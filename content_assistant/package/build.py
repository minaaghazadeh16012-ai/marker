"""Assemble a Content Package from the artifacts a run left behind.

The stage artifacts are the record; this is a view over them. That direction
matters: the builder copies, it does not decide. It re-derives the
deterministic structure from L0 rather than trusting a stored copy of it, and
it carries concepts, objectives and evidence across **unchanged** - no
re-scoring, no re-verification, no re-wording. If a package disagrees with the
artifacts it was built from, the builder has a bug.

The one thing it adds is provenance, and only where an artifact predates the
field. A concept written before per-entity provenance existed still came from a
known model and a known prompt - the stage result recorded both, once, at the
top - so lifting them onto the entity states a fact the artifact already held.
Nothing is invented: an artifact that names no model produces a provenance that
names no model, and ``PROV001`` reports it.

    python -m content_assistant.package.build \\
        --l0 <work>/l0_extraction.json \\
        --concepts <work>/l1 \\
        --objectives <work>/l2 \\
        --out content/

Both stage directories are the ones the runners wrote, holding ``lesson-NN``
subdirectories. Either may be omitted: a package with concepts and no
objectives is a truthful record of a book only half processed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from content_assistant.models.content import (
    BookRef,
    Concept,
    ContentSchema,
    Evidence,
    GenerationProvenance,
    LearningObjective,
    Provenance,
)
from content_assistant.models.extraction import ExtractionResult
from content_assistant.package.schema import (
    BUILDER_VERSION,
    ContentPackage,
    compute_stats,
    default_path,
    save_content,
)
from content_assistant.structuring.segmentation import segment
from content_assistant.validation.engine import (
    render_review_markdown,
    run_validation,
)
from content_assistant.validation.rules import ValidationContext

CONCEPT_ARTIFACT = "concept-verified.json"
OBJECTIVE_ARTIFACT = "objective-verified.json"


class BuildError(RuntimeError):
    """The artifacts cannot be assembled into a package."""


def _read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildError(f"{path} is not valid JSON: {exc}") from exc


def _stage_provenance(payload: Dict, stage: str) -> Provenance:
    """The provenance an artifact already implies, made explicit.

    ``model_id`` and ``prompt_version`` are recorded once per stage result.
    Empty strings become ``None`` so that "not recorded" reads as absent rather
    than as a model called ``""``.
    """
    return Provenance(
        extraction_method="model_proposed",
        stage=stage,
        model_id=payload.get("model_id") or None,
        prompt_version=payload.get("prompt_version") or None,
    )


def load_concept_artifacts(
    directory: Path,
) -> Tuple[List[Concept], List[Evidence]]:
    """Every lesson's concepts, in lesson order."""
    concepts: List[Concept] = []
    evidence: Dict[str, Evidence] = {}
    for path in sorted(Path(directory).glob(f"lesson-*/{CONCEPT_ARTIFACT}")):
        payload = _read_json(path)
        provenance = _stage_provenance(payload, "concepts")
        for raw in payload.get("concepts", []):
            concept = Concept.model_validate(raw)
            if concept.provenance is None:
                concept = concept.model_copy(update={"provenance": provenance})
            concepts.append(concept)
        for raw in payload.get("evidence", []):
            item = Evidence.model_validate(raw)
            evidence.setdefault(item.id, item)
    return concepts, list(evidence.values())


def load_objective_artifacts(
    directory: Path,
) -> Tuple[List[LearningObjective], List[Evidence]]:
    objectives: List[LearningObjective] = []
    evidence: Dict[str, Evidence] = {}
    for path in sorted(Path(directory).glob(f"lesson-*/{OBJECTIVE_ARTIFACT}")):
        payload = _read_json(path)
        provenance = _stage_provenance(payload, "objectives")
        for raw in payload.get("objectives", []):
            objective = LearningObjective.model_validate(raw)
            if objective.provenance is None:
                objective = objective.model_copy(
                    update={"provenance": provenance}
                )
            objectives.append(objective)
        for raw in payload.get("evidence", []):
            item = Evidence.model_validate(raw)
            evidence.setdefault(item.id, item)
    return objectives, list(evidence.values())


def merge_evidence(*groups: Sequence[Evidence]) -> List[Evidence]:
    """One evidence table from many, keyed by id.

    Duplicates across stages are expected rather than exceptional: an evidence
    id is a hash of the document, block and quote, so the same sentence cited
    by a concept and by its objective produces the same id in both artifacts.
    Keeping the first is therefore not a choice about which to trust - they are
    identical by construction.
    """
    merged: Dict[str, Evidence] = {}
    for group in groups:
        for item in group:
            merged.setdefault(item.id, item)
    return list(merged.values())


def build_package(
    *,
    extraction: ExtractionResult,
    concepts: Sequence[Concept] = (),
    objectives: Sequence[LearningObjective] = (),
    evidence: Sequence[Evidence] = (),
    built_at: Optional[str] = None,
) -> ContentPackage:
    """Assemble one book's package from its parts.

    The book's identity has to be declared in L0 rather than guessed here; a
    package that could not say which grade and subject it holds would be
    unusable by a registry, and inferring either from a filename is how the
    wrong book ends up in the wrong shelf.
    """
    book = extraction.document.book
    missing = [
        field
        for field in ("book_id", "grade", "subject")
        if getattr(book, field, None) in (None, "")
    ]
    if missing:
        raise BuildError(
            "the extraction does not declare "
            f"{', '.join(missing)}; a package cannot be filed without it. "
            "Re-run the extraction with the book identity given."
        )

    lessons, sections = segment(extraction)
    content = ContentSchema(
        book=BookRef(
            book_id=book.book_id,
            grade=book.grade,
            subject=book.subject,
            language=book.language,
            title=book.title,
            source=extraction.document.source,
            source_sha256=extraction.document.source_sha256,
            page_count=extraction.document.page_count,
            page_offset=extraction.document.page_offset,
        ),
        lessons=lessons,
        sections=sections,
        concepts=list(concepts),
        objectives=list(objectives),
        evidence=list(evidence),
        provenance=GenerationProvenance(
            extractor_version=extraction.document.extractor_version,
            l0_source_sha256=extraction.document.source_sha256,
            prompt_versions=_prompt_versions(concepts, objectives),
            generated_at=built_at,
        ),
    )
    return ContentPackage(
        package_id=ContentPackage.build_id(
            book.grade, book.subject, book.book_id
        ),
        grade=book.grade,
        subject=book.subject,
        language=book.language,
        book_id=book.book_id,
        title=book.title,
        source_sha256=extraction.document.source_sha256,
        built_at=built_at,
        builder_version=BUILDER_VERSION,
        stats=compute_stats(content),
        content=content,
    )


def _prompt_versions(
    concepts: Sequence[Concept], objectives: Sequence[LearningObjective]
) -> Dict[str, str]:
    """Which prompt produced which stage, taken from what the entities say.

    A stage that used two prompt versions is a real thing - a run resumed after
    a prompt edit - and it is recorded as such rather than collapsed to one, so
    the inconsistency is visible instead of hidden by whichever entity happened
    to be read last.
    """
    out: Dict[str, str] = {}
    for stage, entities in (("concepts", concepts), ("objectives", objectives)):
        versions = sorted(
            {
                entity.provenance.prompt_version
                for entity in entities
                if entity.provenance and entity.provenance.prompt_version
            }
        )
        if versions:
            out[stage] = ", ".join(versions)
    return out


def validate_package(
    package: ContentPackage, extraction: Optional[ExtractionResult] = None
):
    """Run every rule that can see a whole package.

    All three stages, because a package is the first artifact at which the
    ``final`` rules have anything to look at: reference integrity and cycles
    are properties of the assembled document, not of one lesson.
    """
    context = ValidationContext(
        extraction=extraction,
        lessons=package.content.lessons,
        sections=package.content.sections,
        schema_doc=package.content,
    )
    return run_validation(context, stages=["structure", "semantic", "final"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a Content Package from run artifacts"
    )
    parser.add_argument("--l0", required=True, help="l0_extraction.json")
    parser.add_argument(
        "--concepts", default=None, help="directory of lesson-NN/ concept runs"
    )
    parser.add_argument(
        "--objectives",
        default=None,
        help="directory of lesson-NN/ objective runs",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="content root; the package is written to "
        "<root>/grade-N/<subject>/content-package.json",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="write the package even if validation found errors",
    )
    args = parser.parse_args(argv)

    extraction = ExtractionResult.model_validate_json(
        Path(args.l0).read_text(encoding="utf-8")
    )

    concepts: List[Concept] = []
    objectives: List[LearningObjective] = []
    concept_evidence: List[Evidence] = []
    objective_evidence: List[Evidence] = []
    if args.concepts:
        concepts, concept_evidence = load_concept_artifacts(Path(args.concepts))
    if args.objectives:
        objectives, objective_evidence = load_objective_artifacts(
            Path(args.objectives)
        )

    package = build_package(
        extraction=extraction,
        concepts=concepts,
        objectives=objectives,
        evidence=merge_evidence(concept_evidence, objective_evidence),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    report = validate_package(package, extraction)

    root = Path(args.out)
    path = default_path(root, package.grade, package.subject)
    report_path = path.parent / "package-validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (path.parent / "package-review.md").write_text(
        render_review_markdown(report, title="گزارش بررسی بسته محتوا"),
        encoding="utf-8",
    )

    summary = {
        "package_id": package.package_id,
        "content_schema_version": package.content_schema_version,
        "stats": package.stats.model_dump(),
        "validation": report.summary(),
    }

    if report.errors and not args.allow_errors:
        summary["written"] = False
        summary["note"] = (
            f"{len(report.errors)} validation errors; the package was not "
            f"written. The findings are in {report_path}. Pass --allow-errors "
            "to write it anyway."
        )
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 1

    save_content(package, path)
    summary["written"] = True
    summary["path"] = str(path)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
