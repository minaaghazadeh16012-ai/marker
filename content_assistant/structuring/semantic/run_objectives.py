"""Run objective extraction for a single lesson.

Objectives consume the concept stage's own output - the evidence unit it was
given and the concepts it grounded - rather than the raw L0 artifact. That is
the stage boundary made literal: if the concept stage did not admit a claim,
the objective stage cannot see it, and no amount of re-reading the book here
can put it back.

Without ``--llm`` the runner stops in **dry-run**: it builds the exact prompt
that would be sent, writes it out for inspection, and calls nothing. That is
the intended way to review a prompt before spending anything on it.

    python -m content_assistant.structuring.semantic.run_objectives \\
        --unit <work>/l1/lesson-07/evidence-unit.json \\
        --concepts <work>/l1/lesson-07/concept-verified.json \\
        --out <work>/l1-objectives

    # then, with a provider configured:
    ... --llm marker.services.gemini.GoogleGeminiService
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from content_assistant.models.content import (
    BookRef,
    Concept,
    ContentSchema,
    Evidence,
)
from content_assistant.structuring.evidence import EvidenceUnit
from content_assistant.structuring.semantic.objectives import (
    build_objective_prompt,
    concept_blocks,
    extract_objectives,
    load_objective_prompt,
)
from content_assistant.validation.engine import (
    render_review_markdown,
    run_validation,
)
from content_assistant.validation.rules import ValidationContext


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def load_concept_stage(path: Path):
    """Read a ``concept-verified.json`` into concepts and their evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = [Concept.model_validate(c) for c in payload.get("concepts", [])]
    evidence = [Evidence.model_validate(e) for e in payload.get("evidence", [])]
    return concepts, evidence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="L1 objective extraction, one lesson"
    )
    parser.add_argument("--unit", required=True, help="evidence-unit.json")
    parser.add_argument(
        "--concepts", required=True, help="concept-verified.json"
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--llm",
        default=None,
        help="dotted import path of a Marker service; omit for a dry run",
    )
    parser.add_argument("--prompt", default="objective_v1")
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=0,
        help="only send the first N concepts (0 = all); for small trials",
    )
    args = parser.parse_args(argv)

    unit = EvidenceUnit.model_validate_json(
        Path(args.unit).read_text(encoding="utf-8")
    )
    concepts, evidence = load_concept_stage(Path(args.concepts))
    if args.max_concepts:
        concepts = concepts[: args.max_concepts]

    out_dir = Path(args.out) / f"lesson-{unit.lesson_number:02d}"
    template = load_objective_prompt(args.prompt)
    prompt_text = build_objective_prompt(unit, concepts, evidence, template)
    blocks = concept_blocks(concepts, evidence)

    _write(out_dir / "objective-prompt.txt", prompt_text)

    summary = {
        "lesson_number": unit.lesson_number,
        "lesson_id": unit.lesson_id,
        "lesson_title": unit.lesson_title,
        "grade": unit.grade,
        "concepts_in": len(concepts),
        "concepts_with_citable_blocks": sum(1 for v in blocks.values() if v),
        "concept_types": {
            t: sum(1 for c in concepts if c.concept_type == t)
            for t in sorted({c.concept_type for c in concepts})
        },
        "prompt_version": template.full_version,
        "prompt_chars": len(prompt_text),
    }

    if not args.llm:
        summary["mode"] = "dry-run"
        summary["note"] = (
            "no --llm given: the exact prompt was written and nothing was "
            "called"
        )
        _write(out_dir / "objective-run-summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    from content_assistant.structuring.semantic.llm import MarkerServiceClient

    client = MarkerServiceClient.from_import_path(args.llm)
    result, raw, _ = extract_objectives(
        unit=unit,
        concepts=concepts,
        evidence=evidence,
        client=client,
        document_id=unit.book_id,
        template=template,
    )

    _write(out_dir / "objective-raw.json", raw.model_dump(mode="json"))
    _write(
        out_dir / "objective-verified.json", result.model_dump(mode="json")
    )

    schema = ContentSchema(
        book=BookRef(
            book_id=unit.book_id,
            grade=unit.grade,
            subject=unit.subject,
            language=unit.language,
        ),
        concepts=concepts,
        objectives=result.objectives,
        evidence=list({e.id: e for e in list(evidence) + result.evidence}.values()),
    )
    context = ValidationContext(schema_doc=schema)
    report = run_validation(context, stages=["semantic"])
    _write(
        out_dir / "objective-validation.json", report.model_dump(mode="json")
    )
    _write(
        out_dir / "objective-review.md",
        render_review_markdown(report, title="گزارش بررسی هدف‌ها"),
    )

    summary.update(
        {
            "mode": "live",
            "model_id": result.model_id,
            "raw_objectives": len(raw.objectives),
            "accepted": len(result.objectives),
            "rejected": len(result.admission.rejected),
            "ungrounded": len(result.ungrounded),
            "dropped_citations": len(result.admission.dropped_citations),
            "concepts_without_objectives": len(
                result.concepts_without_objectives
            ),
            "needs_review": sum(
                1 for o in result.objectives if o.requires_human_review
            ),
            "validation_ok": report.ok,
            "validation_counts": report.counts(),
        }
    )
    _write(out_dir / "objective-run-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
