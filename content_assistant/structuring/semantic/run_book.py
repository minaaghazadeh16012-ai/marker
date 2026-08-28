"""Run both semantic stages over a whole book, and be safe to run again.

The per-lesson runners are the right unit to *think* in and the wrong unit to
work in: a first-grade farsi book has thirty-three lessons and two stages, and
nobody is going to type sixty-six commands correctly. This drives them.

It owns no extraction logic of its own - it calls
:mod:`content_assistant.structuring.semantic.run_concepts` and
:mod:`~content_assistant.structuring.semantic.run_objectives`, so there is
exactly one implementation of each stage and this cannot drift from it.

Three properties make it usable against a real provider.

**It resumes.** A lesson whose verified artifact already exists is skipped, not
re-run. Re-running the command after a quota reset, a network drop, or an
interrupted evening continues from where it stopped and spends nothing on what
is already done. ``--force`` overrides that for a deliberate re-run.

**It never records a failed call as an empty lesson.** That is
:class:`~content_assistant.structuring.semantic.llm.ModelCallFailed`'s job and
this respects it: a lesson whose call failed is left with no artifact, so the
next run picks it up again. The alternative - writing "0 concepts" - is a claim
about the book that no failed call earns.

**It stops when the provider has stopped.** One lesson failing is a lesson;
several in a row is the provider, and continuing then only converts quota into
noise. ``--max-consecutive-failures`` is where that line sits.

    python -m content_assistant.structuring.semantic.run_book \\
        --l0 <work>/l0_extraction.json \\
        --concepts <work>/l1 --objectives <work>/l2 \\
        --llm marker.services.gemini.GoogleGeminiService

Omitting ``--llm`` runs every lesson in dry-run, which writes each prompt and
calls nothing - the cheap way to see what a whole book would send.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from content_assistant.models.extraction import ExtractionResult
from content_assistant.structuring.segmentation import segment
from content_assistant.structuring.semantic import run_concepts, run_objectives
from content_assistant.structuring.semantic.llm import ModelCallFailed

CONCEPT_ARTIFACT = "concept-verified.json"
OBJECTIVE_ARTIFACT = "objective-verified.json"
EVIDENCE_UNIT = "evidence-unit.json"


def lesson_dir(root: Path, lesson_number: int) -> Path:
    return Path(root) / f"lesson-{lesson_number:02d}"


def _run(runner, argv: Sequence[str]) -> Dict[str, object]:
    """Call a per-lesson runner, keeping its report and quietening its print.

    The runners print their summary for a person watching one lesson. Sixty-six
    of those is not a report, so the text is captured and the structured
    summary each runner already wrote to disk is what stays readable.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = runner.main(list(argv))
    return {"exit_code": code, "output": buffer.getvalue()}


def run_book(
    *,
    l0_path: Path,
    concepts_root: Path,
    objectives_root: Optional[Path],
    llm: Optional[str] = None,
    force: bool = False,
    max_consecutive_failures: int = 3,
    only: Sequence[int] = (),
    with_images: bool = False,
) -> Dict[str, object]:
    """Drive every lesson of one book through both stages."""
    result = ExtractionResult.model_validate_json(
        Path(l0_path).read_text(encoding="utf-8")
    )
    lessons, _ = segment(result)
    wanted = set(only)
    numbers = [
        lesson.lesson_number
        for lesson in lessons
        if not wanted or lesson.lesson_number in wanted
    ]

    report: Dict[str, object] = {
        "book_id": result.document.book.book_id,
        "lessons_total": len(numbers),
        "mode": "live" if llm else "dry-run",
        "lessons": [],
        "stopped_early": False,
    }
    entries: List[Dict[str, object]] = report["lessons"]  # type: ignore[assignment]
    consecutive = 0

    for number in numbers:
        entry: Dict[str, object] = {"lesson": number}
        entries.append(entry)
        concept_dir = lesson_dir(concepts_root, number)
        concept_file = concept_dir / CONCEPT_ARTIFACT

        # -- concepts ----------------------------------------------------
        if concept_file.exists() and not force:
            entry["concepts"] = "already done"
        else:
            argv = [
                "--l0",
                str(l0_path),
                "--lesson",
                str(number),
                "--out",
                str(concepts_root),
            ]
            if llm:
                argv += ["--llm", llm]
            if with_images:
                argv.append("--with-images")
            try:
                outcome = _run(run_concepts, argv)
                entry["concepts"] = (
                    "ok" if outcome["exit_code"] == 0 else "refused"
                )
                consecutive = 0
            except ModelCallFailed as exc:
                entry["concepts"] = f"call failed: {exc}"
                consecutive += 1
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                entry["concepts"] = f"{type(exc).__name__}: {exc}"
                consecutive += 1

        if consecutive >= max_consecutive_failures:
            report["stopped_early"] = True
            break

        # -- objectives --------------------------------------------------
        # Skipped rather than attempted when the concept stage produced
        # nothing: objectives are derived from concepts, so a lesson with no
        # concepts has nothing for this stage to read, and calling anyway would
        # spend a request to be told so.
        if objectives_root is None:
            continue
        objective_file = (
            lesson_dir(objectives_root, number) / OBJECTIVE_ARTIFACT
        )
        if objective_file.exists() and not force:
            entry["objectives"] = "already done"
            continue
        if not concept_file.exists():
            entry["objectives"] = "skipped: no concepts to derive from"
            continue
        argv = [
            "--unit",
            str(concept_dir / EVIDENCE_UNIT),
            "--concepts",
            str(concept_file),
            "--out",
            str(objectives_root),
        ]
        if llm:
            argv += ["--llm", llm]
        try:
            outcome = _run(run_objectives, argv)
            entry["objectives"] = (
                "ok" if outcome["exit_code"] == 0 else "refused"
            )
            consecutive = 0
        except ModelCallFailed as exc:
            entry["objectives"] = f"call failed: {exc}"
            consecutive += 1
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            entry["objectives"] = f"{type(exc).__name__}: {exc}"
            consecutive += 1

        if consecutive >= max_consecutive_failures:
            report["stopped_early"] = True
            break

    report["concepts_done"] = sum(
        1
        for number in numbers
        if (lesson_dir(concepts_root, number) / CONCEPT_ARTIFACT).exists()
    )
    if objectives_root is not None:
        report["objectives_done"] = sum(
            1
            for number in numbers
            if (
                lesson_dir(objectives_root, number) / OBJECTIVE_ARTIFACT
            ).exists()
        )
    report["remaining"] = [
        number
        for number in numbers
        if not (lesson_dir(concepts_root, number) / CONCEPT_ARTIFACT).exists()
        or (
            objectives_root is not None
            and not (
                lesson_dir(objectives_root, number) / OBJECTIVE_ARTIFACT
            ).exists()
        )
    ]
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run both semantic stages over every lesson of a book"
    )
    parser.add_argument("--l0", required=True, help="l0_extraction.json")
    parser.add_argument(
        "--concepts",
        required=True,
        help="directory for lesson-NN/ concept runs",
    )
    parser.add_argument(
        "--objectives",
        default=None,
        help="directory for lesson-NN/ objective runs; omit to stop after "
        "concepts",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="dotted import path of a Marker service; omit for a dry run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run lessons that already have a verified artifact",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="stop after this many failures in a row (the provider, not the "
        "lesson)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated lesson numbers, for a partial run",
    )
    parser.add_argument("--with-images", action="store_true")
    args = parser.parse_args(argv)

    only = [int(n) for n in args.only.split(",") if n.strip()]
    report = run_book(
        l0_path=Path(args.l0),
        concepts_root=Path(args.concepts),
        objectives_root=Path(args.objectives) if args.objectives else None,
        llm=args.llm,
        force=args.force,
        max_consecutive_failures=args.max_consecutive_failures,
        only=only,
        with_images=args.with_images,
    )
    Path(args.concepts).mkdir(parents=True, exist_ok=True)
    (Path(args.concepts) / "run-book.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if report["remaining"] else 0


if __name__ == "__main__":
    sys.exit(main())
