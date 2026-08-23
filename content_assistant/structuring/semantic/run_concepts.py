"""Run concept extraction for a single lesson.

One lesson at a time, on purpose. The first real call against a book should be
small enough to read line by line, and a per-lesson runner keeps every later
re-run cheap: change a prompt and one lesson repeats, not a whole book.

Without ``--llm`` the runner stops in **dry-run**: it builds the evidence unit
and the exact prompt that would be sent, writes both out for inspection, and
calls nothing. That is the intended way to review a prompt before spending
anything on it.

    python -m content_assistant.structuring.semantic.run_concepts \\
        --l0 <work>/l0_extraction.json --lesson 11 --out <work>/l1

    # then, with a provider configured:
    ... --llm marker.services.gemini.GoogleGeminiService
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from content_assistant.models.content import ContentSchema, BookRef
from content_assistant.models.extraction import ExtractionResult
from content_assistant.structuring.evidence import build_evidence_unit
from content_assistant.structuring.segmentation import segment
from content_assistant.structuring.semantic.concepts import (
    build_prompt,
    extract_concepts,
    load_prompt,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="L1 concept extraction, one lesson")
    parser.add_argument("--l0", required=True, help="path to l0_extraction.json")
    parser.add_argument("--lesson", type=int, required=True, help="lesson number")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--llm",
        default=None,
        help="dotted import path of a Marker service; omit for a dry run",
    )
    parser.add_argument("--prompt", default="concept_v1")
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="send page images for a low-density lesson (multimodal)",
    )
    args = parser.parse_args(argv)

    result = ExtractionResult.model_validate_json(
        Path(args.l0).read_text(encoding="utf-8")
    )
    lessons, sections = segment(result)
    match = [x for x in lessons if x.lesson_number == args.lesson]
    if not match:
        print(f"lesson {args.lesson} not found; have "
              f"{[x.lesson_number for x in lessons]}")
        return 2
    lesson = match[0]
    unit = build_evidence_unit(result, lesson, sections)

    out_dir = Path(args.out) / f"lesson-{args.lesson:02d}"
    template = load_prompt(args.prompt)
    prompt_text = build_prompt(unit, template)

    _write(out_dir / "evidence-unit.json", unit.model_dump(mode="json"))
    _write(out_dir / "prompt.txt", prompt_text)

    document_id = result.document.book.book_id or "unknown-book"
    summary = {
        "lesson_number": lesson.lesson_number,
        "lesson_id": lesson.id,
        "lesson_title": lesson.title,
        "printed_pages": [
            lesson.page_range.printed_start,
            lesson.page_range.printed_end,
        ],
        "citable_blocks": len(unit.citable_block_ids()),
        "images": len(unit.citable_asset_ids()),
        "material_profile": unit.material_profile.model_dump(),
        "needs_images": unit.needs_images(),
        "prompt_version": template.full_version,
        "prompt_chars": len(prompt_text),
    }

    if not args.llm:
        summary["mode"] = "dry-run"
        summary["note"] = (
            "no --llm given: the evidence unit and the exact prompt were "
            "written, and nothing was called"
        )
        _write(out_dir / "run-summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    from content_assistant.structuring.semantic.llm import MarkerServiceClient

    client = MarkerServiceClient.from_import_path(args.llm)
    image_paths: List[str] = []
    if args.with_images and unit.needs_images():
        base = Path(args.l0).parent
        image_paths = [
            str(base / image.path)
            for section in unit.sections
            for image in section.images
            if image.path
        ]

    extraction_result, raw, _ = extract_concepts(
        unit=unit,
        client=client,
        document_id=document_id,
        template=template,
        image_paths=image_paths,
    )

    _write(out_dir / "concept-raw.json", raw.model_dump(mode="json"))
    _write(
        out_dir / "concept-verified.json", extraction_result.model_dump(mode="json")
    )

    schema = ContentSchema(
        book=BookRef(
            book_id=document_id,
            grade=lesson.grade,
            subject=lesson.subject,
            language=result.document.book.language,
            page_count=result.document.page_count,
            page_offset=result.document.page_offset,
        ),
        lessons=[lesson],
        concepts=extraction_result.concepts,
        evidence=extraction_result.evidence,
    )
    context = ValidationContext(
        extraction=result, lessons=lessons, sections=sections, schema_doc=schema
    )
    report = run_validation(context, stages=["structure", "semantic", "final"])
    _write(out_dir / "validation-report.json", report.model_dump(mode="json"))
    _write(out_dir / "content-review.md", render_review_markdown(report))

    summary.update(
        {
            "mode": "live",
            "model_id": extraction_result.model_id,
            "images_sent": len(image_paths),
            "raw_concepts": len(raw.concepts),
            "accepted": len(extraction_result.concepts),
            "rejected": len(extraction_result.admission.rejected),
            "ungrounded": len(extraction_result.ungrounded),
            "dropped_citations": len(extraction_result.admission.dropped_citations),
            "evidence": len(extraction_result.evidence),
            "validation_ok": report.ok,
            "validation_counts": report.counts(),
        }
    )
    _write(out_dir / "run-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
