"""L0 extraction pipeline: Marker, then geometric recovery, then normalization.

    PDF
      -> Marker (fast mode, no OCR)            marker_backend
      -> per-page measurement                  page_diagnostics
      -> rescue of lines Marker's layout ate   recovery
      -> deterministic Persian normalization   text.persian
      -> page map + contents reconstruction
      -> l0_extraction.json + validation_report.json

Everything here is deterministic. No LLM, no OCR, no network.

Run it with::

    python -m content_assistant.extraction.pipeline \
        --pdf "<book>.pdf" --out <work-dir> --marker <path to marker_single>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from content_assistant.extraction import marker_backend as mb
from content_assistant.extraction.page_diagnostics import compute_diagnostics
from content_assistant.extraction.recovery import (
    polygon_from_bbox,
    recover_lines,
    reconstruct_decorative_toc,
    text_bboxes_of,
)
from content_assistant.models.extraction import (
    Asset,
    Block,
    DocumentInfo,
    ExtractionConfig,
    ExtractionResult,
    Page,
    PageOffsetEvidence,
    RawLine,
    RawPage,
    TocEntry,
)
from content_assistant.text.persian import (
    PersianNormalizationConfig,
    digits_to_int,
    normalize,
)

#: A printed folio sits in the bottom band of the page.
FOOTER_BAND_FRACTION = 0.85


# ---------------------------------------------------------------------------
# raw text layer
# ---------------------------------------------------------------------------


def load_raw_pages(pdf_path: Path) -> Dict[int, RawPage]:
    """Read the text layer through Marker's own provider, before layout.

    Imported lazily so the rest of the package - and its tests - stay usable
    without paying for Marker's import graph.
    """
    from marker.providers.pdf import PdfProvider  # noqa: PLC0415

    provider = PdfProvider(
        str(pdf_path), {"pdftext_workers": 1, "disable_ocr": True}
    )
    pages: Dict[int, RawPage] = {}
    for page_index, provider_lines in provider.page_lines.items():
        lines = []
        for entry in provider_lines:
            spans = entry.spans or []
            lines.append(
                RawLine(
                    text=entry.raw_text,
                    bbox=[float(v) for v in entry.line.polygon.bbox],
                    n_spans=len(spans),
                    n_single_char_spans=sum(
                        1 for s in spans if len(s.text.strip()) == 1
                    ),
                )
            )
        pages[page_index] = RawPage(pdf_page_index=page_index, lines=lines)
    return pages


# ---------------------------------------------------------------------------
# page map
# ---------------------------------------------------------------------------


def printed_page_candidates(
    blocks: Sequence[mb.MarkerBlock], page_bbox: Sequence[float]
) -> List[int]:
    """Numbers that could be this page's printed folio.

    Two sources, both geometric: an explicit ``PageFooter`` block, and a very
    short numeric block sitting in the bottom band (Marker does not always
    label the folio as a footer).
    """
    height = page_bbox[3] - page_bbox[1]
    cutoff = page_bbox[1] + height * FOOTER_BAND_FRACTION
    out: List[int] = []
    for block in blocks:
        text = block.text.strip()
        if not text or len(text) > 4:
            continue
        in_band = block.bbox[1] >= cutoff
        if block.type != "PageFooter" and not in_band:
            continue
        value = digits_to_int(text)
        if value is not None and 0 < value <= 500:
            out.append(value)
    return out


def build_page_map(
    marker_pages: Sequence[mb.MarkerPage],
) -> Tuple[Optional[int], PageOffsetEvidence, Dict[int, Tuple[int, str]]]:
    """Derive ``printed_page`` per page and the document-wide offset.

    The offset is the majority of ``pdf_page - printed_page`` over every page
    that shows a folio. It is never assumed: with no evidence the offset stays
    ``None`` and printed pages stay ``None`` rather than being invented.
    """
    direct: Dict[int, int] = {}
    for page in marker_pages:
        candidates = printed_page_candidates(page.blocks, page.bbox)
        if candidates:
            direct[page.pdf_page_index] = candidates[0]

    offsets = Counter(
        (index + 1) - printed for index, printed in direct.items()
    )
    if not offsets:
        return None, PageOffsetEvidence(), {}

    offset, votes = offsets.most_common(1)[0]
    evidence = PageOffsetEvidence(
        value=offset,
        samples=len(direct),
        agreement=round(votes / len(direct), 4),
    )

    resolved: Dict[int, Tuple[int, str]] = {}
    for page in marker_pages:
        index = page.pdf_page_index
        if index in direct:
            resolved[index] = (direct[index], "page_footer")
        else:
            resolved[index] = ((index + 1) - offset, "inferred_from_offset")
    return offset, evidence, resolved


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    pdf_path: Path,
    out_dir: Path,
    marker_executable: str,
    config: Optional[ExtractionConfig] = None,
    normalization: Optional[PersianNormalizationConfig] = None,
    page_range: Optional[str] = None,
    force: bool = False,
    raw_pages: Optional[Dict[int, RawPage]] = None,
) -> ExtractionResult:
    config = config or ExtractionConfig()
    normalization = normalization or PersianNormalizationConfig()
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = mb.MarkerBackend(marker_executable, out_dir / "cache")
    run = backend.run(pdf_path, page_range=page_range, force=force)
    assets_dir = out_dir / "assets"
    asset_paths = mb.write_assets(run.pages, assets_dir)

    if raw_pages is None:
        raw_pages = load_raw_pages(pdf_path)
    offset, offset_evidence, page_map = build_page_map(run.pages)

    pages: List[Page] = []
    toc: List[TocEntry] = []

    for marker_page in run.pages:
        index = marker_page.pdf_page_index
        raw_lines = raw_pages.get(index, RawPage(pdf_page_index=index)).lines
        block_dicts = [
            {"type": b.type, "bbox": b.bbox, "text": b.text}
            for b in marker_page.blocks
        ]
        diagnostics = compute_diagnostics(
            pdf_page=index + 1,
            marker_blocks=block_dicts,
            page_bbox=marker_page.bbox,
            raw_lines=raw_lines,
            config=config,
        )

        blocks: List[Block] = []
        assets: List[Asset] = []
        for block in marker_page.blocks:
            asset_ids = []
            for name in (block.images or {}):
                asset_id = name.strip("/").replace("/", "_")
                asset_ids.append(asset_id)
                path = asset_paths.get(name)
                assets.append(
                    Asset(
                        asset_id=asset_id,
                        pdf_page=index + 1,
                        bbox=block.bbox,
                        path=str(Path(path).relative_to(out_dir)) if path else "",
                    )
                )
            clean = normalize(block.text, normalization)
            blocks.append(
                Block(
                    block_id=block.block_id,
                    type=block.type,
                    text=clean,
                    text_raw=block.text if block.text != clean else None,
                    bbox=block.bbox,
                    polygon=block.polygon,
                    source="marker",
                    section_hierarchy=block.section_hierarchy,
                    asset_ids=asset_ids,
                )
            )

        if diagnostics.is_recovery_candidate:
            missing, duplicates = recover_lines(
                raw_lines=raw_lines,
                existing_text_bboxes=text_bboxes_of(block_dicts),
                config=config,
            )
            diagnostics.duplicates_skipped = duplicates
            diagnostics.recovered_lines = len(missing)
            for position, line in enumerate(missing):
                clean = normalize(line.text, normalization)
                if not clean:
                    continue
                diagnostics.recovered_chars += len(clean)
                blocks.append(
                    Block(
                        block_id=f"/page/{index}/RecoveredText/{position}",
                        type="RecoveredText",
                        text=clean,
                        text_raw=line.text if line.text != clean else None,
                        bbox=line.bbox,
                        polygon=polygon_from_bbox(line.bbox),
                        source="pdfprovider_recovery",
                    )
                )

        if diagnostics.is_decorative:
            toc.extend(
                reconstruct_decorative_toc(
                    pdf_page=index + 1,
                    raw_lines=raw_lines,
                    page_count=len(run.pages),
                    config=config,
                )
            )

        printed, printed_source = page_map.get(index, (None, None))
        pages.append(
            Page(
                pdf_page=index + 1,
                pdf_page_index=index,
                printed_page=printed,
                printed_page_source=printed_source,
                blocks=blocks,
                assets=assets,
                diagnostics=diagnostics,
            )
        )

    document = DocumentInfo(
        source=pdf_path.name,
        source_sha256=mb.sha256_of(pdf_path),
        page_count=len(run.pages),
        page_offset=offset,
        page_offset_evidence=offset_evidence,
    )
    return ExtractionResult(document=document, pages=pages, toc=toc)


# ---------------------------------------------------------------------------
# validation report
# ---------------------------------------------------------------------------


def build_validation_report(result: ExtractionResult) -> Dict:
    """Aggregate the per-page diagnostics into the numbers a human reads.

    Character totals come from the per-page diagnostics rather than from the
    raw text layer directly, so a partial run (``--page-range``) reports on the
    pages it actually processed instead of the whole book.
    """
    diagnostics = [p.diagnostics for p in result.pages if p.diagnostics]
    raw_chars = sum(d.raw_chars for d in diagnostics)
    marker_chars = sum(d.marker_chars for d in diagnostics)
    recovered_chars = sum(d.recovered_chars for d in diagnostics)
    duplicates = sum(d.duplicates_skipped for d in diagnostics)
    recovered_blocks = sum(
        1 for p in result.pages for b in p.blocks if b.source == "pdfprovider_recovery"
    )
    final_chars = marker_chars + recovered_chars

    def final_ratio(page: Page) -> Optional[float]:
        diag = page.diagnostics
        if not diag or not diag.raw_chars:
            return None
        return (diag.marker_chars + diag.recovered_chars) / diag.raw_chars

    below_80 = [p.pdf_page for p in result.pages if (final_ratio(p) or 1.0) < 0.80]
    below_90 = [p.pdf_page for p in result.pages if (final_ratio(p) or 1.0) < 0.90]
    remaining = [
        {
            "pdf_page": p.pdf_page,
            "raw_chars": p.diagnostics.raw_chars,
            "final_chars": p.diagnostics.marker_chars + p.diagnostics.recovered_chars,
            "missing_chars": p.diagnostics.raw_chars
            - (p.diagnostics.marker_chars + p.diagnostics.recovered_chars),
        }
        for p in result.pages
        if p.diagnostics and (final_ratio(p) or 1.0) < 0.90
    ]
    return {
        "raw_char_count": raw_chars,
        "marker_char_count": marker_chars,
        "recovered_char_count": recovered_chars,
        "final_char_count": final_chars,
        "final_recovery_ratio": (
            round(final_chars / raw_chars, 4) if raw_chars else None
        ),
        "marker_only_recovery_ratio": (
            round(marker_chars / raw_chars, 4) if raw_chars else None
        ),
        "pages_total": len(result.pages),
        "pages_recovery_candidates": sum(
            1 for d in diagnostics if d.is_recovery_candidate
        ),
        "pages_with_recovered_text": sum(
            1 for d in diagnostics if d.recovered_lines
        ),
        "pages_below_80_percent": len(below_80),
        "pages_below_80_percent_list": below_80,
        "pages_below_90_percent": len(below_90),
        "pages_below_90_percent_list": below_90,
        "duplicates_skipped": duplicates,
        "recovered_blocks": recovered_blocks,
        "assets": sum(len(p.assets) for p in result.pages),
        "page_offset": result.document.page_offset,
        "page_offset_evidence": result.document.page_offset_evidence.model_dump(),
        "toc_entries": len(result.toc),
        "toc": [entry.model_dump() for entry in result.toc],
        "pages_still_incomplete": remaining,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Content Assistant L0 extraction")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--marker", required=True, help="path to marker_single")
    parser.add_argument("--page-range", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    # Read the text layer once and hand it to both stages - it is the second
    # slowest step after Marker itself.
    raw_pages = load_raw_pages(pdf_path)
    result = run_pipeline(
        pdf_path=pdf_path,
        out_dir=out_dir,
        marker_executable=args.marker,
        page_range=args.page_range,
        force=args.force,
        raw_pages=raw_pages,
    )
    report = build_validation_report(result)

    (out_dir / "l0_extraction.json").write_text(
        result.model_dump_json(indent=1, exclude_none=True), encoding="utf-8"
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
