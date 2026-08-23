"""Per-page measurement: how much of the text layer survived Marker's layout?

The diagnostic answers one question per page - *is text missing, and why* -
and it answers it from numbers, not from guessing. A page becomes a recovery
candidate when any of three independent signals fires; each firing signal is
recorded in ``candidate_reasons`` so the decision stays auditable.

The three signals come straight from the measurements taken on the whole book:
every page that lost text lost it because a ``Picture``/``PictureGroup`` box
covered the lines, so picture coverage, the recovery ratio and the
"blocks say nothing, the text layer says plenty" contradiction between them
are exactly the three ways that failure shows up.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from content_assistant.models.extraction import (
    BBox,
    ExtractionConfig,
    GROUP_BLOCK_TYPES,
    PICTURE_BLOCK_TYPES,
    PageDiagnostics,
    RawLine,
    TEXT_BLOCK_TYPES,
)


def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: BBox, b: BBox) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def covered_fraction(inner: BBox, outers: Sequence[BBox]) -> float:
    """Fraction of ``inner``'s own area covered by the union-ish of ``outers``.

    Overlaps between the outer boxes are not subtracted; the value is clamped
    at 1.0. For duplicate detection that is the safe direction - it can only
    make a line look *more* covered, never less, so a genuinely missing line is
    never dropped by accident.
    """
    if not outers:
        return 0.0
    area = bbox_area(inner)
    if area <= 0:
        return 0.0
    total = sum(intersection_area(inner, o) for o in outers)
    return min(1.0, total / area)


def is_text_block(block_type: str) -> bool:
    return block_type in TEXT_BLOCK_TYPES


def is_picture_block(block_type: str) -> bool:
    return block_type in PICTURE_BLOCK_TYPES or block_type in GROUP_BLOCK_TYPES


def single_char_span_ratio(lines: Iterable[RawLine]) -> float:
    """Share of spans that hold exactly one character.

    Ordinary body text arrives as multi-character spans. Text placed glyph by
    glyph - decorative titles set along a curve - arrives as one span per
    letter, which is what this ratio detects.
    """
    total = sum(line.n_spans for line in lines)
    if not total:
        return 0.0
    singles = sum(line.n_single_char_spans for line in lines)
    return singles / total


def compute_diagnostics(
    *,
    pdf_page: int,
    marker_blocks: Sequence[Dict],
    page_bbox: BBox,
    raw_lines: Sequence[RawLine],
    config: ExtractionConfig,
) -> PageDiagnostics:
    """Measure one page. ``marker_blocks`` are top-level blocks of that page."""
    counts: Dict[str, int] = {}
    marker_chars = 0
    picture_area = 0.0
    for block in marker_blocks:
        btype = block["type"]
        counts[btype] = counts.get(btype, 0) + 1
        if is_picture_block(btype):
            picture_area += bbox_area(block["bbox"])
        else:
            marker_chars += len(block.get("text") or "")

    page_area = bbox_area(page_bbox) or 1.0
    picture_area_frac = min(1.0, picture_area / page_area)

    raw_chars = sum(len(line.text.strip()) for line in raw_lines)
    marker_text_blocks = sum(1 for b in marker_blocks if is_text_block(b["type"]))
    ratio = (marker_chars / raw_chars) if raw_chars else None

    reasons: List[str] = []
    if picture_area_frac >= config.picture_area_frac_min:
        reasons.append(
            f"picture_area_frac={picture_area_frac:.2f}>={config.picture_area_frac_min}"
        )
    if ratio is not None and ratio < config.recovery_ratio_min:
        reasons.append(f"recovery_ratio={ratio:.2f}<{config.recovery_ratio_min}")
    if marker_text_blocks == 0 and len(raw_lines) > 0:
        reasons.append("no_text_blocks_but_raw_lines_present")

    decorative = (
        len(raw_lines) >= config.decorative_min_lines
        and single_char_span_ratio(raw_lines)
        >= config.decorative_single_char_span_ratio
    )

    return PageDiagnostics(
        pdf_page=pdf_page,
        raw_chars=raw_chars,
        marker_chars=marker_chars,
        recovery_ratio=None if ratio is None else round(ratio, 4),
        raw_lines=len(raw_lines),
        marker_text_blocks=marker_text_blocks,
        picture_area_frac=round(picture_area_frac, 4),
        has_picture=any(b["type"] in PICTURE_BLOCK_TYPES for b in marker_blocks),
        has_picture_group=any(b["type"] in GROUP_BLOCK_TYPES for b in marker_blocks),
        has_page_header=counts.get("PageHeader", 0) > 0,
        has_page_footer=counts.get("PageFooter", 0) > 0,
        is_decorative=decorative,
        is_recovery_candidate=bool(reasons),
        candidate_reasons=reasons,
    )
