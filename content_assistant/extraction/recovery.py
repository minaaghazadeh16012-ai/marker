"""Geometric recovery of text that Marker's layout pass dropped.

Why this exists
---------------
Measured over the whole book, Marker never *failed to read* text: pdftext
extracted a text layer on every page. What it sometimes fails to do is *place*
that text - when the layout model calls a region ``Picture`` or
``PictureGroup``, every line inside that box is absorbed and the rendered
output shows an image with no words. On the science textbook that cost 12.4%
of the characters, including the entire contents spread.

So the fix is not OCR. The lines are already in ``PdfProvider.page_lines``,
each with its own polygon. Recovery re-attaches them by geometry alone: take
the raw lines, drop the ones an existing text block already covers, order what
is left, and emit it as clearly-labelled ``RecoveredText`` blocks.

No model, no LLM, no second reading of the PDF.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from content_assistant.extraction.page_diagnostics import (
    covered_fraction,
    is_text_block,
)
from content_assistant.models.extraction import (
    BBox,
    ExtractionConfig,
    RawLine,
    TocEntry,
)
from content_assistant.text.persian import digits_to_int

#: Any Arabic-script codepoint. Used only to tell words from numbers.
_PERSIAN_RE = re.compile("[؀-ۿ]")


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def centroid(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def vertical_overlap_ratio(a: BBox, b: BBox) -> float:
    """Shared vertical extent as a fraction of the shorter box's height."""
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    if bottom <= top:
        return 0.0
    shorter = min(a[3] - a[1], b[3] - b[1])
    return (bottom - top) / shorter if shorter > 0 else 0.0


def is_duplicate(
    line_bbox: BBox, existing_bboxes: Sequence[BBox], config: ExtractionConfig
) -> bool:
    """True when an existing text block already covers this line.

    The test is on the *line's own* area: a line sitting inside a large text
    block is covered, while a large block that merely clips a line's corner
    does not hide it.
    """
    return covered_fraction(line_bbox, existing_bboxes) >= config.duplicate_overlap_min


def polygon_from_bbox(box: BBox) -> List[List[float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------


def group_into_rows(
    lines: Sequence[RawLine], config: ExtractionConfig
) -> List[List[RawLine]]:
    """Group lines into visual rows by vertical overlap."""
    rows: List[List[RawLine]] = []
    for line in sorted(lines, key=lambda ln: (ln.bbox[1], -ln.bbox[2])):
        placed = False
        for row in rows:
            if vertical_overlap_ratio(row[0].bbox, line.bbox) >= config.row_overlap_min:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])
    return rows


def sort_reading_order(
    lines: Sequence[RawLine], config: ExtractionConfig
) -> List[RawLine]:
    """Order lines the way a Persian reader would: top to bottom, right to left.

    Rows come first (y ascending), and inside a row the *rightmost* line is
    read first, because the script is right-to-left. This deliberately does not
    try to be clever about columns - Marker already orders multi-column body
    text from the PDF character stream; what arrives here are the leftovers.
    """
    rows = group_into_rows(lines, config)
    rows.sort(key=lambda row: min(ln.bbox[1] for ln in row))
    ordered: List[RawLine] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda ln: -ln.bbox[2]))
    return ordered


# ---------------------------------------------------------------------------
# line-level recovery
# ---------------------------------------------------------------------------


def usable_lines(lines: Sequence[RawLine], config: ExtractionConfig) -> List[RawLine]:
    """Drop empty lines and the 1-pixel newline glyphs pdftext emits."""
    return [
        ln
        for ln in lines
        if ln.text.strip() and ln.height >= config.min_line_height
    ]


def text_bboxes_of(marker_blocks: Sequence[Dict]) -> List[BBox]:
    """Bounding boxes of blocks that actually rendered some text.

    Blocks that came out empty are excluded on purpose: a ``SectionHeader``
    that rendered blank has hidden its content, so the line underneath it is
    still missing and must stay recoverable.
    """
    return [
        b["bbox"]
        for b in marker_blocks
        if is_text_block(b["type"]) and (b.get("text") or "").strip()
    ]


def recover_lines(
    *,
    raw_lines: Sequence[RawLine],
    existing_text_bboxes: Sequence[BBox],
    config: ExtractionConfig,
) -> Tuple[List[RawLine], int]:
    """Return (lines Marker did not render, count of duplicates skipped)."""
    missing: List[RawLine] = []
    duplicates = 0
    for line in usable_lines(raw_lines, config):
        if is_duplicate(line.bbox, existing_text_bboxes, config):
            duplicates += 1
        else:
            missing.append(line)
    return sort_reading_order(missing, config), duplicates


# ---------------------------------------------------------------------------
# decorative pages (curved / per-glyph typesetting, e.g. a contents spread)
# ---------------------------------------------------------------------------


def _is_numeric(text: str) -> bool:
    return digits_to_int(text.strip()) is not None


def find_number_anchors(
    lines: Sequence[RawLine], config: ExtractionConfig
) -> List[RawLine]:
    """Numeric lines set much larger than the rest - the destination pages.

    On a decorative contents page the numbers come in two sizes: the
    destination page in display type, and the small lesson number beside it.
    The split is found from the data - sort the numeric line heights, take the
    widest ratio gap between neighbours, and call everything above it an
    anchor. If no gap reaches ``decorative_big_number_factor`` the numbers are
    all one size and no anchor can be identified, so none is claimed.

    Measuring the gap rather than comparing to a median matters: a median is
    only meaningful while small lines dominate, and it stops being so on a
    sparse page - exactly where this has to keep working.
    """
    numeric = [ln for ln in lines if _is_numeric(ln.text)]
    if len(numeric) < 2:
        return []
    heights = sorted({round(ln.height, 3) for ln in numeric})
    if len(heights) < 2:
        return []

    best_ratio, split = 0.0, None
    for lower, upper in zip(heights, heights[1:]):
        ratio = upper / lower if lower > 0 else math.inf
        if ratio > best_ratio:
            best_ratio, split = ratio, upper
    if split is None or best_ratio < config.decorative_big_number_factor:
        return []
    # The split came from rounded heights; compare with a tolerance so a line
    # whose height differs in the last float digit is not dropped.
    return [ln for ln in numeric if ln.height >= split - 1e-3]


def join_fragments(fragments: Sequence[RawLine], config: ExtractionConfig) -> str:
    """Concatenate title fragments that were laid out along a curve.

    Fragments sharing a visual row continue one another directly; a fragment
    that starts a new row starts a new word. This recovers most titles, but not
    every one: where a single word is split *across* rows, a space lands inside
    it. Callers mark such titles approximate rather than pretend otherwise.
    """
    ordered = sorted(fragments, key=lambda ln: ln.bbox[1])
    out = ""
    prev: Optional[RawLine] = None
    for frag in ordered:
        piece = frag.text.replace("\n", " ")
        if prev is not None:
            same_row = (
                vertical_overlap_ratio(prev.bbox, frag.bbox) >= config.row_overlap_min
            )
            if not same_row and not out.endswith(" ") and not piece.startswith(" "):
                out += " "
        out += piece
        prev = frag
    return re.sub(r"\s+", " ", out).strip()


def reconstruct_decorative_toc(
    *,
    pdf_page: int,
    raw_lines: Sequence[RawLine],
    page_count: int,
    config: ExtractionConfig,
) -> List[TocEntry]:
    """Rebuild ``lesson -> printed page`` rows from a decorative contents page.

    The page is read as a set of spatial clusters. Each display-size number is
    an anchor; every other fragment joins the anchor whose centre is nearest. A
    small numeric fragment in a cluster is the lesson number, the Arabic-script
    fragments are the title.

    Nothing about the book is hard-coded: how many lessons there are, what they
    are called and which page each starts on all come out of the PDF's own text
    layer.
    """
    lines = usable_lines(raw_lines, config)
    anchors = [
        a
        for a in find_number_anchors(lines, config)
        if 0 < (digits_to_int(a.text) or 0) <= page_count
    ]
    if len(anchors) < config.toc_min_anchors:
        return []

    anchor_ids = {id(a) for a in anchors}
    rest = [ln for ln in lines if id(ln) not in anchor_ids]

    clusters: Dict[int, List[RawLine]] = {id(a): [] for a in anchors}
    for line in rest:
        nearest = min(
            anchors,
            key=lambda a: math.dist(centroid(line.bbox), centroid(a.bbox)),
        )
        clusters[id(nearest)].append(line)

    numeric_fragments = [ln for ln in rest if _is_numeric(ln.text)]
    lesson_numbers = _assign_lesson_numbers(anchors, numeric_fragments)

    entries: List[TocEntry] = []
    for anchor in sorted(anchors, key=lambda a: digits_to_int(a.text) or 0):
        words = [m for m in clusters[id(anchor)] if _PERSIAN_RE.search(m.text)]
        entries.append(
            TocEntry(
                lesson_number=lesson_numbers.get(id(anchor)),
                title=join_fragments(words, config),
                printed_page=digits_to_int(anchor.text),
                source_pdf_page=pdf_page,
                title_is_approximate=len(words) > 2,
            )
        )
    return entries


def _assign_lesson_numbers(
    anchors: Sequence[RawLine], numeric_fragments: Sequence[RawLine]
) -> Dict[int, int]:
    """Match anchors to small numeric fragments one-to-one, across the page.

    Taking the nearest number per anchor independently lets one number serve
    two anchors while a third gets none - and restricting candidates to an
    anchor's own spatial cluster makes that worse, because a lesson number can
    sit closer to the neighbouring page number than to its own.

    So the match is global *and* optimal: the assignment chosen is the one that
    minimises total anchor-to-number distance over the whole page. Taking the
    closest pair first instead (a greedy match) is not the same thing and gets
    it wrong in practice - one anchor grabs a number that its neighbour needed
    more, and both end up mislabelled.

    A contents list is a bijection, and this keeps it one.
    """
    values: List[int] = []
    positions: List[Tuple[float, float]] = []
    for fragment in numeric_fragments:
        value = digits_to_int(fragment.text.strip())
        if value is None or value in values:
            continue  # a repeated numeral cannot label two lessons
        values.append(value)
        positions.append(centroid(fragment.bbox))

    if not anchors or not values:
        return {}

    cost = [
        [math.dist(centroid(a.bbox), pos) for pos in positions] for a in anchors
    ]
    chosen = _min_cost_assignment(cost)
    return {
        id(anchors[i]): values[j] for i, j in chosen.items() if j is not None
    }


#: Above this many anchors the exact search is abandoned for a greedy pass.
#: A contents page never comes close - this is a guard, not a workload.
_EXACT_ASSIGNMENT_LIMIT = 16


def _min_cost_assignment(cost: List[List[float]]) -> Dict[int, Optional[int]]:
    """Minimum-total-cost one-to-one assignment of rows to columns.

    Exact, via a bitmask over columns: ``best[mask]`` is the cheapest way to
    serve the first ``popcount(mask)`` rows using exactly the columns in
    ``mask``. Rows outnumbering columns simply go unassigned.
    """
    n_rows, n_cols = len(cost), len(cost[0])
    if n_cols > _EXACT_ASSIGNMENT_LIMIT:
        return _greedy_assignment(cost)

    size = 1 << n_cols
    best = [math.inf] * size
    back: List[Optional[Tuple[int, int]]] = [None] * size
    best[0] = 0.0
    for mask in range(size):
        if best[mask] == math.inf:
            continue
        row = bin(mask).count("1")
        if row >= n_rows:
            continue
        for col in range(n_cols):
            bit = 1 << col
            if mask & bit:
                continue
            candidate = best[mask] + cost[row][col]
            if candidate < best[mask | bit]:
                best[mask | bit] = candidate
                back[mask | bit] = (mask, col)

    usable = min(n_rows, n_cols)
    final = min(
        (m for m in range(size) if bin(m).count("1") == usable),
        key=lambda m: best[m],
    )
    assignment: Dict[int, Optional[int]] = {i: None for i in range(n_rows)}
    mask = final
    while back[mask] is not None:
        previous, col = back[mask]
        assignment[bin(previous).count("1")] = col
        mask = previous
    return assignment


def _greedy_assignment(cost: List[List[float]]) -> Dict[int, Optional[int]]:
    """Fallback for pathologically wide inputs: closest pair first."""
    pairs = sorted(
        (cost[i][j], i, j)
        for i in range(len(cost))
        for j in range(len(cost[0]))
    )
    assignment: Dict[int, Optional[int]] = {i: None for i in range(len(cost))}
    used_cols: set = set()
    for _, i, j in pairs:
        if assignment[i] is not None or j in used_cols:
            continue
        assignment[i] = j
        used_cols.add(j)
    return assignment
