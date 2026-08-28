"""Reading a plain, typeset contents page.

Why this exists
---------------
:func:`~content_assistant.extraction.recovery.reconstruct_decorative_toc` reads
the *decorative* kind of contents spread: the one the science book prints, laid
out along a curve, one glyph per span, with display numerals big enough to
anchor a spatial cluster on. That reader works from geometry because that page
has no rows to read.

The other grade-1 books print the ordinary kind. Every row is a single typeset
line - ``درس دوازدهم قـ ق ــ لـ ل 64`` - the numerals are body-size, and there
is not one display number on the page to anchor on. The same information is
there and every decorative signal is absent, so it needs its own reader.

What a row has to prove before it becomes a lesson boundary
-----------------------------------------------------------
Two things, both printed on the page:

* it **ends in a page number** inside the book, and
* it **names itself a unit** - ``درس <ordinal>``, ``نگاره‌ی <n>``, or a bare
  leading index. A row that only says ``با هم بخوانیم )دریا( 87`` is a piece of
  a lesson, not a lesson, and is skipped.

Nothing about a particular book is hard-coded. The unit *word* is whatever the
book prints in front of its index; the reader learns it from the rows that
carry an index and then accepts the book's other rows that use the same word
(``درس آزاد``, ``درس آخر``) without inventing a number for them.

No model, no lexicon, no network - the same rules as every other L0 module.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from content_assistant.extraction.recovery import (
    group_into_rows,
    usable_lines,
)
from content_assistant.models.extraction import (
    BBox,
    ExtractionConfig,
    RawLine,
    TocEntry,
)
from content_assistant.text.persian import (
    ORDINALS_BY_LENGTH,
    compact,
    digits_to_int,
)

#: A row ends with its page number: ``… خوش‌آمدی 2``.
_TRAILING_PAGE_RE = re.compile(r"^(.*?)[\s.…]*([0-9٠-٩۰-۹]{1,3})\s*$")

#: A bare leading index, as the نگاره rows print it: ``1ــ به خانه‌ی ما …``.
_LEADING_INDEX_RE = re.compile(r"^([0-9]{1,2})(?![0-9])")


def join_row(fragments: Sequence[RawLine]) -> str:
    """Concatenate one row's fragments the way the row is read: right to left.

    :func:`~content_assistant.extraction.recovery.join_fragments` cannot be
    reused here. It orders fragments top-down because it joins a *title set
    along a curve*, where the rows are the fragments. A contents row is the
    opposite shape - one horizontal line, broken into pieces by the PDF - so
    ordering it by height scrambles it, and ``درس هفتم`` ends up behind its own
    page number.
    """
    ordered = sorted(fragments, key=lambda ln: -ln.bbox[2])
    return re.sub(r"\s+", " ", " ".join(ln.text for ln in ordered)).strip()


class ContentsRow:
    """One visual row of a contents page, after its fragments are joined."""

    def __init__(self, text: str, bbox: BBox, fragments: int = 1) -> None:
        self.text = text
        self.bbox = bbox
        #: A row the PDF split into pieces is re-joined with a space at every
        #: break, which can land one inside a word. The page number is exact
        #: either way; the flag says which half of the row to trust.
        self.fragments = fragments
        self.title, self.printed_page = split_trailing_page(text)
        self.unit_word: Optional[str] = None
        self.unit_index: Optional[int] = None

    @property
    def is_unit(self) -> bool:
        return self.unit_word is not None

    def overlaps_column(self, other: "ContentsRow") -> bool:
        """True when two rows share horizontal extent, i.e. sit in one column."""
        left, right = max(self.bbox[0], other.bbox[0]), min(
            self.bbox[2], other.bbox[2]
        )
        if right <= left:
            return False
        narrower = min(self.bbox[2] - self.bbox[0], other.bbox[2] - other.bbox[0])
        return narrower > 0 and (right - left) / narrower >= 0.5


def split_trailing_page(text: str) -> Tuple[str, Optional[int]]:
    """Split ``title … 64`` into its title and its printed page.

    The number has to be at the very end. A numeral anywhere else in the row is
    part of the title - ``نگاره‌ی 1 2`` names unit 1 on page 2, and reading the
    numbers the other way round would put every lesson on the wrong page.
    """
    match = _TRAILING_PAGE_RE.match(text.strip())
    if not match:
        return text.strip(), None
    title, number = match.group(1).strip(), digits_to_int(match.group(2))
    if number is None or not title:
        return text.strip(), None
    return title, number


def unit_marker(
    title: str, config: ExtractionConfig
) -> Optional[Tuple[str, Optional[int]]]:
    """The ``(unit word, index)`` a row declares, or ``None`` if it declares none.

    Three printed shapes, in the order they are tried:

    ``درس دوازدهم``   a word then an ordinal
    ``نگاره‌ی 1``      a word then a numeral
    ``1ــ به خانه…``  a bare leading index, the word being implied by the section

    Matching happens on the compacted title so that a shadda or a space dropped
    inside a word by the PDF does not hide the ordinal. Among several possible
    ordinals the *earliest* one wins, which is what keeps ``یازدهم`` from being
    read as the ``دهم`` inside it.
    """
    text = compact(title)
    if not text:
        return None

    limit = config.plain_toc_unit_word_max_chars
    best: Optional[Tuple[int, int, int]] = None  # (position, -length, value)
    for word, value in ORDINALS_BY_LENGTH:
        position = text.find(word)
        if 0 < position <= limit:
            candidate = (position, -len(word), value)
            if best is None or candidate < best:
                best = candidate
    if best is not None:
        return text[: best[0]], best[2]

    numeral = re.match(r"^([^0-9]{1,%d})([0-9]{1,2})(?![0-9])" % limit, text)
    if numeral:
        return numeral.group(1), int(numeral.group(2))

    leading = _LEADING_INDEX_RE.match(text)
    if leading:
        return "", int(leading.group(1))
    return None


def columns(
    lines: Sequence[RawLine], config: ExtractionConfig
) -> List[Tuple[float, float]]:
    """Find the vertical whitespace corridors that split the page into columns.

    A contents page is usually printed in two columns, and two rows sitting at
    the same height in *different* columns are two rows, not one. Merging them
    produces a row that reads ``درس آزاد … 103 درس چهارم … 37`` and puts a
    lesson on the wrong page, so the columns have to be found before the rows.

    A corridor is a horizontal gap that no ordinary row crosses. Rows that span
    most of the page - a heading, a running title - are excluded from that test
    rather than allowed to weld the columns together: they cross the corridor
    because they sit above it, not because it is not there.
    """
    if not lines:
        return []
    left = min(ln.bbox[0] for ln in lines)
    right = max(ln.bbox[2] for ln in lines)
    extent = right - left
    if extent <= 0:
        return [(left, right)]

    body = [
        ln
        for ln in lines
        if (ln.bbox[2] - ln.bbox[0]) <= config.plain_toc_column_body_width_max * extent
    ]
    if not body:
        return [(left, right)]

    spans = sorted((ln.bbox[0], ln.bbox[2]) for ln in body)
    merged: List[List[float]] = [list(spans[0])]
    minimum_gap = config.plain_toc_column_gap_min * extent
    for start, stop in spans[1:]:
        if start - merged[-1][1] >= minimum_gap:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def _column_of(
    line: RawLine, bands: Sequence[Tuple[float, float]]
) -> int:
    """Which column a line belongs to, decided by its centre."""
    centre = (line.bbox[0] + line.bbox[2]) / 2.0
    for index, (start, stop) in enumerate(bands):
        if start <= centre <= stop:
            return index
    return min(
        range(len(bands)),
        key=lambda i: min(abs(centre - bands[i][0]), abs(centre - bands[i][1])),
    )


def contents_rows(
    raw_lines: Sequence[RawLine], config: ExtractionConfig
) -> List[ContentsRow]:
    """Join each visual row of the page into one string, top to bottom.

    A contents row is one line to a reader but often several fragments to
    pdftext - ``درس او`` and ``ّل آ ا ــ بـ ب 27`` are printed side by side on
    the same row - so the fragments are merged before anything is parsed. The
    merge happens *within a column*: two fragments at the same height in two
    different columns are two rows. Inside a column this reuses the same row
    grouping the recovery pass uses, right-to-left, because it is the same
    script and the same page.
    """
    lines = usable_lines(raw_lines, config)
    bands = columns(lines, config)
    if not bands:
        return []

    per_column: List[List[RawLine]] = [[] for _ in bands]
    for line in lines:
        per_column[_column_of(line, bands)].append(line)

    rows: List[ContentsRow] = []
    for column in per_column:
        for group in group_into_rows(column, config):
            text = join_row(group)
            if not text:
                continue
            rows.append(
                ContentsRow(
                    text=text,
                    bbox=[
                        min(ln.bbox[0] for ln in group),
                        min(ln.bbox[1] for ln in group),
                        max(ln.bbox[2] for ln in group),
                        max(ln.bbox[3] for ln in group),
                    ],
                    fragments=len(group),
                )
            )
    rows.sort(key=lambda row: row.bbox[1])
    return rows


def _label_units(rows: Sequence[ContentsRow], config: ExtractionConfig) -> None:
    """Mark the rows that declare a unit, in two passes.

    The first pass finds the rows that print an index and records the words the
    book puts in front of one. The second accepts the book's *other* rows that
    open with the same word - ``درس آزاد``, ``درس آخر`` are lessons that simply
    have no number, and dropping them would silently merge them into whichever
    lesson came before.
    """
    known: set[str] = set()
    for row in rows:
        marker = unit_marker(row.title, config)
        if marker is None:
            continue
        row.unit_word, row.unit_index = marker
        if row.unit_word:
            known.add(row.unit_word)

    # Longest first, so the match is the most specific one and never depends on
    # the order a set happened to iterate in: the same page must read the same
    # way on every run.
    candidates = sorted(known, key=len, reverse=True)
    for row in rows:
        if row.is_unit or not candidates:
            continue
        text = compact(row.title)
        for word in candidates:
            if text.startswith(word) and len(text) > len(word):
                row.unit_word, row.unit_index = word, None
                break


def _carry_wrapped_pages(rows: Sequence[ContentsRow]) -> None:
    """Give a unit row whose title wrapped the page printed on its next line.

    ``درس بیست‌ویکم`` is too long to share a line with its page number, so the
    number sits on the continuation row underneath it. Without this the lesson
    has no page at all and the lesson before it silently swallows its pages.
    The continuation has to be the next row down *in the same column* and must
    not be a unit of its own.
    """
    for index, row in enumerate(rows):
        if not row.is_unit or row.printed_page is not None:
            continue
        for follower in rows[index + 1 :]:
            if not row.overlaps_column(follower):
                continue
            if follower.is_unit:
                break
            if follower.printed_page is not None:
                row.printed_page = follower.printed_page
            break


def is_plain_contents_page(
    rows: Sequence[ContentsRow], page_count: int, config: ExtractionConfig
) -> bool:
    """Decide whether this page is the book's contents list.

    Three conditions, all of them properties of a contents page rather than
    facts about any particular book: enough rows name a unit *and* a page, the
    pages they name are nearly all different, and between them they reach deep
    into the book. An exercise page can satisfy the first; only a contents page
    satisfies all three.
    """
    pages = [
        row.printed_page
        for row in rows
        if row.is_unit
        and row.printed_page is not None
        and 0 < row.printed_page <= page_count
    ]
    if len(pages) < config.plain_toc_min_unit_rows:
        return False
    if len(set(pages)) < config.plain_toc_distinct_page_ratio * len(pages):
        return False
    return max(pages) >= config.plain_toc_reach_fraction * page_count


def reconstruct_plain_toc(
    *,
    pdf_page: int,
    raw_lines: Sequence[RawLine],
    page_count: int,
    config: ExtractionConfig,
) -> List[TocEntry]:
    """Rebuild ``lesson -> printed page`` rows from a typeset contents page.

    Returns an empty list for any page that is not one, so the caller can offer
    every page and let the evidence decide.

    The ``lesson_number`` left on each entry is the index the *book* prints,
    which restarts at 1 for every kind of unit. Numbering the lessons of the
    whole book is the structuring layer's job and it does it by page order -
    see :func:`~content_assistant.structuring.segmentation.segment_lessons`.
    """
    rows = contents_rows(raw_lines, config)
    _label_units(rows, config)
    _carry_wrapped_pages(rows)

    if not is_plain_contents_page(rows, page_count, config):
        return []

    entries: List[TocEntry] = []
    for row in rows:
        if not row.is_unit or row.printed_page is None:
            continue
        if not 0 < row.printed_page <= page_count:
            continue
        entries.append(
            TocEntry(
                lesson_number=row.unit_index,
                title=row.title,
                printed_page=row.printed_page,
                source_pdf_page=pdf_page,
                title_is_approximate=row.fragments > 1,
            )
        )
    return entries
