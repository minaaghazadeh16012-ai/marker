"""Deterministic structure: lessons, sections, and what belongs to each.

This is the whole of L1 that runs without a model. It answers three questions
from geometry and the book's own printed signals:

* **Where does each lesson start and end?** From the contents spread L0
  reconstructed. A lesson runs from its own printed page to the page before
  the next lesson.
* **What is the lesson actually called?** Two independent sources disagree
  slightly, and the better one wins - see :func:`resolve_lesson_title`.
* **What sits inside it?** Every block and every image is assigned by printed
  page and vertical position. Nothing is inferred, nothing is invented.

The output is the skeleton a model later hangs meaning on. If this layer is
wrong, everything downstream is wrong in a way no amount of prompting fixes,
which is why it takes its answers from the book rather than from a guess.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from content_assistant.models.content import (
    BoundaryMethod,
    Lesson,
    MaterialProfile,
    PageRange,
    Section,
    TextDensity,
    TitleSource,
    id_slug,
    ordinal_id,
)
from content_assistant.text.persian import digits_to_int
from content_assistant.models.extraction import (
    Block,
    ExtractionResult,
    Page,
    TocEntry,
)

#: Block types that carry readable content (as opposed to running heads/feet).
CONTENT_TEXT_TYPES = frozenset(
    {"Text", "RecoveredText", "SectionHeader", "Caption", "ListGroup", "ListItem"}
)
IMAGE_TYPES = frozenset({"Picture", "Figure", "Diagram"})

#: A lesson-opening page states the title as "<number> <title>". The leading
#: number is decorative and unreliable - measured on a real book it is neither
#: the page number nor consistently the lesson number - so it is stripped and
#: discarded rather than trusted.
_OPENING_TITLE_RE = re.compile(r"^\s*[0-9٠-٩۰-۹]{1,3}\s+(\S.*)$")


class SegmentationConfig:
    """Thresholds for the deterministic layer, gathered in one place."""

    def __init__(
        self,
        text_density_low_max: int = 1000,
        text_density_high_min: int = 2500,
        opening_page_max_chars: int = 80,
        title_match_min: float = 0.6,
        section_fallback: BoundaryMethod = "page_fallback",
    ) -> None:
        self.text_density_low_max = text_density_low_max
        self.text_density_high_min = text_density_high_min
        self.opening_page_max_chars = opening_page_max_chars
        self.title_match_min = title_match_min
        self.section_fallback = section_fallback


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def page_key(page: Page) -> int:
    """Printed page when known, otherwise the 1-based PDF page."""
    return page.printed_page if page.printed_page is not None else page.pdf_page


def block_sort_key(block: Block) -> Tuple[float, float]:
    """Reading position of a block on its page: top first, then rightmost.

    Marker emits its own blocks in reading order but appends rescued ones in a
    separate run, so a page's block list is two streams rather than one. Sorting
    by geometry re-merges them; right-to-left ordering within a row matches the
    script.
    """
    return (block.bbox[1], -block.bbox[2])


def sorted_blocks(page: Page) -> List[Block]:
    return sorted(page.blocks, key=block_sort_key)


def title_similarity(left: str, right: str) -> float:
    """Cheap token-overlap similarity, enough to spot two spellings of a title.

    Deliberately not a fuzzy-matching dependency: this only has to answer
    "are these plausibly the same title?", and a Jaccard score over normalized
    tokens does that without pulling anything new in.
    """
    a = set(id_slug(left).split())
    b = set(id_slug(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_bare_number(text: str) -> bool:
    """A block holding nothing but a numeral.

    Lesson-opening pages print decorative numerals beside the title - measured
    on a real book they are neither the page nor reliably the lesson number -
    and Marker emits each as its own block. They are decoration, so they are
    dropped rather than parsed.
    """
    return digits_to_int(text.strip()) is not None


def opening_page_title(
    page: Page, config: SegmentationConfig
) -> Tuple[Optional[str], Optional[TitleSource]]:
    """Read a lesson's title off its first page.

    Two shapes occur, and both are handled by looking at what the page printed
    rather than by assuming a layout:

    * a **bare opener** - a title, a decorative numeral, nothing else. Once the
      numerals are dropped what remains is the title, verbatim.
    * a **working page** that opens the lesson and then gets on with it. Here
      the title is the page's first printed heading, and the body text below is
      not part of it.

    What a title is never made of is *several blocks joined together*. A
    workbook opens a lesson with four short instructions - ``سلام!``,
    ``کامل کن.``, ``رنگ بزن.`` - and running them into one string produces a
    sentence the book does not print anywhere, offered as the lesson's name.
    Short is not the test; being a single printed thing is. So a page with more
    than one block of its own has to state a heading, or it states no title.

    Returns ``(title, source)``, or ``(None, None)`` when the page states no
    title and the contents list has to be trusted instead.
    """
    blocks = [
        b
        for b in sorted_blocks(page)
        if b.text.strip() and b.type in CONTENT_TEXT_TYPES | {"PageHeader"}
    ]
    meaningful = [b for b in blocks if not _is_bare_number(b.text)]
    if not meaningful:
        return None, None

    if len(meaningful) == 1:
        only = meaningful[0].text.strip()
        if len(only) <= config.opening_page_max_chars:
            match = _OPENING_TITLE_RE.match(only)
            candidate = (match.group(1) if match else only).strip()
            return (
                (candidate or None),
                ("lesson_opening_page" if candidate else None),
            )

    heading = next((b for b in meaningful if b.type == "SectionHeader"), None)
    if heading:
        return heading.text.strip(), "section_header"
    return None, None


def resolve_lesson_title(
    toc_entry: TocEntry,
    opening: Optional[str],
    config: SegmentationConfig,
    opening_source: TitleSource = "lesson_opening_page",
    toc_is_verbatim: bool = False,
) -> Tuple[str, TitleSource, bool, Dict[str, str]]:
    """Pick the lesson title from two independent sources.

    The rule is *prefer whichever source printed the title verbatim*, and which
    one that is depends on how the book set its contents list.

    A **decorative** contents spread sets each title along a curve, one glyph
    per span, and words come back split or re-ordered - exact enough to locate
    a lesson, not to name it. There the opening page wins.

    A **typeset** contents table is ordinary text, and its row is the book's own
    name for the lesson, chosen by the book rather than assembled from whatever
    the first page happened to print. There the contents wins - and it matters,
    because a lesson's first page is often a part divider or a worksheet whose
    only heading names a section rather than the lesson.

    ``toc_is_verbatim`` carries that distinction from
    :attr:`DocumentInfo.toc_source`. It defaults to ``False`` so an artifact
    written before the field existed is read exactly as it was then.

    Whichever loses is kept in ``title_alternatives``, so a reviewer sees the
    disagreement rather than only its outcome.
    """
    alternatives: Dict[str, str] = {}
    if toc_entry.title:
        alternatives["toc"] = toc_entry.title
    if opening:
        alternatives[opening_source] = opening

    if toc_is_verbatim and toc_entry.title:
        return (
            toc_entry.title,
            "toc",
            toc_entry.title_is_approximate,
            alternatives,
        )

    if opening:
        agrees = (
            title_similarity(opening, toc_entry.title or "")
            >= config.title_match_min
        )
        # Approximate only when the two sources disagree *and* the contents
        # version was already flagged unreliable.
        approximate = (
            bool(toc_entry.title)
            and not agrees
            and toc_entry.title_is_approximate
        )
        return opening, opening_source, approximate, alternatives
    if toc_entry.title:
        return toc_entry.title, "toc", toc_entry.title_is_approximate, alternatives
    return (
        f"درس {toc_entry.lesson_number}" if toc_entry.lesson_number else "بدون عنوان",
        "fallback",
        True,
        alternatives,
    )


def classify_density(chars: int, config: SegmentationConfig) -> TextDensity:
    if chars <= config.text_density_low_max:
        return "low"
    if chars >= config.text_density_high_min:
        return "high"
    return "medium"


def build_material_profile(
    pages: Sequence[Page], config: SegmentationConfig
) -> MaterialProfile:
    profile = MaterialProfile(pages=len(pages))
    for page in pages:
        for block in page.blocks:
            if block.type in IMAGE_TYPES:
                profile.images += 1
                continue
            if block.type in CONTENT_TEXT_TYPES and block.text.strip():
                profile.text_chars += len(block.text)
                profile.text_blocks += 1
                if block.type == "SectionHeader":
                    profile.section_headers += 1
                if block.source == "pdfprovider_recovery":
                    profile.recovered_blocks += 1
    profile.text_density = classify_density(profile.text_chars, config)
    return profile


# ---------------------------------------------------------------------------
# lessons
# ---------------------------------------------------------------------------


def lesson_page_bounds(
    toc: Sequence[TocEntry], page_count: int
) -> List[Tuple[TocEntry, int, int]]:
    """Turn contents-page starts into closed ranges.

    A lesson runs to the page before the next one begins, and the last runs to
    the end of the book. Entries without a printed page are skipped rather than
    positioned by guesswork.
    """
    entries = sorted(
        (e for e in toc if e.printed_page is not None),
        key=lambda e: e.printed_page or 0,
    )
    bounds: List[Tuple[TocEntry, int, int]] = []
    for index, entry in enumerate(entries):
        start = entry.printed_page or 0
        if index + 1 < len(entries):
            end = (entries[index + 1].printed_page or start) - 1
        else:
            end = page_count
        bounds.append((entry, start, max(start, end)))
    return bounds


def _lesson_number(printed: Optional[int], taken: Sequence[int] | set) -> int:
    """The number that identifies this lesson inside its book.

    The book's own printed index is used whenever it can serve as an
    identifier, which is what a book with one kind of unit gives you: a
    contents list of fourteen lessons numbered one to fourteen.

    Books with more than one kind of unit restart counting at each kind -
    ``نگاره‌ی ۱`` and ``درس اوّل`` are both "1", and a free lesson prints no
    number at all - so the printed index cannot identify anything on its own.
    Reusing it would give two lessons the same id and silently merge them.
    Where that happens the lesson takes the lowest number still free, which is
    its position in the book; the printed index stays visible in the title.
    """
    number = printed if printed is not None and printed not in taken else None
    if number is None:
        number = 1
        while number in taken:
            number += 1
    return number


def segment_lessons(
    result: ExtractionResult, config: Optional[SegmentationConfig] = None
) -> List[Lesson]:
    config = config or SegmentationConfig()
    book = result.document.book
    book_id = book.book_id or "unknown-book"
    by_printed: Dict[int, Page] = {
        page_key(page): page for page in result.pages
    }

    # A document-level fact, read once: every lesson in one book is named from
    # the same contents list, so asking per lesson would be asking the same
    # question fourteen times.
    toc_is_verbatim = result.document.toc_source == "plain"

    lessons: List[Lesson] = []
    taken: set[int] = set()
    for entry, start, end in lesson_page_bounds(
        result.toc, result.document.page_count
    ):
        pages = [by_printed[p] for p in range(start, end + 1) if p in by_printed]
        if not pages:
            continue
        opening, opening_source = opening_page_title(pages[0], config)
        title, source, approximate, alternatives = resolve_lesson_title(
            entry,
            opening,
            config,
            opening_source or "lesson_opening_page",
            toc_is_verbatim=toc_is_verbatim,
        )
        number = _lesson_number(entry.lesson_number, taken)
        taken.add(number)
        lessons.append(
            Lesson(
                id=ordinal_id(book_id, "lesson", number),
                book_id=book_id,
                grade=book.grade or 0,
                subject=book.subject or "unknown",
                lesson_number=number,
                title=title,
                title_source=source,
                title_is_approximate=approximate,
                title_alternatives=alternatives,
                page_range=PageRange(
                    printed_start=start,
                    printed_end=end,
                    pdf_start=pages[0].pdf_page,
                    pdf_end=pages[-1].pdf_page,
                ),
                block_ids=[b.block_id for page in pages for b in sorted_blocks(page)],
                asset_ids=[a.asset_id for page in pages for a in page.assets],
                material_profile=build_material_profile(pages, config),
            )
        )
    return lessons


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _section_page_range(pages: Sequence[Page]) -> PageRange:
    return PageRange(
        printed_start=page_key(pages[0]),
        printed_end=page_key(pages[-1]),
        pdf_start=pages[0].pdf_page,
        pdf_end=pages[-1].pdf_page,
    )


def segment_sections(
    result: ExtractionResult,
    lessons: Sequence[Lesson],
    config: Optional[SegmentationConfig] = None,
) -> List[Section]:
    """Cut each lesson at its printed section headings.

    Headings that rendered empty are skipped - an empty heading marks a place
    but names nothing, and a section with no title is worse than one page-sized
    section. A lesson with no usable heading at all falls back to one section
    per page, which keeps the evidence granular and is recorded in
    ``boundary_method`` so the difference is never invisible.
    """
    config = config or SegmentationConfig()
    book_id = result.document.book.book_id or "unknown-book"
    by_printed: Dict[int, Page] = {page_key(page): page for page in result.pages}

    sections: List[Section] = []
    for lesson in lessons:
        span = range(
            lesson.page_range.printed_start or 0,
            (lesson.page_range.printed_end or 0) + 1,
        )
        pages = [by_printed[p] for p in span if p in by_printed]
        if not pages:
            continue

        headers = [
            (page, block)
            for page in pages
            for block in sorted_blocks(page)
            if block.type == "SectionHeader" and block.text.strip()
        ]
        if headers:
            sections.extend(
                _sections_from_headers(book_id, lesson, pages, headers)
            )
        else:
            sections.extend(
                _sections_from_pages(book_id, lesson, pages, config.section_fallback)
            )
    return sections


def _has_content_before(
    pages: Sequence[Page],
    first_start: Tuple[int, float, Page, Optional[Block]],
) -> bool:
    """True when the lesson prints something above its first heading."""
    boundary = (first_start[0], first_start[1])
    for page in pages:
        for block in page.blocks:
            if (page.pdf_page, block.bbox[1]) < boundary:
                return True
        for asset in page.assets:
            if (page.pdf_page, asset.bbox[1]) < boundary:
                return True
    return False


def _sections_from_headers(
    book_id: str,
    lesson: Lesson,
    pages: Sequence[Page],
    headers: Sequence[Tuple[Page, Block]],
) -> List[Section]:
    """One section per heading, running until the next heading starts.

    A lesson normally starts before its first heading does: the opening page
    carries the title, the picture and often the whole first activity, and the
    first printed heading arrives pages later. That material belongs to the
    lesson and to no printed section, and a section is the only thing later
    stages read - so leaving it out does not merely mislabel it, it deletes it
    from the evidence a model is given. Measured on four grade-1 books it is
    between 6% and 30% of a book's lesson text.

    So the run above the first heading becomes a section of its own. The book
    never said where it starts, which is exactly what ``page_fallback`` records
    everywhere else, and it is named after the lesson because the lesson's name
    is the only name printed for it.
    """
    starts: List[Tuple[int, float, Page, Optional[Block]]] = [
        (page.pdf_page, block.bbox[1], page, block) for page, block in headers
    ]
    if starts and _has_content_before(pages, starts[0]):
        starts.insert(0, (pages[0].pdf_page, float("-inf"), pages[0], None))

    sections: List[Section] = []
    for order, (pdf_page, top, page, header) in enumerate(starts, start=1):
        if order < len(starts):
            next_page, next_top = starts[order][0], starts[order][1]
        else:
            next_page, next_top = float("inf"), float("inf")

        member_pages: List[Page] = []
        block_ids: List[str] = []
        asset_ids: List[str] = []
        chars = 0
        for candidate in pages:
            included = False
            for block in sorted_blocks(candidate):
                position = (candidate.pdf_page, block.bbox[1])
                if position < (pdf_page, top):
                    continue
                if position >= (next_page, next_top):
                    continue
                block_ids.append(block.block_id)
                included = True
                if block.type in CONTENT_TEXT_TYPES and block.text.strip():
                    chars += len(block.text)
            for asset in candidate.assets:
                position = (candidate.pdf_page, asset.bbox[1])
                if (pdf_page, top) <= position < (next_page, next_top):
                    asset_ids.append(asset.asset_id)
                    included = True
            if included:
                member_pages.append(candidate)

        sections.append(
            Section(
                id=ordinal_id(book_id, "section", lesson.lesson_number, order),
                lesson_id=lesson.id,
                order=order,
                title=lesson.title if header is None else header.text.strip(),
                boundary_method=(
                    "page_fallback" if header is None else "section_header"
                ),
                source_block_id=None if header is None else header.block_id,
                page_range=_section_page_range(member_pages or [page]),
                block_ids=block_ids,
                asset_ids=asset_ids,
                text_chars=chars,
            )
        )
    return sections


def _sections_from_pages(
    book_id: str,
    lesson: Lesson,
    pages: Sequence[Page],
    method: BoundaryMethod,
) -> List[Section]:
    """Fallback when a lesson prints no usable heading."""
    if method == "whole_lesson":
        block_ids = [b.block_id for p in pages for b in sorted_blocks(p)]
        asset_ids = [a.asset_id for p in pages for a in p.assets]
        chars = sum(
            len(b.text)
            for p in pages
            for b in p.blocks
            if b.type in CONTENT_TEXT_TYPES and b.text.strip()
        )
        return [
            Section(
                id=ordinal_id(book_id, "section", lesson.lesson_number, 1),
                lesson_id=lesson.id,
                order=1,
                title=lesson.title,
                boundary_method="whole_lesson",
                page_range=_section_page_range(pages),
                block_ids=block_ids,
                asset_ids=asset_ids,
                text_chars=chars,
            )
        ]

    sections = []
    for order, page in enumerate(pages, start=1):
        chars = sum(
            len(b.text)
            for b in page.blocks
            if b.type in CONTENT_TEXT_TYPES and b.text.strip()
        )
        sections.append(
            Section(
                id=ordinal_id(book_id, "section", lesson.lesson_number, order),
                lesson_id=lesson.id,
                order=order,
                title=f"{lesson.title} — صفحه {page_key(page)}",
                boundary_method="page_fallback",
                page_range=_section_page_range([page]),
                block_ids=[b.block_id for b in sorted_blocks(page)],
                asset_ids=[a.asset_id for a in page.assets],
                text_chars=chars,
            )
        )
    return sections


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def segment(
    result: ExtractionResult, config: Optional[SegmentationConfig] = None
) -> Tuple[List[Lesson], List[Section]]:
    """Full deterministic structuring pass over an L0 artifact."""
    config = config or SegmentationConfig()
    lessons = segment_lessons(result, config)
    sections = segment_sections(result, lessons, config)
    return lessons, sections
