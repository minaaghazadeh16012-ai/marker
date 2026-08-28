"""Tests for the deterministic L1 foundation: ids, segmentation, validation,
evidence units and quote verification.

Every fixture is built in memory. No model is called, no network is touched,
no PDF is opened - which is the point: the whole foundation is provable before
a single token is spent.
"""

from __future__ import annotations

import unittest

from content_assistant.models.content import (
    Concept,
    ContentSchema,
    BookRef,
    Evidence,
    LearningObjective,
    Lesson,
    MaterialProfile,
    Misconception,
    PageRange,
    Relation,
    Section,
    Skill,
    id_slug,
    make_id,
    ordinal_id,
)
from content_assistant.models.extraction import (
    Asset,
    Block,
    BookIdentity,
    DocumentInfo,
    ExtractionResult,
    Page,
    TocEntry,
)
from content_assistant.structuring.evidence import (
    build_evidence_unit,
    build_evidence_units,
)
from content_assistant.structuring.segmentation import (
    segment_lessons,
    SegmentationConfig,
    lesson_page_bounds,
    opening_page_title,
    resolve_lesson_title,
    segment,
    sorted_blocks,
    title_similarity,
)
from content_assistant.structuring.semantic.llm import LLMRequest, MockLLMClient
from content_assistant.structuring.verify import (
    ClaimCitation,
    block_page_index,
    match_quote,
    verify_claim,
)
from content_assistant.validation.engine import (
    ValidationReport,
    render_review_markdown,
    run_validation,
)
from content_assistant.validation.rules import ValidationContext

BOOK = "g1-olom"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def blk(page_index, kind, num, text, y0=0.0, source="marker"):
    return Block(
        block_id=f"/page/{page_index}/{kind}/{num}",
        type=kind,
        text=text,
        bbox=[10.0, y0, 500.0, y0 + 20.0],
        polygon=[[10, y0], [500, y0], [500, y0 + 20], [10, y0 + 20]],
        source=source,
    )


def page(pdf_page, blocks, assets=()):
    return Page(
        pdf_page=pdf_page,
        pdf_page_index=pdf_page - 1,
        printed_page=pdf_page,
        printed_page_source="page_footer",
        blocks=list(blocks),
        assets=list(assets),
    )


def asset(page_index, num, y0=100.0):
    return Asset(
        asset_id=f"page_{page_index}_Picture_{num}",
        pdf_page=page_index + 1,
        bbox=[10.0, y0, 400.0, y0 + 80.0],
        path=f"assets/page_{page_index}_Picture_{num}.jpeg",
    )


def make_extraction(pages, toc, page_count=None):
    return ExtractionResult(
        document=DocumentInfo(
            source="olom.pdf",
            source_sha256="deadbeef",
            page_count=page_count or len(pages),
            page_offset=0,
            book=BookIdentity(
                book_id=BOOK, grade=1, subject="science", language="fa"
            ),
        ),
        pages=list(pages),
        toc=list(toc),
    )


def two_lesson_book():
    """Two lessons: one with headings, one with none (fallback path)."""
    pages = [
        page(1, [blk(0, "Text", 0, "10 زنگ علوم", 40)]),
        page(
            2,
            [
                blk(1, "SectionHeader", 0, "چشمها بسته", 50),
                blk(1, "Text", 1, "با چشم بسته اشیا را بشناس", 90),
            ],
            [asset(1, 0, 200.0)],
        ),
        page(3, [blk(2, "Text", 0, "2 دنیای جانوران", 40)]),
        page(4, [blk(3, "Text", 1, "جانوران غذا میخورند", 60)]),
    ]
    toc = [
        TocEntry(lesson_number=1, title="زنگ علوم", printed_page=1, source_pdf_page=0),
        TocEntry(
            lesson_number=2, title="دنیای ج", printed_page=3, source_pdf_page=0
        ),
    ]
    return make_extraction(pages, toc)


# ---------------------------------------------------------------------------
# phase 1 - identity
# ---------------------------------------------------------------------------


class DeterministicIdTests(unittest.TestCase):
    def test_same_input_always_yields_the_same_id(self):
        first = make_id(BOOK, "concept", "زنده بودن", "/page/26/Text/1")
        second = make_id(BOOK, "concept", "زنده بودن", "/page/26/Text/1")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(f"{BOOK}:concept:"))

    def test_different_content_yields_different_ids(self):
        self.assertNotEqual(
            make_id(BOOK, "concept", "زنده بودن"),
            make_id(BOOK, "concept", "رشد گیاه"),
        )

    def test_ids_are_scoped_by_book_so_grades_never_collide(self):
        self.assertNotEqual(
            make_id("g1-olom", "concept", "آب"),
            make_id("g2-olom", "concept", "آب"),
        )

    def test_persian_spelling_variants_collapse_to_one_id(self):
        # Arabic yeh/kaf and a doubled space must not create a second entity.
        self.assertEqual(
            make_id(BOOK, "concept", "يک  کلاس"),
            make_id(BOOK, "concept", "یک کلاس"),
        )

    def test_id_slug_normalizes_and_trims(self):
        self.assertEqual(id_slug("  يک   کتاب  "), "یک کتاب")

    def test_ordinal_ids_are_readable_and_padded(self):
        self.assertEqual(ordinal_id(BOOK, "lesson", 4), f"{BOOK}:lesson:04")
        self.assertEqual(ordinal_id(BOOK, "section", 4, 2), f"{BOOK}:section:04.02")

    def test_evidence_id_depends_on_block_and_quote(self):
        one = Evidence.build_id(BOOK, "/page/1/Text/0", "الف")
        two = Evidence.build_id(BOOK, "/page/1/Text/0", "ب")
        self.assertNotEqual(one, two)


# ---------------------------------------------------------------------------
# phase 2 - segmentation
# ---------------------------------------------------------------------------


class LessonBoundaryTests(unittest.TestCase):
    def test_a_lesson_runs_until_the_next_one_starts(self):
        toc = [
            TocEntry(lesson_number=1, printed_page=10),
            TocEntry(lesson_number=2, printed_page=14),
            TocEntry(lesson_number=3, printed_page=18),
        ]
        bounds = [(e.lesson_number, s, e2) for e, s, e2 in lesson_page_bounds(toc, 25)]
        self.assertEqual(bounds, [(1, 10, 13), (2, 14, 17), (3, 18, 25)])

    def test_the_last_lesson_runs_to_the_end_of_the_book(self):
        toc = [TocEntry(lesson_number=1, printed_page=5)]
        self.assertEqual(lesson_page_bounds(toc, 40)[0][2], 40)

    def test_entries_without_a_page_are_skipped_not_guessed(self):
        toc = [
            TocEntry(lesson_number=1, printed_page=None),
            TocEntry(lesson_number=2, printed_page=6),
        ]
        bounds = lesson_page_bounds(toc, 10)
        self.assertEqual(len(bounds), 1)
        self.assertEqual(bounds[0][0].lesson_number, 2)

    def test_entries_are_ordered_by_page_not_by_input_order(self):
        toc = [
            TocEntry(lesson_number=2, printed_page=14),
            TocEntry(lesson_number=1, printed_page=10),
        ]
        starts = [s for _, s, _ in lesson_page_bounds(toc, 20)]
        self.assertEqual(starts, [10, 14])


class LessonTitleTests(unittest.TestCase):
    def setUp(self):
        self.config = SegmentationConfig()

    def test_a_bare_opening_page_gives_the_title_verbatim(self):
        opener = page(26, [blk(25, "PageHeader", 0, "24 دنیای جانوران", 50)])
        title, source = opening_page_title(opener, self.config)
        self.assertEqual(title, "دنیای جانوران")
        self.assertEqual(source, "lesson_opening_page")

    def test_decorative_numerals_in_their_own_block_are_dropped(self):
        opener = page(
            80,
            [
                blk(79, "RecoveredText", 0, "12", 30),
                blk(79, "PageHeader", 1, "از خانه تا مدرسه", 50),
                blk(79, "RecoveredText", 2, "0", 70),
            ],
        )
        title, _ = opening_page_title(opener, self.config)
        self.assertEqual(title, "از خانه تا مدرسه")

    def test_a_working_opening_page_falls_back_to_its_first_heading(self):
        opener = page(
            58,
            [
                blk(57, "SectionHeader", 0, "چه میخواهم بسازم؟", 40),
                blk(57, "Text", 1, "ب" * 200, 80),
            ],
        )
        title, source = opening_page_title(opener, self.config)
        self.assertEqual(title, "چه میخواهم بسازم؟")
        self.assertEqual(source, "section_header")

    def test_a_dense_page_without_a_heading_states_no_title(self):
        opener = page(3, [blk(2, "Text", 0, "ب" * 200, 40)])
        self.assertEqual(opening_page_title(opener, self.config), (None, None))

    def test_the_opening_page_beats_a_damaged_contents_entry(self):
        entry = TocEntry(
            lesson_number=4, title="دنیای ج", printed_page=26, title_is_approximate=True
        )
        title, source, approximate, alternatives = resolve_lesson_title(
            entry, "دنیای جانوران", self.config
        )
        self.assertEqual(title, "دنیای جانوران")
        self.assertEqual(source, "lesson_opening_page")
        self.assertEqual(alternatives["toc"], "دنیای ج")

    def test_contents_is_used_when_the_page_states_nothing(self):
        entry = TocEntry(lesson_number=1, title="زنگ علوم", printed_page=10)
        title, source, _, _ = resolve_lesson_title(entry, None, self.config)
        self.assertEqual((title, source), ("زنگ علوم", "toc"))

    def test_a_lesson_with_no_title_anywhere_still_gets_one(self):
        entry = TocEntry(lesson_number=7, title="", printed_page=50)
        title, source, approximate, _ = resolve_lesson_title(entry, None, self.config)
        self.assertEqual(source, "fallback")
        self.assertTrue(approximate)
        self.assertIn("7", title)

    def test_several_short_blocks_are_not_run_together_into_a_title(self):
        """The fabrication this rule exists to stop.

        A workbook opens a lesson with four short instructions. Joined, they
        read as a plausible sentence - and it is a sentence the book prints
        nowhere, offered as the lesson's name. Short is not the test; being a
        single printed thing is.
        """
        opener = page(
            8,
            [
                blk(7, "RecoveredText", 0, "سلام!", 20),
                blk(7, "RecoveredText", 1, "کامل کن.", 40),
                blk(7, "RecoveredText", 2, "کامل کن و رنگ بزن.", 60),
                blk(7, "Text", 3, "بنویس.", 80),
            ],
        )
        self.assertEqual(opening_page_title(opener, self.config), (None, None))

    def test_one_short_block_is_still_read_as_the_title(self):
        # The bare-opener case has to keep working; it is how most lessons in
        # a decorative-contents book are named.
        opener = page(14, [blk(13, "RecoveredText", 0, "22 سالم، به من نگاه کن", 40)])
        title, source = opening_page_title(opener, self.config)
        self.assertEqual(title, "سالم، به من نگاه کن")
        self.assertEqual(source, "lesson_opening_page")

    def test_a_typeset_contents_row_beats_whatever_the_page_printed(self):
        """Which source is verbatim depends on how the book set its contents.

        A typeset table is ordinary text and its row is the book's own name for
        the lesson. The first page is often a part divider or a worksheet whose
        only heading names a section - so preferring the page there renames the
        lesson after something else.
        """
        entry = TocEntry(
            lesson_number=3,
            title="3ــ یک و دو و سه، راه مدرسه",
            printed_page=7,
        )
        title, source, _, alternatives = resolve_lesson_title(
            entry, "نگاره‌ها", self.config, "section_header", toc_is_verbatim=True
        )
        self.assertEqual(title, "3ــ یک و دو و سه، راه مدرسه")
        self.assertEqual(source, "toc")
        # The losing candidate is kept, so the disagreement stays reviewable.
        self.assertEqual(alternatives["section_header"], "نگاره‌ها")

    def test_a_decorative_contents_row_still_loses_to_the_page(self):
        entry = TocEntry(
            lesson_number=4,
            title="دنیای ج",
            printed_page=26,
            title_is_approximate=True,
        )
        title, source, _, _ = resolve_lesson_title(
            entry, "دنیای جانوران", self.config, toc_is_verbatim=False
        )
        self.assertEqual((title, source), ("دنیای جانوران", "lesson_opening_page"))

    def test_an_artifact_written_before_the_field_existed_is_read_as_it_was(self):
        # ``toc_is_verbatim`` defaults to False, so an L0 file with no
        # ``toc_source`` segments exactly as it did before the field existed.
        entry = TocEntry(lesson_number=4, title="دنیای ج", printed_page=26)
        title, source, _, _ = resolve_lesson_title(
            entry, "دنیای جانوران", self.config
        )
        self.assertEqual((title, source), ("دنیای جانوران", "lesson_opening_page"))


    def test_similarity_sees_through_spelling_variants(self):
        self.assertGreater(title_similarity("دنیای جانوران", "دنیای  جانوران"), 0.9)
        self.assertEqual(title_similarity("", "x"), 0.0)


class TocSourceTests(unittest.TestCase):
    """How the book set its contents decides which source names a lesson.

    A document-level fact, read once and applied to every lesson - which is why
    it is worth testing at the segmentation level and not only on the resolver:
    the wiring between the two is the part that can quietly be dropped.
    """

    def _book(self, toc_source):
        """One lesson whose two title sources disagree, so the choice shows."""
        pages = [
            page(8, [blk(7, "PageHeader", 0, "دنیای جانوران", 40)]),
            page(9, [blk(8, "Text", 0, "ب" * 120, 30)]),
        ]
        extraction = make_extraction(
            pages,
            [TocEntry(lesson_number=1, title="دنیای ج", printed_page=8)],
            page_count=9,
        )
        return extraction.model_copy(
            update={
                "document": extraction.document.model_copy(
                    update={"toc_source": toc_source, "page_offset": 0}
                )
            }
        )

    def test_a_typeset_contents_names_the_lesson(self):
        lessons = segment_lessons(self._book("plain"))
        self.assertEqual(lessons[0].title, "دنیای ج")
        self.assertEqual(lessons[0].title_source, "toc")

    def test_a_decorative_contents_leaves_the_page_in_charge(self):
        # The same book, the same two candidates, the opposite answer - which
        # is the whole content of the field.
        lessons = segment_lessons(self._book("decorative"))
        self.assertEqual(lessons[0].title, "دنیای جانوران")
        self.assertEqual(lessons[0].title_source, "lesson_opening_page")

    def test_an_l0_artifact_may_say_nothing_about_its_contents_page(self):
        # Older artifacts have no such field, and must segment as they did.
        lessons = segment_lessons(self._book(None))
        self.assertEqual(lessons[0].title, "دنیای جانوران")

    def test_the_losing_candidate_is_kept_either_way(self):
        for source in ("plain", "decorative"):
            alternatives = segment_lessons(self._book(source))[0].title_alternatives
            self.assertEqual(alternatives["toc"], "دنیای ج")
            self.assertEqual(
                alternatives["lesson_opening_page"], "دنیای جانوران"
            )


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.extraction = two_lesson_book()
        self.lessons, self.sections = segment(self.extraction)

    def test_every_contents_entry_becomes_a_lesson(self):
        self.assertEqual(len(self.lessons), 2)
        self.assertEqual([x.lesson_number for x in self.lessons], [1, 2])

    def test_lesson_ids_are_stable_and_readable(self):
        self.assertEqual(self.lessons[0].id, f"{BOOK}:lesson:01")

    def test_page_ranges_are_contiguous_and_closed(self):
        first = self.lessons[0].page_range
        self.assertEqual((first.printed_start, first.printed_end), (1, 2))
        second = self.lessons[1].page_range
        self.assertEqual((second.printed_start, second.printed_end), (3, 4))

    def test_blocks_are_assigned_to_exactly_one_lesson(self):
        first = set(self.lessons[0].block_ids)
        second = set(self.lessons[1].block_ids)
        self.assertFalse(first & second)
        self.assertTrue(first and second)

    def test_a_heading_starts_a_section(self):
        headed = [s for s in self.sections if s.boundary_method == "section_header"]
        self.assertTrue(headed)
        self.assertEqual(headed[0].title, "چشمها بسته")

    def test_a_lesson_without_headings_falls_back_and_says_so(self):
        second = [s for s in self.sections if s.lesson_id == self.lessons[1].id]
        self.assertTrue(all(s.boundary_method == "page_fallback" for s in second))
        self.assertEqual(len(second), 2)

    def test_material_profile_is_measured_not_estimated(self):
        profile = self.lessons[0].material_profile
        self.assertEqual(profile.pages, 2)
        self.assertEqual(profile.images, 0)  # assets are not blocks
        self.assertGreater(profile.text_chars, 0)
        self.assertEqual(profile.section_headers, 1)

    def test_thin_lessons_are_marked_low_density(self):
        self.assertEqual(self.lessons[0].material_profile.text_density, "low")

    def test_blocks_are_returned_in_reading_order(self):
        # Marker's own blocks and rescued ones arrive as two runs; sorting
        # re-merges them top-to-bottom.
        merged = page(
            9,
            [
                blk(8, "Text", 0, "پایین", 400),
                blk(8, "RecoveredText", 1, "بالا", 50, source="pdfprovider_recovery"),
            ],
        )
        self.assertEqual([b.text for b in sorted_blocks(merged)], ["بالا", "پایین"])


# ---------------------------------------------------------------------------
# phase 3 - validation
# ---------------------------------------------------------------------------


def base_context():
    extraction = two_lesson_book()
    lessons, sections = segment(extraction)
    return ValidationContext(
        extraction=extraction, lessons=lessons, sections=sections
    )


class LessonOpeningTests(unittest.TestCase):
    """A lesson starts before its first heading does, and what it prints there
    is the opening activity - not something to drop on the floor."""

    def _book(self):
        pages = [
            page(
                1,
                [
                    blk(0, "Text", 0, "1 زنگ علوم", 40),
                    blk(0, "Text", 1, "با دوستت درباره‌ی کلاس گفت‌وگو کن", 90),
                ],
                [asset(0, 0, 200.0)],
            ),
            page(
                2,
                [
                    blk(1, "SectionHeader", 0, "چشم‌ها بسته", 50),
                    blk(1, "Text", 1, "با چشم بسته اشیا را بشناس", 90),
                ],
            ),
            page(3, [blk(2, "Text", 0, "2 دنیای جانوران", 40)]),
            page(4, [blk(3, "Text", 1, "جانوران غذا می‌خورند", 60)]),
        ]
        toc = [
            TocEntry(
                lesson_number=1, title="زنگ علوم", printed_page=1, source_pdf_page=0
            ),
            TocEntry(
                lesson_number=2, title="دنیای ج", printed_page=3, source_pdf_page=0
            ),
        ]
        return make_extraction(pages, toc)

    def setUp(self):
        self.lessons, self.sections = segment(self._book())
        self.first = [s for s in self.sections if s.lesson_id == self.lessons[0].id]

    def test_material_above_the_first_heading_still_reaches_a_section(self):
        held = {bid for s in self.first for bid in s.block_ids}
        self.assertIn("/page/0/Text/1", held)

    def test_no_block_of_a_lesson_is_left_out_of_every_section(self):
        held = {bid for s in self.first for bid in s.block_ids}
        self.assertEqual(set(self.lessons[0].block_ids) - held, set())

    def test_the_opening_says_the_book_never_drew_that_boundary(self):
        self.assertEqual(self.first[0].boundary_method, "page_fallback")
        self.assertIsNone(self.first[0].source_block_id)

    def test_the_heading_still_owns_what_it_heads(self):
        heading = [s for s in self.first if s.boundary_method == "section_header"]
        self.assertEqual(len(heading), 1)
        self.assertEqual(heading[0].title, "چشم‌ها بسته")
        self.assertIn("/page/1/Text/1", heading[0].block_ids)

    def test_no_block_is_claimed_by_two_sections(self):
        seen = [bid for s in self.sections for bid in s.block_ids]
        self.assertEqual(len(seen), len(set(seen)))

    def test_a_lesson_whose_heading_opens_it_gains_no_extra_section(self):
        pages = [
            page(1, [blk(0, "SectionHeader", 0, "زنگ علوم", 40),
                     blk(0, "Text", 1, "متن درس", 90)]),
            page(2, [blk(1, "Text", 0, "ادامه‌ی درس", 40)]),
        ]
        toc = [
            TocEntry(
                lesson_number=1, title="زنگ علوم", printed_page=1, source_pdf_page=0
            )
        ]
        _, sections = segment(make_extraction(pages, toc))
        self.assertEqual([s.boundary_method for s in sections], ["section_header"])


class LessonNumberingTests(unittest.TestCase):
    """A book with more than one kind of unit restarts its printed counting,
    so the printed index cannot be the lesson's identity on its own."""

    def _book(self, toc):
        pages = [
            page(n, [blk(n - 1, "Text", 0, f"صفحه {n}", 40)]) for n in range(1, 9)
        ]
        return make_extraction(pages, toc)

    def test_a_book_numbered_once_keeps_the_numbers_it_prints(self):
        lessons, _ = segment(
            self._book(
                [
                    TocEntry(
                        lesson_number=1, title="یک", printed_page=1, source_pdf_page=0
                    ),
                    TocEntry(
                        lesson_number=2, title="دو", printed_page=3, source_pdf_page=0
                    ),
                ]
            )
        )
        self.assertEqual([x.lesson_number for x in lessons], [1, 2])

    def test_two_kinds_of_unit_do_not_collide_on_one_number(self):
        lessons, _ = segment(
            self._book(
                [
                    TocEntry(
                        lesson_number=1,
                        title="نگاره‌ی 1",
                        printed_page=1,
                        source_pdf_page=0,
                    ),
                    TocEntry(
                        lesson_number=2,
                        title="نگاره‌ی 2",
                        printed_page=3,
                        source_pdf_page=0,
                    ),
                    TocEntry(
                        lesson_number=1,
                        title="درس اوّل",
                        printed_page=5,
                        source_pdf_page=0,
                    ),
                    TocEntry(
                        lesson_number=2,
                        title="درس دوم",
                        printed_page=7,
                        source_pdf_page=0,
                    ),
                ]
            )
        )
        self.assertEqual([x.lesson_number for x in lessons], [1, 2, 3, 4])
        self.assertEqual(len({x.id for x in lessons}), 4)

    def test_a_lesson_the_book_never_numbered_still_gets_one(self):
        lessons, _ = segment(
            self._book(
                [
                    TocEntry(
                        lesson_number=1,
                        title="درس اوّل",
                        printed_page=1,
                        source_pdf_page=0,
                    ),
                    TocEntry(
                        lesson_number=None,
                        title="درس آزاد",
                        printed_page=3,
                        source_pdf_page=0,
                    ),
                ]
            )
        )
        self.assertEqual([x.lesson_number for x in lessons], [1, 2])

    def test_the_number_is_the_lesson_position_once_counting_restarts(self):
        # The fifth unit in the book is lesson 5 even though it prints "1".
        toc = [
            TocEntry(
                lesson_number=n, title=f"نگاره {n}", printed_page=n, source_pdf_page=0
            )
            for n in range(1, 5)
        ]
        toc.append(
            TocEntry(
                lesson_number=1, title="درس اوّل", printed_page=5, source_pdf_page=0
            )
        )
        lessons, _ = segment(self._book(toc))
        self.assertEqual(lessons[-1].lesson_number, 5)


class StructureValidationTests(unittest.TestCase):
    def test_a_clean_structure_passes_before_any_model_runs(self):
        report = run_validation(base_context(), stages=["structure"])
        self.assertTrue(report.ok, report.summary())

    def test_missing_book_identity_is_an_error(self):
        ctx = base_context()
        ctx.extraction.document.book = BookIdentity()
        report = run_validation(ctx, stages=["structure"])
        codes = report.by_code()
        self.assertEqual(codes.get("STRUCT001"), 3)  # book_id, grade, subject
        self.assertFalse(report.ok)

    def test_overlapping_lessons_are_caught(self):
        ctx = base_context()
        ctx.lessons[1].page_range.pdf_start = 1
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT004", report.by_code())

    def test_reversed_page_range_is_caught(self):
        ctx = base_context()
        ctx.lessons[0].page_range.pdf_end = 0
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT003", report.by_code())

    def test_a_lesson_running_past_the_book_is_caught(self):
        ctx = base_context()
        ctx.lessons[-1].page_range.pdf_end = 999
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT003", report.by_code())

    def test_unknown_block_reference_is_caught(self):
        ctx = base_context()
        ctx.sections[0].block_ids.append("/page/99/Text/9")
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT007", report.by_code())

    def test_unknown_asset_reference_is_caught(self):
        ctx = base_context()
        ctx.sections[0].asset_ids.append("page_99_Picture_9")
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT008", report.by_code())

    def test_section_pointing_at_a_missing_lesson_is_caught(self):
        ctx = base_context()
        ctx.sections[0].lesson_id = "nope"
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT006", report.by_code())

    def test_duplicate_ids_are_caught(self):
        ctx = base_context()
        ctx.sections[1].id = ctx.sections[0].id
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT002", report.by_code())

    def test_lesson_numbering_out_of_page_order_is_a_warning(self):
        ctx = base_context()
        ctx.lessons[1].lesson_number = 0
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT005", report.by_code())
        self.assertTrue(report.ok)  # a warning must not block

    def test_review_markdown_renders_findings(self):
        ctx = base_context()
        ctx.extraction.document.book = BookIdentity()
        report = run_validation(ctx, stages=["structure"])
        text = render_review_markdown(report)
        self.assertIn("STRUCT001", text)

    def test_empty_report_renders_cleanly(self):
        self.assertIn("موردی یافت نشد", render_review_markdown(ValidationReport()))


def schema_with(concepts=(), objectives=(), evidence=(), **kwargs):
    return ContentSchema(
        book=BookRef(book_id=BOOK, grade=1, subject="science"),
        concepts=list(concepts),
        objectives=list(objectives),
        evidence=list(evidence),
        **kwargs,
    )


def concept(
    cid="c1",
    lesson=f"{BOOK}:lesson:01",
    evidence_ids=("e1",),
    level="inferred",
):
    return Concept(
        id=cid,
        lesson_id=lesson,
        label="زنده بودن",
        evidence_ids=list(evidence_ids),
        evidence_level=level,
    )


def evidence(eid="e1", block="/page/0/Text/0", verified=True, page_no=1, printed=1):
    return Evidence(
        id=eid,
        document_id=BOOK,
        block_id=block,
        pdf_page=page_no,
        printed_page=printed,
        quote="زنگ علوم",
        quote_verified=verified,
    )


class SectionCoverageTests(unittest.TestCase):
    def test_a_lesson_block_in_no_section_is_an_error(self):
        """The hole that deletes evidence without deleting anything.

        A section is the only thing the semantic stages read, so a block in a
        lesson and in no section is not mislabelled - it is removed from what a
        model is ever shown, while every count in the lesson still includes it.
        """
        extraction = two_lesson_book()
        lessons, sections = segment(extraction)
        # Take one block away from every section, leaving it in its lesson.
        stripped = [
            section.model_copy(update={"block_ids": section.block_ids[1:]})
            for section in sections
        ]
        ctx = ValidationContext(
            extraction=extraction, lessons=lessons, sections=stripped
        )
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT012", report.by_code())

    def test_a_book_whose_sections_cover_every_lesson_block_is_silent(self):
        extraction = two_lesson_book()
        lessons, sections = segment(extraction)
        ctx = ValidationContext(
            extraction=extraction, lessons=lessons, sections=sections
        )
        report = run_validation(ctx, stages=["structure"])
        self.assertNotIn("STRUCT012", report.by_code())



class EmptyBookTests(unittest.TestCase):
    def test_a_book_that_yielded_no_lessons_says_so(self):
        """The empty result that looks exactly like a successful one.

        Every other rule is silent on an empty package, because there is
        nothing to object to - so a book whose contents page was never found
        and a book that was never run produce identical clean reports.
        """
        extraction = two_lesson_book()
        ctx = ValidationContext(extraction=extraction, lessons=[], sections=[])
        report = run_validation(ctx, stages=["structure"])
        self.assertIn("STRUCT011", report.by_code())

    def test_a_book_with_lessons_does_not_trigger_it(self):
        extraction = two_lesson_book()
        lessons, sections = segment(extraction)
        ctx = ValidationContext(
            extraction=extraction, lessons=lessons, sections=sections
        )
        self.assertNotIn("STRUCT011", run_validation(ctx, stages=["structure"]).by_code())



class EvidenceValidationTests(unittest.TestCase):
    def test_an_entity_without_evidence_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(concepts=[concept(evidence_ids=())])
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("EVID001", report.by_code())
        self.assertFalse(report.ok)

    def test_evidence_pointing_at_a_missing_block_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept()], evidence=[evidence(block="/page/99/Text/9")]
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("EVID002", report.by_code())

    def test_dangling_evidence_id_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(concepts=[concept(evidence_ids=("missing",))])
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("EVID003", report.by_code())

    def test_explicit_without_a_verified_quote_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept(level="explicit")],
            evidence=[evidence(verified=False)],
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("EVID004", report.by_code())

    def test_explicit_with_a_verified_quote_passes(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept(level="explicit")], evidence=[evidence(verified=True)]
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertNotIn("EVID004", report.by_code())

    def test_evidence_from_outside_the_lesson_is_flagged(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept()],
            evidence=[evidence(block="/page/2/Text/0", page_no=3, printed=3)],
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("EVID005", report.by_code())

    def test_printed_page_inconsistent_with_the_offset_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept()], evidence=[evidence(page_no=1, printed=7)]
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("EVID006", report.by_code())

    def test_an_overwhelmingly_inferred_run_is_flagged(self):
        ctx = base_context()
        concepts = [
            concept(cid=f"c{i}", evidence_ids=("e1",), level="inferred")
            for i in range(6)
        ]
        ctx.schema_doc = schema_with(concepts=concepts, evidence=[evidence()])
        report = run_validation(ctx, stages=["final"])
        self.assertIn("EVID007", report.by_code())


class PedagogicalValidationTests(unittest.TestCase):
    def _objective(self, **kwargs):
        defaults = dict(
            id="o1",
            lesson_id=f"{BOOK}:lesson:01",
            statement="دانشآموز بتواند زنده بودن را تشخیص دهد.",
            performance_verb="تشخیص دادن",
            observable=True,
            concept_ids=["c1"],
            evidence_ids=["e1"],
        )
        defaults.update(kwargs)
        return LearningObjective(**defaults)

    def test_an_unobservable_objective_is_sent_to_review(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept()],
            objectives=[self._objective(observable=False)],
            evidence=[evidence()],
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA001", report.by_code())
        self.assertTrue(report.ok)  # review, not a blocking error

    def test_an_objective_with_no_concept_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept()],
            objectives=[self._objective(concept_ids=[])],
            evidence=[evidence()],
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA002", report.by_code())

    def test_an_objective_citing_an_unknown_concept_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept()],
            objectives=[self._objective(concept_ids=["ghost"])],
            evidence=[evidence()],
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA002", report.by_code())

    def test_duplicate_concepts_in_one_lesson_are_flagged(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept(cid="c1"), concept(cid="c2")], evidence=[evidence()]
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA003", report.by_code())

    def test_a_misconception_must_carry_the_review_flag(self):
        ctx = base_context()
        item = Misconception(
            id="m1",
            concept_id="c1",
            statement="گیاه زنده نیست",
            evidence_ids=["e1"],
            requires_human_review=False,
        )
        ctx.schema_doc = schema_with(
            concepts=[concept()], evidence=[evidence()], misconceptions=[item]
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA004", report.by_code())

    def test_an_explicit_misconception_needs_a_verified_quote(self):
        ctx = base_context()
        item = Misconception(
            id="m1",
            concept_id="c1",
            statement="گیاه زنده نیست",
            evidence_ids=["e1"],
            evidence_level="explicit",
        )
        ctx.schema_doc = schema_with(
            concepts=[concept()],
            evidence=[evidence(verified=False)],
            misconceptions=[item],
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA004", report.by_code())


class FinalValidationTests(unittest.TestCase):
    def _relation(self, **kwargs):
        defaults = dict(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            evidence_ids=["e1"],
        )
        defaults.update(kwargs)
        return Relation(**defaults)

    def test_a_relation_endpoint_that_does_not_exist_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept(cid="c1")],
            evidence=[evidence()],
            relations=[self._relation()],
        )
        report = run_validation(ctx, stages=["final"])
        self.assertIn("FINAL001", report.by_code())

    def test_a_self_relation_is_rejected(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept(cid="c1")],
            evidence=[evidence()],
            relations=[self._relation(source_id="c1", target_id="c1")],
        )
        report = run_validation(ctx, stages=["final"])
        self.assertIn("FINAL001", report.by_code())

    def test_a_prerequisite_cycle_is_detected(self):
        ctx = base_context()
        concepts = [concept(cid="c1"), concept(cid="c2", lesson=f"{BOOK}:lesson:02")]
        relations = [
            self._relation(id="r1", source_id="c1", target_id="c2"),
            self._relation(id="r2", source_id="c2", target_id="c1"),
        ]
        ctx.schema_doc = schema_with(
            concepts=concepts, evidence=[evidence()], relations=relations
        )
        report = run_validation(ctx, stages=["final"])
        self.assertIn("FINAL002", report.by_code())

    def test_an_acyclic_chain_is_accepted(self):
        ctx = base_context()
        concepts = [
            concept(cid="c1"),
            concept(cid="c2", lesson=f"{BOOK}:lesson:02"),
            concept(cid="c3", lesson=f"{BOOK}:lesson:02"),
        ]
        relations = [
            self._relation(id="r1", source_id="c1", target_id="c2"),
            self._relation(id="r2", source_id="c2", target_id="c3"),
        ]
        ctx.schema_doc = schema_with(
            concepts=concepts, evidence=[evidence()], relations=relations
        )
        report = run_validation(ctx, stages=["final"])
        self.assertNotIn("FINAL002", report.by_code())

    def test_a_concept_on_an_unknown_lesson_is_flagged(self):
        ctx = base_context()
        ctx.schema_doc = schema_with(
            concepts=[concept(lesson="ghost")], evidence=[evidence()]
        )
        report = run_validation(ctx, stages=["final"])
        self.assertIn("FINAL003", report.by_code())


# ---------------------------------------------------------------------------
# phase 4 - evidence units
# ---------------------------------------------------------------------------


class EvidenceUnitTests(unittest.TestCase):
    def setUp(self):
        self.extraction = two_lesson_book()
        self.lessons, self.sections = segment(self.extraction)
        self.unit = build_evidence_unit(
            self.extraction, self.lessons[0], self.sections
        )

    def test_the_unit_carries_the_lesson_identity(self):
        self.assertEqual(self.unit.lesson_id, self.lessons[0].id)
        self.assertEqual(self.unit.grade, 1)
        self.assertEqual(self.unit.subject, "science")
        self.assertEqual(self.unit.language, "fa")

    def test_every_citable_block_is_labelled_with_its_id_and_page(self):
        blocks = [b for s in self.unit.sections for b in s.blocks]
        self.assertTrue(blocks)
        for block in blocks:
            self.assertTrue(block.block_id.startswith("/page/"))
            self.assertGreater(block.pdf_page, 0)

    def test_running_heads_and_feet_are_not_citable(self):
        extraction = two_lesson_book()
        extraction.pages[0].blocks.append(blk(0, "PageFooter", 9, "1", 700))
        lessons, sections = segment(extraction)
        unit = build_evidence_unit(extraction, lessons[0], sections)
        self.assertNotIn("/page/0/PageFooter/9", unit.citable_block_ids())

    def test_the_unit_only_contains_its_own_lesson(self):
        ids = self.unit.citable_block_ids()
        other = set(self.lessons[1].block_ids)
        self.assertFalse(ids & other)

    def test_images_travel_with_their_section_and_caption(self):
        extraction = two_lesson_book()
        extraction.pages[1].blocks.append(
            blk(1, "Caption", 5, "شکل ۱: چشمها بسته", 300)
        )
        lessons, sections = segment(extraction)
        unit = build_evidence_unit(extraction, lessons[0], sections)
        images = [i for s in unit.sections for i in s.images]
        self.assertTrue(images)
        self.assertEqual(images[0].caption, "شکل ۱: چشمها بسته")

    def test_a_low_density_lesson_asks_for_its_pictures(self):
        self.lessons[0].material_profile = MaterialProfile(
            text_chars=100, images=30, text_density="low"
        )
        unit = build_evidence_unit(self.extraction, self.lessons[0], self.sections)
        self.assertTrue(unit.needs_images())

    def test_a_dense_lesson_does_not_ask_for_pictures(self):
        self.lessons[0].material_profile = MaterialProfile(
            text_chars=5000, images=30, text_density="high"
        )
        unit = build_evidence_unit(self.extraction, self.lessons[0], self.sections)
        self.assertFalse(unit.needs_images())

    def test_one_unit_is_built_per_lesson(self):
        units = build_evidence_units(self.extraction, self.lessons, self.sections)
        self.assertEqual(len(units), len(self.lessons))


# ---------------------------------------------------------------------------
# phase 4 - quote verification
# ---------------------------------------------------------------------------


class QuoteMatchingTests(unittest.TestCase):
    def test_an_exact_quote_matches_with_its_offsets(self):
        result = match_quote("زنده", "کدامها زنده اند؟")
        self.assertTrue(result.matched)
        self.assertEqual(result.method, "exact")
        self.assertIsNotNone(result.char_start)

    def test_a_quote_differing_only_in_spelling_still_matches(self):
        result = match_quote("يک کلاس", "در یک کلاس درس")
        self.assertTrue(result.matched)
        self.assertEqual(result.method, "normalized")

    def test_a_partly_damaged_quote_matches_on_token_overlap(self):
        result = match_quote(
            "جانوران غذا میخورند و رشد", "جانوران غذا میخورند و بزرگ میشوند"
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.method, "token_overlap")

    def test_an_absent_quote_does_not_match(self):
        self.assertFalse(match_quote("آهنربا", "جانوران غذا میخورند").matched)

    def test_an_empty_quote_never_matches(self):
        self.assertFalse(match_quote("", "متن").matched)
        self.assertFalse(match_quote("متن", "").matched)


class ClaimVerificationTests(unittest.TestCase):
    def setUp(self):
        self.extraction = two_lesson_book()
        self.lessons, self.sections = segment(self.extraction)
        self.unit = build_evidence_unit(
            self.extraction, self.lessons[0], self.sections
        )
        self.args = dict(
            document_id=BOOK,
            allowed_block_ids=self.unit.citable_block_ids(),
            block_pages=block_page_index(self.unit),
            block_text=self.unit.block_text(),
            allowed_asset_ids=self.unit.citable_asset_ids(),
        )

    def test_a_citation_outside_the_unit_is_discarded(self):
        outcome = verify_claim(
            citations=[ClaimCitation(block_id="/page/99/Text/0", quote="هرچه")],
            claimed_level="explicit",
            **self.args,
        )
        self.assertEqual(outcome.rejected_citations, ["/page/99/Text/0"])
        self.assertFalse(outcome.grounded)

    def test_a_verified_quote_keeps_the_explicit_level(self):
        outcome = verify_claim(
            citations=[
                ClaimCitation(block_id="/page/1/SectionHeader/0", quote="چشمها بسته")
            ],
            claimed_level="explicit",
            **self.args,
        )
        self.assertTrue(outcome.grounded)
        self.assertEqual(outcome.evidence_level, "explicit")
        self.assertTrue(outcome.evidence[0].quote_verified)
        self.assertFalse(outcome.demoted)

    def test_an_unfindable_quote_is_demoted_not_deleted(self):
        outcome = verify_claim(
            citations=[
                ClaimCitation(block_id="/page/1/SectionHeader/0", quote="آهنربای بزرگ")
            ],
            claimed_level="explicit",
            **self.args,
        )
        self.assertTrue(outcome.grounded)
        self.assertEqual(outcome.evidence_level, "inferred")
        self.assertTrue(outcome.demoted)
        self.assertFalse(outcome.evidence[0].quote_verified)

    def test_verification_never_promotes_a_claim(self):
        outcome = verify_claim(
            citations=[
                ClaimCitation(block_id="/page/1/SectionHeader/0", quote="چشمها بسته")
            ],
            claimed_level="inferred",
            **self.args,
        )
        self.assertEqual(outcome.evidence_level, "inferred")

    def test_evidence_records_the_page_the_pipeline_knows(self):
        outcome = verify_claim(
            citations=[
                ClaimCitation(block_id="/page/1/SectionHeader/0", quote="چشمها بسته")
            ],
            claimed_level="explicit",
            **self.args,
        )
        item = outcome.evidence[0]
        self.assertEqual(item.pdf_page, 2)
        self.assertEqual(item.printed_page, 2)
        self.assertEqual(item.document_id, BOOK)

    def test_evidence_ids_are_deterministic(self):
        def run():
            return verify_claim(
                citations=[
                    ClaimCitation(
                        block_id="/page/1/SectionHeader/0", quote="چشمها بسته"
                    )
                ],
                claimed_level="explicit",
                **self.args,
            ).evidence[0].id

        self.assertEqual(run(), run())

    def test_an_unknown_asset_reference_is_stripped(self):
        outcome = verify_claim(
            citations=[
                ClaimCitation(
                    block_id="/page/1/SectionHeader/0",
                    quote="چشمها بسته",
                    asset_id="page_99_Picture_9",
                )
            ],
            claimed_level="explicit",
            **self.args,
        )
        self.assertIsNone(outcome.evidence[0].asset_id)


# ---------------------------------------------------------------------------
# phase 4 - the model seam
# ---------------------------------------------------------------------------


class MockLLMTests(unittest.TestCase):
    def test_the_mock_returns_queued_responses_in_order(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            value: str

        client = MockLLMClient([Reply(value="one"), Reply(value="two")])
        request = LLMRequest(prompt="p", response_schema=Reply)
        self.assertEqual(client.complete(request).value, "one")
        self.assertEqual(client.complete(request).value, "two")

    def test_an_unscripted_call_fails_loudly(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            value: str

        client = MockLLMClient()
        with self.assertRaises(AssertionError):
            client.complete(LLMRequest(prompt="p", response_schema=Reply))

    def test_every_request_is_recorded_for_inspection(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            value: str

        client = MockLLMClient([Reply(value="x")])
        client.complete(LLMRequest(prompt="hello", response_schema=Reply))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].prompt, "hello")

    def test_the_client_reports_a_model_id_for_provenance(self):
        self.assertEqual(MockLLMClient(model_id="mock-1").model_id, "mock-1")


if __name__ == "__main__":
    unittest.main()
