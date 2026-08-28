"""Deterministic tests for the L0 extraction pipeline.

No LLM, no network, no PDF, no Marker process. Every fixture is built in
memory, so the whole suite runs in well under a second::

    python -m unittest discover -s content_assistant/tests -t .
"""

from __future__ import annotations

import unittest

from content_assistant.extraction.marker_backend import (
    MarkerBlock,
    MarkerPage,
    html_to_text,
)
from content_assistant.extraction.page_diagnostics import (
    bbox_area,
    compute_diagnostics,
    covered_fraction,
    intersection_area,
    single_char_span_ratio,
)
from content_assistant.extraction.contents import (
    columns,
    contents_rows,
    join_row,
    reconstruct_plain_toc,
    split_trailing_page,
    unit_marker,
)
from content_assistant.extraction.pipeline import build_page_map
from content_assistant.extraction.recovery import (
    _greedy_assignment,
    _min_cost_assignment,
    find_number_anchors,
    is_duplicate,
    join_fragments,
    recover_lines,
    reconstruct_decorative_toc,
    sort_reading_order,
    text_bboxes_of,
    vertical_overlap_ratio,
)
from content_assistant.models.extraction import ExtractionConfig, RawLine
from content_assistant.text import persian

CONFIG = ExtractionConfig()


def line(text: str, x0: float, y0: float, x1: float, y1: float, spans=1, singles=0):
    return RawLine(
        text=text,
        bbox=[x0, y0, x1, y1],
        n_spans=spans,
        n_single_char_spans=singles,
    )


# ---------------------------------------------------------------------------
# Persian normalization
# ---------------------------------------------------------------------------


class PersianNormalizationTests(unittest.TestCase):
    def test_arabic_letter_forms_are_folded_to_persian(self):
        self.assertEqual(persian.normalize_characters("كتاب"), "کتاب")
        self.assertEqual(persian.normalize_characters("يک"), "یک")

    def test_arabic_indic_digits_become_persian_digits(self):
        # Persian digits are preserved as digits, not converted to ASCII.
        self.assertEqual(persian.normalize_digits("٤٥"), "۴۵")
        self.assertEqual(persian.normalize_digits("۴۵"), "۴۵")

    def test_double_alef_is_repaired(self):
        # The PDF emits the lam-alef ligature reversed; a double alef cannot
        # occur in correct Persian, so this repair is unambiguous.
        self.assertEqual(persian.fix_double_alef("باال"), "بالا")
        self.assertEqual(persian.fix_double_alef("باالتر"), "بالاتر")

    def test_words_without_double_alef_are_left_alone(self):
        for word in ("سال", "مال", "حال", "سالم"):
            self.assertEqual(persian.fix_double_alef(word), word)

    def test_ambiguous_lam_alef_is_reported_not_rewritten(self):
        # 'کالس'/'کلاس' needs a lexicon to resolve, so normalize must not guess.
        self.assertEqual(persian.normalize("کالس"), "کالس")
        self.assertIn("کالس", persian.find_suspect_lam_alef("در کالس درس"))

    def test_a_mark_between_alef_and_lam_does_not_hide_a_suspect(self):
        # Measured on the real book: the text layer emits خلّاقیت as خاّلقیت,
        # leaving the shadda sitting between the two swapped letters. A
        # literal "ال" test reads straight past that, so the one word whose
        # mangling actually cost a finding was the one word never reported.
        mangled = "خاّلقیت"          # خ + ا + shadda + ل + ق + ی + ت
        self.assertIn(mangled, persian.find_suspect_lam_alef(mangled))
        self.assertIn(
            mangled, persian.find_suspect_lam_alef(f"فکر و {mangled} و ذوق")
        )

    def test_reporting_the_vocalised_form_does_not_start_repairing_it(self):
        # The module reports ambiguous ligatures; it never rewrites them.
        mangled = "خاّلقیت"
        self.assertEqual(persian.normalize(mangled), mangled)

    def test_tolerating_marks_does_not_widen_what_counts_as_a_suspect(self):
        # Unchanged behaviour, pinned: a leading ال is ordinary and stays out,
        # with or without vocalisation, and a word with no ا-ل pair is silent.
        self.assertEqual(persian.find_suspect_lam_alef("الزم"), [])
        self.assertEqual(persian.find_suspect_lam_alef("اّلزم"), [])
        self.assertEqual(persian.find_suspect_lam_alef("کتاب سنگ آب"), [])
        for word in ("سال", "حال", "سالمت", "کالس"):
            with self.subTest(word=word):
                self.assertIn(word, persian.find_suspect_lam_alef(word))

    def test_isolated_diacritic_is_dropped(self):
        self.assertEqual(persian.drop_isolated_marks("گرم ّ تر"), "گرم  تر")

    def test_attached_diacritic_is_kept(self):
        attached = "مُعلم"
        self.assertEqual(persian.drop_isolated_marks(attached), attached)

    def test_whitespace_is_collapsed_without_touching_zwnj(self):
        text = "می" + persian.ZWNJ + "رود   و    می" + persian.ZWNJ + "آید"
        out = persian.collapse_whitespace(text)
        self.assertEqual(out.count(persian.ZWNJ), 2)
        self.assertNotIn("   ", out)

    def test_zwnj_insertion_is_off_by_default(self):
        # 'میز' is a word, not a verb; inserting ZWNJ blindly would corrupt it.
        self.assertEqual(persian.normalize("میرفتیم"), "میرفتیم")

    def test_zwnj_insertion_when_explicitly_enabled(self):
        cfg = persian.PersianNormalizationConfig(insert_zwnj_heuristics=True)
        self.assertIn(persian.ZWNJ, persian.normalize("میرفتیم", cfg))

    def test_digits_to_int_accepts_all_three_encodings(self):
        self.assertEqual(persian.digits_to_int("45"), 45)
        self.assertEqual(persian.digits_to_int("۴۵"), 45)
        self.assertEqual(persian.digits_to_int("٤٥"), 45)
        self.assertIsNone(persian.digits_to_int("۴۵ب"))
        self.assertIsNone(persian.digits_to_int(""))

    def test_normalize_does_not_reorder_text(self):
        text = "خورشید از پشت کوه"
        self.assertEqual(persian.normalize(text), text)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


class BBoxGeometryTests(unittest.TestCase):
    def test_area_and_intersection(self):
        self.assertEqual(bbox_area([0, 0, 10, 10]), 100)
        self.assertEqual(intersection_area([0, 0, 10, 10], [5, 5, 15, 15]), 25)
        self.assertEqual(intersection_area([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_covered_fraction_is_measured_against_the_inner_box(self):
        self.assertEqual(covered_fraction([0, 0, 10, 10], [[0, 0, 10, 10]]), 1.0)
        self.assertEqual(covered_fraction([0, 0, 10, 10], [[0, 0, 5, 10]]), 0.5)
        self.assertEqual(covered_fraction([0, 0, 10, 10], [[50, 50, 60, 60]]), 0.0)
        self.assertEqual(covered_fraction([0, 0, 10, 10], []), 0.0)

    def test_covered_fraction_is_clamped(self):
        # Overlapping outer boxes must never push the value above 1.0.
        value = covered_fraction(
            [0, 0, 10, 10], [[0, 0, 10, 10], [0, 0, 10, 10]]
        )
        self.assertEqual(value, 1.0)

    def test_vertical_overlap_ratio(self):
        self.assertEqual(vertical_overlap_ratio([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(vertical_overlap_ratio([0, 0, 10, 10], [0, 20, 10, 30]), 0.0)


class DuplicateDetectionTests(unittest.TestCase):
    def test_line_inside_a_rendered_text_block_is_a_duplicate(self):
        self.assertTrue(is_duplicate([10, 10, 20, 20], [[0, 0, 100, 100]], CONFIG))

    def test_line_outside_every_text_block_is_not_a_duplicate(self):
        self.assertFalse(is_duplicate([200, 200, 210, 210], [[0, 0, 100, 100]], CONFIG))

    def test_threshold_is_configurable(self):
        # Exactly half the line is covered.
        half = ([0, 0, 10, 10], [[0, 0, 5, 10]])
        self.assertTrue(
            is_duplicate(*half, ExtractionConfig(duplicate_overlap_min=0.5))
        )
        self.assertFalse(
            is_duplicate(*half, ExtractionConfig(duplicate_overlap_min=0.9))
        )

    def test_empty_text_blocks_do_not_count_as_coverage(self):
        # A SectionHeader that rendered blank has hidden its line, so the line
        # underneath must stay recoverable.
        blocks = [
            {"type": "SectionHeader", "bbox": [0, 0, 100, 100], "text": ""},
            {"type": "Text", "bbox": [0, 0, 100, 100], "text": "hello"},
        ]
        self.assertEqual(text_bboxes_of(blocks), [[0, 0, 100, 100]])
        self.assertEqual(len(text_bboxes_of(blocks[:1])), 0)


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------


class ReadingOrderTests(unittest.TestCase):
    def test_right_to_left_within_a_row(self):
        left = line("چپ", 10, 100, 60, 120)
        right = line("راست", 200, 100, 260, 120)
        ordered = sort_reading_order([left, right], CONFIG)
        self.assertEqual([ln.text for ln in ordered], ["راست", "چپ"])

    def test_rows_run_top_to_bottom(self):
        top = line("بالا", 10, 10, 60, 30)
        bottom = line("پایین", 10, 100, 60, 120)
        ordered = sort_reading_order([bottom, top], CONFIG)
        self.assertEqual([ln.text for ln in ordered], ["بالا", "پایین"])

    def test_two_rows_of_two(self):
        lines = [
            line("a", 10, 100, 60, 120),
            line("b", 200, 100, 260, 120),
            line("c", 10, 10, 60, 30),
            line("d", 200, 10, 260, 30),
        ]
        ordered = sort_reading_order(lines, CONFIG)
        self.assertEqual([ln.text for ln in ordered], ["d", "c", "b", "a"])


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


class RecoveryTests(unittest.TestCase):
    def test_only_uncovered_lines_are_recovered(self):
        raw = [
            line("رندر شده", 10, 10, 90, 30),
            line("گم شده", 10, 200, 90, 220),
        ]
        missing, duplicates = recover_lines(
            raw_lines=raw,
            existing_text_bboxes=[[0, 0, 100, 100]],
            config=CONFIG,
        )
        self.assertEqual(duplicates, 1)
        self.assertEqual([ln.text for ln in missing], ["گم شده"])

    def test_newline_glyph_artifacts_are_dropped(self):
        raw = [line("\n", 10, 10, 11, 11), line("متن", 10, 200, 90, 220)]
        missing, _ = recover_lines(
            raw_lines=raw, existing_text_bboxes=[], config=CONFIG
        )
        self.assertEqual([ln.text for ln in missing], ["متن"])

    def test_everything_is_recovered_when_the_page_rendered_nothing(self):
        raw = [line("یک", 10, 10, 90, 30), line("دو", 10, 60, 90, 80)]
        missing, duplicates = recover_lines(
            raw_lines=raw, existing_text_bboxes=[], config=CONFIG
        )
        self.assertEqual(duplicates, 0)
        self.assertEqual(len(missing), 2)


class RecoveryDetectionTests(unittest.TestCase):
    """A page becomes a candidate through three independent signals."""

    def _diag(self, blocks, raw_lines, page=1):
        return compute_diagnostics(
            pdf_page=page,
            marker_blocks=blocks,
            page_bbox=[0, 0, 100, 100],
            raw_lines=raw_lines,
            config=CONFIG,
        )

    def test_clean_text_page_is_not_a_candidate(self):
        diag = self._diag(
            [{"type": "Text", "bbox": [0, 0, 100, 50], "text": "12345"}],
            [line("12345", 0, 0, 100, 50)],
        )
        self.assertFalse(diag.is_recovery_candidate)
        self.assertEqual(diag.candidate_reasons, [])

    def test_large_picture_coverage_flags_the_page(self):
        diag = self._diag(
            [{"type": "Picture", "bbox": [0, 0, 100, 60], "text": ""}],
            [],
        )
        self.assertTrue(diag.is_recovery_candidate)
        self.assertTrue(diag.has_picture)
        self.assertTrue(any("picture_area_frac" in r for r in diag.candidate_reasons))

    def test_low_recovery_ratio_flags_the_page(self):
        diag = self._diag(
            [{"type": "Text", "bbox": [0, 0, 10, 10], "text": "ab"}],
            [line("abcdefghij", 0, 0, 10, 10)],
        )
        self.assertTrue(diag.is_recovery_candidate)
        self.assertTrue(any("recovery_ratio" in r for r in diag.candidate_reasons))

    def test_no_text_blocks_but_raw_lines_flags_the_page(self):
        diag = self._diag(
            [{"type": "Picture", "bbox": [0, 0, 5, 5], "text": ""}],
            [line("متن هست", 0, 0, 10, 10)],
        )
        self.assertIn("no_text_blocks_but_raw_lines_present", diag.candidate_reasons)

    def test_picture_group_is_recognised(self):
        diag = self._diag(
            [{"type": "PictureGroup", "bbox": [0, 0, 100, 100], "text": ""}], []
        )
        self.assertTrue(diag.has_picture_group)


# ---------------------------------------------------------------------------
# page offset
# ---------------------------------------------------------------------------


def marker_page(index: int, footer: str | None):
    blocks = []
    if footer is not None:
        blocks.append(
            MarkerBlock(
                block_id=f"/page/{index}/PageFooter/0",
                type="PageFooter",
                text=footer,
                bbox=[70, 900, 90, 940],
                polygon=[[70, 900], [90, 900], [90, 940], [70, 940]],
            )
        )
    return MarkerPage(
        pdf_page_index=index,
        bbox=[0, 0, 600, 1000],
        polygon=[[0, 0], [600, 0], [600, 1000], [0, 1000]],
        blocks=blocks,
    )


class PageOffsetTests(unittest.TestCase):
    def test_zero_offset_is_derived_from_footers(self):
        pages = [marker_page(i, str(i + 1)) for i in range(5)]
        offset, evidence, resolved = build_page_map(pages)
        self.assertEqual(offset, 0)
        self.assertEqual(evidence.samples, 5)
        self.assertEqual(evidence.agreement, 1.0)
        self.assertEqual(resolved[0], (1, "page_footer"))

    def test_non_zero_offset(self):
        # Printed page 1 sits on PDF page 7 -> offset 6.
        pages = [marker_page(i, str(i + 1 - 6)) for i in range(6, 12)]
        offset, _, _ = build_page_map(pages)
        self.assertEqual(offset, 6)

    def test_pages_without_a_folio_are_inferred_and_labelled(self):
        pages = [marker_page(0, None)]
        pages += [marker_page(i, str(i + 1)) for i in range(1, 5)]
        _, _, resolved = build_page_map(pages)
        self.assertEqual(resolved[0], (1, "inferred_from_offset"))
        self.assertEqual(resolved[1], (2, "page_footer"))

    def test_no_evidence_means_no_offset_rather_than_a_guess(self):
        blank = [marker_page(i, None) for i in range(3)]
        offset, evidence, resolved = build_page_map(blank)
        self.assertIsNone(offset)
        self.assertIsNone(evidence.value)
        self.assertEqual(resolved, {})

    def test_majority_wins_over_a_stray_number(self):
        pages = [marker_page(i, str(i + 1)) for i in range(5)]
        pages.append(marker_page(5, "99"))  # a figure label, not a folio
        offset, evidence, _ = build_page_map(pages)
        self.assertEqual(offset, 0)
        self.assertLess(evidence.agreement, 1.0)


# ---------------------------------------------------------------------------
# decorative contents page
# ---------------------------------------------------------------------------


class DecorativeTocTests(unittest.TestCase):
    """Mirrors the real geometry: display-size page numbers as anchors, small
    lesson numbers, and title fragments scattered along a curve."""

    def _toc_lines(self):
        return [
            # lesson 1 cluster
            line("10", 405, 73, 462, 175, spans=1),
            line("1", 437, 39, 442, 49, spans=1, singles=1),
            line("زنگ", 394, 42, 428, 61, spans=3, singles=3),
            line(" علوم", 364, 57, 393, 86, spans=4, singles=3),
            # lesson 2 cluster
            line("14", 266, 91, 323, 193, spans=1),
            line("2", 321, 65, 328, 75, spans=1, singles=1),
            line("سالم، به", 258, 60, 312, 80, spans=7, singles=5),
            line(" من", 237, 76, 256, 96, spans=3, singles=2),
            line("نگاه", 219, 91, 238, 120, spans=4, singles=4),
            line("کن", 213, 123, 230, 146, spans=2, singles=2),
            # lesson 3 cluster
            line("18", 185, 203, 242, 305, spans=1),
            line("3", 202, 164, 212, 175, spans=1, singles=1),
            line("دنیای ج", 300, 254, 356, 274, spans=7, singles=6),
            line("انوران", 263, 259, 303, 293, spans=7, singles=6),
        ]

    def test_lesson_to_printed_page_mapping_is_extracted_from_the_page(self):
        entries = reconstruct_decorative_toc(
            pdf_page=8, raw_lines=self._toc_lines(), page_count=104, config=CONFIG
        )
        self.assertEqual(len(entries), 3)
        self.assertEqual(
            [(e.lesson_number, e.printed_page) for e in entries],
            [(1, 10), (2, 14), (3, 18)],
        )
        self.assertTrue(all(e.source_pdf_page == 8 for e in entries))

    def test_titles_are_reassembled_from_fragments(self):
        entries = reconstruct_decorative_toc(
            pdf_page=8, raw_lines=self._toc_lines(), page_count=104, config=CONFIG
        )
        titles = {e.printed_page: e.title for e in entries}
        self.assertEqual(titles[10], "زنگ علوم")
        self.assertEqual(titles[14], "سالم، به من نگاه کن")

    def test_fragments_sharing_a_row_are_joined_without_a_space(self):
        # 'دنیای ج' + 'انوران' is one word split across the curve.
        pieces = [
            line("دنیای ج", 300, 254, 356, 274),
            line("انوران", 263, 259, 303, 293),
        ]
        self.assertEqual(join_fragments(pieces, CONFIG), "دنیای جانوران")

    def test_each_lesson_number_is_used_once(self):
        # Two anchors sitting close to the same small number must not both
        # claim it - a contents list is a bijection, not a nearest-neighbour
        # lookup.
        lines = [
            line("10", 400, 70, 460, 170),
            line("14", 260, 90, 320, 190),
            line("18", 180, 200, 240, 300),
            line("1", 430, 40, 436, 50),
            line("2", 320, 60, 327, 70),
            line("3", 200, 160, 210, 170),
        ]
        entries = reconstruct_decorative_toc(
            pdf_page=8, raw_lines=lines, page_count=104, config=CONFIG
        )
        numbers = [e.lesson_number for e in entries]
        self.assertEqual(len(numbers), len(set(numbers)))
        self.assertNotIn(None, numbers)

    def test_assignment_is_globally_optimal_not_greedy(self):
        # Real geometry from the science book's contents page: the numeral '6'
        # sits closer to the '34' anchor than '5' does, so a greedy match
        # labels both lessons wrongly. The optimal match does not.
        lines = [
            line("34", 384.1, 388.1, 449.4, 489.9),
            line("42", 406.6, 527.6, 471.9, 629.4),
            line("50", 271.6, 599.6, 336.9, 701.4),
            line("5", 479.6, 382.7, 490.2, 392.9),
            line("6", 414.2, 497.6, 423.9, 508.7),
            line("7", 330.0, 579.0, 339.8, 589.7),
        ]
        entries = reconstruct_decorative_toc(
            pdf_page=8, raw_lines=lines, page_count=104, config=CONFIG
        )
        self.assertEqual(
            [(e.lesson_number, e.printed_page) for e in entries],
            [(5, 34), (6, 42), (7, 50)],
        )

    def test_min_cost_assignment_beats_the_greedy_choice(self):
        # Greedy takes (0,0) at cost 1 and is then forced into cost 100.
        cost = [[1.0, 4.0], [2.0, 100.0]]
        self.assertEqual(_min_cost_assignment(cost), {0: 1, 1: 0})
        self.assertEqual(_greedy_assignment(cost), {0: 0, 1: 1})

    def test_min_cost_assignment_leaves_extra_rows_unassigned(self):
        cost = [[1.0], [2.0], [3.0]]
        result = _min_cost_assignment(cost)
        self.assertEqual(result[0], 0)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])

    def test_a_page_without_enough_anchors_yields_nothing(self):
        entries = reconstruct_decorative_toc(
            pdf_page=3,
            raw_lines=[line("10", 405, 73, 462, 175), line("متن", 10, 10, 90, 30)],
            page_count=104,
            config=CONFIG,
        )
        self.assertEqual(entries, [])

    def test_numbers_beyond_the_book_are_not_treated_as_anchors(self):
        # Display-size numerals exist here (there is a clear size gap), but
        # they are far past the last page, so they cannot be destinations.
        lines = [line(str(900 + i), 100 * i, 73, 100 * i + 57, 175) for i in range(4)]
        lines += [line(str(i + 1), 100 * i, 40, 100 * i + 8, 50) for i in range(4)]
        self.assertEqual(
            reconstruct_decorative_toc(
                pdf_page=8, raw_lines=lines, page_count=104, config=CONFIG
            ),
            [],
        )

    def test_numbers_of_one_size_yield_no_anchors(self):
        # Without a size gap there is nothing to tell a page number from a
        # lesson number, so the page is left alone rather than guessed at.
        lines = [line(str(10 + i), 100 * i, 40, 100 * i + 20, 60) for i in range(5)]
        self.assertEqual(find_number_anchors(lines, CONFIG), [])

    def test_decorative_pages_are_detected_by_single_char_spans(self):
        decorative = [
            line("ا", 0, i * 10, 5, i * 10 + 8, spans=1, singles=1)
            for i in range(12)
        ]
        self.assertEqual(single_char_span_ratio(decorative), 1.0)
        diag = compute_diagnostics(
            pdf_page=8,
            marker_blocks=[{"type": "Picture", "bbox": [0, 0, 100, 100], "text": ""}],
            page_bbox=[0, 0, 100, 100],
            raw_lines=decorative,
            config=CONFIG,
        )
        self.assertTrue(diag.is_decorative)

    def test_ordinary_body_text_is_not_decorative(self):
        body = [
            line("یک جمله کامل", 0, i * 20, 400, i * 20 + 15, spans=4)
            for i in range(12)
        ]
        diag = compute_diagnostics(
            pdf_page=45,
            marker_blocks=[
                {"type": "Text", "bbox": [0, 0, 400, 300], "text": "x" * 50}
            ],
            page_bbox=[0, 0, 500, 800],
            raw_lines=body,
            config=CONFIG,
        )
        self.assertFalse(diag.is_decorative)


# ---------------------------------------------------------------------------
# plain (typeset) contents page
# ---------------------------------------------------------------------------


class CompactAndOrdinalTests(unittest.TestCase):
    def test_compact_strips_everything_a_reader_ignores(self):
        self.assertEqual(persian.compact("اوّ ل"), "اول")
        self.assertEqual(persian.compact("سی‌ام"), "سیام")
        self.assertEqual(persian.compact("ــ قرآن"), "قرآن")
        self.assertEqual(persian.compact("درس اوّل:"), "درساول")

    def test_compact_keeps_digits_and_folds_them(self):
        self.assertEqual(persian.compact("نگاره ی ۱"), "نگارهی1")

    def test_ordinals_survive_the_spellings_a_book_prints(self):
        self.assertEqual(persian.ordinal_to_int("اوّل"), 1)
        self.assertEqual(persian.ordinal_to_int("بیست و یکم"), 21)
        self.assertEqual(persian.ordinal_to_int("سی‌ام"), 30)
        self.assertEqual(persian.ordinal_to_int("دوازدهم"), 12)

    def test_a_word_that_is_not_an_ordinal_stays_none(self):
        self.assertIsNone(persian.ordinal_to_int("درس"))
        self.assertIsNone(persian.ordinal_to_int("قرآن"))


class ContentsRowParsingTests(unittest.TestCase):
    def test_page_number_is_taken_from_the_end_of_the_row(self):
        self.assertEqual(
            split_trailing_page("درس دوازدهم قـ ق ــ لـ ل 64"),
            ("درس دوازدهم قـ ق ــ لـ ل", 64),
        )

    def test_an_index_before_the_page_stays_in_the_title(self):
        # نگاره ی 1 is on page 2 - reading the two numbers the other way round
        # would put every unit of the book on the wrong page.
        self.assertEqual(split_trailing_page("نگاره ی 1 2"), ("نگاره ی 1", 2))

    def test_a_row_without_a_trailing_number_has_no_page(self):
        self.assertEqual(
            split_trailing_page("درس بیست و یکم"), ("درس بیست و یکم", None)
        )

    def test_a_row_that_is_only_a_number_is_not_a_titled_row(self):
        self.assertEqual(split_trailing_page("64"), ("64", None))


class UnitMarkerTests(unittest.TestCase):
    def test_word_then_ordinal(self):
        self.assertEqual(unit_marker("درس دوازدهم قـ ق", CONFIG), ("درس", 12))

    def test_word_then_numeral(self):
        self.assertEqual(unit_marker("نگاره ی 1", CONFIG), ("نگارهی", 1))

    def test_bare_leading_index(self):
        self.assertEqual(
            unit_marker("1ــ به خانه‌ی ما خوش‌آمدی", CONFIG), ("", 1)
        )

    def test_the_earliest_ordinal_wins_over_the_one_inside_it(self):
        # یازدهم contains دهم; بیست و یکم contains یکم.
        self.assertEqual(unit_marker("درس یازدهم فـ ف", CONFIG), ("درس", 11))
        self.assertEqual(unit_marker("درس بیست و یکم", CONFIG), ("درس", 21))

    def test_a_broken_spelling_still_matches(self):
        self.assertEqual(unit_marker("درس اوّ ل: به نام خدا", CONFIG), ("درس", 1))

    def test_a_sub_item_declares_no_unit(self):
        self.assertIsNone(unit_marker("با هم بخوانیم )دریا(", CONFIG))
        self.assertIsNone(unit_marker("ــ جدول الفبای فارسی", CONFIG))

    def test_an_index_too_far_into_the_row_is_part_of_the_title(self):
        self.assertIsNone(unit_marker("مهربان‌ترین معلم کلاس 2", CONFIG))


class ContentsColumnTests(unittest.TestCase):
    def _two_column_lines(self):
        return [
            line("فهرست", 52, 63, 505, 112),  # spans both columns
            line("درس بیست و یکم", 52, 176, 280, 196),
            line("درس آزاد محلّ زندگی من 103", 52, 200, 280, 220),
            line("1ــ به خانه‌ی ما خوش‌آمدی 2", 300, 176, 521, 196),
            line("درس اوّل آ ا ــ بـ ب 27", 300, 200, 521, 220),
        ]

    def test_two_columns_are_found(self):
        self.assertEqual(len(columns(self._two_column_lines(), CONFIG)), 2)

    def test_a_heading_spanning_the_page_does_not_weld_the_columns(self):
        bands = columns(self._two_column_lines(), CONFIG)
        self.assertLess(bands[0][1], bands[1][0])

    def test_a_single_column_page_is_one_band(self):
        lines = [
            line("درس اوّل: به نام خدا 14", 136, 139, 487, 162),
            line("درس دوم: نعمت‌های خدا 22", 139, 181, 487, 204),
        ]
        self.assertEqual(len(columns(lines, CONFIG)), 1)

    def test_rows_in_different_columns_are_not_joined(self):
        rows = contents_rows(self._two_column_lines(), CONFIG)
        texts = [row.text for row in rows]
        self.assertIn("درس بیست و یکم", texts)
        self.assertIn("1ــ به خانه‌ی ما خوش‌آمدی 2", texts)


class JoinRowTests(unittest.TestCase):
    def test_fragments_are_read_right_to_left_not_top_down(self):
        # The two halves of one printed row, the right-hand half sitting a
        # hair higher. Ordering by height would put the page number first.
        fragments = [
            line("ّل آ ا ــ بـ ب 27", 298, 472, 487, 496),
            line("درس او", 483, 472, 520, 493),
        ]
        self.assertEqual(join_row(fragments), "درس او ّل آ ا ــ بـ ب 27")


class PlainTocTests(unittest.TestCase):
    """Two columns, a sub-item that is not a lesson, a title that wrapped and
    a lesson the book never numbered - the four shapes the real page has."""

    def _toc_lines(self):
        return [
            line("فهرست", 52, 63, 505, 112),
            # right column
            line("1ــ به خانه‌ی ما خوش‌آمدی 2", 300, 176, 521, 196),
            line("با هم بخوانیم )خدای مهربان( 4", 300, 200, 510, 220),
            line("2ــ بچه‌ها، آماده! 5", 300, 224, 521, 244),
            line("درس اوّل آ ا ــ بـ ب 27", 300, 248, 521, 268),
            line("درس دوم اَ ــ د 30", 300, 272, 521, 292),
            # left column
            line("درس بیست و یکم", 52, 176, 280, 196),
            line("ــ لاک‌پشت و مرغابی‌ها 100", 52, 200, 280, 220),
            line("درس آزاد محلّ زندگی من 103", 52, 224, 280, 244),
            line("درس بیست و دوم پیامبر مهربان 104", 52, 248, 280, 268),
        ]

    def _entries(self):
        return reconstruct_plain_toc(
            pdf_page=4,
            raw_lines=self._toc_lines(),
            page_count=120,
            config=CONFIG,
        )

    def test_every_unit_row_becomes_an_entry(self):
        self.assertEqual(len(self._entries()), 7)

    def test_units_map_to_the_page_printed_beside_them(self):
        pages = sorted(entry.printed_page for entry in self._entries())
        self.assertEqual(pages, [2, 5, 27, 30, 100, 103, 104])

    def test_a_sub_item_is_not_a_lesson(self):
        titles = " | ".join(entry.title for entry in self._entries())
        self.assertNotIn("با هم بخوانیم", titles)

    def test_a_title_that_wrapped_takes_the_page_off_its_next_line(self):
        wrapped = [e for e in self._entries() if "بیست و یکم" in e.title]
        self.assertEqual(len(wrapped), 1)
        self.assertEqual(wrapped[0].printed_page, 100)

    def test_a_lesson_the_book_never_numbered_is_still_a_lesson(self):
        free = [e for e in self._entries() if "آزاد" in e.title]
        self.assertEqual(len(free), 1)
        self.assertIsNone(free[0].lesson_number)

    def test_the_index_kept_is_the_one_the_book_prints(self):
        by_page = {e.printed_page: e.lesson_number for e in self._entries()}
        self.assertEqual(by_page[2], 1)
        self.assertEqual(by_page[27], 1)  # نگاره 1 and درس اول both print "1"
        self.assertEqual(by_page[104], 22)

    def test_the_source_page_is_recorded(self):
        self.assertTrue(all(e.source_pdf_page == 4 for e in self._entries()))


class PlainTocRejectionTests(unittest.TestCase):
    def test_an_ordinary_body_page_is_not_a_contents_page(self):
        lines = [
            line("درس دوم", 52, 60, 200, 80),
            line("شکل‌های زیر را بشمار و بگو 3", 52, 100, 400, 120),
            line("هر دانش‌آموز باید بتواند بشمارد", 52, 130, 400, 150),
        ]
        self.assertEqual(
            reconstruct_plain_toc(
                pdf_page=9, raw_lines=lines, page_count=120, config=CONFIG
            ),
            [],
        )

    def test_rows_pointing_at_one_page_over_and_over_are_not_a_contents_list(self):
        words = ["اول", "دوم", "سوم", "چهارم", "پنجم", "ششم"]
        lines = [
            line("درس " + word + " 7", 52, 60 + 24 * i, 280, 80 + 24 * i)
            for i, word in enumerate(words)
        ]
        self.assertEqual(
            reconstruct_plain_toc(
                pdf_page=4, raw_lines=lines, page_count=120, config=CONFIG
            ),
            [],
        )

    def test_a_list_that_never_reaches_into_the_book_is_not_a_contents_list(self):
        words = ["اول", "دوم", "سوم", "چهارم", "پنجم", "ششم"]
        lines = [
            line("درس " + word + " " + str(i + 2), 52, 60 + 24 * i, 280, 80 + 24 * i)
            for i, word in enumerate(words)
        ]
        self.assertEqual(
            reconstruct_plain_toc(
                pdf_page=4, raw_lines=lines, page_count=400, config=CONFIG
            ),
            [],
        )

    def test_a_decorative_page_yields_nothing_here(self):
        # Glyph-scattered fragments, no row ending in a page number: this page
        # belongs to the decorative reader, and this one declines it.
        lines = [
            line("10", 405, 73, 462, 175),
            line("زنگ", 394, 42, 428, 61),
            line("علوم", 364, 57, 393, 86),
        ]
        self.assertEqual(
            reconstruct_plain_toc(
                pdf_page=3, raw_lines=lines, page_count=104, config=CONFIG
            ),
            [],
        )


# ---------------------------------------------------------------------------
# marker output parsing
# ---------------------------------------------------------------------------


class HtmlToTextTests(unittest.TestCase):
    def test_tags_are_stripped_and_entities_resolved(self):
        self.assertEqual(html_to_text("<p block-type='Text'>سلام</p>"), "سلام")
        self.assertEqual(html_to_text("<p>a&nbsp;b</p>"), "a b")

    def test_empty_block_yields_empty_string(self):
        self.assertEqual(html_to_text(""), "")
        self.assertEqual(html_to_text(None), "")
        self.assertEqual(html_to_text("<p block-type='Text'></p>"), "")


if __name__ == "__main__":
    unittest.main()
