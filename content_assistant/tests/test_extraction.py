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
