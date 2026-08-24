"""Adversarial tests for concept extraction.

The interesting cases are all the ways a model can be wrong, so each test names
the attack it defends against. Cases A-H come from the phase-5 brief.

Nothing here touches a network or a model. Every reply is scripted, which is
the only way to test the failure modes that matter: a real model rarely
produces a fabricated citation on demand, but the pipeline still has to be
proved against one.
"""

from __future__ import annotations

import unittest

from content_assistant.models.extraction import (
    Asset,
    Block,
    BookIdentity,
    DocumentInfo,
    ExtractionResult,
    Page,
    TocEntry,
)
from content_assistant.structuring.evidence import build_evidence_unit
from content_assistant.structuring.segmentation import segment
from content_assistant.structuring.semantic.concepts import (
    AUTO_ACCEPT_MIN,
    build_prompt,
    compute_confidence,
    extract_concepts,
    flag_duplicates,
    ground_proposals,
    load_prompt,
    material_note,
    render_evidence_blocks,
)
from content_assistant.structuring.semantic.llm import (
    MockLLMClient,
    ModelCallFailed,
)
from content_assistant.structuring.semantic.proposals import (
    REJECT_ALL_CITATIONS_FOREIGN,
    REJECT_NO_CITATION,
    CitationProposal,
    ConceptProposal,
    ConceptResponse,
    admit_proposals,
)

BOOK = "g1-olom"
DOC = BOOK

QUOTE_A = "جانوران غذا میخورند"
QUOTE_B = "جانوران حرکت میکنند"


def lesson_fixture():
    """One lesson, two text blocks and a picture - enough for every case."""
    pages = [
        Page(
            pdf_page=1,
            pdf_page_index=0,
            printed_page=1,
            printed_page_source="page_footer",
            blocks=[
                Block(
                    block_id="/page/0/Text/0",
                    type="Text",
                    text="1 دنیای جانوران",
                    bbox=[10, 20, 400, 40],
                    polygon=[[10, 20], [400, 20], [400, 40], [10, 40]],
                    source="marker",
                )
            ],
        ),
        Page(
            pdf_page=2,
            pdf_page_index=1,
            printed_page=2,
            printed_page_source="page_footer",
            blocks=[
                Block(
                    block_id="/page/1/SectionHeader/0",
                    type="SectionHeader",
                    text="جانوران گوناگون‌اند",
                    bbox=[10, 30, 400, 50],
                    polygon=[[10, 30], [400, 30], [400, 50], [10, 50]],
                    source="marker",
                ),
                Block(
                    block_id="/page/1/Text/1",
                    type="Text",
                    text=f"{QUOTE_A} و {QUOTE_B} و رشد میکنند.",
                    bbox=[10, 60, 400, 90],
                    polygon=[[10, 60], [400, 60], [400, 90], [10, 90]],
                    source="marker",
                ),
            ],
            assets=[
                Asset(
                    asset_id="page_1_Picture_0",
                    pdf_page=2,
                    bbox=[10, 100, 300, 200],
                    path="assets/page_1_Picture_0.jpeg",
                )
            ],
        ),
    ]
    extraction = ExtractionResult(
        document=DocumentInfo(
            source="olom.pdf",
            source_sha256="feed",
            page_count=2,
            page_offset=0,
            book=BookIdentity(
                book_id=BOOK, grade=1, subject="science", language="fa"
            ),
        ),
        pages=pages,
        toc=[TocEntry(lesson_number=1, title="دنیای جانوران", printed_page=1)],
    )
    lessons, sections = segment(extraction)
    unit = build_evidence_unit(extraction, lessons[0], sections)
    return extraction, unit


def proposal(**kwargs):
    defaults = dict(
        label="تغذیه جانوران",
        definition="جانوران برای زنده ماندن غذا میخورند.",
        concept_type="conceptual",
        citations=[CitationProposal(block_id="/page/1/Text/1", quote=QUOTE_A)],
        claimed_evidence_level="explicit",
    )
    defaults.update(kwargs)
    return ConceptProposal(**defaults)


def run(unit, *proposals):
    response = ConceptResponse(concepts=list(proposals))
    admission = admit_proposals(response, sorted(unit.citable_block_ids()))
    return ground_proposals(unit=unit, admission=admission, document_id=DOC)


class ConceptExtractionCases(unittest.TestCase):
    def setUp(self):
        self.extraction, self.unit = lesson_fixture()

    # -- A ---------------------------------------------------------------
    def test_A_valid_citation_and_quote_is_accepted(self):
        result = run(self.unit, proposal())
        self.assertEqual(len(result.concepts), 1)
        concept = result.concepts[0]
        self.assertEqual(concept.evidence_level, "explicit")
        self.assertTrue(result.evidence[0].quote_verified)
        self.assertGreaterEqual(concept.confidence, AUTO_ACCEPT_MIN - 0.35)
        self.assertEqual(concept.lesson_id, self.unit.lesson_id)

    # -- B ---------------------------------------------------------------
    def test_B_citation_outside_the_evidence_unit_is_rejected(self):
        result = run(
            self.unit,
            proposal(
                citations=[
                    CitationProposal(block_id="/page/99/Text/7", quote=QUOTE_A)
                ]
            ),
        )
        self.assertEqual(result.concepts, [])
        self.assertEqual(
            result.admission.rejected[0].reason, REJECT_ALL_CITATIONS_FOREIGN
        )
        self.assertIn("/page/99/Text/7", result.admission.dropped_citations)

    def test_B2_a_foreign_citation_alongside_a_good_one_is_dropped(self):
        result = run(
            self.unit,
            proposal(
                citations=[
                    CitationProposal(block_id="/page/99/Text/7", quote="هرچه"),
                    CitationProposal(block_id="/page/1/Text/1", quote=QUOTE_A),
                ]
            ),
        )
        self.assertEqual(len(result.concepts), 1)
        self.assertEqual(len(result.concepts[0].evidence_ids), 1)
        self.assertIn("/page/99/Text/7", result.admission.dropped_citations)

    # -- C ---------------------------------------------------------------
    def test_C_a_quote_that_is_not_in_the_block_is_downgraded(self):
        result = run(
            self.unit,
            proposal(
                citations=[
                    CitationProposal(
                        block_id="/page/1/Text/1", quote="آهنربا آهن را جذب میکند"
                    )
                ]
            ),
        )
        self.assertEqual(len(result.concepts), 1)
        concept = result.concepts[0]
        self.assertEqual(concept.evidence_level, "inferred")
        self.assertTrue(concept.requires_human_review)
        self.assertFalse(result.evidence[0].quote_verified)

    # -- D ---------------------------------------------------------------
    def test_D_a_near_miss_quote_verifies_fuzzily_with_less_confidence(self):
        exact = run(self.unit, proposal())
        fuzzy = run(
            self.unit,
            proposal(
                citations=[
                    CitationProposal(
                        block_id="/page/1/Text/1",
                        # A paraphrase: every word is in the block, but the
                        # sentence is not, so only token overlap can find it.
                        quote="غذا و حرکت جانوران",
                    )
                ]
            ),
        )
        self.assertEqual(len(fuzzy.concepts), 1)
        self.assertTrue(fuzzy.evidence[0].quote_verified)
        self.assertEqual(fuzzy.evidence[0].match_method, "token_overlap")
        self.assertLess(
            fuzzy.concepts[0].confidence, exact.concepts[0].confidence
        )

    # -- E ---------------------------------------------------------------
    def test_E_a_concept_without_any_citation_is_rejected(self):
        result = run(self.unit, proposal(citations=[]))
        self.assertEqual(result.concepts, [])
        self.assertEqual(result.admission.rejected[0].reason, REJECT_NO_CITATION)

    # -- F ---------------------------------------------------------------
    def test_F_a_concept_from_general_knowledge_cannot_survive(self):
        # The model "knows" mammals are warm-blooded. The book never says so,
        # so there is nothing in the unit to cite and nothing to quote.
        result = run(
            self.unit,
            proposal(
                label="خونگرمی پستانداران",
                definition="پستانداران خونگرم هستند.",
                citations=[
                    CitationProposal(
                        block_id="/page/1/Text/1",
                        quote="پستانداران خونگرم هستند",
                    )
                ],
            ),
        )
        # It is not silently accepted at face value: the quote is absent from
        # the block, so the claim is demoted and flagged for a human.
        self.assertEqual(result.concepts[0].evidence_level, "inferred")
        self.assertTrue(result.concepts[0].requires_human_review)
        self.assertIn(
            "no cited quotation could be found in the book",
            result.concepts[0].review_reasons,
        )

    def test_F2_invented_block_and_invented_quote_leaves_nothing(self):
        result = run(
            self.unit,
            proposal(
                label="چرخه آب",
                citations=[
                    CitationProposal(block_id="/page/42/Text/3", quote="بخار آب")
                ],
            ),
        )
        self.assertEqual(result.concepts, [])

    # -- G ---------------------------------------------------------------
    def test_G_a_visual_only_concept_is_marked_for_visual_review(self):
        result = run(
            self.unit,
            proposal(
                label="تنوع شکل جانوران",
                visual_only=True,
                claimed_evidence_level="explicit",
                citations=[
                    CitationProposal(
                        block_id="/page/1/SectionHeader/0",
                        quote="جانوران گوناگون‌اند",
                        asset_id="page_1_Picture_0",
                    )
                ],
            ),
        )
        concept = result.concepts[0]
        self.assertEqual(concept.evidence_level, "needs_visual_review")
        self.assertTrue(concept.requires_human_review)
        self.assertLessEqual(concept.confidence, 0.5)

    # -- H ---------------------------------------------------------------
    def test_H_near_identical_concepts_are_flagged_not_merged(self):
        result = run(
            self.unit,
            proposal(label="تغذیه جانوران"),
            proposal(
                label="تغذیه جانوران",
                citations=[
                    CitationProposal(block_id="/page/1/Text/1", quote=QUOTE_B)
                ],
            ),
        )
        self.assertEqual(len(result.concepts), 2)  # flagged, never merged
        self.assertTrue(result.duplicate_flags)
        for concept in result.concepts:
            self.assertTrue(
                any("duplicate" in reason for reason in concept.review_reasons)
            )

    def test_H2_distinct_concepts_are_not_flagged(self):
        result = run(
            self.unit,
            proposal(label="تغذیه جانوران"),
            proposal(
                label="حرکت جانوران",
                citations=[
                    CitationProposal(block_id="/page/1/Text/1", quote=QUOTE_B)
                ],
            ),
        )
        self.assertEqual(result.duplicate_flags, {})


class GroundingInvariants(unittest.TestCase):
    """Properties that must hold no matter what a model returns."""

    def setUp(self):
        self.extraction, self.unit = lesson_fixture()

    def test_every_accepted_concept_carries_evidence(self):
        result = run(self.unit, proposal(), proposal(label="حرکت جانوران"))
        for concept in result.concepts:
            self.assertTrue(concept.evidence_ids)

    def test_every_evidence_id_resolves_in_the_result(self):
        result = run(self.unit, proposal())
        known = {item.id for item in result.evidence}
        for concept in result.concepts:
            for evidence_id in concept.evidence_ids:
                self.assertIn(evidence_id, known)

    def test_every_cited_block_belongs_to_this_lesson(self):
        result = run(self.unit, proposal())
        citable = self.unit.citable_block_ids()
        for item in result.evidence:
            self.assertIn(item.block_id, citable)

    def test_ids_are_deterministic_across_runs(self):
        first = run(self.unit, proposal())
        second = run(self.unit, proposal())
        self.assertEqual(
            [c.id for c in first.concepts], [c.id for c in second.concepts]
        )

    def test_the_model_cannot_talk_its_way_to_high_confidence(self):
        # Self-reported 1.0 on an unverifiable claim must stay low.
        result = run(
            self.unit,
            proposal(
                model_confidence=1.0,
                citations=[
                    CitationProposal(block_id="/page/1/Text/1", quote="چیزی که نیست")
                ],
            ),
        )
        self.assertLess(result.concepts[0].confidence, 0.6)

    def test_confidence_is_explained_by_its_components(self):
        result = run(self.unit, proposal())
        breakdown = list(result.confidence_breakdowns.values())[0]
        self.assertIn("quote_verified_exact", breakdown.components)
        self.assertAlmostEqual(
            breakdown.score,
            round(min(1.0, sum(breakdown.components.values())), 4),
            places=3,
        )

    def test_an_empty_reply_is_a_valid_answer(self):
        result = run(self.unit)
        self.assertEqual(result.concepts, [])
        self.assertEqual(result.admission.rejected, [])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.extraction, self.unit = lesson_fixture()

    def test_the_prompt_version_is_the_content_hash(self):
        template = load_prompt()
        self.assertEqual(template.name, "concept_v1")
        self.assertEqual(len(template.version), 8)
        self.assertEqual(load_prompt().version, template.version)

    def test_every_citable_block_id_appears_in_the_prompt(self):
        text = build_prompt(self.unit, load_prompt())
        for block_id in self.unit.citable_block_ids():
            self.assertIn(block_id, text)

    def test_the_prompt_carries_the_lesson_identity(self):
        text = build_prompt(self.unit, load_prompt())
        self.assertIn(self.unit.lesson_title, text)
        self.assertIn(str(self.unit.grade), text)

    def test_no_text_from_another_lesson_leaks_in(self):
        rendered = render_evidence_blocks(self.unit)
        self.assertIn(QUOTE_A, rendered)
        self.assertNotIn("آهنربا", rendered)

    def test_a_thin_lesson_is_told_it_is_thin(self):
        note = material_note(self.unit)
        self.assertIn("چگالی متن", note)
        self.assertIn("کم است", note)

    def test_the_prompt_forbids_outside_knowledge(self):
        text = load_prompt().text
        self.assertIn("دانش عمومی خودت را وارد نکن", text)
        self.assertIn("citation", text)


class EndToEndWithMockTests(unittest.TestCase):
    def setUp(self):
        self.extraction, self.unit = lesson_fixture()

    def test_a_full_pass_runs_without_a_model(self):
        client = MockLLMClient([ConceptResponse(concepts=[proposal()])])
        result, raw, prompt = extract_concepts(
            unit=self.unit, client=client, document_id=DOC
        )
        self.assertEqual(len(result.concepts), 1)
        self.assertEqual(len(raw.concepts), 1)
        self.assertIn(self.unit.lesson_title, prompt)
        self.assertTrue(result.prompt_version.startswith("concept_v1@"))
        self.assertEqual(result.model_id, "mock")

    def test_the_model_is_shown_only_its_own_lesson(self):
        client = MockLLMClient([ConceptResponse(concepts=[])])
        extract_concepts(unit=self.unit, client=client, document_id=DOC)
        sent = client.requests[0].prompt
        self.assertNotIn("/page/42/", sent)
        for block_id in self.unit.citable_block_ids():
            self.assertIn(block_id, sent)

    def test_a_failed_call_is_not_a_lesson_without_concepts(self):
        """Marker's services return ``{}`` once their retries are exhausted,
        and ``{}`` validates into a well-formed reply carrying no concepts -
        indistinguishable from a model that read the lesson and correctly
        found nothing.

        Measured on the objective stage, which shares this seam: ten lessons
        in a row wrote "0 items, validation ok" while every call behind them
        had failed on quota. Saying a lesson holds nothing is a claim about
        the book, and a call that never arrived may not make it.
        """
        client = MockLLMClient([{}])
        with self.assertRaises(ModelCallFailed):
            extract_concepts(
                unit=self.unit, client=client, document_id=DOC
            )

    def test_a_model_that_genuinely_found_nothing_is_believed(self):
        # The other half: an empty list is a real answer and must survive.
        client = MockLLMClient([{"concepts": [], "notes": "هیچ"}])
        result, raw, _ = extract_concepts(
            unit=self.unit, client=client, document_id=DOC
        )
        self.assertEqual(result.concepts, [])
        self.assertEqual(raw.notes, "هیچ")

    def test_images_are_only_sent_when_asked_for(self):
        client = MockLLMClient([ConceptResponse(concepts=[])])
        extract_concepts(unit=self.unit, client=client, document_id=DOC)
        self.assertEqual(client.requests[0].image_paths, [])


if __name__ == "__main__":
    unittest.main()
