"""Tests for vocabulary containment.

The failure this guards against was found in a real run: every citation
verified exactly, and every definition still explained the idea in words the
lesson does not contain - ``مایع``, ``جامد``, ``ذوب``, ``انرژی``. Quote
verification cannot see that, because the quote really was in the book; only
the sentence built around it was not.

The rule these tests lock in: out-of-book wording flags a concept for review
and never invalidates a citation.
"""

from __future__ import annotations

import unittest

from content_assistant.models.content import ContentSchema, BookRef
from content_assistant.structuring.semantic.proposals import (
    CitationProposal,
    ConceptProposal,
    ConceptResponse,
    admit_proposals,
)
from content_assistant.structuring.semantic.concepts import ground_proposals
from content_assistant.text.vocabulary import (
    VocabularyConfig,
    build_vocabulary,
    check_wording,
    find_out_of_vocabulary,
    tokenize,
)
from content_assistant.validation.engine import run_validation
from content_assistant.validation.rules import ValidationContext
from content_assistant.tests.test_concepts import (
    BOOK,
    DOC,
    QUOTE_A,
    lesson_fixture,
)

#: The sentence the real book actually prints.
BOOK_TEXT = "وقتی آب به اندازه کافی سرد شود، یخ میزند. آدم برفی زودتر آب میشود."


class TokenizationTests(unittest.TestCase):
    def test_only_arabic_script_words_are_tokens(self):
        self.assertEqual(tokenize("گرما 12 heat سرما"), ["گرما", "سرما"])

    def test_vocabulary_is_a_set_of_the_lessons_words(self):
        vocab = build_vocabulary(["گرما و سرما", "یخ"])
        self.assertIn("گرما", vocab)
        self.assertIn("یخ", vocab)


class ContainmentTests(unittest.TestCase):
    def setUp(self):
        self.vocab = build_vocabulary([BOOK_TEXT])

    def test_the_books_own_words_are_contained(self):
        self.assertEqual(find_out_of_vocabulary("آب سرد یخ", self.vocab), [])

    def test_scientific_words_the_book_never_uses_are_reported(self):
        missing = find_out_of_vocabulary("مایع جامد انجماد", self.vocab)
        self.assertEqual(missing, sorted(["مایع", "جامد", "انجماد"]))

    def test_inflections_of_a_books_word_are_accepted(self):
        # The book says گرم; a definition saying گرمای or گرمتر is still its
        # vocabulary, and flagging that would be noise, not signal.
        vocab = build_vocabulary(["هوا گرما دارد"])
        self.assertEqual(find_out_of_vocabulary("گرمای", vocab), [])

    def test_function_words_are_never_reported(self):
        missing = find_out_of_vocabulary("برای اینکه مانند", self.vocab)
        self.assertEqual(missing, [])

    def test_short_words_are_skipped(self):
        self.assertEqual(find_out_of_vocabulary("در از به", self.vocab), [])

    def test_results_are_sorted_and_deduplicated(self):
        missing = find_out_of_vocabulary("انرژی انرژی دمای", self.vocab)
        self.assertEqual(missing, sorted(set(missing)))
        self.assertEqual(missing.count("انرژی"), 1)

    def test_strictness_is_configurable(self):
        vocab = build_vocabulary(["گرم"])
        loose = find_out_of_vocabulary(
            "گرمایش", vocab, VocabularyConfig(stem_length=3)
        )
        strict = find_out_of_vocabulary(
            "گرمایش", vocab, VocabularyConfig(stem_length=6)
        )
        self.assertEqual(loose, [])
        self.assertEqual(strict, ["گرمایش"])

    def test_check_wording_covers_label_and_definition_together(self):
        missing = check_wording(
            label="انجماد آب",
            definition="آب سرد یخ میزند",
            lesson_texts=[BOOK_TEXT],
        )
        self.assertIn("انجماد", missing)


class GroundedConceptWordingTests(unittest.TestCase):
    """The four adversarial cases from the brief."""

    def setUp(self):
        self.extraction, self.unit = lesson_fixture()

    def _run(self, **kwargs):
        defaults = dict(
            label="جانوران گوناگون",
            definition="جانوران غذا میخورند",
            concept_type="conceptual",
            citations=[CitationProposal(block_id="/page/1/Text/1", quote=QUOTE_A)],
            claimed_evidence_level="explicit",
        )
        defaults.update(kwargs)
        response = ConceptResponse(concepts=[ConceptProposal(**defaults)])
        admission = admit_proposals(
            response, sorted(self.unit.citable_block_ids())
        )
        return ground_proposals(
            unit=self.unit, admission=admission, document_id=DOC
        )

    def test_a_definition_in_the_books_words_is_accepted_cleanly(self):
        result = self._run()
        concept = result.concepts[0]
        self.assertEqual(concept.out_of_book_vocabulary, [])
        self.assertFalse(
            any("wording" in r for r in concept.review_reasons),
            concept.review_reasons,
        )

    def test_a_scientific_definition_outside_the_book_is_flagged(self):
        result = self._run(
            definition="جانوران برای تامین انرژی متابولیسم پروتئین میکنند"
        )
        concept = result.concepts[0]
        self.assertTrue(concept.out_of_book_vocabulary)
        self.assertTrue(concept.requires_human_review)
        self.assertTrue(any("wording" in r for r in concept.review_reasons))

    def test_a_label_outside_the_vocabulary_is_flagged(self):
        result = self._run(label="متابولیسم", definition="جانوران غذا میخورند")
        concept = result.concepts[0]
        self.assertIn("متابولیسم", concept.out_of_book_vocabulary)
        self.assertTrue(concept.requires_human_review)

    def test_the_citation_survives_a_flagged_definition(self):
        # This is the whole point: wording is a review signal, not a verdict on
        # the evidence. The quote was really in the block, and stays valid.
        result = self._run(definition="انرژی متابولیسم پروتئین")
        concept = result.concepts[0]
        self.assertTrue(concept.out_of_book_vocabulary)
        self.assertEqual(len(concept.evidence_ids), 1)
        self.assertTrue(result.evidence[0].quote_verified)
        self.assertEqual(result.evidence[0].match_method, "exact")
        self.assertEqual(concept.evidence_level, "explicit")

    def test_flagging_does_not_change_the_confidence_score(self):
        clean = self._run().concepts[0]
        flagged = self._run(definition="انرژی متابولیسم").concepts[0]
        self.assertEqual(clean.confidence, flagged.confidence)


class VocabularyValidationRuleTests(unittest.TestCase):
    def _schema(self, concepts):
        return ContentSchema(
            book=BookRef(book_id=BOOK, grade=1, subject="science"),
            concepts=concepts,
        )

    def test_the_rule_reports_a_flagged_concept_for_review(self):
        extraction, unit = lesson_fixture()
        response = ConceptResponse(
            concepts=[
                ConceptProposal(
                    label="متابولیسم",
                    definition="انرژی پروتئین",
                    citations=[
                        CitationProposal(
                            block_id="/page/1/Text/1", quote=QUOTE_A
                        )
                    ],
                    claimed_evidence_level="explicit",
                )
            ]
        )
        admission = admit_proposals(response, sorted(unit.citable_block_ids()))
        grounded = ground_proposals(
            unit=unit, admission=admission, document_id=DOC
        )
        ctx = ValidationContext(
            extraction=extraction, schema_doc=self._schema(grounded.concepts)
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertIn("PEDA005", report.by_code())
        # A review finding must not block the run.
        self.assertTrue(
            all(f.severity != "error" for f in report.findings if f.code == "PEDA005")
        )

    def test_a_clean_concept_raises_no_wording_finding(self):
        extraction, unit = lesson_fixture()
        response = ConceptResponse(
            concepts=[
                ConceptProposal(
                    label="جانوران گوناگون",
                    definition="جانوران غذا میخورند",
                    citations=[
                        CitationProposal(
                            block_id="/page/1/Text/1", quote=QUOTE_A
                        )
                    ],
                    claimed_evidence_level="explicit",
                )
            ]
        )
        admission = admit_proposals(response, sorted(unit.citable_block_ids()))
        grounded = ground_proposals(
            unit=unit, admission=admission, document_id=DOC
        )
        ctx = ValidationContext(
            extraction=extraction, schema_doc=self._schema(grounded.concepts)
        )
        report = run_validation(ctx, stages=["semantic"])
        self.assertNotIn("PEDA005", report.by_code())


class PromptWordingTests(unittest.TestCase):
    def test_the_prompt_states_the_wording_rule_and_shows_both_examples(self):
        from content_assistant.structuring.semantic.concepts import load_prompt

        text = load_prompt().text
        self.assertIn("با واژه‌های خودِ همین درس بنویس", text)
        self.assertIn("مثال درست", text)
        self.assertIn("مثال نادرست", text)
        # The counter-example names the exact words the real run produced.
        for word in ("مایع", "جامد", "ذوب", "انجماد"):
            self.assertIn(word, text)

    def test_the_prompt_pushes_back_on_defaulting_to_conceptual(self):
        from content_assistant.structuring.semantic.concepts import load_prompt

        text = load_prompt().text
        self.assertIn("همه را `conceptual` نگذار", text)
        self.assertIn("procedural", text)


if __name__ == "__main__":
    unittest.main()
