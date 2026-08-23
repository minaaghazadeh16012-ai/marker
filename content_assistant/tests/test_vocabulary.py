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
    is_verb_form,
    tokenize,
    word_forms,
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


class PunctuationIsNotPartOfAWordTests(unittest.TestCase):
    """Arabic punctuation lives inside the U+0600-U+06FF block, so a comma used
    to be a word character and ``چشم،`` never matched the book's ``چشم``."""

    def test_arabic_comma_does_not_glue_onto_a_word(self):
        self.assertEqual(tokenize("چشم، گوش، دست"), ["چشم", "گوش", "دست"])

    def test_arabic_semicolon_and_question_mark_are_not_letters(self):
        self.assertEqual(tokenize("کنید؛ چرا؟ بله"), ["کنید", "چرا", "بله"])

    def test_persian_and_arabic_digits_are_not_words(self):
        self.assertEqual(tokenize("صفحه ۱۲۳۴ و ٥٦٧٨"), ["صفحه", "و"])

    def test_a_comma_in_the_book_still_vouches_for_the_bare_word(self):
        # The regression that produced two of the run's false positives.
        vocab = build_vocabulary(["به کمک چشم، گوش، و پوست"])
        self.assertEqual(find_out_of_vocabulary("چشم گوش پوست", vocab), [])

    def test_a_comma_on_the_candidate_side_is_stripped_too(self):
        vocab = build_vocabulary(["به کمک چشمها و گوشها"])
        self.assertEqual(find_out_of_vocabulary("چشمها، گوشها،", vocab), [])

    def test_tatweel_is_removed_rather_than_treated_as_a_boundary(self):
        # A kashida stretches a letter for justification; it is the same word,
        # and splitting there would invent two words that are not in the text.
        vocab = build_vocabulary(["بله درست است"])
        self.assertEqual(find_out_of_vocabulary("بـــله", vocab), [])

    def test_vocalised_words_are_not_split_by_their_harakat(self):
        # Marks sit on letters; cutting there would shatter the word.
        self.assertEqual(len(tokenize("دقّت")), 1)


class ZwnjSymmetryTests(unittest.TestCase):
    """A PDF text layer loses ZWNJ; a model writes it. Both spellings have to
    reach each other, or the comparison is about typography not vocabulary."""

    def test_word_forms_yields_both_readings(self):
        forms = word_forms("دانش\u200cآموز")
        self.assertIn("دانشآموز", forms)
        self.assertIn("دانش", forms)
        self.assertIn("آموز", forms)

    def test_book_without_zwnj_covers_a_definition_with_zwnj(self):
        vocab = build_vocabulary(["دانشآموزان به مدرسه میروند"])
        self.assertEqual(find_out_of_vocabulary("دانش\u200cآموزان", vocab), [])

    def test_book_with_zwnj_covers_a_definition_without_zwnj(self):
        vocab = build_vocabulary(["دانش\u200cآموزان به مدرسه میروند"])
        self.assertEqual(find_out_of_vocabulary("دانشآموزان", vocab), [])

    def test_a_genuinely_foreign_word_is_still_reported_either_way(self):
        vocab = build_vocabulary(["دانش\u200cآموزان به مدرسه میروند"])
        self.assertIn("متابولیسم", find_out_of_vocabulary("متابولیسم", vocab))


class ShortBookWordCoverTests(unittest.TestCase):
    """A book word shorter than ``stem_length`` used to vouch for nothing:
    ``'هوا'[:4]`` is still ``'هوا'`` and never equals ``'هوای'[:4]``."""

    def test_a_three_letter_book_word_covers_its_inflection(self):
        vocab = build_vocabulary(["در اطراف ما هوا وجود دارد"])
        self.assertEqual(find_out_of_vocabulary("هوای پاکیزه", vocab), ["پاکیزه"])

    def test_the_cover_is_one_character_wide_not_unlimited(self):
        # The guarantee the original docstring made: at stem_length 6 a
        # three-letter word must not reach گرمایش.
        vocab = build_vocabulary(["گرم"])
        self.assertEqual(
            find_out_of_vocabulary("گرمایش", vocab, VocabularyConfig(stem_length=6)),
            ["گرمایش"],
        )

    def test_a_short_book_word_does_not_cover_an_unrelated_word(self):
        vocab = build_vocabulary(["هوا"])
        self.assertEqual(find_out_of_vocabulary("هزینه", vocab), ["هزینه"])


class VerbFormsAreOutOfScopeTests(unittest.TestCase):
    """Persian conjugation is suppletive, so ``شستن`` cannot be matched to the
    book's ``بشویید`` without a verb lexicon. Verbs are skipped instead."""

    def test_infinitives_and_conjugations_are_recognised(self):
        for word in (
            "کردن", "بودن", "دادن", "چسبیدن", "ساییدن", "ریختن", "شستن",
            "گرفتن", "داشتن", "میکنند", "نمیچسبند", "نزنید", "کنند",
            "شوند", "هستند", "نیستند", "توان",
        ):
            with self.subTest(word=word):
                self.assertTrue(is_verb_form(word))

    def test_nouns_that_merely_end_like_an_infinitive_are_not_verbs(self):
        # The trap: a bare (تن|دن)$ rule exempts معدن from a lesson on rocks.
        for word in ("معدن", "تمدن", "متن", "بتن"):
            with self.subTest(word=word):
                self.assertFalse(is_verb_form(word))

    def test_the_words_this_check_exists_to_catch_are_not_verbs(self):
        for word in ("مایع", "جامد", "انجماد", "انرژی", "متابولیسم",
                     "بررسی", "خلاقیت", "اهمیت", "مختلف"):
            with self.subTest(word=word):
                self.assertFalse(is_verb_form(word))

    def test_a_verb_the_lesson_never_uses_is_not_reported(self):
        vocab = build_vocabulary(["سنگها را به هم بسایید"])
        self.assertEqual(find_out_of_vocabulary("ساییدن سنگها", vocab), [])

    def test_skipping_verbs_can_be_switched_off_and_only_gets_stricter(self):
        vocab = build_vocabulary(["سنگها را به هم بسایید"])
        strict = find_out_of_vocabulary(
            "ساییدن سنگها", vocab, VocabularyConfig(skip_verb_forms=False)
        )
        self.assertEqual(strict, ["ساییدن"])

    def test_an_imported_noun_beside_a_skipped_verb_still_surfaces(self):
        # Skipping verbs must not become a way for content words to hide.
        vocab = build_vocabulary(["آب سرد میشود"])
        self.assertEqual(
            find_out_of_vocabulary("انجماد مایع میشود", vocab),
            sorted(["انجماد", "مایع"]),
        )


class VerbSkipDoesNotSwallowNounsTests(unittest.TestCase):
    """The dangerous direction of the verb filter, pinned down.

    A verb wrongly checked produces a flag, and a flag costs a glance. A noun
    wrongly called a verb is never checked at all, and nothing about the run
    says so. Only the second failure is silent, so it is the one under test.

    An earlier draft allowed any two characters after a light-verb stem, which
    made ``زن`` + ``ده`` a verb - and ``زنده`` is the word lesson 4's own
    concept is built on. These are the words that broke it.
    """

    def test_nouns_beginning_with_a_light_verb_stem_are_not_verbs(self):
        for word in (
            "زنده",      # زن + ده, and the subject of a whole lesson
            "دهان",      # ده + ان, in a lesson about the senses
            "دارو", "داروی", "کندو", "کنار", "هسته", "گیره", "بودجه",
            "زندگی", "زنبور", "کنجکاوی", "دانش", "توانایی", "دانه",
        ):
            with self.subTest(word=word):
                self.assertFalse(is_verb_form(word))

    def test_the_personal_endings_are_all_still_caught(self):
        for word in (
            "کنند", "کنید", "کنیم", "هستند", "دارند", "شوند", "نیستند",
            "دهند", "نزنید", "توان",
        ):
            with self.subTest(word=word):
                self.assertTrue(is_verb_form(word))

    def test_a_content_word_shaped_like_a_conjugation_is_still_reported(self):
        # End to end: the lesson never says زنده, so a definition that does
        # must be flagged rather than quietly skipped as a verb.
        vocab = build_vocabulary(["گیاهان رشد میکنند و بزرگ میشوند"])
        self.assertIn("زنده", find_out_of_vocabulary("موجود زنده", vocab))

    def test_a_lesson_that_uses_the_word_still_vouches_for_it(self):
        vocab = build_vocabulary(["چیزها را در دو گروه زنده و غیر زنده بگذار"])
        self.assertEqual(find_out_of_vocabulary("گروه زنده", vocab), [])


class BookRunRegressionTests(unittest.TestCase):
    """The four findings that survived a full 14-lesson run, and the noise that
    surrounded them. Each case is the real lesson wording, shortened."""

    def test_a_synonym_the_book_does_not_use_is_still_flagged(self):
        # Lesson 7 says گوناگون throughout; the model wrote مختلف.
        vocab = build_vocabulary(["سنگها گوناگوناند و در جاهای گوناگونی هستند"])
        self.assertIn("مختلف", find_out_of_vocabulary("جاهای مختلف", vocab))

    def test_an_abstraction_above_the_books_register_is_still_flagged(self):
        # Lesson 8 says فکر و ذوق و هنرمندی; the model wrote خلاقیت.
        vocab = build_vocabulary(["به فکر و ذوق و هنرمندی نیاز است"])
        self.assertIn("خلاقیت", find_out_of_vocabulary("فکر و خلاقیت", vocab))

    def test_academic_framing_words_are_still_flagged(self):
        vocab = build_vocabulary(["آن را نقاشی کن و با دقت نگاه کردی"])
        missing = find_out_of_vocabulary("بررسی اهمیت آن", vocab)
        self.assertIn("بررسی", missing)
        self.assertIn("اهمیت", missing)

    def test_the_verb_noise_that_drowned_them_is_gone(self):
        # 22 of the 30 false positives in the first run were verb forms of a
        # verb the lesson plainly uses. None of them is reported now.
        vocab = build_vocabulary(
            ["بچهها با کنجکاوی نگاه کردی و چیزهای تازه پیدا میکنیم"]
        )
        self.assertEqual(find_out_of_vocabulary("کردن و نگاه کردن", vocab), [])

    def test_a_noun_inflection_the_prefix_cannot_reach_is_still_reported(self):
        # An honest limit, recorded rather than hidden: the book says چیزهای
        # and a definition saying چیزی shares only three leading characters, so
        # the four-character stem rule still calls it foreign. Nothing here
        # stems nouns, and guessing at one would risk the real findings.
        vocab = build_vocabulary(["چیزهای تازه پیدا میکنیم"])
        self.assertEqual(find_out_of_vocabulary("چیزی تازه", vocab), ["چیزی"])


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
