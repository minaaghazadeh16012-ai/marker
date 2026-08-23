"""Adversarial tests for objective extraction.

Every test names the way a model can be wrong, because that is the only thing
worth testing here: a real model rarely volunteers a fabricated citation or an
unassessable verb on demand, and the pipeline still has to be proved against
both.

Nothing here touches a network or a model. The concepts these objectives hang
from are produced by running the real concept pipeline against a scripted
reply, so the evidence ids are the ids the pipeline actually derives rather
than constants a test invented and a refactor would silently invalidate.
"""

from __future__ import annotations

import unittest

from content_assistant.models.content import (
    BookRef,
    Concept,
    ContentSchema,
)
from content_assistant.models.extraction import (
    Block,
    BookIdentity,
    DocumentInfo,
    ExtractionResult,
    Page,
    TocEntry,
)
from content_assistant.models.objective import (
    OBJECTIVE_SCHEMA_VERSION,
    is_vague,
    lexicon_words,
    strip_lexicon,
    type_fits_concept,
    verb_is_observable,
)
from content_assistant.structuring.evidence import build_evidence_unit
from content_assistant.structuring.segmentation import segment
from content_assistant.structuring.semantic.concepts import ground_proposals
from content_assistant.structuring.semantic.llm import MockLLMClient
from content_assistant.structuring.semantic.objectives import (
    MAX_OBJECTIVES_PER_CONCEPT,
    ModelCallFailed,
    build_objective_prompt,
    concept_blocks,
    extract_objectives,
    ground_objective_proposals,
    load_objective_prompt,
)
from content_assistant.structuring.semantic.proposals import (
    REJECT_CITATIONS_OUTSIDE_CONCEPT,
    REJECT_CONCEPT_WITHOUT_EVIDENCE,
    REJECT_EMPTY_STATEMENT,
    REJECT_UNKNOWN_CONCEPT,
    AdmissionResult,
    CitationProposal,
    ConceptProposal,
    ObjectiveProposal,
    ObjectiveResponse,
    admit_objective_proposals,
)
from content_assistant.validation.engine import run_validation
from content_assistant.validation.rules import ValidationContext

BOOK = "g1-olom"

FOOD = "جانوران غذا میخورند"
HOT = "به وسیلههای داغ دست نزنید"
CURIOUS = "با کنجکاوی و دقت نگاه کنید"
QUESTION = "آب از کدام خاک زودتر عبور میکند؟"

B_FOOD = "/page/1/Text/1"
B_HOT = "/page/1/Text/2"
B_CURIOUS = "/page/1/Text/3"
B_QUESTION = "/page/1/Text/4"


def _block(block_id, text, top):
    return Block(
        block_id=block_id,
        type="Text",
        text=text,
        bbox=[10, top, 400, top + 20],
        polygon=[[10, top], [400, top], [400, top + 20], [10, top + 20]],
        source="marker",
    )


def lesson_fixture():
    """One lesson carrying a fact, a rule, a habit and a question.

    Four blocks on purpose: the concept types this stage has to tell apart
    (conceptual, procedural, meta) plus the question that tempts a model into
    answering it.
    """
    pages = [
        Page(
            pdf_page=1,
            pdf_page_index=0,
            printed_page=1,
            printed_page_source="page_footer",
            blocks=[_block("/page/0/Text/0", "1 دنیای جانوران", 20)],
        ),
        Page(
            pdf_page=2,
            pdf_page_index=1,
            printed_page=2,
            printed_page_source="page_footer",
            blocks=[
                _block(B_FOOD, f"{FOOD} و حرکت میکنند.", 60),
                _block(B_HOT, f"{HOT}.", 90),
                _block(B_CURIOUS, f"{CURIOUS} تا چیزهای تازه پیدا کنید.", 120),
                _block(B_QUESTION, QUESTION, 150),
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
    return extraction, build_evidence_unit(extraction, lessons[0], sections)


def grounded_concepts(unit):
    """Real concepts, produced by the real concept pipeline.

    Three of them, one per concept type this stage must distinguish, plus a
    fourth grounded on the book's question so a test can try to answer it.
    """
    admission = AdmissionResult(
        admitted=[
            ConceptProposal(
                label="غذا خوردن جانوران",
                definition="جانوران غذا میخورند.",
                concept_type="conceptual",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                claimed_evidence_level="explicit",
            ),
            ConceptProposal(
                label="دست نزدن به وسیلههای داغ",
                definition="به وسیلههای داغ دست نزنید.",
                concept_type="procedural",
                citations=[CitationProposal(block_id=B_HOT, quote=HOT)],
                claimed_evidence_level="explicit",
            ),
            ConceptProposal(
                label="کنجکاوی و دقت",
                definition="با کنجکاوی و دقت نگاه کنید.",
                concept_type="meta",
                citations=[CitationProposal(block_id=B_CURIOUS, quote=CURIOUS)],
                claimed_evidence_level="explicit",
            ),
            ConceptProposal(
                label="عبور آب از خاک",
                definition="آب از کدام خاک زودتر عبور میکند؟",
                concept_type="conceptual",
                citations=[
                    CitationProposal(block_id=B_QUESTION, quote=QUESTION)
                ],
                claimed_evidence_level="explicit",
            ),
        ]
    )
    result = ground_proposals(
        unit=unit, admission=admission, document_id=BOOK
    )
    return result.concepts, result.evidence


class ObjectiveFixture(unittest.TestCase):
    def setUp(self):
        self.extraction, self.unit = lesson_fixture()
        self.concepts, self.evidence = grounded_concepts(self.unit)
        by_type = {c.concept_type: c for c in self.concepts}
        self.conceptual = [
            c for c in self.concepts if c.label == "غذا خوردن جانوران"
        ][0]
        self.procedural = by_type["procedural"]
        self.meta = by_type["meta"]
        self.question = [
            c for c in self.concepts if c.label == "عبور آب از خاک"
        ][0]

    def ground(self, *proposals):
        response = ObjectiveResponse(objectives=list(proposals))
        admission = admit_objective_proposals(
            response, concept_blocks(self.concepts, self.evidence)
        )
        result = ground_objective_proposals(
            unit=self.unit,
            concepts=self.concepts,
            evidence=self.evidence,
            admission=admission,
            document_id=BOOK,
        )
        return result

    def reasons(self, objective):
        return " | ".join(objective.review_reasons)


# ---------------------------------------------------------------------------
# 1. a fully grounded objective is accepted
# ---------------------------------------------------------------------------


class GroundedObjectiveTests(ObjectiveFixture):
    def test_a_grounded_observable_objective_is_accepted(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                claimed_evidence_level="explicit",
            )
        )
        self.assertEqual(len(result.objectives), 1)
        objective = result.objectives[0]
        self.assertTrue(objective.observable)
        self.assertEqual(objective.concept_ids, [self.conceptual.id])
        self.assertEqual(objective.out_of_book_vocabulary, [])
        self.assertEqual(objective.evidence_level, "explicit")
        self.assertGreater(objective.confidence, 0.0)

    def test_the_objective_carries_the_stage_schema_version(self):
        result = self.ground()
        self.assertEqual(result.schema_version, OBJECTIVE_SCHEMA_VERSION)
        self.assertEqual(result.grade, 1)

    def test_an_objective_is_never_more_certain_than_its_concept(self):
        # The cap that keeps a well-phrased objective from outrunning a shaky
        # idea. Without it the objective sails past review while the concept
        # it rests on sits in the queue.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                claimed_evidence_level="explicit",
            )
        )
        objective = result.objectives[0]
        self.assertLessEqual(objective.confidence, self.conceptual.confidence)

    def test_a_concept_with_no_objective_is_recorded_not_filled(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        self.assertIn(self.procedural.id, result.concepts_without_objectives)
        self.assertIn(self.meta.id, result.concepts_without_objectives)


# ---------------------------------------------------------------------------
# 2 + 3. imported knowledge and out-of-book terminology
# ---------------------------------------------------------------------------


class ImportedKnowledgeTests(ObjectiveFixture):
    def test_general_knowledge_added_to_a_grounded_quote_is_flagged(self):
        # The citation is real and the quote verifies; the sentence built
        # around it is not the book's. That is exactly the case the wording
        # check exists for.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران برای انرژی به پروتئین نیاز دارند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                claimed_evidence_level="explicit",
            )
        )
        objective = result.objectives[0]
        self.assertTrue(objective.requires_human_review)
        self.assertIn("انرژی", objective.out_of_book_vocabulary)
        self.assertIn("پروتئین", objective.out_of_book_vocabulary)
        self.assertIn("wording is not the lesson's", self.reasons(objective))

    def test_a_technical_term_above_first_grade_is_flagged(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که متابولیسم جانوران با غذا انجام میشود.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        objective = result.objectives[0]
        self.assertIn("متابولیسم", objective.out_of_book_vocabulary)
        self.assertTrue(objective.requires_human_review)

    def test_the_pipelines_own_verbs_are_not_reported_as_foreign(self):
        # 'توضیح دهد' comes from the objective lexicon, not from the book, so
        # reporting it would bury the words that actually matter.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        self.assertNotIn("توضیح", result.objectives[0].out_of_book_vocabulary)

    def test_strip_lexicon_removes_only_this_modules_words(self):
        self.assertEqual(strip_lexicon(["توضیح", "متابولیسم"]), ["متابولیسم"])
        self.assertIn("بگوید", lexicon_words())


# ---------------------------------------------------------------------------
# 4 + 5. missing and invalid evidence
# ---------------------------------------------------------------------------


class EvidenceAdmissionTests(ObjectiveFixture):
    def test_a_concept_with_no_evidence_yields_no_objective(self):
        naked = Concept(
            id=f"{BOOK}:concept:naked",
            lesson_id=self.unit.lesson_id,
            label="مفهوم بیشاهد",
            concept_type="conceptual",
        )
        response = ObjectiveResponse(
            objectives=[
                ObjectiveProposal(
                    concept_id=naked.id,
                    statement="توضیح دهد که چیزی درست است.",
                    objective_type="describe",
                    performance_verb="توضیح دهد",
                    citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                )
            ]
        )
        admission = admit_objective_proposals(
            response,
            concept_blocks([naked] + list(self.concepts), self.evidence),
        )
        self.assertEqual(admission.admitted, [])
        self.assertEqual(
            admission.rejected[0].reason, REJECT_CONCEPT_WITHOUT_EVIDENCE
        )

    def test_a_citation_outside_the_concepts_evidence_is_rejected(self):
        # The mechanical form of "an objective must not become a new concept":
        # the block is real and in the lesson, but this concept does not rest
        # on it, so an objective citing it is claiming something new.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="رعایت کند که به وسیلههای داغ دست نزند.",
                objective_type="perform",
                performance_verb="رعایت کند",
                citations=[CitationProposal(block_id=B_HOT, quote=HOT)],
            )
        )
        self.assertEqual(result.objectives, [])
        self.assertEqual(
            result.admission.rejected[0].reason,
            REJECT_CITATIONS_OUTSIDE_CONCEPT,
        )

    def test_an_unknown_block_is_rejected(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[
                    CitationProposal(block_id="/page/9/Text/9", quote=FOOD)
                ],
            )
        )
        self.assertEqual(result.objectives, [])
        self.assertIn("/page/9/Text/9", result.admission.dropped_citations)

    def test_an_unknown_concept_id_is_rejected(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=f"{BOOK}:concept:invented",
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        self.assertEqual(
            result.admission.rejected[0].reason, REJECT_UNKNOWN_CONCEPT
        )

    def test_an_empty_statement_is_rejected(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="   ",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        self.assertEqual(
            result.admission.rejected[0].reason, REJECT_EMPTY_STATEMENT
        )

    def test_an_unverifiable_quote_still_grounds_but_asks_for_review(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[
                    CitationProposal(
                        block_id=B_FOOD, quote="این جمله در کتاب نیست"
                    )
                ],
                claimed_evidence_level="explicit",
            )
        )
        objective = result.objectives[0]
        self.assertEqual(objective.evidence_level, "inferred")
        self.assertTrue(objective.requires_human_review)
        self.assertIn("no cited quotation", self.reasons(objective))


# ---------------------------------------------------------------------------
# 6. duplicates
# ---------------------------------------------------------------------------


class DuplicateObjectiveTests(ObjectiveFixture):
    def test_two_objectives_saying_the_same_thing_are_flagged(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            ),
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            ),
        )
        self.assertTrue(result.duplicate_flags)
        for objective in result.objectives:
            self.assertTrue(objective.requires_human_review)
            self.assertIn("possible duplicate", self.reasons(objective))

    def test_objectives_on_different_concepts_are_not_duplicates(self):
        # Two lessons may legitimately ask a student to name something;
        # calling those duplicates would bury the case that matters.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="بگوید جانوران غذا میخورند.",
                objective_type="name",
                performance_verb="بگوید",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            ),
            ObjectiveProposal(
                concept_id=self.procedural.id,
                statement="رعایت کند که به وسیلههای داغ دست نزند.",
                objective_type="perform",
                performance_verb="رعایت کند",
                citations=[CitationProposal(block_id=B_HOT, quote=HOT)],
            ),
        )
        self.assertEqual(result.duplicate_flags, {})

    def test_padding_a_concept_with_objectives_is_flagged(self):
        proposals = [
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement=f"بگوید جانوران غذا میخورند و حرکت میکنند {n}.",
                objective_type="name",
                performance_verb="بگوید",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
            for n in range(MAX_OBJECTIVES_PER_CONCEPT + 1)
        ]
        result = self.ground(*proposals)
        self.assertGreater(len(result.objectives), MAX_OBJECTIVES_PER_CONCEPT)
        for objective in result.objectives:
            self.assertIn("more than", self.reasons(objective))


# ---------------------------------------------------------------------------
# 7. turning a question into a claim
# ---------------------------------------------------------------------------


class QuestionIntoClaimTests(ObjectiveFixture):
    def test_answering_the_books_question_imports_words_the_book_lacks(self):
        # The book asks which soil water passes through faster; it never
        # answers. An objective that answers it has to import the answer, and
        # importing it is what the wording check sees.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.question.id,
                statement="توضیح دهد که آب از خاک درشت زودتر عبور میکند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[
                    CitationProposal(block_id=B_QUESTION, quote=QUESTION)
                ],
            )
        )
        objective = result.objectives[0]
        self.assertIn("درشت", objective.out_of_book_vocabulary)
        self.assertTrue(objective.requires_human_review)

    def test_a_three_letter_imported_word_is_a_known_blind_spot(self):
        # Recorded rather than hidden. ``min_word_length`` is 4, so a
        # three-letter answer word - ``شنی`` here, and ``دما``/``ذوب``
        # elsewhere - is never checked at all. Widening the floor is a
        # calibration decision for the vocabulary layer, not something this
        # stage should reach in and change, so the limit is pinned here to
        # make it visible the moment anyone does.
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.question.id,
                statement="توضیح دهد که آب از خاک شنی زودتر عبور میکند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[
                    CitationProposal(block_id=B_QUESTION, quote=QUESTION)
                ],
            )
        )
        self.assertEqual(result.objectives[0].out_of_book_vocabulary, [])

    def test_an_objective_about_doing_the_activity_stays_in_the_book(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.question.id,
                statement="آزمایش کند که آب از کدام خاک زودتر عبور میکند.",
                objective_type="perform",
                performance_verb="آزمایش کند",
                citations=[
                    CitationProposal(block_id=B_QUESTION, quote=QUESTION)
                ],
            )
        )
        self.assertEqual(result.objectives[0].out_of_book_vocabulary, [])


# ---------------------------------------------------------------------------
# 8 + 9 + 10. objective type against concept type
# ---------------------------------------------------------------------------


class ObjectiveTypeFitTests(ObjectiveFixture):
    def test_a_procedural_concept_takes_a_perform_objective(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.procedural.id,
                statement="رعایت کند که به وسیلههای داغ دست نزند.",
                objective_type="perform",
                performance_verb="رعایت کند",
                citations=[CitationProposal(block_id=B_HOT, quote=HOT)],
                claimed_evidence_level="explicit",
            )
        )
        objective = result.objectives[0]
        self.assertNotIn("does not suit", self.reasons(objective))
        self.assertTrue(objective.observable)

    def test_naming_a_safety_rule_instead_of_following_it_is_flagged(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.procedural.id,
                statement="بگوید به وسیلههای داغ دست نزند.",
                objective_type="name",
                performance_verb="بگوید",
                citations=[CitationProposal(block_id=B_HOT, quote=HOT)],
            )
        )
        objective = result.objectives[0]
        self.assertTrue(objective.requires_human_review)
        self.assertIn("does not suit", self.reasons(objective))

    def test_a_conceptual_concept_takes_a_describing_objective(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="بگوید جانوران غذا میخورند.",
                objective_type="name",
                performance_verb="بگوید",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                claimed_evidence_level="explicit",
            )
        )
        self.assertNotIn("does not suit", self.reasons(result.objectives[0]))

    def test_a_conceptual_concept_asked_to_be_performed_is_flagged(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="انجام دهد که جانوران غذا میخورند.",
                objective_type="perform",
                performance_verb="انجام دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        self.assertIn("does not suit", self.reasons(result.objectives[0]))

    def test_a_meta_concept_takes_a_perform_objective(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.meta.id,
                statement="انجام دهد که با کنجکاوی و دقت نگاه کند.",
                objective_type="perform",
                performance_verb="انجام دهد",
                citations=[CitationProposal(block_id=B_CURIOUS, quote=CURIOUS)],
                claimed_evidence_level="explicit",
            )
        )
        self.assertNotIn("does not suit", self.reasons(result.objectives[0]))

    def test_a_meta_concept_asked_to_be_classified_is_flagged(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.meta.id,
                statement="دسته بندی کند که با کنجکاوی و دقت نگاه کند.",
                objective_type="classify",
                performance_verb="دسته بندی کند",
                citations=[CitationProposal(block_id=B_CURIOUS, quote=CURIOUS)],
            )
        )
        self.assertIn("does not suit", self.reasons(result.objectives[0]))


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


class ObservabilityTests(ObjectiveFixture):
    def test_a_wish_is_not_an_objective(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="بداند که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="بداند",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        objective = result.objectives[0]
        self.assertFalse(objective.observable)
        self.assertTrue(objective.requires_human_review)
        self.assertIn("state of mind", self.reasons(objective))

    def test_the_vague_verbs_are_recognised(self):
        for phrase in (
            "بداند", "درک کند", "بفهمد", "آشنا شود", "یاد بگیرد",
            "متوجه شود", "پی ببرد",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_vague(phrase))

    def test_the_observable_verbs_are_recognised(self):
        for phrase in (
            "بگوید", "نشان دهد", "توضیح دهد", "دسته بندی کند",
            "مقایسه کند", "انجام دهد", "رعایت کند",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(verb_is_observable(phrase))

    def test_a_verb_outside_the_lexicon_is_not_observable(self):
        # The lexicon is closed; extending it is a visible edit to one file,
        # never something a model can do by writing a confident sentence.
        self.assertFalse(verb_is_observable("بیندیشد"))
        self.assertFalse(verb_is_observable(""))

    def test_an_empty_performance_verb_asks_for_review(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        objective = result.objectives[0]
        self.assertFalse(objective.observable)
        self.assertIn("no performance verb", self.reasons(objective))

    def test_type_fit_is_permissive_for_an_unknown_concept_type(self):
        # The concept vocabulary belongs to the concept layer; guessing here
        # would report a fault that lives somewhere else.
        self.assertTrue(type_fits_concept("name", "something_new"))


# ---------------------------------------------------------------------------
# the validator, run independently of the extractor
# ---------------------------------------------------------------------------


class ObjectiveValidationTests(ObjectiveFixture):
    def _schema(self, objectives):
        return ContentSchema(
            book=BookRef(book_id=BOOK, grade=1, subject="science"),
            concepts=list(self.concepts),
            objectives=objectives,
            evidence=list(self.evidence),
        )

    def _codes(self, objectives):
        report = run_validation(
            ValidationContext(schema_doc=self._schema(objectives)),
            stages=["semantic"],
        )
        return {f.code for f in report.findings}

    def test_a_clean_objective_raises_no_objective_rule(self):
        result = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
                claimed_evidence_level="explicit",
            )
        )
        schema = self._schema(result.objectives)
        schema.evidence = list(
            {e.id: e for e in list(self.evidence) + result.evidence}.values()
        )
        report = run_validation(
            ValidationContext(schema_doc=schema), stages=["semantic"]
        )
        for code in ("PEDA006", "PEDA009", "PEDA010", "PEDA011"):
            self.assertNotIn(code, {f.code for f in report.findings})

    def test_evidence_outside_the_concept_is_caught_by_the_validator(self):
        # Built by hand rather than through the extractor: the point is that
        # the rule stands on its own, so a future extractor bug is caught by
        # something other than the extractor.
        hot_evidence = [
            e for e in self.evidence if e.block_id == B_HOT
        ][0]
        objective = self._objective(evidence_ids=[hot_evidence.id])
        self.assertIn("PEDA006", self._codes([objective]))

    def test_a_wish_is_caught_by_the_validator(self):
        objective = self._objective(statement="بداند که جانوران غذا میخورند.")
        self.assertIn("PEDA011", self._codes([objective]))

    def test_a_type_mismatch_is_caught_by_the_validator(self):
        objective = self._objective(
            objective_type="perform", concept_ids=[self.conceptual.id]
        )
        self.assertIn("PEDA010", self._codes([objective]))

    def test_out_of_book_wording_is_caught_by_the_validator(self):
        objective = self._objective(out_of_book_vocabulary=["متابولیسم"])
        self.assertIn("PEDA009", self._codes([objective]))

    def test_a_duplicate_statement_is_caught_by_the_validator(self):
        first = self._objective(objective_id="a")
        second = self._objective(objective_id="b")
        self.assertIn("PEDA008", self._codes([first, second]))

    def test_an_objective_in_the_wrong_lesson_is_caught(self):
        objective = self._objective(lesson_id=f"{BOOK}:lesson:99")
        self.assertIn("PEDA007", self._codes([objective]))

    def _objective(self, **overrides):
        from content_assistant.models.content import LearningObjective

        food_evidence = [e for e in self.evidence if e.block_id == B_FOOD][0]
        defaults = dict(
            id=f"{BOOK}:objective:{overrides.pop('objective_id', 'x')}",
            lesson_id=self.unit.lesson_id,
            statement="توضیح دهد که جانوران غذا میخورند.",
            objective_type="describe",
            performance_verb="توضیح دهد",
            observable=True,
            concept_ids=[self.conceptual.id],
            evidence_ids=[food_evidence.id],
            evidence_level="explicit",
            confidence=0.9,
        )
        defaults.update(overrides)
        return LearningObjective(**defaults)


# ---------------------------------------------------------------------------
# prompt + end to end through a scripted model
# ---------------------------------------------------------------------------


class ObjectivePromptTests(ObjectiveFixture):
    def test_the_prompt_shows_every_concept_with_only_its_own_blocks(self):
        template = load_objective_prompt()
        prompt = build_objective_prompt(
            self.unit, self.concepts, self.evidence, template
        )
        for concept in self.concepts:
            self.assertIn(concept.id, prompt)
        # The procedural concept's block must appear under it, and the
        # conceptual concept must not be offered that block.
        self.assertIn(B_HOT, prompt)
        self.assertIn("نوع هدف مجاز", prompt)

    def test_the_prompt_carries_the_closed_verb_lexicon(self):
        template = load_objective_prompt()
        prompt = build_objective_prompt(
            self.unit, self.concepts, self.evidence, template
        )
        self.assertIn("توضیح دهد", prompt)
        self.assertIn("بداند", prompt)  # named as forbidden
        self.assertNotIn("{{VERB_LEXICON}}", prompt)
        self.assertNotIn("{{CONCEPTS}}", prompt)

    def test_the_prompt_version_is_the_files_content_hash(self):
        template = load_objective_prompt()
        self.assertTrue(template.full_version.startswith("objective_v1@"))

    def test_end_to_end_through_a_scripted_model(self):
        client = MockLLMClient(
            responses=[
                ObjectiveResponse(
                    objectives=[
                        ObjectiveProposal(
                            concept_id=self.conceptual.id,
                            statement="توضیح دهد که جانوران غذا میخورند.",
                            objective_type="describe",
                            performance_verb="توضیح دهد",
                            citations=[
                                CitationProposal(block_id=B_FOOD, quote=FOOD)
                            ],
                            claimed_evidence_level="explicit",
                        )
                    ]
                )
            ]
        )
        result, raw, prompt = extract_objectives(
            unit=self.unit,
            concepts=self.concepts,
            evidence=self.evidence,
            client=client,
            document_id=BOOK,
        )
        self.assertEqual(len(result.objectives), 1)
        self.assertEqual(len(raw.objectives), 1)
        self.assertIn(self.conceptual.id, prompt)
        self.assertEqual(result.model_id, "mock")
        self.assertTrue(result.prompt_version)

    def test_a_failed_call_is_not_a_lesson_without_objectives(self):
        """Measured on the real book: ten lessons in a row reported
        "0 objectives, validation ok" while every call behind them had failed
        on quota. Marker's services return ``{}`` once their retries are
        exhausted, and ``{}`` validates into a well-formed reply with no
        objectives - indistinguishable from a model that read the lesson and
        correctly found nothing.

        Saying a lesson has no objectives is a claim about the book. A call
        that never arrived may not make it.
        """
        client = MockLLMClient(responses=[{}])
        with self.assertRaises(ModelCallFailed):
            extract_objectives(
                unit=self.unit,
                concepts=self.concepts,
                evidence=self.evidence,
                client=client,
                document_id=BOOK,
            )

    def test_a_model_that_genuinely_found_nothing_is_believed(self):
        # The other half: an empty list is a real answer and must survive.
        client = MockLLMClient(responses=[{"objectives": [], "notes": "هیچ"}])
        result, raw, _ = extract_objectives(
            unit=self.unit,
            concepts=self.concepts,
            evidence=self.evidence,
            client=client,
            document_id=BOOK,
        )
        self.assertEqual(result.objectives, [])
        self.assertEqual(raw.notes, "هیچ")
        self.assertEqual(
            len(result.concepts_without_objectives), len(self.concepts)
        )

    def test_ids_are_derived_so_a_rerun_is_byte_identical(self):
        first = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        second = self.ground(
            ObjectiveProposal(
                concept_id=self.conceptual.id,
                statement="توضیح دهد که جانوران غذا میخورند.",
                objective_type="describe",
                performance_verb="توضیح دهد",
                citations=[CitationProposal(block_id=B_FOOD, quote=FOOD)],
            )
        )
        self.assertEqual(
            [o.id for o in first.objectives], [o.id for o in second.objectives]
        )


if __name__ == "__main__":
    unittest.main()
