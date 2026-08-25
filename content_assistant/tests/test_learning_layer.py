"""The learning-experience layer, and the rules that hold it together.

Two things are being proved here, and they pull in opposite directions.

The first is that activities and questions are **not** claims about the book,
so nothing may demand a quotation from them. The second is that they are not
therefore unconstrained: what replaces evidence is linkage, and a question that
measures no objective, or names one that does not exist, is refused just as
firmly as an ungrounded concept.

Every test names the way the schema can be wrong rather than the feature it
exercises, because a linkage rule that has never been shown firing on a real
fault is a rule nobody should trust.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

from content_assistant.models.content import (
    BookRef,
    Concept,
    ContentSchema,
    Evidence,
    LearningObjective,
    Provenance,
    Relation,
    Skill,
    human_relation,
    make_id,
    skill_from_objectives,
)
from content_assistant.models.learning import (
    AUTO_GRADABLE_TYPES,
    LearningActivity,
    Question,
    QuestionOption,
)
from content_assistant.validation.engine import run_validation
from content_assistant.validation.rules import ValidationContext

BOOK = "g1-olom"
LESSON = f"{BOOK}:lesson:01"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def evidence(eid="e1", verified=True, block="/page/1/Text/0"):
    return Evidence(
        id=eid,
        document_id=BOOK,
        block_id=block,
        pdf_page=1,
        quote="آهنربا آهن را جذب می‌کند",
        quote_verified=verified,
        match_method="exact" if verified else "token_overlap",
    )


def concept(cid="c1", evidence_ids=("e1",), confidence=0.8):
    return Concept(
        id=cid,
        lesson_id=LESSON,
        label="آهنربا",
        evidence_ids=list(evidence_ids),
        evidence_level="explicit",
        confidence=confidence,
        provenance=Provenance(
            extraction_method="model_proposed",
            stage="concepts",
            model_id="test-model",
            prompt_version="concept_v1@abc",
        ),
    )


def objective(
    oid="o1",
    concept_ids=("c1",),
    evidence_ids=("e1",),
    confidence=0.7,
    statement="دانش‌آموز بتواند نشان دهد آهنربا چه چیزهایی را جذب می‌کند.",
    level="explicit",
):
    return LearningObjective(
        id=oid,
        lesson_id=LESSON,
        statement=statement,
        objective_type="identify",
        performance_verb="نشان دهد",
        concept_ids=list(concept_ids),
        evidence_ids=list(evidence_ids),
        evidence_level=level,
        confidence=confidence,
        provenance=Provenance(
            extraction_method="model_proposed",
            stage="objectives",
            model_id="test-model",
            prompt_version="objective_v1@abc",
        ),
    )


def question(qid="q1", objective_ids=("o1",), question_type="multiple_choice"):
    return Question(
        id=qid,
        question_type=question_type,
        prompt="کدام یک را آهنربا جذب می‌کند؟",
        objective_ids=list(objective_ids),
        options=[
            QuestionOption(option_id="a", text="میخ", is_correct=True),
            QuestionOption(
                option_id="b",
                text="چوب",
                feedback="چوب فلز نیست؛ دوباره به جدول نگاه کن.",
            ),
        ],
    )


def activity(
    aid="a1", objective_ids=("o1",), activity_type="practice", question_ids=()
):
    return LearningActivity(
        id=aid,
        activity_type=activity_type,
        title="بازی آهنربا",
        objective_ids=list(objective_ids),
        question_ids=list(question_ids),
    )


def schema_with(**kwargs):
    kwargs.setdefault("evidence", [evidence()])
    kwargs.setdefault("concepts", [concept()])
    kwargs.setdefault("objectives", [objective()])
    return ContentSchema(
        book=BookRef(book_id=BOOK, grade=1, subject="science"), **kwargs
    )


def codes(schema, stages=("final",)):
    report = run_validation(
        ValidationContext(schema_doc=schema), stages=list(stages)
    )
    return report.by_code()


# ---------------------------------------------------------------------------
# the layer boundary itself
# ---------------------------------------------------------------------------


class LayerSeparationTests(unittest.TestCase):
    """A question is not a claim, and the schema says so structurally."""

    def test_a_question_carries_no_evidence_fields_at_all(self):
        # Not "carries empty evidence" - carries none. If the field existed, a
        # rule would eventually be written against it and authored material
        # would start being asked to quote the textbook.
        self.assertFalse(hasattr(question(), "evidence_ids"))
        self.assertFalse(hasattr(question(), "evidence_level"))
        self.assertFalse(hasattr(question(), "confidence"))

    def test_an_activity_carries_no_evidence_fields_at_all(self):
        self.assertFalse(hasattr(activity(), "evidence_ids"))
        self.assertFalse(hasattr(activity(), "confidence"))

    def test_an_activity_still_carries_provenance_and_review(self):
        # It is exempt from evidence, not from accountability.
        item = activity()
        self.assertEqual(item.review_status, "pending")
        self.assertIsNone(item.provenance)
        self.assertFalse(item.requires_human_review)

    def test_evidence_rules_never_fire_on_authored_material(self):
        schema = schema_with(
            questions=[question()], activities=[activity(question_ids=["q1"])]
        )
        found = codes(schema, stages=["semantic", "final"])
        self.assertNotIn("EVID001", found)

    def test_a_question_does_not_store_the_concepts_it_touches(self):
        self.assertFalse(hasattr(question(), "concept_ids"))


class ImportDirectionTests(unittest.TestCase):
    """``content`` may import ``learning``; ``learning`` may not import back.

    ``content`` needs ``learning`` to give ``ContentSchema`` its activity and
    question lists. If ``learning`` referred back, the pair would resolve or
    fail depending on which was imported first - and every test in this file
    imports ``content`` first, so the broken order would never show up here.
    A fresh interpreter is the only place that question can honestly be asked.
    """

    def _import_first(self, module: str):
        return subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
        )

    def test_the_experience_layer_imports_on_its_own(self):
        done = self._import_first("content_assistant.models.learning")
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_so_does_the_knowledge_layer(self):
        done = self._import_first("content_assistant.models.content")
        self.assertEqual(done.returncode, 0, done.stderr)


class QuestionOwnsNoKnowledgeTests(unittest.TestCase):
    """The concepts a question touches are reached through its objectives."""

    def test_concepts_are_derived_through_the_objective(self):
        schema = schema_with(questions=[question()])
        found = schema.concepts_for_question("q1")
        self.assertEqual([c.id for c in found], ["c1"])

    def test_repointing_the_objective_moves_the_question_with_it(self):
        # The whole reason the field is derived: one edit, one consequence.
        schema = schema_with(
            concepts=[concept(), concept(cid="c2", evidence_ids=["e1"])],
            objectives=[objective(concept_ids=["c2"])],
            questions=[question()],
        )
        self.assertEqual(
            [c.id for c in schema.concepts_for_question("q1")], ["c2"]
        )

    def test_a_question_naming_an_unknown_objective_reaches_no_concept(self):
        schema = schema_with(questions=[question(objective_ids=["nope"])])
        self.assertEqual(schema.concepts_for_question("q1"), [])

    def test_two_objectives_on_one_concept_do_not_duplicate_it(self):
        schema = schema_with(
            objectives=[objective(), objective(oid="o2")],
            questions=[question(objective_ids=["o1", "o2"])],
        )
        self.assertEqual(
            [c.id for c in schema.concepts_for_question("q1")], ["c1"]
        )


class TraversalTests(unittest.TestCase):
    """The questions an adaptive engine actually asks of the content layer."""

    def test_questions_for_objective_finds_what_would_measure_it(self):
        schema = schema_with(questions=[question(), question(qid="q2")])
        self.assertEqual(
            {q.id for q in schema.questions_for_objective("o1")}, {"q1", "q2"}
        )

    def test_remediation_is_reachable_without_a_string_at_the_call_site(self):
        schema = schema_with(
            activities=[
                activity(),
                activity(aid="a2", activity_type="remediation"),
            ]
        )
        remedial = schema.activities_for_objective("o1", "remediation")
        self.assertEqual([a.id for a in remedial], ["a2"])
        self.assertTrue(remedial[0].is_remedial)
        self.assertFalse(activity().is_remedial)

    def test_prerequisites_read_the_edge_in_the_stored_direction(self):
        edge = human_relation(
            book_id=BOOK,
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            reason="a child sorts before they compare",
            authored_by="teacher",
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[edge]
        )
        self.assertEqual(schema.prerequisites_of("c2"), ["c1"])
        self.assertEqual(schema.dependents_of("c1"), ["c2"])
        self.assertEqual(schema.prerequisites_of("c1"), [])

    def test_related_to_reads_from_either_end(self):
        edge = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="related_to",
            evidence_ids=["e1"],
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[edge]
        )
        self.assertEqual(schema.related_to("c1"), ["c2"])
        self.assertEqual(schema.related_to("c2"), ["c1"])

    def test_a_prerequisite_is_not_reported_as_merely_related(self):
        edge = human_relation(
            book_id=BOOK,
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            reason="ordering",
            authored_by="teacher",
        )
        schema = schema_with(relations=[edge])
        self.assertEqual(schema.related_to("c1"), [])


# ---------------------------------------------------------------------------
# deterministic identity
# ---------------------------------------------------------------------------


class DeterministicIdTests(unittest.TestCase):
    def test_the_same_question_hashes_to_the_same_id_twice(self):
        first = Question.build_id(BOOK, "multiple_choice", "کدام یک؟")
        second = Question.build_id(BOOK, "multiple_choice", "کدام یک؟")
        self.assertEqual(first, second)

    def test_two_spellings_of_one_prompt_are_one_question(self):
        # Arabic yeh against Persian yeh. Two ids here would mean re-running an
        # authoring tool silently doubled the question bank.
        persian = Question.build_id(BOOK, "true_false", "آیا آهنربا میخ را می‌گیرد؟")
        arabic = Question.build_id(BOOK, "true_false", "آيا آهنربا ميخ را مي‌گيرد؟")
        self.assertEqual(persian, arabic)

    def test_changing_the_form_changes_the_item(self):
        self.assertNotEqual(
            Question.build_id(BOOK, "multiple_choice", "کدام یک؟"),
            Question.build_id(BOOK, "short_answer", "کدام یک؟"),
        )

    def test_activity_ids_are_derived_the_same_way(self):
        first = LearningActivity.build_id(BOOK, "game", "بازی آهنربا")
        self.assertEqual(first, LearningActivity.build_id(BOOK, "game", "بازی آهنربا"))
        self.assertNotEqual(
            first, LearningActivity.build_id(BOOK, "practice", "بازی آهنربا")
        )

    def test_an_id_is_scoped_to_its_book(self):
        self.assertNotEqual(
            Question.build_id(BOOK, "true_false", "س"),
            Question.build_id("g2-olom", "true_false", "س"),
        )

    def test_a_relation_id_is_derived_from_its_endpoints(self):
        first = Relation.build_id(BOOK, "c1", "prerequisite_of", "c2")
        self.assertEqual(
            first, Relation.build_id(BOOK, "c1", "prerequisite_of", "c2")
        )
        # Direction is part of the fact, so reversing it is a different edge.
        self.assertNotEqual(
            first, Relation.build_id(BOOK, "c2", "prerequisite_of", "c1")
        )

    def test_auto_gradable_is_a_property_of_the_form(self):
        self.assertTrue(question(question_type="true_false").auto_gradable)
        self.assertFalse(question(question_type="drawing").auto_gradable)
        self.assertIn("matching", AUTO_GRADABLE_TYPES)


# ---------------------------------------------------------------------------
# LINK001 - references
# ---------------------------------------------------------------------------


class ReferenceIntegrityTests(unittest.TestCase):
    def test_a_question_naming_a_missing_objective_is_an_error(self):
        schema = schema_with(questions=[question(objective_ids=["ghost"])])
        self.assertIn("LINK001", codes(schema))

    def test_an_activity_naming_a_missing_question_is_an_error(self):
        schema = schema_with(
            activities=[activity(question_ids=["ghost"])],
        )
        self.assertIn("LINK001", codes(schema))

    def test_a_reference_of_the_wrong_kind_is_reported_as_such(self):
        # An id that exists but is a concept where an objective was expected is
        # the failure a plain "does it exist?" check waves through.
        schema = schema_with(questions=[question(objective_ids=["c1"])])
        report = run_validation(
            ValidationContext(schema_doc=schema), stages=["final"]
        )
        messages = [f.message for f in report.findings if f.code == "LINK001"]
        self.assertTrue(any("but it is a concept" in m for m in messages))

    def test_an_objective_naming_a_missing_skill_is_an_error(self):
        schema = schema_with(
            objectives=[
                objective().model_copy(update={"skill_id": "no-such-skill"})
            ]
        )
        self.assertIn("LINK001", codes(schema))

    def test_a_fully_wired_package_reports_nothing(self):
        schema = schema_with(
            questions=[question()],
            activities=[activity(question_ids=["q1"])],
        )
        self.assertNotIn("LINK001", codes(schema))
        self.assertNotIn("LINK002", codes(schema))


# ---------------------------------------------------------------------------
# LINK002 - the experience layer must serve something
# ---------------------------------------------------------------------------


class ExperienceServesSomethingTests(unittest.TestCase):
    def test_a_question_that_measures_nothing_is_refused(self):
        schema = schema_with(questions=[question(objective_ids=[])])
        self.assertIn("LINK002", codes(schema))

    def test_an_activity_serving_nothing_is_refused(self):
        schema = schema_with(activities=[activity(objective_ids=[])])
        self.assertIn("LINK002", codes(schema))

    def test_an_activity_with_no_questions_is_still_fine(self):
        # Most of what a six-year-old does is not a question.
        schema = schema_with(activities=[activity(activity_type="game")])
        self.assertNotIn("LINK002", codes(schema))


# ---------------------------------------------------------------------------
# LINK003 - a prerequisite must be accountable
# ---------------------------------------------------------------------------


class PrerequisiteAccountabilityTests(unittest.TestCase):
    def test_a_guessed_prerequisite_is_an_error(self):
        guess = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            provenance=Provenance(
                extraction_method="model_proposed",
                stage="relations",
                model_id="test-model",
                prompt_version="v1",
            ),
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[guess]
        )
        self.assertIn("LINK003", codes(schema))

    def test_a_quoted_prerequisite_stands_on_the_book(self):
        quoted = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            evidence_ids=["e1"],
            evidence_level="explicit",
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[quoted]
        )
        self.assertNotIn("LINK003", codes(schema))

    def test_an_unverified_quotation_does_not_count_as_one(self):
        # A citation that could not be found in the block it named is exactly
        # the case a laxer check would let through.
        quoted = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            evidence_ids=["e-bad"],
        )
        schema = schema_with(
            evidence=[evidence(), evidence(eid="e-bad", verified=False)],
            concepts=[concept(), concept(cid="c2")],
            relations=[quoted],
        )
        self.assertIn("LINK003", codes(schema))

    def test_a_named_person_may_author_one(self):
        edge = human_relation(
            book_id=BOOK,
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            reason="sorting comes before comparing",
            authored_by="m.aghazadeh",
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[edge]
        )
        self.assertNotIn("LINK003", codes(schema))
        self.assertTrue(edge.requires_human_review)

    def test_human_provenance_without_a_name_is_not_accountable(self):
        anonymous = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            provenance=Provenance(extraction_method="human", stage="relations"),
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[anonymous]
        )
        self.assertIn("LINK003", codes(schema))

    def test_authoring_one_demands_a_name_and_a_reason(self):
        with self.assertRaises(ValueError):
            human_relation(
                book_id=BOOK,
                source_id="c1",
                target_id="c2",
                relation_type="prerequisite_of",
                reason="because",
                authored_by="   ",
            )
        with self.assertRaises(ValueError):
            human_relation(
                book_id=BOOK,
                source_id="c1",
                target_id="c2",
                relation_type="prerequisite_of",
                reason="",
                authored_by="teacher",
            )

    def test_only_prerequisites_are_held_to_this(self):
        # A "related_to" edge sends nobody anywhere; it does not need a signature.
        loose = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="related_to",
            evidence_ids=["e1"],
        )
        schema = schema_with(
            concepts=[concept(), concept(cid="c2")], relations=[loose]
        )
        self.assertNotIn("LINK003", codes(schema))

    def test_a_cycle_is_still_caught_when_every_edge_is_accountable(self):
        # Each edge is individually defensible; the graph is not.
        edges = [
            human_relation(
                book_id=BOOK,
                source_id=a,
                target_id=b,
                relation_type="prerequisite_of",
                reason="judgement",
                authored_by="teacher",
            )
            for a, b in (("c1", "c2"), ("c2", "c3"), ("c3", "c1"))
        ]
        schema = schema_with(
            concepts=[concept(), concept(cid="c2"), concept(cid="c3")],
            relations=edges,
        )
        self.assertIn("FINAL002", codes(schema))


class EvidenceExemptionTests(unittest.TestCase):
    """EVID001's one exemption, and the fence around it."""

    def test_a_human_authored_record_may_stand_without_a_quotation(self):
        edge = human_relation(
            book_id=BOOK,
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            reason="judgement",
            authored_by="teacher",
        )
        schema = schema_with(relations=[edge])
        self.assertNotIn("EVID001", codes(schema, stages=["semantic"]))

    def test_a_model_proposed_record_may_not(self):
        guess = Relation(
            id="r1",
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            provenance=Provenance(
                extraction_method="model_proposed",
                stage="relations",
                model_id="m",
                prompt_version="v",
            ),
        )
        schema = schema_with(relations=[guess])
        self.assertIn("EVID001", codes(schema, stages=["semantic"]))

    def test_a_record_with_no_provenance_at_all_may_not(self):
        # Silence is not a claim of human authorship.
        schema = schema_with(
            concepts=[concept(evidence_ids=[])],
        )
        self.assertIn("EVID001", codes(schema, stages=["semantic"]))


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


class SkillDerivationTests(unittest.TestCase):
    def test_a_skill_is_no_more_certain_than_its_weakest_objective(self):
        skill = skill_from_objectives(
            book_id=BOOK,
            label="پیش‌بینی نتیجه یک آزمایش ساده",
            objectives=[
                objective(confidence=0.82),
                objective(oid="o2", confidence=0.61),
            ],
            authored_by="teacher",
        )
        self.assertEqual(skill.confidence, 0.61)

    def test_its_evidence_is_the_union_of_theirs_without_duplicates(self):
        skill = skill_from_objectives(
            book_id=BOOK,
            label="مهارت",
            objectives=[
                objective(evidence_ids=["e1", "e2"]),
                objective(oid="o2", evidence_ids=["e2", "e3"]),
            ],
            authored_by="teacher",
        )
        self.assertEqual(skill.evidence_ids, ["e1", "e2", "e3"])

    def test_one_inferred_objective_makes_the_whole_skill_inferred(self):
        skill = skill_from_objectives(
            book_id=BOOK,
            label="مهارت",
            objectives=[
                objective(level="explicit"),
                objective(oid="o2", level="inferred"),
            ],
            authored_by="teacher",
        )
        self.assertEqual(skill.evidence_level, "inferred")

    def test_a_skill_is_always_attributed_to_a_person(self):
        # Nothing but a person decides two objectives are one ability.
        skill = skill_from_objectives(
            book_id=BOOK,
            label="مهارت",
            objectives=[objective()],
            authored_by="teacher",
        )
        self.assertEqual(skill.provenance.extraction_method, "human")
        self.assertEqual(skill.provenance.authored_by, "teacher")

    def test_review_is_inherited_from_the_objectives_it_groups(self):
        under_review = objective().model_copy(
            update={"requires_human_review": True}
        )
        skill = skill_from_objectives(
            book_id=BOOK,
            label="مهارت",
            objectives=[under_review],
            authored_by="teacher",
        )
        self.assertTrue(skill.requires_human_review)

    def test_a_skill_over_nothing_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            skill_from_objectives(
                book_id=BOOK, label="مهارت", objectives=[], authored_by="t"
            )

    def test_its_id_is_derived_from_the_label(self):
        skill = skill_from_objectives(
            book_id=BOOK,
            label="مهارت",
            objectives=[objective()],
            authored_by="teacher",
        )
        self.assertEqual(skill.id, make_id(BOOK, "skill", "مهارت"))


class SkillIsNotAnObjectiveTests(unittest.TestCase):
    def test_a_skill_restating_its_only_objective_is_flagged(self):
        statement = "دانش‌آموز بتواند میخ را پیدا کند."
        skill = Skill(
            id="s1",
            label=statement,
            objective_ids=["o1"],
            evidence_ids=["e1"],
        )
        schema = schema_with(
            objectives=[objective(statement=statement)], skills=[skill]
        )
        self.assertIn("LINK004", codes(schema))

    def test_a_skill_that_generalises_two_objectives_is_not(self):
        skill = skill_from_objectives(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective(), objective(oid="o2")],
            authored_by="teacher",
        )
        schema = schema_with(
            objectives=[objective(), objective(oid="o2")], skills=[skill]
        )
        self.assertNotIn("LINK004", codes(schema))

    def test_a_skill_grouping_nothing_is_flagged(self):
        skill = Skill(id="s1", label="مهارت", evidence_ids=["e1"])
        schema = schema_with(skills=[skill])
        self.assertIn("LINK004", codes(schema))


# ---------------------------------------------------------------------------
# provenance and review
# ---------------------------------------------------------------------------


class ProvenanceTests(unittest.TestCase):
    def test_a_claim_of_model_authorship_must_name_the_model(self):
        nameless = concept().model_copy(
            update={
                "provenance": Provenance(
                    extraction_method="model_proposed", stage="concepts"
                )
            }
        )
        schema = schema_with(concepts=[nameless])
        self.assertIn("PROV001", codes(schema, stages=["semantic"]))

    def test_a_record_with_no_provenance_is_silence_not_a_fault(self):
        # A 1.0.0 artifact has none. Reporting those would bury the real cases.
        older = concept().model_copy(update={"provenance": None})
        schema = schema_with(concepts=[older])
        self.assertNotIn("PROV001", codes(schema, stages=["semantic"]))

    def test_a_fully_recorded_model_claim_passes(self):
        self.assertNotIn(
            "PROV001", codes(schema_with(), stages=["semantic"])
        )

    def test_a_human_record_is_not_asked_for_a_prompt_version(self):
        edge = human_relation(
            book_id=BOOK,
            source_id="c1",
            target_id="c2",
            relation_type="prerequisite_of",
            reason="judgement",
            authored_by="teacher",
        )
        schema = schema_with(relations=[edge])
        self.assertNotIn("PROV001", codes(schema, stages=["semantic"]))

    def test_the_pipeline_stamps_provenance_without_a_timestamp(self):
        # A per-entity timestamp would make two identical runs produce
        # artifacts that differ, which destroys the point of a diff.
        item = concept()
        self.assertIsNone(item.provenance.generated_at)
        self.assertEqual(item.provenance.stage, "concepts")


class ReviewLifecycleTests(unittest.TestCase):
    def test_an_unsigned_verdict_is_an_error(self):
        decided = concept().model_copy(update={"review_status": "accepted"})
        schema = schema_with(concepts=[decided])
        self.assertIn("PROV002", codes(schema, stages=["semantic"]))

    def test_a_signed_verdict_passes(self):
        decided = concept().model_copy(
            update={
                "review_status": "accepted",
                "reviewed_by": "m.aghazadeh",
                "reviewed_at": "2026-08-25T10:00:00Z",
            }
        )
        schema = schema_with(concepts=[decided])
        self.assertNotIn("PROV002", codes(schema, stages=["semantic"]))

    def test_pending_is_the_absence_of_a_decision_not_one(self):
        self.assertNotIn("PROV002", codes(schema_with(), stages=["semantic"]))

    def test_a_reviewer_may_accept_something_still_flagged_for_review(self):
        # The flag records why it was queued; the status records the verdict.
        # Neither overwrites the other, and that is the audit trail.
        decided = concept().model_copy(
            update={
                "requires_human_review": True,
                "review_reasons": ["confidence 0.80 below auto-accept"],
                "review_status": "accepted",
                "reviewed_by": "m.aghazadeh",
                "reviewed_at": "2026-08-25T10:00:00Z",
                "review_notes": "read back against page 12; correct",
            }
        )
        schema = schema_with(concepts=[decided])
        self.assertNotIn("PROV002", codes(schema, stages=["semantic"]))
        self.assertTrue(decided.requires_human_review)
        self.assertEqual(decided.review_status, "accepted")

    def test_a_verdict_on_authored_material_is_held_to_the_same_rule(self):
        item = activity().model_copy(update={"review_status": "rejected"})
        schema = schema_with(activities=[item])
        self.assertIn("PROV002", codes(schema, stages=["semantic"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
