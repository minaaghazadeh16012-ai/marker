"""Human authoring, and the rules that hold authored material to account.

Four things an adaptive engine needs are not printed in a first-grade textbook:
which objectives add up to one transferable skill, which concept must come
before which, what a child should do to practise, and what would show they can.
:mod:`content_assistant.authoring` is the only door for those, and this file
proves that the door is narrow.

Two properties are being tested, and they are easy to confuse.

*Refusal at authoring time.* A record that could not be used by anything
downstream is not built at all, because the person is right there and can fix
it. That is what the ``AuthoringError`` tests are about.

*Refusal at validation time.* The same faults, arriving by some other route -
a hand-edited file, a future generator, a migration - are caught by rules that
know nothing about how the record was made. Authoring is convenience;
validation is the guarantee, and neither substitutes for the other.

Every test names the way the schema can be wrong rather than the feature it
exercises.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from content_assistant.authoring import (
    AuthoredContent,
    AuthoringError,
    author_activity,
    author_prerequisite,
    author_question,
    author_relation,
    author_skill,
    load_authored,
    question_option,
    save_authored,
)
from content_assistant.authoring.store import (
    AUTHORED_FILENAME,
    AuthoredContentError,
    authored_path,
)
from content_assistant.models.common import SCHEMA_VERSION
from content_assistant.models.content import (
    BookRef,
    Concept,
    ContentSchema,
    Evidence,
    LearningObjective,
    Provenance,
    Relation,
)
from content_assistant.models.learning import (
    AUTO_GRADABLE_TYPES,
    DEFAULT_GRADING_MODE,
    QUESTION_TYPES,
    GradingSpec,
    LearningActivity,
    Question,
    QuestionOption,
)
from content_assistant.package.registry import ContentRegistry
from content_assistant.package.schema import ContentPackage, compute_stats
from content_assistant.validation.engine import run_validation
from content_assistant.validation.rules import ValidationContext

BOOK = "g1-olom"
LESSON = f"{BOOK}:lesson:01"
AUTHOR = "م. آقازاده"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def evidence(eid="e1", verified=True):
    return Evidence(
        id=eid,
        document_id=BOOK,
        block_id="/page/1/Text/0",
        pdf_page=1,
        quote="آهنربا آهن را جذب می‌کند",
        quote_verified=verified,
        match_method="exact" if verified else "token_overlap",
    )


def concept(cid="c1"):
    return Concept(
        id=cid,
        lesson_id=LESSON,
        label="آهنربا",
        evidence_ids=["e1"],
        evidence_level="explicit",
        confidence=0.8,
        provenance=Provenance(
            extraction_method="model_proposed",
            stage="concepts",
            model_id="test-model",
            prompt_version="concept_v1@abc",
        ),
    )


def objective(oid="o1", concept_ids=("c1",), confidence=0.7, statement=None):
    return LearningObjective(
        id=oid,
        lesson_id=LESSON,
        statement=statement
        or "دانش‌آموز بتواند نشان دهد آهنربا چه چیزهایی را جذب می‌کند.",
        objective_type="identify",
        performance_verb="نشان دهد",
        concept_ids=list(concept_ids),
        evidence_ids=["e1"],
        evidence_level="explicit",
        confidence=confidence,
        provenance=Provenance(
            extraction_method="model_proposed",
            stage="objectives",
            model_id="test-model",
            prompt_version="objective_v1@abc",
        ),
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


def mc_question(**kwargs):
    kwargs.setdefault("book_id", BOOK)
    kwargs.setdefault("question_type", "multiple_choice")
    kwargs.setdefault("prompt", "کدام یک را آهنربا جذب می‌کند؟")
    kwargs.setdefault("objective_ids", ["o1"])
    kwargs.setdefault("authored_by", AUTHOR)
    kwargs.setdefault(
        "options",
        [
            question_option("میخ", is_correct=True),
            question_option("چوب", feedback="چوب فلز نیست."),
        ],
    )
    return author_question(**kwargs)


# ---------------------------------------------------------------------------
# authorship is never inferred
# ---------------------------------------------------------------------------


class AccountabilityTests(unittest.TestCase):
    """The exemption from evidence costs a name, every time."""

    def test_a_question_with_no_named_author_is_refused(self):
        with self.assertRaises(AuthoringError):
            mc_question(authored_by="")

    def test_an_activity_with_no_named_author_is_refused(self):
        with self.assertRaises(AuthoringError):
            author_activity(
                book_id=BOOK,
                activity_type="practice",
                title="بازی آهنربا",
                objective_ids=["o1"],
                authored_by="   ",
            )

    def test_a_skill_with_no_named_author_is_refused(self):
        with self.assertRaises(AuthoringError):
            author_skill(
                book_id=BOOK,
                label="پیش‌بینی",
                objectives=[objective()],
                authored_by="",
            )

    def test_a_relation_with_no_named_author_is_refused(self):
        with self.assertRaises(AuthoringError):
            author_prerequisite(
                book_id=BOOK,
                earlier_id="c1",
                later_id="c2",
                reason="بدون شناخت آهنربا نمی‌توان جذب را فهمید.",
                authored_by="",
            )

    def test_everything_authored_records_a_human_provenance(self):
        # ``human`` is the only extraction method that exempts a record from
        # needing a quotation, so nothing may reach it by accident.
        made = [
            mc_question(),
            author_activity(
                book_id=BOOK,
                activity_type="practice",
                title="بازی آهنربا",
                objective_ids=["o1"],
                authored_by=AUTHOR,
            ),
            author_skill(
                book_id=BOOK,
                label="پیش‌بینی",
                objectives=[objective()],
                authored_by=AUTHOR,
            ),
        ]
        for record in made:
            self.assertIsNotNone(record.provenance)
            self.assertEqual(record.provenance.extraction_method, "human")
            self.assertEqual(record.provenance.authored_by, AUTHOR)

    def test_authoring_never_writes_a_review_decision(self):
        # A reviewer decides; an author does not get to sign their own work off.
        self.assertEqual(mc_question().review_status, "pending")


# ---------------------------------------------------------------------------
# refusals: records nothing downstream could use
# ---------------------------------------------------------------------------


class UnusableRecordsAreRefusedTests(unittest.TestCase):
    def test_a_question_measuring_nothing_is_refused(self):
        with self.assertRaises(AuthoringError):
            mc_question(objective_ids=[])

    def test_an_activity_serving_nothing_is_refused(self):
        with self.assertRaises(AuthoringError):
            author_activity(
                book_id=BOOK,
                activity_type="practice",
                title="بازی",
                objective_ids=[],
                authored_by=AUTHOR,
            )

    def test_a_choice_among_one_thing_is_refused(self):
        with self.assertRaises(AuthoringError):
            mc_question(options=[question_option("میخ", is_correct=True)])

    def test_options_with_no_correct_answer_are_refused(self):
        with self.assertRaises(AuthoringError):
            mc_question(
                options=[question_option("میخ"), question_option("چوب")]
            )

    def test_automatic_marking_with_nothing_to_mark_against_is_refused(self):
        # The failure this prevents lands in front of a child, at the moment
        # they say they are done - not in a report.
        with self.assertRaises(AuthoringError) as caught:
            author_question(
                book_id=BOOK,
                question_type="short_answer",
                prompt="آهنربا چه چیزی را جذب می‌کند؟",
                objective_ids=["o1"],
                authored_by=AUTHOR,
            )
        self.assertIn("nothing to mark", str(caught.exception))

    def test_the_same_item_is_accepted_once_it_has_an_answer_key(self):
        item = author_question(
            book_id=BOOK,
            question_type="short_answer",
            prompt="آهنربا چه چیزی را جذب می‌کند؟",
            objective_ids=["o1"],
            answer="آهن",
            authored_by=AUTHOR,
        )
        self.assertEqual(item.answer_key(), ["آهن"])

    def test_the_same_item_is_accepted_once_marking_is_declared_manual(self):
        item = author_question(
            book_id=BOOK,
            question_type="short_answer",
            prompt="چه دیدی؟",
            objective_ids=["o1"],
            grading=GradingSpec(mode="manual"),
            authored_by=AUTHOR,
        )
        self.assertFalse(item.auto_gradable)

    def test_a_relation_pointing_at_itself_is_refused(self):
        with self.assertRaises(AuthoringError):
            author_prerequisite(
                book_id=BOOK,
                earlier_id="c1",
                later_id="c1",
                reason="چون",
                authored_by=AUTHOR,
            )

    def test_a_skill_generalising_nothing_is_refused(self):
        with self.assertRaises(AuthoringError):
            author_skill(
                book_id=BOOK, label="پیش‌بینی", objectives=[], authored_by=AUTHOR
            )


# ---------------------------------------------------------------------------
# derived identity and inherited certainty
# ---------------------------------------------------------------------------


class DerivedIdentityTests(unittest.TestCase):
    def test_authoring_the_same_item_twice_yields_one_id(self):
        # Otherwise a corrected item silently becomes a second item, and the
        # first goes on being scheduled.
        self.assertEqual(mc_question().id, mc_question().id)

    def test_an_option_id_follows_its_text_not_its_position(self):
        # A stored attempt says "the child chose this option". Re-ordering the
        # options must not change which one that was.
        first = question_option("میخ", is_correct=True)
        again = question_option("میخ")
        self.assertEqual(first.option_id, again.option_id)
        self.assertNotEqual(first.option_id, question_option("چوب").option_id)

    def test_a_skill_is_no_more_certain_than_its_weakest_objective(self):
        skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی نتیجه‌ی یک آزمایش ساده",
            objectives=[
                objective(oid="o1", confidence=0.9),
                objective(oid="o2", confidence=0.4),
            ],
            authored_by=AUTHOR,
        )
        self.assertEqual(skill.confidence, 0.4)

    def test_a_skill_inherits_the_evidence_of_what_it_groups(self):
        skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective(oid="o1"), objective(oid="o2")],
            authored_by=AUTHOR,
        )
        self.assertEqual(skill.evidence_ids, ["e1"])
        self.assertEqual(skill.objective_ids, ["o1", "o2"])


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


class AuthoredStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_book_nobody_has_authored_for_reads_as_empty(self):
        # Not an error. Forcing every caller to create an empty file first
        # would only produce empty files.
        store = load_authored(self.root / AUTHORED_FILENAME, BOOK)
        self.assertTrue(store.is_empty())
        self.assertEqual(store.book_id, BOOK)

    def test_a_round_trip_keeps_every_record(self):
        store = AuthoredContent(book_id=BOOK, questions=[mc_question()])
        path = save_authored(store, self.root / AUTHORED_FILENAME)
        back = load_authored(path, BOOK)
        self.assertEqual(back.counts()["questions"], 1)
        self.assertEqual(back.questions[0].id, store.questions[0].id)

    def test_saving_over_existing_work_is_refused_by_default(self):
        # Nothing regenerates this file; a package can always be rebuilt.
        store = AuthoredContent(book_id=BOOK)
        path = save_authored(store, self.root / AUTHORED_FILENAME)
        with self.assertRaises(AuthoredContentError):
            save_authored(store, path)
        save_authored(store, path, overwrite=True)

    def test_content_authored_for_another_book_is_refused(self):
        store = AuthoredContent(book_id="g1-farsi")
        path = save_authored(store, self.root / AUTHORED_FILENAME)
        with self.assertRaises(AuthoredContentError):
            load_authored(path, BOOK)

    def test_content_written_by_newer_code_is_refused_rather_than_dropped(self):
        path = self.root / AUTHORED_FILENAME
        path.write_text(
            json.dumps({"schema_version": "99.0.0", "book_id": BOOK}),
            encoding="utf-8",
        )
        with self.assertRaises(Exception):
            load_authored(path, BOOK)

    def test_authored_content_sits_beside_the_package_it_feeds(self):
        self.assertEqual(
            authored_path(self.root, 1, "science").parent.name, "science"
        )

    def test_a_directory_is_accepted_where_the_file_is_expected(self):
        save_authored(AuthoredContent(book_id=BOOK), self.root)
        self.assertTrue((self.root / AUTHORED_FILENAME).exists())
        self.assertEqual(load_authored(self.root, BOOK).book_id, BOOK)


# ---------------------------------------------------------------------------
# LINK005 - the two fields that hold one fact
# ---------------------------------------------------------------------------


class SkillLinkTests(unittest.TestCase):
    def test_an_objective_naming_a_skill_that_disowns_it_is_an_error(self):
        skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective(oid="o2")],
            authored_by=AUTHOR,
        )
        schema = schema_with(
            objectives=[objective().model_copy(update={"skill_id": skill.id})],
            skills=[skill],
        )
        self.assertIn("LINK005", codes(schema))

    def test_the_two_agreeing_is_silent(self):
        target = objective()
        skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[target, objective(oid="o2")],
            authored_by=AUTHOR,
        )
        schema = schema_with(
            objectives=[
                target.model_copy(update={"skill_id": skill.id}),
                objective(oid="o2"),
            ],
            skills=[skill],
        )
        self.assertNotIn("LINK005", codes(schema))

    def test_skills_for_objective_reads_the_side_that_owns_the_grouping(self):
        skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective(), objective(oid="o2")],
            authored_by=AUTHOR,
        )
        schema = schema_with(skills=[skill])
        found = schema.skills_for_objective("o1")
        self.assertEqual([s.id for s in found], [skill.id])


# ---------------------------------------------------------------------------
# LINK006 - an activity that tests what it never taught
# ---------------------------------------------------------------------------


class ActivityQuestionAgreementTests(unittest.TestCase):
    def test_an_activity_asking_about_another_objective_is_reported(self):
        # Both halves are individually valid, which is why nothing else in the
        # schema compares them.
        item = mc_question(objective_ids=["o2"])
        activity = author_activity(
            book_id=BOOK,
            activity_type="practice",
            title="بازی آهنربا",
            objective_ids=["o1"],
            question_ids=[item.id],
            authored_by=AUTHOR,
        )
        schema = schema_with(
            objectives=[objective(), objective(oid="o2")],
            questions=[item],
            activities=[activity],
        )
        self.assertIn("LINK006", codes(schema))

    def test_an_activity_asking_about_its_own_objective_is_silent(self):
        item = mc_question()
        activity = author_activity(
            book_id=BOOK,
            activity_type="practice",
            title="بازی آهنربا",
            objective_ids=["o1"],
            question_ids=[item.id],
            authored_by=AUTHOR,
        )
        schema = schema_with(questions=[item], activities=[activity])
        self.assertNotIn("LINK006", codes(schema))


# ---------------------------------------------------------------------------
# LINK007 / FINAL004 - the same thing entered twice
# ---------------------------------------------------------------------------


class DuplicateTests(unittest.TestCase):
    def test_two_skills_with_one_label_are_reported(self):
        first = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective()],
            authored_by=AUTHOR,
        )
        second = first.model_copy(update={"id": "skill-2"})
        schema = schema_with(skills=[first, second])
        self.assertIn("LINK007", codes(schema))

    def test_two_skills_grouping_one_set_of_objectives_are_reported(self):
        # The duplicate a label check misses: two names, one ability.
        first = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective(), objective(oid="o2")],
            authored_by=AUTHOR,
        )
        second = author_skill(
            book_id=BOOK,
            label="حدس زدن نتیجه",
            objectives=[objective(oid="o2"), objective()],
            authored_by=AUTHOR,
        )
        schema = schema_with(
            objectives=[objective(), objective(oid="o2")],
            skills=[first, second],
        )
        self.assertIn("LINK007", codes(schema))

    def test_one_edge_stated_twice_under_two_ids_is_reported(self):
        edge = author_prerequisite(
            book_id=BOOK,
            earlier_id="c1",
            later_id="o1",
            reason="بدون شناخت آهنربا، هدف قابل سنجش نیست.",
            authored_by=AUTHOR,
        )
        twin = edge.model_copy(update={"id": "hand-written-id"})
        schema = schema_with(relations=[edge, twin])
        self.assertIn("FINAL004", codes(schema))

    def test_two_different_edges_are_not_duplicates(self):
        schema = schema_with(
            objectives=[objective(), objective(oid="o2")],
            relations=[
                author_prerequisite(
                    book_id=BOOK,
                    earlier_id="c1",
                    later_id="o1",
                    reason="یک",
                    authored_by=AUTHOR,
                ),
                author_prerequisite(
                    book_id=BOOK,
                    earlier_id="c1",
                    later_id="o2",
                    reason="دو",
                    authored_by=AUTHOR,
                ),
            ],
        )
        self.assertNotIn("FINAL004", codes(schema))


# ---------------------------------------------------------------------------
# QUEST001-003 - would the item work in front of a child
# ---------------------------------------------------------------------------


class QuestionIntegrityTests(unittest.TestCase):
    def test_an_auto_marked_item_with_no_answer_key_is_an_error(self):
        # Reachable only around the authoring API - a hand-edited file, a
        # future generator - which is exactly why the rule exists as well.
        item = Question(
            id="q9",
            question_type="true_false",
            prompt="آهنربا آهن را جذب می‌کند.",
            objective_ids=["o1"],
        )
        self.assertIn("QUEST002", codes(schema_with(questions=[item])))

    def test_a_manually_marked_item_needs_no_answer_key(self):
        item = Question(
            id="q9",
            question_type="drawing",
            prompt="آنچه دیدی را بکش.",
            objective_ids=["o1"],
        )
        self.assertNotIn("QUEST002", codes(schema_with(questions=[item])))

    def test_a_multiple_choice_item_with_one_option_is_an_error(self):
        item = Question(
            id="q9",
            question_type="multiple_choice",
            prompt="کدام؟",
            objective_ids=["o1"],
            options=[QuestionOption(option_id="a", text="میخ", is_correct=True)],
        )
        self.assertIn("QUEST003", codes(schema_with(questions=[item])))

    def test_two_options_sharing_an_id_is_an_error(self):
        # An attempt on it could not say which was chosen.
        item = Question(
            id="q9",
            question_type="multiple_choice",
            prompt="کدام؟",
            objective_ids=["o1"],
            options=[
                QuestionOption(option_id="a", text="میخ", is_correct=True),
                QuestionOption(option_id="a", text="چوب"),
            ],
        )
        self.assertIn("QUEST003", codes(schema_with(questions=[item])))

    def test_an_item_that_cannot_be_got_wrong_is_an_error(self):
        item = Question(
            id="q9",
            question_type="multiple_choice",
            prompt="کدام؟",
            objective_ids=["o1"],
            options=[
                QuestionOption(option_id="a", text="میخ", is_correct=True),
                QuestionOption(option_id="b", text="سنجاق", is_correct=True),
            ],
        )
        found = run_validation(
            ValidationContext(schema_doc=schema_with(questions=[item])),
            stages=["final"],
        )
        messages = [f.message for f in found.findings if f.code == "QUEST003"]
        self.assertTrue(any("cannot be got wrong" in m for m in messages))

    def test_a_type_outside_the_closed_vocabulary_is_an_error(self):
        # Pydantic refuses it on the way in, so this can only fire on a package
        # assembled by code that widened the vocabulary. That is exactly when a
        # value would otherwise reach a renderer with no template for it.
        item = mc_question()
        object.__setattr__(item, "question_type", "interpretive_dance")
        self.assertIn("QUEST001", codes(schema_with(questions=[item])))


# ---------------------------------------------------------------------------
# grading is a decision, not a guess from the form
# ---------------------------------------------------------------------------


class GradingTests(unittest.TestCase):
    def test_every_response_form_declares_who_marks_it(self):
        # A form missing from the table would raise at runtime, in the middle
        # of a package build, on whichever book happened to use it first.
        for form in QUESTION_TYPES:
            self.assertIn(form, DEFAULT_GRADING_MODE)

    def test_auto_gradable_is_derived_from_the_table_not_written_twice(self):
        expected = {
            form
            for form, mode in DEFAULT_GRADING_MODE.items()
            if mode == "auto"
        }
        self.assertEqual(set(AUTO_GRADABLE_TYPES), expected)

    def test_an_author_can_overrule_what_the_form_implies(self):
        item = author_question(
            book_id=BOOK,
            question_type="short_answer",
            prompt="چه فکر می‌کنی؟",
            objective_ids=["o1"],
            answer="هرچه",
            grading=GradingSpec(mode="manual"),
            authored_by=AUTHOR,
        )
        self.assertEqual(DEFAULT_GRADING_MODE["short_answer"], "auto")
        self.assertEqual(item.grading_mode, "manual")
        self.assertFalse(item.auto_gradable)

    def test_hybrid_is_not_automatic(self):
        # A form a machine can partly mark still needs a person before a score
        # means anything; reading "partly" as "yes" shows a child an unchecked
        # mark.
        item = author_question(
            book_id=BOOK,
            question_type="handwriting",
            prompt="بنویس.",
            objective_ids=["o1"],
            answer="آ",
            authored_by=AUTHOR,
        )
        self.assertEqual(item.grading_mode, "hybrid")
        self.assertFalse(item.auto_gradable)

    def test_tolerated_spellings_join_the_answer_key_without_replacing_it(self):
        item = author_question(
            book_id=BOOK,
            question_type="fill_blank",
            prompt="آهنربا ... را جذب می‌کند.",
            objective_ids=["o1"],
            answer="آهن",
            grading=GradingSpec(mode="auto", accepted_answers=["آهنی", "آهن"]),
            authored_by=AUTHOR,
        )
        # The shown answer first, then the tolerances, and no duplicate.
        self.assertEqual(item.answer_key(), ["آهن", "آهنی"])

    def test_the_template_that_draws_it_is_free_text_and_optional(self):
        # The template list belongs to the UI and grows without this schema.
        self.assertIsNone(mc_question().template_id)
        item = mc_question(template_id="multiple-choice")
        self.assertEqual(item.template_id, "multiple-choice")


# ---------------------------------------------------------------------------
# traversal: the questions an engine actually asks
# ---------------------------------------------------------------------------


class EngineTraversalTests(unittest.TestCase):
    """``question -> objective -> concept -> evidence``, end to end."""

    def setUp(self):
        self.item = mc_question()
        self.remedial = author_activity(
            book_id=BOOK,
            activity_type="remediation",
            title="دوباره با آهنربا",
            objective_ids=["o1"],
            authored_by=AUTHOR,
        )
        self.practice = author_activity(
            book_id=BOOK,
            activity_type="practice",
            title="بازی آهنربا",
            objective_ids=["o1"],
            question_ids=[self.item.id],
            order=1,
            authored_by=AUTHOR,
        )
        self.skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی نتیجه‌ی یک آزمایش ساده",
            objectives=[objective(), objective(oid="o2")],
            authored_by=AUTHOR,
        )
        self.prereq = author_prerequisite(
            book_id=BOOK,
            earlier_id="c1",
            later_id="o1",
            reason="بدون شناخت آهنربا این هدف سنجیدنی نیست.",
            authored_by=AUTHOR,
        )
        self.schema = schema_with(
            objectives=[objective(), objective(oid="o2")],
            questions=[self.item],
            activities=[self.practice, self.remedial],
            skills=[self.skill],
            relations=[self.prereq],
        )

    def test_a_question_reaches_its_objectives(self):
        found = self.schema.objectives_for_question(self.item.id)
        self.assertEqual([o.id for o in found], ["o1"])

    def test_a_question_reaches_its_concepts_only_through_its_objectives(self):
        # Re-point the question and the answer changes with it, which is the
        # whole reason it is derived rather than stored.
        self.assertEqual(
            [c.id for c in self.schema.concepts_for_question(self.item.id)],
            ["c1"],
        )
        moved = self.item.model_copy(update={"objective_ids": ["o2"]})
        schema = schema_with(
            objectives=[objective(oid="o2", concept_ids=["c2"])],
            concepts=[concept(cid="c2")],
            questions=[moved],
        )
        self.assertEqual(
            [c.id for c in schema.concepts_for_question(moved.id)], ["c2"]
        )

    def test_a_concept_reaches_the_quotations_behind_it(self):
        found = self.schema.evidence_for("c1")
        self.assertEqual([e.id for e in found], ["e1"])
        self.assertTrue(found[0].quote_verified)

    def test_an_objective_reaches_its_skills(self):
        self.assertEqual(
            [s.id for s in self.schema.skills_for_objective("o1")],
            [self.skill.id],
        )

    def test_an_objective_reaches_what_to_offer_after_a_failure(self):
        remedial = self.schema.activities_for_objective("o1", "remediation")
        self.assertEqual([a.id for a in remedial], [self.remedial.id])
        self.assertTrue(remedial[0].is_remedial)

    def test_an_objective_reaches_everything_that_would_show_it(self):
        self.assertEqual(
            [q.id for q in self.schema.questions_for_objective("o1")],
            [self.item.id],
        )

    def test_a_question_reaches_the_activity_that_asks_it(self):
        found = self.schema.activity_for_question(self.item.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, self.practice.id)

    def test_a_question_nobody_asks_belongs_to_no_activity(self):
        self.assertIsNone(self.schema.activity_for_question("q-unknown"))

    def test_prerequisites_read_in_the_direction_they_are_stored(self):
        self.assertEqual(self.schema.prerequisites_of("o1"), ["c1"])
        self.assertEqual(self.schema.dependents_of("c1"), ["o1"])

    def test_the_whole_chain_holds_in_one_walk(self):
        objectives = self.schema.objectives_for_question(self.item.id)
        concepts = self.schema.concepts_for_objective(objectives[0].id)
        quotes = self.schema.evidence_for(concepts[0].id)
        self.assertTrue(quotes and quotes[0].quote_verified)

    def test_no_learner_state_leaked_into_the_schema(self):
        # Student, attempt, mastery and learning events belong to the engine.
        # The content layer answers what exists, never what a child should do.
        fields = set(ContentSchema.model_fields)
        for forbidden in (
            "students",
            "attempts",
            "mastery",
            "learning_events",
            "profiles",
        ):
            self.assertNotIn(forbidden, fields)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def package_with(content):
    return ContentPackage(
        package_id=ContentPackage.build_id(1, "science", BOOK),
        grade=1,
        subject="science",
        book_id=BOOK,
        stats=compute_stats(content),
        content=content,
    )


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.item = mc_question()
        self.activity = author_activity(
            book_id=BOOK,
            activity_type="remediation",
            title="دوباره",
            objective_ids=["o1"],
            authored_by=AUTHOR,
        )
        self.skill = author_skill(
            book_id=BOOK,
            label="پیش‌بینی",
            objectives=[objective(), objective(oid="o2")],
            authored_by=AUTHOR,
        )
        self.relation = author_prerequisite(
            book_id=BOOK,
            earlier_id="c1",
            later_id="o1",
            reason="چون",
            authored_by=AUTHOR,
        )
        content = schema_with(
            objectives=[objective(), objective(oid="o2")],
            questions=[self.item],
            activities=[self.activity],
            skills=[self.skill],
            relations=[self.relation],
        )
        self.package = package_with(content)
        self.registry = ContentRegistry.from_packages([self.package])

    def test_every_kind_has_a_typed_getter(self):
        self.assertEqual(self.registry.get_concept("c1").id, "c1")
        self.assertEqual(self.registry.get_objective("o1").id, "o1")
        self.assertEqual(self.registry.get_question(self.item.id).id, self.item.id)
        self.assertEqual(
            self.registry.get_activity(self.activity.id).id, self.activity.id
        )
        self.assertEqual(self.registry.get_skill(self.skill.id).id, self.skill.id)
        self.assertEqual(
            self.registry.get_relation(self.relation.id).id, self.relation.id
        )
        self.assertEqual(self.registry.get_evidence("e1").id, "e1")

    def test_a_typed_getter_refuses_the_wrong_kind_rather_than_returning_it(self):
        # An id that is a concept where an objective was expected reads
        # perfectly well - it has an id, it has a label - and the caller finds
        # out several traversals later, if at all.
        self.assertIsNone(self.registry.get_objective("c1"))
        self.assertIsNone(self.registry.get_question("c1"))

    def test_the_book_is_reachable_by_package_id(self):
        book = self.registry.get_book(self.package.package_id)
        self.assertEqual(book.book_id, BOOK)
        self.assertEqual(book.grade, 1)

    def test_a_relation_id_is_indexed_without_becoming_a_valid_endpoint(self):
        # ``entity_ids`` is what FINAL001 checks a relation's endpoints
        # against; a relation pointing at a relation states nothing.
        self.assertEqual(self.registry.kind_of(self.relation.id), "relation")
        self.assertNotIn(self.relation.id, self.package.content.entity_ids())
        self.assertIn(self.relation.id, self.package.content.all_ids())

    def test_the_engine_traversals_delegate_to_the_owning_package(self):
        self.assertEqual(
            [o.id for o in self.registry.objectives_for_question(self.item.id)],
            ["o1"],
        )
        self.assertEqual(
            [s.id for s in self.registry.skills_for_objective("o1")],
            [self.skill.id],
        )
        self.assertEqual(
            [a.id for a in self.registry.remediation_for_objective("o1")],
            [self.activity.id],
        )
        self.assertEqual(self.registry.prerequisites_of("o1"), ["c1"])
        self.assertEqual(
            [e.id for e in self.registry.evidence_for("c1")], ["e1"]
        )

    def test_an_unknown_id_answers_empty_rather_than_raising(self):
        self.assertEqual(self.registry.questions_for_objective("ghost"), [])
        self.assertIsNone(self.registry.get_concept("ghost"))


# ---------------------------------------------------------------------------
# the package carries authored material through unchanged
# ---------------------------------------------------------------------------


class BuildWithAuthoredContentTests(unittest.TestCase):
    def test_an_empty_store_produces_empty_lists_not_invented_ones(self):
        # A package that is incomplete and says so beats one that is complete
        # and padded.
        content = schema_with()
        package = package_with(content)
        self.assertEqual(package.stats.questions, 0)
        self.assertEqual(package.stats.activities, 0)
        self.assertEqual(package.stats.skills, 0)

    def test_authored_material_is_counted_where_a_reader_looks_for_it(self):
        content = schema_with(
            questions=[mc_question()],
            activities=[
                author_activity(
                    book_id=BOOK,
                    activity_type="practice",
                    title="بازی",
                    objective_ids=["o1"],
                    authored_by=AUTHOR,
                )
            ],
        )
        stats = compute_stats(content)
        self.assertEqual(stats.questions, 1)
        self.assertEqual(stats.activities, 1)

    def test_authored_content_for_the_wrong_book_is_refused_by_the_builder(self):
        from content_assistant.package.build import BuildError, build_package
        from content_assistant.models.extraction import (
            BookIdentity,
            DocumentInfo,
            ExtractionResult,
        )

        extraction = ExtractionResult(
            document=DocumentInfo(
                source="x.pdf",
                source_sha256="0" * 64,
                page_count=10,
                book=BookIdentity(book_id=BOOK, grade=1, subject="science"),
            )
        )
        with self.assertRaises(BuildError):
            build_package(
                extraction=extraction,
                authored=AuthoredContent(book_id="g1-farsi"),
            )


# ---------------------------------------------------------------------------
# 1.1.0 -> 1.2.0
#
# The bump this file's features caused. A minor version promises that an
# artifact written before it still reads, and a promise proved only against
# data this code produced is not proved - so the fixtures below are written in
# the old shape by hand.
# ---------------------------------------------------------------------------


class MigrationTests(unittest.TestCase):
    def _package_payload(self, questions=(), activities=()):
        content = schema_with(
            questions=list(questions), activities=list(activities)
        )
        package = package_with(content)
        return json.loads(package.model_dump_json())

    def test_a_1_1_0_question_loads_without_the_fields_1_2_0_added(self):
        payload = self._package_payload(questions=[mc_question()])
        payload["content"]["schema_version"] = "1.1.0"
        for question in payload["content"]["questions"]:
            question.pop("grading", None)
            question.pop("template_id", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "content-package.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            from content_assistant.package.schema import load_content

            loaded = load_content(path)
        item = loaded.content.questions[0]
        self.assertEqual(loaded.content.schema_version, SCHEMA_VERSION)
        self.assertIsNone(item.grading)
        self.assertIsNone(item.template_id)
        # An item that made no decision about marking still answers the
        # question an engine asks, from what its form implies.
        self.assertTrue(item.auto_gradable)

    def test_a_1_1_0_activity_loads_with_no_claim_about_its_order(self):
        # ``None`` is not "first". A scheduler reading a missing order as
        # position zero would put every un-ordered activity ahead of every
        # ordered one.
        activity = author_activity(
            book_id=BOOK,
            activity_type="practice",
            title="بازی",
            objective_ids=["o1"],
            authored_by=AUTHOR,
        )
        payload = self._package_payload(activities=[activity])
        payload["content"]["schema_version"] = "1.1.0"
        for entry in payload["content"]["activities"]:
            entry.pop("order", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "content-package.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            from content_assistant.package.schema import load_content

            loaded = load_content(path)
        self.assertIsNone(loaded.content.activities[0].order)

    def test_authored_content_written_against_1_1_0_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / AUTHORED_FILENAME
            store = AuthoredContent(book_id=BOOK, questions=[mc_question()])
            save_authored(store, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schema_version"] = "1.1.0"
            for question in payload["questions"]:
                question.pop("grading", None)
                question.pop("template_id", None)
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            back = load_authored(path, BOOK)
        self.assertEqual(back.schema_version, SCHEMA_VERSION)
        self.assertEqual(back.counts()["questions"], 1)

    def test_the_response_form_vocabulary_only_ever_grew(self):
        # Removing a value would be a major bump: a stored question naming it
        # would stop loading. This is the list as of 1.1.0.
        for form in (
            "multiple_choice",
            "true_false",
            "fill_blank",
            "matching",
            "ordering",
            "short_answer",
            "drawing",
            "spoken",
            "physical_task",
        ):
            self.assertIn(form, QUESTION_TYPES)


# ---------------------------------------------------------------------------
# the module boundary
# ---------------------------------------------------------------------------


class ImportDirectionTests(unittest.TestCase):
    def test_authoring_imports_cleanly_in_a_fresh_interpreter(self):
        # Every test here imports ``content`` first. A circular import between
        # the layers would resolve here and fail only for a consumer who
        # imported the other one first.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import content_assistant.authoring as a; "
                "print(a.AuthoredContent(book_id='b').is_empty())",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("True", result.stdout)

    def test_no_pipeline_stage_can_reach_the_authoring_api(self):
        # The whole point of the module: a model cannot author a prerequisite,
        # because nothing a model drives imports this.
        import content_assistant.structuring.semantic.concepts as concepts
        import content_assistant.structuring.semantic.objectives as objectives

        for module in (concepts, objectives):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("content_assistant.authoring", source)


if __name__ == "__main__":
    unittest.main()
