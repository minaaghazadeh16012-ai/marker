"""The learning-experience layer: what a student *does*, and what it measures.

Three layers have to stay apart, and this module is the third of them:

===========================  =========================================
layer                        bound by
===========================  =========================================
content knowledge            evidence - a concept must quote the book
learning intent              evidence - an objective must quote its
                             concept's own blocks
learning experience          **linkage** - an activity must serve an
                             objective, a question must test one
===========================  =========================================

Nothing here inherits :class:`~content_assistant.models.content.Grounded`, and
that is the design rather than an oversight. A matching game is not an
assertion about the textbook; asking it for a verified quotation would be a
category error, and the rule that would follow ("every activity must cite a
page") would either be ignored or satisfied with a decorative citation. What
an activity *can* be held to is that it serves something real, which
``LINK001`` and ``LINK002`` check.

The other half of the separation runs the other way. **A question does not own
pedagogical knowledge.** It names the objectives it tests and stops there; the
concepts it touches are reached *through* those objectives and are never
stored on it. That is why there is no ``concept_ids`` field below and why
:meth:`~content_assistant.models.content.ContentSchema.concepts_for_question`
exists instead: one fact, one place, and a question that drifts from its
objective cannot silently keep pointing at the right concept.

Nothing in this pipeline generates activities or questions. This module is the
contract an authoring tool or a generator writes against, and the validation
rules it will be held to.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Only ``common``, never ``content``. ``content`` imports this module to give
# ContentSchema its activity and question lists, so a reference back would put
# the two in a circle that resolves or fails depending on which is imported
# first - the worst kind, because the tests that happen to import ``content``
# first would never see it.
from content_assistant.models.common import (
    Attributed,
    ContentType,
    DifficultyBand,
    make_id,
)

#: Bumped with the content schema it belongs to.
LEARNING_SCHEMA_VERSION = "1.2.0"

#: What a student is doing. Closed, because an open list stops being a
#: vocabulary: a scheduler choosing "something else to try" has to be able to
#: reason over these values, and it cannot reason over free text.
#:
#: The three that carry scheduling meaning rather than format meaning are
#: ``explanation`` (teach it), ``remediation`` (teach it again, differently,
#: after a failure) and ``review`` (return to it later). The rest describe how
#: the activity is presented.
ActivityType = Literal[
    "explanation",
    "guided_practice",
    "practice",
    "game",
    "matching",
    "drag_drop",
    "sorting",
    "handwriting",
    "audio",
    "visual",
    "remediation",
    "review",
]

ACTIVITY_TYPES = (
    "explanation",
    "guided_practice",
    "practice",
    "game",
    "matching",
    "drag_drop",
    "sorting",
    "handwriting",
    "audio",
    "visual",
    "remediation",
    "review",
)

#: The **response form** a question takes - what the student does to answer,
#: not what it is rendered with. The distinction is the whole reason this list
#: is closed and short while the renderer's template list is open and long: a
#: template is a way of drawing a question and new ones arrive whenever the UI
#: grows, whereas a response form is what a scheduler and a grader reason over.
#: ``template_id`` on :class:`Question` is the seam between the two.
#:
#: The last four exist because a first-grade book asks a child to draw, to
#: colour, to write by hand and to say something out loud, and a schema that
#: could not express those would push real assessment out of the system.
QuestionType = Literal[
    "multiple_choice",
    "true_false",
    "fill_blank",
    "short_answer",
    "long_answer",
    "matching",
    "ordering",
    "grouping",
    "table_fill",
    "column_arithmetic",
    "text_selection",
    "drawing",
    "coloring",
    "handwriting",
    "spoken",
    "physical_task",
]

QUESTION_TYPES = (
    "multiple_choice",
    "true_false",
    "fill_blank",
    "short_answer",
    "long_answer",
    "matching",
    "ordering",
    "grouping",
    "table_fill",
    "column_arithmetic",
    "text_selection",
    "drawing",
    "coloring",
    "handwriting",
    "spoken",
    "physical_task",
)

#: Who can mark an answer. ``hybrid`` is not a hedge: it is the form whose
#: shape a machine can check while its content still needs a person - a
#: handwriting task where the letter is recognisable but the stroke order is
#: not, a spoken answer matched against a word list.
GradingMode = Literal["auto", "manual", "hybrid"]

#: What each response form can be marked by when nobody has said otherwise.
#: A default, never a verdict: an author who knows this particular item cannot
#: be machine-marked says so on the item, and :attr:`Question.grading` wins.
#:
#: ``fill_blank`` and ``short_answer`` are ``auto`` here because a one-word
#: answer in a first-grade book is checked against an answer key, not judged.
#: ``long_answer`` and the two picture forms are not, because there is nothing
#: to check them against that would not be invented.
DEFAULT_GRADING_MODE = {
    "multiple_choice": "auto",
    "true_false": "auto",
    "fill_blank": "auto",
    "short_answer": "auto",
    "long_answer": "manual",
    "matching": "auto",
    "ordering": "auto",
    "grouping": "auto",
    "table_fill": "auto",
    "column_arithmetic": "auto",
    "text_selection": "auto",
    "drawing": "manual",
    "coloring": "manual",
    "handwriting": "hybrid",
    "spoken": "hybrid",
    "physical_task": "manual",
}

#: Question forms a machine can mark on its own, absent an explicit decision.
#: Derived from the table above rather than written twice, so the two can never
#: disagree about a form.
AUTO_GRADABLE_TYPES = frozenset(
    form for form, mode in DEFAULT_GRADING_MODE.items() if mode == "auto"
)


class GradingSpec(BaseModel):
    """How this particular item is marked, when the form's default is wrong.

    Two facts an engine needs before it can promise a child instant feedback,
    and neither is derivable from the response form alone. *Who marks it* -
    because an author can know that this short answer is open-ended even though
    short answers usually are not. *What counts as right* - because Persian is
    written more than one way and an answer key that accepts exactly one
    spelling marks a correct child wrong.

    ``accepted_answers`` does not duplicate :attr:`Question.answer`: the
    question holds the answer that will be *shown*, this holds the other
    spellings that also *count*. One is what a student is told; the other is
    what a marker tolerates.
    """

    mode: GradingMode
    #: Whether a partly-right answer earns part of the credit. Off by default:
    #: a scheduler reading "0.5" has to know what half of this item means, and
    #: for most first-grade forms nothing does.
    partial_credit: bool = False
    #: Other spellings of the right answer that a marker must accept.
    accepted_answers: List[str] = Field(default_factory=list)
    #: Apply the pipeline's own Persian normalization before comparing. On by
    #: default, because the alternative is failing a child over an Arabic yeh.
    normalize_answer: bool = True
    #: What this item is worth relative to its siblings.
    points: float = 1.0


class QuestionOption(BaseModel):
    """One choice offered for a question.

    ``feedback`` is per-option on purpose. "Wrong, try again" tells a student
    nothing and tells an error-analysis pass even less; what a particular wrong
    choice reveals is the thing worth writing down, and it can only be written
    beside that choice.
    """

    option_id: str
    text: str
    is_correct: bool = False
    feedback: str = ""


class Question(Attributed):
    """One assessable item. It tests objectives; it does not own concepts."""

    id: str
    question_type: QuestionType
    prompt: str
    #: The objectives this item measures. At least one, always: an item that
    #: measures nothing produces a score nobody can interpret, and ``LINK002``
    #: refuses it. This is the only link a question owns - see the module
    #: docstring on why ``concept_ids`` is absent.
    objective_ids: List[str] = Field(default_factory=list)
    options: List[QuestionOption] = Field(default_factory=list)
    #: The expected answer for the forms that have one and no options
    #: (``fill_blank``, ``short_answer``). Left empty for the rest.
    answer: str = ""
    #: Ordered from the most abstract nudge to the most concrete step, so a
    #: tutor can escalate rather than choose. Empty is fine; a wrong order is
    #: not, which is why the field is a list and not a set.
    hints: List[str] = Field(default_factory=list)
    #: Why the right answer is right. Shown after the attempt, not before.
    explanation: str = ""
    #: How this item is marked, when the form's default is not right for it.
    #: ``None`` means "whatever this form usually implies" - see
    #: :data:`DEFAULT_GRADING_MODE` - which is the honest reading of an item
    #: nobody has made a decision about.
    grading: Optional[GradingSpec] = None
    #: Which renderer draws it, when the content layer knows. Free text and
    #: deliberately so: the template list belongs to the UI, it grows without
    #: the content schema, and a closed copy here would have to be edited every
    #: time a template is added. Left empty, a consumer picks a template from
    #: :attr:`question_type`.
    template_id: Optional[str] = None
    difficulty: Optional[DifficultyBand] = None
    content_types: List[ContentType] = Field(default_factory=list)

    @property
    def grading_mode(self) -> str:
        """Who marks this item: what the author said, or what the form implies.

        A form this version does not know answers ``"manual"`` rather than
        raising. Two reasons, and the second is the important one. A record
        can hold an unknown form - a package written by newer code, a value
        that got past pydantic - and the safe reading of "I do not know what
        this is" is never "a machine can mark it". And ``QUEST001`` is the rule
        that reports such a value; a property that raised here would take the
        whole validation report down with it, losing that finding and every
        other one alongside it.
        """
        if self.grading is not None:
            return self.grading.mode
        return DEFAULT_GRADING_MODE.get(self.question_type, "manual")

    @property
    def auto_gradable(self) -> bool:
        """Can a machine mark this without a person looking at it?

        ``hybrid`` answers ``False``. A form a machine can *partly* mark still
        needs a person before a score means anything, and an engine that read
        "partly" as "yes" would show a child a mark nobody had checked.
        """
        return self.grading_mode == "auto"

    def answer_key(self) -> List[str]:
        """Everything that counts as a right answer, in one list.

        The shown answer first, then the tolerated spellings. Empty for the
        forms whose correctness lives in :attr:`options` instead.
        """
        keys = [self.answer] if self.answer else []
        if self.grading:
            keys += [
                text
                for text in self.grading.accepted_answers
                if text and text not in keys
            ]
        return keys

    def correct_options(self) -> List[QuestionOption]:
        return [option for option in self.options if option.is_correct]

    @staticmethod
    def build_id(book_id: str, question_type: str, prompt: str) -> str:
        return make_id(book_id, "question", question_type, prompt)


class LearningActivity(Attributed):
    """Something a student works through in order to meet an objective.

    Separate from :class:`Question` because most of what a six-year-old does is
    not a question - it is a game, a sorting task, a thing to build. An
    activity may own questions, and an activity with none is still an activity.

    Remediation is an ``activity_type``, not a separate entity or a second set
    of links: the activity that follows a failure serves exactly the objective
    that was failed. Giving it its own ``remediates`` field would restate
    ``objective_ids`` and let the two disagree.
    """

    id: str
    activity_type: ActivityType
    title: str
    instructions: str = ""
    #: What this activity is for. At least one, always - ``LINK002``.
    objective_ids: List[str] = Field(default_factory=list)
    #: Questions asked inside this activity, in the order they are asked.
    question_ids: List[str] = Field(default_factory=list)
    difficulty: Optional[DifficultyBand] = None
    content_types: List[ContentType] = Field(default_factory=list)
    estimated_minutes: Optional[int] = None
    #: Where this activity sits among the ones serving the same objective, when
    #: the order matters - explain it, then practise it, then play with it.
    #: ``None`` means the author made no claim about order, which is different
    #: from claiming it comes first; a scheduler must not read a missing order
    #: as position zero.
    order: Optional[int] = None

    # The concepts, prerequisites and skills behind this activity are all
    # reached through its objectives and are deliberately not stored here. See
    # ContentSchema.concepts_for_activity() and .prerequisites_of().

    @property
    def is_remedial(self) -> bool:
        """Is this what a scheduler reaches for after a student fails?"""
        return self.activity_type == "remediation"

    @staticmethod
    def build_id(book_id: str, activity_type: str, title: str) -> str:
        return make_id(book_id, "activity", activity_type, title)
