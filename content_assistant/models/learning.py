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
LEARNING_SCHEMA_VERSION = "1.1.0"

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

#: The forms a question can take. Grade one is mostly the first five; the last
#: three exist because a first-grade science book asks a child to draw, to say
#: something out loud, and to do something with their hands, and a schema that
#: could not express those would push real assessment out of the system.
QuestionType = Literal[
    "multiple_choice",
    "true_false",
    "fill_blank",
    "matching",
    "ordering",
    "short_answer",
    "drawing",
    "spoken",
    "physical_task",
]

QUESTION_TYPES = (
    "multiple_choice",
    "true_false",
    "fill_blank",
    "matching",
    "ordering",
    "short_answer",
    "drawing",
    "spoken",
    "physical_task",
)

#: Question forms a machine can mark on its own. The rest need a person or a
#: separate grader, and an engine has to know which it is holding before it
#: promises a student instant feedback.
AUTO_GRADABLE_TYPES = frozenset(
    {"multiple_choice", "true_false", "matching", "ordering"}
)


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
    difficulty: Optional[DifficultyBand] = None
    content_types: List[ContentType] = Field(default_factory=list)

    @property
    def auto_gradable(self) -> bool:
        """Can a machine mark this without a person looking at it?"""
        return self.question_type in AUTO_GRADABLE_TYPES

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
