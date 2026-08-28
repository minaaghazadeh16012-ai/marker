"""Constructors for the four entity kinds a person has to author.

Every function here does the same three things, and the order matters.

**It refuses what could not be used.** Not what is low quality - what is
structurally unusable. An activity serving no objective would never be
scheduled; a multiple-choice item with one option is not a choice; an item that
promises instant marking with nothing to mark against will fail a child at
runtime rather than at authoring time. These raise, rather than producing a
record for the validator to complain about later, because the person is right
here and can fix it now.

**It records who.** ``extraction_method="human"`` with a named ``authored_by``
is the one provenance that lets a record stand without a quotation - see
``EVID001`` - so it is never inferred and never defaulted. A call without a
name raises.

**It derives the id.** Same book, same content, same id, on every machine and
every re-run. Re-authoring an item that already exists updates it instead of
producing a second copy nobody notices.

Nothing here is reachable by a model. That is the point of the module.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from content_assistant.models.common import (
    DifficultyBand,
    Provenance,
    make_id,
)
from content_assistant.models.content import (
    LearningObjective,
    Relation,
    RelationType,
    Skill,
    human_relation,
    skill_from_objectives,
)
from content_assistant.models.learning import (
    ActivityType,
    ContentType,
    GradingSpec,
    LearningActivity,
    Question,
    QuestionOption,
    QuestionType,
    DEFAULT_GRADING_MODE,
)


class AuthoringError(ValueError):
    """The record as described could not be used by anything downstream."""


def _named(authored_by: str, what: str) -> str:
    name = (authored_by or "").strip()
    if not name:
        raise AuthoringError(
            f"a{'n' if what[0] in 'aeiou' else ''} {what} authored by nobody "
            "cannot be argued with afterwards; pass authored_by"
        )
    return name


def _provenance(stage: str, authored_by: str, generated_at: Optional[str]):
    return Provenance(
        extraction_method="human",
        stage=stage,
        authored_by=authored_by,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# learning intent
# ---------------------------------------------------------------------------


def author_skill(
    *,
    book_id: str,
    label: str,
    objectives: Sequence[LearningObjective],
    concept_ids: Sequence[str] = (),
    description: str = "",
    authored_by: str,
    generated_at: Optional[str] = None,
) -> Skill:
    """Group objectives a person has judged to exercise one ability.

    Thin over :func:`~content_assistant.models.content.skill_from_objectives`,
    which is where the arithmetic lives: the evidence is the union of the
    objectives' and the confidence is the *minimum* of theirs, so a skill can
    never be more certain than its weakest member. What this adds is the
    refusal - a skill needs a label and a person, and the objectives have to be
    the real ones rather than their ids, because inheriting evidence from an id
    is not possible and pretending otherwise is how an ungrounded skill gets in.
    """
    name = _named(authored_by, "skill")
    if not label.strip():
        raise AuthoringError("a skill needs a label; an unnamed ability is not one")
    if not objectives:
        raise AuthoringError(
            "a skill must generalise at least one objective; one with none is "
            "a label with nothing a student could be seen doing"
        )
    return skill_from_objectives(
        book_id=book_id,
        label=label,
        objectives=objectives,
        concept_ids=concept_ids,
        description=description,
        authored_by=name,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


def author_relation(
    *,
    book_id: str,
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    reason: str,
    authored_by: str,
    strength: Literal["hard", "soft"] = "soft",
    confidence: float = 0.0,
    evidence_ids: Sequence[str] = (),
    generated_at: Optional[str] = None,
) -> Relation:
    """Record a typed edge on judgement rather than on a quotation."""
    name = _named(authored_by, "relation")
    if source_id == target_id:
        raise AuthoringError(
            f"{source_id} cannot be related to itself; a self-edge states "
            "nothing and a prerequisite one is a cycle of length one"
        )
    return human_relation(
        book_id=book_id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        reason=reason,
        authored_by=name,
        strength=strength,
        confidence=confidence,
        evidence_ids=evidence_ids,
        generated_at=generated_at,
    )


def author_prerequisite(
    *,
    book_id: str,
    earlier_id: str,
    later_id: str,
    reason: str,
    authored_by: str,
    strength: Literal["hard", "soft"] = "soft",
    evidence_ids: Sequence[str] = (),
    generated_at: Optional[str] = None,
) -> Relation:
    """``earlier_id`` must be met before ``later_id``.

    Named for the direction rather than for the field names, because
    ``source``/``target`` on a prerequisite edge is the one thing everybody
    gets backwards once. The edge is stored exactly as
    :meth:`ContentSchema.prerequisites_of` reads it: source is the
    prerequisite, target is what needs it.
    """
    return author_relation(
        book_id=book_id,
        source_id=earlier_id,
        target_id=later_id,
        relation_type="prerequisite_of",
        reason=reason,
        authored_by=authored_by,
        strength=strength,
        evidence_ids=evidence_ids,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# learning experience
# ---------------------------------------------------------------------------


def question_option(
    text: str, *, is_correct: bool = False, feedback: str = "", option_id: str = ""
) -> QuestionOption:
    """One choice, with an id derived from its own text.

    Deriving rather than counting means re-ordering the options does not
    renumber them, so a stored attempt that recorded "the child chose b" still
    means the same choice after an edit.
    """
    if not text.strip():
        raise AuthoringError("an option with no text cannot be chosen")
    return QuestionOption(
        option_id=option_id or make_id("option", "opt", text)[-10:],
        text=text.strip(),
        is_correct=is_correct,
        feedback=feedback.strip(),
    )


def author_question(
    *,
    book_id: str,
    question_type: QuestionType,
    prompt: str,
    objective_ids: Sequence[str],
    authored_by: str,
    options: Sequence[QuestionOption] = (),
    answer: str = "",
    hints: Sequence[str] = (),
    explanation: str = "",
    grading: Optional[GradingSpec] = None,
    template_id: Optional[str] = None,
    difficulty: Optional[DifficultyBand] = None,
    content_types: Sequence[ContentType] = (),
    generated_at: Optional[str] = None,
) -> Question:
    """One assessable item, refused unless it could actually be marked.

    Three refusals, and each is a runtime failure moved to authoring time.

    An item that measures no objective produces a score nobody can interpret -
    there is no sentence of the form "the student can now ..." that getting it
    right would support. ``LINK002`` reports it in an assembled package; here
    it simply cannot be built.

    A choice among fewer than two things is not a choice, and a set of options
    with none marked correct has no right answer for a marker to find.

    An item whose grading resolves to ``auto`` needs *something to mark
    against*: a correct option, or an answer key. Without one, the engine
    promises a child instant feedback and then has nothing to say.
    """
    name = _named(authored_by, "question")
    if not prompt.strip():
        raise AuthoringError("a question needs a prompt")
    objectives = [o for o in objective_ids if o]
    if not objectives:
        raise AuthoringError(
            "a question must test at least one objective; a score on an item "
            "that measures nothing means nothing"
        )
    options = list(options)
    if question_type == "multiple_choice" and len(options) < 2:
        raise AuthoringError(
            "a multiple-choice question needs at least two options; a choice "
            "among one is not a choice"
        )
    if options and not any(option.is_correct for option in options):
        raise AuthoringError(
            "no option is marked correct; a marker would have no right answer "
            "to find"
        )

    question = Question(
        id=Question.build_id(book_id, question_type, prompt),
        question_type=question_type,
        prompt=prompt.strip(),
        objective_ids=list(dict.fromkeys(objectives)),
        options=options,
        answer=answer.strip(),
        hints=[h.strip() for h in hints if h.strip()],
        explanation=explanation.strip(),
        grading=grading,
        template_id=template_id,
        difficulty=difficulty,
        content_types=list(content_types),
        provenance=_provenance("questions", name, generated_at),
    )
    if question.auto_gradable and not (
        question.correct_options() or question.answer_key()
    ):
        mode_source = "you asked for" if grading else "this form implies"
        raise AuthoringError(
            f"{mode_source} automatic marking, but the item carries neither a "
            "correct option nor an answer key; there is nothing to mark "
            "against. Give it one, or set grading mode 'manual'."
        )
    return question


def author_activity(
    *,
    book_id: str,
    activity_type: ActivityType,
    title: str,
    objective_ids: Sequence[str],
    authored_by: str,
    instructions: str = "",
    question_ids: Sequence[str] = (),
    difficulty: Optional[DifficultyBand] = None,
    content_types: Sequence[ContentType] = (),
    estimated_minutes: Optional[int] = None,
    order: Optional[int] = None,
    generated_at: Optional[str] = None,
) -> LearningActivity:
    """Something a student works through, refused unless it serves something.

    An activity with no objective is material with no place in any path through
    the book: nothing would ever schedule it, and nothing failing would ever
    lead back to it. That is not a quality problem to be scored down, so it is
    not built.
    """
    name = _named(authored_by, "activity")
    if not title.strip():
        raise AuthoringError("an activity needs a title")
    objectives = [o for o in objective_ids if o]
    if not objectives:
        raise AuthoringError(
            "an activity must serve at least one objective; nothing would ever "
            "schedule one that serves none"
        )
    return LearningActivity(
        id=LearningActivity.build_id(book_id, activity_type, title),
        activity_type=activity_type,
        title=title.strip(),
        instructions=instructions.strip(),
        objective_ids=list(dict.fromkeys(objectives)),
        question_ids=list(dict.fromkeys(q for q in question_ids if q)),
        difficulty=difficulty,
        content_types=list(content_types),
        estimated_minutes=estimated_minutes,
        order=order,
        provenance=_provenance("activities", name, generated_at),
    )


def default_grading_mode(question_type: str) -> str:
    """What this response form implies when nobody has decided otherwise."""
    return DEFAULT_GRADING_MODE[question_type]


__all__: List[str] = [
    "AuthoringError",
    "author_activity",
    "author_prerequisite",
    "author_question",
    "author_relation",
    "author_skill",
    "default_grading_mode",
    "question_option",
]
