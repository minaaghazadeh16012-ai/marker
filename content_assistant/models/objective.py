"""The objective vocabulary: what an objective may say, and how it is judged.

Versioned on its own, and deliberately separate from the L1 content model. The
content model says what a
:class:`~content_assistant.models.content.LearningObjective` *is*; this module
says what makes one acceptable - which performance verbs count as observable,
which ones are wishes rather than behaviour, and which kinds of objective
belong to which kind of concept. Those are pedagogical judgements that will be
argued with and revised, so they live in one file with their own version
rather than scattered through the extractor.

The lexicon is closed on purpose. An open-ended "any verb the model likes"
makes the observability check unenforceable: every objective looks assessable
if the grader is the same model that wrote it. A fixed lexicon makes the check
a lookup, and a verb outside it is a visible decision to extend this file.

Field naming, since it is the first question anyone asks: this module does not
introduce a second objective type. The pipeline writes ``LearningObjective``
records straight into ``ContentSchema``, and the stage artifact carries the
run-scoped fields around them.

===========================  ==========================================
asked for                    where it lives
===========================  ==========================================
``objective_id``             ``LearningObjective.id``
``concept_id``               ``LearningObjective.concept_ids`` - a list,
                             because the field already existed as one
``objective_type``           ``LearningObjective.objective_type``
``statement``                ``LearningObjective.statement``
``evidence_refs``            ``LearningObjective.evidence_ids``
``source_lesson``            ``LearningObjective.lesson_id``
``grade``                    ``ObjectiveExtractionResult.grade`` - it is a
                             property of the book, and copying it onto
                             every objective would make a second source of
                             truth for one fact
``confidence``               ``LearningObjective.confidence``
``needs_review``             ``LearningObjective.requires_human_review``
``review_reasons``           ``LearningObjective.review_reasons``
===========================  ==========================================
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from content_assistant.models.content import ObjectiveType  # noqa: F401
from content_assistant.text.vocabulary import word_forms

#: Bumped when a field's meaning or the vocabulary below changes, so a stored
#: artifact can always be read against the rules that produced it.
OBJECTIVE_SCHEMA_VERSION = "1.0.0"

#: The six kinds, as data. :data:`ObjectiveType` is the same list as a type,
#: and it is declared in :mod:`content_assistant.models.content` beside the
#: other closed vocabularies rather than here, so the L1 model never has to
#: import this module to describe its own field.
OBJECTIVE_TYPES: Tuple[str, ...] = (
    "name",
    "identify",
    "describe",
    "classify",
    "compare",
    "perform",
)

#: The performance verbs each kind of objective may use, in the plain
#: classroom Persian a first-grade book speaks.
#:
#: These are the pipeline's own words rather than the book's, which matters to
#: the wording check: an objective is written *about* the lesson, so its verb
#: comes from here while every other word must come from the lesson itself.
#: See :func:`lexicon_words`.
PERFORMANCE_VERBS: Dict[str, Tuple[str, ...]] = {
    "name": ("بگوید", "نام ببرد"),
    "identify": ("نشان دهد", "پیدا کند", "تشخیص دهد", "مشخص کند"),
    "describe": ("توضیح دهد", "شرح دهد"),
    "classify": ("دسته بندی کند", "گروه کند", "جدا کند"),
    "compare": ("مقایسه کند", "فرق را بگوید"),
    "perform": (
        "انجام دهد",
        "بسازد",
        "رعایت کند",
        "درست کند",
        "بکشد",
        "آزمایش کند",
    ),
}

#: Verbs naming a state of mind rather than a behaviour. An objective built on
#: one of these cannot be observed, so it cannot be assessed, so no one can
#: ever say whether the lesson achieved it. It is a wish, not an objective.
VAGUE_PERFORMANCES: Tuple[str, ...] = (
    "بداند",
    "بدانند",
    "درک کند",
    "درک کنند",
    "بفهمد",
    "بفهمند",
    "آشنا شود",
    "آشنا شوند",
    "آگاه شود",
    "آگاه باشد",
    "یاد بگیرد",
    "یاد بگیرند",
    "متوجه شود",
    "پی ببرد",
    "پی ببرند",
    "علاقه مند شود",
    "احساس کند",
    "باور کند",
    "شناخت پیدا کند",
)

#: Which kinds of objective suit which kind of concept.
#:
#: A concept about *doing* something wants an objective about doing it. Asking
#: a first grader to name a safety rule instead of following it measures the
#: wrong thing, and a concept the book teaches as an idea does not turn into a
#: procedure because an objective was phrased as one.
#:
#: A mismatch is reported for review, never rejected: the pairing is a
#: judgement, and this pipeline does not overrule a person on judgement.
OBJECTIVE_TYPES_FOR_CONCEPT: Dict[str, FrozenSet[str]] = {
    "conceptual": frozenset(
        {"name", "identify", "describe", "classify", "compare"}
    ),
    "procedural": frozenset({"perform", "describe"}),
    "representational": frozenset({"identify", "describe", "name"}),
    "language": frozenset({"name", "describe", "identify"}),
    "meta": frozenset({"perform", "describe"}),
}

#: Words an objective sentence needs in order to be a sentence about a student,
#: which say nothing about whether it followed the book.
_SCAFFOLDING: Tuple[str, ...] = ("دانش آموز", "بتواند", "بتوانند")


def _phrase_words(text: str) -> Set[str]:
    """The word forms of a phrase, read through any ZWNJ.

    Shared by every comparison here so that ``دانش‌آموز`` and ``دانشآموز`` are
    one phrase and not two - the book usually lost its ZWNJ, a model writes it.
    """
    return word_forms(text)


def is_vague(text: str) -> bool:
    """Does this text rest on a state of mind rather than a behaviour?

    A phrase matches when every one of its words is present, so
    ``دانش‌آموز باید بداند`` is caught by ``بداند`` without the list having to
    enumerate the sentences it might appear in.
    """
    present = _phrase_words(text)
    return any(
        _phrase_words(phrase) <= present for phrase in VAGUE_PERFORMANCES
    )


def known_performance_verbs() -> Set[str]:
    """Every verb phrase the lexicon allows, across all objective types."""
    return {verb for verbs in PERFORMANCE_VERBS.values() for verb in verbs}


def verb_is_observable(verb: str) -> bool:
    """Is this verb one the lexicon recognises as a visible performance?"""
    if not verb or not verb.strip():
        return False
    if is_vague(verb):
        return False
    given = _phrase_words(verb)
    if not given:
        return False
    return any(
        _phrase_words(known) <= given for known in known_performance_verbs()
    )


def verbs_for(objective_type: str) -> Tuple[str, ...]:
    """The verbs this kind of objective is written with."""
    return PERFORMANCE_VERBS.get(objective_type, ())


def type_fits_concept(objective_type: str, concept_type: str) -> bool:
    """Does this kind of objective belong to this kind of concept?"""
    allowed = OBJECTIVE_TYPES_FOR_CONCEPT.get(concept_type)
    if allowed is None:
        # An unfamiliar concept type is not evidence of a bad pairing. That
        # vocabulary belongs to the concept layer, and guessing here would
        # report a fault that lives somewhere else.
        return True
    return objective_type in allowed


def lexicon_words() -> Set[str]:
    """Every word this module contributes to an objective statement.

    The wording check asks whether an objective speaks its lesson's language.
    Its performance verb never can - that comes from :data:`PERFORMANCE_VERBS`,
    not from the book - so those words are subtracted before the question is
    put. Everything else in the statement still has to be the lesson's.

    The vague verbs are subtracted too. They have their own check, with a
    message that says what is actually wrong; letting them surface again as
    "wording the lesson never uses" would report one fault twice and bury the
    half that tells a reviewer what to do.
    """
    words: Set[str] = set()
    for phrase in known_performance_verbs():
        words |= _phrase_words(phrase)
    for phrase in VAGUE_PERFORMANCES:
        words |= _phrase_words(phrase)
    for phrase in _SCAFFOLDING:
        words |= _phrase_words(phrase)
    return words


def strip_lexicon(words: Sequence[str]) -> List[str]:
    """Drop this module's own vocabulary from a wording report."""
    ours = lexicon_words()
    return [word for word in words if word not in ours]
