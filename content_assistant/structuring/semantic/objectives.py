"""Objective extraction: what a student can be seen doing, from concepts alone.

This stage runs *after* concepts and takes them as its input. That ordering is
the design, not a convenience. A concept is a claim about what the lesson
teaches, already grounded and already scored; an objective is a claim about
what a student does with it. Deriving the second from the first means an
objective can never be better evidenced than the idea it serves, and the code
enforces exactly that - see the cap in :func:`compute_objective_confidence`.

Three properties are structural here rather than editorial, because editorial
rules are the ones a model talks its way around:

**An objective may only cite its own concept's blocks.** Enforced by
:func:`~content_assistant.structuring.semantic.proposals.admit_objective_proposals`.
An objective needing other evidence is asserting something its concept does
not, which is a new concept wearing an objective's clothes.

**An objective must name a performance from a closed lexicon.** Enforced by
:mod:`content_assistant.models.objective`. ``بداند`` and ``درک کند`` are not
assessable, and a check that let the writing model choose its own verbs would
be graded by the same judgement that produced them.

**An objective is written in the lesson's words.** The same wording check the
concept layer uses, with this pipeline's own performance verbs subtracted -
those come from the lexicon, not from the book, and reporting them would drown
the words that actually matter.

Nothing here produces skills, misconceptions or relations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from content_assistant.models.content import (
    Concept,
    Evidence,
    LearningObjective,
    make_id,
)
from content_assistant.models.objective import (
    OBJECTIVE_SCHEMA_VERSION,
    is_vague,
    strip_lexicon,
    type_fits_concept,
    verb_is_observable,
    verbs_for,
)
from content_assistant.structuring.evidence import EvidenceUnit
from content_assistant.structuring.semantic.concepts import (
    ConfidenceBreakdown,
    PromptTemplate,
    load_prompt,
    material_note,
)
from content_assistant.structuring.semantic.llm import LLMRequest
from content_assistant.structuring.semantic.proposals import (
    ObjectiveAdmissionResult,
    ObjectiveProposal,
    ObjectiveResponse,
    admit_objective_proposals,
)
from content_assistant.structuring.verify import (
    ClaimCitation,
    VerificationOutcome,
    block_page_index,
    verify_claim,
)
from content_assistant.text.vocabulary import (
    VocabularyConfig,
    check_wording,
    joined_tokens,
)

#: Review routing for objectives. Deliberately separate constants from the
#: concept layer's: the two stages measure different things, and tying them
#: together would mean re-tuning concepts to move objectives.
OBJECTIVE_AUTO_ACCEPT_MIN = 0.85
OBJECTIVE_REVIEW_QUEUE_MIN = 0.60

#: Two objectives on one concept whose statements overlap this much are the
#: same objective said twice. Flagged, never silently dropped.
OBJECTIVE_DUPLICATE_OVERLAP = 0.75

#: More objectives than this on a single concept is a symptom of padding
#: rather than of a rich concept, so the concept is sent for review. A
#: first-grade concept that genuinely supports four distinct performances is
#: rare enough to be worth a person's glance.
MAX_OBJECTIVES_PER_CONCEPT = 3


class ModelCallFailed(RuntimeError):
    """The provider returned nothing usable, and that is not an answer.

    Marker's services return an empty dict when their retries are exhausted -
    a 429, a timeout, an unparseable reply all end the same way. Validated
    against the response schema, ``{}`` becomes a perfectly well-formed reply
    with zero objectives, which is indistinguishable from a model that read
    the lesson and correctly concluded there was nothing to write.

    That collapse was measured, not imagined: ten lessons in a row recorded
    "0 objectives, validation ok" while every call behind them had failed on
    quota. An artifact that says a lesson has no objectives is a claim about
    the book, and it may not be produced by a call that never arrived.
    """



# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def load_objective_prompt(
    name: str = "objective_v1", language: str = "fa"
) -> PromptTemplate:
    """Load the versioned objective prompt.

    Same loader as the concept stage, so the version is the file's own content
    hash and an edited prompt is a different version without anyone having to
    remember to bump a number.
    """
    return load_prompt(name, language)


def concept_blocks(
    concepts: Sequence[Concept], evidence: Sequence[Evidence]
) -> Dict[str, Set[str]]:
    """Which blocks each concept is actually grounded in.

    This is the admission rule's input and the prompt's input at once, which is
    the point: the model is shown exactly the blocks it will be allowed to
    cite, so a rejection is a model ignoring what it was given rather than a
    model guessing at a boundary nobody drew for it.
    """
    by_id = {item.id: item for item in evidence}
    out: Dict[str, Set[str]] = {}
    for concept in concepts:
        blocks = {
            by_id[evidence_id].block_id
            for evidence_id in concept.evidence_ids
            if evidence_id in by_id
        }
        out[concept.id] = blocks
    return out


def render_concepts(
    concepts: Sequence[Concept],
    evidence: Sequence[Evidence],
    unit: EvidenceUnit,
) -> str:
    """Lay out each concept with the exact text it rests on.

    A model writing objectives needs the concept and its sentences together;
    giving it the whole lesson again would invite objectives grounded in
    whatever else it found there, which is the failure this stage exists to
    prevent.
    """
    by_id = {item.id: item for item in evidence}
    texts = unit.block_text()
    lines: List[str] = []
    for concept in concepts:
        lines.append(f"### مفهوم `{concept.id}`")
        lines.append(f"- عنوان: {concept.label}")
        if concept.definition:
            lines.append(f"- تعریف: {concept.definition}")
        lines.append(f"- نوع مفهوم: `{concept.concept_type}`")
        allowed = verbs_for_concept(concept.concept_type)
        if allowed:
            lines.append(f"- نوع هدف مجاز برای این مفهوم: {allowed}")
        lines.append("- بلاک‌های مجاز برای citation (فقط همین‌ها):")
        seen: Set[str] = set()
        for evidence_id in concept.evidence_ids:
            item = by_id.get(evidence_id)
            if item is None or item.block_id in seen:
                continue
            seen.add(item.block_id)
            text = texts.get(item.block_id, item.quote)
            lines.append(f"  - `{item.block_id}`: {text}")
        if not seen:
            lines.append("  - (هیچ بلاکی — برای این مفهوم هدف ننویس)")
        lines.append("")
    return "\n".join(lines).strip()


def verbs_for_concept(concept_type: str) -> str:
    """The objective types allowed for a concept type, as prompt text."""
    from content_assistant.models.objective import OBJECTIVE_TYPES_FOR_CONCEPT

    allowed = OBJECTIVE_TYPES_FOR_CONCEPT.get(concept_type)
    if not allowed:
        return ""
    return "، ".join(f"`{name}`" for name in sorted(allowed))


def render_verb_lexicon() -> str:
    """The closed verb list, as the prompt's own table."""
    from content_assistant.models.objective import OBJECTIVE_TYPES

    lines = ["| `objective_type` | فعل‌های مجاز |", "|---|---|"]
    for name in OBJECTIVE_TYPES:
        verbs = "، ".join(verbs_for(name))
        lines.append(f"| `{name}` | {verbs} |")
    return "\n".join(lines)


def build_objective_prompt(
    unit: EvidenceUnit,
    concepts: Sequence[Concept],
    evidence: Sequence[Evidence],
    template: PromptTemplate,
) -> str:
    replacements = {
        "{{LESSON_NUMBER}}": str(unit.lesson_number),
        "{{LESSON_TITLE}}": unit.lesson_title,
        "{{GRADE}}": str(unit.grade),
        "{{SUBJECT}}": unit.subject,
        "{{MATERIAL_NOTE}}": material_note(unit),
        "{{VERB_LEXICON}}": render_verb_lexicon(),
        "{{CONCEPTS}}": render_concepts(concepts, evidence, unit),
    }
    text = template.text
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


# ---------------------------------------------------------------------------
# confidence + review routing
# ---------------------------------------------------------------------------


def compute_objective_confidence(
    outcome: VerificationOutcome,
    proposal: ObjectiveProposal,
    concept: Concept,
    unit: EvidenceUnit,
    observable: bool,
    type_fits: bool,
    foreign_words: Sequence[str],
) -> ConfidenceBreakdown:
    """Score an objective from what could be checked, not from what was said.

    The last step is the one that matters most: the score is capped at the
    confidence of the concept it serves. An objective is a claim *about* a
    concept, so believing it more than the concept is incoherent - and without
    the cap a well-phrased objective on a shaky concept would sail past review
    while the concept itself sat in the queue.
    """
    breakdown = ConfidenceBreakdown()
    verified = [e for e in outcome.evidence if e.quote_verified]

    if verified:
        methods = {e.match_method for e in verified}
        if methods & {"exact", "normalized"}:
            breakdown.add("quote_verified_exact", 0.50)
        else:
            breakdown.add("quote_verified_fuzzy", 0.30)
    else:
        breakdown.add("no_verified_quote", 0.0)

    if observable:
        breakdown.add("observable_performance", 0.15)
    if type_fits:
        breakdown.add("type_suits_concept", 0.10)
    if not foreign_words:
        breakdown.add("wording_is_the_books", 0.10)
    if len(verified) >= 2:
        breakdown.add("multiple_independent_citations", 0.05)
    if unit.material_profile.text_density == "low":
        breakdown.add("thin_lesson_penalty", -0.10)

    if proposal.model_confidence is not None:
        breakdown.add(
            "model_self_report_capped",
            min(0.10, max(0.0, proposal.model_confidence) * 0.10),
        )

    if outcome.evidence_level == "needs_visual_review":
        breakdown.score = round(min(breakdown.score, 0.50), 4)
        breakdown.components["visual_cap"] = 0.50

    if breakdown.score > concept.confidence:
        breakdown.score = round(concept.confidence, 4)
        breakdown.components["capped_at_concept_confidence"] = round(
            concept.confidence, 4
        )

    return breakdown


def objective_review_reasons(
    *,
    outcome: VerificationOutcome,
    confidence: float,
    concept: Concept,
    observable: bool,
    vague: bool,
    type_fits: bool,
    objective_type: str,
    foreign_words: Sequence[str],
) -> Tuple[bool, List[str]]:
    """Decide whether a person has to look at this objective, and say why."""
    reasons: List[str] = []

    if vague:
        reasons.append(
            "states a state of mind rather than a behaviour; it cannot be "
            "observed, so it cannot be assessed"
        )
    elif not observable:
        reasons.append(
            "no performance verb from the objective lexicon; nobody can say "
            "what the student would be seen doing"
        )

    if not type_fits:
        reasons.append(
            f"objective_type {objective_type!r} does not suit a "
            f"{concept.concept_type!r} concept"
        )

    if foreign_words:
        words = "، ".join(foreign_words[:8])
        reasons.append(f"wording is not the lesson's: {words}")

    if not any(e.quote_verified for e in outcome.evidence):
        reasons.append("no cited quotation could be found in the book")
    if outcome.demoted:
        reasons.append("claim was demoted during verification")
    if outcome.evidence_level == "needs_visual_review":
        reasons.append("rests on an image; a person must look at the page")

    if concept.requires_human_review:
        reasons.append(
            f"the concept it rests on ({concept.id}) is itself under review"
        )

    if confidence < OBJECTIVE_REVIEW_QUEUE_MIN:
        reasons.append(f"confidence {confidence:.2f} below "
                       f"{OBJECTIVE_REVIEW_QUEUE_MIN}")
    elif confidence < OBJECTIVE_AUTO_ACCEPT_MIN:
        reasons.append(f"confidence {confidence:.2f} below auto-accept")

    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# duplicate flagging
# ---------------------------------------------------------------------------


def flag_duplicate_objectives(
    objectives: Sequence[LearningObjective],
) -> Dict[str, List[str]]:
    """Report objectives that say the same thing. Flag only, never merge.

    Scoped to objectives sharing a concept: two lessons may legitimately ask a
    student to name something, and calling those duplicates would bury the
    case that matters - one concept padded out with the same objective twice.

    Statements are cut into words rather than split on spaces, so a trailing
    ``.`` does not make one sentence two. Splitting on whitespace let the same
    objective written twice, once with a full stop, fall under the threshold -
    which is precisely the pair this check exists to catch.
    """
    flags: Dict[str, List[str]] = {}
    tokens = {o.id: set(joined_tokens(o.statement)) for o in objectives}
    for i, first in enumerate(objectives):
        for second in objectives[i + 1:]:
            if not set(first.concept_ids) & set(second.concept_ids):
                continue
            a, b = tokens[first.id], tokens[second.id]
            if not a or not b:
                continue
            overlap = len(a & b) / len(a | b)
            if overlap >= OBJECTIVE_DUPLICATE_OVERLAP:
                flags.setdefault(first.id, []).append(second.id)
                flags.setdefault(second.id, []).append(first.id)
    return flags


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class ObjectiveExtractionResult(BaseModel):
    """One lesson's objectives, with everything needed to argue with them."""

    #: Versioned apart from the L1 content model, so an artifact on disk can
    #: always be read against the rules that produced it.
    schema_version: str = OBJECTIVE_SCHEMA_VERSION
    lesson_id: str
    #: The book's grade. Recorded once here rather than on every objective:
    #: it is a property of the book, and copying it per row would create a
    #: second source of truth for one fact.
    grade: int = 0
    objectives: List[LearningObjective] = Field(default_factory=list)
    #: Evidence records produced while verifying objective citations. They may
    #: repeat ids already in the concept stage's table - the ids are derived
    #: from content, so the two agree by construction.
    evidence: List[Evidence] = Field(default_factory=list)
    admission: ObjectiveAdmissionResult = Field(
        default_factory=ObjectiveAdmissionResult
    )
    #: Admitted but left ungrounded once their citations were checked.
    ungrounded: List[str] = Field(default_factory=list)
    duplicate_flags: Dict[str, List[str]] = Field(default_factory=dict)
    #: Concepts that produced no objective at all, and why - a concept with
    #: nothing to say is a fact about the lesson, not a gap to be filled.
    concepts_without_objectives: List[str] = Field(default_factory=list)
    confidence_breakdowns: Dict[str, ConfidenceBreakdown] = Field(
        default_factory=dict
    )
    verification_notes: Dict[str, List[str]] = Field(default_factory=dict)
    prompt_version: str = ""
    model_id: str = ""


def ground_objective_proposals(
    *,
    unit: EvidenceUnit,
    concepts: Sequence[Concept],
    evidence: Sequence[Evidence],
    admission: ObjectiveAdmissionResult,
    document_id: str,
    prompt_version: str = "",
    model_id: str = "",
    vocabulary_config: Optional[VocabularyConfig] = None,
) -> ObjectiveExtractionResult:
    """Turn admitted objective proposals into grounded objectives, or drop them."""
    result = ObjectiveExtractionResult(
        lesson_id=unit.lesson_id,
        grade=unit.grade,
        admission=admission,
        prompt_version=prompt_version,
        model_id=model_id,
    )
    by_concept = {c.id: c for c in concepts}
    blocks_for = concept_blocks(concepts, evidence)
    pages = block_page_index(unit)
    texts = unit.block_text()
    lesson_texts = list(texts.values())
    evidence_by_id: Dict[str, Evidence] = {}

    for proposal in admission.admitted:
        concept = by_concept[proposal.concept_id]
        outcome = verify_claim(
            document_id=document_id,
            citations=[
                ClaimCitation(
                    block_id=c.block_id, quote=c.quote, asset_id=c.asset_id
                )
                for c in proposal.citations
            ],
            claimed_level=proposal.claimed_evidence_level,
            allowed_block_ids=blocks_for[concept.id],
            block_pages=pages,
            block_text=texts,
            allowed_asset_ids=unit.citable_asset_ids(),
        )
        if not outcome.grounded:
            result.ungrounded.append(proposal.statement)
            continue

        vague = is_vague(proposal.statement) or is_vague(
            proposal.performance_verb
        )
        observable = verb_is_observable(proposal.performance_verb) and not vague
        type_fits = type_fits_concept(
            proposal.objective_type, concept.concept_type
        )

        # The statement is checked against the lesson the same way a concept's
        # wording is, minus this pipeline's own verbs - those come from the
        # lexicon and could never be the book's.
        foreign_words = strip_lexicon(
            check_wording(
                label="",
                definition=proposal.statement,
                lesson_texts=lesson_texts,
                config=vocabulary_config,
            )
        )

        breakdown = compute_objective_confidence(
            outcome,
            proposal,
            concept,
            unit,
            observable=observable,
            type_fits=type_fits,
            foreign_words=foreign_words,
        )
        needs_review, reasons = objective_review_reasons(
            outcome=outcome,
            confidence=breakdown.score,
            concept=concept,
            observable=observable,
            vague=vague,
            type_fits=type_fits,
            objective_type=proposal.objective_type,
            foreign_words=foreign_words,
        )

        objective_id = make_id(
            unit.book_id, "objective", concept.id, proposal.statement
        )
        for item in outcome.evidence:
            evidence_by_id[item.id] = item

        result.objectives.append(
            LearningObjective(
                id=objective_id,
                lesson_id=unit.lesson_id,
                section_id=concept.section_id,
                statement=proposal.statement.strip(),
                objective_type=proposal.objective_type,
                performance_verb=proposal.performance_verb.strip(),
                observable=observable,
                concept_ids=[concept.id],
                evidence_ids=[item.id for item in outcome.evidence],
                evidence_level=outcome.evidence_level,
                confidence=breakdown.score,
                requires_human_review=needs_review,
                review_reasons=reasons,
                out_of_book_vocabulary=foreign_words,
            )
        )
        result.confidence_breakdowns[objective_id] = breakdown
        if outcome.notes:
            result.verification_notes[objective_id] = outcome.notes

    result.evidence = list(evidence_by_id.values())
    result.duplicate_flags = flag_duplicate_objectives(result.objectives)
    for objective in result.objectives:
        for other in result.duplicate_flags.get(objective.id, []):
            objective.review_reasons.append(f"possible duplicate of {other}")
            objective.requires_human_review = True

    covered = {c for o in result.objectives for c in o.concept_ids}
    result.concepts_without_objectives = [
        concept.id for concept in concepts if concept.id not in covered
    ]

    # Padding shows up as a count, so it is checked as one.
    per_concept: Dict[str, int] = {}
    for objective in result.objectives:
        for concept_id in objective.concept_ids:
            per_concept[concept_id] = per_concept.get(concept_id, 0) + 1
    for objective in result.objectives:
        for concept_id in objective.concept_ids:
            if per_concept[concept_id] > MAX_OBJECTIVES_PER_CONCEPT:
                objective.review_reasons.append(
                    f"concept {concept_id} carries {per_concept[concept_id]} "
                    f"objectives, more than {MAX_OBJECTIVES_PER_CONCEPT}"
                )
                objective.requires_human_review = True

    return result


def extract_objectives(
    *,
    unit: EvidenceUnit,
    concepts: Sequence[Concept],
    evidence: Sequence[Evidence],
    client,
    document_id: str,
    template: Optional[PromptTemplate] = None,
) -> Tuple[ObjectiveExtractionResult, ObjectiveResponse, str]:
    """One lesson's concepts in, grounded objectives out.

    Returns the grounded result, the model's raw reply and the prompt that was
    sent - all three, because a run that cannot be inspected afterwards cannot
    be reviewed.
    """
    template = template or load_objective_prompt()
    prompt = build_objective_prompt(unit, concepts, evidence, template)
    request = LLMRequest(prompt=prompt, response_schema=ObjectiveResponse)
    reply = client.complete(request)
    if isinstance(reply, ObjectiveResponse):
        raw = reply
    else:
        payload = reply if isinstance(reply, dict) else reply.model_dump()
        # A model that found nothing still answers with the field, empty. A
        # call that never landed has no field at all, and the difference is
        # the whole point - see :class:`ModelCallFailed`.
        if "objectives" not in payload:
            raise ModelCallFailed(
                "the provider returned no 'objectives' field; the call "
                "failed rather than finding nothing. Nothing was written "
                "for this lesson."
            )
        raw = ObjectiveResponse.model_validate(payload)
    admission = admit_objective_proposals(
        raw, concept_blocks(concepts, evidence)
    )
    result = ground_objective_proposals(
        unit=unit,
        concepts=concepts,
        evidence=evidence,
        admission=admission,
        document_id=document_id,
        prompt_version=template.full_version,
        model_id=getattr(client, "model_id", "unknown"),
    )
    return result, raw, prompt
