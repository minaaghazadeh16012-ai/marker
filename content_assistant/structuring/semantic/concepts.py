"""Concept extraction: propose with a model, admit and ground deterministically.

The model's only job is to notice what a lesson teaches and point at the words
that say so. Everything after that - whether the citation exists, whether the
quote is real, how much to trust the result, whether a person must look at it -
is decided here, by code, from facts.

Confidence in particular is *computed*, never taken. A model reports high
confidence in the same tone whether it is right or wrong, so its own number is
capped at a small contribution and the weight sits on things that can be
checked: did the quotation turn up in the block it was attributed to, how many
independent citations survived, and how much text the lesson had to begin with.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from content_assistant.models.content import Concept, Evidence, make_id
from content_assistant.structuring.evidence import EvidenceUnit
from content_assistant.structuring.semantic.llm import LLMRequest
from content_assistant.structuring.semantic.proposals import (
    AdmissionResult,
    ConceptProposal,
    ConceptResponse,
    admit_proposals,
)
from content_assistant.structuring.verify import (
    ClaimCitation,
    VerificationOutcome,
    block_page_index,
    verify_claim,
)
from content_assistant.models.content import id_slug

PROMPT_DIR = Path(__file__).parent / "prompts"

#: Two concepts whose labels overlap this much are probably the same idea.
#: They are flagged, never merged - merging is a later, reviewable step.
DUPLICATE_TOKEN_OVERLAP = 0.8

#: Review routing thresholds.
AUTO_ACCEPT_MIN = 0.85
REVIEW_QUEUE_MIN = 0.60


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class PromptTemplate(BaseModel):
    name: str
    version: str
    text: str

    @property
    def full_version(self) -> str:
        return f"{self.name}@{self.version}"


def load_prompt(name: str = "concept_v1", language: str = "fa") -> PromptTemplate:
    """Load a versioned prompt and fingerprint it.

    The version is the file's own content hash, so an edited prompt is a
    different version automatically. That is what keeps the cache key honest -
    nobody has to remember to bump a number.
    """
    path = PROMPT_DIR / language / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return PromptTemplate(name=name, version=digest, text=text)


def render_evidence_blocks(unit: EvidenceUnit) -> str:
    """Lay the unit out so every citable id is impossible to miss."""
    lines: List[str] = []
    for section in unit.sections:
        lines.append(f"### بخش {section.order}: {section.title}")
        if not section.blocks:
            lines.append("(بدون متن)")
        for block in section.blocks:
            page = block.printed_page if block.printed_page is not None else "?"
            lines.append(f"- `{block.block_id}` (صفحه {page}): {block.text}")
        if section.images:
            ids = ", ".join(f"`{image.asset_id}`" for image in section.images)
            lines.append(f"- تصاویر این بخش: {ids}")
        lines.append("")
    return "\n".join(lines).strip()


def material_note(unit: EvidenceUnit) -> str:
    """Tell the model, plainly, how thin the lesson is.

    A lesson carrying a few hundred characters against thirty pictures cannot
    support many grounded concepts, and saying so up front is more effective
    than hoping the model notices.
    """
    profile = unit.material_profile
    note = (
        f"این درس {profile.text_chars} نویسه متن و {profile.images} تصویر دارد "
        f"(چگالی متن: {profile.text_density})."
    )
    if profile.text_density == "low":
        note += (
            " متن این درس کم است؛ انتظار نداشته باش مفاهیم زیادی از متن "
            "پشتیبانی شوند. آنچه را متن پشتیبانی نمی‌کند ننویس."
        )
    return note


def build_prompt(unit: EvidenceUnit, template: PromptTemplate) -> str:
    replacements = {
        "{{LESSON_NUMBER}}": str(unit.lesson_number),
        "{{LESSON_TITLE}}": unit.lesson_title,
        "{{GRADE}}": str(unit.grade),
        "{{SUBJECT}}": unit.subject,
        "{{PAGE_START}}": str(unit.printed_page_start or "?"),
        "{{PAGE_END}}": str(unit.printed_page_end or "?"),
        "{{MATERIAL_NOTE}}": material_note(unit),
        "{{EVIDENCE_BLOCKS}}": render_evidence_blocks(unit),
    }
    text = template.text
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


# ---------------------------------------------------------------------------
# confidence + review routing
# ---------------------------------------------------------------------------


class ConfidenceBreakdown(BaseModel):
    """Every term that produced a score, so the number can be argued with."""

    score: float = 0.0
    components: Dict[str, float] = Field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.components[name] = round(value, 4)
        self.score = round(min(1.0, max(0.0, self.score + value)), 4)


def compute_confidence(
    outcome: VerificationOutcome,
    proposal: ConceptProposal,
    unit: EvidenceUnit,
) -> ConfidenceBreakdown:
    """Score a concept from what could be checked, not from what was claimed."""
    breakdown = ConfidenceBreakdown()
    verified = [e for e in outcome.evidence if e.quote_verified]

    if verified:
        # Keyed on *how* the quote was found, not on the numeric score: a
        # token-overlap match can score 1.0 while still being a paraphrase, and
        # a paraphrase is weaker evidence than the sentence itself.
        methods = {e.match_method for e in verified}
        if methods & {"exact", "normalized"}:
            breakdown.add("quote_verified_exact", 0.55)
        else:
            breakdown.add("quote_verified_fuzzy", 0.35)
    else:
        breakdown.add("no_verified_quote", 0.0)

    if len(verified) >= 2:
        breakdown.add("multiple_independent_citations", 0.10)

    if outcome.evidence_level == "explicit" and verified:
        breakdown.add("explicit_and_grounded", 0.10)

    if unit.material_profile.text_density == "low":
        breakdown.add("thin_lesson_penalty", -0.10)

    if proposal.model_confidence is not None:
        # Capped hard: a model's self-assessment is a hint, not a measurement.
        breakdown.add(
            "model_self_report_capped",
            min(0.10, max(0.0, proposal.model_confidence) * 0.10),
        )

    if outcome.evidence_level == "needs_visual_review":
        breakdown.score = round(min(breakdown.score, 0.50), 4)
        breakdown.components["visual_cap"] = 0.50

    return breakdown


def review_reasons(
    outcome: VerificationOutcome, confidence: float, proposal: ConceptProposal
) -> Tuple[bool, List[str]]:
    """Decide whether a person has to look at this, and say why."""
    reasons: List[str] = []
    if outcome.evidence_level == "needs_visual_review" or proposal.visual_only:
        reasons.append("rests on an image; a person must look at the page")
    if outcome.demoted:
        reasons.append("claim was demoted during verification")
    if not any(e.quote_verified for e in outcome.evidence):
        reasons.append("no cited quotation could be found in the book")
    if confidence < REVIEW_QUEUE_MIN:
        reasons.append(f"confidence {confidence:.2f} below {REVIEW_QUEUE_MIN}")
    elif confidence < AUTO_ACCEPT_MIN:
        reasons.append(f"confidence {confidence:.2f} below auto-accept")
    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# duplicate flagging
# ---------------------------------------------------------------------------


def flag_duplicates(concepts: Sequence[Concept]) -> Dict[str, List[str]]:
    """Report concepts that look like the same idea. Flag only, never merge.

    Merging is a separate, reviewable step: two labels can overlap heavily and
    still be different ideas, and collapsing them here would destroy evidence
    without anyone seeing it happen.
    """
    flags: Dict[str, List[str]] = {}
    tokens = {c.id: set(id_slug(c.label).split()) for c in concepts}
    for i, first in enumerate(concepts):
        for second in concepts[i + 1 :]:
            a, b = tokens[first.id], tokens[second.id]
            if not a or not b:
                continue
            overlap = len(a & b) / len(a | b)
            if overlap >= DUPLICATE_TOKEN_OVERLAP:
                flags.setdefault(first.id, []).append(second.id)
                flags.setdefault(second.id, []).append(first.id)
    return flags


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class ConceptExtractionResult(BaseModel):
    lesson_id: str
    concepts: List[Concept] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    admission: AdmissionResult = Field(default_factory=AdmissionResult)
    #: Proposals admitted but left ungrounded once their citations were checked.
    ungrounded: List[str] = Field(default_factory=list)
    duplicate_flags: Dict[str, List[str]] = Field(default_factory=dict)
    confidence_breakdowns: Dict[str, ConfidenceBreakdown] = Field(
        default_factory=dict
    )
    verification_notes: Dict[str, List[str]] = Field(default_factory=dict)
    prompt_version: str = ""
    model_id: str = ""


def ground_proposals(
    *,
    unit: EvidenceUnit,
    admission: AdmissionResult,
    document_id: str,
    prompt_version: str = "",
    model_id: str = "",
) -> ConceptExtractionResult:
    """Turn admitted proposals into grounded concepts, or drop them."""
    result = ConceptExtractionResult(
        lesson_id=unit.lesson_id,
        admission=admission,
        prompt_version=prompt_version,
        model_id=model_id,
    )
    allowed_blocks = unit.citable_block_ids()
    allowed_assets = unit.citable_asset_ids()
    pages = block_page_index(unit)
    texts = unit.block_text()
    evidence_by_id: Dict[str, Evidence] = {}

    for proposal in admission.admitted:
        outcome = verify_claim(
            document_id=document_id,
            citations=[
                ClaimCitation(
                    block_id=c.block_id, quote=c.quote, asset_id=c.asset_id
                )
                for c in proposal.citations
            ],
            claimed_level=proposal.claimed_evidence_level,
            allowed_block_ids=allowed_blocks,
            block_pages=pages,
            block_text=texts,
            allowed_asset_ids=allowed_assets,
        )
        if not outcome.grounded:
            result.ungrounded.append(proposal.label)
            continue

        level = outcome.evidence_level
        if proposal.visual_only and level == "explicit":
            level = "needs_visual_review"
        outcome = outcome.model_copy(update={"evidence_level": level})

        breakdown = compute_confidence(outcome, proposal, unit)
        needs_review, reasons = review_reasons(outcome, breakdown.score, proposal)

        concept_id = make_id(
            unit.book_id,
            "concept",
            proposal.label,
            outcome.evidence[0].block_id,
        )
        for item in outcome.evidence:
            evidence_by_id[item.id] = item

        result.concepts.append(
            Concept(
                id=concept_id,
                lesson_id=unit.lesson_id,
                section_id=None,
                label=proposal.label.strip(),
                definition=proposal.definition.strip(),
                concept_type=proposal.concept_type,
                evidence_ids=[item.id for item in outcome.evidence],
                evidence_level=outcome.evidence_level,
                confidence=breakdown.score,
                requires_human_review=needs_review,
                review_reasons=reasons,
            )
        )
        result.confidence_breakdowns[concept_id] = breakdown
        if outcome.notes:
            result.verification_notes[concept_id] = outcome.notes

    result.evidence = list(evidence_by_id.values())
    result.duplicate_flags = flag_duplicates(result.concepts)
    for concept in result.concepts:
        for other in result.duplicate_flags.get(concept.id, []):
            concept.review_reasons.append(f"possible duplicate of {other}")
            concept.requires_human_review = True
    return result


def extract_concepts(
    *,
    unit: EvidenceUnit,
    client,
    document_id: str,
    template: Optional[PromptTemplate] = None,
    image_paths: Optional[Sequence[str]] = None,
) -> Tuple[ConceptExtractionResult, ConceptResponse, str]:
    """One lesson in, grounded concepts out.

    Returns the grounded result, the model's raw reply, and the prompt that was
    sent - all three, because a run that cannot be inspected afterwards cannot
    be reviewed.
    """
    template = template or load_prompt()
    prompt = build_prompt(unit, template)
    request = LLMRequest(
        prompt=prompt,
        response_schema=ConceptResponse,
        image_paths=list(image_paths or []),
    )
    raw = client.complete(request)
    if not isinstance(raw, ConceptResponse):
        raw = ConceptResponse.model_validate(
            raw if isinstance(raw, dict) else raw.model_dump()
        )
    admission = admit_proposals(raw, sorted(unit.citable_block_ids()))
    result = ground_proposals(
        unit=unit,
        admission=admission,
        document_id=document_id,
        prompt_version=template.full_version,
        model_id=getattr(client, "model_id", "unknown"),
    )
    return result, raw, prompt
