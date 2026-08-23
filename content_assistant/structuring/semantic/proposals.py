"""What a model is allowed to say, and how a claim is admitted or thrown out.

These are *proposals*, not entities. A model's reply is untrusted input: it
arrives here, gets checked against the evidence unit it was given, and only
what survives becomes a :class:`~content_assistant.models.content.Concept`.

The admission rule is deliberately blunt. A citation naming a block that was
not in the unit is not scored down - it is dropped, because the model cannot
have read a block it was never shown, and a claim resting only on such
citations is unfounded by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from content_assistant.models.content import ConceptType, EvidenceLevel


class CitationProposal(BaseModel):
    """A model's claim that ``quote`` appears in ``block_id``.

    Both halves are checked later and independently: the id against the
    evidence unit, the quote against that block's actual text.
    """

    block_id: str
    quote: str = ""
    #: Set when the claim rests on a picture. Text is still preferred.
    asset_id: Optional[str] = None


class ConceptProposal(BaseModel):
    """One concept as proposed. Nothing here is trusted yet."""

    label: str
    definition: str = ""
    concept_type: ConceptType = "conceptual"
    citations: List[CitationProposal] = Field(default_factory=list)
    claimed_evidence_level: EvidenceLevel = "inferred"
    #: The model's own confidence. Recorded for comparison, never used on its
    #: own - a model states this with the same assurance whether it is right or
    #: wrong, so it can only ever be a small input to the computed score.
    model_confidence: Optional[float] = None
    #: Set by the model when a concept is carried by a picture, not by text.
    visual_only: bool = False


class ConceptResponse(BaseModel):
    """The whole reply. This is the schema handed to the provider."""

    concepts: List[ConceptProposal] = Field(default_factory=list)
    #: Optional free-text note from the model, for the review file only.
    notes: str = ""


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------

REJECT_NO_CITATION = "no_citation"
REJECT_ALL_CITATIONS_FOREIGN = "all_citations_outside_evidence_unit"
REJECT_EMPTY_LABEL = "empty_label"


class RejectedProposal(BaseModel):
    label: str
    reason: str
    detail: str = ""


class AdmissionResult(BaseModel):
    """Proposals that may proceed, and the ones that may not, with reasons."""

    admitted: List[ConceptProposal] = Field(default_factory=list)
    rejected: List[RejectedProposal] = Field(default_factory=list)
    #: Citations dropped from otherwise-admitted proposals.
    dropped_citations: List[str] = Field(default_factory=list)


def admit_proposals(
    response: ConceptResponse, allowed_block_ids: Sequence[str]
) -> AdmissionResult:
    """Keep only what the evidence unit can support.

    Three things end a proposal here, all of them structural rather than
    editorial: an empty label, no citations at all, or citations that every one
    of them names a block outside the unit. Whether the *quotes* are real is a
    separate question, answered by the verifier.
    """
    allowed = set(allowed_block_ids)
    result = AdmissionResult()

    for proposal in response.concepts:
        if not proposal.label.strip():
            result.rejected.append(
                RejectedProposal(label=proposal.label, reason=REJECT_EMPTY_LABEL)
            )
            continue
        if not proposal.citations:
            result.rejected.append(
                RejectedProposal(
                    label=proposal.label,
                    reason=REJECT_NO_CITATION,
                    detail="a concept with no citation cannot be grounded",
                )
            )
            continue

        kept = [c for c in proposal.citations if c.block_id in allowed]
        foreign = [c.block_id for c in proposal.citations if c.block_id not in allowed]
        result.dropped_citations.extend(foreign)

        if not kept:
            result.rejected.append(
                RejectedProposal(
                    label=proposal.label,
                    reason=REJECT_ALL_CITATIONS_FOREIGN,
                    detail=", ".join(sorted(set(foreign))),
                )
            )
            continue

        result.admitted.append(proposal.model_copy(update={"citations": kept}))

    return result
