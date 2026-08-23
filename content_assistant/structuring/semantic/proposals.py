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

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from content_assistant.models.content import (
    ConceptType,
    EvidenceLevel,
    ObjectiveType,
)


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


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------


class ObjectiveProposal(BaseModel):
    """One learning objective as proposed. Nothing here is trusted yet.

    An objective names the concept it serves by id, the same way a citation
    names its block by id: the model is handed the ids and may only give them
    back. Anything else is not scored down, it is dropped.
    """

    concept_id: str
    statement: str
    objective_type: ObjectiveType = "identify"
    #: The action the student performs. Checked against the closed lexicon in
    #: :mod:`content_assistant.models.objective`, never taken on trust.
    performance_verb: str = ""
    citations: List[CitationProposal] = Field(default_factory=list)
    claimed_evidence_level: EvidenceLevel = "inferred"
    #: The model's own confidence. Recorded, capped, never decisive.
    model_confidence: Optional[float] = None


class ObjectiveResponse(BaseModel):
    """The whole reply. This is the schema handed to the provider."""

    objectives: List[ObjectiveProposal] = Field(default_factory=list)
    notes: str = ""


REJECT_EMPTY_STATEMENT = "empty_statement"
REJECT_UNKNOWN_CONCEPT = "concept_not_in_this_lesson"
REJECT_CITATIONS_OUTSIDE_CONCEPT = "citations_outside_concept_evidence"
#: The concept exists but stands on nothing, so there is no text an objective
#: about it could ever cite. Reported separately from a bad citation because
#: the fault is in the concept, and pointing a reviewer at the objective would
#: send them to the wrong place.
REJECT_CONCEPT_WITHOUT_EVIDENCE = "concept_has_no_evidence"


class ObjectiveAdmissionResult(BaseModel):
    """Objectives that may proceed, and the ones that may not, with reasons."""

    admitted: List[ObjectiveProposal] = Field(default_factory=list)
    rejected: List[RejectedProposal] = Field(default_factory=list)
    dropped_citations: List[str] = Field(default_factory=list)


def admit_objective_proposals(
    response: ObjectiveResponse,
    concept_blocks: Mapping[str, Set[str]],
) -> ObjectiveAdmissionResult:
    """Keep only the objectives their own concept's evidence can support.

    ``concept_blocks`` maps each concept id to the blocks that concept is
    already grounded in. That mapping is the whole admission rule, and it is
    what stops an objective from quietly becoming a new concept.

    An objective is a statement *about* a concept. If it can only be justified
    by a sentence the concept does not rest on, then it is asserting something
    the concept does not, which is a new claim wearing an objective's clothes.
    Restricting citations to the concept's own blocks makes that structural
    instead of editorial: there is no wording a model can choose that gets
    around it, and no judgement call for anyone to disagree with later.

    The consequence is deliberate and worth stating plainly: an objective that
    needs other evidence is not repaired here, it is rejected, and the right
    fix is a concept that covers that evidence.
    """
    result = ObjectiveAdmissionResult()

    for proposal in response.objectives:
        if not proposal.statement.strip():
            result.rejected.append(
                RejectedProposal(
                    label=proposal.statement, reason=REJECT_EMPTY_STATEMENT
                )
            )
            continue

        allowed = concept_blocks.get(proposal.concept_id)
        if allowed is None:
            result.rejected.append(
                RejectedProposal(
                    label=proposal.statement,
                    reason=REJECT_UNKNOWN_CONCEPT,
                    detail=(
                        f"{proposal.concept_id!r} is not a grounded concept of "
                        "this lesson"
                    ),
                )
            )
            continue

        if not allowed:
            result.rejected.append(
                RejectedProposal(
                    label=proposal.statement,
                    reason=REJECT_CONCEPT_WITHOUT_EVIDENCE,
                    detail=(
                        f"concept {proposal.concept_id} rests on no block, so "
                        "no objective about it can be grounded"
                    ),
                )
            )
            continue

        if not proposal.citations:
            result.rejected.append(
                RejectedProposal(
                    label=proposal.statement,
                    reason=REJECT_NO_CITATION,
                    detail="an objective with no citation cannot be grounded",
                )
            )
            continue

        kept = [c for c in proposal.citations if c.block_id in allowed]
        foreign = [
            c.block_id for c in proposal.citations if c.block_id not in allowed
        ]
        result.dropped_citations.extend(foreign)

        if not kept:
            result.rejected.append(
                RejectedProposal(
                    label=proposal.statement,
                    reason=REJECT_CITATIONS_OUTSIDE_CONCEPT,
                    detail=", ".join(sorted(set(foreign))),
                )
            )
            continue

        result.admitted.append(proposal.model_copy(update={"citations": kept}))

    return result
