"""Grounding: checking a model's claims against the book, mechanically.

The model proposes; this module disposes. Two things are checked, and neither
asks the model's opinion:

**Did it cite something it was given?** An Evidence Unit hands over a closed set
of block ids. A citation outside that set is not a mistake to be scored down -
it is fabrication, and the claim is dropped.

**Does the quotation actually appear there?** Exact match first, then a
normalized match, then a token-overlap match to survive the text layer's known
defects (missing ZWNJ, a reversed lam-alef ligature, stray diacritics). If the
quote cannot be found in the cited block, the claim is not rejected outright -
it is *demoted* from ``explicit`` to ``inferred`` and flagged, because a real
observation with a sloppy quotation is still worth a reviewer's attention.

Demotion, never silent promotion: nothing in this module can raise a claim's
evidence level.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from content_assistant.models.content import Evidence, EvidenceLevel
from content_assistant.text.persian import normalize

#: Below this token overlap a quotation is not considered found.
DEFAULT_FUZZY_MIN = 0.75

_WORD_RE = re.compile(r"[^\s]+")


class QuoteMatch(BaseModel):
    matched: bool
    score: float = 0.0
    method: str = "none"
    char_start: Optional[int] = None
    char_end: Optional[int] = None


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(normalize(text or ""))


def match_quote(
    quote: str, block_text: str, fuzzy_min: float = DEFAULT_FUZZY_MIN
) -> QuoteMatch:
    """Find ``quote`` inside ``block_text``, tolerating the text layer's faults.

    Three passes, strongest first, so a clean quotation is reported as clean and
    only a damaged one falls back to a looser test.
    """
    if not quote.strip() or not block_text.strip():
        return QuoteMatch(matched=False)

    index = block_text.find(quote)
    if index >= 0:
        return QuoteMatch(
            matched=True,
            score=1.0,
            method="exact",
            char_start=index,
            char_end=index + len(quote),
        )

    normalized_block = normalize(block_text)
    normalized_quote = normalize(quote)
    index = normalized_block.find(normalized_quote)
    if index >= 0:
        return QuoteMatch(
            matched=True,
            score=0.95,
            method="normalized",
            char_start=index,
            char_end=index + len(normalized_quote),
        )

    quote_tokens = _tokens(quote)
    block_tokens = set(_tokens(block_text))
    if not quote_tokens:
        return QuoteMatch(matched=False)
    overlap = sum(1 for token in quote_tokens if token in block_tokens)
    score = overlap / len(quote_tokens)
    if score >= fuzzy_min:
        return QuoteMatch(matched=True, score=round(score, 4), method="token_overlap")
    return QuoteMatch(matched=False, score=round(score, 4), method="token_overlap")


class ClaimCitation(BaseModel):
    """One (block, quote) pair a model offered in support of a claim."""

    block_id: str
    quote: str = ""
    asset_id: Optional[str] = None


class VerificationOutcome(BaseModel):
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_level: EvidenceLevel = "inferred"
    rejected_citations: List[str] = Field(default_factory=list)
    demoted: bool = False
    notes: List[str] = Field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """A claim survives only if at least one citation stood up."""
        return bool(self.evidence)


def verify_claim(
    *,
    document_id: str,
    citations: Sequence[ClaimCitation],
    claimed_level: EvidenceLevel,
    allowed_block_ids: set,
    block_pages: Dict[str, Tuple[int, Optional[int]]],
    block_text: Dict[str, str],
    allowed_asset_ids: Optional[set] = None,
    fuzzy_min: float = DEFAULT_FUZZY_MIN,
) -> VerificationOutcome:
    """Turn a model's citations into evidence records, or throw them out.

    ``claimed_level`` is what the model asserted. The returned level is what the
    book supports, which can only be the same or weaker.
    """
    outcome = VerificationOutcome(evidence_level=claimed_level)
    allowed_asset_ids = allowed_asset_ids or set()
    any_verified = False
    visual_only = True

    for citation in citations:
        if citation.block_id not in allowed_block_ids:
            outcome.rejected_citations.append(citation.block_id)
            outcome.notes.append(
                f"citation {citation.block_id!r} was not in the evidence unit "
                "and was discarded"
            )
            continue

        pdf_page, printed_page = block_pages.get(citation.block_id, (0, None))
        text = block_text.get(citation.block_id, "")
        match = match_quote(citation.quote, text, fuzzy_min)
        if citation.asset_id and citation.asset_id not in allowed_asset_ids:
            outcome.notes.append(
                f"asset {citation.asset_id!r} was not in the evidence unit"
            )
            citation = citation.model_copy(update={"asset_id": None})
        if citation.asset_id is None:
            visual_only = False

        outcome.evidence.append(
            Evidence(
                id=Evidence.build_id(document_id, citation.block_id, citation.quote),
                document_id=document_id,
                block_id=citation.block_id,
                pdf_page=pdf_page,
                printed_page=printed_page,
                quote=citation.quote,
                char_start=match.char_start,
                char_end=match.char_end,
                quote_verified=match.matched,
                match_score=match.score,
                match_method=match.method,
                asset_id=citation.asset_id,
            )
        )
        any_verified = any_verified or match.matched

    if claimed_level == "explicit" and not any_verified:
        outcome.evidence_level = "inferred"
        outcome.demoted = True
        outcome.notes.append(
            "demoted from 'explicit' to 'inferred': no cited quotation could be "
            "found in the block it was attributed to"
        )
    elif claimed_level == "explicit" and visual_only and outcome.evidence:
        outcome.evidence_level = "needs_visual_review"
        outcome.demoted = True
        outcome.notes.append(
            "claim rests on images only; a person has to look at the page"
        )

    return outcome


def block_page_index(unit) -> Dict[str, Tuple[int, Optional[int]]]:
    """``block_id -> (pdf_page, printed_page)`` for an Evidence Unit."""
    return {
        block.block_id: (block.pdf_page, block.printed_page)
        for section in unit.sections
        for block in section.blocks
    }
