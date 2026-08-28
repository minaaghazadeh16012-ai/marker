"""What every layer of the content schema needs: identity, provenance, review.

Three layers sit on top of this one - content knowledge and learning intent in
:mod:`content_assistant.models.content`, learning experience in
:mod:`content_assistant.models.learning` - and all three need the same
identity function and the same answer to "who said this, and who checked it?".
Putting those here is what lets the experience layer exist as its own module
without the two importing each other in a circle.

Nothing in this file knows about lessons, concepts or questions. If a type here
starts needing to, it belongs one layer up.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from content_assistant.text.persian import normalize

#: Bumped whenever a field is added, removed, or changes meaning. 1.0.0 held
#: lessons, sections, concepts and objectives; 1.1.0 added per-entity
#: provenance, the review lifecycle, and the learning-experience layer; 1.2.0
#: adds what an engine needs to *run* that layer - how an item is graded, which
#: template draws it, where an activity sits in a sequence - and widens the
#: response-form vocabulary to the forms a first-grade book actually asks for.
#: Every addition since 1.0.0 is optional with a default, so an artifact
#: written against any earlier minor still loads - see
#: :mod:`content_assistant.package.migrate`, which is the only place allowed to
#: decide whether a stored version can be read.
SCHEMA_VERSION = "1.2.0"

#: The public name for the same number. Consumers outside this package ask for
#: "the content schema version"; there is exactly one, and this is it.
CONTENT_SCHEMA_VERSION = SCHEMA_VERSION

BBox = List[float]

EvidenceLevel = Literal["explicit", "inferred", "needs_visual_review"]

#: How an objective can be evidenced or assessed.
ContentType = Literal[
    "text", "image", "audio", "interactive", "handwriting", "physical"
]

#: Three bands rather than a number, and the same three everywhere they are
#: used. A textbook states no difficulty at all, so any decimal here would be
#: invented precision; three bands are what an author can actually judge and
#: what a scheduler actually branches on.
DifficultyBand = Literal["intro", "core", "stretch"]

#: How a record came to exist. The distinction is load-bearing rather than
#: descriptive: ``model_proposed`` is the only value the semantic stages may
#: write, and ``human`` is the only one that exempts a record from needing a
#: quotation - because a person can be asked why, and a model cannot.
ExtractionMethod = Literal["deterministic", "model_proposed", "human"]

#: What a reviewer decided. ``pending`` is not a judgement, it is the absence
#: of one, and it stays that way until a person writes something else here.
ReviewStatus = Literal["pending", "accepted", "rejected", "needs_changes"]

REVIEW_DECIDED = ("accepted", "rejected", "needs_changes")


# ---------------------------------------------------------------------------
# deterministic identity
# ---------------------------------------------------------------------------

_ID_CLEAN = re.compile(r"\s+")


def id_slug(text: str) -> str:
    """Canonical form of a label for hashing.

    Persian normalization runs first so that two spellings of the same title -
    Arabic vs Persian yeh, a stray diacritic, doubled spaces - hash to one id
    instead of creating a duplicate entity.
    """
    return _ID_CLEAN.sub(" ", normalize(text or "")).strip().lower()


def make_id(book_id: str, kind: str, *parts: object) -> str:
    """Build a stable id: ``{book}:{kind}:{digest}``.

    The digest is a hash of the canonicalised parts, so the same book and the
    same content always yield the same id - across machines, across runs, and
    regardless of the order entities happened to be produced in. Ids are never
    counters for exactly that reason.
    """
    payload = "|".join(id_slug(str(p)) for p in parts)
    digest = hashlib.sha256(f"{book_id}|{kind}|{payload}".encode("utf-8"))
    return f"{book_id}:{kind}:{digest.hexdigest()[:10]}"


def ordinal_id(book_id: str, kind: str, *numbers: int) -> str:
    """Readable id for entities whose position *is* their identity.

    A lesson is the fourth lesson of this book; that is more stable and far
    more legible than a hash of its title, which changes the moment a title is
    corrected.
    """
    tail = ".".join(f"{n:02d}" for n in numbers)
    return f"{book_id}:{kind}:{tail}"


# ---------------------------------------------------------------------------
# provenance and review
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Who produced this record, with what, and when.

    Every entity carries its own, rather than inheriting the run's: one package
    is assembled from many runs, and after a merge "which model wrote this?"
    has no answer at the document level. It is optional only so that a 1.0.0
    artifact still loads; anything this pipeline writes sets it.

    ``extraction_method`` is the field the validator keys off, and the three
    values are not interchangeable. ``deterministic`` means code derived it
    from the book. ``model_proposed`` means a model said it and the verifier
    let it through. ``human`` means a person is accountable for it - which is
    the only way a claim can enter without a quotation behind it.
    """

    extraction_method: ExtractionMethod = "model_proposed"
    #: The stage that produced it: ``"concepts"``, ``"objectives"``, ... Free
    #: text, because stages are added over time and a closed list here would
    #: have to be edited before a new stage could record anything.
    stage: str = ""
    model_id: Optional[str] = None
    prompt_version: Optional[str] = None
    run_id: Optional[str] = None
    #: ISO-8601. A string rather than a datetime so a stored artifact reads
    #: back byte-identical instead of through a parser's idea of a timezone.
    generated_at: Optional[str] = None
    #: Set when ``extraction_method`` is ``"human"``: who authored it.
    authored_by: Optional[str] = None


class Attributed(BaseModel):
    """Where a record came from, and what a reviewer decided about it.

    Two questions that look like one and are not.
    ``requires_human_review`` asks whether a person *has to look*; the pipeline
    sets it, and ``review_reasons`` says why. ``review_status`` records what
    the person then *decided*, and nothing in the pipeline may write it. An
    entity can perfectly well be ``accepted`` while still carrying the reasons
    that sent it to review - that is the audit trail, not a contradiction.
    """

    provenance: Optional[Provenance] = None
    requires_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    review_status: ReviewStatus = "pending"
    review_notes: str = ""
    reviewed_by: Optional[str] = None
    #: ISO-8601, set at the same time as ``reviewed_by``. A decision with no
    #: author and no date cannot be audited, and the validator says so.
    reviewed_at: Optional[str] = None
