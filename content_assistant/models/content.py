"""The L1 content model: what a textbook *teaches*, and where each claim came from.

Three rules shape every type here.

**Nothing exists without evidence.** Every educational entity carries at least
one :class:`EvidenceRef` pointing at a block in the L0 artifact. An entity with
no evidence is not "low confidence" - it is invalid, and the validation layer
rejects it. This is the mechanical form of "the book is the source of truth".

**Evidence is a table, not a field.** One sentence in the book can support a
concept, an objective and a misconception at once. Storing the quote three
times invites three copies that drift apart, so quotes live in
:class:`Evidence` and entities reference them by id.

**Ids are derived, not allocated.** Re-running the pipeline on an unchanged
book must produce byte-identical ids, otherwise every run looks like a total
rewrite and no one can review a diff. See :func:`make_id`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from content_assistant.text.persian import normalize

SCHEMA_VERSION = "1.0.0"

BBox = List[float]

EvidenceLevel = Literal["explicit", "inferred", "needs_visual_review"]
TitleSource = Literal["toc", "lesson_opening_page", "section_header", "fallback"]
BoundaryMethod = Literal["section_header", "page_fallback", "whole_lesson"]
TextDensity = Literal["low", "medium", "high"]

#: What kind of thinking a concept asks for. Borrowed as an idea only - no
#: external taxonomy data is imported into this project.
ConceptType = Literal[
    "conceptual", "procedural", "representational", "language", "meta"
]

#: How an objective can be evidenced or assessed.
ContentType = Literal[
    "text", "image", "audio", "interactive", "handwriting", "physical"
]

#: The closed relation vocabulary. A model may not invent a type outside this
#: list, and the validator rejects anything that tries. Structural links
#: (part_of, follows, assesses, teaches) are deliberately absent: they are
#: derivable from the entity fields, and storing them too would create a second
#: source of truth for the same fact.
RelationType = Literal[
    "prerequisite_of",
    "related_to",
    "elaborates",
    "example_of",
    "commonly_misunderstood_as",
]

RELATION_TYPES = (
    "prerequisite_of",
    "related_to",
    "elaborates",
    "example_of",
    "commonly_misunderstood_as",
)


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
# provenance
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single grounded citation into the L0 artifact.

    ``quote_verified`` is set by the verifier, never by the model that produced
    the claim: whether a quotation really appears in a block is a fact about
    the text, and asking the author of the claim to grade it defeats the point.
    """

    id: str
    document_id: str
    block_id: str
    pdf_page: int
    printed_page: Optional[int] = None
    pdf_page_index: Optional[int] = None
    bbox: Optional[BBox] = None
    quote: str = ""
    #: Character span of the quote inside the block's normalized text.
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    quote_verified: bool = False
    match_score: Optional[float] = None
    match_method: Optional[str] = None
    #: Set when the claim rests on a picture rather than on text.
    asset_id: Optional[str] = None
    block_source: Optional[str] = None

    @staticmethod
    def build_id(document_id: str, block_id: str, quote: str) -> str:
        return make_id(document_id, "ev", block_id, quote)


class Grounded(BaseModel):
    """Mixin for everything a model proposes and a human may have to check."""

    evidence_ids: List[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel = "inferred"
    #: Computed by the pipeline from verification facts - see the review layer.
    #: A model's own stated confidence is at most one small input to this.
    confidence: float = 0.0
    requires_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# structure (deterministic - produced without any model)
# ---------------------------------------------------------------------------


class PageRange(BaseModel):
    printed_start: Optional[int] = None
    printed_end: Optional[int] = None
    pdf_start: int
    pdf_end: int

    def contains_pdf_page(self, pdf_page: int) -> bool:
        return self.pdf_start <= pdf_page <= self.pdf_end


class MaterialProfile(BaseModel):
    """How much a lesson actually gives you to work with.

    Measured, not estimated. It decides whether a lesson can be read from text
    alone or needs its pages looked at, and it is the honest answer to "why did
    this lesson yield so little?" - some lessons in a first-grade book carry a
    few hundred characters of text against thirty pictures.
    """

    text_chars: int = 0
    text_blocks: int = 0
    recovered_blocks: int = 0
    images: int = 0
    section_headers: int = 0
    pages: int = 0
    text_density: TextDensity = "low"


class Lesson(BaseModel):
    id: str
    book_id: str
    grade: int
    subject: str
    lesson_number: int
    title: str
    title_source: TitleSource
    title_is_approximate: bool = False
    #: The other candidate title, kept so a reviewer can see the disagreement.
    title_alternatives: Dict[str, str] = Field(default_factory=dict)
    page_range: PageRange
    block_ids: List[str] = Field(default_factory=list)
    asset_ids: List[str] = Field(default_factory=list)
    material_profile: MaterialProfile = Field(default_factory=MaterialProfile)
    evidence_ids: List[str] = Field(default_factory=list)


class Section(BaseModel):
    id: str
    lesson_id: str
    order: int
    title: str
    boundary_method: BoundaryMethod
    source_block_id: Optional[str] = None
    page_range: PageRange
    block_ids: List[str] = Field(default_factory=list)
    asset_ids: List[str] = Field(default_factory=list)
    text_chars: int = 0


# ---------------------------------------------------------------------------
# semantics (model-proposed, verifier-grounded - not produced in this phase)
# ---------------------------------------------------------------------------


class Concept(Grounded):
    id: str
    lesson_id: str
    section_id: Optional[str] = None
    label: str
    definition: str = ""
    concept_type: ConceptType = "conceptual"
    aliases: List[str] = Field(default_factory=list)
    #: Ids this concept absorbed during cross-referencing, so a merge is
    #: reversible and reviewable rather than silent.
    merged_from: List[str] = Field(default_factory=list)


class LearningObjective(Grounded):
    id: str
    lesson_id: str
    section_id: Optional[str] = None
    statement: str
    #: The action the student performs - what makes an objective assessable.
    performance_verb: str = ""
    #: False marks an objective that cannot be observed, so it is flagged
    #: rather than quietly shipped as if it could be tested.
    observable: bool = True
    concept_ids: List[str] = Field(default_factory=list)
    skill_id: Optional[str] = None
    content_types: List[ContentType] = Field(default_factory=list)


class Skill(Grounded):
    id: str
    label: str
    concept_ids: List[str] = Field(default_factory=list)
    lesson_ids: List[str] = Field(default_factory=list)


class Misconception(Grounded):
    """A likely student error.

    Held to a stricter standard than anything else in the model: a first-grade
    textbook almost never states a misconception outright, so ``explicit`` is
    only permitted when a verified quotation backs it, and human review is on
    by default. A run that produces many confident misconceptions is a symptom,
    not an achievement.
    """

    id: str
    concept_id: str
    statement: str
    correction: str = ""
    requires_human_review: bool = True


class Relation(Grounded):
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: Literal["hard", "soft"] = "soft"
    reason: str = ""


# ---------------------------------------------------------------------------
# assembled document
# ---------------------------------------------------------------------------


class GenerationProvenance(BaseModel):
    """Everything needed to explain, or reproduce, a run."""

    extractor_version: Optional[str] = None
    structurer_version: str = "0.1.0"
    l0_source_sha256: Optional[str] = None
    model_id: Optional[str] = None
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    run_id: Optional[str] = None
    generated_at: Optional[str] = None


class BookRef(BaseModel):
    book_id: str
    grade: int
    subject: str
    language: str = "fa"
    title: Optional[str] = None
    source: Optional[str] = None
    source_sha256: Optional[str] = None
    page_count: int = 0
    page_offset: Optional[int] = None


class ContentSchema(BaseModel):
    """The L1 source of truth. Index and knowledge graph derive from this."""

    schema_version: str = SCHEMA_VERSION
    book: BookRef
    lessons: List[Lesson] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    concepts: List[Concept] = Field(default_factory=list)
    objectives: List[LearningObjective] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    misconceptions: List[Misconception] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    provenance: GenerationProvenance = Field(default_factory=GenerationProvenance)

    def evidence_by_id(self) -> Dict[str, Evidence]:
        return {item.id: item for item in self.evidence}

    def entity_ids(self) -> Dict[str, str]:
        """Every entity id mapped to its kind, for reference checking."""
        out: Dict[str, str] = {}
        for lesson in self.lessons:
            out[lesson.id] = "lesson"
        for section in self.sections:
            out[section.id] = "section"
        for concept in self.concepts:
            out[concept.id] = "concept"
        for objective in self.objectives:
            out[objective.id] = "objective"
        for skill in self.skills:
            out[skill.id] = "skill"
        for item in self.misconceptions:
            out[item.id] = "misconception"
        return out
