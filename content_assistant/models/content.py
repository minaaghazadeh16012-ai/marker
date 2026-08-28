"""The L1 content model: what a textbook *teaches*, and where each claim came from.

Three rules shape every type here.

**No claim about the book exists without evidence.** Everything that asserts
what a lesson teaches - every :class:`Grounded` entity - carries at least one
:class:`Evidence` id pointing at a block in the L0 artifact. Such an entity
with no evidence is not "low confidence", it is invalid, and ``EVID001``
rejects it. This is the mechanical form of "the book is the source of truth".
The single exception is a record a named person authored, because a person can
be asked why and a model cannot.

**Evidence is a table, not a field.** One sentence in the book can support a
concept, an objective and a misconception at once. Storing the quote three
times invites three copies that drift apart, so quotes live in
:class:`Evidence` and entities reference them by id.

**Ids are derived, not allocated.** Re-running the pipeline on an unchanged
book must produce byte-identical ids, otherwise every run looks like a total
rewrite and no one can review a diff. See :func:`make_id`.

The file is organised in the three layers a consumer has to keep apart:
*content knowledge* (what the book says), *learning intent* (what a student
should be able to do), and - in :mod:`content_assistant.models.learning` -
*learning experience* (what a student does in order to get there). The first
two are bound by evidence; the third is bound by linkage, because a practice
game is not a claim about the book and demanding a quotation for one would be
a category error.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

# Identity, provenance and review live one layer down, in ``common``, because
# the learning-experience layer needs exactly the same three and importing
# this module for them would put the two in a circle. They are re-exported
# here so every existing import of them keeps working.
from content_assistant.models.common import (  # noqa: F401
    CONTENT_SCHEMA_VERSION,
    REVIEW_DECIDED,
    SCHEMA_VERSION,
    Attributed,
    BBox,
    ContentType,
    DifficultyBand,
    EvidenceLevel,
    ExtractionMethod,
    Provenance,
    ReviewStatus,
    id_slug,
    make_id,
    ordinal_id,
)
from content_assistant.models.learning import (  # noqa: F401
    LearningActivity,
    Question,
    QuestionOption,
)

TitleSource = Literal["toc", "lesson_opening_page", "section_header", "fallback"]
BoundaryMethod = Literal["section_header", "page_fallback", "whole_lesson"]
TextDensity = Literal["low", "medium", "high"]

#: What kind of thinking a concept asks for. Borrowed as an idea only - no
#: external taxonomy data is imported into this project.
ConceptType = Literal[
    "conceptual", "procedural", "representational", "language", "meta"
]

#: What a student visibly does to satisfy an objective. Closed, and short on
#: purpose: nothing here can be met by thinking about something. The verbs that
#: realise each kind, and which kinds suit which concept type, live in
#: :mod:`content_assistant.models.objective`.
ObjectiveType = Literal[
    "name", "identify", "describe", "classify", "compare", "perform"
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


class Grounded(Attributed):
    """Everything that makes a claim about the book, and must cite it.

    The learning-experience layer deliberately does **not** inherit from this:
    an activity is designed material, not an assertion about the text, and
    demanding a quotation for a matching game would be a category error. It
    inherits :class:`Attributed` instead - provenance and review without
    evidence. See :mod:`content_assistant.models.learning`.
    """

    evidence_ids: List[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel = "inferred"
    #: Computed by the pipeline from verification facts - see the review layer.
    #: A model's own stated confidence is at most one small input to this.
    confidence: float = 0.0


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
    #: Words in the label or definition that the lesson never uses. A quotation
    #: can verify perfectly while the sentence around it imports vocabulary the
    #: book does not have - this records that, without touching the citation.
    out_of_book_vocabulary: List[str] = Field(default_factory=list)
    #: How hard this idea is. **Never set by extraction**, because a textbook
    #: does not state it and inferring it from text length or word rarity
    #: would be a number with nothing behind it. It exists so an authoring
    #: tool or a teacher has somewhere to put a judgement a scheduler can then
    #: read; until one does it is ``None``, which is the honest value.
    difficulty: Optional[DifficultyBand] = None

    # Prerequisites and related concepts are deliberately *not* fields here.
    # They live in Relation, which carries the evidence and the provenance
    # that make such a link auditable; a copy on the concept would be a second
    # source of truth for the same fact and would drift from the first. Ask
    # ContentSchema.prerequisites_of() and .related_to() instead.
    #
    # grade and subject are absent for the same reason: they are properties of
    # the book, and ContentSchema.book already holds them once.


class LearningObjective(Grounded):
    id: str
    lesson_id: str
    section_id: Optional[str] = None
    statement: str
    #: What the student visibly does. The closed vocabulary and the rules for
    #: pairing it with a concept type live in
    #: :mod:`content_assistant.models.objective`, which is versioned apart from
    #: this file because it encodes pedagogical judgement rather than shape.
    objective_type: ObjectiveType = "identify"
    #: The action the student performs - what makes an objective assessable.
    performance_verb: str = ""
    #: False marks an objective that cannot be observed, so it is flagged
    #: rather than quietly shipped as if it could be tested.
    observable: bool = True
    concept_ids: List[str] = Field(default_factory=list)
    skill_id: Optional[str] = None
    content_types: List[ContentType] = Field(default_factory=list)
    #: Words in the statement that the lesson never uses, with this pipeline's
    #: own performance verbs already subtracted. Same meaning as the field of
    #: the same name on :class:`Concept`: a signal to read the sentence, never
    #: a reason to drop the citation.
    out_of_book_vocabulary: List[str] = Field(default_factory=list)


class Skill(Grounded):
    """A transferable ability that several objectives all exercise.

    The line against :class:`LearningObjective` is the whole reason this type
    exists, so it is worth stating: an objective is bound to one concept in one
    lesson - *say what a magnet does to a paperclip*. A skill is what carries
    across them - *predict what will happen in a simple physical setup*. One
    skill grouping one objective is not a skill; it is that objective spelled
    twice, and ``LINK004`` reports it.

    Nothing in this pipeline proposes skills. :func:`skill_from_objectives`
    builds one from objectives a person has decided belong together, which
    keeps its evidence and its ceiling honest: both are inherited from the
    objectives it groups rather than asserted about it.
    """

    id: str
    label: str
    description: str = ""
    concept_ids: List[str] = Field(default_factory=list)
    #: The objectives this skill generalises. A skill with none is unfounded -
    #: there is nothing a student could do that would show they have it.
    objective_ids: List[str] = Field(default_factory=list)
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
    """A typed edge between two entities, carrying why anyone believes it.

    This is where the graph lives. ``prerequisite_of`` in particular is the
    edge an adaptive scheduler walks backwards when a student fails, so it is
    held to the strictest rule in the schema: it must be quoted from the book
    or authored by a named person, never guessed. ``LINK003`` enforces that,
    and ``FINAL002`` refuses a cycle.
    """

    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: Literal["hard", "soft"] = "soft"
    reason: str = ""

    @staticmethod
    def build_id(
        book_id: str, source_id: str, relation_type: str, target_id: str
    ) -> str:
        return make_id(book_id, "rel", source_id, relation_type, target_id)


# ---------------------------------------------------------------------------
# derived entities
#
# Two constructors, and no extractor. Both build an entity out of entities that
# already stood up to verification, so neither can be more certain than what it
# was built from - the same rule that caps an objective at its concept.
# ---------------------------------------------------------------------------


def skill_from_objectives(
    *,
    book_id: str,
    label: str,
    objectives: Sequence["LearningObjective"],
    concept_ids: Sequence[str] = (),
    description: str = "",
    authored_by: str = "",
    generated_at: Optional[str] = None,
) -> "Skill":
    """Group objectives a person has judged to exercise one ability.

    The evidence is the union of theirs and the confidence is the *minimum* of
    theirs, both for the same reason: a skill is a claim about all of them at
    once, so the weakest member is what it can be trusted to. Taking the mean
    would let one well-grounded objective carry four shaky ones.

    Authorship is human by construction. Nothing decides that two objectives
    describe one transferable ability except a person, and recording anything
    else here would misattribute the judgement.
    """
    if not objectives:
        raise ValueError(
            "a skill must generalise at least one objective; one with none is "
            "a label with nothing a student could be seen doing"
        )
    evidence_ids: List[str] = []
    for objective in objectives:
        for evidence_id in objective.evidence_ids:
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    lesson_ids: List[str] = []
    for objective in objectives:
        if objective.lesson_id not in lesson_ids:
            lesson_ids.append(objective.lesson_id)
    concepts = list(concept_ids) or [
        concept_id
        for objective in objectives
        for concept_id in objective.concept_ids
    ]
    levels = {objective.evidence_level for objective in objectives}
    level: EvidenceLevel = "explicit" if levels == {"explicit"} else "inferred"
    return Skill(
        id=make_id(book_id, "skill", label),
        label=label.strip(),
        description=description.strip(),
        concept_ids=list(dict.fromkeys(concepts)),
        objective_ids=[objective.id for objective in objectives],
        lesson_ids=lesson_ids,
        evidence_ids=evidence_ids,
        evidence_level=level,
        confidence=round(min(o.confidence for o in objectives), 4),
        requires_human_review=any(
            o.requires_human_review for o in objectives
        ),
        review_reasons=(
            ["groups objectives that are themselves under review"]
            if any(o.requires_human_review for o in objectives)
            else []
        ),
        provenance=Provenance(
            extraction_method="human",
            stage="skills",
            authored_by=authored_by or None,
            generated_at=generated_at,
        ),
    )


def human_relation(
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
    """Author a relation on pedagogical judgement rather than a quotation.

    The book states a prerequisite about as often as it states a
    misconception, which is to say almost never - yet the ordering is real and
    a scheduler needs it. This is the sanctioned way to record one, and it
    costs a name: ``authored_by`` is required, the record is marked for review,
    and ``LINK003`` refuses any prerequisite that has neither evidence nor a
    person behind it. A model cannot reach this function.
    """
    if not authored_by.strip():
        raise ValueError(
            "a relation with no evidence needs a person accountable for it; "
            "pass authored_by"
        )
    if not reason.strip():
        raise ValueError(
            "a relation authored on judgement must say what the judgement was"
        )
    return Relation(
        id=Relation.build_id(book_id, source_id, relation_type, target_id),
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        strength=strength,
        reason=reason.strip(),
        evidence_ids=list(evidence_ids),
        evidence_level="inferred",
        confidence=confidence,
        requires_human_review=True,
        review_reasons=["authored on judgement rather than quoted"],
        provenance=Provenance(
            extraction_method="human",
            stage="relations",
            authored_by=authored_by.strip(),
            generated_at=generated_at,
        ),
    )


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
    """One book's content, whole. Index and knowledge graph derive from this.

    The resolver methods below are the reason several fields a reader might
    expect are missing from the entities themselves. A concept has no
    ``prerequisite_concepts`` list and a question has no ``concept_ids``,
    because both facts are already stored once - in :class:`Relation` and in
    :attr:`Question.objective_ids` - and a second copy is a second thing to
    keep in step. Ask here instead; the traversal is cheap and cannot drift.
    """

    schema_version: str = SCHEMA_VERSION
    book: BookRef
    lessons: List[Lesson] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    concepts: List[Concept] = Field(default_factory=list)
    objectives: List[LearningObjective] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    misconceptions: List[Misconception] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    #: The learning-experience layer. Empty for a package that has only been
    #: extracted: nothing in this pipeline writes activities or questions, and
    #: an empty list is the truthful record of that.
    activities: List[LearningActivity] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    provenance: GenerationProvenance = Field(default_factory=GenerationProvenance)

    def evidence_by_id(self) -> Dict[str, Evidence]:
        return {item.id: item for item in self.evidence}

    def entity_ids(self) -> Dict[str, str]:
        """Every id a reference may point *at*, mapped to its kind.

        Relations are absent on purpose, and it is not an oversight: this is
        what ``FINAL001`` checks a relation's own endpoints against, and a
        relation whose target is another relation states nothing anyone can
        act on. :meth:`all_ids` is the wider map, for a registry that has to be
        able to find any record at all.
        """
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
        for activity in self.activities:
            out[activity.id] = "activity"
        for question in self.questions:
            out[question.id] = "question"
        return out

    def all_ids(self) -> Dict[str, str]:
        """Every record in the package, relations and evidence included.

        What a lookup index is built from. Wider than :meth:`entity_ids`
        because "find me this id" and "may a relation point here?" are
        different questions, and answering both from one map would have to
        settle for the wrong answer to one of them.
        """
        out = self.entity_ids()
        for relation in self.relations:
            out[relation.id] = "relation"
        for item in self.evidence:
            out[item.id] = "evidence"
        return out

    # -- traversal -------------------------------------------------------
    #
    # Everything an adaptive scheduler asks of the content layer, and nothing
    # about a particular student. Where a learner's state would go, these
    # return *options* instead: which activities exist for this objective,
    # which of them are remedial. Choosing between them is the engine's job,
    # and needs a learner - which is exactly why no learner appears here.

    def objectives_for_concept(
        self, concept_id: str
    ) -> List[LearningObjective]:
        return [o for o in self.objectives if concept_id in o.concept_ids]

    def concepts_for_objective(self, objective_id: str) -> List[Concept]:
        objective = self.by_id(objective_id)
        if not isinstance(objective, LearningObjective):
            return []
        wanted = set(objective.concept_ids)
        return [c for c in self.concepts if c.id in wanted]

    def questions_for_objective(self, objective_id: str) -> List[Question]:
        """What would show that this objective has been met."""
        return [q for q in self.questions if objective_id in q.objective_ids]

    def activities_for_objective(
        self, objective_id: str, activity_type: Optional[str] = None
    ) -> List[LearningActivity]:
        """What a student can do towards this objective.

        Pass ``activity_type="remediation"`` for the after-a-failure list.
        """
        found = [
            a for a in self.activities if objective_id in a.objective_ids
        ]
        if activity_type is None:
            return found
        return [a for a in found if a.activity_type == activity_type]

    def concepts_for_question(self, question_id: str) -> List[Concept]:
        """The knowledge an item touches, reached through its objectives.

        Derived rather than stored, which is the whole rule: a question is a
        consumer of the content schema, never an owner of it. Re-point the
        question at a different objective and this answer changes with it.
        """
        question = self.by_id(question_id)
        if not isinstance(question, Question):
            return []
        by_id = {o.id: o for o in self.objectives}
        wanted: List[str] = []
        for objective_id in question.objective_ids:
            objective = by_id.get(objective_id)
            if objective is None:
                continue
            for concept_id in objective.concept_ids:
                if concept_id not in wanted:
                    wanted.append(concept_id)
        by_concept = {c.id: c for c in self.concepts}
        return [by_concept[c] for c in wanted if c in by_concept]

    def objectives_for_question(
        self, question_id: str
    ) -> List[LearningObjective]:
        """What this item claims to measure.

        The first hop of ``question -> objective -> concept``, offered on its
        own because a scheduler that has just marked an attempt needs the
        objectives and not the concepts behind them.
        """
        question = self.by_id(question_id)
        if not isinstance(question, Question):
            return []
        wanted = set(question.objective_ids)
        return [o for o in self.objectives if o.id in wanted]

    def objectives_for_activity(
        self, activity_id: str
    ) -> List[LearningObjective]:
        activity = self.by_id(activity_id)
        if not isinstance(activity, LearningActivity):
            return []
        wanted = set(activity.objective_ids)
        return [o for o in self.objectives if o.id in wanted]

    def skills_for_objective(self, objective_id: str) -> List[Skill]:
        """The transferable abilities this objective exercises.

        Read off :attr:`Skill.objective_ids`, which is the side that owns the
        grouping: a skill is *defined* by the objectives it generalises, while
        :attr:`LearningObjective.skill_id` is a convenience pointer back that an
        author may or may not have filled in. ``LINK005`` refuses the two to
        disagree; this method never has to choose between them.
        """
        return [s for s in self.skills if objective_id in s.objective_ids]

    def activity_for_question(self, question_id: str) -> Optional[
        LearningActivity
    ]:
        """The activity that asks this question, if one does.

        Derived rather than stored. An activity owns an *ordered* list of the
        questions it asks, and that order is a fact only the activity can hold;
        a back-pointer on the question would restate the membership half of it
        and be free to disagree.
        """
        for activity in self.activities:
            if question_id in activity.question_ids:
                return activity
        return None

    def sections_for_lesson(self, lesson_id: str) -> List[Section]:
        return sorted(
            (s for s in self.sections if s.lesson_id == lesson_id),
            key=lambda s: s.order,
        )

    def evidence_for(self, entity_id: str) -> List[Evidence]:
        """The quotations behind a claim, resolved.

        ``concept -> evidence`` in one call, for a reviewer or an interface
        that has to show why a claim is in the book.
        """
        entity = self.by_id(entity_id)
        wanted = list(getattr(entity, "evidence_ids", []) or [])
        by_id = self.evidence_by_id()
        return [by_id[e] for e in wanted if e in by_id]

    def concepts_for_activity(self, activity_id: str) -> List[Concept]:
        activity = self.by_id(activity_id)
        if not isinstance(activity, LearningActivity):
            return []
        by_id = {o.id: o for o in self.objectives}
        wanted: List[str] = []
        for objective_id in activity.objective_ids:
            objective = by_id.get(objective_id)
            if objective is None:
                continue
            for concept_id in objective.concept_ids:
                if concept_id not in wanted:
                    wanted.append(concept_id)
        by_concept = {c.id: c for c in self.concepts}
        return [by_concept[c] for c in wanted if c in by_concept]

    def prerequisites_of(self, entity_id: str) -> List[str]:
        """What must come before ``entity_id``, one hop out.

        Reads ``prerequisite_of`` edges in the direction they are stored:
        ``source_id`` is the prerequisite, ``target_id`` is what needs it.
        """
        return [
            relation.source_id
            for relation in self.relations
            if relation.relation_type == "prerequisite_of"
            and relation.target_id == entity_id
        ]

    def dependents_of(self, entity_id: str) -> List[str]:
        """What ``entity_id`` unlocks - the other direction of the same edge."""
        return [
            relation.target_id
            for relation in self.relations
            if relation.relation_type == "prerequisite_of"
            and relation.source_id == entity_id
        ]

    def related_to(self, entity_id: str) -> List[str]:
        """Neighbours on every edge that is not a prerequisite.

        Undirected on purpose: ``related_to``, ``elaborates`` and
        ``example_of`` are useful from either end when a student is looking for
        another way in.
        """
        out: List[str] = []
        for relation in self.relations:
            if relation.relation_type == "prerequisite_of":
                continue
            if relation.source_id == entity_id:
                out.append(relation.target_id)
            elif relation.target_id == entity_id:
                out.append(relation.source_id)
        return out

    def by_id(self, entity_id: str):
        """Any entity by id, or ``None``. Evidence included."""
        for group in (
            self.lessons,
            self.sections,
            self.concepts,
            self.objectives,
            self.skills,
            self.misconceptions,
            self.relations,
            self.activities,
            self.questions,
            self.evidence,
        ):
            for item in group:
                if item.id == entity_id:
                    return item
        return None
