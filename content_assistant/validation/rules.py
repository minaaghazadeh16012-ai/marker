"""Validation rules. Deterministic, and runnable before a model is involved.

Validation is built and proved *before* the semantic layer exists, on purpose.
If the checks were written after seeing a model's first output, they would be
quietly shaped to accept it - the measure has to exist before the thing it
measures.

Rules are grouped by the stage they can run at:

``structure``
    Everything that only needs the deterministic skeleton. Runs today.
``semantic``
    Everything about model-proposed entities and their grounding.
``final``
    Whole-document properties - reference integrity, cycles, orphans.

Each rule is a small class with a stable ``code`` so a finding can be traced,
suppressed, or counted over time without matching on message text.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

from pydantic import BaseModel, Field

from content_assistant.models.content import (
    RELATION_TYPES,
    ContentSchema,
    Lesson,
    Section,
)
from content_assistant.models.extraction import ExtractionResult

Severity = str  # "error" | "warning" | "review"
Stage = str  # "structure" | "semantic" | "final"


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str
    stage: Stage = "structure"
    entity_id: Optional[str] = None
    entity_kind: Optional[str] = None
    page: Optional[int] = None
    details: Dict[str, object] = Field(default_factory=dict)


class ValidationContext(BaseModel):
    """Everything the rules are allowed to look at."""

    model_config = {"arbitrary_types_allowed": True}

    extraction: Optional[ExtractionResult] = None
    lessons: List[Lesson] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    schema_doc: Optional[ContentSchema] = None
    #: Ratio of inferred to explicit entities above which a run is suspect.
    max_inferred_ratio: float = 0.85

    def block_ids(self) -> Set[str]:
        if not self.extraction:
            return set()
        return {
            block.block_id
            for page in self.extraction.pages
            for block in page.blocks
        }

    def asset_ids(self) -> Set[str]:
        if not self.extraction:
            return set()
        return {
            asset.asset_id
            for page in self.extraction.pages
            for asset in page.assets
        }

    def block_text(self) -> Dict[str, str]:
        if not self.extraction:
            return {}
        return {
            block.block_id: block.text
            for page in self.extraction.pages
            for block in page.blocks
        }


class Rule:
    code = "UNSET"
    severity: Severity = "error"
    stage: Stage = "structure"
    description = ""

    def check(self, ctx: ValidationContext) -> Iterable[Finding]:  # pragma: no cover
        raise NotImplementedError

    def finding(self, message: str, **kwargs) -> Finding:
        return Finding(
            code=self.code,
            severity=self.severity,
            stage=self.stage,
            message=message,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# structure rules
# ---------------------------------------------------------------------------


class BookIdentityDeclared(Rule):
    code = "STRUCT001"
    description = "Book identity must be declared, not inferred."

    def check(self, ctx):
        if not ctx.extraction:
            return
        book = ctx.extraction.document.book
        for field in ("book_id", "grade", "subject"):
            if getattr(book, field, None) in (None, ""):
                yield self.finding(
                    f"book identity is missing '{field}'; it must be declared "
                    "explicitly and never guessed",
                    entity_kind="book",
                    details={"field": field},
                )


class UniqueIds(Rule):
    code = "STRUCT002"
    description = "No two entities may share an id."

    def check(self, ctx):
        seen: Dict[str, str] = {}
        groups = [("lesson", ctx.lessons), ("section", ctx.sections)]
        if ctx.schema_doc:
            groups += [
                ("concept", ctx.schema_doc.concepts),
                ("objective", ctx.schema_doc.objectives),
                ("skill", ctx.schema_doc.skills),
                ("misconception", ctx.schema_doc.misconceptions),
                ("relation", ctx.schema_doc.relations),
                ("evidence", ctx.schema_doc.evidence),
            ]
        for kind, items in groups:
            for item in items:
                if item.id in seen:
                    yield self.finding(
                        f"duplicate id {item.id!r} used by "
                        f"{seen[item.id]} and {kind}",
                        entity_id=item.id,
                        entity_kind=kind,
                    )
                seen[item.id] = kind


class LessonPageRangeSane(Rule):
    code = "STRUCT003"
    description = "A lesson's page range must be ordered and inside the book."

    def check(self, ctx):
        page_count = ctx.extraction.document.page_count if ctx.extraction else None
        for lesson in ctx.lessons:
            rng = lesson.page_range
            if rng.pdf_start > rng.pdf_end:
                yield self.finding(
                    f"lesson {lesson.lesson_number} ends before it starts",
                    entity_id=lesson.id,
                    entity_kind="lesson",
                )
            if page_count and rng.pdf_end > page_count:
                yield self.finding(
                    f"lesson {lesson.lesson_number} runs past the last page "
                    f"({rng.pdf_end} > {page_count})",
                    entity_id=lesson.id,
                    entity_kind="lesson",
                )
            if rng.pdf_start < 1:
                yield self.finding(
                    f"lesson {lesson.lesson_number} starts before page 1",
                    entity_id=lesson.id,
                    entity_kind="lesson",
                )


class LessonsDoNotOverlap(Rule):
    code = "STRUCT004"
    description = "Two lessons may not claim the same page."

    def check(self, ctx):
        ordered = sorted(ctx.lessons, key=lambda x: x.page_range.pdf_start)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.page_range.pdf_start <= earlier.page_range.pdf_end:
                yield self.finding(
                    f"lessons {earlier.lesson_number} and {later.lesson_number} "
                    f"overlap on pages "
                    f"{later.page_range.pdf_start}-{earlier.page_range.pdf_end}",
                    entity_id=later.id,
                    entity_kind="lesson",
                )


class LessonOrderMatchesPages(Rule):
    code = "STRUCT005"
    severity = "warning"
    description = "Lesson numbering should follow page order."

    def check(self, ctx):
        ordered = sorted(ctx.lessons, key=lambda x: x.page_range.pdf_start)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.lesson_number < earlier.lesson_number:
                yield self.finding(
                    f"lesson {later.lesson_number} is printed after lesson "
                    f"{earlier.lesson_number} but numbered before it",
                    entity_id=later.id,
                    entity_kind="lesson",
                )


class SectionsBelongToTheirLesson(Rule):
    code = "STRUCT006"
    description = "A section must reference a real lesson and stay inside it."

    def check(self, ctx):
        lessons = {lesson.id: lesson for lesson in ctx.lessons}
        for section in ctx.sections:
            lesson = lessons.get(section.lesson_id)
            if lesson is None:
                yield self.finding(
                    f"section {section.id} references unknown lesson "
                    f"{section.lesson_id!r}",
                    entity_id=section.id,
                    entity_kind="section",
                )
                continue
            if (
                section.page_range.pdf_start < lesson.page_range.pdf_start
                or section.page_range.pdf_end > lesson.page_range.pdf_end
            ):
                yield self.finding(
                    f"section {section.id} covers pages outside its lesson",
                    entity_id=section.id,
                    entity_kind="section",
                )


class ReferencedBlocksExist(Rule):
    code = "STRUCT007"
    description = "Every referenced block id must exist in the L0 artifact."

    def check(self, ctx):
        known = ctx.block_ids()
        if not known:
            return
        for lesson in ctx.lessons:
            for block_id in lesson.block_ids:
                if block_id not in known:
                    yield self.finding(
                        f"lesson {lesson.lesson_number} references unknown block "
                        f"{block_id!r}",
                        entity_id=lesson.id,
                        entity_kind="lesson",
                    )
        for section in ctx.sections:
            for block_id in section.block_ids:
                if block_id not in known:
                    yield self.finding(
                        f"section {section.id} references unknown block "
                        f"{block_id!r}",
                        entity_id=section.id,
                        entity_kind="section",
                    )


class ReferencedAssetsExist(Rule):
    code = "STRUCT008"
    description = "Every referenced asset id must exist in the L0 artifact."

    def check(self, ctx):
        known = ctx.asset_ids()
        if not known:
            return
        for section in ctx.sections:
            for asset_id in section.asset_ids:
                if asset_id not in known:
                    yield self.finding(
                        f"section {section.id} references unknown asset "
                        f"{asset_id!r}",
                        entity_id=section.id,
                        entity_kind="section",
                    )


class LessonHasContent(Rule):
    code = "STRUCT009"
    severity = "warning"
    description = "A lesson with no text at all cannot be structured."

    def check(self, ctx):
        for lesson in ctx.lessons:
            if lesson.material_profile.text_chars == 0:
                yield self.finding(
                    f"lesson {lesson.lesson_number} has no extractable text",
                    entity_id=lesson.id,
                    entity_kind="lesson",
                )


class PagesCovered(Rule):
    code = "STRUCT010"
    severity = "warning"
    description = "Report pages that belong to no lesson."

    def check(self, ctx):
        if not ctx.extraction or not ctx.lessons:
            return
        covered: Set[int] = set()
        for lesson in ctx.lessons:
            covered.update(
                range(lesson.page_range.pdf_start, lesson.page_range.pdf_end + 1)
            )
        missing = [
            page.pdf_page
            for page in ctx.extraction.pages
            if page.pdf_page not in covered
        ]
        if missing:
            yield self.finding(
                f"{len(missing)} pages belong to no lesson (front/back matter "
                "is expected here)",
                details={"pages": missing},
            )


# ---------------------------------------------------------------------------
# evidence rules
# ---------------------------------------------------------------------------


def _grounded_entities(schema: ContentSchema):
    yield from (("concept", c) for c in schema.concepts)
    yield from (("objective", o) for o in schema.objectives)
    yield from (("skill", s) for s in schema.skills)
    yield from (("misconception", m) for m in schema.misconceptions)
    yield from (("relation", r) for r in schema.relations)


class EntityHasEvidence(Rule):
    code = "EVID001"
    stage = "semantic"
    description = "Nothing enters the schema without evidence."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        for kind, entity in _grounded_entities(ctx.schema_doc):
            if not entity.evidence_ids:
                yield self.finding(
                    f"{kind} {entity.id} has no evidence and cannot be grounded "
                    "in the book",
                    entity_id=entity.id,
                    entity_kind=kind,
                )


class EvidenceReferencesRealBlock(Rule):
    code = "EVID002"
    stage = "semantic"
    description = "Evidence must point at a block that exists."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        known = ctx.block_ids()
        for item in ctx.schema_doc.evidence:
            if known and item.block_id not in known:
                yield self.finding(
                    f"evidence {item.id} cites unknown block {item.block_id!r}",
                    entity_id=item.id,
                    entity_kind="evidence",
                )


class EvidenceIdsResolve(Rule):
    code = "EVID003"
    stage = "semantic"
    description = "An entity's evidence ids must exist in the evidence table."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        known = {item.id for item in ctx.schema_doc.evidence}
        for kind, entity in _grounded_entities(ctx.schema_doc):
            for evidence_id in entity.evidence_ids:
                if evidence_id not in known:
                    yield self.finding(
                        f"{kind} {entity.id} cites unknown evidence "
                        f"{evidence_id!r}",
                        entity_id=entity.id,
                        entity_kind=kind,
                    )


class ExplicitNeedsVerifiedQuote(Rule):
    code = "EVID004"
    stage = "semantic"
    description = "'explicit' is only allowed with a verified quotation."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        evidence = ctx.schema_doc.evidence_by_id()
        for kind, entity in _grounded_entities(ctx.schema_doc):
            if entity.evidence_level != "explicit":
                continue
            supports = [
                evidence[e]
                for e in entity.evidence_ids
                if e in evidence
            ]
            if not any(item.quote_verified for item in supports):
                yield self.finding(
                    f"{kind} {entity.id} claims 'explicit' but no cited quote "
                    "was verified against the book",
                    entity_id=entity.id,
                    entity_kind=kind,
                )


class EvidencePageInsideLesson(Rule):
    code = "EVID005"
    stage = "semantic"
    severity = "warning"
    description = "Evidence for a lesson entity should come from that lesson."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        lessons = {lesson.id: lesson for lesson in ctx.lessons}
        evidence = ctx.schema_doc.evidence_by_id()
        scoped = list(ctx.schema_doc.concepts) + list(ctx.schema_doc.objectives)
        for entity in scoped:
            lesson = lessons.get(getattr(entity, "lesson_id", ""))
            if not lesson:
                continue
            for evidence_id in entity.evidence_ids:
                item = evidence.get(evidence_id)
                if item and not lesson.page_range.contains_pdf_page(item.pdf_page):
                    yield self.finding(
                        f"{entity.id} cites page {item.pdf_page}, outside its "
                        f"lesson ({lesson.page_range.pdf_start}-"
                        f"{lesson.page_range.pdf_end})",
                        entity_id=entity.id,
                        page=item.pdf_page,
                    )


class PrintedPageConsistent(Rule):
    code = "EVID006"
    stage = "semantic"
    description = "printed_page must agree with the document page offset."

    def check(self, ctx):
        if not ctx.schema_doc or not ctx.extraction:
            return
        offset = ctx.extraction.document.page_offset
        if offset is None:
            return
        for item in ctx.schema_doc.evidence:
            if item.printed_page is None:
                continue
            if item.pdf_page - item.printed_page != offset:
                yield self.finding(
                    f"evidence {item.id} has printed page {item.printed_page} "
                    f"for pdf page {item.pdf_page}, but the book offset is "
                    f"{offset}",
                    entity_id=item.id,
                    entity_kind="evidence",
                    page=item.pdf_page,
                )


class InferredRatioReasonable(Rule):
    code = "EVID007"
    stage = "final"
    severity = "warning"
    description = "Too much inference means the model is writing, not reading."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        entities = [entity for _, entity in _grounded_entities(ctx.schema_doc)]
        if len(entities) < 5:
            return
        inferred = sum(1 for e in entities if e.evidence_level != "explicit")
        ratio = inferred / len(entities)
        if ratio > ctx.max_inferred_ratio:
            yield self.finding(
                f"{ratio:.0%} of entities are inferred rather than quoted; the "
                "run may be producing content instead of extracting it",
                details={"ratio": round(ratio, 3), "count": len(entities)},
            )


# ---------------------------------------------------------------------------
# pedagogical + final rules
# ---------------------------------------------------------------------------


class ObjectiveIsObservable(Rule):
    code = "PEDA001"
    stage = "semantic"
    severity = "review"
    description = "An objective that cannot be observed cannot be assessed."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        for objective in ctx.schema_doc.objectives:
            if not objective.observable or not objective.performance_verb.strip():
                yield self.finding(
                    f"objective {objective.id} states no observable performance",
                    entity_id=objective.id,
                    entity_kind="objective",
                )


class ObjectiveHasConcept(Rule):
    code = "PEDA002"
    stage = "semantic"
    description = "An objective must be about at least one concept."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        concept_ids = {c.id for c in ctx.schema_doc.concepts}
        for objective in ctx.schema_doc.objectives:
            if not objective.concept_ids:
                yield self.finding(
                    f"objective {objective.id} is attached to no concept",
                    entity_id=objective.id,
                    entity_kind="objective",
                )
                continue
            for concept_id in objective.concept_ids:
                if concept_id not in concept_ids:
                    yield self.finding(
                        f"objective {objective.id} references unknown concept "
                        f"{concept_id!r}",
                        entity_id=objective.id,
                        entity_kind="objective",
                    )


class NoDuplicateConcepts(Rule):
    code = "PEDA003"
    stage = "semantic"
    severity = "warning"
    description = "The same concept must not appear twice under one lesson."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        from content_assistant.models.content import id_slug

        seen: Dict[tuple, str] = {}
        for concept in ctx.schema_doc.concepts:
            key = (concept.lesson_id, id_slug(concept.label))
            if key in seen:
                yield self.finding(
                    f"concept {concept.id} duplicates {seen[key]} "
                    f"({concept.label!r}) in the same lesson",
                    entity_id=concept.id,
                    entity_kind="concept",
                )
            seen[key] = concept.id


class MisconceptionIsGuarded(Rule):
    code = "PEDA004"
    stage = "semantic"
    description = "Misconceptions need stronger evidence and human review."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        concept_ids = {c.id for c in ctx.schema_doc.concepts}
        evidence = ctx.schema_doc.evidence_by_id()
        for item in ctx.schema_doc.misconceptions:
            if item.concept_id not in concept_ids:
                yield self.finding(
                    f"misconception {item.id} references unknown concept "
                    f"{item.concept_id!r}",
                    entity_id=item.id,
                    entity_kind="misconception",
                )
            if not item.requires_human_review:
                yield self.finding(
                    f"misconception {item.id} must be marked for human review",
                    entity_id=item.id,
                    entity_kind="misconception",
                )
            if item.evidence_level == "explicit":
                verified = any(
                    evidence[e].quote_verified
                    for e in item.evidence_ids
                    if e in evidence
                )
                if not verified:
                    yield self.finding(
                        f"misconception {item.id} claims 'explicit' without a "
                        "verified quotation; the book rarely states these "
                        "outright",
                        entity_id=item.id,
                        entity_kind="misconception",
                    )


class RelationsAreWellFormed(Rule):
    code = "FINAL001"
    stage = "final"
    description = "Relations must use the closed vocabulary and real endpoints."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        known = ctx.schema_doc.entity_ids()
        for relation in ctx.schema_doc.relations:
            if relation.relation_type not in RELATION_TYPES:
                yield self.finding(
                    f"relation {relation.id} uses type "
                    f"{relation.relation_type!r}, which is outside the closed "
                    "vocabulary",
                    entity_id=relation.id,
                    entity_kind="relation",
                )
            for role, target in (
                ("source", relation.source_id),
                ("target", relation.target_id),
            ):
                if target not in known:
                    yield self.finding(
                        f"relation {relation.id} has a {role} "
                        f"{target!r} that does not exist",
                        entity_id=relation.id,
                        entity_kind="relation",
                    )
            if relation.source_id == relation.target_id:
                yield self.finding(
                    f"relation {relation.id} points at itself",
                    entity_id=relation.id,
                    entity_kind="relation",
                )


class PrerequisiteGraphIsAcyclic(Rule):
    code = "FINAL002"
    stage = "final"
    description = "Prerequisites must form a DAG - A cannot precede itself."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        edges: Dict[str, List[str]] = {}
        for relation in ctx.schema_doc.relations:
            if relation.relation_type == "prerequisite_of":
                edges.setdefault(relation.source_id, []).append(relation.target_id)

        WHITE, GREY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {}

        def visit(node: str, trail: List[str]) -> Optional[List[str]]:
            colour[node] = GREY
            for nxt in edges.get(node, []):
                state = colour.get(nxt, WHITE)
                if state == GREY:
                    return trail + [node, nxt]
                if state == WHITE:
                    found = visit(nxt, trail + [node])
                    if found:
                        return found
            colour[node] = BLACK
            return None

        for node in list(edges):
            if colour.get(node, WHITE) == WHITE:
                cycle = visit(node, [])
                if cycle:
                    yield self.finding(
                        "prerequisite cycle: " + " -> ".join(cycle),
                        details={"cycle": cycle},
                    )
                    return


class NoOrphanEntities(Rule):
    code = "FINAL003"
    stage = "final"
    severity = "warning"
    description = "Every concept should belong to a lesson that exists."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        lesson_ids = {lesson.id for lesson in ctx.schema_doc.lessons} or {
            lesson.id for lesson in ctx.lessons
        }
        for concept in ctx.schema_doc.concepts:
            if concept.lesson_id not in lesson_ids:
                yield self.finding(
                    f"concept {concept.id} belongs to unknown lesson "
                    f"{concept.lesson_id!r}",
                    entity_id=concept.id,
                    entity_kind="concept",
                )


#: Order matters only for readability of the report.
ALL_RULES: Sequence[Rule] = (
    BookIdentityDeclared(),
    UniqueIds(),
    LessonPageRangeSane(),
    LessonsDoNotOverlap(),
    LessonOrderMatchesPages(),
    SectionsBelongToTheirLesson(),
    ReferencedBlocksExist(),
    ReferencedAssetsExist(),
    LessonHasContent(),
    PagesCovered(),
    EntityHasEvidence(),
    EvidenceReferencesRealBlock(),
    EvidenceIdsResolve(),
    ExplicitNeedsVerifiedQuote(),
    EvidencePageInsideLesson(),
    PrintedPageConsistent(),
    InferredRatioReasonable(),
    ObjectiveIsObservable(),
    ObjectiveHasConcept(),
    NoDuplicateConcepts(),
    MisconceptionIsGuarded(),
    RelationsAreWellFormed(),
    PrerequisiteGraphIsAcyclic(),
    NoOrphanEntities(),
)
