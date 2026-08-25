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
    REVIEW_DECIDED,
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
    """Everything that asserts something about the book, and so must cite it.

    Activities and questions are absent by design: they are designed material
    rather than claims, and holding them to an evidence rule would either be
    ignored or satisfied with a decorative citation. What holds them together
    is ``LINK001``/``LINK002`` instead.
    """
    yield from (("concept", c) for c in schema.concepts)
    yield from (("objective", o) for o in schema.objectives)
    yield from (("skill", s) for s in schema.skills)
    yield from (("misconception", m) for m in schema.misconceptions)
    yield from (("relation", r) for r in schema.relations)


def _reviewable_entities(schema: ContentSchema):
    """Everything a person can pass judgement on - both layers.

    Wider than :func:`_grounded_entities`, because a reviewer signs off on an
    activity or a question exactly as they do on a concept, even though only
    one of the two has to quote the book.
    """
    yield from _grounded_entities(schema)
    yield from (("activity", a) for a in schema.activities)
    yield from (("question", q) for q in schema.questions)


class EntityHasEvidence(Rule):
    code = "EVID001"
    stage = "semantic"
    description = "Nothing enters the schema without evidence, or a person."

    def check(self, ctx):
        """One exemption, and it costs a name.

        A record whose provenance says ``human`` may stand without a quotation,
        because a person is accountable for it and can be asked why. Nothing
        else can: ``deterministic`` and ``model_proposed`` both still have to
        cite the book, so no stage of this pipeline can reach the exemption -
        it is only reachable by a person editing a package, through
        :func:`~content_assistant.models.content.human_relation` or an
        authoring tool.

        The exemption exists because the one relation an adaptive engine most
        needs - ``A is a prerequisite of B`` - is almost never printed in a
        first-grade textbook, while being perfectly real. Without this, the
        only way to record it would have been a decorative citation, which is
        worse: it would look grounded. ``LINK003`` closes the other half, by
        refusing a prerequisite that has neither evidence nor an author.
        """
        if not ctx.schema_doc:
            return
        for kind, entity in _grounded_entities(ctx.schema_doc):
            if entity.evidence_ids:
                continue
            provenance = getattr(entity, "provenance", None)
            if provenance and provenance.extraction_method == "human":
                continue
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


class WordingIsTheBooks(Rule):
    code = "PEDA005"
    stage = "semantic"
    severity = "review"
    description = "A concept should be worded the way its lesson words things."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        for concept in ctx.schema_doc.concepts:
            if concept.out_of_book_vocabulary:
                words = "، ".join(concept.out_of_book_vocabulary[:8])
                yield self.finding(
                    f"concept {concept.id} is explained with wording the lesson "
                    f"never uses ({words}); the citation still stands, but a "
                    "person should read the sentence",
                    entity_id=concept.id,
                    entity_kind="concept",
                    details={"words": concept.out_of_book_vocabulary},
                )


class ObjectiveRestsOnItsConceptsEvidence(Rule):
    code = "PEDA006"
    stage = "semantic"
    description = "An objective may only rest on the evidence of its concept."

    def check(self, ctx):
        """Re-derive the admission rule from the assembled document.

        The extractor already enforces this, and that is exactly why it is
        checked again here: a rule proved only by the code that implements it
        is not proved. This works from block ids rather than evidence ids
        because an objective quoting a different sentence of the same block
        produces a different evidence id by construction.
        """
        if not ctx.schema_doc:
            return
        evidence = ctx.schema_doc.evidence_by_id()
        concepts = {c.id: c for c in ctx.schema_doc.concepts}

        def blocks_of(ids):
            return {
                evidence[i].block_id for i in ids if i in evidence
            }

        for objective in ctx.schema_doc.objectives:
            allowed = set()
            for concept_id in objective.concept_ids:
                concept = concepts.get(concept_id)
                if concept is not None:
                    allowed |= blocks_of(concept.evidence_ids)
            if not allowed:
                continue
            stray = sorted(blocks_of(objective.evidence_ids) - allowed)
            if stray:
                yield self.finding(
                    f"objective {objective.id} cites {', '.join(stray)}, which "
                    "its concept does not rest on; an objective that needs "
                    "other evidence is a new concept, not an objective",
                    entity_id=objective.id,
                    entity_kind="objective",
                    details={"blocks": stray},
                )


class ObjectiveStaysInsideItsLesson(Rule):
    code = "PEDA007"
    stage = "semantic"
    description = "An objective belongs to the lesson its concept belongs to."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        concepts = {c.id: c for c in ctx.schema_doc.concepts}
        for objective in ctx.schema_doc.objectives:
            for concept_id in objective.concept_ids:
                concept = concepts.get(concept_id)
                if concept is None:
                    continue
                if concept.lesson_id != objective.lesson_id:
                    yield self.finding(
                        f"objective {objective.id} sits in lesson "
                        f"{objective.lesson_id} but its concept {concept_id} "
                        f"is in {concept.lesson_id}",
                        entity_id=objective.id,
                        entity_kind="objective",
                    )


class NoDuplicateObjectives(Rule):
    code = "PEDA008"
    stage = "semantic"
    severity = "warning"
    description = "One concept must not carry the same objective twice."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        from content_assistant.models.content import id_slug

        seen: Dict[tuple, str] = {}
        for objective in ctx.schema_doc.objectives:
            for concept_id in objective.concept_ids:
                key = (concept_id, id_slug(objective.statement))
                if key in seen:
                    yield self.finding(
                        f"objective {objective.id} repeats {seen[key]} on "
                        f"concept {concept_id}",
                        entity_id=objective.id,
                        entity_kind="objective",
                    )
                seen[key] = objective.id


class ObjectiveWordingIsTheBooks(Rule):
    code = "PEDA009"
    stage = "semantic"
    severity = "review"
    description = "An objective should be worded the way its lesson words things."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        for objective in ctx.schema_doc.objectives:
            if objective.out_of_book_vocabulary:
                words = "، ".join(objective.out_of_book_vocabulary[:8])
                yield self.finding(
                    f"objective {objective.id} is written with wording the "
                    f"lesson never uses ({words}); a person should read it",
                    entity_id=objective.id,
                    entity_kind="objective",
                    details={"words": objective.out_of_book_vocabulary},
                )


class ObjectiveTypeSuitsItsConcept(Rule):
    code = "PEDA010"
    stage = "semantic"
    severity = "review"
    description = "An objective's kind must suit the kind of concept it serves."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        from content_assistant.models.objective import type_fits_concept

        concepts = {c.id: c for c in ctx.schema_doc.concepts}
        for objective in ctx.schema_doc.objectives:
            for concept_id in objective.concept_ids:
                concept = concepts.get(concept_id)
                if concept is None:
                    continue
                if not type_fits_concept(
                    objective.objective_type, concept.concept_type
                ):
                    yield self.finding(
                        f"objective {objective.id} is "
                        f"{objective.objective_type!r} but concept "
                        f"{concept_id} is {concept.concept_type!r}; asking for "
                        "the wrong kind of performance measures the wrong "
                        "thing",
                        entity_id=objective.id,
                        entity_kind="objective",
                    )


class ObjectiveIsNotAWish(Rule):
    code = "PEDA011"
    stage = "semantic"
    severity = "review"
    description = "An objective must name a behaviour, not a state of mind."

    def check(self, ctx):
        """Read the statement itself rather than trusting ``observable``.

        PEDA001 reports what the extractor concluded; this reports what the
        sentence says. Checking the text independently is the only way a
        mislabelled objective ever surfaces.
        """
        if not ctx.schema_doc:
            return
        from content_assistant.models.objective import is_vague

        for objective in ctx.schema_doc.objectives:
            if is_vague(objective.statement):
                yield self.finding(
                    f"objective {objective.id} asks the student to know or "
                    "understand something; nobody can observe that, so it "
                    "cannot be assessed",
                    entity_id=objective.id,
                    entity_kind="objective",
                )


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


# ---------------------------------------------------------------------------
# provenance rules
#
# "Every important claim must be traceable to its source" is only a slogan
# until something refuses a claim that is not. These two are that refusal.
# ---------------------------------------------------------------------------


class ModelProposedEntityNamesItsModel(Rule):
    code = "PROV001"
    stage = "semantic"
    description = "A model-proposed record must say which model, and which prompt."

    def check(self, ctx):
        """Only fires on a record that claims a model produced it.

        A missing provenance block is silence, not a false claim - a 1.0.0
        artifact has none, and reporting those as faults would flood the report
        with the schema's own history. What cannot stand is a record that says
        "a model wrote me" and then cannot say which one: that is exactly the
        claim nobody can check later.
        """
        if not ctx.schema_doc:
            return
        for kind, entity in _grounded_entities(ctx.schema_doc):
            provenance = getattr(entity, "provenance", None)
            if provenance is None:
                continue
            if provenance.extraction_method != "model_proposed":
                continue
            missing = [
                field
                for field in ("model_id", "prompt_version")
                if not getattr(provenance, field, None)
            ]
            if missing:
                yield self.finding(
                    f"{kind} {entity.id} says a model proposed it but records "
                    f"no {', '.join(missing)}; the claim cannot be traced back "
                    "to the run that made it",
                    entity_id=entity.id,
                    entity_kind=kind,
                    details={"missing": missing},
                )


class ReviewDecisionIsAttributed(Rule):
    code = "PROV002"
    stage = "semantic"
    description = "A review decision must name who made it and when."

    def check(self, ctx):
        """A verdict nobody signed is not a review.

        ``pending`` is exempt because it is the absence of a decision rather
        than one. Everything else is a person overriding, or confirming, what
        the pipeline computed - and an unattributed override is the one edit in
        the whole schema that cannot be argued with afterwards.
        """
        if not ctx.schema_doc:
            return
        for kind, entity in _reviewable_entities(ctx.schema_doc):
            status = getattr(entity, "review_status", "pending")
            if status not in REVIEW_DECIDED:
                continue
            missing = [
                field
                for field in ("reviewed_by", "reviewed_at")
                if not getattr(entity, field, None)
            ]
            if missing:
                yield self.finding(
                    f"{kind} {entity.id} is marked {status!r} but records no "
                    f"{', '.join(missing)}",
                    entity_id=entity.id,
                    entity_kind=kind,
                    details={"missing": missing, "review_status": status},
                )


# ---------------------------------------------------------------------------
# linkage rules
#
# The content-knowledge layers are held together by evidence; the
# learning-experience layer is held together by these. An activity is not a
# claim about the book and cannot be asked to quote one, so what it *can* be
# asked is that it serves something real and that everything it names exists.
# ---------------------------------------------------------------------------

#: ``(kind, id-list attribute, what the target must be)``. Kept as data so
#: adding an entity type is a row rather than another loop.
_REFERENCE_FIELDS: Sequence[tuple] = (
    ("objective", "concept_ids", "concept"),
    ("skill", "concept_ids", "concept"),
    ("skill", "objective_ids", "objective"),
    ("skill", "lesson_ids", "lesson"),
    ("activity", "objective_ids", "objective"),
    ("activity", "question_ids", "question"),
    ("question", "objective_ids", "objective"),
)


class ReferencesResolve(Rule):
    code = "LINK001"
    stage = "final"
    description = "Every reference between content entities must point at one."

    def check(self, ctx):
        if not ctx.schema_doc:
            return
        schema = ctx.schema_doc
        kinds = schema.entity_ids()
        groups = {
            "objective": schema.objectives,
            "skill": schema.skills,
            "activity": schema.activities,
            "question": schema.questions,
        }
        for kind, attribute, expected in _REFERENCE_FIELDS:
            for entity in groups[kind]:
                for target in getattr(entity, attribute, []) or []:
                    found = kinds.get(target)
                    if found is None:
                        yield self.finding(
                            f"{kind} {entity.id} references {target!r} in "
                            f"{attribute}, which does not exist",
                            entity_id=entity.id,
                            entity_kind=kind,
                        )
                    elif found != expected:
                        yield self.finding(
                            f"{kind} {entity.id} lists {target!r} as a "
                            f"{expected} in {attribute}, but it is a {found}",
                            entity_id=entity.id,
                            entity_kind=kind,
                        )
        # ``skill_id`` is a single value rather than a list, so it does not fit
        # the table above; the check it needs is the same one.
        for objective in schema.objectives:
            if objective.skill_id and kinds.get(objective.skill_id) != "skill":
                yield self.finding(
                    f"objective {objective.id} names skill "
                    f"{objective.skill_id!r}, which is not a skill in this "
                    "package",
                    entity_id=objective.id,
                    entity_kind="objective",
                )


class ExperienceServesSomething(Rule):
    code = "LINK002"
    stage = "final"
    description = "An activity and a question must each serve an objective."

    def check(self, ctx):
        """The learning-experience layer's whole integrity rule.

        A question measuring no objective yields a score that means nothing -
        there is no statement of the form "the student can now ..." that
        getting it right would support. An activity serving no objective is
        material with no place in any path through the book. Neither is a
        quality problem to be scored down; both are unusable, so both are
        errors.
        """
        if not ctx.schema_doc:
            return
        for question in ctx.schema_doc.questions:
            if not question.objective_ids:
                yield self.finding(
                    f"question {question.id} tests no objective; a score on it "
                    "measures nothing anyone can name",
                    entity_id=question.id,
                    entity_kind="question",
                )
        for activity in ctx.schema_doc.activities:
            if not activity.objective_ids:
                yield self.finding(
                    f"activity {activity.id} serves no objective; nothing "
                    "would ever schedule it",
                    entity_id=activity.id,
                    entity_kind="activity",
                )


class PrerequisiteIsAccountable(Rule):
    code = "LINK003"
    stage = "final"
    description = "A prerequisite must be quoted from the book or signed by a person."

    def check(self, ctx):
        """The one edge an adaptive engine trusts most, held to the most.

        When a student fails, the scheduler walks ``prerequisite_of`` backwards
        and sends them somewhere else. A guessed edge therefore does not
        produce a slightly worse recommendation, it sends a child to the wrong
        lesson - so a guess is not allowed to look like the other two.

        Either the book says so, in a verified quotation, or a named person
        does. ``EVID001`` lets a human-authored record stand without evidence;
        this makes sure that is the *only* way one stands.
        """
        if not ctx.schema_doc:
            return
        evidence = ctx.schema_doc.evidence_by_id()
        for relation in ctx.schema_doc.relations:
            if relation.relation_type != "prerequisite_of":
                continue
            verified = any(
                evidence[e].quote_verified
                for e in relation.evidence_ids
                if e in evidence
            )
            if verified:
                continue
            provenance = relation.provenance
            authored = (
                provenance is not None
                and provenance.extraction_method == "human"
                and bool(provenance.authored_by)
            )
            if authored:
                continue
            yield self.finding(
                f"prerequisite {relation.source_id} -> {relation.target_id} "
                "has neither a verified quotation nor a named author; a "
                "guessed prerequisite sends a student to the wrong lesson",
                entity_id=relation.id,
                entity_kind="relation",
            )


class SkillIsMoreThanOneObjective(Rule):
    code = "LINK004"
    stage = "final"
    severity = "warning"
    description = "A skill must generalise; one that does not is its objective."

    def check(self, ctx):
        """What separates a skill from the objective it was copied from.

        A skill is the ability that carries across objectives. Grouping one
        objective and repeating its sentence does not describe anything the
        objective did not, and it doubles the vocabulary a dashboard has to
        show. Reported rather than refused: a single-objective skill can be a
        legitimate first entry in a group that is still being filled in.
        """
        if not ctx.schema_doc:
            return
        from content_assistant.models.content import id_slug

        statements = {
            objective.id: id_slug(objective.statement)
            for objective in ctx.schema_doc.objectives
        }
        for skill in ctx.schema_doc.skills:
            if not skill.objective_ids:
                yield self.finding(
                    f"skill {skill.id} groups no objective; there is nothing a "
                    "student could do that would show they have it",
                    entity_id=skill.id,
                    entity_kind="skill",
                )
                continue
            if len(skill.objective_ids) > 1:
                continue
            only = skill.objective_ids[0]
            if statements.get(only) == id_slug(skill.label):
                yield self.finding(
                    f"skill {skill.id} restates objective {only} and "
                    "generalises nothing",
                    entity_id=skill.id,
                    entity_kind="skill",
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
    WordingIsTheBooks(),
    ObjectiveRestsOnItsConceptsEvidence(),
    ObjectiveStaysInsideItsLesson(),
    NoDuplicateObjectives(),
    ObjectiveWordingIsTheBooks(),
    ObjectiveTypeSuitsItsConcept(),
    ObjectiveIsNotAWish(),
    MisconceptionIsGuarded(),
    RelationsAreWellFormed(),
    PrerequisiteGraphIsAcyclic(),
    NoOrphanEntities(),
    ModelProposedEntityNamesItsModel(),
    ReviewDecisionIsAttributed(),
    ReferencesResolve(),
    ExperienceServesSomething(),
    PrerequisiteIsAccountable(),
    SkillIsMoreThanOneObjective(),
)
