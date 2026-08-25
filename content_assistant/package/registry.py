"""One place that knows what content exists, and where every id lives.

A consumer holding an objective id should not have to know which book it came
from, and an engine choosing what to show next should not have to open files to
find out what grades exist. The registry answers both, over any number of
packages, from one index.

**It stores no content of its own.** The index maps an id to the very object
inside the package that was loaded - the same instance, not a copy - so there
is nothing here that can fall out of step with a package, and nothing to
invalidate when one is reloaded. What the registry owns is the *mapping*, which
exists nowhere else.

The traversals it offers are the ones an adaptive engine actually performs, and
none of them takes a student. That boundary is the point: the registry answers
"what could come next", never "what should this child do next". The second
question needs a learner's state, and learner state is deliberately not part of
the content schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from content_assistant.models.content import (
    Concept,
    Lesson,
    LearningObjective,
    Skill,
)
from content_assistant.models.learning import LearningActivity, Question
from content_assistant.package.schema import (
    PACKAGE_FILENAME,
    ContentPackage,
    load_content,
)


class ContentRegistry:
    """Every loaded package, indexed by id."""

    def __init__(self) -> None:
        self._packages: Dict[str, ContentPackage] = {}
        #: ``entity_id -> (package_id, kind)``. The object itself is fetched
        #: from the package on demand rather than held here, so the registry
        #: never becomes a second copy of the content.
        self._index: Dict[str, Tuple[str, str]] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def from_directory(cls, root: Path) -> "ContentRegistry":
        """Load every package under ``root``.

        Discovery is by file name, not by directory depth, so a tree organised
        some other way still works as long as each package is written to
        ``content-package.json``.
        """
        registry = cls()
        for path in sorted(Path(root).rglob(PACKAGE_FILENAME)):
            registry.add(load_content(path))
        return registry

    @classmethod
    def from_packages(
        cls, packages: Iterable[ContentPackage]
    ) -> "ContentRegistry":
        registry = cls()
        for package in packages:
            registry.add(package)
        return registry

    def add(self, package: ContentPackage) -> None:
        """Index a package, refusing an id that already belongs to another.

        Two packages sharing an entity id means one of them is not the book it
        says it is - ids are derived from the book id, so a collision across
        books cannot happen by accident. Loading both would make every lookup
        depend on load order.
        """
        if package.package_id in self._packages:
            raise ValueError(
                f"package {package.package_id!r} is already registered; two "
                "files claim the same grade, subject and book"
            )
        for entity_id, kind in package.content.entity_ids().items():
            owner = self._index.get(entity_id)
            if owner is not None and owner[0] != package.package_id:
                raise ValueError(
                    f"id {entity_id!r} is claimed by both "
                    f"{owner[0]!r} and {package.package_id!r}"
                )
            self._index[entity_id] = (package.package_id, kind)
        self._packages[package.package_id] = package

    # -- what exists -----------------------------------------------------

    def packages(self) -> List[ContentPackage]:
        return [self._packages[key] for key in sorted(self._packages)]

    def package(self, package_id: str) -> Optional[ContentPackage]:
        return self._packages.get(package_id)

    def grades(self) -> List[int]:
        return sorted({p.grade for p in self._packages.values()})

    def subjects(self, grade: Optional[int] = None) -> List[str]:
        return sorted(
            {
                p.subject
                for p in self._packages.values()
                if grade is None or p.grade == grade
            }
        )

    def find(self, grade: int, subject: str) -> Optional[ContentPackage]:
        for package in self._packages.values():
            if package.grade == grade and package.subject == subject:
                return package
        return None

    def lessons(self, grade: Optional[int] = None) -> List[Lesson]:
        return list(self._collect("lessons", grade))

    def concepts(self, grade: Optional[int] = None) -> List[Concept]:
        return list(self._collect("concepts", grade))

    def objectives(
        self, grade: Optional[int] = None
    ) -> List[LearningObjective]:
        return list(self._collect("objectives", grade))

    def skills(self, grade: Optional[int] = None) -> List[Skill]:
        return list(self._collect("skills", grade))

    def activities(
        self, grade: Optional[int] = None
    ) -> List[LearningActivity]:
        return list(self._collect("activities", grade))

    def questions(self, grade: Optional[int] = None) -> List[Question]:
        return list(self._collect("questions", grade))

    def _collect(self, attribute: str, grade: Optional[int]) -> Iterator:
        for package in self.packages():
            if grade is not None and package.grade != grade:
                continue
            yield from getattr(package.content, attribute)

    # -- lookup ----------------------------------------------------------

    def kind_of(self, entity_id: str) -> Optional[str]:
        found = self._index.get(entity_id)
        return found[1] if found else None

    def package_of(self, entity_id: str) -> Optional[ContentPackage]:
        found = self._index.get(entity_id)
        return self._packages.get(found[0]) if found else None

    def get(self, entity_id: str):
        """Any entity in any loaded package, or ``None``."""
        package = self.package_of(entity_id)
        return package.content.by_id(entity_id) if package else None

    # -- traversal -------------------------------------------------------
    #
    # Each of these delegates to the package that owns the id, because every
    # link in this schema is within one book. Cross-book relations would need
    # a decision about what a prerequisite across two textbooks means, and
    # nothing has needed one yet - so nothing here pretends to support it.

    def objectives_for_concept(
        self, concept_id: str
    ) -> List[LearningObjective]:
        package = self.package_of(concept_id)
        if package is None:
            return []
        return package.content.objectives_for_concept(concept_id)

    def concepts_for_objective(self, objective_id: str) -> List[Concept]:
        package = self.package_of(objective_id)
        if package is None:
            return []
        return package.content.concepts_for_objective(objective_id)

    def questions_for_objective(self, objective_id: str) -> List[Question]:
        package = self.package_of(objective_id)
        if package is None:
            return []
        return package.content.questions_for_objective(objective_id)

    def activities_for_objective(
        self, objective_id: str, activity_type: Optional[str] = None
    ) -> List[LearningActivity]:
        package = self.package_of(objective_id)
        if package is None:
            return []
        return package.content.activities_for_objective(
            objective_id, activity_type
        )

    def remediation_for_objective(
        self, objective_id: str
    ) -> List[LearningActivity]:
        """What to offer after a student fails this objective.

        A named method rather than a magic string at every call site, because
        this is the single traversal an adaptive engine makes most often and
        the one whose meaning is worth stating in the API.
        """
        return self.activities_for_objective(objective_id, "remediation")

    def concepts_for_question(self, question_id: str) -> List[Concept]:
        package = self.package_of(question_id)
        if package is None:
            return []
        return package.content.concepts_for_question(question_id)

    def prerequisites_of(self, entity_id: str) -> List[str]:
        package = self.package_of(entity_id)
        return package.content.prerequisites_of(entity_id) if package else []

    def dependents_of(self, entity_id: str) -> List[str]:
        package = self.package_of(entity_id)
        return package.content.dependents_of(entity_id) if package else []

    def related_to(self, entity_id: str) -> List[str]:
        package = self.package_of(entity_id)
        return package.content.related_to(entity_id) if package else []

    def summary(self) -> Dict[str, object]:
        """What is loaded, for a dashboard or a startup log."""
        return {
            "packages": len(self._packages),
            "grades": self.grades(),
            "entities": len(self._index),
            "by_package": {
                package.package_id: package.stats.model_dump()
                for package in self.packages()
            },
        }
