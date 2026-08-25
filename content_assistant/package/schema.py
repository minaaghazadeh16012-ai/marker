"""The Content Package: one book's content as a single loadable file.

Everything upstream of here works in per-lesson stage artifacts, which is right
for a pipeline and wrong for a consumer. An adaptive engine does not want
fourteen directories of intermediate JSON; it wants one file it can load,
check, and ask questions of. That file is a Content Package.

A package is *derived*, never authored. It is assembled from stage artifacts by
:mod:`content_assistant.package.build`, and everything in it can be rebuilt
from them - which is what makes overwriting one safe and makes a diff between
two runs mean something.

Two properties are enforced rather than documented:

**Stats are recomputed on load.** Counts stored in a file are a summary of what
was in it *when it was written*. If the two disagree the file has been edited
outside the builder, and :func:`load_content` says so rather than reporting the
numbers the file would like to be true.

**A version is checked before anything is read.** See
:mod:`content_assistant.package.migrate`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from content_assistant.models.common import SCHEMA_VERSION
from content_assistant.models.content import ContentSchema
from content_assistant.package.migrate import upgrade_payload

#: Bumped when the *builder* changes how it assembles a package, even if no
#: field changed. Two packages built from the same artifacts by different
#: builders can differ, and this is how anyone finds out which built which.
BUILDER_VERSION = "1.0.0"

#: The file name a package is always written to inside its directory, so a
#: registry can find packages without being told where each one is.
PACKAGE_FILENAME = "content-package.json"


class ContentPackageError(RuntimeError):
    """A package on disk cannot be trusted to be what it says it is."""


class PackageStats(BaseModel):
    """What is in the package, counted. Derived - never a source of truth.

    Kept because the first question anyone asks of a package is "how much is in
    here, and how much of it is verified?", and answering it should not require
    loading and traversing the whole thing. Every field is recomputed by
    :func:`compute_stats`, so a stale count is a detectable fault rather than a
    quiet lie.
    """

    lessons: int = 0
    sections: int = 0
    concepts: int = 0
    objectives: int = 0
    skills: int = 0
    misconceptions: int = 0
    relations: int = 0
    activities: int = 0
    questions: int = 0
    evidence: int = 0
    #: Of the evidence records, how many had their quotation found in the block
    #: they attributed it to. The single most useful number about a package.
    quotations_verified: int = 0
    #: Entities the pipeline decided a person has to look at.
    awaiting_review: int = 0
    #: Of those, how many a person has since decided about.
    reviewed: int = 0


def compute_stats(content: ContentSchema) -> PackageStats:
    """Count what is actually there, right now."""
    reviewable = (
        list(content.concepts)
        + list(content.objectives)
        + list(content.skills)
        + list(content.misconceptions)
        + list(content.relations)
        + list(content.activities)
        + list(content.questions)
    )
    return PackageStats(
        lessons=len(content.lessons),
        sections=len(content.sections),
        concepts=len(content.concepts),
        objectives=len(content.objectives),
        skills=len(content.skills),
        misconceptions=len(content.misconceptions),
        relations=len(content.relations),
        activities=len(content.activities),
        questions=len(content.questions),
        evidence=len(content.evidence),
        quotations_verified=sum(
            1 for item in content.evidence if item.quote_verified
        ),
        awaiting_review=sum(
            1
            for entity in reviewable
            if entity.requires_human_review
            and entity.review_status == "pending"
        ),
        reviewed=sum(
            1 for entity in reviewable if entity.review_status != "pending"
        ),
    )


class ContentPackage(BaseModel):
    """One (grade, subject) book, whole and self-contained."""

    #: ``grade-1:science:g1-olom``. Deterministic, readable, and stable across
    #: rebuilds - it is composed of facts about the book rather than of
    #: anything a particular run decided.
    package_id: str
    grade: int
    subject: str
    language: str = "fa"
    book_id: str
    #: The book's own title, when the extraction recorded one.
    title: Optional[str] = None
    #: Checksum of the source PDF, carried up from L0 so a package can be
    #: matched back to the file it came from.
    source_sha256: Optional[str] = None
    #: ISO-8601, stamped once per package rather than once per entity. Two
    #: rebuilds from identical artifacts differ in this field and nowhere else,
    #: which is exactly what makes a diff readable.
    built_at: Optional[str] = None
    builder_version: str = BUILDER_VERSION
    stats: PackageStats = Field(default_factory=PackageStats)
    content: ContentSchema

    @property
    def content_schema_version(self) -> str:
        """The schema version this package is written against.

        Read through to the content rather than stored twice: one number, one
        place, and no way for a package header to claim a version its body
        does not hold.
        """
        return self.content.schema_version

    @staticmethod
    def build_id(grade: int, subject: str, book_id: str) -> str:
        return f"grade-{grade}:{subject}:{book_id}"

    def stats_are_current(self) -> bool:
        return self.stats == compute_stats(self.content)


def default_path(root: Path, grade: int, subject: str) -> Path:
    """Where a package for this grade and subject lives under ``root``.

    ``<root>/grade-1/science/content-package.json``. The layout is a
    convention rather than a requirement - :func:`load_content` reads any path
    - but it is the one :class:`~content_assistant.package.registry`
    scans for, and following it is what makes a package discoverable.
    """
    return Path(root) / f"grade-{grade}" / subject / PACKAGE_FILENAME


def save_content(package: ContentPackage, path: Path) -> Path:
    """Write a package, recomputing its stats first.

    Recomputing on the way out is what makes the check on the way in
    meaningful: the writer can never be the reason the two disagree.
    """
    package = package.model_copy(
        update={"stats": compute_stats(package.content)}
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return path


def load_content(path: Path, expected_version: str = SCHEMA_VERSION) -> ContentPackage:
    """Read a package from disk, deterministically.

    Three things happen, in this order, and each can refuse: the version is
    checked and upgraded if it is an older minor, the payload is validated
    against the model, and the stored counts are compared against the content
    they claim to describe.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContentPackageError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContentPackageError(f"{path} does not hold a content package")

    payload = upgrade_payload(payload, expected_version)
    package = ContentPackage.model_validate(payload)

    actual = compute_stats(package.content)
    if package.stats != actual:
        differences = _stat_differences(package.stats, actual)
        raise ContentPackageError(
            f"{path} states counts that do not match its content "
            f"({differences}); it was edited outside the builder and its "
            "summary can no longer be trusted. Rebuild it."
        )
    return package


def _stat_differences(stated: PackageStats, actual: PackageStats) -> str:
    stated_map: Dict[str, int] = stated.model_dump()
    actual_map: Dict[str, int] = actual.model_dump()
    parts = [
        f"{field}: says {stated_map[field]}, holds {actual_map[field]}"
        for field in stated_map
        if stated_map[field] != actual_map[field]
    ]
    return "; ".join(parts)
