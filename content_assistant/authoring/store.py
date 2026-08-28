"""Where authored content lives between authoring it and building a package.

One file per book, beside the package it feeds:

    content/grade-1/science/authored-content.json
    content/grade-1/science/content-package.json

Two properties are worth stating, because both are choices.

**One file, not four.** Skills, relations, activities and questions could each
have their own, and then four files would each carry a ``book_id`` free to
disagree with the others. They belong to one book and are written by one
person; one file is the shape of that.

**It is source, not output.** Everything else in this package is derived and
can be rebuilt from L0 plus the stage artifacts. This cannot: it is somebody's
judgement, and nothing regenerates it. :func:`save_authored` therefore never
writes over a file it did not read - see the ``overwrite`` argument - because
losing this file loses work no re-run brings back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from content_assistant.models.common import SCHEMA_VERSION
from content_assistant.models.content import Relation, Skill
from content_assistant.models.learning import LearningActivity, Question
from content_assistant.package.migrate import check_version

#: The file name authored content is always written to inside its directory,
#: so a builder can find it without being told where each book keeps it.
AUTHORED_FILENAME = "authored-content.json"


class AuthoredContentError(RuntimeError):
    """Authored content on disk cannot be read as what it claims to be."""


class AuthoredContent(BaseModel):
    """Everything a person has authored for one book.

    Empty lists are the normal state and the honest one. A book whose skills
    have not been authored yet has no skills, and saying so is the difference
    between a package that is incomplete and one that is padded.
    """

    schema_version: str = SCHEMA_VERSION
    #: Which book these records belong to. Checked against the package on
    #: build: every id in here is derived from a book id, so authored content
    #: pointed at the wrong book would produce references that resolve to
    #: nothing, one entity at a time, with no single obvious failure.
    book_id: str
    #: Free text, for the person keeping the file: what round of authoring this
    #: is, what is still missing, who to ask.
    notes: str = ""
    skills: List[Skill] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    activities: List[LearningActivity] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.skills or self.relations or self.activities or self.questions
        )

    def counts(self) -> dict:
        return {
            "skills": len(self.skills),
            "relations": len(self.relations),
            "activities": len(self.activities),
            "questions": len(self.questions),
        }

    def entity_ids(self) -> List[str]:
        return (
            [s.id for s in self.skills]
            + [r.id for r in self.relations]
            + [a.id for a in self.activities]
            + [q.id for q in self.questions]
        )


def authored_path(root: Path, grade: int, subject: str) -> Path:
    """Where authored content for this grade and subject lives under ``root``.

    Deliberately the same layout as
    :func:`~content_assistant.package.schema.default_path`, so the source and
    the artifact built from it sit side by side and neither can be moved
    without the other being noticed.
    """
    return Path(root) / f"grade-{grade}" / subject / AUTHORED_FILENAME


def resolve_authored_path(path: Path) -> Path:
    """Accept either the file itself or the directory holding it."""
    path = Path(path)
    if path.is_dir():
        return path / AUTHORED_FILENAME
    return path


def load_authored(
    path: Path, expected_book_id: Optional[str] = None
) -> AuthoredContent:
    """Read authored content, or return an empty store if there is none.

    A missing file is not an error: a book nobody has authored for yet is the
    ordinary case, and forcing every caller to create an empty file first would
    only produce empty files. What *is* an error is a file that exists and
    cannot be trusted - unreadable, written against an unreadable schema
    version, or belonging to another book.
    """
    path = resolve_authored_path(path)
    if not path.exists():
        return AuthoredContent(book_id=expected_book_id or "")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthoredContentError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuthoredContentError(f"{path} does not hold authored content")

    # The same rule the packages are read by, and for the same reason: content
    # written by newer code holds fields this version would drop on the next
    # save, and this is the one file where a dropped field is lost work.
    check_version(payload.get("schema_version", SCHEMA_VERSION))
    payload = {**payload, "schema_version": SCHEMA_VERSION}
    store = AuthoredContent.model_validate(payload)

    if expected_book_id and store.book_id and store.book_id != expected_book_id:
        raise AuthoredContentError(
            f"{path} holds content authored for {store.book_id!r}, but it is "
            f"being built into {expected_book_id!r}; every id in it was "
            "derived from the other book and would resolve to nothing"
        )
    return store


def save_authored(
    store: AuthoredContent, path: Path, *, overwrite: bool = False
) -> Path:
    """Write authored content, refusing to clobber by default.

    Nothing regenerates this file. A builder overwriting a package is fine
    because the package can be rebuilt; overwriting this loses the judgement
    that went into it, so the caller has to say so.
    """
    path = resolve_authored_path(path)
    if path.exists() and not overwrite:
        raise AuthoredContentError(
            f"{path} already exists and nothing can regenerate it; read it, "
            "merge into it, and save with overwrite=True"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(store.model_dump(mode="json"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return path
