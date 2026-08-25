"""Whether a stored package can be read by this code, and what to do if not.

A schema that changes without a migration story quietly invalidates everything
already written against it. This module is the single place allowed to answer
"can I read this?", and it answers in three ways rather than two - the third
being the one that is usually forgotten.

============================  =====================================
stored version                what happens
============================  =====================================
same major, same minor        read as-is
same major, older minor       upgraded: pydantic fills the fields
                              added since, and the version is
                              re-stamped on save
same major, **newer** minor   refused
different major               refused
============================  =====================================

The third row is the important one. Reading a package written by newer code
would work - unknown fields are simply dropped - and saving it again would
then delete them without a word. Refusing is the only behaviour that cannot
silently destroy someone's data.
"""

from __future__ import annotations

from typing import Dict, Tuple

from content_assistant.models.common import SCHEMA_VERSION


class SchemaVersionError(RuntimeError):
    """A stored artifact cannot be read by this version of the code."""


def parse_version(value: str) -> Tuple[int, int, int]:
    """``"1.2.3"`` -> ``(1, 2, 3)``. Anything else is not a version."""
    parts = (value or "").split(".")
    if len(parts) != 3:
        raise SchemaVersionError(
            f"{value!r} is not a schema version; expected major.minor.patch"
        )
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise SchemaVersionError(
            f"{value!r} is not a schema version; expected major.minor.patch"
        ) from exc


def check_version(stored: str, current: str = SCHEMA_VERSION) -> str:
    """Return the version the payload should be read as, or refuse.

    The patch number is deliberately ignored in both directions: by definition
    it marks a change that alters no field, so a package differing only there
    needs nothing done to it.
    """
    stored_parts = parse_version(stored)
    current_parts = parse_version(current)

    if stored_parts[0] != current_parts[0]:
        raise SchemaVersionError(
            f"package is schema {stored}, this code reads {current}; a major "
            "version differs, which means a field changed meaning rather than "
            "being added. Rebuild the package from its run artifacts."
        )
    if stored_parts[1] > current_parts[1]:
        raise SchemaVersionError(
            f"package is schema {stored}, this code reads {current}; it was "
            "written by newer code and holds fields this version would drop "
            "on the next save. Upgrade content_assistant instead."
        )
    return current


def upgrade_payload(payload: Dict, current: str = SCHEMA_VERSION) -> Dict:
    """Bring a decoded package payload up to the current schema version.

    Upgrading within a major version is exactly "let the model fill in its own
    defaults", because every addition since 1.0.0 is optional. The work here is
    therefore checking that the upgrade is *allowed* and re-stamping the
    version; the field-filling happens when pydantic validates the result.

    A future change that needs real rewriting adds its step here, keyed off the
    stored version - and gets a test that reads a fixture written in the old
    shape, because a migration proved only against data this code produced is
    not proved.
    """
    content = payload.get("content")
    if not isinstance(content, dict):
        raise SchemaVersionError(
            "package has no 'content' object; it is not a content package"
        )
    stored = content.get("schema_version", "")
    check_version(stored, current)
    upgraded = dict(payload)
    upgraded["content"] = {**content, "schema_version": current}
    return upgraded
