"""Data models for the Content Assistant.

Only the L0 extraction types are re-exported here. The content schema is large
enough that a flat namespace over it would hide which layer a type belongs to,
which is the one thing its readers most need to see - so import those from the
module that owns them:

``common``
    identity, :class:`~content_assistant.models.common.Provenance`, review.
``content``
    evidence, structure, concepts, objectives, skills, relations, and the
    assembled :class:`~content_assistant.models.content.ContentSchema`.
``learning``
    the learning-experience layer: activities and questions.
``objective``
    the closed performance-verb lexicon, versioned on its own.
"""

from content_assistant.models.extraction import (  # noqa: F401
    Asset,
    BBox,
    Block,
    BlockSource,
    DocumentInfo,
    ExtractionConfig,
    ExtractionResult,
    GROUP_BLOCK_TYPES,
    Page,
    PageDiagnostics,
    PageOffsetEvidence,
    PICTURE_BLOCK_TYPES,
    Polygon,
    RawLine,
    RawPage,
    TEXT_BLOCK_TYPES,
    TocEntry,
)
