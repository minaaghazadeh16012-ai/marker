"""Evidence Units: the only thing a model is ever allowed to see.

A model in this pipeline never receives a PDF, a page image dump, or free text.
It receives an Evidence Unit - one lesson, already segmented, with every block
labelled by the id it must cite. That single design choice does most of the
work of keeping output grounded:

* the model cannot invent a page number, because it is never asked for one;
* the model cannot cite a source outside the lesson, because the only ids it
  has are the ones handed to it, and anything else is rejected on arrival;
* the unit records what it contains, so a thin lesson is visibly thin instead
  of silently producing thin results.

Images travel with the unit as references. Whether they are actually rendered
and sent is the caller's decision, taken from ``material_profile.text_density``
- a lesson carrying a few hundred characters against thirty pictures cannot be
read from its text alone, and pretending otherwise produces confident nonsense.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from content_assistant.models.content import Lesson, MaterialProfile, Section
from content_assistant.models.extraction import Block, ExtractionResult, Page


class EvidenceBlock(BaseModel):
    """One citable unit of text, as the model sees it."""

    block_id: str
    block_type: str
    text: str
    pdf_page: int
    printed_page: Optional[int] = None
    source: str = "marker"


class EvidenceImage(BaseModel):
    asset_id: str
    pdf_page: int
    printed_page: Optional[int] = None
    path: str = ""
    #: Nearby caption text, when the book printed one.
    caption: str = ""


class EvidenceSection(BaseModel):
    section_id: str
    title: str
    order: int
    blocks: List[EvidenceBlock] = Field(default_factory=list)
    images: List[EvidenceImage] = Field(default_factory=list)


class EvidenceUnit(BaseModel):
    """Everything a model is given for one lesson, and nothing else."""

    book_id: str
    grade: int
    subject: str
    language: str = "fa"
    lesson_id: str
    lesson_number: int
    lesson_title: str
    printed_page_start: Optional[int] = None
    printed_page_end: Optional[int] = None
    material_profile: MaterialProfile = Field(default_factory=MaterialProfile)
    sections: List[EvidenceSection] = Field(default_factory=list)

    def citable_block_ids(self) -> set:
        """The complete set of ids a model may cite for this lesson."""
        return {
            block.block_id
            for section in self.sections
            for block in section.blocks
        }

    def citable_asset_ids(self) -> set:
        return {
            image.asset_id
            for section in self.sections
            for image in section.images
        }

    def block_text(self) -> Dict[str, str]:
        return {
            block.block_id: block.text
            for section in self.sections
            for block in section.blocks
        }

    def needs_images(self, low_density_only: bool = True) -> bool:
        """Whether this lesson should be read with its pictures.

        Decided from the measured profile, not from a hunch: a low-density
        lesson has more meaning in its illustrations than in its sentences.
        """
        if not low_density_only:
            return bool(self.material_profile.images)
        return (
            self.material_profile.text_density == "low"
            and self.material_profile.images > 0
        )


#: Block types worth citing. Running heads and feet are excluded: a page number
#: is not evidence for anything a lesson teaches.
CITABLE_TYPES = frozenset(
    {"Text", "RecoveredText", "SectionHeader", "Caption", "ListGroup", "ListItem"}
)


def _index_blocks(result: ExtractionResult) -> Dict[str, tuple]:
    index: Dict[str, tuple] = {}
    for page in result.pages:
        for block in page.blocks:
            index[block.block_id] = (page, block)
    return index


def _index_assets(result: ExtractionResult) -> Dict[str, tuple]:
    index: Dict[str, tuple] = {}
    for page in result.pages:
        for asset in page.assets:
            index[asset.asset_id] = (page, asset)
    return index


def _nearest_caption(page: Page, bbox: Sequence[float]) -> str:
    """Caption printed closest below an image, if any.

    Captions are the one textual handle on a picture that the book itself
    provides, so they travel with the image rather than being left in the block
    stream where their subject is unclear.
    """
    captions = [
        b
        for b in page.blocks
        if b.type == "Caption" and b.text.strip()
    ]
    if not captions:
        return ""
    below = [c for c in captions if c.bbox[1] >= bbox[3] - 1]
    pool = below or captions
    nearest = min(pool, key=lambda c: abs(c.bbox[1] - bbox[3]))
    return nearest.text.strip()


def build_evidence_unit(
    result: ExtractionResult,
    lesson: Lesson,
    sections: Sequence[Section],
) -> EvidenceUnit:
    """Assemble the packet for one lesson from the L0 artifact."""
    blocks = _index_blocks(result)
    assets = _index_assets(result)
    book = result.document.book

    unit_sections: List[EvidenceSection] = []
    for section in sorted(
        (s for s in sections if s.lesson_id == lesson.id), key=lambda s: s.order
    ):
        evidence_blocks: List[EvidenceBlock] = []
        for block_id in section.block_ids:
            found = blocks.get(block_id)
            if not found:
                continue
            page, block = found
            if block.type not in CITABLE_TYPES or not block.text.strip():
                continue
            evidence_blocks.append(
                EvidenceBlock(
                    block_id=block.block_id,
                    block_type=block.type,
                    text=block.text,
                    pdf_page=page.pdf_page,
                    printed_page=page.printed_page,
                    source=block.source,
                )
            )

        evidence_images: List[EvidenceImage] = []
        for asset_id in section.asset_ids:
            found = assets.get(asset_id)
            if not found:
                continue
            page, asset = found
            evidence_images.append(
                EvidenceImage(
                    asset_id=asset.asset_id,
                    pdf_page=page.pdf_page,
                    printed_page=page.printed_page,
                    path=asset.path,
                    caption=_nearest_caption(page, asset.bbox),
                )
            )

        unit_sections.append(
            EvidenceSection(
                section_id=section.id,
                title=section.title,
                order=section.order,
                blocks=evidence_blocks,
                images=evidence_images,
            )
        )

    return EvidenceUnit(
        book_id=lesson.book_id,
        grade=lesson.grade,
        subject=lesson.subject,
        language=book.language,
        lesson_id=lesson.id,
        lesson_number=lesson.lesson_number,
        lesson_title=lesson.title,
        printed_page_start=lesson.page_range.printed_start,
        printed_page_end=lesson.page_range.printed_end,
        material_profile=lesson.material_profile,
        sections=unit_sections,
    )


def build_evidence_units(
    result: ExtractionResult,
    lessons: Sequence[Lesson],
    sections: Sequence[Section],
) -> List[EvidenceUnit]:
    return [build_evidence_unit(result, lesson, sections) for lesson in lessons]
