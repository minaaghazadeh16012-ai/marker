"""Typed model for the L0 extraction output.

Design rules for this phase:
  * Nothing here is educational content. No Concept / Objective / Skill / Lesson.
    L0 answers one question only: *what text, where, and where did it come from?*
  * Every block carries its provenance (``source``) so a later layer can tell
    Marker's own output apart from text this pipeline had to rescue.
  * Page numbering keeps three separate fields on purpose - they are three
    different numbers and conflating them is the classic off-by-one bug:
      ``pdf_page_index``  0-based, exactly Marker's ``page_id``
      ``pdf_page``        1-based, what a human sees in a PDF reader
      ``printed_page``    the number printed on the paper, or ``None``
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

BBox = List[float]  # [x0, y0, x1, y1]
Polygon = List[List[float]]

BlockSource = Literal["marker", "pdfprovider_recovery"]
PrintedPageSource = Literal["page_footer", "inferred_from_offset"]


# ---------------------------------------------------------------------------
# Configuration - every threshold in the pipeline lives here, nowhere else.
# ---------------------------------------------------------------------------


class ExtractionConfig(BaseModel):
    """Single source of truth for every tunable threshold.

    Thresholds are deliberately gathered in one model instead of being spread
    through the modules: the diagnostics that *select* pages for recovery and
    the geometry that *performs* it have to agree, and that is only checkable
    when the numbers sit next to each other.
    """

    # -- page diagnostics: when is a page a recovery candidate? --------------
    picture_area_frac_min: float = Field(
        0.30,
        description="Page is a candidate when picture/figure blocks cover at "
        "least this fraction of the page area.",
    )
    recovery_ratio_min: float = Field(
        0.80,
        description="Page is a candidate when marker_chars/raw_chars falls "
        "below this.",
    )

    # -- duplicate suppression ----------------------------------------------
    duplicate_overlap_min: float = Field(
        0.50,
        description="A raw line counts as already-extracted when this fraction "
        "of its own area is covered by an existing text block.",
    )

    # -- reading order -------------------------------------------------------
    row_overlap_min: float = Field(
        0.50,
        description="Two lines share a row when their vertical overlap reaches "
        "this fraction of the shorter line's height.",
    )

    # -- decorative (curved / per-glyph) page handling -----------------------
    decorative_single_char_span_ratio: float = Field(
        0.50,
        description="A page is 'decorative' when at least this fraction of its "
        "spans hold a single character.",
    )
    decorative_min_lines: int = Field(
        10, description="Minimum raw lines before the decorative test applies."
    )
    decorative_big_number_factor: float = Field(
        3.0,
        description="A numeric line is an anchor ('big number') when its height "
        "is at least this many times the page's median line height.",
    )
    toc_min_anchors: int = Field(
        3,
        description="A decorative page is a table of contents when it holds at "
        "least this many big-number anchors.",
    )

    # -- noise filtering -----------------------------------------------------
    min_line_height: float = Field(
        2.0,
        description="Lines shorter than this are layout artefacts (stray "
        "newline glyphs) and are dropped.",
    )


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class Asset(BaseModel):
    """An image Marker cropped out of the page. Never analysed in this phase."""

    asset_id: str
    pdf_page: int
    bbox: BBox
    path: str
    type: str = "image"
    format: str = "jpeg"


class Block(BaseModel):
    block_id: str
    type: str
    text: str
    bbox: BBox
    polygon: Polygon
    source: BlockSource
    #: text exactly as it came out of Marker/pdftext, before normalization
    text_raw: Optional[str] = None
    section_hierarchy: Optional[Dict[str, str]] = None
    asset_ids: List[str] = Field(default_factory=list)


class PageDiagnostics(BaseModel):
    pdf_page: int
    raw_chars: int
    marker_chars: int
    recovery_ratio: Optional[float]
    raw_lines: int
    marker_text_blocks: int
    picture_area_frac: float
    has_picture: bool
    has_picture_group: bool
    has_page_header: bool
    has_page_footer: bool
    is_decorative: bool = False
    is_recovery_candidate: bool = False
    candidate_reasons: List[str] = Field(default_factory=list)
    recovered_lines: int = 0
    duplicates_skipped: int = 0
    recovered_chars: int = 0


class Page(BaseModel):
    pdf_page: int
    pdf_page_index: int
    printed_page: Optional[int] = None
    printed_page_source: Optional[PrintedPageSource] = None
    blocks: List[Block] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)
    diagnostics: Optional[PageDiagnostics] = None


class TocEntry(BaseModel):
    """One row rebuilt from a decorative contents page.

    Nothing here is hard-coded: the lesson number, the title and the printed
    page all come out of the PDF's own text layer.
    """

    lesson_number: Optional[int] = None
    title: str = ""
    printed_page: Optional[int] = None
    source_pdf_page: int = 0
    #: Titles set along a curve can be split mid-word; the page mapping is
    #: exact either way, so the flag says which half to trust.
    title_is_approximate: bool = False


class PageOffsetEvidence(BaseModel):
    value: Optional[int] = None
    samples: int = 0
    agreement: Optional[float] = None
    method: str = "page_footer_majority"


class DocumentInfo(BaseModel):
    source: str
    source_sha256: str
    page_count: int
    page_offset: Optional[int] = None
    page_offset_evidence: PageOffsetEvidence = Field(
        default_factory=PageOffsetEvidence
    )
    marker_version: Optional[str] = None
    extractor_version: str = "0.1.0"


class ExtractionResult(BaseModel):
    document: DocumentInfo
    pages: List[Page] = Field(default_factory=list)
    toc: List[TocEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Raw text-layer structures (input to diagnostics/recovery, not output)
# ---------------------------------------------------------------------------


class RawLine(BaseModel):
    """One line as pdftext produced it, before Marker's layout classified it.

    This is the material recovery works from: Marker never loses the text, it
    only sometimes fails to place it in a rendered block.
    """

    text: str
    bbox: BBox
    n_spans: int = 0
    n_single_char_spans: int = 0

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


class RawPage(BaseModel):
    pdf_page_index: int
    lines: List[RawLine] = Field(default_factory=list)


#: Marker block types that are expected to carry text in the rendered output.
TEXT_BLOCK_TYPES = frozenset(
    {
        "Text",
        "TextInlineMath",
        "SectionHeader",
        "ListItem",
        "ListGroup",
        "Caption",
        "Footnote",
        "PageHeader",
        "PageFooter",
        "Code",
        "Table",
        "TableOfContents",
        "Form",
        "Reference",
        "Equation",
        "Handwriting",
        "ComplexRegion",
    }
)

#: Marker block types that render as an image and swallow any text inside them.
PICTURE_BLOCK_TYPES = frozenset({"Picture", "Figure", "Diagram"})
GROUP_BLOCK_TYPES = frozenset({"PictureGroup", "FigureGroup", "TableGroup"})
