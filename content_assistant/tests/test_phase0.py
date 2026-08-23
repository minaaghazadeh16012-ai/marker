"""Regression tests for the three L0 gaps closed in phase 0.

Each test states the defect it locks out, because all three were found by
auditing a real 104-page run rather than by reasoning about the code:

1. asset paths were written with the host separator, so a Content Package
   produced on Windows could not be read anywhere else;
2. the 179 rescued blocks carried no ``section_hierarchy``, orphaning exactly
   the text that was hardest to recover from the heading chain;
3. the document carried no identity - no book, grade, subject or language -
   so nothing downstream could mint a stable id or enforce a grade boundary.

Backward compatibility is part of the contract: an artifact written before
these changes must still load.
"""

from __future__ import annotations

import unittest

from content_assistant.extraction.marker_backend import MarkerBlock, MarkerPage
from content_assistant.extraction.pipeline import _section_hierarchy_resolver
from content_assistant.models.extraction import (
    BookIdentity,
    DocumentInfo,
    ExtractionResult,
    Page,
)


def block(block_id: str, y0: float, hierarchy=None) -> MarkerBlock:
    return MarkerBlock(
        block_id=block_id,
        type="Text",
        text="x",
        bbox=[0.0, y0, 100.0, y0 + 20.0],
        polygon=[[0, y0], [100, y0], [100, y0 + 20], [0, y0 + 20]],
        section_hierarchy=hierarchy,
    )


class AssetPathPortabilityTests(unittest.TestCase):
    """Gap 1: asset paths must be POSIX so a package travels between hosts."""

    def test_asset_path_uses_forward_slashes(self):
        from pathlib import PurePosixPath, PureWindowsPath

        # The pipeline stores Path(...).relative_to(out).as_posix(); this is
        # that expression's contract, checked on a Windows-style path.
        windows = PureWindowsPath(r"C:\\work\\assets\\page_44_Picture_0.jpeg")
        relative = windows.relative_to(PureWindowsPath(r"C:\\work"))
        self.assertEqual(relative.as_posix(), "assets/page_44_Picture_0.jpeg")
        self.assertNotIn("\\", relative.as_posix())
        self.assertEqual(
            PurePosixPath(relative.as_posix()).name, "page_44_Picture_0.jpeg"
        )


class RecoveredSectionHierarchyTests(unittest.TestCase):
    """Gap 2: rescued lines must inherit the headings they sit under."""

    def test_line_inherits_the_nearest_heading_above_it(self):
        blocks = [
            block("/page/5/Text/0", 100.0, {"2": "/page/4/SectionHeader/0"}),
            block("/page/5/Text/1", 300.0, {"2": "/page/5/SectionHeader/1"}),
        ]
        resolve = _section_hierarchy_resolver(blocks, None)
        self.assertEqual(resolve([0, 350, 100, 370]), {"2": "/page/5/SectionHeader/1"})
        self.assertEqual(resolve([0, 150, 100, 170]), {"2": "/page/4/SectionHeader/0"})

    def test_line_above_every_block_takes_the_first_heading_on_the_page(self):
        blocks = [block("/page/5/Text/0", 300.0, {"2": "/page/5/SectionHeader/0"})]
        resolve = _section_hierarchy_resolver(blocks, None)
        self.assertEqual(resolve([0, 10, 100, 30]), {"2": "/page/5/SectionHeader/0"})

    def test_page_without_any_heading_carries_the_previous_page_forward(self):
        resolve = _section_hierarchy_resolver([], {"2": "/page/3/SectionHeader/0"})
        self.assertEqual(resolve([0, 10, 100, 30]), {"2": "/page/3/SectionHeader/0"})

    def test_no_heading_anywhere_yields_none_rather_than_an_empty_dict(self):
        # None means "unknown"; {} would claim the block sits under no heading.
        self.assertIsNone(_section_hierarchy_resolver([], None)([0, 0, 1, 1]))

    def test_resolver_returns_a_copy_not_a_shared_reference(self):
        shared = {"2": "/page/4/SectionHeader/0"}
        resolve = _section_hierarchy_resolver(
            [block("/page/5/Text/0", 10.0, shared)], None
        )
        result = resolve([0, 50, 100, 70])
        result["2"] = "mutated"
        self.assertEqual(shared["2"], "/page/4/SectionHeader/0")


class BookIdentityTests(unittest.TestCase):
    """Gap 3: identity is declared, never inferred."""

    def test_identity_is_carried_on_the_document(self):
        doc = DocumentInfo(
            source="olom.pdf",
            source_sha256="abc",
            page_count=104,
            book=BookIdentity(
                book_id="g1-olom", grade=1, subject="science", language="fa"
            ),
        )
        self.assertEqual(doc.book.book_id, "g1-olom")
        self.assertEqual(doc.book.grade, 1)
        self.assertEqual(doc.book.language, "fa")

    def test_language_defaults_to_persian(self):
        self.assertEqual(BookIdentity().language, "fa")

    def test_identity_is_optional_so_older_artifacts_still_load(self):
        # An artifact written before phase 0 has no "book" key at all.
        legacy = {
            "document": {
                "source": "olom.pdf",
                "source_sha256": "abc",
                "page_count": 104,
                "page_offset": 0,
            },
            "pages": [],
            "toc": [],
        }
        result = ExtractionResult.model_validate(legacy)
        self.assertEqual(result.document.page_count, 104)
        self.assertIsNone(result.document.book.book_id)

    def test_legacy_pages_still_validate(self):
        legacy_page = {
            "pdf_page": 45,
            "pdf_page_index": 44,
            "printed_page": 45,
            "printed_page_source": "page_footer",
            "blocks": [
                {
                    "block_id": "/page/44/Text/1",
                    "type": "Text",
                    "text": "x",
                    "bbox": [0, 0, 1, 1],
                    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "source": "marker",
                }
            ],
            "assets": [],
        }
        page = Page.model_validate(legacy_page)
        self.assertEqual(page.blocks[0].source, "marker")
        self.assertIsNone(page.diagnostics)


if __name__ == "__main__":
    unittest.main()
