"""Thin wrapper around the installed Marker CLI.

Marker is the extraction engine and is used exactly as shipped: this module
shells out to ``marker_single`` and parses what comes back. Nothing in the
``marker/`` package is imported for the conversion itself, nothing is patched,
and no Marker source file is touched.

The flag set is fixed on purpose:

``--mode fast``
    The VLM layout path needs an inference server; the lightweight detector
    does not, and this phase runs without any backend.
``--disable_ocr``
    The books targeted here have a healthy text layer. Turning OCR off keeps
    the pipeline deterministic and free of a model server.
``--keep_pageheader_in_output`` / ``--keep_pagefooter_in_output``
    Marker strips running heads and feet by default. Those carry the printed
    page number and, on lesson-opening pages, the lesson title - the two things
    the page map and the structure layer need most.
``--output_format json``
    Only the JSON renderer exposes block ids, bboxes and section hierarchy.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_MARKER_FLAGS: Sequence[str] = (
    "--mode",
    "fast",
    "--disable_ocr",
    "--keep_pageheader_in_output",
    "--keep_pagefooter_in_output",
    "--output_format",
    "json",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(fragment: Optional[str]) -> str:
    """Flatten a Marker HTML fragment to plain text.

    Deliberately regex-based rather than a parser: the fragments are Marker's
    own output, they are small, and this keeps the module free of a parsing
    dependency.
    """
    if not fragment:
        return ""
    text = _TAG_RE.sub(" ", fragment)
    return _WS_RE.sub(" ", html_lib.unescape(text)).strip()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class MarkerBlock:
    """One leaf block of Marker's JSON output."""

    block_id: str
    type: str
    text: str
    bbox: List[float]
    polygon: List[List[float]]
    section_hierarchy: Optional[Dict[str, str]] = None
    images: Dict[str, str] = field(default_factory=dict)

    @property
    def pdf_page_index(self) -> int:
        # ids look like "/page/44/Text/5"
        return int(self.block_id.split("/")[2])


@dataclass
class MarkerPage:
    pdf_page_index: int
    bbox: List[float]
    polygon: List[List[float]]
    blocks: List[MarkerBlock]


@dataclass
class MarkerRun:
    json_path: Path
    meta_path: Path
    work_dir: Path
    pages: List[MarkerPage]
    meta: Dict
    cached: bool = False


def _iter_leaf_blocks(node: Dict):
    """Yield leaf blocks; groups only hold ``<content-ref>`` placeholders."""
    children = node.get("children") or []
    if not children:
        yield node
        return
    for child in children:
        yield from _iter_leaf_blocks(child)


class MarkerBackend:
    """Runs ``marker_single`` and parses its JSON output.

    Results are cached on disk keyed by the source checksum and the flag set,
    so re-running the pipeline never re-extracts an unchanged PDF - extraction
    is by far the slowest step and later layers are re-run far more often.
    """

    def __init__(
        self,
        executable: str,
        work_dir: Path,
        flags: Sequence[str] = DEFAULT_MARKER_FLAGS,
    ) -> None:
        self.executable = executable
        self.work_dir = Path(work_dir)
        self.flags = list(flags)

    def _cache_key(self, source_sha: str, page_range: Optional[str]) -> str:
        payload = source_sha + "|" + " ".join(self.flags) + "|" + (page_range or "all")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def run(
        self,
        pdf_path: Path,
        page_range: Optional[str] = None,
        force: bool = False,
    ) -> MarkerRun:
        pdf_path = Path(pdf_path)
        source_sha = sha256_of(pdf_path)
        key = self._cache_key(source_sha, page_range)
        out_dir = self.work_dir / f"marker-{key}"
        stem = pdf_path.stem
        json_path = out_dir / stem / f"{stem}.json"
        meta_path = out_dir / stem / f"{stem}_meta.json"

        cached = json_path.exists() and meta_path.exists() and not force
        if not cached:
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [self.executable, str(pdf_path), *self.flags]
            if page_range:
                cmd += ["--page_range", page_range]
            cmd += ["--output_dir", str(out_dir)]
            completed = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if completed.returncode != 0 or not json_path.exists():
                raise RuntimeError(
                    "marker_single failed "
                    f"(exit {completed.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            (out_dir / "marker_cmd.txt").write_text(
                " ".join(cmd), encoding="utf-8"
            )

        pages, meta = self.parse(json_path, meta_path)
        return MarkerRun(
            json_path=json_path,
            meta_path=meta_path,
            work_dir=out_dir,
            pages=pages,
            meta=meta,
            cached=cached,
        )

    @staticmethod
    def parse(json_path: Path, meta_path: Path):
        document = json.loads(Path(json_path).read_text(encoding="utf-8"))
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        pages: List[MarkerPage] = []
        for page_node in document.get("children") or []:
            page_index = int(page_node["id"].split("/")[2])
            blocks = [
                MarkerBlock(
                    block_id=node["id"],
                    type=node["block_type"],
                    text=html_to_text(node.get("html")),
                    bbox=list(node["bbox"]),
                    polygon=[list(p) for p in node["polygon"]],
                    section_hierarchy=node.get("section_hierarchy") or None,
                    images=node.get("images") or {},
                )
                for node in _iter_leaf_blocks(page_node)
                if node is not page_node
            ]
            pages.append(
                MarkerPage(
                    pdf_page_index=page_index,
                    bbox=list(page_node["bbox"]),
                    polygon=[list(p) for p in page_node["polygon"]],
                    blocks=blocks,
                )
            )
        pages.sort(key=lambda p: p.pdf_page_index)
        return pages, meta


def write_assets(pages: Sequence[MarkerPage], assets_dir: Path) -> Dict[str, Path]:
    """Write the base64 images Marker embedded in its JSON out to files.

    Images are never analysed in this phase - they are preserved so a later
    layer can look at them without re-running extraction.
    """
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for page in pages:
        for block in page.blocks:
            for name, payload in (block.images or {}).items():
                asset_id = name.strip("/").replace("/", "_")
                target = assets_dir / f"{asset_id}.jpeg"
                if not target.exists():
                    target.write_bytes(base64.b64decode(payload))
                written[name] = target
    return written
