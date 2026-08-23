# Content Assistant — L0 Extraction

Turns a raw textbook PDF into a structured, traceable extraction artifact.

This package sits **on top of** Marker. It imports Marker's public API and runs
its CLI; it never modifies, forks, or patches anything under `marker/`.

**L0 (extraction)** and the deterministic half of **L1 (structuring)** exist
today. There is still no Concept, Skill, Learning Objective, Misconception,
Index, or Knowledge Graph — and no model has been called — see
[Not built yet](#not-built-yet).

---

## Why no OCR

Because the text is already there.

Running Marker over all 104 pages of the grade-1 science book, every single
page reported `text_extraction_method = "pdftext"`. The PDF has a healthy text
layer and pdftext read all of it — 32,922 characters. OCR had nothing to add
that pdftext had missed.

What *was* missing from the rendered output was 12.4% of those characters, and
they went missing for a completely different reason: **layout classification**.
When Marker's layout model labels a region `Picture` or `PictureGroup`, every
text line whose box falls inside it is absorbed, and the block renders as an
image with no words. On this book that swallowed:

- the entire two-page contents spread (0 of 438 characters survived),
- seven lesson-opening titles,
- the top half of several activity pages.

So the deficit is a *placement* problem, not a *reading* problem. Turning on
OCR would spin up a VLM inference server to re-read text that pdftext had
already read correctly. Recovering it geometrically costs milliseconds and no
hardware. That is what this package does.

This decision is scoped, not permanent. A scanned book with no text layer will
need OCR, and Marker already supports that path — see
[When OCR will be needed](#when-ocr-will-be-needed).

## What Marker does

Everything that involves reading the PDF:

| Marker provides | Used for |
| --- | --- |
| page images, block detection, reading order | the page's block structure |
| `block_id`, `bbox`, `polygon` | source traceability |
| `PageHeader` / `PageFooter` blocks | the printed page number, lesson titles |
| `section_hierarchy` | heading nesting |
| cropped images (base64 in the JSON) | assets |
| `PdfProvider.page_lines` | the raw text layer, **before** layout |

The CLI is invoked with a fixed flag set (`marker_backend.DEFAULT_MARKER_FLAGS`):

```
--mode fast --disable_ocr
--keep_pageheader_in_output --keep_pagefooter_in_output
--output_format json
```

`--keep_page*_in_output` matters more than it looks. Marker strips running
heads and feet by default, and on this book that alone destroyed every printed
page number and every lesson title on a lesson-opening page.

## Why recovery is needed

Marker's JSON is the *rendered* view of the page. `PdfProvider.page_lines` is
the *raw* view — the same lines, each with its own polygon, before any layout
model saw them. When a line exists in the raw view but nothing in the rendered
view carries its text, that line was swallowed.

## How recovery works

Pure geometry. No model, no LLM, no second read of the PDF.

1. **Take the raw lines** for the page from `PdfProvider.page_lines`.
2. **Drop artifacts** — empty lines and the 1-pixel newline glyphs pdftext
   emits (`min_line_height`).
3. **Remove duplicates.** For each raw line, measure how much of *its own* area
   is covered by blocks that actually rendered text. At or above
   `duplicate_overlap_min`, Marker already has it. Blocks that rendered *empty*
   don't count as coverage — a blank `SectionHeader` has hidden its line, not
   published it.
4. **Order what's left** top-to-bottom, then right-to-left within each row,
   because the script is RTL (`sort_reading_order`).
5. **Emit** each survivor as a `RecoveredText` block with
   `source: "pdfprovider_recovery"`, so recovered text is never confused with
   Marker's own output.

## When a page is a candidate

A page enters recovery when **any** of three independent signals fires. Every
signal that fired is recorded in `diagnostics.candidate_reasons`, so the
decision is auditable per page.

| # | Signal | Default | Catches |
| --- | --- | --- | --- |
| 1 | `picture_area_frac >= 0.30` | 0.30 | pages where picture blocks cover enough area to swallow text |
| 2 | `marker_chars / raw_chars < 0.80` | 0.80 | pages that measurably lost characters |
| 3 | `marker_text_blocks == 0 and raw_lines > 0` | — | pages where the rendered view says nothing and the text layer says plenty |

Every threshold lives in one place — `ExtractionConfig` in
`models/extraction.py`. None are duplicated in the modules.

## How the page map is built

Three page numbers are kept apart on purpose, because conflating them is the
classic off-by-one bug:

| Field | Base | Meaning |
| --- | --- | --- |
| `pdf_page_index` | 0 | exactly Marker's `page_id` |
| `pdf_page` | 1 | what a PDF reader shows |
| `printed_page` | — | the number printed on the paper, or `null` |

The offset is **derived, never assumed**:

1. On each page, look for a short numeric block that is either a `PageFooter`
   or sits in the bottom 15% of the page.
2. That number is the printed page; `pdf_page - printed_page` is one vote.
3. The document offset is the majority vote, reported with its sample count
   and agreement ratio in `page_offset_evidence`.
4. Pages with no folio get `printed_page` inferred from the offset and labelled
   `printed_page_source: "inferred_from_offset"`, so inferred values are never
   mistaken for observed ones.

With no evidence at all, the offset stays `null` and printed pages stay `null`.
Nothing is guessed.

## Decorative pages and the contents spread

A contents page can be typeset along a curve, one glyph per span. The layout
model reasonably calls it a picture, and line order carries no meaning.

Such pages are detected generically — not by page number — via the share of
single-character spans (`decorative_single_char_span_ratio`). For a detected
page, `reconstruct_decorative_toc` reads it as spatial clusters: each
display-size numeric line is an anchor (a destination page), every other
fragment joins the nearest anchor, the small number in a cluster is the lesson
number and the Arabic-script fragments are the title.

**The lesson → page mapping is exact.** Titles are best-effort: where a word is
split across rows, a space can land inside it. Entries carry
`title_is_approximate` so a consumer knows which half to trust. Nothing about
any particular book is hard-coded — lesson count, titles, and pages all come
out of the PDF's own text layer.

## Persian normalization limits

`text/persian.py` is deterministic and conservative. Its rule is **do not
guess**: it repairs what cannot be anything else, and *reports* what would
require a lexicon.

**Repaired** (the input form cannot occur in correct Persian):

- Arabic letter forms folded to Persian (`ي`→`ی`, `ك`→`ک`, …)
- Arabic-Indic digits folded to Persian digits (Persian numerals stay numerals)
- `اال` → `الا` — a double alef is impossible in Persian, so it is always a
  lam-alef ligature the PDF decomposed backwards (`باال` → `بالا`)
- combining marks left stranded between spaces (`گرم ّ تر`)
- whitespace and zero-width noise (ZWNJ is never stripped)

**Deliberately not repaired:**

- The same reversed-ligature defect *inside* a word — `کالس`/`کلاس`,
  `سالمت`/`سلامت`. The broken form collides with real words (`سال`, `سالم`), so
  a blind rewrite would corrupt correct text. `find_suspect_lam_alef()` reports
  these for human review instead.
- Missing ZWNJ (`میرفتیم` → `می‌رفتیم`). `می` also starts ordinary words
  (`میز`, `میان`). Insertion is available via
  `PersianNormalizationConfig(insert_zwnj_heuristics=True)` and is **off by
  default**.
- Missing word spaces (`کمیگرم` → `کمی گرم`), which needs a lexicon.

No character is ever reordered; text stays exactly as right-to-left as it
arrived. `text_raw` on each block preserves the pre-normalization string
whenever normalization changed anything.

## Usage

```bash
python -m content_assistant.extraction.pipeline \
    --pdf "path/to/book.pdf" \
    --out  path/to/work-dir \
    --marker "path/to/marker_single"
```

Outputs, all under `--out` (keep it outside the repository):

| File | Contents |
| --- | --- |
| `l0_extraction.json` | the extraction result: pages, blocks, assets, diagnostics, page map, contents |
| `validation_report.json` | aggregate metrics and the pages still incomplete |
| `assets/` | images cropped by Marker |
| `cache/` | Marker's raw output, keyed by source checksum + flags |

Extraction is cached on the source checksum and flag set, so re-running only
repeats the cheap stages.

### Tests

Standard-library `unittest` — no pytest, no network, no PDF, no Marker run:

```bash
python -m unittest discover -s content_assistant/tests -t .
```

143 tests, all deterministic: no model, no network, no PDF, no Marker process.

## L1 — deterministic structuring

Everything L1 can do without a model is done, and proved, before a model is
introduced. `structuring/segmentation.py` turns the L0 artifact into lessons
and sections:

- **Lesson boundaries** come from the contents spread: a lesson runs from its
  printed page to the page before the next one starts.
- **Lesson titles** are settled between two independent sources. The contents
  page lists every lesson but sets its lettering along a curve, so words come
  back split; a lesson's opening page states the title verbatim but only some
  lessons have one. The opening page wins where it exists, the contents version
  is kept in `title_alternatives`, and a lesson that opens straight into its
  content falls back to that page's first printed heading.
- **Sections** come from the book's own printed headings. A lesson that prints
  none falls back to one section per page and records that in
  `boundary_method`, so a fallback is never mistaken for a real heading.
- **`material_profile`** measures what a lesson actually offers — characters,
  blocks, images, headings — and classifies its text density. This is what
  later decides whether a lesson can be read from its text at all.

`validation/` runs before any model exists, on purpose: if the checks were
written after seeing a model's first output they would be quietly shaped to
accept it. Rules are grouped by the stage they can run at (`structure`,
`semantic`, `final`) and each carries a stable code.

`structuring/evidence.py` builds the **Evidence Unit** — the only thing a model
is ever shown. One lesson, already segmented, every block labelled with the id
it must cite. A model cannot invent a page number because it is never asked for
one, and cannot cite outside the lesson because ids outside the unit are
rejected on arrival.

`structuring/verify.py` grounds what comes back: a citation outside the unit is
discarded, and a quotation that cannot be found in the block it was attributed
to **demotes** the claim from `explicit` to `inferred` rather than deleting it.
Nothing in the verifier can raise a claim's level.

`structuring/semantic/llm.py` is the model seam — a protocol, an adapter over
Marker's existing `BaseService` (so no new SDK and every provider Marker
supports), and a mock. No provider name appears anywhere except the import
string an operator passes on the command line.

## Not built yet

Deliberately absent, and no model has been run:

- Concept, Skill, Learning Objective, Misconception, Relation extraction
- Content Schema assembly, Content Index, Knowledge Graph
- the packaging step that emits a self-contained Content Package
- any real LLM call, and any use of the page images beyond storing them
  and flagging which lessons need them

## When OCR will be needed

The no-OCR decision follows from a measurement, not from principle. OCR becomes
necessary when a book has no usable text layer — a scan, or a PDF whose fonts
carry no recoverable encoding. The signal is already in the diagnostics: pages
where `raw_lines == 0` while the page clearly holds content, or where Marker
reports `text_extraction_method = "surya"`. Marker supports that path already;
it needs an inference backend (vLLM or llama.cpp), which this phase does not
install or require.
