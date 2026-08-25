# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (`uv.lock` is committed; CI runs `uv sync --frozen --group dev --extra full`).

```bash
uv sync --frozen --group dev --extra full   # install dev + non-PDF format deps

uv run pytest                               # full suite (needs a GPU / inference server)
uv run pytest -m cpu                        # CPU-only subset (config, providers) - what the ubuntu CI job runs
uv run pytest tests/converters/test_modes.py::test_fast_mode_digital   # single test
uv run pytest -m "not integration"          # skip slow end-to-end VLM quality checks

uv run ruff check --fix . && uv run ruff format .   # lint/format (also wired as pre-commit hooks)

uv run marker_single path/to/file.pdf --page_range 0 --output_format markdown
uv run marker path/to/folder --max_files 1 --page_range 0     # batch (multiprocessing pool)
uv run marker_single --help                 # every config key is exposed as a CLI flag
uv run marker_single config --help          # dump all builders/processors/renderers + their config
uv run marker_gui                           # streamlit playground
uv run marker_server --port 8001            # fastapi wrapper (small-scale only)
```

Test fixtures pull PDFs from the `datalab-to/pdfs` HuggingFace dataset, so tests need network access and `HF_TOKEN`. Custom pytest markers: `filename` (which dataset PDF the `temp_doc`/`pdf_document` fixtures load), `config` (dict merged into the converter config), `output_format` (picks the renderer fixture), plus `cpu` and `integration`.

Benchmarks live in [benchmarks/](benchmarks/) and are run against an external checkout of olmOCR-bench — see [benchmarks/README.md](benchmarks/README.md).

## Architecture

Conversion is a linear pipeline over a single mutable `Document` tree. `PdfConverter.build_document` ([marker/converters/pdf.py](marker/converters/pdf.py)) is the place to read first — everything else hangs off it:

1. **Provider** ([marker/providers/](marker/providers/)) — opens the source file and yields per-page text lines/chars plus rendered images. `provider_from_filepath` sniffs the file type; non-PDF formats (docx, pptx, xlsx, epub, html) are converted to PDF via weasyprint and then go through `PdfProvider`.
2. **Builders** ([marker/builders/](marker/builders/)) — `DocumentBuilder` orchestrates `LayoutBuilder` (block regions), `LineBuilder` (decides *per page* whether the embedded text layer is usable or needs OCR), then `OcrBuilder`. `StructureBuilder` runs after and groups blocks (lists, captions).
3. **Processors** ([marker/processors/](marker/processors/)) — ~30 single-purpose passes over the document, run in the order declared by `PdfConverter.default_processors` (order matters). Those under [marker/processors/llm/](marker/processors/llm/) are no-ops unless `use_llm` is set.
4. **Renderer** ([marker/renderers/](marker/renderers/)) — blocks render to HTML first (`Block.assemble_html` emits `<content-ref src='...'>` placeholders that the renderer splices); markdown/chunks are derived from that HTML, so fixing output usually means fixing the HTML, not the markdown.

### Modes and the inference server

`mode` is `balanced` (VLM layout + full-page OCR; defaults on CUDA) or `fast` (rf-detr layout + pdftext, VLM only for equations and surgical per-block repair; defaults on CPU/MPS). `--disable_ocr` bypasses the VLM entirely.

`create_model_dict()` ([marker/models.py](marker/models.py)) returns **thin clients**, not models: the heavy surya VLM runs in a separate inference server (vllm or llama.cpp) that is auto-spawned on first use. Marker worker processes hold only the small ocr-error model, which is why many workers can share one GPU. Server behavior is controlled by surya env vars (`SURYA_INFERENCE_URL`, `SURYA_INFERENCE_BACKEND`, `SURYA_INFERENCE_PARALLEL`, `VLLM_GPUS`); `marker` (batch) budgets per-worker concurrency from the server's reported capacity.

### Config system

There is no config file format — every tunable is an `Annotated` class attribute with a default on a Builder/Processor/Converter/Provider/Renderer/Service. `assign_config` ([marker/util.py](marker/util.py)) copies matching keys from the config dict onto the instance, and also supports class-scoped keys like `MarkdownRenderer_page_separator`. `ConfigCrawler` ([marker/config/crawler.py](marker/config/crawler.py)) reflects over all subclasses to build the CLI help and `config --help` output, so **adding an annotated attribute is all that's needed to expose a new option** to the CLI, JSON config, and server. `BaseConverter.resolve_dependencies` injects `config` plus anything in `artifact_dict` (the model clients, `llm_service`) into constructors by parameter name.

### Schema

`BlockTypes` ([marker/schema/__init__.py](marker/schema/__init__.py)) is the enum every block keys off. Blocks are pydantic models under [marker/schema/blocks/](marker/schema/blocks/); a block's children are *not* nested objects — `structure` holds a list of `BlockId`s resolved through `document.get_block()`. Block classes are looked up through [marker/schema/registry.py](marker/schema/registry.py), so a custom subclass is installed with `register_block_class` (or `PdfConverter.override_map`) rather than by import.

### LLM services

[marker/services/](marker/services/) wraps Gemini (default), Claude, OpenAI, Azure OpenAI, Vertex, OpenRouter, and Ollama behind `BaseService.__call__(prompt, image, block, response_schema)` — all calls are structured-output against a pydantic schema. Select one with `--llm_service <full.import.Path>`; each declares its own annotated config keys (`gemini_api_key`, `claude_model_name`, `openai_base_url`, …) which `verify_config_keys` checks at init. `LLMSimpleBlockMetaProcessor` batches all the simple per-block LLM processors into one threaded pass, so simple LLM processors should subclass `BaseLLMSimpleBlockProcessor` rather than calling the service themselves.

## Content Assistant (`content_assistant/`)

A second, independent package that sits **on top of** Marker and never modifies
it. Marker is the extraction engine; `content_assistant` turns its output into a
measured, traceable artifact - first L0/L1, then the concepts a lesson
teaches and the objectives derived from them, then a single loadable
**Content Package** an adaptive learning engine consumes. See
[content_assistant/README.md](content_assistant/README.md) and, for the schema
itself, [content_assistant/CONTENT_SCHEMA.md](content_assistant/CONTENT_SCHEMA.md).

```bash
python -m content_assistant.extraction.pipeline \
    --pdf book.pdf --out work-dir --marker "<path to marker_single>"

# semantic stages, in order - objectives take the concept stage's output
python -m content_assistant.structuring.semantic.run_concepts --l0 work-dir/l0_extraction.json --lesson 4 --out l1-dir --llm marker.services.gemini.GoogleGeminiService
python -m content_assistant.structuring.semantic.run_objectives --unit l1-dir/lesson-04/evidence-unit.json --concepts l1-dir/lesson-04/concept-verified.json --out l2-dir --llm marker.services.gemini.GoogleGeminiService

# assemble the per-lesson stage artifacts into one validated package
python -m content_assistant.package.build --l0 work-dir/l0_extraction.json --concepts l1-dir --objectives l2-dir --out content/

python -m unittest discover -s content_assistant/tests -t .   # 396 tests, no deps
```

Omitting `--llm` puts either runner in dry-run: it builds and writes the exact
prompt and stops, without calling a model. Credentials are read from Marker's
`local.env`, never passed on the command line.

Key facts when working on it:

- **No OCR, no network beyond the model call, and no new dependencies.**
  Extraction and structuring are entirely deterministic and model-free; only
  the semantic stages call a model, through Marker's own service layer. The
  target books have
  a healthy text layer; the character loss Marker shows is a layout-classification
  effect (text inside `Picture`/`PictureGroup` boxes is swallowed), so it is
  recovered geometrically from `PdfProvider.page_lines` instead.
- Marker is invoked as a **subprocess** (`marker_single`), not imported for
  conversion. The only in-process Marker import is `PdfProvider`, used to read
  the raw text layer before layout — done lazily so tests never load Marker.
- **Every threshold lives in `ExtractionConfig`** ([content_assistant/models/extraction.py](content_assistant/models/extraction.py)) — never inline a new one.
- **Three layers, bound by two different rules.** Content knowledge (`Concept`)
  and learning intent (`LearningObjective`, `Skill`) inherit `Grounded` and
  cannot exist without a citation. Learning experience (`LearningActivity`,
  `Question` in [content_assistant/models/learning.py](content_assistant/models/learning.py))
  inherits `Attributed` and carries **no evidence fields at all** — an activity
  is designed material, not a claim about the book, so what holds it together
  is linkage (`LINK001`/`LINK002`), not evidence. Do not add `evidence_ids` to
  either.
- **A question does not own pedagogical knowledge.** It names the objectives it
  tests; the concepts it touches are reached *through* them
  (`ContentSchema.concepts_for_question`) and are never stored on it. The same
  reasoning keeps `prerequisite_concepts`, `grade` and `subject` off `Concept`
  — each is already stored once elsewhere.
- **Learner state is not part of the content schema.** Student, attempt,
  mastery and learning events belong to the engine. The content layer answers
  what *exists* (`activities_for_objective`, `prerequisites_of`), never what a
  particular child should do next.
- **A prerequisite must be quoted from the book or signed by a person.**
  `EVID001` exempts `provenance.extraction_method == "human"` from needing
  evidence — the only such exemption, unreachable by any pipeline stage — and
  `LINK003` refuses a prerequisite that has neither. Use `human_relation()`;
  never let a model author one.
- Three page numbers are kept distinct on purpose: `pdf_page_index` (0-based,
  Marker's `page_id`), `pdf_page` (1-based), `printed_page` (on the paper, or
  `null`). `page_offset` is derived from footer evidence, never assumed.
- The semantic layer runs in stages, each consuming the previous one's
  artifacts: concepts ([structuring/semantic/concepts.py](content_assistant/structuring/semantic/concepts.py))
  then objectives ([structuring/semantic/objectives.py](content_assistant/structuring/semantic/objectives.py)).
  An objective may only cite the blocks its own concept rests on, and its
  confidence is capped at that concept's — so a stage can never be more certain
  than the one it was derived from. Prompts are versioned by content hash under
  `structuring/semantic/prompts/fa/`.
- **L2 is complete and frozen for the grade-1 science book**: 14/14 lessons,
  73 concepts, 79 objectives, 82/82 quotations verified by exact match, zero
  citing outside their concept. Measurements, known limits and the thresholds
  that were deliberately *not* moved are in
  [content_assistant/L2_QUALITY_REPORT.md](content_assistant/L2_QUALITY_REPORT.md).
  Read it before re-tuning anything in the semantic layer — in particular, the
  auto-accept floor is not the reason nothing auto-accepts, and lowering it
  changes nothing at any value.
- **A failed model call is not an empty answer.** Marker's services return `{}`
  once their retries are exhausted, which validates into a well-formed reply
  carrying no items. Both semantic stages check for the response field itself
  and raise `ModelCallFailed` ([structuring/semantic/llm.py](content_assistant/structuring/semantic/llm.py))
  rather than recording that a lesson has nothing in it.
- **The schema is versioned and additive.** `SCHEMA_VERSION` lives once, in
  [content_assistant/models/common.py](content_assistant/models/common.py), and
  [content_assistant/package/migrate.py](content_assistant/package/migrate.py)
  is the only code allowed to decide whether a stored version can be read — it
  upgrades an older minor, and **refuses a newer one** rather than silently
  dropping fields on the next save. Adding a field means an optional one with a
  default plus a minor bump; anything else is a major bump.
- Tests use stdlib `unittest` because pytest is not a runtime dependency here;
  `pytest.ini` limits `testpaths` to `tests`, so Marker's own suite is unaffected.
