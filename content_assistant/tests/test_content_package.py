"""Content Packages: assembling one, storing it, reading it back, indexing it.

The book these fixtures imitate is the grade-1 science book the pipeline was
built against, but **the content is invented for the test**. The real
artifacts are run outputs and the book is not ours to redistribute, so what is
proved here is the shape and the machinery - that a package assembled from
stage artifacts survives a round trip, that a tampered one is caught, that a
registry can answer for it - and never that any particular concept is in any
particular book.

Nothing here touches a network, a model, a PDF or a Marker process.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from content_assistant.models.content import (
    Concept,
    Evidence,
    LearningObjective,
    Provenance,
    SCHEMA_VERSION,
)
from content_assistant.models.extraction import (
    Block,
    BookIdentity,
    DocumentInfo,
    ExtractionResult,
    Page,
    TocEntry,
)
from content_assistant.models.learning import LearningActivity, Question
from content_assistant.package.build import (
    BuildError,
    build_package,
    load_concept_artifacts,
    load_objective_artifacts,
    main,
    merge_evidence,
    validate_package,
)
from content_assistant.package.migrate import (
    SchemaVersionError,
    check_version,
    upgrade_payload,
)
from content_assistant.package.registry import ContentRegistry
from content_assistant.package.schema import (
    PACKAGE_FILENAME,
    ContentPackageError,
    compute_stats,
    default_path,
    load_content,
    save_content,
)

BOOK = "g1-olom"
LESSON_1 = f"{BOOK}:lesson:01"
#: Block ids carry the 0-based page index, exactly as Marker emits them, so
#: these are the blocks on printed pages 1 and 3 respectively.
BLOCK_1 = "/page/0/Text/0"
BLOCK_3 = "/page/2/Text/0"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def blk(page_index, kind, num, text, y0=0.0):
    return Block(
        block_id=f"/page/{page_index}/{kind}/{num}",
        type=kind,
        text=text,
        bbox=[10.0, y0, 500.0, y0 + 20.0],
        polygon=[[10, y0], [500, y0], [500, y0 + 20], [10, y0 + 20]],
        source="marker",
    )


def page(pdf_page, blocks):
    return Page(
        pdf_page=pdf_page,
        pdf_page_index=pdf_page - 1,
        printed_page=pdf_page,
        printed_page_source="page_footer",
        blocks=list(blocks),
    )


def science_book(book=None):
    """A two-lesson stand-in with the grade-1 science book's shape."""
    return ExtractionResult(
        document=DocumentInfo(
            source="olom.pdf",
            source_sha256="deadbeef",
            page_count=4,
            page_offset=0,
            book=book
            if book is not None
            else BookIdentity(
                book_id=BOOK,
                grade=1,
                subject="science",
                language="fa",
                title="علوم اول دبستان",
            ),
        ),
        pages=[
            page(1, [blk(0, "Text", 0, "1 زنگ علوم", 40)]),
            page(
                2,
                [
                    blk(1, "SectionHeader", 0, "چشم‌ها بسته", 50),
                    blk(1, "Text", 1, "با چشم بسته اشیا را بشناس", 90),
                ],
            ),
            page(3, [blk(2, "Text", 0, "2 دنیای جانوران", 40)]),
            page(4, [blk(3, "Text", 1, "جانوران غذا می‌خورند", 60)]),
        ],
        toc=[
            TocEntry(
                lesson_number=1,
                title="زنگ علوم",
                printed_page=1,
                source_pdf_page=0,
            ),
            TocEntry(
                lesson_number=2,
                title="دنیای جانوران",
                printed_page=3,
                source_pdf_page=0,
            ),
        ],
    )


def evidence(eid="e1", block=BLOCK_1, page_no=1, verified=True):
    return Evidence(
        id=eid,
        document_id=BOOK,
        block_id=block,
        pdf_page=page_no,
        printed_page=page_no,
        quote="زنگ علوم",
        quote_verified=verified,
        match_method="exact" if verified else "token_overlap",
    )


def concept(cid="c1", evidence_ids=("e1",), provenance=True):
    return Concept(
        id=cid,
        lesson_id=LESSON_1,
        label="زنگ علوم",
        definition="ساعتی که در آن علوم می‌خوانیم",
        evidence_ids=list(evidence_ids),
        evidence_level="explicit",
        confidence=0.8,
        requires_human_review=True,
        review_reasons=["confidence 0.80 below auto-accept"],
        provenance=Provenance(
            extraction_method="model_proposed",
            stage="concepts",
            model_id="gemini-test",
            prompt_version="concept_v1@abc12345",
        )
        if provenance
        else None,
    )


def objective(oid="o1", concept_ids=("c1",), evidence_ids=("e1",)):
    return LearningObjective(
        id=oid,
        lesson_id=LESSON_1,
        statement="دانش‌آموز بتواند نام زنگ علوم را بگوید.",
        objective_type="name",
        performance_verb="بگوید",
        concept_ids=list(concept_ids),
        evidence_ids=list(evidence_ids),
        evidence_level="explicit",
        confidence=0.75,
        provenance=Provenance(
            extraction_method="model_proposed",
            stage="objectives",
            model_id="gemini-test",
            prompt_version="objective_v1@def67890",
        ),
    )


def a_package(**kwargs):
    kwargs.setdefault("concepts", [concept()])
    kwargs.setdefault("objectives", [objective()])
    kwargs.setdefault("evidence", [evidence()])
    return build_package(extraction=science_book(), **kwargs)


def run_builder(argv):
    """Run the builder CLI, returning its exit code and its printed summary.

    The summary is captured rather than let through: it is the command's
    answer and worth asserting on, and a test suite that prints a page of JSON
    per case buries its own failures.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(argv)
    return code, json.loads(buffer.getvalue())


# ---------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------


class BuildTests(unittest.TestCase):
    def test_the_package_id_says_what_the_package_holds(self):
        package = a_package()
        self.assertEqual(package.package_id, "grade-1:science:g1-olom")
        self.assertEqual(package.grade, 1)
        self.assertEqual(package.subject, "science")

    def test_structure_is_re_derived_rather_than_trusted(self):
        # The builder reads L0 and segments it again, so a stale copy of the
        # lessons stored anywhere else can never leak into a package.
        package = a_package()
        self.assertEqual(len(package.content.lessons), 2)
        self.assertEqual(package.content.lessons[0].id, LESSON_1)

    def test_content_is_carried_across_unchanged(self):
        original = concept()
        package = a_package(concepts=[original])
        stored = package.content.concepts[0]
        self.assertEqual(stored.confidence, original.confidence)
        self.assertEqual(stored.review_reasons, original.review_reasons)
        self.assertEqual(stored.evidence_ids, original.evidence_ids)

    def test_a_book_that_does_not_say_what_it_is_cannot_be_filed(self):
        anonymous = science_book(book=BookIdentity())
        with self.assertRaises(BuildError) as caught:
            build_package(extraction=anonymous)
        self.assertIn("book_id", str(caught.exception))

    def test_the_stats_describe_what_was_actually_assembled(self):
        package = a_package()
        self.assertEqual(package.stats.concepts, 1)
        self.assertEqual(package.stats.objectives, 1)
        self.assertEqual(package.stats.evidence, 1)
        self.assertEqual(package.stats.quotations_verified, 1)
        self.assertEqual(package.stats.awaiting_review, 1)
        self.assertEqual(package.stats.reviewed, 0)

    def test_an_unverified_quotation_is_not_counted_as_one(self):
        package = a_package(evidence=[evidence(verified=False)])
        self.assertEqual(package.stats.evidence, 1)
        self.assertEqual(package.stats.quotations_verified, 0)

    def test_prompt_versions_are_read_off_the_entities_themselves(self):
        package = a_package()
        self.assertEqual(
            package.content.provenance.prompt_versions["concepts"],
            "concept_v1@abc12345",
        )

    def test_two_prompt_versions_in_one_stage_are_both_reported(self):
        # A run resumed after a prompt edit. Collapsing this to one would hide
        # exactly the inconsistency someone needs to see.
        second = concept(cid="c2").model_copy(
            update={
                "provenance": Provenance(
                    extraction_method="model_proposed",
                    stage="concepts",
                    model_id="gemini-test",
                    prompt_version="concept_v1@99999999",
                )
            }
        )
        package = a_package(concepts=[concept(), second])
        self.assertIn(
            "concept_v1@99999999",
            package.content.provenance.prompt_versions["concepts"],
        )

    def test_a_package_with_no_semantic_stage_is_still_a_package(self):
        # A book only extracted so far is a truthful thing to record.
        package = build_package(extraction=science_book())
        self.assertEqual(package.stats.concepts, 0)
        self.assertEqual(len(package.content.lessons), 2)

    def test_nothing_in_this_pipeline_writes_activities_or_questions(self):
        package = a_package()
        self.assertEqual(package.content.activities, [])
        self.assertEqual(package.content.questions, [])


class EvidenceMergeTests(unittest.TestCase):
    def test_the_same_citation_from_two_stages_is_stored_once(self):
        # The id is a hash of document, block and quote, so the concept stage
        # and the objective stage derive the same id for the same sentence.
        merged = merge_evidence([evidence()], [evidence(), evidence("e2")])
        self.assertEqual({item.id for item in merged}, {"e1", "e2"})

    def test_merging_nothing_yields_nothing(self):
        self.assertEqual(merge_evidence(), [])


# ---------------------------------------------------------------------------
# storing and reading back
# ---------------------------------------------------------------------------


class RoundTripTests(unittest.TestCase):
    def test_a_package_survives_being_written_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            save_content(a_package(), path)
            loaded = load_content(path)
        self.assertEqual(loaded.package_id, "grade-1:science:g1-olom")
        self.assertEqual(loaded.content.concepts[0].label, "زنگ علوم")
        self.assertEqual(loaded.content.objectives[0].performance_verb, "بگوید")

    def test_provenance_survives_the_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            save_content(a_package(), path)
            loaded = load_content(path)
        stored = loaded.content.concepts[0].provenance
        self.assertEqual(stored.model_id, "gemini-test")
        self.assertEqual(stored.prompt_version, "concept_v1@abc12345")

    def test_persian_text_is_stored_readably_rather_than_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            save_content(a_package(), path)
            raw = path.read_text(encoding="utf-8")
        self.assertIn("زنگ علوم", raw)

    def test_the_writer_can_never_be_the_reason_stats_are_stale(self):
        stale = a_package().model_copy(
            update={"stats": compute_stats(a_package().content)}
        )
        stale.stats.concepts = 99
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            save_content(stale, path)
            loaded = load_content(path)
        self.assertEqual(loaded.stats.concepts, 1)

    def test_a_hand_edited_package_is_refused(self):
        # Deleting a concept from the file without rebuilding leaves a summary
        # that claims more than the file holds. Reading it as if the numbers
        # were true is the failure this catches.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            save_content(a_package(), path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["content"]["concepts"] = []
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(ContentPackageError) as caught:
                load_content(path)
        self.assertIn("says 1, holds 0", str(caught.exception))

    def test_a_file_that_is_not_json_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(ContentPackageError):
                load_content(path)

    def test_the_default_layout_files_a_package_by_grade_and_subject(self):
        path = default_path(Path("content"), 1, "science")
        self.assertEqual(path.parts[-3:], ("grade-1", "science", PACKAGE_FILENAME))


class SchemaVersionTests(unittest.TestCase):
    def test_a_package_reports_one_version_read_from_its_content(self):
        package = a_package()
        self.assertEqual(package.content_schema_version, SCHEMA_VERSION)
        self.assertEqual(
            package.content_schema_version, package.content.schema_version
        )

    def test_the_same_version_is_read_as_is(self):
        self.assertEqual(check_version("1.1.0", "1.1.0"), "1.1.0")

    def test_an_older_minor_is_upgraded(self):
        self.assertEqual(check_version("1.0.0", "1.1.0"), "1.1.0")

    def test_a_newer_minor_is_refused_rather_than_silently_downgraded(self):
        # Reading it would work and saving it would delete fields this code
        # does not know about, without a word.
        with self.assertRaises(SchemaVersionError) as caught:
            check_version("1.9.0", "1.1.0")
        self.assertIn("newer code", str(caught.exception))

    def test_a_different_major_is_refused(self):
        with self.assertRaises(SchemaVersionError):
            check_version("2.0.0", "1.1.0")

    def test_the_patch_number_is_ignored_in_both_directions(self):
        self.assertEqual(check_version("1.1.9", "1.1.0"), "1.1.0")
        self.assertEqual(check_version("1.1.0", "1.1.9"), "1.1.9")

    def test_nonsense_is_not_a_version(self):
        with self.assertRaises(SchemaVersionError):
            check_version("one point one", "1.1.0")

    def test_a_1_0_0_payload_loads_and_is_restamped(self):
        # The whole promise of an additive minor version: an artifact written
        # before per-entity provenance existed still reads.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PACKAGE_FILENAME
            save_content(a_package(), path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["content"]["schema_version"] = "1.0.0"
            for entity in payload["content"]["concepts"]:
                entity.pop("provenance", None)
                entity.pop("review_status", None)
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            loaded = load_content(path)
        self.assertEqual(loaded.content.schema_version, SCHEMA_VERSION)
        self.assertIsNone(loaded.content.concepts[0].provenance)
        self.assertEqual(loaded.content.concepts[0].review_status, "pending")

    def test_a_payload_with_no_content_is_not_a_package(self):
        with self.assertRaises(SchemaVersionError):
            upgrade_payload({"package_id": "x"})


# ---------------------------------------------------------------------------
# reading the artifacts a run left behind
# ---------------------------------------------------------------------------


def write_stage_artifacts(root: Path, *, with_provenance: bool = True):
    """Write the two stage artifacts a runner produces, for one lesson."""
    concept_dir = root / "l1" / "lesson-01"
    concept_dir.mkdir(parents=True, exist_ok=True)
    stored = concept(provenance=with_provenance)
    concept_dir.joinpath("concept-verified.json").write_text(
        json.dumps(
            {
                "lesson_id": LESSON_1,
                "model_id": "gemini-test",
                "prompt_version": "concept_v1@abc12345",
                "concepts": [stored.model_dump(mode="json")],
                "evidence": [evidence().model_dump(mode="json")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    objective_dir = root / "l2" / "lesson-01"
    objective_dir.mkdir(parents=True, exist_ok=True)
    objective_dir.joinpath("objective-verified.json").write_text(
        json.dumps(
            {
                "lesson_id": LESSON_1,
                "model_id": "gemini-test",
                "prompt_version": "objective_v1@def67890",
                "objectives": [objective().model_dump(mode="json")],
                "evidence": [evidence().model_dump(mode="json")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    root.joinpath("l0_extraction.json").write_text(
        science_book().model_dump_json(),
        encoding="utf-8",
    )


class ArtifactLoadingTests(unittest.TestCase):
    def test_concepts_are_read_out_of_the_stage_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root)
            concepts, found = load_concept_artifacts(root / "l1")
        self.assertEqual([c.id for c in concepts], ["c1"])
        self.assertEqual([e.id for e in found], ["e1"])

    def test_an_artifact_written_before_provenance_existed_gets_the_stages(self):
        # Lifted from what the stage result already recorded, not invented:
        # the artifact named its model and its prompt, once, at the top.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root, with_provenance=False)
            concepts, _ = load_concept_artifacts(root / "l1")
        self.assertEqual(concepts[0].provenance.model_id, "gemini-test")
        self.assertEqual(
            concepts[0].provenance.prompt_version, "concept_v1@abc12345"
        )
        self.assertEqual(concepts[0].provenance.stage, "concepts")

    def test_an_artifact_that_names_no_model_produces_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "l1" / "lesson-01"
            root.mkdir(parents=True)
            root.joinpath("concept-verified.json").write_text(
                json.dumps(
                    {
                        "concepts": [
                            concept(provenance=False).model_dump(mode="json")
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            concepts, _ = load_concept_artifacts(root.parent)
        self.assertIsNone(concepts[0].provenance.model_id)

    def test_objectives_are_read_the_same_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root)
            objectives, _ = load_objective_artifacts(root / "l2")
        self.assertEqual([o.id for o in objectives], ["o1"])

    def test_a_missing_stage_directory_yields_nothing_rather_than_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            concepts, found = load_concept_artifacts(Path(tmp) / "absent")
        self.assertEqual(concepts, [])
        self.assertEqual(found, [])


class BuilderCommandTests(unittest.TestCase):
    def test_the_builder_writes_a_package_a_registry_can_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root)
            code, summary = run_builder(
                [
                    "--l0",
                    str(root / "l0_extraction.json"),
                    "--concepts",
                    str(root / "l1"),
                    "--objectives",
                    str(root / "l2"),
                    "--out",
                    str(root / "content"),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(summary["written"])
            self.assertEqual(summary["package_id"], "grade-1:science:g1-olom")
            path = default_path(root / "content", 1, "science")
            self.assertTrue(path.exists())
            loaded = load_content(path)
            self.assertEqual(loaded.stats.concepts, 1)
            self.assertEqual(loaded.stats.objectives, 1)
            self.assertTrue((path.parent / "package-validation.json").exists())
            self.assertTrue((path.parent / "package-review.md").exists())

    def test_two_builds_of_the_same_artifacts_differ_only_in_when(self):
        # Reproducibility in the form that matters: a rebuild must not look
        # like a rewrite. Ids, counts and content have to be identical.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root)
            first = build_package(
                extraction=science_book(),
                concepts=load_concept_artifacts(root / "l1")[0],
                objectives=load_objective_artifacts(root / "l2")[0],
                evidence=[evidence()],
                built_at="2026-01-01T00:00:00+00:00",
            )
            second = build_package(
                extraction=science_book(),
                concepts=load_concept_artifacts(root / "l1")[0],
                objectives=load_objective_artifacts(root / "l2")[0],
                evidence=[evidence()],
                built_at="2026-06-30T00:00:00+00:00",
            )
        self.assertEqual(
            first.model_dump(exclude={"built_at", "content"}),
            second.model_dump(exclude={"built_at", "content"}),
        )
        self.assertEqual(
            first.content.model_dump(exclude={"provenance"}),
            second.content.model_dump(exclude={"provenance"}),
        )

    def test_a_package_with_validation_errors_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root)
            broken = root / "l2" / "lesson-01" / "objective-verified.json"
            payload = json.loads(broken.read_text(encoding="utf-8"))
            # An objective citing evidence its concept does not rest on: the
            # structural rule the objective stage exists to enforce.
            payload["objectives"][0]["evidence_ids"] = ["e-stray"]
            payload["evidence"] = [
                evidence(eid="e-stray", block=BLOCK_3, page_no=3).model_dump(
                    mode="json"
                )
            ]
            broken.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            code, summary = run_builder(
                [
                    "--l0",
                    str(root / "l0_extraction.json"),
                    "--concepts",
                    str(root / "l1"),
                    "--objectives",
                    str(root / "l2"),
                    "--out",
                    str(root / "content"),
                ]
            )
            path = default_path(root / "content", 1, "science")
            self.assertEqual(code, 1)
            self.assertFalse(summary["written"])
            self.assertFalse(path.exists())
            # The findings are still written, because "it failed" without
            # saying why is not a usable answer.
            report = json.loads(
                (path.parent / "package-validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                "PEDA006", {f["code"] for f in report["findings"]}
            )

    def test_a_clean_package_validates_with_no_errors(self):
        package = a_package()
        report = validate_package(package, science_book())
        self.assertEqual(report.errors, [])


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def other_package():
    """A second book, so cross-package behaviour has something to cross."""
    extraction = science_book(
        book=BookIdentity(
            book_id="g1-farsi", grade=1, subject="farsi", language="fa"
        )
    )
    return build_package(extraction=extraction)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ContentRegistry.from_packages(
            [a_package(), other_package()]
        )

    def test_it_knows_what_grades_and_subjects_exist(self):
        self.assertEqual(self.registry.grades(), [1])
        self.assertEqual(self.registry.subjects(), ["farsi", "science"])
        self.assertEqual(self.registry.subjects(grade=2), [])

    def test_it_finds_a_package_by_grade_and_subject(self):
        found = self.registry.find(1, "science")
        self.assertIsNotNone(found)
        self.assertEqual(found.book_id, BOOK)
        self.assertIsNone(self.registry.find(6, "science"))

    def test_it_answers_which_package_an_id_belongs_to(self):
        self.assertEqual(self.registry.kind_of("c1"), "concept")
        self.assertEqual(
            self.registry.package_of("c1").package_id, "grade-1:science:g1-olom"
        )
        self.assertIsNone(self.registry.kind_of("no-such-id"))

    def test_it_hands_back_the_object_the_package_holds_not_a_copy(self):
        # The registry duplicates no content. If it copied, an edit to a
        # package would leave the registry serving a stale entity.
        package = self.registry.find(1, "science")
        self.assertIs(self.registry.get("c1"), package.content.concepts[0])

    def test_it_lists_entities_across_packages_and_filters_by_grade(self):
        self.assertEqual([c.id for c in self.registry.concepts()], ["c1"])
        self.assertEqual(len(self.registry.lessons()), 4)
        self.assertEqual(len(self.registry.lessons(grade=2)), 0)

    def test_two_files_claiming_one_book_are_refused(self):
        registry = ContentRegistry.from_packages([a_package()])
        with self.assertRaises(ValueError) as caught:
            registry.add(a_package())
        self.assertIn("already registered", str(caught.exception))

    def test_two_books_claiming_one_entity_id_are_refused(self):
        # Ids are derived from the book id, so this cannot happen by accident;
        # it means one file is not the book it says it is, and loading both
        # would make every lookup depend on load order.
        colliding = build_package(
            extraction=science_book(
                book=BookIdentity(
                    book_id="g1-riazi", grade=1, subject="math", language="fa"
                )
            ),
            concepts=[concept()],
            evidence=[evidence()],
        )
        with self.assertRaises(ValueError) as caught:
            self.registry.add(colliding)
        self.assertIn("claimed by both", str(caught.exception))

    def test_it_answers_the_traversals_an_engine_makes(self):
        self.assertEqual(
            [o.id for o in self.registry.objectives_for_concept("c1")], ["o1"]
        )
        self.assertEqual(
            [c.id for c in self.registry.concepts_for_objective("o1")], ["c1"]
        )

    def test_a_traversal_from_an_unknown_id_is_empty_not_an_error(self):
        self.assertEqual(self.registry.objectives_for_concept("ghost"), [])
        self.assertEqual(self.registry.prerequisites_of("ghost"), [])
        self.assertIsNone(self.registry.get("ghost"))

    def test_remediation_is_asked_for_by_name(self):
        package = a_package(
            concepts=[concept()],
            objectives=[objective()],
            evidence=[evidence()],
        )
        package.content.activities.extend(
            [
                LearningActivity(
                    id="a1",
                    activity_type="practice",
                    title="تمرین",
                    objective_ids=["o1"],
                ),
                LearningActivity(
                    id="a2",
                    activity_type="remediation",
                    title="دوباره",
                    objective_ids=["o1"],
                ),
            ]
        )
        package.content.questions.append(
            Question(
                id="q1",
                question_type="true_false",
                prompt="زنگ علوم؟",
                objective_ids=["o1"],
            )
        )
        registry = ContentRegistry.from_packages([package])
        self.assertEqual(
            [a.id for a in registry.remediation_for_objective("o1")], ["a2"]
        )
        self.assertEqual(
            [a.id for a in registry.activities_for_objective("o1")],
            ["a1", "a2"],
        )
        self.assertEqual(
            [q.id for q in registry.questions_for_objective("o1")], ["q1"]
        )
        self.assertEqual(
            [c.id for c in registry.concepts_for_question("q1")], ["c1"]
        )

    def test_it_reports_what_is_loaded(self):
        summary = self.registry.summary()
        self.assertEqual(summary["packages"], 2)
        self.assertEqual(summary["grades"], [1])
        self.assertIn("grade-1:science:g1-olom", summary["by_package"])


class RegistryFromDirectoryTests(unittest.TestCase):
    def test_it_finds_every_package_under_a_content_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "content"
            save_content(a_package(), default_path(root, 1, "science"))
            save_content(other_package(), default_path(root, 1, "farsi"))
            registry = ContentRegistry.from_directory(root)
        self.assertEqual(len(registry.packages()), 2)
        self.assertEqual(registry.subjects(1), ["farsi", "science"])

    def test_an_empty_content_root_is_an_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ContentRegistry.from_directory(Path(tmp))
        self.assertEqual(registry.packages(), [])
        self.assertEqual(registry.grades(), [])


class GradeOneLoadingTests(unittest.TestCase):
    """Loading a grade-1 package end to end, on the shape the real one has.

    The content is invented - the real artifacts are run outputs and the book
    is not ours to redistribute - so this proves the path from artifacts to a
    loaded, indexed, traversable package, and claims nothing about the book.
    """

    def test_artifacts_become_a_loadable_indexed_grade_one_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stage_artifacts(root)
            code, _ = run_builder(
                [
                    "--l0",
                    str(root / "l0_extraction.json"),
                    "--concepts",
                    str(root / "l1"),
                    "--objectives",
                    str(root / "l2"),
                    "--out",
                    str(root / "content"),
                ]
            )
            self.assertEqual(code, 0)
            registry = ContentRegistry.from_directory(root / "content")

        self.assertEqual(registry.grades(), [1])
        package = registry.find(1, "science")
        self.assertEqual(package.content_schema_version, SCHEMA_VERSION)
        self.assertEqual(package.stats.quotations_verified, 1)

        objective_id = registry.objectives(grade=1)[0].id
        concepts = registry.concepts_for_objective(objective_id)
        self.assertEqual([c.id for c in concepts], ["c1"])
        # Nothing has authored an activity yet, and the registry says so
        # rather than inventing one.
        self.assertEqual(registry.activities_for_objective(objective_id), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
