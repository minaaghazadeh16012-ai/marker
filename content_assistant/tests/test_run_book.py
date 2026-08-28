"""The whole-book runner: resumable, and honest about what did not happen.

Sixty-six commands typed by hand is not a workflow, so something has to drive
the per-lesson runners. What that something must never do is turn a failure
into a result - and the failure mode it exists in the presence of is a provider
that answers some lessons and refuses others.

Three properties are proved here.

*It resumes.* A lesson already done is skipped, not re-run and not re-paid for.

*It writes nothing for a lesson whose call failed.* The artifact's absence is
what makes the next run pick the lesson up again; an empty artifact would be a
claim that the lesson holds nothing, which no failed call earns.

*It stops when the provider has stopped.* One lesson failing is a lesson;
several in a row is the provider, and continuing then converts quota into noise.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from content_assistant.models.extraction import (
    Block,
    BookIdentity,
    DocumentInfo,
    ExtractionResult,
    Page,
    TocEntry,
)
from content_assistant.structuring.semantic import run_book
from content_assistant.structuring.semantic.llm import ModelCallFailed

BOOK = "g1-test"


def blk(page_index, num, text, y0=0.0, kind="Text"):
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
        assets=[],
    )


def a_book(tmp: Path) -> Path:
    """A two-lesson L0 artifact on disk, enough for both stages to build on."""
    pages = [
        page(1, [blk(0, 0, "زنگ علوم", 20), blk(0, 1, "ب" * 120, 60)]),
        page(2, [blk(1, 0, "ج" * 120, 30)]),
        page(3, [blk(2, 0, "دنیای جانوران", 20), blk(2, 1, "د" * 120, 60)]),
        page(4, [blk(3, 0, "ه" * 120, 30)]),
    ]
    result = ExtractionResult(
        document=DocumentInfo(
            source="test.pdf",
            source_sha256="0" * 64,
            page_count=4,
            page_offset=0,
            toc_source="decorative",
            book=BookIdentity(
                book_id=BOOK, grade=1, subject="science", language="fa"
            ),
        ),
        pages=pages,
        toc=[
            TocEntry(lesson_number=1, title="زنگ علوم", printed_page=1),
            TocEntry(lesson_number=2, title="دنیای جانوران", printed_page=3),
        ],
    )
    path = tmp / "l0_extraction.json"
    path.write_text(result.model_dump_json(indent=1), encoding="utf-8")
    return path


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.l0 = a_book(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_every_lesson_is_visited_and_nothing_is_called(self):
        report = run_book.run_book(
            l0_path=self.l0,
            concepts_root=self.tmp / "l1",
            objectives_root=self.tmp / "l2",
        )
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["lessons_total"], 2)
        self.assertEqual([e["lesson"] for e in report["lessons"]], [1, 2])

    def test_a_dry_run_writes_the_prompt_it_would_have_sent(self):
        run_book.run_book(
            l0_path=self.l0,
            concepts_root=self.tmp / "l1",
            objectives_root=None,
        )
        self.assertTrue((self.tmp / "l1" / "lesson-01" / "prompt.txt").exists())

    def test_nothing_counts_as_done_until_a_verified_artifact_exists(self):
        # A dry run writes prompts, not results. Reporting those as done would
        # make the next live run skip every lesson.
        report = run_book.run_book(
            l0_path=self.l0,
            concepts_root=self.tmp / "l1",
            objectives_root=self.tmp / "l2",
        )
        self.assertEqual(report["concepts_done"], 0)
        self.assertEqual(report["remaining"], [1, 2])

    def test_only_narrows_the_run_to_the_lessons_named(self):
        report = run_book.run_book(
            l0_path=self.l0,
            concepts_root=self.tmp / "l1",
            objectives_root=None,
            only=[2],
        )
        self.assertEqual([e["lesson"] for e in report["lessons"]], [2])

    def test_objectives_are_skipped_when_there_are_no_concepts_to_derive_from(
        self,
    ):
        # Calling anyway would spend a request to be told so.
        report = run_book.run_book(
            l0_path=self.l0,
            concepts_root=self.tmp / "l1",
            objectives_root=self.tmp / "l2",
        )
        self.assertIn("no concepts", report["lessons"][0]["objectives"])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.l0 = a_book(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def _finish(self, lesson, stage_root, filename):
        directory = stage_root / f"lesson-{lesson:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_text("{}", encoding="utf-8")

    def test_a_lesson_already_done_is_not_run_again(self):
        self._finish(1, self.tmp / "l1", "concept-verified.json")
        with mock.patch.object(run_book, "_run") as called:
            called.return_value = {"exit_code": 0, "output": ""}
            report = run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=None,
            )
        self.assertEqual(report["lessons"][0]["concepts"], "already done")
        # Only lesson 2 reached a runner.
        self.assertEqual(called.call_count, 1)

    def test_force_runs_a_finished_lesson_again(self):
        self._finish(1, self.tmp / "l1", "concept-verified.json")
        with mock.patch.object(run_book, "_run") as called:
            called.return_value = {"exit_code": 0, "output": ""}
            run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=None,
                force=True,
            )
        self.assertEqual(called.call_count, 2)

    def test_what_is_left_is_reported_by_lesson_number(self):
        self._finish(1, self.tmp / "l1", "concept-verified.json")
        self._finish(1, self.tmp / "l2", "objective-verified.json")
        with mock.patch.object(run_book, "_run") as called:
            called.return_value = {"exit_code": 0, "output": ""}
            report = run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=self.tmp / "l2",
            )
        self.assertEqual(report["remaining"], [2])


class FailureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.l0 = a_book(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_failed_call_leaves_the_lesson_to_be_picked_up_again(self):
        with mock.patch.object(
            run_book, "_run", side_effect=ModelCallFailed("quota")
        ):
            report = run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=None,
                max_consecutive_failures=99,
            )
        self.assertIn("call failed", report["lessons"][0]["concepts"])
        self.assertEqual(report["concepts_done"], 0)
        self.assertEqual(report["remaining"], [1, 2])

    def test_enough_failures_in_a_row_stop_the_run(self):
        with mock.patch.object(
            run_book, "_run", side_effect=ModelCallFailed("quota")
        ) as called:
            report = run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=None,
                max_consecutive_failures=1,
            )
        self.assertTrue(report["stopped_early"])
        self.assertEqual(called.call_count, 1)

    def test_one_failure_between_successes_does_not_stop_the_run(self):
        outcomes = [
            ModelCallFailed("blip"),
            {"exit_code": 0, "output": ""},
        ]

        def answer(*_args, **_kwargs):
            got = outcomes.pop(0)
            if isinstance(got, Exception):
                raise got
            return got

        with mock.patch.object(run_book, "_run", side_effect=answer):
            report = run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=None,
                max_consecutive_failures=2,
            )
        self.assertFalse(report["stopped_early"])
        self.assertEqual(len(report["lessons"]), 2)

    def test_an_unexpected_error_is_recorded_rather_than_swallowed(self):
        with mock.patch.object(
            run_book, "_run", side_effect=RuntimeError("disk full")
        ):
            report = run_book.run_book(
                l0_path=self.l0,
                concepts_root=self.tmp / "l1",
                objectives_root=None,
                max_consecutive_failures=99,
            )
        self.assertIn("RuntimeError", report["lessons"][0]["concepts"])


class ProvenanceTests(unittest.TestCase):
    """An artifact must be able to say which model answered, not which adapter."""

    def test_the_concrete_model_name_joins_the_service_path(self):
        from content_assistant.structuring.semantic.llm import (
            MarkerServiceClient,
            concrete_model_name,
        )

        class FakeGemini:
            gemini_model_name = "gemini-9.9-flash"

        self.assertEqual(concrete_model_name(FakeGemini()), "gemini-9.9-flash")

        with mock.patch(
            "marker.util.strings_to_classes", return_value=[lambda cfg: FakeGemini()]
        ), mock.patch(
            "content_assistant.structuring.semantic.llm.build_service_config",
            return_value={},
        ):
            client = MarkerServiceClient.from_import_path("some.Service")
        self.assertEqual(client.model_id, "some.Service@gemini-9.9-flash")

    def test_a_service_naming_no_model_still_identifies_itself(self):
        from content_assistant.structuring.semantic.llm import (
            MarkerServiceClient,
            concrete_model_name,
        )

        class Nameless:
            pass

        self.assertIsNone(concrete_model_name(Nameless()))
        with mock.patch(
            "marker.util.strings_to_classes", return_value=[lambda cfg: Nameless()]
        ), mock.patch(
            "content_assistant.structuring.semantic.llm.build_service_config",
            return_value={},
        ):
            client = MarkerServiceClient.from_import_path("some.Service")
        self.assertEqual(client.model_id, "some.Service")

    def test_a_semantic_call_asks_for_more_time_than_a_block_repair(self):
        # Marker's 30s default is right for repairing one block and wrong for
        # a whole lesson; two of ten quran lessons failed on "deadline
        # expired" until this was raised.
        from content_assistant.structuring.semantic.llm import (
            SEMANTIC_TIMEOUT_SECONDS,
            LLMRequest,
        )

        request = LLMRequest(prompt="x", response_schema=dict)
        self.assertEqual(request.timeout, SEMANTIC_TIMEOUT_SECONDS)
        self.assertGreater(SEMANTIC_TIMEOUT_SECONDS, 30)


if __name__ == "__main__":
    unittest.main()
