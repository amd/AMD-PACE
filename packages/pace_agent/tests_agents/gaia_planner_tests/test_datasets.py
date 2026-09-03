# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gaia_planner.datasets import (
    _named_attachment,
    answer_verdicts,
    answers_match,
    load_gaia_from_hf,
    load_tasks,
)


class TestDatasets(unittest.TestCase):
    def test_named_attachment_keeps_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blob = (
                tmp_path / "cafebabe"
            )  # extensionless hash blob, as .resolve() would give
            blob.write_bytes(b"PK\x03\x04 docx bytes")
            link = tmp_path / "friendly.docx"
            link.symlink_to(blob)
            out = _named_attachment(link.resolve(), "friendly.docx", key="task-a")
            self.assertTrue(out.name.endswith("__friendly.docx"))
            self.assertEqual(out.read_bytes(), b"PK\x03\x04 docx bytes")

    def test_named_attachment_restores_missing_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blob = (
                tmp_path / "deadbeefhash"
            )  # extensionless blob, as .resolve() would give
            blob.write_bytes(b"%PDF-1.7 data")
            out = _named_attachment(blob, "paper.pdf", key="task-b")
            self.assertTrue(out.name.endswith("__paper.pdf"))
            self.assertEqual(out.read_bytes(), b"%PDF-1.7 data")

    def test_named_attachment_does_not_collide_on_shared_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            one = tmp_path / "blob-one"
            two = tmp_path / "blob-two"
            one.write_bytes(b"first")
            two.write_bytes(b"second")
            out_one = _named_attachment(one, "report.pdf", key="task-1")
            out_two = _named_attachment(two, "report.pdf", key="task-2")
            self.assertNotEqual(out_one, out_two)
            self.assertEqual(out_one.read_bytes(), b"first")
            self.assertEqual(out_two.read_bytes(), b"second")

    def test_loads_webvoyager_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "tasks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "web-1",
                        "ques": "Find a fact",
                        "Final answer": "A  value",
                        "web": "https://example.com",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            task = load_tasks(path)[0]

            self.assertEqual(task.task_id, "web-1")
            self.assertEqual(task.question, "Find a fact")
            self.assertEqual(task.expected, "A  value")
            self.assertEqual(task.start_url, "https://example.com")

    def test_loads_nsjail_json_and_resolves_attachments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            attachment = tmp_path / "data.csv"
            attachment.write_text("x\n1\n", encoding="utf-8")
            path = tmp_path / "tasks.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "code-1",
                            "question": "Inspect the file",
                            "expected": "1",
                            "files": ["data.csv"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            task = load_tasks(path)[0]

            self.assertEqual(task.files, (attachment.resolve(),))
            self.assertTrue(answers_match("  ONE value ", "one   value"))
            self.assertTrue(
                answers_match("$1,700", "1700")
            )  # pace_agent.score normalization
            self.assertFalse(
                answers_match("the answer is 17", "17")
            )  # strict, not loose
            self.assertIsNone(answers_match("1", None))
            self.assertEqual(
                answer_verdicts("the answer is 17", "17"),
                {
                    "gold": "17",
                    "match_strict": False,
                    "match_loose": True,
                },
            )

    def test_rejects_missing_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "tasks.json"
            path.write_text('[{"task_id": "bad"}]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no question"):
                load_tasks(path)

    def test_load_gaia_from_hf_filters_level_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            attachment = tmp_path / "sheet.xlsx"
            attachment.write_bytes(b"binary")
            rows = [
                {
                    "task_id": "L1",
                    "Question": "  Easy question ",
                    "Level": "1",
                    "Final answer": "alpha",
                    "file_name": "",
                    "file_path": "",
                },
                {
                    "task_id": "L2",
                    "Question": "Hard question with a file",
                    "Level": "2",
                    "Final answer": "beta",
                    "file_name": "sheet.xlsx",
                    "file_path": str(attachment),
                },
            ]

            def fake_loader(*, config, split, **kwargs):
                self.assertEqual(config, "2023_level2")
                self.assertEqual(split, "validation")
                return rows

            level2 = load_gaia_from_hf(level=2, _loader=fake_loader)
            self.assertEqual([task.task_id for task in level2], ["L2"])
            task = level2[0]
            self.assertEqual(task.expected, "beta")
            self.assertEqual(task.metadata["level"], "2")
            self.assertEqual(task.metadata["source"], "huggingface")
            self.assertEqual(task.files, (attachment.resolve(),))

            def all_loader(*, config, split, **kwargs):
                self.assertEqual(config, "2023_all")
                return rows

            all_levels = load_gaia_from_hf(level=None, _loader=all_loader)
            self.assertEqual([task.task_id for task in all_levels], ["L1", "L2"])
            self.assertEqual(all_levels[0].question, "Easy question")

    def test_load_gaia_from_hf_test_split_is_unscored(self):
        rows = [
            {"task_id": "T1", "Question": "Q", "Level": "1", "Final answer": "?"},
        ]

        tasks = load_gaia_from_hf(split="test", _loader=lambda *a, **k: rows)

        self.assertIsNone(tasks[0].expected)
        self.assertIsNone(answers_match("anything", tasks[0].expected))


if __name__ == "__main__":
    unittest.main()
