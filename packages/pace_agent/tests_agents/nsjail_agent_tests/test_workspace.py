# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for Workspace."""

from __future__ import annotations

import base64
import shutil
import tempfile
import unittest
from pathlib import Path

from nsjail_agent.workspace import Workspace
from nsjail_agent.models import WorkspaceFile


class TestWorkspace(unittest.TestCase):
    def test_context_manager_creates_and_cleans_up(self):
        with Workspace() as ws:
            path = ws.host_path
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())

    def test_write_and_read_file(self):
        with Workspace() as ws:
            ws.write_file("hello.txt", "hello world")
            content = ws.read_file("hello.txt")
        self.assertEqual(content, "hello world")

    def test_missing_file_returns_none(self):
        with Workspace() as ws:
            self.assertIsNone(ws.read_file("nonexistent.txt"))

    def test_initial_files_written(self):
        files = [
            WorkspaceFile(name="a.py", content="x = 1"),
            WorkspaceFile(name="b.py", content="y = 2"),
        ]
        with Workspace(files=files) as ws:
            self.assertEqual(ws.read_file("a.py"), "x = 1")
            self.assertEqual(ws.read_file("b.py"), "y = 2")

    def test_initial_binary_file_written(self):
        payload = b"not utf8: \x00\xff"
        files = [
            WorkspaceFile(
                name="data.bin",
                content="",
                content_base64=base64.b64encode(payload).decode("ascii"),
            )
        ]

        with Workspace(files=files) as ws:
            self.assertEqual((ws.host_path / "data.bin").read_bytes(), payload)

    def test_collect_files(self):
        with Workspace() as ws:
            ws.write_file("out.txt", "result")
            ws.write_file("log.txt", "logs")
            collected = ws.collect_files(["out.txt", "missing.txt"])
        self.assertEqual(collected, {"out.txt": "result"})

    def test_collect_files_encodes_binary_files(self):
        payload = b"\x89PNG\r\n\x1a\n"
        with Workspace() as ws:
            ws.write_binary_file("processed_out.png", payload)
            collected = ws.collect_files(["processed_out.png"])

        self.assertTrue(collected["processed_out.png"].startswith("[base64:8 bytes]\n"))
        self.assertEqual(
            base64.b64decode(collected["processed_out.png"].splitlines()[1]),
            payload,
        )

    def test_rejects_absolute_write_path(self):
        with Workspace() as ws:
            with self.assertRaisesRegex(ValueError, "relative"):
                ws.write_file("/tmp/escape.txt", "bad")

    def test_rejects_parent_directory_escape(self):
        with Workspace() as ws:
            with self.assertRaisesRegex(ValueError, "relative"):
                ws.write_file("../escape.txt", "bad")

    def test_rejects_symlink_read_escape(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        target = Path(tmp) / "secret.txt"
        target.write_text("secret", encoding="utf-8")

        with Workspace() as ws:
            (ws.host_path / "leak").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes workspace"):
                ws.read_file("leak")


if __name__ == "__main__":
    unittest.main()
