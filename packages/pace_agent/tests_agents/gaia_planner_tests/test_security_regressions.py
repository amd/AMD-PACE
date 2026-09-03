# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Regression tests for the gaia_planner review findings.

`fetch_url` and `provision` run on the *host*, outside the sandbox, driven by
model-supplied arguments — so their input policies are security boundaries.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from gaia_planner.adapters import (
    _PublicOnlyRedirectHandler,
    _assert_public_http_url,
    _download_to_workspace,
    _pip_install,
)
from gaia_planner.datasets import PACKAGE_ROOT, _named_attachment, load_tasks
from gaia_planner.langgraph_planner import _llm_is_verifying


class _Workspace:
    """Minimal stand-in for the sandbox workspace fetch_url writes into."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def write_binary_file(self, name, data):
        path = self.directory / name
        path.write_bytes(data)
        return path


class TestFetchUrlEgressPolicy(unittest.TestCase):
    """fetch_url must not become a way to read local files or reach the intranet."""

    NON_PUBLIC = [
        "file:///etc/passwd",  # urllib serves file:// via its own FileHandler
        "file:///etc/hostname",
        "ftp://example.com/x",
        "gopher://example.com/",
        "http://169.254.169.254/latest/meta-data/",  # cloud instance metadata
        "http://127.0.0.1:8080/",
        "http://localhost/admin",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
    ]

    def test_non_public_destinations_are_blocked(self):
        for url in self.NON_PUBLIC:
            with self.subTest(url=url):
                with self.assertRaises(PermissionError):
                    _assert_public_http_url(url)

    def test_public_web_is_still_allowed(self):
        for url in ("https://arxiv.org/abs/2301.00001", "http://example.com/paper.pdf"):
            with self.subTest(url=url):
                _assert_public_http_url(url)

    def test_file_scheme_cannot_leak_host_files_into_the_workspace(self):
        workspace = _Workspace(tempfile.mkdtemp())
        with self.assertRaises(PermissionError):
            _download_to_workspace("file:///etc/hostname", None, workspace)
        self.assertEqual(list(workspace.directory.iterdir()), [])

    def test_redirect_targets_are_revalidated(self):
        """A public URL can 302 to the metadata service."""
        handler = _PublicOnlyRedirectHandler()
        with self.assertRaises(PermissionError):
            handler.redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "http://169.254.169.254/latest/meta-data/",
            )


class TestProvisionPipSpecs(unittest.TestCase):
    def test_option_injection_is_refused(self):
        """`--index-url` in the package list would retarget the index and run
        that index's setup.py on the host."""
        target = Path(tempfile.mkdtemp())
        ok, log = _pip_install(
            ["--index-url", "https://evil.example/simple", "numpy"], target
        )
        self.assertFalse(ok)
        self.assertIn("refusing pip option", log)
        self.assertEqual(list(target.iterdir()), [])

    def test_leading_dash_of_any_form_is_refused(self):
        for spec in ("--extra-index-url", "-i", "--find-links"):
            with self.subTest(spec=spec):
                ok, log = _pip_install([spec, "http://x/"], Path(tempfile.mkdtemp()))
                self.assertFalse(ok)
                self.assertIn("refusing pip option", log)


class TestAttachmentNaming(unittest.TestCase):
    """The symlink farm must be private, not a fixed world-shared /tmp path."""

    def test_extensionless_blob_is_exposed_under_the_friendly_name(self):
        work = Path(tempfile.mkdtemp())
        blob = work / "deadbeefhash"
        blob.write_bytes(b"%PDF-1.7 data")
        out = _named_attachment(blob, "paper.pdf", key="task-b")
        self.assertTrue(out.name.endswith("__paper.pdf"))
        self.assertEqual(out.read_bytes(), b"%PDF-1.7 data")

    def test_link_dir_is_private_and_not_the_fixed_shared_path(self):
        work = Path(tempfile.mkdtemp())
        blob = work / "blob"
        blob.write_bytes(b"data")
        out = _named_attachment(blob, "doc.pdf", key="task-p")
        link_dir = out.parent
        self.assertNotEqual(link_dir, Path(tempfile.gettempdir()) / "gaia_named_files")
        self.assertEqual(stat.S_IMODE(link_dir.stat().st_mode), 0o700)
        self.assertEqual(link_dir.stat().st_uid, os.getuid())

    def test_source_already_named_is_used_as_is(self):
        work = Path(tempfile.mkdtemp())
        real = work / "real.docx"
        real.write_bytes(b"PK\x03\x04")
        link = work / "friendly.docx"
        link.symlink_to(real)
        out = _named_attachment(link, "friendly.docx", key="task-a")
        self.assertEqual(out, link)

    def test_shared_friendly_name_does_not_collide(self):
        work = Path(tempfile.mkdtemp())
        one, two = work / "blob-one", work / "blob-two"
        one.write_bytes(b"first")
        two.write_bytes(b"second")
        out_one = _named_attachment(one, "report.pdf", key="task-1")
        out_two = _named_attachment(two, "report.pdf", key="task-2")
        self.assertNotEqual(out_one, out_two)
        self.assertTrue(out_one.name.endswith("__report.pdf"))
        self.assertEqual(out_one.read_bytes(), b"first")
        self.assertEqual(out_two.read_bytes(), b"second")


class TestTaskAttachmentContainment(unittest.TestCase):
    """A task file may come from an untrusted dataset drop."""

    def _task_file(self, directory, entry):
        path = Path(directory) / "tasks.json"
        path.write_text(
            json.dumps([{"task_id": "t1", "question": "q", "files": [entry]}]),
            encoding="utf-8",
        )
        return path

    def test_traversal_and_absolute_paths_rejected(self):
        for entry in (
            "../../../../../../etc/passwd",
            "../../../../../../../root/.ssh/id_rsa",
            "/etc/passwd",
        ):
            with self.subTest(entry=entry):
                path = self._task_file(tempfile.mkdtemp(), entry)
                with self.assertRaises(ValueError):
                    load_tasks(path)

    def test_attachment_beside_the_task_file_is_allowed(self):
        work = Path(tempfile.mkdtemp())
        (work / "data.pdf").write_bytes(b"%PDF")
        path = self._task_file(work, "data.pdf")
        self.assertEqual(load_tasks(path)[0].files, ((work / "data.pdf").resolve(),))

    def test_shipped_fixture_layout_still_loads(self):
        """configs/ points at pace_agent/examples/fixtures/, one level up."""
        tasks = load_tasks(PACKAGE_ROOT / "configs" / "gaia_sample_tasks.json")
        self.assertTrue(tasks)
        for task in tasks:
            for attachment in task.files:
                self.assertTrue(attachment.is_file(), attachment)

    def test_extra_root_env_widens_the_boundary(self):
        work = Path(tempfile.mkdtemp())
        external = work / "external"
        external.mkdir()
        (external / "doc.pdf").write_bytes(b"%PDF")
        tasks_dir = work / "tasks"
        tasks_dir.mkdir()
        path = self._task_file(tasks_dir, "../external/doc.pdf")

        with self.assertRaises(ValueError):
            load_tasks(path)

        os.environ["GAIA_TASK_FILES_ROOT"] = str(external)
        try:
            self.assertEqual(
                load_tasks(path)[0].files, ((external / "doc.pdf").resolve(),)
            )
        finally:
            os.environ.pop("GAIA_TASK_FILES_ROOT", None)


class TestVerifyGate(unittest.TestCase):
    """A failed execute_code must not satisfy the verify-by-execution gate."""

    FAILURES = [
        '{"status": "error", "error": "ValueError: unknown tool"}',
        '{"status": "error", "error": "download failed: URLError"}',
        "[tool error] invalid JSON arguments: Expecting value (char 0)",
        "[status: timeout]",
        "[status: oom]",
        "[status: security_violation]",
    ]

    def test_failed_execute_code_is_not_verification(self):
        for output in self.FAILURES:
            with self.subTest(output=output[:34]):
                self.assertFalse(
                    _llm_is_verifying([{"name": "execute_code", "output": output}])
                )

    def test_successful_execute_code_is_verification(self):
        self.assertTrue(
            _llm_is_verifying(
                [{"name": "execute_code", "output": "[status: success]\nSTDOUT:\n42"}]
            )
        )

    def test_evidence_gathering_is_not_verification(self):
        self.assertFalse(
            _llm_is_verifying(
                [
                    {"name": "execute_code", "output": "[status: success]"},
                    {"name": "browse_web", "output": "{}"},
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
