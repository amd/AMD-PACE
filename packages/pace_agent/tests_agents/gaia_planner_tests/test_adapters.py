# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gaia_planner.adapters import (
    VideoAnalysisAdapter,
    VideoAnalysisConfig,
    WebVoyagerAdapter,
    WebVoyagerConfig,
    _echo_last_expression,
    _host_allowed,
    _prepare_execute_arguments,
)


def _exec_capture(code: str) -> str:
    import io
    import contextlib

    buf = io.StringIO()
    ns: dict = {}
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<t>", "exec"), ns)
    return buf.getvalue()


class TestAdapters(unittest.TestCase):
    def test_echo_last_expression_displays_bare_trailing_value(self):
        code = "vals = ['a.xml', 'b.xlsx']\nvals"
        self.assertEqual(
            _exec_capture(_echo_last_expression(code)).strip(), "['a.xml', 'b.xlsx']"
        )

    def test_echo_last_expression_leaves_non_display_cases_alone(self):
        # trailing assignment, existing print, and docstring must not gain extra output
        self.assertEqual(_exec_capture(_echo_last_expression("x = 5")), "")
        self.assertEqual(
            _exec_capture(_echo_last_expression("print('hi')")).strip(), "hi"
        )
        self.assertEqual(_exec_capture(_echo_last_expression("y = 1\n'note'")), "")
        # a value of None is suppressed
        self.assertEqual(
            _exec_capture(_echo_last_expression("def f():\n    pass\nf()")), ""
        )

    def test_prepare_execute_arguments_echoes_last_expression(self):
        prepared = _prepare_execute_arguments(json.dumps({"code": "z = 2\nz"}))
        self.assertIsInstance(prepared, dict)
        self.assertEqual(_exec_capture(prepared["code"]).strip(), "2")

    def test_provision_allowlist_permits_trusted_hosts(self):
        for url in (
            "https://github.com/official-stockfish/Stockfish/releases/download/x/y.tar",
            "https://files.pythonhosted.org/packages/a/chess.whl",
            "https://huggingface.co/m/resolve/main/w.bin",
            "https://cdn-lfs.huggingface.co/x",
            "https://raw.githubusercontent.com/a/b/c.txt",
        ):
            self.assertTrue(_host_allowed(url))

    def test_provision_allowlist_blocks_untrusted_and_nonhttp(self):
        for url in (
            "https://evil.example.com/pwn.sh",
            "http://169.254.169.254/latest/meta-data",  # cloud metadata SSRF
            "file:///etc/passwd",
            "https://github.com.attacker.net/payload",
        ):
            self.assertFalse(_host_allowed(url))

    def test_execute_preamble_injected_for_python(self):
        out = _prepare_execute_arguments({"code": "print(pd.__version__)"})
        self.assertIn("import pandas as pd", out["code"])
        self.assertTrue(out["code"].rstrip().endswith("print(pd.__version__)"))

    def test_execute_preamble_skips_future_and_non_python(self):
        future = _prepare_execute_arguments(
            {
                "code": "from __future__ import annotations\nprint(1)",
                "language": "python",
            }
        )
        self.assertTrue(future["code"].startswith("from __future__"))
        bash = _prepare_execute_arguments({"code": "echo hi", "language": "bash"})
        self.assertEqual(bash["code"], "echo hi")

    def test_webvoyager_adapter_reads_machine_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "WebVoyager"
            root.mkdir()
            (root / "run.py").write_text("", encoding="utf-8")
            output_root = tmp_path / "results"

            captured_env = {}
            captured_command: list[str] = []

            def fake_run(command, **kwargs):
                captured_env.update(kwargs["env"])
                captured_command.extend(command)
                output_dir = Path(command[command.index("--output_dir") + 1])
                result_dir = output_dir / "timestamp" / "taskplanner-web"
                result_dir.mkdir(parents=True)
                (result_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "task_id": "planner-web",
                            "status": "success",
                            "answer": "fact",
                            "iterations": 2,
                            "error_context": "",
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch.object(subprocess, "run", fake_run):
                adapter = WebVoyagerAdapter(
                    WebVoyagerConfig(
                        root=root,
                        output_root=output_root,
                        api_key="secret",
                        base_url="https://gateway.example/v1",
                        model="gateway-model",
                        use_gateway_headers=True,
                        max_attached_imgs=1,
                    )
                )

                result = json.loads(adapter.call("Find a fact", "https://example.com"))

            self.assertEqual(result["answer"], "fact")
            self.assertTrue(result["artifact_dir"].endswith("taskplanner-web"))
            self.assertEqual(captured_env["LLM_GATEWAY_KEY"], "secret")
            self.assertEqual(
                captured_env["LLM_GATEWAY_BASE_URL"], "https://gateway.example/v1"
            )
            self.assertEqual(captured_env["LLM_GATEWAY_MODEL"], "gateway-model")
            self.assertEqual(
                captured_command[captured_command.index("--max_attached_imgs") + 1], "1"
            )

    def test_webvoyager_adapter_returns_subprocess_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "WebVoyager"
            root.mkdir()

            def fake_run(command, **_kwargs):
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="browser failed"
                )

            with patch.object(subprocess, "run", fake_run):
                adapter = WebVoyagerAdapter(
                    WebVoyagerConfig(root=root, output_root=tmp_path / "out")
                )

                result = json.loads(adapter.call("Find a fact"))

            self.assertEqual(result["status"], "error")
            self.assertIn("browser failed", result["error"])

    def test_video_adapter_samples_frames_and_extracts_answer(self):
        client = _FakeVisionClient("The busiest frame shows three species.\nANSWER: 3")
        adapter = VideoAnalysisAdapter(client, VideoAnalysisConfig(model="vision"))

        with (
            patch.object(
                VideoAnalysisAdapter,
                "_download",
                lambda self, url, work: Path("/tmp/video.mp4"),
            ),
            patch("gaia_planner.adapters._probe_duration", lambda path: 12.0),
            patch.object(
                VideoAnalysisAdapter,
                "_extract_frames",
                lambda self, video, start, end, interval, work: [
                    (0.0, b"a"),
                    (2.0, b"b"),
                ],
            ),
        ):
            result = json.loads(
                adapter.call(
                    {
                        "url": "https://youtu.be/x",
                        "question": "How many species at once?",
                    }
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "3")
        self.assertEqual(result["frames_analyzed"], 2)
        # The vision request must carry the sampled frames as image parts.
        image_parts = [
            part
            for message in client.messages
            if isinstance(message["content"], list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        self.assertEqual(len(image_parts), 2)

    def test_video_adapter_reports_when_no_frames(self):
        client = _FakeVisionClient("unused")
        adapter = VideoAnalysisAdapter(client, VideoAnalysisConfig(model="vision"))

        with (
            patch.object(
                VideoAnalysisAdapter,
                "_download",
                lambda self, url, work: Path("/tmp/video.mp4"),
            ),
            patch("gaia_planner.adapters._probe_duration", lambda path: 5.0),
            patch.object(
                VideoAnalysisAdapter,
                "_extract_frames",
                lambda self, video, start, end, interval, work: [],
            ),
        ):
            result = json.loads(adapter.call({"url": "u", "question": "q"}))

        self.assertEqual(result["status"], "error")
        self.assertIn("frame", result["error"].lower())


class _FakeVisionClient:
    def __init__(self, content):
        self._content = content
        self.messages = None

        def create(**kwargs):
            self.messages = kwargs["messages"]
            message = SimpleNamespace(content=self._content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


if __name__ == "__main__":
    unittest.main()
