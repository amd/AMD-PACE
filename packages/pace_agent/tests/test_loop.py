# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from pace_agent.profile.attribution import attribute
from pace_agent.registry import get_tool
from pace_agent.replay import Player, Recorder
from pace_agent.replay import player as player_module
from pace_agent.replay.trajectory import ToolCall, Trajectory
from pace_agent.report.render import render
from pace_agent.rundir import RunDir


class StubLLM:
    def __init__(self):
        self._tape = None
        self._rec = False
        self._pos = 0
        self.calls = []

    def record_into(self, tape):
        self._tape, self._rec, self._pos = tape, True, 0

    def replay_from(self, tape):
        self._tape, self._rec, self._pos = list(tape), False, 0

    def complete(self, messages):
        if self._tape is not None and not self._rec:
            r = self._tape[self._pos]
            self._pos += 1
            return r
        self.calls.append(0.05)
        if self._rec:
            self._tape.append("ANSWER")
        return "ANSWER"


class StubAgent:
    def run(self, llm):
        q = get_tool("tokenize").run("naive", "which op attributes cpu time")
        get_tool("dedup").run("naive", ["a", "a", "b"])
        return llm.complete([{"role": "user", "content": " ".join(q)}])


def _llm_up(endpoint="http://localhost:8000"):
    """True when *endpoint*'s host:port accepts a TCP connection.

    Protocol-agnostic on purpose (same approach as the WebVoyager live test): it
    avoids guessing the models route (vLLM `/v1/models` vs a gateway `/models`)
    and works for gateways that require auth, where an unauthenticated GET would
    look like "down".
    """
    parsed = urllib.parse.urlparse(str(endpoint or ""))
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False  # e.g. the shipped placeholder, not a URL
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=4):
            return True
    except OSError:
        return False


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_record_replay_deterministic(self):
        agent = StubAgent()
        traj = Recorder().record(agent, StubLLM(), "test")
        self.assertEqual(len(traj.tools), 2)
        self.assertEqual(traj.llm_tape, ["ANSWER"])
        out = Player().replay(traj, agent, StubLLM(), "naive", 3)
        self.assertEqual(len(out["e2e"]), 3)
        self.assertIn("tokenize", out["per_tool"])

    def test_attribution_math(self):
        tools = [
            ToolCall("bm25", "retrieval", False, (), {}, None, 0.010),
            ToolCall("tokenize", "serialization", False, (), {}, None, 0.002),
        ]
        traj = Trajectory(tools=tools, llm_tape=["x"], llm_time=0.1, e2e=0.2)
        a = attribute(traj)
        self.assertEqual(a["model"], 0.1)
        self.assertAlmostEqual(a["non_model_cpu"], 0.012, delta=1e-9)
        self.assertGreater(a["unattributed"], 0)
        self.assertGreater(a["non_model_frac"], 0)
        self.assertLess(a["non_model_frac"], 1)

    def test_isolate_and_render(self):
        op = get_tool("vector_search")
        args, kw = op.bench_inputs({"N": 300, "dim": 16, "top_k": 5})
        traj = Trajectory(
            tools=[ToolCall("vector_search", "retrieval", False, args, kw, None, 0.01)],
            llm_tape=[],
            llm_time=0.0,
            e2e=0.02,
        )
        r = Player().isolate(traj, "vector_search", repeat=2)
        self.assertEqual(r["n_calls"], 1)
        self.assertGreater(r["naive_time"], 0)
        rd = RunDir(self.tmp)
        rd.write_result(
            {
                "cmd": "profile",
                "attribution": attribute(traj),
            }
        )
        path = render(self.tmp)
        with open(path) as f:
            self.assertIn("Attribution", f.read())

    def test_headroom_command_is_removed(self):
        proc = subprocess.run(
            [sys.executable, "-m", "pace_agent.cli", "headroom", self.tmp],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)

    def test_profile_task_prints_progress(self):
        from pace_agent.config import RunConfig
        from pace_agent.profile.profiler import profile_task

        out = os.path.join(self.tmp, "query_0")

        def fake_profile(agent_path, cfg, out_dir):
            RunDir(out_dir).write_result(
                {
                    "query": cfg.tasks[0]["query"],
                    "completed": True,
                    "steps": 2,
                    "e2e": {"median": 1.5},
                }
            )

        buf = StringIO()
        with (
            patch("pace_agent.profile.profiler.profile", side_effect=fake_profile),
            redirect_stdout(buf),
        ):
            profile_task(
                (
                    "agent",
                    RunConfig(),
                    {"query": "demo query"},
                    out,
                    0,
                    1,
                )
            )
        output = buf.getvalue()
        self.assertIn("[profile 1/1] starting", output)
        self.assertIn("[profile 1/1] completed", output)

    def test_profile_task_persists_failure_result(self):
        from pace_agent.config import RunConfig
        from pace_agent.profile.profiler import profile_task

        out = os.path.join(self.tmp, "query_0")
        with patch(
            "pace_agent.profile.profiler.profile",
            side_effect=RuntimeError("profile failed"),
        ):
            result = profile_task(
                (
                    "agent",
                    RunConfig(),
                    {"query": "demo query"},
                    out,
                    0,
                    1,
                )
            )
        saved = RunDir(out).read_result()
        # the exception type is kept: "profile failed" alone doesn't say whether the browser died,
        # the endpoint refused the call, or the task stalled
        self.assertEqual(saved["error"], "RuntimeError: profile failed")
        self.assertEqual(saved["query"], "demo query")
        self.assertFalse(saved["completed"])
        self.assertEqual(result, saved)

    def test_isolate_uses_median_timing(self):
        measure = getattr(player_module, "_median", None)
        self.assertIsNotNone(measure)
        with patch.object(
            player_module.time,
            "perf_counter",
            side_effect=[0.0, 3.0, 3.0, 4.0, 4.0, 6.0],
        ):
            self.assertEqual(measure(lambda: None, 3), 2.0)

    def test_live_profile(self):
        from pace_agent.config import load_config
        from pace_agent.profile.profiler import profile

        cfg = load_config(
            os.path.join(
                os.path.dirname(__file__), "..", "configs", "basic_example.json"
            )
        )
        # Gate on the endpoint this test actually drives, not a hardcoded one:
        # basic_example.json ships with a placeholder, so probing localhost:8000
        # would open the gate for anyone with an unrelated server on that port
        # and then fail on connect instead of skipping.
        endpoint = getattr(cfg, "llm", None) and cfg.llm.get("endpoint")
        if not _llm_up(endpoint):
            self.skipTest(f"no LLM endpoint reachable at {endpoint!r}")
        cfg.num_runs = 1
        cfg.warmup = 0
        out = profile(
            "pace_agent.examples.basic_example", cfg, os.path.join(self.tmp, "run")
        )
        res = RunDir(out).read_result()
        self.assertGreater(res["attribution"]["model"], 0)
        self.assertIn("e2e", res)
