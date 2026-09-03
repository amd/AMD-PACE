# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy
import os
import pickle
import shutil
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pace_agent.capture import active_capture, capturing
from pace_agent.config import SCHEMA_VERSION
from pace_agent.errors import ReplayDivergence, UnsupportedType
from pace_agent.optimize import optimize
from pace_agent.profile.profiler import import_agent
from pace_agent.replay import codec
from pace_agent.replay.modes import get_replay_mode, replay_modes
from pace_agent.replay.trajectory import ToolCall, Trajectory, register_redactor
from pace_agent.replay import Player, Recorder
from pace_agent.rundir import RunDir


class _StubLLM:
    def __init__(self):
        self._tape = None
        self._rec = False
        self._pos = 0
        self.calls = []

    def record_into(self, tape):
        self._tape, self._rec, self._pos = tape, True, 0

    def replay_from(self, tape):
        self._tape, self._rec, self._pos = list(tape), False, 0

    def complete(self, m):
        if self._tape is not None and not self._rec:
            r = self._tape[self._pos]
            self._pos += 1
            return r
        self.calls.append(0.01)
        if self._rec:
            self._tape.append("ANSWER")
        return "ANSWER"


_AGENT_SRC = """
from pace_agent import get_tool
def run(llm):
    op = get_tool("vector_search")
    args, kw = op.bench_inputs({"N": 200, "dim": 16, "top_k": 5})
    op.run("naive", *args, **kw)
    return llm.complete([{"role": "user", "content": "hi"}])
"""


class TestCodec(unittest.TestCase):
    def test_roundtrip_native_and_numpy(self):
        obj = {
            "a": np.arange(6, dtype="float32").reshape(2, 3),
            "b": b"raw",
            "c": [1, 2.5, "s", None, True],
            "d": {"n": np.float64(3.5)},
        }
        r = codec.loads(codec.dumps(obj))
        self.assertTrue(np.array_equal(r["a"], obj["a"]))
        self.assertEqual(r["a"].dtype, obj["a"].dtype)
        self.assertEqual(r["b"], b"raw")
        self.assertEqual(r["c"], [1, 2.5, "s", None, True])
        self.assertEqual(r["d"]["n"], 3.5)

    def test_dataclass_roundtrip(self):
        tc = ToolCall(
            "vector_search",
            "retrieval",
            False,
            (np.ones(3),),
            {"k": 1},
            np.zeros(2),
            0.01,
        )
        traj = Trajectory(
            tools=[tc], llm_tape=["ANSWER"], llm_time=0.1, e2e=0.2, meta={"agent": "x"}
        )
        r = codec.loads(codec.dumps(traj))
        self.assertIsInstance(r, Trajectory)
        self.assertEqual(r.tools[0].op, "vector_search")
        self.assertTrue(np.array_equal(r.tools[0].output, np.zeros(2)))
        self.assertEqual(r.llm_tape, ["ANSWER"])

    def test_unsupported_type_is_loud(self):
        class Weird:
            pass

        with self.assertRaises(UnsupportedType):
            codec.dumps(Weird())

    def test_descriptor(self):
        d = codec.descriptor(np.zeros((2, 3), dtype="float64"))
        self.assertEqual(d["shape"], [2, 3])
        self.assertEqual(d["dtype"], "float64")
        self.assertIn("digest", d)


class TestCapture(unittest.TestCase):
    def test_nested_isolation(self):
        self.assertIsNone(active_capture())
        with capturing() as c1:
            c1.record_node("n", 0.1)
            with capturing() as c2:
                c2.record_node("m", 0.2)
                self.assertIs(active_capture(), c2)
            self.assertIs(active_capture(), c1)
            self.assertEqual(c1.node_count(), 1)
        self.assertIsNone(active_capture())


class TestReplayModes(unittest.TestCase):
    def test_deterministic_registered(self):
        self.assertIn("deterministic", replay_modes())
        self.assertTrue(callable(get_replay_mode("deterministic")))

    def test_unknown_mode_raises(self):
        with self.assertRaises(KeyError):
            get_replay_mode("no_such_mode")

    def test_multi_replay_defaults_missing_mode_to_deterministic(self):
        import pace_agent.optimize as optimize_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = Trajectory(meta={}).save(tmp)
        with patch.object(
            optimize_module,
            "get_replay_mode",
            return_value=lambda traj, run_dir, opts: {
                "replay_mode": "deterministic",
                "valid": True,
            },
        ) as get_mode:
            optimize_module._replay_one((path, {}))
        get_mode.assert_called_once_with("deterministic")

    def test_mixed_multi_report_has_no_browser_aggregate(self):
        import pace_agent.optimize as optimize_module

        build = getattr(optimize_module, "_multi_report", None)
        self.assertIsNotNone(build)
        report = build(
            [
                {"replay_mode": "deterministic", "valid": True},
                {
                    "replay_mode": "browser-snapshot",
                    "n_snapshots": 1,
                    "observe_parity_all": True,
                    "e2e": {},
                },
            ]
        )
        self.assertEqual(report["mode"], "multi")
        self.assertNotIn("observe_speedup_median", report["aggregate"])

    def test_snapshots_force_browser_snapshot_replay(self):
        import pace_agent.optimize as optimize_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        traj = Trajectory(
            tools=[
                ToolCall("dedup", "cpu", False, ("x",), {}, "x", 0.01),
            ],
            meta={
                "replay_mode": "deterministic",
                "snapshots": ["<html><body>snap</body></html>"],
            },
        )
        traj.save(tmp)
        RunDir(tmp).write_result({"cmd": "profile"})
        requested = []

        def _fake_get_mode(name):
            requested.append(name)
            return lambda traj, run_dir, opts: {
                "replay_mode": "browser-snapshot",
                "n_snapshots": len(traj.meta.get("snapshots") or []),
                "observe_parity_all": True,
                "e2e": {"reference": {"median": 1.0}, "candidate": {"median": 0.5}},
            }

        with (
            patch.object(
                optimize_module, "get_replay_mode", side_effect=_fake_get_mode
            ),
            patch("pace_agent.report.render.render"),
        ):
            optimize_module.optimize(tmp, impl="pace", num_runs=1)
        self.assertEqual(requested, ["browser-snapshot"])

    def test_flat_layout_uses_mode_aware_multi_report(self):
        import pace_agent.optimize as optimize_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "trajectory_task.msgpack")
        with open(path, "wb") as f:
            f.write(codec.dumps(Trajectory(meta={"replay_mode": "deterministic"})))
        RunDir(tmp).write_result({"cmd": "profile"})
        expected = {
            "cmd": "optimize",
            "mode": "multi",
            "aggregate": {"valid": True},
            "tasks": [],
        }
        with (
            patch.object(
                optimize_module,
                "get_replay_mode",
                return_value=lambda traj, run_dir, opts: {
                    "replay_mode": "deterministic",
                    "valid": True,
                },
            ),
            patch.object(
                optimize_module, "_multi_report", return_value=expected
            ) as multi_report,
            patch("pace_agent.report.render.render"),
        ):
            result = optimize_module.optimize(tmp, impl="optimized", num_runs=1)
        multi_report.assert_called_once()
        self.assertIs(result, expected)


class TestTrajectoryFormat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_save_msgpack_0600_and_load(self):
        traj = Trajectory(
            tools=[ToolCall("bm25", "retrieval", False, (), {}, "o", 0.01)],
            llm_tape=["A"],
            e2e=0.1,
            meta={"agent": "x"},
        )
        path = traj.save(self.tmp)
        self.assertTrue(path.endswith("trajectory.msgpack"))
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        r = Trajectory.load(self.tmp)
        self.assertEqual(r.tools[0].op, "bm25")
        self.assertEqual(r.meta["schema_version"], SCHEMA_VERSION)

    def test_refuses_legacy_pickle_without_optin(self):
        with open(os.path.join(self.tmp, "trajectory.pkl"), "wb") as f:
            pickle.dump(Trajectory(meta={"agent": "x"}), f)
        with self.assertRaises(ValueError):
            Trajectory.load(self.tmp)
        r = Trajectory.load(self.tmp, trust_pickle=True)
        self.assertIsInstance(r, Trajectory)

    def test_redactor_applied_before_persist(self):
        register_redactor(lambda v: "[REDACTED]" if v == "SECRET" else v)
        try:
            traj = Trajectory(llm_tape=["SECRET", "ok"], meta={"agent": "x"})
            traj.save(self.tmp)
            r = Trajectory.load(self.tmp)
            self.assertEqual(r.llm_tape, ["[REDACTED]", "ok"])
        finally:
            from pace_agent.replay import trajectory as _t

            _t._REDACTORS.clear()


class TestOptimizeCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        p = os.path.join(self.tmp, "agent.py")
        with open(p, "w") as f:
            f.write(_AGENT_SRC)
        traj = Recorder().record(import_agent(p), _StubLLM(), os.path.abspath(p))
        self.run_dir = os.path.join(self.tmp, "run")
        RunDir(self.run_dir).write_result({"cmd": "profile"})
        traj.save(self.run_dir)

    def test_candidate_dispatch_and_parity(self):
        r = optimize(self.run_dir, impl="optimized", num_runs=2)
        self.assertEqual(r["candidate_impl"], "optimized")
        self.assertEqual(r["coverage"]["candidate_calls"], 1)
        self.assertTrue(r["valid"])
        self.assertTrue(r["parity_all_pass"])

    def test_parity_checks_every_recorded_call(self):
        import pace_agent.optimize as optimize_module

        class Tool:
            name = "multi_call_parity"

            def implementations(self):
                return {"optimized": SimpleNamespace(is_fallback=False)}

            def parity(self, args, kw, **tiers):
                return args[0] == 1

        traj = Trajectory(
            tools=[
                ToolCall("multi_call_parity", "retrieval", False, (1,), {}, 1, 0.1),
                ToolCall("multi_call_parity", "retrieval", False, (2,), {}, 2, 0.1),
            ],
            e2e=1.0,
            meta={"agent": "unused"},
        )
        player = Mock()
        player.replay.return_value = {
            "per_tool": {"multi_call_parity": [0.1, 0.1]},
            "e2e": [1.0, 1.0],
            "contract": {
                "validated": True,
                "tool_calls_expected": 2,
                "tool_calls_consumed": 2,
                "llm_calls_expected": 0,
            },
        }
        player.isolate.return_value = {
            "n_calls": 2,
            "naive_time": 0.2,
            "candidate_time": 0.1,
            "recoverable": 0.1,
        }
        with (
            patch.object(optimize_module, "import_agent", return_value=object()),
            patch.object(optimize_module, "get_tool", return_value=Tool()),
            patch.object(optimize_module, "Player", return_value=player),
        ):
            result = optimize_module._deterministic_replay(
                traj, self.tmp, {"impl": "optimized", "num_runs": 2}
            )
        self.assertFalse(result["parity"]["multi_call_parity"])
        self.assertFalse(result["parity_all_pass"])
        self.assertFalse(result["valid"])
        self.assertIsNone(result["e2e_speedup"])

    def test_honest_e2e_projection(self):
        r = optimize(self.run_dir, impl="optimized", num_runs=2)
        ref = r["e2e"]["reference"]["median"]
        cand = r["e2e"]["candidate"]["median"]
        self.assertLessEqual(cand, ref)
        self.assertGreater(r["coverage"]["covered_time_fraction"], 0.0)

    def test_zero_coverage_is_invalid_not_a_speedup(self):
        r = optimize(
            self.run_dir, impl="pace", num_runs=2
        )  # no pace provider registered
        self.assertEqual(r["coverage"]["candidate_calls"], 0)
        self.assertFalse(r["valid"])
        self.assertIsNone(r["e2e_speedup"])

    def test_available_provider_dispatches(self):
        from pace_agent import get_tool
        from pace_agent.providers import _PROVIDERS, register_provider

        # a fake provider that returns exactly the naive output (parity holds)
        register_provider(
            "fakeprov",
            "vector_search",
            lambda *a, **k: get_tool("vector_search").naive(*a, **k),
            source="fake-pkg",
            version="9.9",
        )
        try:
            r = optimize(self.run_dir, impl="fakeprov", num_runs=2)
            self.assertEqual(r["candidate_impl"], "fakeprov")
            self.assertEqual(r["coverage"]["candidate_calls"], 1)
            self.assertTrue(r["valid"])
            self.assertTrue(r["parity_all_pass"])
            self.assertEqual(r["provider"]["source"], "fake-pkg")
        finally:
            _PROVIDERS.pop(("fakeprov", "vector_search"), None)

    def test_config_match_integrity(self):
        import hashlib
        import json

        from pace_agent.optimize import _config_match

        cfg = {"llm": {"model": "m"}, "num_runs": 10}
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        self.assertIs(
            _config_match(
                Trajectory(meta={"identity": {"config": cfg, "config_digest": digest}})
            ),
            True,
        )
        self.assertIs(
            _config_match(
                Trajectory(meta={"identity": {"config": cfg, "config_digest": "bad"}})
            ),
            False,
        )
        self.assertIsNone(_config_match(Trajectory(meta={"identity": {}})))


class TestReplayContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        p = os.path.join(self.tmp, "agent.py")
        with open(p, "w") as f:
            f.write(_AGENT_SRC)  # one vector_search tool call + one llm.complete
        self.agent = import_agent(p)
        self.traj = Recorder().record(self.agent, _StubLLM(), os.path.abspath(p))

    def _replay(self, traj):
        return Player().replay(traj, self.agent, _StubLLM(), "naive", 1)

    def test_valid_contract(self):
        out = self._replay(self.traj)
        self.assertEqual(out["contract"]["tool_calls_expected"], 1)
        self.assertTrue(out["contract"]["validated"])

    def test_extra_call_beyond_recorded_raises(self):
        short = Trajectory(tools=[], llm_tape=self.traj.llm_tape, meta=self.traj.meta)
        with self.assertRaises(ReplayDivergence):
            self._replay(short)

    def test_missing_recorded_call_raises(self):
        dup = Trajectory(
            tools=self.traj.tools + [self.traj.tools[0]],
            llm_tape=self.traj.llm_tape,
            meta=self.traj.meta,
        )
        with self.assertRaises(ReplayDivergence):
            self._replay(dup)

    def test_op_mismatch_raises(self):
        t = copy.deepcopy(self.traj)
        t.tools[0].op = "bm25"
        with self.assertRaises(ReplayDivergence):
            self._replay(t)

    def test_unconsumed_llm_tape_raises(self):
        t = Trajectory(
            tools=self.traj.tools,
            llm_tape=self.traj.llm_tape + ["leftover"],
            meta=self.traj.meta,
        )
        with self.assertRaises(ReplayDivergence):
            self._replay(t)

    def test_source_drift_invalidates_under_strict(self):
        run_dir = os.path.join(self.tmp, "run")
        RunDir(run_dir).write_result({"cmd": "profile"})
        self.traj.save(run_dir)
        with open(self.traj.meta["agent"], "a") as f:
            f.write("\n# changed after recording\n")
        r = optimize(run_dir, impl="optimized", num_runs=2)
        self.assertIs(r["replay_contract"]["source_match"], False)

    def test_side_effecting_op_not_reexecuted_on_replay(self):
        from pace_agent import op
        from pace_agent.registry import TOOLS

        state = {"runs": 0}

        @op("sideeffect_probe", bucket="custom", side_effect_free=False)
        def _probe(x):
            state["runs"] += 1
            return x + 1

        try:
            src = (
                "from pace_agent import get_tool\n"
                "def run(llm):\n"
                "    get_tool('sideeffect_probe').run('naive', 41)\n"
                "    return llm.complete([{'role': 'user', 'content': 'hi'}])\n"
            )
            p = os.path.join(self.tmp, "sfx_agent.py")
            with open(p, "w") as f:
                f.write(src)
            agent = import_agent(p)
            traj = Recorder().record(agent, _StubLLM(), os.path.abspath(p))
            after_record = state["runs"]
            Player().replay(traj, agent, _StubLLM(), "naive", 3)
            self.assertEqual(state["runs"], after_record)
            self.assertGreaterEqual(after_record, 1)
        finally:
            TOOLS.pop("sideeffect_probe", None)
