# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""The consolidated `pace-agent profile` path: grading, artifacts and the multi-task roll-up."""

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from pace_agent.config import RunConfig
from pace_agent.profile.profiler import profile, profile_task
from pace_agent.rundir import RunDir
from tests.stubs import StubLLM


class _Agent:
    """A browser-style agent double: not replay-safe, records the task it was handed."""

    REPLAY_SAFE = False

    def __init__(self, answer="17"):
        self.answer = answer
        self.seen = None

    def run(self, llm, task):
        self.seen = dict(task or {})
        llm.complete([{"role": "user", "content": self.seen.get("query", "")}])
        return {"answer": self.answer, "reached_answer": True, "iterations": 3}


class TestSingleTaskScoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def _profile(self, task, scoring=None, answer="17", **cfg_kw):
        agent = _Agent(answer)
        cfg = RunConfig(
            tasks=[task],
            scoring=scoring or {},
            warmup=0,
            telemetry="off",
            **{"num_runs": 1, **cfg_kw},
        )
        out = os.path.join(self.tmp, "run")
        with (
            patch("pace_agent.profile.profiler.import_agent", return_value=agent),
            patch("pace_agent.profile.profiler._llm", return_value=StubLLM()),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as err,
        ):
            profile("agent.py", cfg, out)
        return agent, RunDir(out).read_result(), out, err.getvalue()

    def test_expected_answer_is_graded_and_hidden_from_the_agent(self):
        agent, res, _, _ = self._profile(
            {"query": "how many?", "gold": "17"}, {"gold_key": "gold"}
        )
        self.assertNotIn("gold", agent.seen)
        self.assertEqual(agent.seen["query"], "how many?")
        self.assertEqual(res["gold"], "17")
        self.assertTrue(res["match_loose"])
        self.assertEqual(res["scoring"], {"gold_key": "gold"})

    def test_the_configured_gold_key_is_used_and_both_verdicts_reported(self):
        agent, res, _, _ = self._profile(
            {"query": "q", "expected": "17"},
            {"gold_key": "expected"},
            answer="the answer is 17",
        )
        self.assertNotIn("expected", agent.seen)
        self.assertFalse(res["match_strict"])  # the sentence is not the bare answer
        self.assertTrue(res["match_loose"])

    def test_without_a_scoring_block_the_run_is_ungraded(self):
        agent, res, _, _ = self._profile({"query": "q", "gold": "17"})
        self.assertNotIn("scoring", res)
        self.assertNotIn("match_strict", res)
        # still withheld from the agent: enabling grading must not change what the agent was given
        self.assertNotIn("gold", agent.seen)

    def test_num_runs_is_not_silently_ignored(self):
        _, res, _, err = self._profile({"query": "q"}, num_runs=10)
        self.assertIn("not replay-safe", err)
        self.assertIn("num_runs=10", err)
        self.assertEqual(res["e2e"]["n"], 1)

    def test_the_stall_watchdog_is_armed_around_the_task(self):
        with patch("pace_agent.watchdog.stall_limit_s", return_value=0.05):
            agent = _Agent()

            def slow(llm, task):
                import time

                time.sleep(10)

            agent.run = slow
            cfg = RunConfig(tasks=[{"query": "q"}], num_runs=1, warmup=0)
            with (
                patch("pace_agent.profile.profiler.import_agent", return_value=agent),
                patch("pace_agent.profile.profiler._llm", return_value=StubLLM()),
                self.assertRaises(Exception) as caught,
            ):
                profile("agent.py", cfg, os.path.join(self.tmp, "slow"))
        self.assertIn("no progress", str(caught.exception))


class TestFailedTaskScoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_a_failed_task_counts_as_a_miss_and_leaves_artifacts(self):
        out = os.path.join(self.tmp, "query_0")
        cfg = RunConfig(scoring={"gold_key": "gold"})
        buf = io.StringIO()
        with (
            patch(
                "pace_agent.profile.profiler.profile",
                side_effect=RuntimeError("chrome died"),
            ),
            redirect_stdout(buf),
        ):
            res = profile_task(("agent", cfg, {"query": "q", "gold": "17"}, out, 0, 1))
        self.assertEqual(res["error"], "RuntimeError: chrome died")
        self.assertFalse(res["match_strict"])
        self.assertFalse(res["match_loose"])
        self.assertEqual(res["gold"], "17")
        self.assertIn("[profile 1/1] failed: RuntimeError: chrome died", buf.getvalue())
        self.assertEqual(RunDir(out).read_result()["error"], res["error"])


class TestMultiTaskRollup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def _rows(self):
        return [
            {
                "query": "a",
                "answer": "17",
                "completed": True,
                "steps": 3,
                "gold": "17",
                "match_strict": True,
                "match_loose": True,
                "scoring": {"gold_key": "gold"},
                "e2e": {"median": 10.0},
                "per_node": {"observe": {"total_ms": 400.0}},
                "run_metrics": {"llm_calls": 3},
                "observe_ab": {"speedup": 4.0},
                "attribution": {"non_model_cpu": 1.0},
            },
            {
                "query": "b",
                "answer": None,
                "completed": False,
                "steps": 0,
                "gold": "42",
                "match_strict": False,
                "match_loose": False,
                "error": "TaskStalled: task made no progress for 600s",
                "e2e": {"n": 0},
                "per_node": {"observe": {"total_ms": 100.0}},
                "run_metrics": {},
            },
            {
                "query": "c",
                "answer": "9",
                "completed": True,
                "steps": 1,
                "gold": None,
                "match_strict": None,
                "match_loose": None,
                "e2e": {"median": 5.0},
                "per_node": {},
                "run_metrics": {"llm_calls": 1},
            },
        ]

    def _run(self, scoring=None, **cfg_kw):
        from pace_agent.cli import _profile_multi

        cfg = RunConfig(scoring=scoring or {}, **{"parallel": 1, **cfg_kw})
        out = os.path.join(self.tmp, "multi")
        os.makedirs(out, exist_ok=True)
        buf, err = io.StringIO(), io.StringIO()
        with (
            patch("pace_agent.parallel.run_parallel", return_value=self._rows()),
            redirect_stdout(buf),
            redirect_stderr(err),
        ):
            rc = _profile_multi("agent.py", cfg, [{}, {}, {}], out)
        return rc, RunDir(out).read_result(), buf.getvalue(), err.getvalue()

    def test_env_records_the_worker_core_layout(self):
        import json

        self._run(parallel=2, cores_per_worker=3)
        with open(os.path.join(self.tmp, "multi", "env.json")) as f:
            pin = json.load(f)["pinning"]
        self.assertEqual(pin["parallel"], 2)
        self.assertEqual(pin["cores_per_worker"], 3)
        self.assertEqual(pin["worker_cpus"], ["0-2", "3-5"])

    def test_aggregate_counts_answers_errors_and_grades(self):
        rc, res, out, _ = self._run({"gold_key": "gold"})
        agg = res["aggregate"]
        self.assertEqual(rc, 0)
        self.assertEqual(
            (agg["n_queries"], agg["answered"], agg["n_errors"]), (3, 2, 1)
        )
        self.assertEqual(agg["auto_match_loose"], 1)
        self.assertEqual(agg["auto_match_strict"], 1)
        self.assertEqual(agg["n_scored"], 2)  # the failed task stays in the denominator
        self.assertEqual(agg["n_unscored"], 1)
        self.assertEqual(agg["scoring"], {"gold_key": "gold"})
        self.assertIn("match=strict:1 loose:1/2", out)
        rows = {q["query"]: q for q in res["per_query"]}
        self.assertEqual(rows["a"]["match_loose"], True)
        self.assertEqual(rows["b"]["gold"], "42")
        self.assertEqual(rows["c"]["match_loose"], None)

    def test_aggregate_pools_node_time_and_observe_speedup(self):
        _, res, _, _ = self._run()
        agg = res["aggregate"]
        self.assertEqual(agg["per_node_total_ms"], {"observe": 500.0})
        self.assertEqual(agg["observe_speedup_median"], 4.0)
        self.assertEqual(agg["e2e_median_total_s"], 15.0)
        self.assertNotIn("auto_match_loose", agg)

    def test_a_task_lost_with_its_worker_is_named_graded_and_on_disk(self):
        from pace_agent.cli import _profile_multi

        # what run_parallel hands back for a task whose worker was killed: it never reached the
        # in-worker failure path, so the row is bare
        rows = [
            {
                "query": "a",
                "answer": "42",
                "completed": True,
                "steps": 1,
                "gold": "42",
                "match_strict": True,
                "match_loose": True,
                "e2e": {"median": 1.0},
                "per_node": {},
                "run_metrics": {},
            },
            {"error": "worker process died before this task produced a result"},
        ]
        out = os.path.join(self.tmp, "killed")
        os.makedirs(out, exist_ok=True)
        with (
            patch("pace_agent.parallel.run_parallel", return_value=rows),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            rc = _profile_multi(
                "agent.py",
                RunConfig(scoring={"gold_key": "gold"}),
                [{"query": "a", "gold": "42"}, {"query": "lost", "gold": "17"}],
                out,
            )
        res = RunDir(out).read_result()
        lost = res["per_query"][1]
        self.assertEqual(rc, 0)
        self.assertEqual(lost["query"], "lost")
        self.assertEqual((lost["gold"], lost["match_loose"]), ("17", False))
        self.assertEqual(
            res["aggregate"]["n_scored"], 2
        )  # a miss, not an unscored task
        self.assertEqual(res["aggregate"]["auto_match_loose"], 1)
        self.assertEqual(
            RunDir(os.path.join(out, "query_1")).read_result()["query"], "lost"
        )

    def test_the_aggregate_names_the_candidate_impl(self):
        _, res, _, _ = self._run(impl="pace")
        self.assertEqual(res["impl"], "pace")  # else the report renders the impl as "-"

    def test_a_run_where_nothing_succeeded_fails(self):
        from pace_agent.cli import _profile_multi

        rows = [{"query": "a", "error": "boom", "e2e": {"n": 0}, "run_metrics": {}}]
        out = os.path.join(self.tmp, "all_failed")
        os.makedirs(out, exist_ok=True)
        err = io.StringIO()
        with (
            patch("pace_agent.parallel.run_parallel", return_value=rows),
            redirect_stdout(io.StringIO()),
            redirect_stderr(err),
        ):
            rc = _profile_multi("agent.py", RunConfig(), [{}], out)
        self.assertEqual(rc, 1)
        self.assertIn("every task failed", err.getvalue())

    def test_failures_are_repeated_at_the_end_of_the_log(self):
        _, _, _, err = self._run()
        self.assertIn("TaskStalled", err)
        self.assertIn("'b'", err)

    def test_multi_task_run_writes_no_report(self):
        _, _, out, _ = self._run()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "multi", "report.md")))
        self.assertIn("pace-agent report", out)


if __name__ == "__main__":
    unittest.main()
