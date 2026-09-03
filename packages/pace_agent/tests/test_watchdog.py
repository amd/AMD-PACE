# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import signal
import threading
import time
import unittest
from unittest.mock import patch

from pace_agent.errors import ConfigError, TaskStalled
from pace_agent.watchdog import (
    DEFAULT_STALL_S,
    ENV_VAR,
    armable,
    beat,
    stall_limit_s,
    watch,
)


class TestStallLimitConfig(unittest.TestCase):
    def test_the_limit_comes_from_the_environment(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop(ENV_VAR, None)
            self.assertEqual(stall_limit_s(), 600.0)
            self.assertEqual(DEFAULT_STALL_S, 600.0)
        with patch.dict("os.environ", {ENV_VAR: "45"}):
            self.assertEqual(stall_limit_s(), 45.0)
        with patch.dict(
            "os.environ", {ENV_VAR: "0"}
        ):  # 0 disables enforcement entirely
            self.assertEqual(stall_limit_s(), 0)
            with watch() as armed:
                self.assertIsNone(armed)

    def test_unusable_values_are_rejected(self):
        for bad in ("soon", "-1"):
            with patch.dict("os.environ", {ENV_VAR: bad}):
                with self.assertRaises(ConfigError):
                    stall_limit_s()


@unittest.skipUnless(armable(), "needs SIGALRM on the main thread")
class TestWatchdog(unittest.TestCase):
    """Unarmed, `watch()` enforces nothing and these would sit out their sleeps instead of failing."""

    def test_silence_beyond_the_limit_stops_the_task(self):
        started = time.time()
        with self.assertRaises(TaskStalled) as caught:
            with watch(label="wedged task", limit=0.2):
                beat("node observe started")
                time.sleep(30)
        self.assertLess(time.time() - started, 10)
        for expected in ("wedged task", "no progress", "node observe started", ENV_VAR):
            self.assertIn(expected, str(caught.exception))

    def test_a_slow_task_that_keeps_progressing_is_left_alone(self):
        # 1.2s of work under a 0.3s limit: total runtime is irrelevant while steps keep landing
        with watch(limit=0.3):
            for i in range(12):
                time.sleep(0.1)
                beat(f"step {i}")

    def test_progress_reported_from_a_worker_thread_counts(self):
        # LangGraph runs nodes on its own threads, so beats must not require the main thread
        stop = threading.Event()

        def worker():
            while not stop.is_set():
                beat("background step")
                time.sleep(0.05)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            with watch(limit=0.3):
                time.sleep(1.2)
        finally:
            stop.set()
            t.join(timeout=1)

    def test_cleanup_runs_and_is_not_interrupted_again(self):
        cleaned = []

        def slow_cleanup():
            time.sleep(0.5)  # longer than the limit: the timer must already be disarmed
            cleaned.append("browser closed")

        with self.assertRaises(TaskStalled):
            with watch(limit=0.2):
                try:
                    time.sleep(30)
                finally:
                    slow_cleanup()
        self.assertEqual(cleaned, ["browser closed"])

    def test_timer_and_handler_are_restored_afterwards(self):
        before = signal.getsignal(signal.SIGALRM)
        with watch(limit=30) as armed:
            self.assertEqual(armed, 30)
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0.0)
        time.sleep(0.1)  # a cancelled timer must not fire into the next task

    def test_an_inner_watch_leaves_the_outer_one_armed(self):
        with watch(label="outer", limit=0.6):
            with watch(label="inner", limit=5):
                time.sleep(0.1)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            with self.assertRaises(TaskStalled) as caught:
                time.sleep(30)
        self.assertIn("outer", str(caught.exception))


@unittest.skipUnless(armable(), "needs SIGALRM on the main thread")
class TestHungGraphNode(unittest.TestCase):
    def test_a_wedged_node_is_stopped_and_its_cleanup_still_runs(self):
        """Injected through LangGraph, not a bare sleep: enforcement rests on a linear graph
        running its node on the calling thread, so that assumption is asserted too."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        seen = {}

        def wedged(state):
            seen["main_thread"] = threading.current_thread() is threading.main_thread()
            try:
                time.sleep(30)  # a hung page load or a model that never answers
            finally:
                seen["cleanup"] = True
            return {"done": True}

        g = StateGraph(TypedDict("S", {"done": bool}, total=False))
        g.add_node("work", wedged)
        g.set_entry_point("work")
        g.add_edge("work", END)

        started = time.time()
        with self.assertRaises(TaskStalled):
            with watch(label="wedged node", limit=0.2):
                g.compile().invoke({})
        self.assertLess(time.time() - started, 10)
        self.assertTrue(seen["cleanup"], "the node's cleanup must still run")
        self.assertTrue(
            seen["main_thread"],
            "a linear graph must run its node on the calling thread for SIGALRM to reach it",
        )


class TestProgressSignals(unittest.TestCase):
    """The instrumented paths must report progress, or a long run would look wedged."""

    def test_node_timings_report_progress(self):
        from pace_agent import node_counter

        with patch("pace_agent.watchdog.beat") as spy:
            node_counter._record("observe", 0.01)
        spy.assert_called_once()
        self.assertIn("observe", spy.call_args[0][0])

    def test_a_replayed_model_response_reports_progress(self):
        from pace_agent.llm.client import LLMClient

        llm = LLMClient(endpoint="http://127.0.0.1:1", model="m")
        llm.replay_from(["cached answer"])
        with patch("pace_agent.llm.client.beat") as spy:
            self.assertEqual(
                llm.complete([{"role": "user", "content": "x"}]), "cached answer"
            )
        spy.assert_called_once()

    def test_a_recorded_tool_call_reports_progress(self):
        from pace_agent.replay import Recorder

        from tests.stubs import StubLLM

        class Agent:
            def run(self, llm):
                from pace_agent.registry import get_tool

                get_tool("tokenize").run("naive", "one two three")
                return llm.complete([{"role": "user", "content": "x"}])

        with patch("pace_agent.replay.recorder.beat") as spy:
            Recorder().record(Agent(), StubLLM(), "test")
        self.assertTrue(any("tokenize" in c[0][0] for c in spy.call_args_list))


if __name__ == "__main__":
    unittest.main()
