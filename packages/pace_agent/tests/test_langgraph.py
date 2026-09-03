# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import shutil
import tempfile
import threading
import unittest

from pace_agent import instrument, list_tools, op, profile_invoke, with_pace_agent
from pace_agent.node_counter import PaceAgentNodeCounter
from pace_agent.profile.profiler import import_agent
from pace_agent.replay import Recorder


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


def _basic_graph(llm):
    from pace_agent.examples.basic_example import _build

    return _build(llm)


# a plain graph module (no run()): exercises B5 (PACE Agent drives any compiled graph)
_GRAPH_AGENT_SRC = """
from typing import TypedDict
from pace_agent import get_tool

class _S(TypedDict):
    question: str
    answer: str

def _node(state, llm):
    get_tool("tokenize").run("naive", state["question"])
    return {"answer": llm.complete([{"role": "user", "content": state["question"]}])}

def build(llm):
    from langgraph.graph import END, StateGraph
    g = StateGraph(_S)
    g.add_node("work", lambda s: _node(s, llm))
    g.set_entry_point("work")
    g.add_edge("work", END)
    return g.compile()

def inputs(task):
    return {"question": (task or {}).get("query", "hi")}
"""


class TestWithPaceAgent(unittest.TestCase):
    def test_merges_callbacks_and_tags(self):
        sentinel = object()
        cfg = with_pace_agent(
            {"callbacks": [sentinel], "tags": ["keep"]}, run_name="r1"
        )
        self.assertIn(sentinel, cfg["callbacks"])
        self.assertTrue(
            any(isinstance(c, PaceAgentNodeCounter) for c in cfg["callbacks"])
        )
        self.assertIn("keep", cfg["tags"])
        self.assertIn("pace_agent:r1", cfg["tags"])


class TestProfileInvoke(unittest.TestCase):
    def test_node_timings_for_any_graph(self):
        run = profile_invoke(
            _basic_graph(_StubLLM()), {"question": "q"}, return_run=True
        )
        self.assertEqual(run["result"]["answer"], "ANSWER")
        self.assertIn("retrieve", run["node_timings"])
        self.assertIn("reason", run["node_timings"])
        self.assertEqual(run["nodes_traversed"], 2)

    def test_record_writes_replayable_trajectory(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        run = profile_invoke(
            _basic_graph(_StubLLM()), {"question": "q"}, record=d, return_run=True
        )
        self.assertTrue(os.path.exists(os.path.join(d, "trajectory.msgpack")))
        self.assertTrue(run["tool_calls"])

    def test_concurrent_runs_isolated(self):
        results = {}

        def go(key):
            run = profile_invoke(
                _basic_graph(_StubLLM()), {"question": key}, return_run=True
            )
            results[key] = run["nodes_traversed"]

        threads = [threading.Thread(target=go, args=(f"q{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(set(results.values()), {2})


class TestInstrument(unittest.TestCase):
    def test_drop_in_invoke_is_profiled(self):
        g = instrument(_basic_graph(_StubLLM()))
        out = g.invoke({"question": "q"})
        self.assertEqual(out["answer"], "ANSWER")
        self.assertEqual(len(g.pace_agent_runs), 1)
        self.assertIn("retrieve", g.pace_agent_runs[0]["node_timings"])


class TestOpDecorator(unittest.TestCase):
    def test_registers_and_routes_through_hook(self):
        @op("double_it", bucket="custom")
        def double_it(xs):
            return [v * 2 for v in xs]

        self.assertEqual(double_it([1, 2]), [2, 4])
        self.assertIn("double_it", {t.name for t in list_tools()})

        from pace_agent.capture import capturing
        from pace_agent.registry import set_tool_hook

        seen = []
        with capturing():
            set_tool_hook(
                lambda tool, tier, a, k: (
                    seen.append(tool.name),
                    tool.dispatch(tier, *a, **k),
                )[1]
            )
            double_it([3])
            set_tool_hook(None)
        self.assertIn("double_it", seen)


class TestProfileAnyGraph(unittest.TestCase):
    def test_b5_plain_graph_module_recorded(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, "graph_agent.py")
        with open(p, "w") as f:
            f.write(_GRAPH_AGENT_SRC)
        agent = import_agent(p)  # module has build(llm)/inputs(task), NO run()
        traj = Recorder().record(agent, _StubLLM(), p, task={"query": "hello"})
        self.assertEqual(traj.llm_tape, ["ANSWER"])
        self.assertTrue(any(t.op == "tokenize" for t in traj.tools))
        self.assertEqual(traj.run_metrics["nodes_traversed"], 1)
