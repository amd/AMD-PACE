# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

"""The LangGraph planner engine must behave identically to the hand-rolled loop
engine: same tool dispatch, nudges, turn budget, final answer, and message trace."""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gaia_planner.langgraph_planner import LangGraphGaiaPlanner
from gaia_planner.models import GaiaTask, PlannerConfig
from gaia_planner.planner import GaiaPlanner

_HAS_LANGGRAPH = importlib.util.find_spec("langgraph.graph") is not None


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    """Replays a scripted list of assistant messages, recording each tool_choice."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.tool_choices: list[str] = []
        outer = self

        class _Completions:
            def create(self, *, model, messages, tools, tool_choice):
                outer.tool_choices.append(tool_choice)
                message = outer._script[outer._i]
                outer._i += 1
                return _response(message)

        self.chat = SimpleNamespace(completions=_Completions())


class _FakeExecution:
    schema = {
        "type": "function",
        "function": {
            "name": "execute_code",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def call(self, arguments):
        return '{"stdout": "42"}'


def _build(cls, script, max_turns=2):
    return cls(
        client=_FakeClient(script),
        web_adapter=SimpleNamespace(call=lambda *a, **k: "{}"),
        execution_factory=lambda task: _FakeExecution(),
        config=PlannerConfig(model="m", max_turns=max_turns),
    )


def _summary(result, planner):
    return (
        result.answer,
        result.raw_answer,
        [event["name"] for event in result.tool_events],
        len(result.messages),
        tuple(planner.client.tool_choices),
    )


SCENARIOS = {
    "tool_then_nudge_then_finalize": [
        _msg(tool_calls=[_tool_call("c1", "execute_code", '{"code": "print(42)"}')]),
        _msg(content="thinking, no final line yet"),
        _msg(content="FINAL ANSWER: 42"),
    ],
    "turn_limit_forces_finalize": [
        _msg(tool_calls=[_tool_call("c1", "execute_code", '{"code": "1"}')]),
        _msg(tool_calls=[_tool_call("c2", "execute_code", '{"code": "2"}')]),
        _msg(content="FINAL ANSWER: done"),
    ],
    "immediate_answer": [
        _msg(content="FINAL ANSWER: right"),
    ],
}


class TestLangGraphPlanner(unittest.TestCase):
    @unittest.skipUnless(_HAS_LANGGRAPH, "langgraph not installed")
    def test_langgraph_engine_matches_loop_engine(self):
        # Base engine-equivalence: the verification gate is a langgraph-only enforcement layer,
        # so disable it here to compare the shared loop behavior of the two engines.
        with patch.dict(os.environ, {"GAIA_ENFORCE_VERIFY": "0"}):
            for name in SCENARIOS:
                with self.subTest(name=name):
                    script = SCENARIOS[name]
                    loop = _build(GaiaPlanner, script)
                    graph = _build(LangGraphGaiaPlanner, script)
                    loop_result = loop.run(GaiaTask(task_id="t", question="q"))
                    graph_result = graph.run(GaiaTask(task_id="t", question="q"))
                    self.assertEqual(
                        _summary(graph_result, graph), _summary(loop_result, loop)
                    )

    @unittest.skipUnless(_HAS_LANGGRAPH, "langgraph not installed")
    def test_langgraph_dispatches_tools_and_answers(self):
        with patch.dict(os.environ, {"GAIA_ENFORCE_VERIFY": "0"}):
            graph = _build(
                LangGraphGaiaPlanner, SCENARIOS["tool_then_nudge_then_finalize"]
            )
            result = graph.run(GaiaTask(task_id="t", question="q"))
        self.assertEqual(result.answer, "42")
        self.assertEqual([e["name"] for e in result.tool_events], ["execute_code"])

    @unittest.skipUnless(_HAS_LANGGRAPH, "langgraph not installed")
    def test_verify_gate_nudges_unverified_final_answer(self):
        # Model finalizes with no sandbox verification -> one verify nudge, then its re-stated
        # answer is accepted. Enforcement is on by default.
        env = {k: v for k, v in os.environ.items() if k != "GAIA_ENFORCE_VERIFY"}
        with patch.dict(os.environ, env, clear=True):
            script = [
                _msg(content="FINAL ANSWER: 7"),
                _msg(content="FINAL ANSWER: 7"),
            ]
            graph = _build(LangGraphGaiaPlanner, script, max_turns=2)
            result = graph.run(GaiaTask(task_id="t", question="q"))
        self.assertEqual(result.answer, "7")
        self.assertEqual(result.tool_events, [])
        # Two agent turns: the initial answer plus the re-statement after the verify nudge.
        self.assertEqual(len(graph.client.tool_choices), 2)
        from gaia_planner.langgraph_planner import VERIFY_BY_EXECUTION_NUDGE

        self.assertTrue(
            any(m.get("content") == VERIFY_BY_EXECUTION_NUDGE for m in result.messages)
        )

    @unittest.skipUnless(_HAS_LANGGRAPH, "langgraph not installed")
    def test_verify_gate_accepts_answer_verified_by_execution(self):
        # Last tool action before finalizing was a successful execute_code -> accept, no nudge.
        env = {k: v for k, v in os.environ.items() if k != "GAIA_ENFORCE_VERIFY"}
        with patch.dict(os.environ, env, clear=True):
            script = [
                _msg(
                    tool_calls=[
                        _tool_call("c1", "execute_code", '{"code": "print(42)"}')
                    ]
                ),
                _msg(content="FINAL ANSWER: 42"),
            ]
            graph = _build(LangGraphGaiaPlanner, script, max_turns=3)
            result = graph.run(GaiaTask(task_id="t", question="q"))
        self.assertEqual(result.answer, "42")
        self.assertEqual([e["name"] for e in result.tool_events], ["execute_code"])
        from gaia_planner.langgraph_planner import VERIFY_BY_EXECUTION_NUDGE

        self.assertFalse(
            any(m.get("content") == VERIFY_BY_EXECUTION_NUDGE for m in result.messages)
        )

    def test_llm_is_verifying_heuristic(self):
        from gaia_planner.langgraph_planner import _llm_is_verifying

        self.assertFalse(_llm_is_verifying([]))
        self.assertTrue(
            _llm_is_verifying([{"name": "execute_code", "output": '{"stdout": "1"}'}])
        )
        self.assertFalse(
            _llm_is_verifying(
                [{"name": "execute_code", "output": "[status: error] boom"}]
            )
        )
        # Most recent action is evidence-gathering, not verification.
        self.assertFalse(
            _llm_is_verifying(
                [
                    {"name": "execute_code", "output": "ok"},
                    {"name": "browse_web", "output": "{}"},
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
