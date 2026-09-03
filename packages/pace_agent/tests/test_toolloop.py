# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import unittest
from types import SimpleNamespace

from pace_agent.replay import Player, Recorder
from pace_agent.toolloop import assistant_message, parse_arguments, run_tool_loop
from tests.stubs import StubLLM


def _tool_turn(name, args, call_id="c1"):
    return {
        "content": None,
        "tool_calls": [{"id": call_id, "name": name, "arguments": json.dumps(args)}],
    }


def _answer_turn(text):
    return {"content": text, "tool_calls": []}


class TestRunToolLoop(unittest.TestCase):
    def test_dispatches_tool_then_returns_answer(self):
        llm = StubLLM(
            tool_script=[_tool_turn("echo", {"x": 3}), _answer_turn("done: 4")]
        )
        seen = []

        def dispatch(name, arguments):
            args = parse_arguments(arguments)
            seen.append((name, args["x"]))
            return str(args["x"] + 1)

        answer, messages, events = run_tool_loop(
            llm,
            [{"role": "user", "content": "go"}],
            tools=[{"x": 1}],
            dispatch=dispatch,
        )
        self.assertEqual(answer, "done: 4")
        self.assertEqual(seen, [("echo", 3)])
        self.assertEqual(events[0]["output"], "4")
        # the assistant tool_call and the tool result are both threaded into the history
        roles = [m["role"] for m in messages]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

    def test_forces_final_answer_when_turns_exhausted(self):
        # every scripted turn asks for a tool -> loop hits max_turns, then forces one
        # tool_choice="none" answer (the last scripted item, returned once tools are off)
        llm = StubLLM(
            tool_script=[
                _tool_turn("noop", {}),
                _tool_turn("noop", {}),
                _answer_turn("forced"),
            ]
        )
        answer, _m, events = run_tool_loop(
            llm,
            [{"role": "user", "content": "go"}],
            tools=[{}],
            dispatch=lambda n, a: "ok",
            max_turns=2,
        )
        self.assertEqual(answer, "forced")
        self.assertEqual(len(events), 2)

    def test_assistant_message_shape(self):
        msg = _tool_turn("execute_code", {"code": "print(1)"}, call_id="abc")
        am = assistant_message(msg)
        self.assertEqual(am["role"], "assistant")
        self.assertEqual(am["tool_calls"][0]["id"], "abc")
        self.assertEqual(am["tool_calls"][0]["function"]["name"], "execute_code")

    def test_assistant_message_accepts_sdk_shaped_objects(self):
        # OpenAI SDK ChatCompletionMessage uses attributes, not a dict.
        fn = SimpleNamespace(name="browse_web", arguments='{"goal":"x"}')
        call = SimpleNamespace(id="c9", function=fn)
        msg = SimpleNamespace(content=None, tool_calls=[call])
        am = assistant_message(msg)
        self.assertEqual(am["tool_calls"][0]["function"]["name"], "browse_web")
        self.assertEqual(am["tool_calls"][0]["id"], "c9")


class TestSandboxAgentIntegration(unittest.TestCase):
    """The sandbox tool agent is the nsjail integration template: execute_code is a
    side-effecting PACE Agent op (recorded, never re-run on replay)."""

    def _script(self):
        return [
            _tool_turn("execute_code", {"code": "print(sum(range(1, 101)))"}),
            _answer_turn("FINAL ANSWER: 5050"),
        ]

    def test_runs_offline_with_default_executor(self):
        from pace_agent.examples import sandbox_tool_agent as sta

        answer = sta.run(StubLLM(tool_script=self._script()), {"query": "sum 1..100"})
        self.assertIn("5050", answer)

    def test_execute_code_is_side_effecting_and_not_replayed(self):
        from pace_agent.examples import sandbox_tool_agent as sta
        from pace_agent.registry import get_tool

        self.assertFalse(get_tool("execute_code").side_effect_free)

        runs = {"n": 0}

        def counting_executor(code, language="python", stdin=""):
            runs["n"] += 1
            return "5050\n"

        sta.set_executor(counting_executor)
        try:
            traj = Recorder().record(
                sta,
                StubLLM(tool_script=self._script()),
                "pace_agent.examples.sandbox_tool_agent",
            )
            after_record = runs["n"]
            self.assertGreaterEqual(
                after_record, 1
            )  # the real run executed the sandbox
            # find the recorded execute_code call
            ops = [t.op for t in traj.tools]
            self.assertIn("execute_code", ops)
            Player().replay(traj, sta, StubLLM(), "naive", 3)
            self.assertEqual(
                runs["n"], after_record
            )  # replay did NOT re-run the sandbox
        finally:
            sta.set_executor(sta._default_executor)

    def test_replay_contract_validates(self):
        from pace_agent.examples import sandbox_tool_agent as sta

        traj = Recorder().record(
            sta,
            StubLLM(tool_script=self._script()),
            "pace_agent.examples.sandbox_tool_agent",
        )
        out = Player().replay(traj, sta, StubLLM(), "naive", 1)
        self.assertTrue(out["contract"]["validated"])
        self.assertEqual(out["contract"]["tool_calls_expected"], len(traj.tools))


if __name__ == "__main__":
    unittest.main()
