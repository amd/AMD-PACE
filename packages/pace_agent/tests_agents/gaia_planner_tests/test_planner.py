# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from gaia_planner.models import GaiaTask, PlannerConfig
from gaia_planner.planner import GaiaPlanner, _messages_size, _truncate_text

EXECUTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
    },
}


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


class FakeWeb:
    def __init__(self):
        self.calls = []

    def call(self, goal, start_url):
        self.calls.append((goal, start_url))
        return json.dumps({"status": "success", "answer": "web evidence: 6"})


class FakeExecution:
    schema = EXECUTE_SCHEMA

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def call(self, arguments):
        self.calls.append(arguments)
        return "[status: success]\nSTDOUT:\n42"

    def provision(self, arguments):
        self.provisioned = arguments
        return json.dumps({"status": "success"})


def response(*, content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeVideo:
    def __init__(self):
        self.calls = []

    def call(self, arguments):
        self.calls.append(arguments)
        return json.dumps({"status": "success", "answer": "3", "frames_analyzed": 5})


class BigExecution:
    schema = EXECUTE_SCHEMA

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def call(self, _arguments):
        return self.payload


def make_planner(responses, *, max_turns=4, video=None):
    client = FakeClient(responses)
    web = FakeWeb()
    execution = FakeExecution()
    planner = GaiaPlanner(
        client=client,
        web_adapter=web,
        execution_factory=lambda _task: execution,
        config=PlannerConfig(model="fake", max_turns=max_turns),
        video_adapter=video,
    )
    return planner, client, web, execution


class TestPlanner(unittest.TestCase):
    def test_returns_direct_answer_without_tools(self):
        planner, client, _web, _execution = make_planner(
            [response(content="Reasoning\nFINAL ANSWER: 9")]
        )

        result = planner.run(GaiaTask(task_id="one", question="Count words"))

        self.assertEqual(result.answer, "9")
        self.assertEqual(result.tool_events, [])
        self.assertEqual(client.completions.requests[0]["tool_choice"], "auto")

    def test_rejects_placeholder_answer_and_retries(self):
        planner, client, _web, _execution = make_planner(
            [
                response(
                    content="I looked but could not find it.\nFINAL ANSWER: unknown"
                ),
                response(content="After deriving it.\nFINAL ANSWER: +4.6"),
            ]
        )

        result = planner.run(GaiaTask(task_id="p", question="Compute the percentage"))

        # The "unknown" placeholder must be rejected, forcing a second model call
        # whose concrete answer is the one that gets returned.
        self.assertEqual(result.answer, "+4.6")
        self.assertEqual(len(client.completions.requests), 2)

    def test_composes_web_then_code(self):
        planner, _client, web, execution = make_planner(
            [
                response(
                    tool_calls=[
                        tool_call(
                            "web-1", "browse_web", '{"goal":"Find the source value"}'
                        )
                    ]
                ),
                response(
                    tool_calls=[
                        tool_call(
                            "code-1",
                            "execute_code",
                            '{"code":"print(6 * 7)","language":"python"}',
                        )
                    ]
                ),
                response(content="FINAL ANSWER: 42"),
            ]
        )
        task = GaiaTask(
            task_id="mixed",
            question="Research and calculate",
            start_url="https://example.com",
        )

        result = planner.run(task)

        self.assertEqual(result.answer, "42")
        self.assertEqual(web.calls, [("Find the source value", "https://example.com")])
        self.assertEqual(execution.calls[0]["code"], "print(6 * 7)")
        self.assertEqual(
            [event["name"] for event in result.tool_events],
            ["browse_web", "execute_code"],
        )

    def test_reports_malformed_tool_arguments_to_model(self):
        planner, _client, _web, _execution = make_planner(
            [
                response(tool_calls=[tool_call("bad", "execute_code", "{broken")]),
                response(content="FINAL ANSWER: recovered"),
            ]
        )

        result = planner.run(GaiaTask(task_id="bad", question="Recover"))

        self.assertEqual(result.answer, "recovered")
        self.assertIn('"status": "error"', result.tool_events[0]["output"])

    def test_dispatches_analyze_video_when_configured(self):
        video = FakeVideo()
        planner, client, _web, _execution = make_planner(
            [
                response(
                    tool_calls=[
                        tool_call(
                            "vid-1",
                            "analyze_video",
                            '{"url":"https://youtu.be/x","question":"How many species?"}',
                        )
                    ]
                ),
                response(content="FINAL ANSWER: 3"),
            ],
            video=video,
        )

        result = planner.run(
            GaiaTask(task_id="vid", question="Count birds in the video")
        )

        self.assertEqual(result.answer, "3")
        self.assertEqual(
            video.calls,
            [{"url": "https://youtu.be/x", "question": "How many species?"}],
        )
        self.assertEqual(
            [event["name"] for event in result.tool_events], ["analyze_video"]
        )
        tool_names = {
            tool["function"]["name"] for tool in client.completions.requests[0]["tools"]
        }
        self.assertIn("analyze_video", tool_names)

    def test_dispatches_provision_and_offers_tool(self):
        planner, client, _web, execution = make_planner(
            [
                response(
                    tool_calls=[
                        tool_call(
                            "prov-1",
                            "provision",
                            '{"pip":["python-chess"],'
                            '"downloads":[{"url":"https://github.com/x/y","executable":true}]}',
                        )
                    ]
                ),
                response(content="FINAL ANSWER: done"),
            ]
        )

        result = planner.run(GaiaTask(task_id="prov", question="Need a tool"))

        self.assertEqual(result.answer, "done")
        self.assertEqual(execution.provisioned["pip"], ["python-chess"])
        self.assertEqual([event["name"] for event in result.tool_events], ["provision"])
        tool_names = {
            tool["function"]["name"] for tool in client.completions.requests[0]["tools"]
        }
        self.assertIn("provision", tool_names)

    def test_truncate_text_shortens_and_marks(self):
        text = "A" * 1000
        out = _truncate_text(text, 100)
        self.assertLess(len(out), len(text))
        self.assertIn("truncated", out)
        self.assertEqual(_truncate_text("short", 100), "short")

    def test_truncates_large_tool_output(self):
        client = FakeClient(
            [
                response(tool_calls=[tool_call("c1", "execute_code", '{"code":"1"}')]),
                response(content="FINAL ANSWER: ok"),
            ]
        )
        planner = GaiaPlanner(
            client=client,
            web_adapter=FakeWeb(),
            execution_factory=lambda _task: BigExecution("Z" * 500_000),
            config=PlannerConfig(model="fake", max_turns=4, max_tool_output_chars=500),
        )

        result = planner.run(GaiaTask(task_id="big", question="q"))

        self.assertEqual(result.answer, "ok")
        stored = result.tool_events[0]["output"]
        self.assertLess(len(stored), 2000)
        self.assertIn("truncated", stored)

    def test_enforces_context_budget_by_trimming_old_tool_outputs(self):
        responses = [
            response(tool_calls=[tool_call(f"c{i}", "execute_code", '{"code":"1"}')])
            for i in range(5)
        ]
        responses.append(response(content="FINAL ANSWER: done"))
        client = FakeClient(responses)
        planner = GaiaPlanner(
            client=client,
            web_adapter=FakeWeb(),
            execution_factory=lambda _task: BigExecution("Y" * 1000),
            config=PlannerConfig(
                model="fake",
                max_turns=6,
                max_tool_output_chars=2000,
                context_char_budget=2500,
                keep_recent_tool_outputs=1,
            ),
        )

        result = planner.run(GaiaTask(task_id="budget", question="q"))

        self.assertEqual(result.answer, "done")
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        trimmed = [m for m in tool_messages if "trimmed" in (m.get("content") or "")]
        full = [m for m in tool_messages if m.get("content") == "Y" * 1000]
        self.assertEqual(len(tool_messages), 5)
        # Oldest four dropped to a placeholder; the most recent one (keep=1) preserved.
        self.assertEqual(len(trimmed), 4)
        self.assertEqual(len(full), 1)
        # Trimming keeps the history far smaller than leaving all payloads intact.
        self.assertLess(
            _messages_size(result.messages),
            _messages_size(
                result.messages + [{"role": "tool", "content": "Y" * 1000}] * 4
            ),
        )

    def test_analyze_video_absent_without_adapter(self):
        planner, client, _web, _execution = make_planner(
            [response(content="FINAL ANSWER: ok")]
        )

        planner.run(GaiaTask(task_id="novideo", question="No video here"))

        tool_names = {
            tool["function"]["name"] for tool in client.completions.requests[0]["tools"]
        }
        self.assertNotIn("analyze_video", tool_names)

    def test_reasks_when_final_answer_blank(self):
        planner, _client, _web, _execution = make_planner(
            [
                response(content="Reasoning here.\nFINAL ANSWER:"),
                response(content="FINAL ANSWER: 42"),
            ]
        )

        result = planner.run(GaiaTask(task_id="blank", question="Compute"))

        self.assertEqual(result.answer, "42")
        # The blank answer must be rejected with the empty-answer nudge.
        self.assertTrue(
            any(
                message.get("role") == "user" and "blank" in message.get("content", "")
                for message in result.messages
            )
        )

    def test_nudges_when_final_answer_missing(self):
        planner, client, _web, _execution = make_planner(
            [
                response(content="I could not find the article. Tell me more."),
                response(content="FINAL ANSWER: egalitarian"),
            ]
        )

        result = planner.run(GaiaTask(task_id="nudge", question="Which word?"))

        self.assertEqual(result.answer, "egalitarian")
        # The non-answer must not be accepted, and a follow-up user nudge is injected.
        self.assertTrue(
            any(
                message.get("role") == "user"
                and "FINAL ANSWER:" in message.get("content", "")
                and "did not end" in message.get("content", "")
                for message in result.messages
            )
        )

    def test_forces_final_answer_after_turn_limit(self):
        planner, client, _web, _execution = make_planner(
            [
                response(
                    tool_calls=[
                        tool_call("code-1", "execute_code", '{"code":"print(1)"}')
                    ]
                ),
                response(content="FINAL ANSWER: 1"),
            ],
            max_turns=1,
        )

        result = planner.run(GaiaTask(task_id="limit", question="Calculate"))

        self.assertEqual(result.answer, "1")
        self.assertEqual(client.completions.requests[-1]["tool_choice"], "none")


if __name__ == "__main__":
    unittest.main()
