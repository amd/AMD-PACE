# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the reusable execution-agent loop."""

from __future__ import annotations

import json
import unittest
import warnings
from types import SimpleNamespace

from nsjail_agent.agent import AgentConfig, ExecutionAgent
from nsjail_agent.local_executor import LocalExecutor


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


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _message(*, content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _execute_call(call_id: str, code: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="execute_code",
            arguments=json.dumps({"code": code, "language": "python"}),
        ),
    )


class TestExecutionAgent(unittest.TestCase):
    def test_execution_agent_preserves_workspace_between_tool_calls(self):
        client = FakeClient(
            [
                _response(
                    _message(
                        tool_calls=[
                            _execute_call(
                                "call-1", "open('x.txt','w').write('persisted')"
                            )
                        ]
                    )
                ),
                _response(
                    _message(
                        tool_calls=[
                            _execute_call("call-2", "print(open('x.txt').read())")
                        ]
                    )
                ),
                _response(_message(content="FINAL ANSWER: persisted")),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = ExecutionAgent(
                client=client,
                executor=LocalExecutor(),
                config=AgentConfig(model="fake-model", max_turns=2),
            )

        result = agent.run("Read the value back from the generated file.")

        self.assertEqual(result.answer, "FINAL ANSWER: persisted")
        self.assertTrue(
            any(
                message["role"] == "tool" and "persisted" in message["content"]
                for message in result.messages
            )
        )
        self.assertEqual(client.completions.requests[-1]["tool_choice"], "none")

    def test_execution_agent_accepts_pace_agent_chat_client(self):
        class FakeLLMClient:
            def __init__(self, responses):
                self.responses = list(responses)
                self.requests = []

            def chat(self, messages, tools=None, tool_choice="auto"):
                self.requests.append(
                    {"messages": messages, "tools": tools, "tool_choice": tool_choice}
                )
                return self.responses.pop(0)

        llm = FakeLLMClient(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "execute_code",
                            "arguments": json.dumps(
                                {
                                    "code": "print('via-llm-client')",
                                    "language": "python",
                                }
                            ),
                        }
                    ],
                },
                {"content": "FINAL ANSWER: via-llm-client", "tool_calls": []},
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = ExecutionAgent(
                client=llm,
                executor=LocalExecutor(),
                config=AgentConfig(
                    model="fake-model",
                    max_turns=2,
                    force_first_tool=False,
                    force_tool_every_turn=False,
                ),
            )

        result = agent.run("Run code and answer.")

        self.assertEqual(result.answer, "FINAL ANSWER: via-llm-client")
        self.assertEqual(llm.requests[0]["tool_choice"], "auto")
        # Early return on a no-tool final answer uses tool_choice="auto", not "none".
        self.assertEqual(llm.requests[-1]["tool_choice"], "auto")
        self.assertTrue(
            any(
                "via-llm-client" in message.get("content", "")
                for message in result.messages
            )
        )


if __name__ == "__main__":
    unittest.main()
