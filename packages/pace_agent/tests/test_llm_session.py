# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import unittest
import unittest.mock
from types import SimpleNamespace

from pace_agent.llm.client import (
    LLMClient,
    _AnthropicApi,
    _ChatApi,
    _ResponsesApi,
    _content_text,
)


class TestClientSession(unittest.TestCase):
    """B6: the client owns conversation history; the agent sends only the new turn."""

    def test_send_replays_tape_and_accumulates_history(self):
        c = LLMClient(endpoint="", model="")
        c.replay_from(["first", "second"])
        c.reset_session()
        self.assertEqual(c.send([{"role": "user", "content": "hi"}]), "first")
        self.assertEqual(c.send([{"role": "user", "content": "again"}]), "second")
        roles = [m["role"] for m in c._history]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertEqual(c._history[1]["content"], "first")
        self.assertEqual(c._history[3]["content"], "second")

    def test_reset_session_clears_history(self):
        c = LLMClient(endpoint="", model="")
        c.replay_from(["x"])
        c.send([{"role": "user", "content": "a"}])
        c.reset_session()
        self.assertEqual(c._history, [])
        self.assertIsNone(c._prev_response_id)

    def test_every_request_on_both_surfaces_disables_tool_choice(self):
        calls = []

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output_text="ok", output=[], usage=None, id="r")

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    usage=None,
                )

        responses = object.__new__(_ResponsesApi)
        responses.client = SimpleNamespace(responses=Responses())
        chat = object.__new__(_ChatApi)
        chat.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        chat._max_key = "max_tokens"
        for api in (responses, chat):
            api.model = "m"
            api.temperature = 0
            api.seed = 0
            api.max_tokens = 16
            api._send_temp = True

        responses._create([{"role": "user", "content": "hello"}])
        responses.call_stateful([{"role": "user", "content": "hello"}], None)
        responses.ping()
        chat._create([{"role": "user", "content": "hello"}])

        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call["tool_choice"] == "none" for call in calls))


class TestAnthropicBackendSelection(unittest.TestCase):
    """Offline (no network / no `anthropic` install needed): the Claude routing gate and the
    OpenAI->Anthropic request/usage shaping done by the native Messages backend."""

    def test_is_anthropic_gate(self):
        self.assertFalse(LLMClient(endpoint="", model="gpt-oss-20b")._is_anthropic())
        c = LLMClient(endpoint="", model="whatever")
        c.image_format = "anthropic"
        self.assertTrue(c._is_anthropic())
        # a claude* model id routes to the Messages API even with the default image_format
        self.assertTrue(LLMClient(endpoint="", model="Claude-Opus-5")._is_anthropic())

    def test_split_lifts_system_and_keeps_turns(self):
        system, convo = _AnthropicApi._split(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "also be kind"},
            ]
        )
        self.assertEqual(system, "be terse\n\nalso be kind")
        self.assertEqual(
            convo,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )

    def test_split_no_system_is_none(self):
        system, convo = _AnthropicApi._split([{"role": "user", "content": "hi"}])
        self.assertIsNone(system)
        self.assertEqual(convo, [{"role": "user", "content": "hi"}])

    def test_content_text_flattens_multimodal(self):
        self.assertEqual(_content_text("plain"), "plain")
        self.assertEqual(
            _content_text(
                [
                    {"type": "text", "text": "look:"},
                    {"type": "image", "source": {"data": "..."}},
                ]
            ),
            "look:",
        )

    def test_usage_maps_anthropic_fields(self):
        # dict stands in for the SDK usage model (_get reads dicts or attributes)
        u = {
            "input_tokens": 29,
            "output_tokens": 103,
            "cache_read_input_tokens": 5,
            "output_tokens_details": {"thinking_tokens": 32},
        }
        got = _AnthropicApi._usage_of(u)
        self.assertEqual(got["prompt"], 29)
        self.assertEqual(got["completion"], 103)
        self.assertEqual(got["cached"], 5)
        self.assertEqual(got["reasoning"], 32)


class TestAnthropicToolCalling(unittest.TestCase):
    """Offline: the OpenAI<->Anthropic tool dialect translation.

    A native Anthropic endpoint serves only /v1/messages, so tool calling must go
    through the Messages transport rather than /chat/completions.
    """

    def test_chat_backend_uses_messages_api_for_anthropic(self):
        c = LLMClient(endpoint="https://gw/Anthropic", model="Claude-Opus-5")
        with unittest.mock.patch.object(_AnthropicApi, "__init__", lambda s, *a: None):
            self.assertIsInstance(c._chat_backend(), _AnthropicApi)

    def test_chat_backend_stays_on_chat_completions_for_openai(self):
        from pace_agent.llm.client import _ChatApi

        c = LLMClient(endpoint="http://vllm:8000", model="gpt-oss-20b")
        with unittest.mock.patch.object(_ChatApi, "__init__", lambda s, *a: None):
            self.assertIsInstance(c._chat_backend(), _ChatApi)

    def test_tool_schemas_convert_to_input_schema(self):
        params = {"type": "object", "properties": {"city": {"type": "string"}}}
        got = _AnthropicApi._tool_schemas(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "w",
                        "parameters": params,
                    },
                }
            ]
        )
        self.assertEqual(
            got, [{"name": "get_weather", "input_schema": params, "description": "w"}]
        )

    def test_tool_choice_conversion(self):
        self.assertEqual(_AnthropicApi._tool_choice("auto"), {"type": "auto"})
        self.assertEqual(_AnthropicApi._tool_choice("none"), {"type": "none"})
        self.assertEqual(
            _AnthropicApi._tool_choice(
                {"type": "function", "function": {"name": "execute_code"}}
            ),
            {"type": "tool", "name": "execute_code"},
        )

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        system, convo = _AnthropicApi._split_tool_messages(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "run",
                                "arguments": '{"code": "print(1)"}',
                            },
                        }
                    ],
                },
            ]
        )
        self.assertIsNone(system)
        self.assertEqual(convo[1]["role"], "assistant")
        self.assertEqual(
            convo[1]["content"],
            [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "run",
                    # Anthropic wants a JSON object, not the OpenAI string
                    "input": {"code": "print(1)"},
                },
            ],
        )

    def test_tool_results_merge_into_one_user_turn(self):
        """Anthropic requires all tool_results for one assistant turn in a single message."""
        _, convo = _AnthropicApi._split_tool_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "g", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "ra"},
                {"role": "tool", "tool_call_id": "b", "content": "rb"},
            ]
        )
        self.assertEqual(len(convo), 2)
        self.assertEqual(convo[1]["role"], "user")
        self.assertEqual(
            convo[1]["content"],
            [
                {"type": "tool_result", "tool_use_id": "a", "content": "ra"},
                {"type": "tool_result", "tool_use_id": "b", "content": "rb"},
            ],
        )

    def test_malformed_tool_arguments_become_empty_object(self):
        _, convo = _AnthropicApi._split_tool_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "f", "arguments": "{broken"}}
                    ],
                }
            ]
        )
        self.assertEqual(convo[0]["content"][0]["input"], {})

    def test_parse_tools_normalizes_tool_use_blocks(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="on it"),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="execute_code",
                    input={"code": "print(2)"},
                ),
            ],
            usage={"input_tokens": 10, "output_tokens": 3},
        )
        msg, usage = _AnthropicApi._parse_tools(_AnthropicApi, response)
        self.assertEqual(msg["content"], "on it")
        self.assertEqual(
            msg["tool_calls"],
            [
                {
                    "id": "toolu_1",
                    "name": "execute_code",
                    # agents json.loads this, so it must be a string
                    "arguments": '{"code": "print(2)"}',
                }
            ],
        )
        self.assertEqual(usage["prompt"], 10)

    def test_parse_tools_text_only_answer(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="FINAL ANSWER: 42")],
            usage={"input_tokens": 5, "output_tokens": 6},
        )
        msg, _ = _AnthropicApi._parse_tools(_AnthropicApi, response)
        self.assertEqual(msg["content"], "FINAL ANSWER: 42")
        self.assertEqual(msg["tool_calls"], [])


class TestApiSelection(unittest.TestCase):
    """`api` in the config forces a surface; unset keeps the inferred /responses-first probe."""

    def test_config_api_chat_selects_chat_without_probing(self):
        c = LLMClient.from_config(
            {"endpoint": "http://x:8000", "model": "m", "api": "chat"}
        )
        probes = []

        def ping(self):
            probes.append(1)
            return True

        with unittest.mock.patch.object(_ResponsesApi, "ping", ping):
            self.assertEqual(c._backend().kind, "chat")
        self.assertEqual(probes, [])

    def test_config_api_responses_selects_responses_without_probing(self):
        c = LLMClient.from_config(
            {"endpoint": "http://x:8000", "model": "m", "api": "responses"}
        )
        with unittest.mock.patch.object(
            _ResponsesApi, "ping", lambda self: self.fail("probed")
        ):
            self.assertEqual(c._backend().kind, "responses")

    def test_unset_api_probes_and_falls_back_to_chat(self):
        c = LLMClient.from_config({"endpoint": "http://x:8000", "model": "m"})
        with unittest.mock.patch.object(_ResponsesApi, "ping", lambda self: False):
            self.assertEqual(c._backend().kind, "chat")

    def test_unknown_api_is_rejected(self):
        c = LLMClient.from_config(
            {"endpoint": "http://x:8000", "model": "m", "api": "grpc"}
        )
        with self.assertRaises(ValueError):
            c._backend()
