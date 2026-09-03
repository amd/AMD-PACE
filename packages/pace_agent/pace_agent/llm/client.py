# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json as _json
import re as _re
import time

from pace_agent.watchdog import beat


def _get(obj, key):
    """Read `key` from an SDK model (attribute) or a plain dict; None if absent."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _usage(prompt=None, completion=None, cached=None, reasoning=None):
    """Normalized token accounting shared by both API surfaces (None = not reported)."""
    return {
        "prompt": prompt,
        "completion": completion,
        "cached": cached,  # input tokens served from the prefix cache
        "reasoning": reasoning,  # output tokens spent "thinking"
    }


_INLINE_TOOL_CALL_RE = _re.compile(
    r"<\|tool_call_begin\|>(?P<id>.*?)<\|tool_call_argument_begin\|>"
    r"(?P<args>.*?)<\|tool_call_end\|>",
    _re.DOTALL,
)
_INLINE_TOOL_SECTION_RE = _re.compile(
    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", _re.DOTALL
)


def _parse_inline_tool_calls(content):
    """Recover tool calls that a model emitted as SPECIAL TOKENS in the message *content*
    instead of the structured ``tool_calls`` field. Some OpenAI-compatible gateways (e.g.
    Moonshot/Kimi models) return, as plain content::

        <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX
        <|tool_call_argument_begin|>{...json...}<|tool_call_end|><|tool_calls_section_end|>

    Without this, an agent loop sees no tool_calls and leaks the raw tokens as a final answer.
    Returns ``(calls, cleaned_content)``; ``calls`` is ``[]`` (and content is unchanged) when
    there are no such tokens."""
    if not content or "<|tool_call_begin|>" not in content:
        return [], content
    calls = []
    for raw in _INLINE_TOOL_CALL_RE.finditer(content):
        token_id = (raw.group("id") or "").strip()
        args = (raw.group("args") or "").strip()
        # token_id looks like "functions.browse_web:8" -> tool name "browse_web"
        name = token_id.split(".", 1)[1] if "." in token_id else token_id
        name = name.split(":", 1)[0].strip()
        if not name:
            continue
        calls.append(
            {
                "id": "call_"
                + _re.sub(r"[^0-9A-Za-z_]", "_", token_id or str(len(calls))),
                "name": name,
                "arguments": args or "{}",
            }
        )
    if not calls:
        return [], content
    cleaned = _INLINE_TOOL_SECTION_RE.sub("", content).strip() or None
    return calls, cleaned


def _json_object(raw):
    """Parse a tool-call ``arguments`` payload into a dict (Anthropic wants an object)."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = _json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _unsupported(e):
    """True when the /responses ROUTE itself is not served (so we fall back to /chat).
    Narrow by design: a per-parameter 400 must NOT count. We treat as route-unsupported a
    404, the gateway's explicit api_not_supported signal, and the Azure/OpenAI gateway 400
    that reports the Responses API is not enabled for the negotiated api-version (e.g.
    "Responses API is enabled only for api-version 2025-03-01-preview and later") — on such a
    gateway the route is effectively unavailable, so /chat/completions is the right fallback.
    Also covers the AMD-style gateway that reports no Responses deployment at all
    (``Deployment of "v1" for "Responses" is not found!``).
    """
    if getattr(e, "status_code", None) == 404:
        return True
    msg = str(getattr(e, "message", "") or e).lower()
    return (
        "api_not_supported" in msg
        or "requested api is currently not supported" in msg
        or "responses api is enabled only for api-version" in msg
        or "responses api is not enabled" in msg
        or "responses api is not supported" in msg
        # gateway routes the model but has no Responses deployment behind it
        or ('for "responses" is not found' in msg)
        or ("responses" in msg and "deployment" in msg and "not found" in msg)
    )


def _sanitize_model_text(text):
    """Collapse a provider *refusal* payload that leaked into the ``content`` field.

    Some gateways, when the upstream model issues an Anthropic safety refusal, place the
    raw provider message object (e.g. ``{"stop_reason": "refusal", "content": [], ...}``)
    verbatim into the OpenAI-shaped ``content`` string. Returned as-is it pollutes the
    agent's final answer and the recorded trajectory, so detect that specific shape and
    collapse it to an empty string — a refusal carries no usable answer. Any other content
    (including ordinary JSON answers) is returned untouched."""
    if not isinstance(text, str):
        return text
    s = text.strip()
    if not (s.startswith("{") and "refusal" in s):
        return text
    try:
        obj = _json.loads(s)
    except (ValueError, TypeError):
        return text
    if not isinstance(obj, dict):
        return text
    details = obj.get("stop_details")
    is_refusal = obj.get("stop_reason") == "refusal" or (
        isinstance(details, dict) and details.get("type") == "refusal"
    )
    return "" if is_refusal else text


def _is_transient_upstream(e) -> bool:
    """True for gateway errors worth retrying: the usual retryable HTTP statuses, an
    unknown status (network/timeout), *and* a 404 that carries the gateway's transient
    ``upstream_error`` / ``api_not_supported`` signal.

    The last case matters for the AMD gateway: a valid chat model intermittently 404s with
    ``api_not_supported`` under load, and a single such drop otherwise aborts the whole
    task. It is only ever consulted on the /chat surface (see callers), where the route is
    already known to exist, so retrying a transient 404 there cannot mask a genuinely
    missing model — the fail-fast preflight catches that up front."""
    status = getattr(e, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504) or status is None:
        return True
    if status == 404:
        msg = str(getattr(e, "message", "") or e).lower()
        return (
            "api_not_supported" in msg
            or "upstream_error" in msg
            or "requested api is currently not supported" in msg
        )
    return False


class _Api:
    """One LLM API surface. Owns its OpenAI client (built once here, reused for every call).

    Subclasses build the request in their native shape (`_create`) and parse `(text, usage)`
    back (`_parse`); `call()` adds a param-compat retry: if the model rejects `temperature`
    (reasoning models do), drop it and retry, remembering the fix. Instantiated lazily by
    LLMClient on the first real call, so `openai` stays optional for replay/bench clients.
    """

    kind = "?"

    def __init__(
        self, endpoint, model, temperature, seed, max_tokens, api_key, headers
    ):
        from openai import OpenAI

        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.client = OpenAI(
            base_url=base, api_key=api_key, default_headers=headers, timeout=120
        )
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self._send_temp = True  # send `temperature` until the model rejects it

    def call(self, messages):
        # relax-and-retry: on a 400 that names an unsupported param, learn the fix and retry.
        # `_relax` only returns True when it changed state (monotonic flags) -> loop terminates.
        while True:
            try:
                return self._parse(self._create(messages))
            except Exception as e:  # noqa: BLE001
                if self._relax(e):
                    continue
                raise

    def _relax(self, e):
        if (
            getattr(e, "status_code", None) == 400
            and self._send_temp
            and "temperature" in str(getattr(e, "message", "") or e).lower()
        ):
            self._send_temp = False  # reasoning models reject a non-default temperature
            return True
        return False

    def _create(self, messages):
        raise NotImplementedError

    def _parse(self, r):
        raise NotImplementedError


class _ChatApi(_Api):
    """OpenAI /chat/completions — the universal path; consumes chat messages as-is."""

    kind = "chat"

    def __init__(self, *args):
        super().__init__(*args)
        self._max_key = "max_tokens"  # reasoning models want "max_completion_tokens"

    def _create(self, messages):
        kw = {
            "model": self.model,
            "messages": messages,
            "seed": self.seed,
            self._max_key: self.max_tokens,
            "tool_choice": "none",
        }
        if self._send_temp:
            kw["temperature"] = self.temperature
        return self.client.chat.completions.create(**kw)

    def _parse(self, r):
        m = r.choices[0].message
        text = m.content or getattr(m, "reasoning_content", "") or ""
        return _sanitize_model_text(text), self._usage_of(getattr(r, "usage", None))

    def _relax(self, e):
        if (
            getattr(e, "status_code", None) == 400
            and self._max_key == "max_tokens"
            and "max_completion_tokens" in str(getattr(e, "message", "") or e).lower()
        ):
            self._max_key = "max_completion_tokens"
            return True
        return super()._relax(e)

    def call_tools(self, messages, tools, tool_choice):
        """Tool-calling turn: like ``call`` but advertises ``tools`` and returns the full
        assistant message (content + any tool_calls) in a backend-agnostic normalized shape.
        Shares the same relax-and-retry as ``call``."""
        while True:
            try:
                return self._parse_tools(
                    self._create_tools(messages, tools, tool_choice)
                )
            except Exception as e:  # noqa: BLE001
                if self._relax(e):
                    continue
                raise

    def _create_tools(self, messages, tools, tool_choice):
        kw = {
            "model": self.model,
            "messages": messages,
            "seed": self.seed,
            self._max_key: self.max_tokens,
        }
        if self._send_temp:
            kw["temperature"] = self.temperature
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kw)

    def _parse_tools(self, r):
        m = r.choices[0].message
        calls = [
            {
                "id": getattr(c, "id", None),
                "name": c.function.name,
                "arguments": c.function.arguments,
            }
            for c in (getattr(m, "tool_calls", None) or [])
        ]
        content = m.content
        if not calls:
            # some gateways surface tool calls as special tokens in the content rather than
            # the structured field — recover them so they aren't leaked as a final answer.
            inline, content = _parse_inline_tool_calls(content)
            if inline:
                calls = inline
            else:
                # no tool calls: a bare refusal payload would otherwise become the answer.
                content = _sanitize_model_text(content)
        msg = {"content": content, "tool_calls": calls}
        return msg, self._usage_of(getattr(r, "usage", None))

    @staticmethod
    def _usage_of(u):
        return _usage(
            prompt=_get(u, "prompt_tokens"),
            completion=_get(u, "completion_tokens"),
            cached=_get(_get(u, "prompt_tokens_details"), "cached_tokens"),
            reasoning=_get(_get(u, "completion_tokens_details"), "reasoning_tokens"),
        )


class _ResponsesApi(_Api):
    """OpenAI /responses — richer usage (cached prefix + reasoning tokens); rewrites messages
    into the Responses `input` shape."""

    kind = "responses"

    def _create(self, messages):
        kw = {
            "model": self.model,
            "input": self._to_input(messages),
            "max_output_tokens": self.max_tokens,
            "tool_choice": "none",
        }
        if self._send_temp:
            kw["temperature"] = self.temperature
        return self.client.responses.create(**kw)

    def _parse(self, r):
        return _sanitize_model_text(self._text(r)), self._usage_of(
            getattr(r, "usage", None)
        )

    def call_stateful(self, new_messages, previous_response_id):
        """Server-side threaded turn: send only the new input with store=true and (after the
        first turn) previous_response_id. Returns (text, usage, response_id)."""
        while True:
            try:
                kw = {
                    "model": self.model,
                    "input": self._to_input(new_messages),
                    "max_output_tokens": self.max_tokens,
                    "store": True,
                    "tool_choice": "none",
                }
                if previous_response_id:
                    kw["previous_response_id"] = previous_response_id
                if self._send_temp:
                    kw["temperature"] = self.temperature
                r = self.client.responses.create(**kw)
                return (
                    self._text(r),
                    self._usage_of(getattr(r, "usage", None)),
                    getattr(r, "id", None),
                )
            except Exception as e:  # noqa: BLE001
                if self._relax(e):
                    continue
                raise

    def ping(self):
        """One-shot capability check: does this endpoint actually serve /responses?"""
        try:
            self.client.responses.create(
                model=self.model,
                input="ping",
                max_output_tokens=16,
                tool_choice="none",
            )
            return True
        except (
            Exception
        ) as e:  # noqa: BLE001 - a non-route error still means the route exists
            return not _unsupported(e)

    @staticmethod
    def _to_input(messages):
        # string content passes straight through; only list content (multimodal) is rewritten
        # into input_text / input_image parts.
        def part(p):
            if not isinstance(p, dict):
                return {"type": "input_text", "text": str(p)}
            if p.get("type") == "text":
                return {"type": "input_text", "text": p.get("text", "")}
            if p.get("type") == "image_url":
                u = p.get("image_url")
                return {
                    "type": "input_image",
                    "image_url": u.get("url") if isinstance(u, dict) else u,
                }
            return p

        out = []
        for m in messages:
            c = m.get("content", "")
            out.append(
                {
                    "role": m.get("role", "user"),
                    "content": [part(p) for p in c] if isinstance(c, list) else c,
                }
            )
        return out

    @staticmethod
    def _text(r):
        txt = getattr(r, "output_text", None)
        if txt:
            return txt
        chunks = []
        for item in getattr(r, "output", None) or []:
            if _get(item, "type") != "message":
                continue
            for p in _get(item, "content") or []:
                if _get(p, "type") in ("output_text", "text"):
                    chunks.append(_get(p, "text") or "")
        return "".join(chunks)

    @staticmethod
    def _usage_of(u):
        return _usage(
            prompt=_get(u, "input_tokens"),
            completion=_get(u, "output_tokens"),
            cached=_get(_get(u, "input_tokens_details"), "cached_tokens"),
            reasoning=_get(_get(u, "output_tokens_details"), "reasoning_tokens"),
        )


def _content_text(content):
    """Flatten OpenAI-style message content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else ""
            for p in content
        ).strip()
    return str(content or "")


class _AnthropicApi(_Api):
    """Anthropic Messages API (POST /v1/messages) via the `anthropic` SDK — the transport
    for Claude models. Differences from the OpenAI surface: `max_tokens` is required (not
    `max_completion_tokens`), system prompts are a top-level `system` param (not a message),
    the reply `content` is a list of blocks, and usage reports input/output (+ cache-read,
    thinking) tokens. Built lazily by LLMClient so `anthropic` stays an optional dependency.
    """

    kind = "anthropic"

    def __init__(
        self, endpoint, model, temperature, seed, max_tokens, api_key, headers
    ):
        import anthropic

        base = endpoint.rstrip("/")
        if base.endswith("/v1"):  # the SDK appends /v1/messages itself
            base = base[:-3].rstrip("/")
        self.client = anthropic.Anthropic(
            base_url=base or None,
            api_key=api_key
            or "dummy",  # real auth is a header (e.g. gateway subscription)
            default_headers=headers,
            timeout=120,
        )
        self.model = model
        self.temperature = temperature
        self.seed = seed  # unused by Anthropic; kept for constructor parity
        self.max_tokens = max_tokens
        self._send_temp = True

    @staticmethod
    def _split(messages):
        """Lift system messages to Anthropic's top-level `system`; keep user/assistant turns."""
        system, convo = [], []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                system.append(_content_text(m.get("content", "")))
            else:
                convo.append({"role": role, "content": m.get("content", "")})
        return ("\n\n".join(s for s in system if s) or None), convo

    def _create(self, messages):
        system, convo = self._split(messages)
        kw = {"model": self.model, "max_tokens": self.max_tokens, "messages": convo}
        if system:
            kw["system"] = system
        if self._send_temp:
            kw["temperature"] = self.temperature
        return self.client.messages.create(**kw)

    def _parse(self, r):
        text = "".join(
            getattr(b, "text", "") or ""
            for b in (getattr(r, "content", None) or [])
            if getattr(b, "type", None) == "text"
        )
        return text, self._usage_of(getattr(r, "usage", None))

    # -- tool calling -------------------------------------------------------
    # PACE Agent agents speak the OpenAI tool dialect (flat `tools` schemas, a
    # `role: "tool"` result message). Anthropic instead nests tool activity in
    # content blocks, so translate both directions here rather than making a
    # native Messages endpoint look like /chat/completions.

    def call_tools(self, messages, tools, tool_choice):
        """Tool-calling turn against /v1/messages, in the normalized PACE Agent shape."""
        while True:
            try:
                return self._parse_tools(
                    self._create_tools(messages, tools, tool_choice)
                )
            except Exception as e:  # noqa: BLE001
                if self._relax(e):
                    continue
                raise

    @staticmethod
    def _tool_schemas(tools):
        """OpenAI ``{"function": {...}}`` schemas -> Anthropic ``input_schema`` schemas."""
        out = []
        for tool in tools or []:
            fn = tool.get("function", tool) if isinstance(tool, dict) else tool
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            schema = {"name": fn["name"], "input_schema": fn.get("parameters") or {}}
            if fn.get("description"):
                schema["description"] = fn["description"]
            out.append(schema)
        return out

    @staticmethod
    def _tool_choice(choice):
        """OpenAI tool_choice -> Anthropic tool_choice (None means "send no choice")."""
        if isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            return {"type": "tool", "name": name} if name else {"type": "auto"}
        if choice == "none":
            return {"type": "none"}
        if choice == "required":
            return {"type": "any"}
        return {"type": "auto"}

    @classmethod
    def _split_tool_messages(cls, messages):
        """Convert an OpenAI-shaped history into ``(system, anthropic_messages)``.

        Assistant ``tool_calls`` become ``tool_use`` blocks and ``role: "tool"``
        messages become ``tool_result`` blocks. Anthropic requires every
        ``tool_use`` in one assistant turn to be answered by ``tool_result``
        blocks in a *single* following user message, so consecutive tool results
        are merged instead of emitted as one message each.
        """
        system, convo = [], []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                system.append(_content_text(message.get("content", "")))
                continue
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id"),
                    "content": _content_text(message.get("content", "")) or "",
                }
                # Append to the open tool-result turn when there is one.
                if (
                    convo
                    and convo[-1]["role"] == "user"
                    and isinstance(convo[-1]["content"], list)
                    and convo[-1]["content"]
                    and convo[-1]["content"][0].get("type") == "tool_result"
                ):
                    convo[-1]["content"].append(block)
                else:
                    convo.append({"role": "user", "content": [block]})
                continue
            if role == "assistant" and message.get("tool_calls"):
                blocks = []
                text = _content_text(message.get("content", ""))
                if text:
                    blocks.append({"type": "text", "text": text})
                for call in message["tool_calls"]:
                    fn = call.get("function") or call
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id"),
                            "name": fn.get("name"),
                            "input": _json_object(fn.get("arguments")),
                        }
                    )
                convo.append({"role": "assistant", "content": blocks})
                continue
            convo.append({"role": role, "content": message.get("content", "")})
        return ("\n\n".join(s for s in system if s) or None), convo

    def _create_tools(self, messages, tools, tool_choice):
        system, convo = self._split_tool_messages(messages)
        kw = {"model": self.model, "max_tokens": self.max_tokens, "messages": convo}
        if system:
            kw["system"] = system
        if self._send_temp:
            kw["temperature"] = self.temperature
        schemas = self._tool_schemas(tools)
        if schemas:
            kw["tools"] = schemas
            kw["tool_choice"] = self._tool_choice(tool_choice)
        return self.client.messages.create(**kw)

    def _parse_tools(self, r):
        text_parts, calls = [], []
        for block in getattr(r, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif kind == "tool_use":
                calls.append(
                    {
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        # PACE Agent agents parse `arguments` as a JSON string.
                        "arguments": _json.dumps(getattr(block, "input", None) or {}),
                    }
                )
        content = "".join(text_parts) or None
        if not calls:
            content = _sanitize_model_text(content)
        return {"content": content, "tool_calls": calls}, self._usage_of(
            getattr(r, "usage", None)
        )

    @staticmethod
    def _usage_of(u):
        return _usage(
            prompt=_get(u, "input_tokens"),
            completion=_get(u, "output_tokens"),
            cached=_get(u, "cache_read_input_tokens"),
            reasoning=_get(_get(u, "output_tokens_details"), "thinking_tokens"),
        )


class LLMClient:
    def __init__(
        self,
        endpoint,
        model,
        temperature=0.0,
        seed=0,
        max_tokens=1024,
        api_key="none",
        headers=None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.api_key = api_key or "none"  # Bearer token (OpenAI-compatible)
        self.headers = headers or None  # extra headers (e.g. gateway subscription key)
        self.api = None  # pinned surface "chat" | "responses"; None = infer
        self.vision = False  # backend can accept images (set via from_config)
        self.image_format = "openai"  # openai (vLLM) | anthropic (Claude)
        self.reasoning = None  # gpt-oss harmony reasoning effort: low|medium|high
        self.no_tools = (
            False  # closed-book: never advertise tools to the model (chat())
        )
        self._tape = None  # list of recorded responses (replay) or sink (record)
        self._pos = 0
        self._recording = False
        self._api = None  # backend, selected + built once on first real call
        self._chat_api = None  # lazy; see _chat_backend()
        self._responses_ok = True  # False once /responses is probed and unsupported
        self.calls = []  # per-call timings
        self.prompt_tokens = []
        self.completion_tokens = []
        self.cached_tokens = []
        self.reasoning_tokens = []
        self._history = (
            []
        )  # client-owned conversation (B6): the agent sends only new turns
        self._prev_response_id = None  # /responses server-side thread id
        self._store_ok = True  # server-side storage available (learned on first use)

    @classmethod
    def from_config(cls, llm):
        """Build from a cfg.llm dict. Secrets come from the environment, never the config file:
        `api_key_env` names the env var holding the key; `subscription_header` (optional) sends
        that same key as a custom header.
        """
        import os

        llm = llm or {}
        key = os.environ.get(llm["api_key_env"]) if llm.get("api_key_env") else None
        key = key or llm.get(
            "api_key"
        )  # direct value fallback (discouraged; prefer env)
        headers = (
            {llm["subscription_header"]: key}
            if (llm.get("subscription_header") and key)
            else None
        )
        c = cls(
            endpoint=llm.get("endpoint", "http://localhost:8000"),
            model=llm.get("model", ""),
            temperature=llm.get("temperature", 0.0),
            seed=llm.get("seed", 0),
            max_tokens=llm.get("max_tokens", 2048),
            api_key=key or "none",
            headers=headers,
        )
        c.api = llm.get("api") or None
        c.vision = bool(llm.get("vision", False))
        c.image_format = llm.get("image_format", "openai")  # "anthropic" for Claude
        c.reasoning = llm.get("reasoning")  # gpt-oss reasoning effort (opt-in)
        c.no_tools = bool(llm.get("no_tools", False))  # closed-book mode (opt-in)
        return c

    def image_content(self, b64, media_type="image/png"):
        """A single image block in the backend's expected shape. Default is OpenAI `image_url`
        (vLLM, gpt-5.x, Llama-4); Claude needs the Anthropic `image`/`source`
        block (set `image_format: "anthropic"`)."""
        if self.image_format == "anthropic":
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        }

    def record_into(self, tape):
        self._tape, self._recording, self._pos = tape, True, 0

    def replay_from(self, tape):
        self._tape, self._recording, self._pos = list(tape), False, 0

    def metrics(self):
        """Token accounting from the API `usage`; each field is None when unavailable
        (replay tape, or a backend that does not report it)."""

        def tot(xs):
            v = [x for x in xs if x is not None]
            return sum(v) if v else None

        out = tot(self.completion_tokens)
        rea = tot(self.reasoning_tokens)
        return {
            "llm_calls": len(self.calls),
            "api": self._api.kind if self._api else None,
            "context_tokens_total": tot(self.prompt_tokens),
            "context_tokens_max": max(
                [t for t in self.prompt_tokens if t is not None], default=None
            ),
            "cached_tokens_total": tot(self.cached_tokens),
            "output_tokens_total": out,
            "reasoning_tokens_total": rea,
            "work_tokens_total": (
                (out - rea) if (out is not None and rea is not None) else out
            ),
        }

    def complete(self, messages):
        if self._tape is not None and not self._recording:
            resp = self._tape[self._pos]
            self._pos += 1
            beat("replayed model response")
            return resp
        from pace_agent import shims

        t = time.perf_counter()
        with shims.tagged("model"):  # LLM time is model, not tool-io (7.13a)
            text, u = self._send(messages)
        self.calls.append(time.perf_counter() - t)
        self.prompt_tokens.append(u["prompt"])
        self.completion_tokens.append(u["completion"])
        self.cached_tokens.append(u["cached"])
        self.reasoning_tokens.append(u["reasoning"])
        if self._recording:
            self._tape.append(text)
        return text

    def chat(self, messages, tools=None, tool_choice="auto"):
        """One OpenAI-style tool-calling turn.

        Returns a normalized assistant message ``{"content": str|None, "tool_calls":
        [{"id", "name", "arguments"}]}`` — the shape an agent loop needs to decide whether
        to call a tool or answer. Recorded/replayed on the SAME tape as ``complete`` (each
        entry is just the returned message), so a function-calling agent is deterministically
        replayable. Tool calls always use the /chat/completions surface.

        In closed-book mode (``no_tools``) any supplied ``tools`` are dropped so the model is
        never told a tool exists; it must answer from its own reasoning.
        """
        if self.no_tools:
            tools, tool_choice = None, "none"
        if self._tape is not None and not self._recording:
            msg = self._tape[self._pos]
            self._pos += 1
            return msg
        from pace_agent import shims

        t = time.perf_counter()
        with shims.tagged("model"):
            msg, u = self._send_tools(messages, tools, tool_choice)
        self.calls.append(time.perf_counter() - t)
        self.prompt_tokens.append(u["prompt"])
        self.completion_tokens.append(u["completion"])
        self.cached_tokens.append(u["cached"])
        self.reasoning_tokens.append(u["reasoning"])
        if self._recording:
            self._tape.append(msg)
        return msg

    def _chat_backend(self):
        """Return the lazily built backend used for tool calling.

        On an OpenAI-style endpoint tool calling is a chat-completions concept, so route
        ``chat`` through /chat/completions even when :meth:`_backend` inferred /responses
        for plain ``complete``/``send`` (responses has no tool-calling surface). A NATIVE
        Anthropic endpoint serves only /v1/messages — there is no /chat/completions to
        fall back to — so tool calling goes through the Messages transport instead, which
        translates the OpenAI tool dialect in :meth:`_AnthropicApi.call_tools`.

        Returns:
            An ``_Api`` exposing ``call_tools`` for the endpoint's real wire protocol.
        """
        if self._chat_api is None:
            args = (
                self.endpoint,
                self.model,
                self.temperature,
                self.seed,
                self.max_tokens,
                self.api_key,
                self.headers,
            )
            self._chat_api = (
                _AnthropicApi(*args) if self._is_anthropic() else _ChatApi(*args)
            )
        return self._chat_api

    def _send_tools(self, messages, tools, tool_choice):
        """Call ``/chat/completions`` with tools, retrying transient upstream errors.

        Retries up to six times with exponential backoff on timeouts, rate limits, and
        gateway/server failures (including transient upstream 404s on a known-good model).
        Returns ``(assistant_message, usage)`` — the same normalized pair as
        :meth:`_ChatApi.call_tools`.
        """
        api = self._chat_backend()
        last = None
        for attempt in range(6):
            try:
                return api.call_tools(messages, tools, tool_choice)
            except Exception as e:  # noqa: BLE001 - retry any transient error
                last = e
                # tool calling always runs on /chat, so a transient upstream 404 here is a
                # gateway drop of a known-good model — retry it rather than lose the task.
                transient = _is_transient_upstream(e)
                if attempt == 5 or not transient:
                    break
                time.sleep(min(2 ** (attempt + 1), 32) + (len(messages) % 5) * 0.1)
        raise last

    def reset_session(self):
        """Start a fresh client-owned conversation."""
        self._history = []
        self._prev_response_id = None

    def send(self, new_messages):
        """Client-owned conversation turn (B6): the agent passes only the new messages; the
        client keeps the running history and picks the transport. On a /responses endpoint
        with server-side storage it sends just the new turn with previous_response_id +
        store=true (token-efficient); otherwise it resends the full history (portable).
        Correctness never depends on server state, and deterministic replay is unaffected
        (it replays the recorded tape)."""
        self._history.extend(new_messages)
        if self._tape is not None and not self._recording:  # deterministic replay
            resp = self._tape[self._pos]
            self._pos += 1
            self._history.append({"role": "assistant", "content": resp})
            return resp
        from pace_agent import shims

        t = time.perf_counter()
        with shims.tagged("model"):
            text, u, rid = self._send_session(new_messages)
        self.calls.append(time.perf_counter() - t)
        self.prompt_tokens.append(u["prompt"])
        self.completion_tokens.append(u["completion"])
        self.cached_tokens.append(u["cached"])
        self.reasoning_tokens.append(u["reasoning"])
        self._prev_response_id = rid
        self._history.append({"role": "assistant", "content": text})
        if self._recording:
            self._tape.append(text)
        return text

    def _send_session(self, new_messages):
        api = self._backend()
        # opportunistic server-side threading; falls back permanently if unsupported
        if api.kind == "responses" and self._store_ok:
            try:
                return api.call_stateful(new_messages, self._prev_response_id)
            except Exception:  # noqa: BLE001 - storage not supported -> portable resend
                self._store_ok = False
        text, u = self._send(self._history)  # portable full-history resend
        return text, u, None

    def _is_anthropic(self):
        # Claude speaks the native Messages API, not OpenAI /chat|/responses. The explicit
        # switch is image_format="anthropic" (config-driven); a "claude*" model id is a
        # convenience fallback so a mis-set image_format still routes correctly.
        return self.image_format == "anthropic" or str(self.model).lower().startswith(
            "claude"
        )

    def _backend(self):
        # `api` pins the surface and wins over the Claude guess because it is deliberate; left
        # unset, /responses is preferred for its richer usage and probed because not every
        # endpoint serves it
        if self.api not in (None, "chat", "responses"):
            raise ValueError(
                f"llm api must be 'chat', 'responses' or unset, got {self.api!r}"
            )
        if self._api is None:
            args = (
                self.endpoint,
                self.model,
                self.temperature,
                self.seed,
                self.max_tokens,
                self.api_key,
                self.headers,
            )
            if self.api == "chat":
                self._api = _ChatApi(*args)
            elif self.api == "responses":
                self._api = _ResponsesApi(*args)
            elif self._is_anthropic():
                self._api = _AnthropicApi(*args)
                self._responses_ok = False
            else:
                resp = _ResponsesApi(*args)
                self._responses_ok = resp.ping()
                self._api = resp if self._responses_ok else _ChatApi(*args)
        return self._api

    def _send(self, messages):
        # retry with exponential backoff — under high concurrency the gateway may 429/5xx/timeout
        api = self._backend()
        last = None
        for attempt in range(6):
            try:
                out = api.call(messages)
                beat(f"model answered ({api.kind})")
                return out
            except Exception as e:  # noqa: BLE001 - retry any transient error
                # a rejected request is still the endpoint responding: progress, not a stall
                beat(f"model call rejected, retry {attempt + 1}/6")
                last = e
                # A transient upstream 404 is only retried on the /chat surface (where the
                # route is known to exist); on /responses a 404 means the route is genuinely
                # unsupported and must fall through fast to the /chat fallback below.
                transient = _is_transient_upstream(e) and (
                    api.kind == "chat" or getattr(e, "status_code", None) != 404
                )
                if attempt == 5 or not transient:
                    break
                # backoff 2,4,8,16,32s with a deterministic per-attempt jitter (no RNG in workers)
                time.sleep(min(2 ** (attempt + 1), 32) + (len(messages) % 5) * 0.1)
        # both surfaces rejected the route: /responses was probed and unsupported, and now
        # /chat too -> a combined diagnosis instead of just the raw chat 404.
        if api.kind == "chat" and not self._responses_ok and _unsupported(last):
            raise RuntimeError(
                f"LLM endpoint {self.endpoint!r} served neither /responses nor "
                f"/chat/completions for model {self.model!r}; check the base_url path "
                f"and model id (gateway said: "
                f"{str(getattr(last, 'message', '') or last)[:160]})"
            ) from last
        raise last
