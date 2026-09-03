# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************


class StubLLM:
    """In-memory LLM double for tests — deterministic, offline, record/replay-capable.

    This is a TEST-ONLY path: the pace-agent product itself is real-LLM only. It lets the
    example agents and the profile loop be exercised without a live endpoint.
    """

    def __init__(self, replies=None, tool_script=None):
        self._replies = list(replies) if replies else ["ANSWER: ok"]
        self._i = 0
        # optional scripted tool-calling turns for chat(): each item is a normalized
        # assistant message {"content": str|None, "tool_calls": [{"id","name","arguments"}]}
        self._tool_script = list(tool_script) if tool_script else None
        self._ti = 0
        self._tape = None
        self._recording = False
        self._pos = 0
        self.calls = []

    def record_into(self, tape):
        self._tape, self._recording, self._pos = tape, True, 0

    def replay_from(self, tape):
        self._tape, self._recording, self._pos = list(tape), False, 0

    def complete(self, messages):
        if self._tape is not None and not self._recording:
            r = self._tape[self._pos]
            self._pos += 1
            return r
        self.calls.append(0.0)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        if self._recording:
            self._tape.append(reply)
        return reply

    def chat(self, messages, tools=None, tool_choice="auto"):
        if self._tape is not None and not self._recording:
            m = self._tape[self._pos]
            self._pos += 1
            return m
        self.calls.append(0.0)
        if self._tool_script is not None:
            m = self._tool_script[min(self._ti, len(self._tool_script) - 1)]
            self._ti += 1
        else:
            m = {
                "content": self._replies[min(self._i, len(self._replies) - 1)],
                "tool_calls": [],
            }
            self._i += 1
        if self._recording:
            self._tape.append(m)
        return m
