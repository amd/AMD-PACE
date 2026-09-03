# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Shared data contracts for the GAIA planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GaiaTask:
    task_id: str
    question: str
    expected: str | None = None
    start_url: str = "https://duckduckgo.com/"
    files: tuple[Path, ...] = ()
    source: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerConfig:
    model: str
    max_turns: int = 16
    system_prompt: str | None = None
    # Closed-book mode: never advertise any tools to the model and never open the
    # execution sandbox. The model is not told which tools exist and must reason to
    # the answer from its own knowledge, ending with a `FINAL ANSWER:` line. Used to
    # measure the model's unaided reasoning.
    no_tools: bool = False
    # In closed-book mode the model cannot open files, so each attached file's extracted
    # text is inlined into the task prompt instead. This bounds how many characters of each
    # file are inlined (0 disables inlining, keeping it strictly filename-only).
    attached_file_char_budget: int = 20000
    # Post-loop forced-answer retries. When the research budget is exhausted the
    # model is asked (with tool_choice="none") to commit an answer. A single call
    # sometimes returns empty content or a "Let me..." preamble instead of a
    # committed value; this many extra escalating retries recover a concrete
    # answer from the evidence already gathered.
    final_answer_retries: int = 2
    # Context-window safety. A single oversized tool output (or many accumulated
    # ones) can exceed the model's context limit; these bounds keep the running
    # message history well under it.
    max_tool_output_chars: int = 8000
    context_char_budget: int = 240000
    keep_recent_tool_outputs: int = 3


@dataclass
class PlannerRunResult:
    task_id: str
    answer: str
    raw_answer: str
    messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    # Per-node wall-clock timings (ms stats keyed by graph node), populated by the LangGraph
    # engine; None for the loop engine, which has no graph nodes.
    per_node: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "task_id": self.task_id,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "messages": self.messages,
            "tool_events": self.tool_events,
        }
        if self.per_node is not None:
            record["per_node"] = self.per_node
        return record
