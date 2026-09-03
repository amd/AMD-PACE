# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
from typing import TypedDict

from pace_agent.examples.common import op, reason

NAME = "tool_agent"
KIND = "standard"
QUESTION = "What is the status and owner of ticket T-42?"

_TOOL_RESULT = json.dumps(
    {
        "tickets": [
            {
                "id": f"T-{i}",
                "status": "open" if i % 2 else "closed",
                "owner": f"user{i}",
                "note": f"ticket note body {i}",
            }
            for i in range(60)
        ]
    }
)


class _S(TypedDict):
    question: str
    context: str
    answer: str


def _parse(state):
    data = op("json_deserialize", _TOOL_RESULT)
    ids = op("regex_extract", _TOOL_RESULT, r"\"id\": \"(T-\d+)\"")
    rows = [
        f"{t['id']} status={t['status']} owner={t['owner']}" for t in data["tickets"]
    ]
    return {
        "context": op("assemble", [{"role": "row", "content": r} for r in rows[:20]])
        or " ".join(ids[:1])
    }


def _build(llm):
    from langgraph.graph import END, StateGraph

    g = StateGraph(_S)
    g.add_node("parse", _parse)
    g.add_node("reason", lambda s: {"answer": reason(llm, s["context"], s["question"])})
    g.set_entry_point("parse")
    g.add_edge("parse", "reason")
    g.add_edge("reason", END)
    return g.compile()


def run(llm, task=None):
    from pace_agent.node_counter import PaceAgentNodeCounter

    q = (task or {}).get("query") or QUESTION
    return _build(llm).invoke(
        {"question": q}, config={"callbacks": [PaceAgentNodeCounter()]}
    )["answer"]
