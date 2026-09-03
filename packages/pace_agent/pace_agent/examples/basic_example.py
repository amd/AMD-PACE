# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import TypedDict

from pace_agent.registry import get_tool

NAME = "basic_example"
KIND = "showcase"
QUESTION = "Which component attributes CPU time to buckets in PACE Agent?"

_DOCS = [
    "pace-agent profile attributes wall clock time to model and the five cpu buckets",
    "the attribution module rolls up op time by bucket and reports unattributed",
    "optimize re-times candidate-covered ops and projects parity-gated end-to-end gain",
    "pace-agent bench tools compares naive and available optimized or provider tiers",
    "opt-in telemetry collects whole-process replay counters through perf or uProf",
    "the recorder captures per op inputs outputs and timing for deterministic replay",
] + [f"filler document number {i} about unrelated topics" for i in range(32)]

_CORPUS = [d.split() for d in _DOCS]


def _tool(name, *args, **kw):
    return get_tool(name).run("naive", *args, **kw)


class _S(TypedDict):
    question: str
    context: str
    answer: str


def _retrieve(state):
    q = _tool("tokenize", state["question"])
    idx, _ = _tool("bm25", _CORPUS, q, 4)
    picked = [" ".join(_CORPUS[i]) for i in idx]
    picked = _tool("dedup", picked)
    msgs = [{"role": "doc", "content": c} for c in picked]
    return {"context": _tool("assemble", msgs)}


def _reason(state, llm):
    prompt = [
        {
            "role": "system",
            "content": "Answer the question using only the context. Be concise.",
        },
        {
            "role": "user",
            "content": f"Context:\n{state['context']}\n\nQuestion: {state['question']}",
        },
    ]
    return {"answer": llm.complete(prompt)}


def _build(llm):
    from langgraph.graph import END, StateGraph

    g = StateGraph(_S)
    g.add_node("retrieve", _retrieve)
    g.add_node("reason", lambda s: _reason(s, llm))
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "reason")
    g.add_edge("reason", END)
    return g.compile()


def run(llm, task=None):
    from pace_agent.node_counter import PaceAgentNodeCounter

    q = (task or {}).get("query") or QUESTION
    return _build(llm).invoke(
        {"question": q}, config={"callbacks": [PaceAgentNodeCounter()]}
    )["answer"]
