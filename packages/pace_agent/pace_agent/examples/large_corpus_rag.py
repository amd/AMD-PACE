# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import TypedDict

import numpy as np

from pace_agent.examples.common import op, reason
from pace_agent.registry import get_tool

NAME = "large_corpus_rag"
KIND = "showcase"
QUESTION = "Summarize what the documents say about cpu cache performance."

_DIM = 128
_N = 15000
_VOCAB = [
    "cpu",
    "cache",
    "memory",
    "latency",
    "bandwidth",
    "kernel",
    "vector",
    "search",
    "index",
    "token",
    "model",
    "agent",
] + [f"t{i}" for i in range(300)]
_DOCS = None
_INDEX = None


class _S(TypedDict):
    question: str
    context: str
    answer: str


def _data():
    global _DOCS, _INDEX
    if _INDEX is None:
        rng = np.random.default_rng(0)
        _DOCS = [" ".join(rng.choice(_VOCAB, size=12)) for _ in range(_N)]
        _INDEX = get_tool("embed").dispatch("naive", _DOCS, _DIM)
    return _DOCS, _INDEX


def _retrieve(state):
    docs, index = _data()
    qvec = op("embed", [state["question"]], _DIM)[0]
    idx, _ = op("vector_search", index, qvec, 5)
    selected = op("dedup", [docs[i] for i in idx])
    return {
        "context": op("assemble", [{"role": "doc", "content": doc} for doc in selected])
    }


def _build(llm):
    from langgraph.graph import END, StateGraph

    g = StateGraph(_S)
    g.add_node("retrieve", _retrieve)
    g.add_node("reason", lambda s: {"answer": reason(llm, s["context"], s["question"])})
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
