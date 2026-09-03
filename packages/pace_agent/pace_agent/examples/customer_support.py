# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import TypedDict

from pace_agent.examples.common import op, reason

NAME = "customer_support"
KIND = "standard"
QUESTION = "How do I reset my password if I lost access to my email?"

_FAQ = [
    "to reset your password open settings then security then reset password",
    "if you lost email access contact support with your account id for identity verification",
    "billing questions are handled under the billing tab in your account",
    "two factor authentication can be enabled in the security settings",
    "to close your account contact support and confirm the closure by email",
] + [f"unrelated help article number {i}" for i in range(45)]

_CORPUS = [d.split() for d in _FAQ]


class _S(TypedDict):
    question: str
    context: str
    answer: str


def _retrieve(state):
    q = op("tokenize", state["question"])
    idx, _ = op("bm25", _CORPUS, q, 3)
    docs = op("dedup", [" ".join(_CORPUS[i]) for i in idx])
    return {"context": op("assemble", [{"role": "faq", "content": d} for d in docs])}


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
