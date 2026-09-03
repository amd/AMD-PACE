# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""E2E fixture agent: a deterministic run(llm, task) that exercises every op which
has an optimized tier, so the record -> replay -> optimize loop can re-time them.

Importable by absolute path (PACE Agent re-imports it during `optimize`). Pure/replay-safe.
"""

import numpy as np

from pace_agent.registry import get_tool

NAME = "e2e_ops"
KIND = "showcase"
REPLAY_SAFE = True


def _op(name, *a, **k):
    return get_tool(name).run("naive", *a, **k)


def run(llm, task=None):
    rng = np.random.default_rng(0)  # fixed => identical inputs across record & replay

    # retrieval
    texts = ["w1 w2 w3 w1 w4", "w4 w2 w2", "w5 w6 w1", "w1 w1 w2 w3"] * 8
    _op("embed", texts, 64)
    index = rng.standard_normal((600, 64), dtype=np.float32)
    query = rng.standard_normal(64, dtype=np.float32)
    _op("vector_search", index, query, 10)
    cands = rng.standard_normal((400, 64), dtype=np.float32)
    _op("rerank", query, cands, 10)

    # context
    _op("dedup", [f"line {i % 50}" for i in range(1500)])

    # serialization
    obj = {
        "items": [
            {"id": i, "name": f"item{i}", "vals": list(range(5))} for i in range(400)
        ]
    }
    s = _op("json_serialize", obj)
    _op("json_deserialize", s)
    body = "".join(
        f"<div><span>cell {i}</span><a href='#'>link {i}</a></div>" for i in range(300)
    )
    _op("html_parse", f"<html><body>{body}</body></html>")
    _op(
        "ast_parse",
        "\n".join(
            f"def f{i}(x):\n    y = x + {i}\n    return y * 2" for i in range(60)
        ),
    )
    _op(
        "regex_extract", " ".join(f"key{i}=val{i}" for i in range(800)), r"key\d+=(\w+)"
    )

    ans = llm.complete([{"role": "user", "content": (task or {}).get("query", "go")}])
    return {"answer": ans, "reached_answer": True}
