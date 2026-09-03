# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from pace_agent.registry import get_tool


def query_of(task, default):
    return (task or {}).get("query") or default


def op(name, *args, **kw):
    return get_tool(name).run("naive", *args, **kw)


def reason(llm, context, question):
    return llm.complete(
        [
            {"role": "system", "content": "Answer using only the context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    )
