# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""LangGraph ReAct example that executes code through the vendored nsjail tool.

Requires an installed ``pace-agent`` plus ``langchain-openai`` / ``langchain-core``,
and endpoint credentials via the environment (``LLM_GATEWAY_KEY`` /
``LLM_GATEWAY_BASE_URL`` / ``LLM_GATEWAY_MODEL``, or the OpenAI equivalents).

Usage (from the repo root, after ``pip install -e ./packages/pace_agent``)::

    python -m pace_agent.examples.nsjail_langgraph_example
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from nsjail_agent.tool_adapter import make_langchain_tool


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def build_app(*, model: str, base_url: str, api_key: str):
    """Compile a LangGraph agent with nsjail-backed ``execute_code``."""
    execute_code = make_langchain_tool()
    tools = [execute_code]
    llm_kwargs = {
        "model": model,
        "temperature": 0,
        "api_key": api_key,
        "base_url": base_url,
    }
    # AMD gateway also expects the subscription key header when LLM_GATEWAY_KEY is set.
    if os.environ.get("LLM_GATEWAY_KEY"):
        llm_kwargs["default_headers"] = {
            "Ocp-Apim-Subscription-Key": os.environ["LLM_GATEWAY_KEY"],
        }
    llm = ChatOpenAI(**llm_kwargs).bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


if __name__ == "__main__":
    api_key = _env("LLM_GATEWAY_KEY", "OPENAI_API_KEY")
    base_url = _env("LLM_GATEWAY_BASE_URL", "OPENAI_BASE_URL")
    model = _env("LLM_GATEWAY_MODEL", "OPENAI_MODEL")
    if not api_key:
        print("ERROR: Set LLM_GATEWAY_KEY or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)
    if not base_url:
        print("ERROR: Set LLM_GATEWAY_BASE_URL or OPENAI_BASE_URL.", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("ERROR: Set LLM_GATEWAY_MODEL or OPENAI_MODEL.", file=sys.stderr)
        sys.exit(1)

    app = build_app(model=model, base_url=base_url, api_key=api_key)
    print(f"Using model: {model}")
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Write a Python script that computes the first 20 Fibonacci "
                        "numbers and prints them."
                    )
                )
            ]
        }
    )
    for msg in result["messages"]:
        print(f"[{type(msg).__name__}] {msg.content}")
