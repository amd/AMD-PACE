# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""CAMEL example: code-execution tool inside a CAMEL ChatAgent.

Requires the ``camel-ai`` package in addition to ``pace-agent``.

Usage::

    python -m pace_agent.examples.nsjail_camel_example
"""

from __future__ import annotations

from camel.agents import ChatAgent  # type: ignore
from camel.messages import BaseMessage  # type: ignore
from camel.models import ModelFactory  # type: ignore
from camel.types import ModelPlatformType, ModelType  # type: ignore

from nsjail_agent.tool_adapter import CAMELToolAdapter

# Wrap the nsjail executor as a CAMEL-compatible tool
code_tool = CAMELToolAdapter()

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
)

agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Code Assistant",
        content="You are a helpful assistant that can write and run code safely.",
    ),
    model=model,
    tools=[code_tool],
)


if __name__ == "__main__":
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content="Sort the list [5, 3, 8, 1, 9, 2] using Python and show the result.",
    )
    response = agent.step(user_msg)
    print(response.msg.content)
