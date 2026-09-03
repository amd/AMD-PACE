# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from pace_agent.registry import (
    PaceAgentTool,
    SkipTool,
    register_tool,
    op,
    get_tool,
    list_tools,
    load_plugins,
    TOOLS,
)
from pace_agent import tools as _tools  # noqa: F401
from pace_agent.providers import register_provider, providers_for  # noqa: F401
from pace_agent.langgraph import (
    profile_invoke,
    with_pace_agent,
    instrument,
)  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "PaceAgentTool",
    "SkipTool",
    "register_tool",
    "op",
    "register_provider",
    "providers_for",
    "get_tool",
    "list_tools",
    "load_plugins",
    "TOOLS",
    "profile_invoke",
    "with_pace_agent",
    "instrument",
    "__version__",
]
