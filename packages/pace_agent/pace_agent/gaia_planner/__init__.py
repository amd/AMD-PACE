# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PACE Agent namespace re-export of the bundled ``gaia_planner`` package.

Use ``from pace_agent.gaia_planner import ...`` in PACE Agent code. The implementation
remains in the internal top-level ``gaia_planner`` package.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from gaia_planner import *  # noqa: F403,F401
from gaia_planner import __all__ as __all__  # noqa: F401


def __getattr__(name: str) -> Any:
    """Forward submodule lookups (``cli``, ``planner``, ...) to ``gaia_planner``."""
    try:
        mod = importlib.import_module(f"gaia_planner.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    sys.modules[f"{__name__}.{name}"] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(
        set(__all__) | {"cli", "datasets", "models", "planner", "langgraph_planner"}
    )
