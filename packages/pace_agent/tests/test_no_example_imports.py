# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import ast
import os
import unittest

_PACE_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "pace_agent")
)
_EXAMPLES = os.path.join(_PACE_AGENT, "examples")


class TestNoExampleImports(unittest.TestCase):
    """Architectural invariant: examples depend on PACE Agent; PACE Agent core NEVER imports from
    pace_agent.examples. (A namespace string like "pace_agent.examples.webvoyager" resolved at runtime
    is fine — this checks real import statements via the AST, not string literals.)"""

    def test_core_never_imports_examples(self):
        offenders = []
        for root, _dirs, files in os.walk(_PACE_AGENT):
            if os.path.commonpath([os.path.normpath(root), _EXAMPLES]) == _EXAMPLES:
                continue  # skip pace_agent/examples/ itself
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                with open(path) as f:
                    tree = ast.parse(f.read(), path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if (node.module or "").startswith("pace_agent.examples"):
                            offenders.append(f"{path}:{node.lineno} from {node.module}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("pace_agent.examples"):
                                offenders.append(
                                    f"{path}:{node.lineno} import {alias.name}"
                                )
        self.assertEqual(
            offenders,
            [],
            f"pace-agent core must not import pace_agent.examples: {offenders}",
        )
