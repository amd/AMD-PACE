# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import unittest

from pace_agent import get_tool
from pace_agent.providers import get_provider_impl, providers_for, register_provider
from pace_agent.registry import SkipTool


class TestProviders(unittest.TestCase):
    def test_register_and_lookup(self):
        register_provider("dummy", "quantize", lambda x: "dummy-result")
        self.assertIsNotNone(get_provider_impl("dummy", "quantize"))
        self.assertIsNone(get_provider_impl("dummy", "vector_search"))
        self.assertIn("dummy", providers_for("quantize"))

    def test_registered_provider_dispatches(self):
        register_provider("dummy2", "quantize", lambda x: "quantized-by-dummy2")
        self.assertEqual(
            get_tool("quantize").run("dummy2", [1.0, 2.0]), "quantized-by-dummy2"
        )

    def test_unknown_provider_skips_cleanly(self):
        # An unloaded provider/plugin tier must skip (SkipTool), never crash the caller.
        with self.assertRaises(SkipTool):
            get_tool("quantize").run("no_such_provider")
