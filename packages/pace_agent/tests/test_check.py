# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import unittest
from unittest.mock import patch

import pace_agent.check as check_module
from pace_agent.check import run_checks


class TestCheck(unittest.TestCase):
    def test_run_checks_shape(self):
        checks = run_checks()
        names = {c["name"] for c in checks}
        for expected in ["pace_agent", "numpy", "tools registered"]:
            self.assertIn(expected, names)
        for c in checks:
            self.assertGreaterEqual(set(c), {"name", "ok", "detail", "doc"})
            self.assertIsInstance(c["ok"], bool)

    def test_pace_agent_and_tools_ok(self):
        checks = {c["name"]: c for c in run_checks()}
        self.assertIs(checks["pace_agent"]["ok"], True)
        self.assertIs(checks["tools registered"]["ok"], True)

    def test_chrome_check_uses_browser_resolver(self):
        resolver = getattr(check_module, "_chrome_binary", None)
        self.assertIsNotNone(resolver)
        with patch.object(
            check_module, "_chrome_binary", return_value="/custom/chrome"
        ):
            checks = {c["name"]: c for c in run_checks()}
        self.assertTrue(checks["chrome"]["ok"])
        self.assertEqual(checks["chrome"]["detail"], "/custom/chrome")
