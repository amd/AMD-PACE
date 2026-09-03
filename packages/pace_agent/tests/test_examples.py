# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import importlib
import os
import unittest

from pace_agent.cli import _EXAMPLES as CLI_EXAMPLES
from pace_agent.config import load_config
from pace_agent.replay import Player, Recorder
from tests.stubs import StubLLM

_EXAMPLES = [
    "basic_example",
    "customer_support",
    "tool_agent",
    "large_corpus_rag",
    "testgen",
]


class TestExamples(unittest.TestCase):
    def test_examples_importable(self):
        for name in _EXAMPLES:
            with self.subTest(example=name):
                m = importlib.import_module(f"pace_agent.examples.{name}")
                self.assertTrue(hasattr(m, "run"))
                self.assertTrue(hasattr(m, "NAME"))
                self.assertTrue(hasattr(m, "KIND"))

    def test_testgen_is_listed_and_imported(self):
        self.assertIn("testgen", _EXAMPLES)
        self.assertIn("testgen", CLI_EXAMPLES)

    def test_testgen_config_uses_supported_llm_shape(self):
        config = load_config(
            os.path.join(
                os.path.dirname(__file__), "..", "configs", "pace_agent_testgen.json"
            )
        )
        self.assertTrue(config.llm["endpoint"])

    def test_large_corpus_is_initialized_lazily(self):
        module = importlib.reload(
            importlib.import_module("pace_agent.examples.large_corpus_rag")
        )
        self.assertIsNone(module._INDEX)

    def test_testgen_warns_about_generated_code_execution(self):
        module = importlib.import_module("pace_agent.examples.testgen")
        self.assertIn("executes model-generated Python", module.__doc__)

    def test_showcase_has_recoverable(self):
        m = importlib.import_module("pace_agent.examples.large_corpus_rag")
        traj = Recorder().record(m, StubLLM(), "pace_agent.examples.large_corpus_rag")
        r = Player().isolate(traj, "vector_search", repeat=1)
        self.assertGreater(r["recoverable"], 0)
