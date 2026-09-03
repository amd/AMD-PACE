# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import os
import shutil
import tempfile
import unittest

from pace_agent.config import RunConfig, load_config
from pace_agent.replay.player import call_agent


class StubLLM:
    def __init__(self, replies=None):
        self.replies = replies or ["ANSWER: ok"]
        self.i = 0
        self.calls = []

    def complete(self, msgs):
        self.calls.append(0.0)
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r


class TestQueryRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_oldconfig_compat(self):
        p = os.path.join(self.tmp, "wv.json")
        with open(p, "w") as f:
            json.dump(
                {
                    "impl": "pace",
                    "graphs": [
                        {
                            "spec": "webvoyager",
                            "kwargs": {
                                "query": "cheapest flight",
                                "start_url": "https://x.com",
                                "max_iter": 25,
                            },
                        }
                    ],
                    "llm_backend": {
                        "type": "openai",
                        "base_url": "http://localhost:8000/v1",
                        "model": "m",
                        "temperature": 0.0,
                        "seed": 0,
                    },
                    "num_runs": 1,
                    "warmup_runs": 0,
                },
                f,
            )
        cfg = load_config(p)
        self.assertIsInstance(cfg, RunConfig)
        self.assertTrue(cfg.tasks)
        self.assertEqual(cfg.tasks[0]["query"], "cheapest flight")
        self.assertEqual(cfg.tasks[0]["start_url"], "https://x.com")
        self.assertEqual(cfg.tasks[0]["max_iter"], 25)
        self.assertEqual(cfg.llm["endpoint"], "http://localhost:8000")
        self.assertEqual(cfg.warmup, 0)

    def test_llm_base_url_only_strips_trailing_v1(self):
        p = os.path.join(self.tmp, "endpoint.json")
        with open(p, "w") as f:
            json.dump(
                {
                    "llm_backend": {
                        "base_url": "https://example.com/v1/proxy",
                        "model": "m",
                    }
                },
                f,
            )
        cfg = load_config(p)
        self.assertEqual(cfg.llm["endpoint"], "https://example.com/v1/proxy")

    def test_llm_backend_api_choice_is_mapped(self):
        p = os.path.join(self.tmp, "api.json")
        with open(p, "w") as f:
            json.dump(
                {
                    "llm_backend": {
                        "base_url": "http://x/v1",
                        "model": "m",
                        "api": "chat",
                    }
                },
                f,
            )
        self.assertEqual(load_config(p).llm["api"], "chat")

    def test_llm_backend_api_defaults_to_inferred(self):
        p = os.path.join(self.tmp, "api_default.json")
        with open(p, "w") as f:
            json.dump({"llm_backend": {"base_url": "http://x/v1", "model": "m"}}, f)
        self.assertIsNone(load_config(p).llm["api"])

    def test_rag_agent_accepts_query(self):
        import pace_agent.examples.basic_example as m

        ans = call_agent(
            m, StubLLM(["ANSWER: bucketed"]), {"query": "what buckets time?"}
        )
        self.assertEqual(ans, "ANSWER: bucketed")

    def test_call_agent_arity(self):
        class Old:
            def run(self, llm):
                return "old"

        class New:
            def run(self, llm, task=None):
                return (task or {}).get("query", "new")

        self.assertEqual(call_agent(Old(), StubLLM(), {"query": "q"}), "old")
        self.assertEqual(call_agent(New(), StubLLM(), {"query": "q"}), "q")
