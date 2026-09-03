# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""The command surface after `bench workload` was folded into `profile`."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch


def _cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "pace_agent.cli", *argv], capture_output=True, text=True
    )


class TestCliSurface(unittest.TestCase):
    def test_examples_are_listed_by_profile(self):
        proc = _cli("profile", "--list")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("webvoyager", proc.stdout)
        self.assertIn("pace-agent profile", proc.stdout)

    def test_profile_needs_an_agent_and_a_config(self):
        self.assertEqual(_cli("profile").returncode, 2)
        self.assertIn("agent is required", _cli("profile").stdout)
        self.assertIn("requires --config", _cli("profile", "webvoyager").stdout)


class TestRunDirRecordsWhatRan(unittest.TestCase):
    """The copied config says what ran, except when --query replaced its tasks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def _profile(self, cfg_payload, query=None):
        from pace_agent.cli import cmd_profile

        cfg_path = os.path.join(self.tmp, "run.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_payload, f)
        out = os.path.join(self.tmp, "out")
        args = Namespace(
            agent="pace_agent.examples.basic_example",
            config=cfg_path,
            out=out,
            query=query,
            list=False,
        )
        with (
            patch("pace_agent.cli._profile_multi", return_value=0),
            patch("pace_agent.profile.profiler.profile"),
        ):
            cmd_profile(args)
        return out

    def test_query_overrides_are_recorded(self):
        out = self._profile({"tasks": [{"query": "from config"}]}, query=["a", "b"])
        with open(os.path.join(out, "tasks.json")) as f:
            self.assertEqual([t["query"] for t in json.load(f)["tasks"]], ["a", "b"])

    def test_an_inline_config_needs_no_extra_file(self):
        out = self._profile({"tasks": [{"query": "q1"}, {"query": "q2"}]})
        self.assertFalse(os.path.exists(os.path.join(out, "tasks.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "config.json")))


if __name__ == "__main__":
    unittest.main()
