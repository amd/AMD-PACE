# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for nsjail command construction."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from nsjail_agent.models import Language, NetworkPolicy, ResourceLimits, SandboxConfig
from nsjail_agent.sandbox import build_nsjail_cmd, sandbox_workdir


def _flag_values(cmd: list[str], flag: str) -> list[str]:
    return [cmd[index + 1] for index, value in enumerate(cmd) if value == flag]


class TestBuildNsjailCmd(unittest.TestCase):
    def test_build_nsjail_cmd_uses_supported_cwd_flag(self):
        cmd = build_nsjail_cmd(
            workspace_host_path=Path("/tmp/workspace"),
            language=Language.PYTHON,
            resources=ResourceLimits(),
            config=SandboxConfig(),
        )

        self.assertIn("--cwd", cmd)
        self.assertNotIn("--chdir", cmd)
        self.assertEqual(cmd[cmd.index("--cwd") + 1], sandbox_workdir())

    def test_build_nsjail_cmd_uses_absolute_interpreter_path(self):
        cmd = build_nsjail_cmd(
            workspace_host_path=Path("/tmp/workspace"),
            language=Language.PYTHON,
            resources=ResourceLimits(),
            config=SandboxConfig(),
        )

        separator_index = cmd.index("--")
        self.assertEqual(cmd[separator_index + 1], "/usr/bin/python3")
        self.assertNotIn("python3", cmd[separator_index + 1 :])

    def test_build_nsjail_cmd_allows_python_bin_override(self):
        with patch.dict("os.environ", {"NSJAIL_PYTHON_BIN": "/opt/conda/bin/python"}):
            cmd = build_nsjail_cmd(
                workspace_host_path=Path("/tmp/workspace"),
                language=Language.PYTHON,
                resources=ResourceLimits(),
                config=SandboxConfig(),
            )

        separator_index = cmd.index("--")
        self.assertEqual(cmd[separator_index + 1], "/opt/conda/bin/python")

    def test_build_nsjail_cmd_includes_extra_ro_mounts(self):
        with patch.dict(
            "os.environ", {"NSJAIL_EXTRA_RO_MOUNTS": "/opt/conda,/host/data:/data"}
        ):
            cmd = build_nsjail_cmd(
                workspace_host_path=Path("/tmp/workspace"),
                language=Language.PYTHON,
                resources=ResourceLimits(),
                config=SandboxConfig(),
            )

        self.assertEqual(
            _flag_values(cmd, "--bindmount_ro")[-2:],
            ["/opt/conda", "/host/data:/data"],
        )

    def test_network_none_disables_loopback_without_unsupported_flag(self):
        cmd = build_nsjail_cmd(
            workspace_host_path=Path("/tmp/workspace"),
            language=Language.PYTHON,
            resources=ResourceLimits(),
            config=SandboxConfig(network=NetworkPolicy.NONE),
        )

        self.assertIn("--iface_no_lo", cmd)
        self.assertNotIn("--iface_lo", cmd)

    def test_network_loopback_keeps_default_loopback(self):
        cmd = build_nsjail_cmd(
            workspace_host_path=Path("/tmp/workspace"),
            language=Language.PYTHON,
            resources=ResourceLimits(),
            config=SandboxConfig(network=NetworkPolicy.LOOPBACK),
        )

        self.assertNotIn("--iface_no_lo", cmd)
        self.assertNotIn("--iface_lo", cmd)

    def test_build_nsjail_cmd_sets_sandbox_pythonpath(self):
        with patch.dict("os.environ", {"NSJAIL_SANDBOX_PYTHONPATH": "/sandbox_deps"}):
            cmd = build_nsjail_cmd(
                workspace_host_path=Path("/tmp/workspace"),
                language=Language.PYTHON,
                resources=ResourceLimits(),
                config=SandboxConfig(),
            )

        self.assertIn("PYTHONPATH=/sandbox_deps", _flag_values(cmd, "--env"))

    def test_build_nsjail_cmd_omits_pythonpath_when_unset(self):
        env = dict(os.environ)
        env.pop("NSJAIL_SANDBOX_PYTHONPATH", None)
        with patch.dict("os.environ", env, clear=True):
            cmd = build_nsjail_cmd(
                workspace_host_path=Path("/tmp/workspace"),
                language=Language.PYTHON,
                resources=ResourceLimits(),
                config=SandboxConfig(),
            )

        self.assertFalse(
            any(v.startswith("PYTHONPATH=") for v in _flag_values(cmd, "--env"))
        )


if __name__ == "__main__":
    unittest.main()
