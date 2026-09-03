# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for sandbox path configuration."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from nsjail_agent.config import (
    configure_sandbox_paths,
    get_sandbox_paths,
    load_sandbox_paths_config,
    reset_sandbox_paths_cache,
)

_ENV_KEYS = (
    "NSJAIL_SANDBOX_CONFIG",
    "NSJAIL_SANDBOX_WORKDIR",
    "NSJAIL_SANDBOX_DEPS_MOUNT",
    "NSJAIL_SANDBOX_STOCKFISH_PATH",
)


class SandboxPathsConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_sandbox_paths_cache()
        for key in _ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        reset_sandbox_paths_cache()
        for key in _ENV_KEYS:
            os.environ.pop(key, None)

    def test_load_defaults_from_config_file(self) -> None:
        cfg = load_sandbox_paths_config()
        self.assertEqual(cfg.workdir, "/sandbox")
        self.assertEqual(cfg.deps_mount, "/sandbox_deps")
        self.assertEqual(cfg.stockfish_path, "/opt/stockfish")

    def test_json_file_overrides_builtin_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sandbox.json"
            path.write_text(
                json.dumps(
                    {
                        "workdir": "/custom/sandbox",
                        "deps_mount": "/custom/deps",
                        "stockfish_path": "/custom/stockfish",
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_sandbox_paths_config(config_path=path)
            self.assertEqual(cfg.workdir, "/custom/sandbox")
            self.assertEqual(cfg.deps_mount, "/custom/deps")
            self.assertEqual(cfg.stockfish_path, "/custom/stockfish")

    def test_mapping_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sandbox.json"
            path.write_text(json.dumps({"workdir": "/json/workdir"}), encoding="utf-8")
            with unittest.mock.patch.dict(
                "os.environ", {"NSJAIL_SANDBOX_DEPS_MOUNT": "/env/deps"}, clear=False
            ):
                cfg = load_sandbox_paths_config(
                    config_path=path,
                    overrides={"stockfish_path": "/override/stockfish"},
                )
            self.assertEqual(cfg.workdir, "/json/workdir")
            self.assertEqual(cfg.deps_mount, "/env/deps")
            self.assertEqual(cfg.stockfish_path, "/override/stockfish")

    def test_configure_and_get_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sandbox.json"
            path.write_text(
                json.dumps({"workdir": "/cached/workdir"}), encoding="utf-8"
            )
            configured = configure_sandbox_paths(config_path=path)
            self.assertEqual(configured.workdir, "/cached/workdir")
            self.assertEqual(get_sandbox_paths().workdir, "/cached/workdir")


if __name__ == "__main__":
    unittest.main()
