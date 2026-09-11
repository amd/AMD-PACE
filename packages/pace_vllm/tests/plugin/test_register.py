# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the `pace_vllm.register()` plugin entry point: success path,
version-mismatch / invalid-spec failure paths, idempotence."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import pace_vllm


class TestRegister(unittest.TestCase):
    def test_supported_returns_platform_qualname(self) -> None:
        self.assertEqual(pace_vllm.register(), "pace_vllm.platform.PacePlatform")

    def test_version_mismatch_returns_none_and_skips_native_load(self) -> None:
        # An unsupported version must short-circuit *before* _load_pace_native
        # runs; asserting only the return value would let a future reorder slip.
        with (
            patch.object(pace_vllm, "_PACE_VLLM_SUPPORTED_VLLM_RANGE", "==99.0.0"),
            patch.object(pace_vllm, "_load_pace_native") as mock_load,
        ):
            self.assertIsNone(pace_vllm.register())
        mock_load.assert_not_called()

    def test_invalid_range_returns_none_and_skips_native_load(self) -> None:
        with (
            patch.object(pace_vllm, "_PACE_VLLM_SUPPORTED_VLLM_RANGE", "not-a-spec"),
            patch.object(pace_vllm, "_load_pace_native") as mock_load,
        ):
            self.assertIsNone(pace_vllm.register())
        mock_load.assert_not_called()

    def test_prerelease_inside_range_is_supported(self) -> None:
        with patch("vllm.__version__", "0.28.0rc1"):
            supported, _, _, _ = pace_vllm._check_vllm_version_supported()
        self.assertTrue(supported)

    def test_idempotent(self) -> None:
        first = pace_vllm.register()
        second = pace_vllm.register()
        self.assertEqual(first, second)


class TestLoadPaceNativeFakesSnapshotGuard(unittest.TestCase):
    """When pace's torch surfaces are already registered (e.g. `pace` was
    imported before us), `_load_pace_native()` must NOT re-import
    `_fakes_snapshot`; doing so calls `torch.library.register_fake` twice
    on the same names and raises -- which `register()` swallows."""

    def setUp(self) -> None:
        pace_vllm.register()
        self._saved_ops_loaded = pace_vllm._ops_loaded
        self._saved_sys_modules_entry = sys.modules.pop(
            "pace_vllm._fakes_snapshot", None
        )
        self._saved_attr = pace_vllm.__dict__.pop("_fakes_snapshot", None)

    def tearDown(self) -> None:
        pace_vllm._ops_loaded = self._saved_ops_loaded
        if self._saved_sys_modules_entry is not None:
            sys.modules["pace_vllm._fakes_snapshot"] = self._saved_sys_modules_entry
        else:
            sys.modules.pop("pace_vllm._fakes_snapshot", None)
        if self._saved_attr is not None:
            pace_vllm._fakes_snapshot = self._saved_attr

    def test_skips_fakes_snapshot_when_pace_ops_already_registered(self) -> None:
        import torch.library

        pace_vllm._ops_loaded = False
        with (
            patch.object(
                torch.library, "register_fake", return_value=lambda f: f
            ) as mock_register,
            patch.object(pace_vllm, "_pace_ops_already_registered", return_value=True),
        ):
            pace_vllm._load_pace_native()

        self.assertEqual(
            mock_register.call_count,
            0,
            "H1 regression: torch.library.register_fake was called "
            f"{mock_register.call_count} times from _fakes_snapshot "
            "despite pace ops already being registered.",
        )


if __name__ == "__main__":
    unittest.main()
