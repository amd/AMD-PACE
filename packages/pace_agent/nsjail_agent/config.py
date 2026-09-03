# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Sandbox path defaults and configuration loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_PACE_AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SANDBOX_CONFIG_PATH = _PACE_AGENT_ROOT / "configs" / "nsjail_sandbox.json"

_ENV_CONFIG_PATH = "NSJAIL_SANDBOX_CONFIG"
_ENV_WORKDIR = "NSJAIL_SANDBOX_WORKDIR"
_ENV_DEPS_MOUNT = "NSJAIL_SANDBOX_DEPS_MOUNT"
_ENV_STOCKFISH_PATH = "NSJAIL_SANDBOX_STOCKFISH_PATH"

_BUILTIN_DEFAULTS: dict[str, str] = {
    "workdir": "/sandbox",
    "deps_mount": "/sandbox_deps",
    "stockfish_path": "/opt/stockfish",
}


@dataclass(frozen=True)
class SandboxPathsConfig:
    """In-sandbox mount points and fixed tool paths."""

    workdir: str
    deps_mount: str
    stockfish_path: str


_cached: SandboxPathsConfig | None = None


def _resolve_config_path(config_path: Path | str | None) -> Path:
    if config_path is not None:
        return Path(config_path)
    env_path = os.environ.get(_ENV_CONFIG_PATH, "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_SANDBOX_CONFIG_PATH


def _load_json_defaults(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        return dict(_BUILTIN_DEFAULTS)
    try:
        with config_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        # A malformed config must not surface as a bare JSONDecodeError from
        # whichever call site happened to warm the cache first.
        raise ValueError(f"Could not read sandbox config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Sandbox config must be a JSON object: {config_path}")
    merged = dict(_BUILTIN_DEFAULTS)
    for key in _BUILTIN_DEFAULTS:
        value = raw.get(key)
        if value is not None and value != "":
            merged[key] = str(value)
    return merged


def _apply_env_overrides(values: dict[str, str]) -> dict[str, str]:
    out = dict(values)
    for key, env_name in (
        ("workdir", _ENV_WORKDIR),
        ("deps_mount", _ENV_DEPS_MOUNT),
        ("stockfish_path", _ENV_STOCKFISH_PATH),
    ):
        override = os.environ.get(env_name, "").strip()
        if override:
            out[key] = override
    return out


def _apply_mapping_overrides(
    values: dict[str, str], overrides: Mapping[str, Any] | None
) -> dict[str, str]:
    if not overrides:
        return values
    out = dict(values)
    for key in _BUILTIN_DEFAULTS:
        if key in overrides and overrides[key] not in (None, ""):
            out[key] = str(overrides[key])
    return out


def load_sandbox_paths_config(
    *,
    config_path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SandboxPathsConfig:
    """Load sandbox path settings from the JSON config file, then apply overrides."""
    path = _resolve_config_path(config_path)
    values = _load_json_defaults(path)
    values = _apply_mapping_overrides(values, overrides)
    values = _apply_env_overrides(values)
    return SandboxPathsConfig(
        workdir=values["workdir"],
        deps_mount=values["deps_mount"],
        stockfish_path=values["stockfish_path"],
    )


def _publish_to_env(config: SandboxPathsConfig) -> None:
    """Expose resolved paths in the environment for child processes."""
    os.environ[_ENV_WORKDIR] = config.workdir
    os.environ[_ENV_DEPS_MOUNT] = config.deps_mount
    os.environ[_ENV_STOCKFISH_PATH] = config.stockfish_path


def configure_sandbox_paths(
    *,
    config_path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SandboxPathsConfig:
    """Load and cache sandbox path settings for the current process."""
    global _cached
    _cached = load_sandbox_paths_config(config_path=config_path, overrides=overrides)
    _publish_to_env(_cached)
    return _cached


def get_sandbox_paths() -> SandboxPathsConfig:
    """Return cached sandbox path settings, loading defaults on first use."""
    global _cached
    if _cached is None:
        _cached = load_sandbox_paths_config()
        _publish_to_env(_cached)
    return _cached


def reset_sandbox_paths_cache() -> None:
    """Clear cached settings (for tests)."""
    global _cached
    _cached = None
