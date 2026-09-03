# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Shared dataset I/O helpers (local JSON/JSONL and HuggingFace parquet repos)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL task file into a list of objects."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {path}")
        records = value
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every task in {path} must be an object")
    return records


def load_hf_parquet_repo(
    repo_id: str,
    config: str,
    split: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> tuple[str, Iterable[dict[str, Any]]]:
    """Download a HuggingFace dataset repo and load one parquet-backed config split.

    Returns ``(repo_root, rows)`` where ``repo_root`` is the snapshot directory
    (attachment paths in rows are relative to it).
    """
    # The Xet download backend can fail with "Unable to parse string as hex hash
    # value" on some repos; fall back to standard HTTP transfers. Must be set
    # before huggingface_hub is imported, since it reads this at import time.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Loading HuggingFace datasets requires the 'datasets' and "
            "'huggingface_hub' packages. Install them with: "
            "pip install -e ./packages/pace_agent (or pip install datasets huggingface_hub)."
        ) from exc

    data_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        cache_dir=cache_dir,
    )
    rows = load_dataset(data_dir, config, split=split)
    return data_dir, rows
