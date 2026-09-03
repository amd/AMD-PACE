# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Load GAIA tasks from local files or the HuggingFace GAIA dataset.

Local/HF row parsing and :class:`~gaia_planner.models.GaiaTask` normalization live
here; shared JSON/JSONL and HuggingFace parquet I/O live in :mod:`pace_agent.datasets`.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pace_agent.datasets import load_hf_parquet_repo, read_json_or_jsonl

from .models import GaiaTask

# packages/pace_agent — holds both configs/ and pace_agent/examples/fixtures/.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


_LINK_DIR: Path | None = None


def _cleanup_link_dir(path: Path, owner_pid: int) -> None:
    """Remove the symlink farm, but only in the process that created it.

    A forked worker inherits both ``_LINK_DIR`` and this atexit hook, and would
    otherwise delete the directory its parent is still handing to tasks.
    """
    if os.getpid() == owner_pid:
        shutil.rmtree(path, ignore_errors=True)


def _link_dir() -> Path:
    """Return this process's private directory for friendly-name symlinks.

    A fixed shared path (``/tmp/gaia_named_files``) is unusable and unsafe on a
    multi-user host: whoever creates it first owns it, so every other user's
    ``symlink()`` fails, and because the link names are derived from the task id
    they are predictable enough for another user to pre-plant or swap a link and
    redirect an attachment read at an arbitrary file. ``mkdtemp`` gives us a
    0700 directory with an unguessable name, private to this process.
    """
    global _LINK_DIR
    if _LINK_DIR is None:
        created = Path(tempfile.mkdtemp(prefix="gaia_named_files-"))
        atexit.register(_cleanup_link_dir, created, os.getpid())
        _LINK_DIR = created
    return _LINK_DIR


def _named_attachment(
    resolved: Path,
    file_name: str | None,
    *,
    key: str | None = None,
) -> Path:
    """Return an absolute path that keeps the friendly (extensioned) filename.

    The HF snapshot exposes attachments as ``<id>.docx`` symlinks into hash-named
    blobs; ``Path.resolve()`` would dereference to the extensionless blob. We keep
    the real name so downstream tools (and the model) can tell a .docx/.pdf/.xlsx
    from an image. If the source name lacks the extension the metadata provides, we
    expose a correctly-named symlink to the underlying blob.

    Symlinks are keyed by ``key`` (typically ``task_id``) plus the basename so two
    tasks that share a friendly ``file_name`` but point at different blobs cannot
    clobber each other within a run.
    """
    path = resolved.absolute()
    desired = Path((file_name or "").strip()).name
    if not desired or path.name == desired:
        return path
    target = path.resolve()
    digest = hashlib.sha256(f"{key or target}\0{desired}".encode()).hexdigest()[:16]
    link_name = f"{digest}__{desired}"
    try:
        link_dir = _link_dir()
        link = link_dir / link_name
        if link.is_symlink() or link.exists():
            if link.resolve() == target:
                return link
            link.unlink()
        link.symlink_to(target)
        return link
    except OSError as exc:
        # Losing the extension makes a .docx indistinguishable from a .pdf for
        # both the toolkit and the model, so say so instead of failing mute.
        warnings.warn(
            f"could not expose attachment {target} as {desired!r} "
            f"({type(exc).__name__}: {exc}); using the original filename",
            RuntimeWarning,
            stacklevel=2,
        )
        return path


GAIA_HF_REPO = "gaia-benchmark/GAIA"
GAIA_HF_CONFIG = "2023_all"


def load_tasks(path: str | Path) -> list[GaiaTask]:
    source = Path(path).expanduser().resolve()
    records = read_json_or_jsonl(source)
    return [
        _normalize_record(record, source, index) for index, record in enumerate(records)
    ]


def load_gaia_from_hf(
    *,
    level: int | None = None,
    split: str = "validation",
    config: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    _loader: Any | None = None,
) -> list[GaiaTask]:
    """Load the official GAIA dataset from HuggingFace, optionally by level.

    The GAIA repo is gated: accept its terms on the Hub and provide a token
    (``HF_TOKEN`` env var or ``--hf-token``). ``level`` filters to a single
    difficulty (1, 2, or 3); ``None`` keeps every level.

    Since October 2025 GAIA ships as Parquet (no dataset script), so we snapshot
    the repo and load the Parquet-backed config; ``file_path`` is relative to the
    downloaded repository root.
    """
    resolved_config = config or (
        f"2023_level{level}" if level in {1, 2, 3} else GAIA_HF_CONFIG
    )
    if _loader is not None:
        rows = _loader(
            config=resolved_config, split=split, token=token, cache_dir=cache_dir
        )
        return _tasks_from_hf_rows(rows, level=level, split=split, data_dir="")

    data_dir, rows = load_hf_parquet_repo(
        GAIA_HF_REPO,
        resolved_config,
        split,
        token=token,
        cache_dir=cache_dir,
    )
    return _tasks_from_hf_rows(rows, level=level, split=split, data_dir=data_dir)


def _tasks_from_hf_rows(
    rows: Iterable[dict[str, Any]],
    *,
    level: int | None,
    split: str,
    data_dir: str,
) -> list[GaiaTask]:
    tasks: list[GaiaTask] = []
    for index, record in enumerate(rows):
        if level is not None and str(record.get("Level", "")).strip() != str(level):
            continue
        tasks.append(_normalize_hf_record(record, index, split, data_dir))
    return tasks


def _normalize_hf_record(
    record: dict[str, Any],
    index: int,
    split: str,
    data_dir: str,
) -> GaiaTask:
    question = record.get("Question") or record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"GAIA row {index} has no Question string")

    task_id = record.get("task_id") or f"gaia-{split}-{index}"

    final_answer = record.get("Final answer", record.get("Final Answer"))
    # The test split ships placeholder answers ("?"); treat those as unscored.
    if final_answer is None or str(final_answer).strip() in {"", "?"}:
        expected = None
    else:
        expected = str(final_answer)

    files: tuple[Path, ...] = ()
    file_path = record.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        resolved = Path(file_path).expanduser()
        if data_dir and not resolved.is_absolute():
            resolved = Path(data_dir) / file_path
        if resolved.is_file():
            # Do NOT resolve() the symlink: in the HF snapshot the friendly name
            # (e.g. "<id>.docx") is a symlink into a hash-named blob, and resolving
            # would strip the extension - leaving the model unable to tell a Word
            # doc from an image. Keep the extensioned name so tools read it right.
            files = (
                _named_attachment(resolved, record.get("file_name"), key=str(task_id)),
            )

    level_value = record.get("Level")
    metadata: dict[str, Any] = {
        "source": "huggingface",
        "split": split,
        "level": str(level_value) if level_value is not None else None,
        "file_name": record.get("file_name"),
    }
    return GaiaTask(
        task_id=str(task_id),
        question=question.strip(),
        expected=expected,
        start_url="https://duckduckgo.com/",
        files=files,
        source=None,
        metadata=metadata,
    )


def normalize_answer(value: str) -> str:
    """Same normalization as the PACE Agent harness scorer (not the official GAIA metric)."""
    from pace_agent.score import normalize

    return normalize(value)


def answers_match(actual: str, expected: str | None) -> bool | None:
    """Strict PACE Agent match; ``None`` when the expected answer is missing/blank."""
    return answer_verdicts(actual, expected)["match_strict"]


def answer_verdicts(actual: str, expected: str | None) -> dict[str, Any]:
    """Both PACE Agent harness verdicts (``match_strict`` / ``match_loose``) for one answer."""
    from pace_agent.score import score as pace_agent_score

    return pace_agent_score(actual, expected)


def _attachment_roots(task_dir: Path) -> tuple[Path, ...]:
    """Directories a task file may reference attachments from.

    The task file's own directory plus the PACE Agent package root, because the
    shipped sample sets live in ``configs/`` and point at fixtures under
    ``pace_agent/examples/fixtures/``. Extra roots can be added with
    ``GAIA_TASK_FILES_ROOT`` (os.pathsep-separated) for task sets that keep
    their attachments elsewhere.
    """
    roots = [task_dir, PACKAGE_ROOT]
    extra = os.environ.get("GAIA_TASK_FILES_ROOT", "")
    roots.extend(
        Path(part).expanduser().resolve()
        for part in extra.split(os.pathsep)
        if part.strip()
    )
    return tuple(roots)


def _resolve_attachment(base: Path, name: str, task_id: Any) -> Path:
    """Resolve one task attachment, rejecting anything outside the allowed roots.

    ``resolve()`` silently follows ``..``, so without this a task file (which may
    come from an untrusted dataset drop) could name ``../../../etc/passwd`` or a
    private key and have those bytes copied into the sandbox and shown to the
    model.
    """
    if Path(name).is_absolute():
        raise ValueError(
            f"Task {task_id} attachment {name!r} must be relative to the task file"
        )
    resolved = (base / name).resolve()
    for root in _attachment_roots(base):
        if resolved == root or root in resolved.parents:
            return resolved
    raise ValueError(
        f"Task {task_id} attachment {name!r} resolves to {resolved}, outside the "
        f"task directory and the PACE Agent package root. Set GAIA_TASK_FILES_ROOT to "
        f"allow an additional directory."
    )


def _normalize_record(record: dict[str, Any], source: Path, index: int) -> GaiaTask:
    question = record.get("question") or record.get("ques")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Task {index} in {source} has no question/ques string")

    task_id = record.get("task_id") or record.get("id") or f"task-{index}"
    expected = record.get("expected")
    if expected is None:
        expected = record.get("Final answer")
    if expected is not None:
        expected = str(expected)

    start_url = record.get("web") or "https://www.google.com/"
    raw_files = record.get("files") or []
    if not isinstance(raw_files, list) or not all(
        isinstance(name, str) for name in raw_files
    ):
        raise ValueError(f"Task {task_id} in {source} has an invalid files list")
    # Attachments are named relative to the task file, so they must stay inside
    # its directory: `resolve()` happily follows `../../` out of the task set and
    # those bytes are later copied into the sandbox and shown to the model.
    base = source.parent.resolve()
    files = tuple(_resolve_attachment(base, name, task_id) for name in raw_files)

    return GaiaTask(
        task_id=str(task_id),
        question=question.strip(),
        expected=expected,
        start_url=str(start_url),
        files=files,
        source=source,
        metadata=dict(record),
    )
