# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Manages ephemeral on-disk workspaces mounted into the nsjail sandbox."""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from .models import WorkspaceFile

if TYPE_CHECKING:
    pass


class Workspace:
    """Creates a temporary directory that serves as the sandbox's working directory.

    Use as a context manager to guarantee cleanup:

        async with Workspace(files=[WorkspaceFile(name="main.py", content="print('hi')")]) as ws:
            print(ws.host_path)   # /tmp/nsjail_ws_XXXXXX
    """

    def __init__(self, files: list[WorkspaceFile] | None = None) -> None:
        self._files = files or []
        self._tmpdir: str | None = None

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "Workspace":
        self._tmpdir = tempfile.mkdtemp(prefix="nsjail_ws_")
        self._write_files()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def host_path(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError(
                "Workspace not yet initialised — use as a context manager."
            )
        return Path(self._tmpdir)

    def write_file(self, name: str, content: str) -> Path:
        """Write (or overwrite) a file inside the workspace."""
        dest = self._safe_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_within_workspace(dest.parent.resolve(strict=True))
        if dest.exists():
            self._ensure_within_workspace(dest.resolve(strict=True))
        dest.write_text(content, encoding="utf-8")
        return dest

    def write_binary_file(self, name: str, content: bytes) -> Path:
        """Write binary content inside the workspace."""
        dest = self._safe_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_within_workspace(dest.parent.resolve(strict=True))
        if dest.exists():
            self._ensure_within_workspace(dest.resolve(strict=True))
        dest.write_bytes(content)
        return dest

    def read_file(self, name: str, max_bytes: int | None = None) -> str | None:
        """Read a file from the workspace; returns None if missing.

        *max_bytes* caps how much is pulled into the host process. Sandboxed
        code may write up to ``max_fsize_mb`` (64 MB by default) per file, so
        reading requested files whole would blow past the response budget that
        stdout/stderr already respect.
        """
        dest = self._safe_path(name)
        if not dest.exists():
            return None
        self._ensure_within_workspace(dest.resolve(strict=True))
        if max_bytes is None:
            data = dest.read_bytes()
            truncated = False
        else:
            with dest.open("rb") as fh:
                data = fh.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            if truncated:
                data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            encoded = base64.b64encode(data).decode("ascii")
            text = f"[base64:{len(data)} bytes]\n{encoded}"
        if truncated:
            text += f"\n[truncated: file exceeds {max_bytes} bytes]"
        return text

    def collect_files(
        self, names: list[str], max_bytes: int | None = None
    ) -> dict[str, str]:
        """Return a mapping of {name: content} for each requested file that exists."""
        result: dict[str, str] = {}
        for name in names:
            content = self.read_file(name, max_bytes=max_bytes)
            if content is not None:
                result[name] = content
        return result

    def cleanup(self) -> None:
        if self._tmpdir and Path(self._tmpdir).exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_files(self) -> None:
        for f in self._files:
            if f.content_base64 is not None:
                self.write_binary_file(f.name, base64.b64decode(f.content_base64))
            else:
                self.write_file(f.name, f.content)

    def _safe_path(self, name: str) -> Path:
        """Return a workspace-local path, rejecting absolute or parent escapes."""
        path = Path(name)
        if not name or path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Workspace path must be relative and stay inside workspace: {name!r}"
            )
        candidate = self.host_path / path
        self._ensure_within_workspace(candidate.resolve(strict=False))
        return candidate

    def _ensure_within_workspace(self, path: Path) -> None:
        root = self.host_path.resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Workspace path escapes workspace: {path}") from exc
