# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Pydantic models for nsjail-agent-tool request/response contracts."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Language(str, Enum):
    PYTHON = "python"
    BASH = "bash"
    JAVASCRIPT = "javascript"
    GO = "go"


class ResourceLimits(BaseModel):
    max_cpu_seconds: int = Field(
        default=10, ge=1, le=300, description="CPU time limit in seconds"
    )
    max_wall_seconds: int = Field(
        default=30, ge=1, le=600, description="Wall-clock time limit in seconds"
    )
    max_memory_mb: int = Field(
        default=256, ge=16, le=4096, description="Memory limit in MB"
    )
    max_output_bytes: int = Field(
        default=1_048_576,
        ge=1024,
        le=10_485_760,
        description="Max bytes returned per stdout/stderr/log stream (1 MB default)",
    )
    max_pids: int = Field(
        default=32, ge=1, le=512, description="Max number of processes/threads"
    )
    max_fsize_mb: int = Field(
        default=64, ge=1, le=512, description="Max file size writable in sandbox (MB)"
    )


class NetworkPolicy(str, Enum):
    NONE = "none"  # No network access (default, safest)
    LOOPBACK = "loopback"  # Loopback only


class SandboxConfig(BaseModel):
    network: NetworkPolicy = NetworkPolicy.NONE
    allow_proc: bool = False
    uid: int = Field(
        default=65534, description="UID inside sandbox (nobody by default)"
    )
    gid: int = Field(default=65534, description="GID inside sandbox")


# Payload ceilings. Everything a request carries is written to the host
# filesystem *before* nsjail starts, so none of the in-jail rlimits apply yet;
# without an explicit cap a single large request can exhaust host disk or RAM.
MAX_CODE_CHARS = 1_000_000
MAX_STDIN_CHARS = 1_000_000
MAX_FILE_CHARS = 8_000_000
MAX_FILES = 64
MAX_RETURN_FILES = 64


class WorkspaceFile(BaseModel):
    name: str = Field(description="Filename as seen inside the sandbox")
    content: str = Field(
        max_length=MAX_FILE_CHARS, description="Text content of the file"
    )
    content_base64: str | None = Field(
        default=None,
        max_length=MAX_FILE_CHARS,
        description="Optional base64-encoded binary content. Takes precedence over content.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_workspace_path(value)

    @model_validator(mode="after")
    def validate_content(self) -> "WorkspaceFile":
        if self.content_base64 is not None and self.content:
            raise ValueError("Set either content or content_base64, not both")
        return self


class ExecutionRequest(BaseModel):
    code: str = Field(
        max_length=MAX_CODE_CHARS,
        description="Source code or shell script to execute",
    )
    language: Language = Language.PYTHON
    stdin: str = Field(
        default="",
        max_length=MAX_STDIN_CHARS,
        description="Data piped to the process's stdin",
    )
    files: list[WorkspaceFile] = Field(
        default_factory=list,
        max_length=MAX_FILES,
        description="Extra files placed in workspace before execution",
    )
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    return_files: list[str] = Field(
        default_factory=list,
        max_length=MAX_RETURN_FILES,
        description="Paths (relative to workspace) to read back after execution",
    )

    @field_validator("return_files")
    @classmethod
    def validate_return_files(cls, values: list[str]) -> list[str]:
        return [_validate_workspace_path(value) for value in values]


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    OOM = "oom"
    ERROR = "error"
    SECURITY_VIOLATION = "security_violation"


class ExecutionResult(BaseModel):
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    wall_time_ms: float
    cpu_time_ms: float | None = None
    output_files: dict[str, str] = Field(
        default_factory=dict, description="Content of requested return_files"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


def _validate_workspace_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("Workspace path must be relative and stay inside workspace")
    return value
