# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Builds and launches the nsjail process for a given sandbox configuration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import get_sandbox_paths
from .models import Language, NetworkPolicy, SandboxConfig, ResourceLimits

# Name given to the user's entrypoint file inside the sandbox workspace
_ENTRYPOINT_NAME: dict[Language, str] = {
    Language.PYTHON: "main.py",
    Language.BASH: "main.sh",
    Language.JAVASCRIPT: "main.js",
    Language.GO: "main.go",
}

NSJAIL_BIN = (
    os.environ.get("NSJAIL_BIN") or shutil.which("nsjail") or "/usr/sbin/nsjail"
)
_ALLOWED_ENV_VARS = ["PATH", "HOME", "LANG", "LC_ALL"]


def sandbox_workdir() -> str:
    """Path inside the jail where the workspace is bind-mounted."""
    return get_sandbox_paths().workdir


def build_nsjail_cmd(
    *,
    workspace_host_path: Path,
    language: Language,
    resources: ResourceLimits,
    config: SandboxConfig,
    log_fd: int | None = None,
) -> list[str]:
    """Construct the full nsjail argv list.

    Args:
        workspace_host_path: The host-side temporary directory to bind-mount.
        language: The target language (determines entrypoint and interpreter).
        resources: CPU/memory/pid limits.
        config: Security policy (network, uid, …).
        log_fd: File descriptor number for nsjail's own log output. If None,
                nsjail logs are suppressed via /dev/null.

    Returns:
        A list suitable for passing to ``subprocess.Popen``.
    """
    entrypoint = _ENTRYPOINT_NAME[language]
    interpreter_cmd = _language_cmd(language)
    workdir = sandbox_workdir()

    # Route nsjail's own diagnostic output
    log_arg = str(log_fd) if log_fd is not None else "/dev/null"
    log_flag = ["--log_fd", log_arg] if log_fd is not None else ["--log", "/dev/null"]

    cmd: list[str] = [
        NSJAIL_BIN,
        "--mode",
        "o",  # once — run once and exit
        *log_flag,
        # nsjail's --time_limit is WALL clock; CPU time is a separate rlimit.
        # Passing max_cpu_seconds here would cap wall time at the CPU budget
        # (killing an I/O-bound script early) and leave CPU time unbounded.
        "--time_limit",
        str(resources.max_wall_seconds),
        "--rlimit_cpu",
        str(resources.max_cpu_seconds),
        "--rlimit_as",
        str(resources.max_memory_mb),
        "--rlimit_fsize",
        str(resources.max_fsize_mb),
        "--rlimit_nproc",
        str(resources.max_pids),
        "--max_cpus",
        "1",
        "--user",
        str(config.uid),
        "--group",
        str(config.gid),
        # Bind-mount the workspace read-write so the code can write outputs
        "--bindmount",
        f"{workspace_host_path}:{workdir}",
        "--cwd",
        workdir,
    ]

    # Read-only system mounts needed by the interpreter (added conditionally)
    for ro_path in [*_required_ro_mounts(language), *_extra_ro_mounts()]:
        cmd += ["--bindmount_ro", ro_path]

    # Network policy. nsjail creates an isolated network namespace by default;
    # explicitly keep loopback down for NetworkPolicy.NONE.
    if config.network == NetworkPolicy.NONE:
        cmd += ["--iface_no_lo"]

    # Proc filesystem
    if not config.allow_proc:
        cmd += ["--disable_proc"]

    # Keep the inherited environment narrow and server-owned. In particular,
    # never allow request payloads to expose secrets such as API keys.
    for var in _ALLOWED_ENV_VARS:
        cmd += ["--env", var]

    # Optional explicit PYTHONPATH for the sandbox. Callers use this to expose a
    # bind-mounted directory of pip packages (installed with `pip --target`) on
    # sys.path *without* polluting a shared user-site. Set as an explicit value
    # (VAR=value) rather than a passthrough so the host's own PYTHONPATH never
    # leaks in. Only meaningful for Python.
    sandbox_pythonpath = os.environ.get("NSJAIL_SANDBOX_PYTHONPATH", "").strip()
    if language == Language.PYTHON and sandbox_pythonpath:
        cmd += ["--env", f"PYTHONPATH={sandbox_pythonpath}"]

    # Separator between nsjail flags and the user command
    cmd += ["--", *interpreter_cmd, entrypoint]

    return cmd


def _required_ro_mounts(language: Language) -> list[str]:
    """Return host paths that must be bind-mounted read-only for the given language."""
    common = ["/usr", "/lib", "/lib64", "/etc/passwd", "/etc/group"]
    # Resolving shared libraries installed via Debian's "alternatives" system
    # (e.g. libblas/liblapack, required by numpy/pandas/opencv) needs the
    # /etc/alternatives symlink tree, and the dynamic linker uses
    # /etc/ld.so.cache to locate libraries. Without these, importing numpy in
    # the sandbox fails with "libblas.so.3: cannot open shared object file".
    # Mounted only when present so hosts lacking them still work.
    for optional in ("/etc/alternatives", "/etc/ld.so.cache"):
        if Path(optional).exists():
            common.append(optional)
    extra: dict[Language, list[str]] = {
        Language.PYTHON: ["/usr/lib/python3"],
        Language.JAVASCRIPT: [],
        Language.BASH: ["/bin"],
        Language.GO: ["/usr/local/go"],
    }
    return common + extra.get(language, [])


def _language_cmd(language: Language) -> list[str]:
    """Return the interpreter command for a language, allowing server env overrides."""
    commands: dict[Language, list[str]] = {
        Language.PYTHON: [
            os.environ.get("NSJAIL_PYTHON_BIN", "/usr/bin/python3"),
            "-u",
        ],
        Language.BASH: [os.environ.get("NSJAIL_BASH_BIN", "/usr/bin/bash")],
        Language.JAVASCRIPT: [os.environ.get("NSJAIL_NODE_BIN", "/usr/bin/node")],
        Language.GO: [os.environ.get("NSJAIL_GO_BIN", "/usr/bin/go"), "run"],
    }
    return commands[language]


def _extra_ro_mounts() -> list[str]:
    """Return server-configured read-only mounts, separated by commas.

    Commas keep nsjail's source:dest bind syntax usable, e.g.
    /host/prefix:/sandbox/prefix,/host/data:/data.
    """
    configured = os.environ.get("NSJAIL_EXTRA_RO_MOUNTS", "")
    return [path.strip() for path in configured.split(",") if path.strip()]


def check_nsjail_available() -> None:
    """Raise RuntimeError if nsjail binary is not found on PATH."""
    if not Path(NSJAIL_BIN).exists():
        raise RuntimeError(
            f"nsjail binary not found at '{NSJAIL_BIN}'. "
            "Install nsjail or set the NSJAIL_BIN environment variable."
        )
