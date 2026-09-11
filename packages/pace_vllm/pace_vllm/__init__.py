# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""pace-vllm: vLLM platform plugin for PACE (CPU).

Self-contained at runtime: ships its own `libpace_cpp.so` and a build-time
snapshot of `pace/_register_fake.py`. The bundled `libpace_cpp.so` is loaded
via `torch.ops.load_library`, which fires the `TORCH_LIBRARY_FRAGMENT` static
initializers and populates `torch.ops.pace.*` + `torch.classes.pace.*`. The
`pace` Python package is never imported.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from ._version import __version__, __version_tuple__  # noqa: F401
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0)


# Extend this hard-coded range only after end-to-end verification because
# pace-vllm integrates with vLLM's internal APIs.
_PACE_VLLM_SUPPORTED_VLLM_RANGE = ">=0.25.0,<=0.28.0"


# Module-level logger. Re-initialised by `_init_logging()` on first
# `register()` call so vLLM imports stay lazy and `import pace_vllm` succeeds
# even in environments where vllm isn't installed yet.
logger: logging.Logger = logging.getLogger("pace_vllm")


def _init_logging() -> None:
    """Attach vLLM's configured handlers to the `pace_vllm` logger so pace
    INFO/WARNING lines land in the same stream as vLLM's. Idempotent."""
    global logger
    from vllm.envs import VLLM_LOGGING_LEVEL
    from vllm.logger import init_logger

    vllm_logger = logging.getLogger("vllm")
    pace_logger = logging.getLogger("pace_vllm")
    pace_logger.setLevel(logging.getLevelName(VLLM_LOGGING_LEVEL))
    if vllm_logger.handlers and not pace_logger.handlers:
        for handler in vllm_logger.handlers:
            pace_logger.addHandler(handler)
        pace_logger.propagate = False
    logger = init_logger("pace_vllm")


_ops_loaded = False


def _check_vllm_version_supported() -> "tuple[bool, str, str, str]":
    """Return `(supported, vllm_version, range_spec, reason)`. `reason` is
    empty on success and a short string explaining the mismatch on failure."""
    range_spec = _PACE_VLLM_SUPPORTED_VLLM_RANGE

    try:
        import vllm
    except ImportError as exc:
        return False, "(not installed)", range_spec, f"vllm import failed: {exc}"

    vllm_version = getattr(vllm, "__version__", None) or "(unknown)"

    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:  # pragma: no cover
        return False, vllm_version, range_spec, f"packaging unavailable: {exc}"

    try:
        spec = SpecifierSet(range_spec)
    except InvalidSpecifier as exc:  # pragma: no cover
        return (
            False,
            vllm_version,
            range_spec,
            f"_PACE_VLLM_SUPPORTED_VLLM_RANGE is not a valid PEP 440 "
            f"specifier (fix in pace_vllm/__init__.py): {exc}",
        )

    try:
        version = Version(vllm_version)
    except InvalidVersion as exc:
        return False, vllm_version, range_spec, f"unparseable vllm version: {exc}"

    if spec.contains(version, prereleases=True):
        return True, vllm_version, range_spec, ""

    return (
        False,
        vllm_version,
        range_spec,
        f"vllm {vllm_version} is outside the supported range",
    )


def _pace_ops_already_registered() -> bool:
    """True if pace's torch surfaces are already populated (e.g. another
    import of `pace` in this process loaded the C++ library).

    Uses `torch.classes.pace.SlabPool` as the sentinel rather than a specific
    op like `pace::rmsnorm`. The SlabPool class is the central pace
    abstraction and far less likely to be renamed than individual ops.
    """
    import torch

    try:
        torch.classes.pace.SlabPool  # noqa: B018
        return True
    except (AttributeError, RuntimeError):
        return False


def _load_pace_native() -> None:
    """Load the bundled pace C++ library + fake-op snapshot exactly once.

    Both `torch.ops.load_library` and `_fakes_snapshot` sit under
    `_pace_ops_already_registered()`: if pace's torch surfaces are already
    populated (e.g. `pace` was imported elsewhere in this process),
    re-importing our snapshot would call `torch.library.register_fake` a
    second time on the same op names and raise -- which `register()`'s outer
    try/except would swallow, silently disabling the plugin.
    """
    global _ops_loaded
    if _ops_loaded:
        return

    import torch

    if not _pace_ops_already_registered():
        lib_path = Path(__file__).resolve().parent / "lib" / "libpace_cpp.so"
        torch.ops.load_library(str(lib_path))
        from . import _fakes_snapshot  # noqa: F401

        logger.info(
            "pace-vllm: loaded bundled libpace_cpp.so via torch.ops.load_library."
        )
    else:
        logger.info(
            "pace-vllm: pace ops already registered; skipping load_library "
            "and _fakes_snapshot import."
        )

    if not hasattr(torch.ops.pace, "rmsnorm"):
        raise RuntimeError(
            "pace-vllm: torch.ops.pace.rmsnorm missing after libpace_cpp.so load"
        )
    if not hasattr(torch.classes.pace, "SlabPool"):
        raise RuntimeError(
            "pace-vllm: torch.classes.pace.SlabPool missing after libpace_cpp.so load"
        )
    logger.info(
        "pace-vllm: native surfaces live (torch.ops.pace.rmsnorm, "
        "torch.classes.pace.SlabPool)."
    )

    _ops_loaded = True


def register() -> "str | None":
    """vLLM platform-plugin entry point. Returns the fully-qualified
    `PacePlatform` import path string (`"pace_vllm.platform.PacePlatform"`)
    on supported hosts, or `None` on a vLLM version mismatch, native load
    failure, or unsupported CPU."""
    # Version check first: _init_logging() imports vllm.envs / vllm.logger
    # and would raise ImportError without vllm, defeating the graceful-None
    # path. Early-return falls back to the stdlib logger (lastResort -> stderr).
    supported, vllm_version, range_spec, reason = _check_vllm_version_supported()
    if not supported:
        logger.warning(
            "pace-vllm: skipping plugin registration -- %s (current=%s, "
            "supported=%r). Upgrade / downgrade vllm to a supported version, "
            "or update _PACE_VLLM_SUPPORTED_VLLM_RANGE in pace_vllm/__init__.py "
            "after verifying end-to-end.",
            reason,
            vllm_version,
            range_spec,
        )
        return None

    _init_logging()
    logger.info(
        "pace-vllm: vllm %s satisfies supported range %r.",
        vllm_version,
        range_spec,
    )

    try:
        _load_pace_native()
    except Exception as exc:  # pragma: no cover
        logger.warning("pace-vllm: native load failed (%s); plugin disabled.", exc)
        return None

    from pace_vllm.platform import PacePlatform

    if not PacePlatform.is_available():
        logger.info("pace-vllm: platform not available on this host; plugin inactive.")
        return None

    logger.info("pace-vllm: plugin active, PacePlatform registered.")
    return "pace_vllm.platform.PacePlatform"


__all__ = ["__version__", "__version_tuple__", "register"]
