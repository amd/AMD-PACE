# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PacePlatform: vLLM CPU platform for PACE.

Subclass of `CpuPlatform` that swaps the vLLM worker class to `PaceWorker`
and routes attention to `PaceAttentionBackend`. Prefix caching is supported
via `PaceSlabConnector` (a KVConnectorBase_V1 implementation that manages
prefix cache lifecycle in SlabPool). All other CPU defaults are inherited.
"""

from __future__ import annotations

import os
import platform as py_platform
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger("pace_vllm.platform")

_PACE_WORKER_CLS = "pace_vllm.v1.worker.cpu_worker.PaceWorker"
_PACE_BACKEND_CLS = "pace_vllm.v1.attention.backends.pace_attn.PaceAttentionBackend"

# Pace `CustomOp` OOTs that need explicit `compilation_config.custom_ops`
# opt-in (vLLM defaults custom_ops=['none'] under Inductor on CPU, which
# routes every CustomOp through forward_native and bypasses our forward_cpu
# overrides). Grouped by the friendly shorthand PACE_VLLM_CUSTOM_OPS accepts.
_PACE_CUSTOM_OP_GROUPS: dict[str, list[str]] = {
    "rms_norm": ["RMSNorm", "GemmaRMSNorm"],
}

_PACE_CUSTOM_OPS_ENV = "PACE_VLLM_CUSTOM_OPS"


def _cpu_supports_avx512_bf16() -> bool:
    """Return True on x86_64 Linux with AVX512F + AVX512_BF16 (Zen4+, SPR+)."""
    if py_platform.system() != "Linux":
        return False
    if py_platform.machine() != "x86_64":
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("flags"):
                    continue
                flags = line.split(":", 1)[1].split()
                return "avx512f" in flags and "avx512_bf16" in flags
    except OSError:
        return False
    return False


def _resolve_pace_custom_op_groups() -> list[str]:
    """Resolve PACE_VLLM_CUSTOM_OPS into a list of group keys.

    Accepts: unset / `all` / `true` / `1` (default = all groups), `none` /
    `""` / `false` / `0` (empty list), or a comma-separated subset of the
    keys in `_PACE_CUSTOM_OP_GROUPS` (currently `rms_norm`). Unknown
    groups log a warning and are ignored. Group keys are what users pass
    in PACE_VLLM_CUSTOM_OPS, so logging *these* (not the underlying class
    names) gives copy-pasteable diagnostics.
    """
    raw = os.environ.get(_PACE_CUSTOM_OPS_ENV)
    if raw is None:
        return list(_PACE_CUSTOM_OP_GROUPS.keys())

    normalized = raw.strip().lower()
    if normalized in ("", "none", "0", "false"):
        return []
    if normalized in ("all", "1", "true"):
        return list(_PACE_CUSTOM_OP_GROUPS.keys())

    requested = [tok.strip().lower() for tok in normalized.split(",") if tok.strip()]
    groups: list[str] = []
    for tok in requested:
        if tok in _PACE_CUSTOM_OP_GROUPS:
            groups.append(tok)
        else:
            logger.warning(
                "pace-vllm: ignoring unknown %s group %r (known: %s).",
                _PACE_CUSTOM_OPS_ENV,
                tok,
                sorted(_PACE_CUSTOM_OP_GROUPS.keys()),
            )
    return groups


def _resolve_pace_custom_op_names() -> list[str]:
    """Flatten the active groups into vLLM's class-name contract for
    `compilation_config.custom_ops`."""
    return [
        name
        for group in _resolve_pace_custom_op_groups()
        for name in _PACE_CUSTOM_OP_GROUPS[group]
    ]


def _is_hybrid_mamba(cache_config) -> bool:
    if getattr(cache_config, "mamba_block_size", None) is not None:
        return True
    mode = getattr(cache_config, "mamba_cache_mode", None)
    return mode not in (None, "none")


def _hf_text_config(hf_config):
    """Prefer the nested text tower on multimodal wrappers (Gemma4, Qwen3.5)."""
    text = getattr(hf_config, "text_config", None)
    if text is not None and hasattr(text, "num_attention_heads"):
        return text
    return hf_config


def _attention_geometry(model_config) -> tuple[int, int]:
    """(num_kv_heads, head_dim) for slab autotune.

    vLLM ModelConfig getters already unwrap `text_config` and pick a
    representative head_dim on heterogeneous models. Flat HF configs
    (and unit-test stubs) fall back to the Llama-style attributes.
    """
    if hasattr(model_config, "get_head_size") and hasattr(
        model_config, "get_total_num_kv_heads"
    ):
        return model_config.get_total_num_kv_heads(), model_config.get_head_size()

    hf_config = _hf_text_config(getattr(model_config, "hf_config", model_config))
    num_attn = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_attn)
    head_dim = getattr(hf_config, "head_dim", None)
    if not head_dim:
        head_dim = hf_config.hidden_size // num_attn
    return int(num_kv_heads), int(head_dim)


def _resolve_slab_block_size_for_connector(vllm_config: "VllmConfig") -> int:
    """Pre-compute slab block size for the connector's extra_config.

    Reads the env override first, otherwise uses the C++ L2 autotuner.
    """
    from pace_vllm.v1.kv_cache import (
        _autotune_slab_block_size,
        _resolve_slab_block_size_override,
    )

    env_override = _resolve_slab_block_size_override()
    if env_override is not None:
        return env_override

    num_kv_heads, head_dim = _attention_geometry(vllm_config.model_config)
    return _autotune_slab_block_size(num_kv_heads, head_dim)


def _inject_slab_connector(vllm_config: "VllmConfig") -> None:
    """Install the connector with scheduler and slab block sizes aligned."""
    from vllm.config import KVTransferConfig

    cache_config = vllm_config.cache_config
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError(
            "pace-vllm: prefix caching cannot yet be combined with speculative "
            "decoding: lookahead slots allocate blocks the step never writes."
        )
    executor = vllm_config.parallel_config.distributed_executor_backend
    if executor not in ("uni", "mp"):
        raise ValueError(
            "pace-vllm: multiprocessing prefix caching requires the 'uni' or "
            f"'mp' executor, got {executor!r}."
        )
    async_scheduling = vllm_config.scheduler_config.async_scheduling
    if async_scheduling is True:
        raise ValueError(
            "pace-vllm: prefix caching does not yet support async scheduling."
        )
    if async_scheduling is None:
        vllm_config.scheduler_config.async_scheduling = False
    existing = vllm_config.kv_transfer_config
    if getattr(cache_config, "kv_offloading_size", None) is not None:
        raise ValueError(
            "pace-vllm: prefix caching cannot yet be combined with KV offloading."
        )
    if existing is not None and existing.kv_connector != "PaceSlabConnector":
        raise ValueError(
            "pace-vllm: prefix caching cannot yet be combined with "
            f"KV connector {existing.kv_connector!r}."
        )
    slab_block_size = _resolve_slab_block_size_for_connector(vllm_config)
    cache_config.enable_prefix_caching = False

    # vLLM derives single-group hash granularity from block_size rather than
    # hash_block_size, so scheduler and slab physical blocks must share it.
    prev_block_size = cache_config.block_size
    cache_config.block_size = slab_block_size

    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="PaceSlabConnector",
        kv_role="kv_both",
        kv_connector_module_path="pace_vllm.v1.connector",
        kv_connector_extra_config={
            **(existing.kv_connector_extra_config if existing is not None else {}),
            "slab_block_size": slab_block_size,
        },
    )

    logger.info(
        "pace-vllm: injected PaceSlabConnector for prefix caching "
        "(slab_block_size=%d, prev_block_size=%d).",
        slab_block_size,
        prev_block_size,
    )


def _enable_pace_custom_ops(vllm_config: "VllmConfig") -> list[str]:
    """Append pace OOT opt-ins to `compilation_config.custom_ops`.

    Each pace OOT selected by PACE_VLLM_CUSTOM_OPS is added as `+<Name>`,
    skipping any already `+`-listed by the user / another plugin or
    `-`-listed (explicitly disabled). Returns the list of *group keys*
    that were resolved -- these are the env-var-shape strings users can
    copy back into `PACE_VLLM_CUSTOM_OPS`.
    """
    cops = vllm_config.compilation_config.custom_ops
    appended_any = False
    for name in _resolve_pace_custom_op_names():
        plus = f"+{name}"
        minus = f"-{name}"
        if plus in cops or minus in cops:
            continue
        cops.append(plus)
        appended_any = True

    # CustomOp.default_on() asserts exactly one of "none" / "all" is in cops.
    # Only add a "none" baseline when we actually appended a `+<Name>` and
    # no baseline exists -- otherwise we'd silently flip vLLM-wide
    # CustomOp dispatch to forward_native for the user.
    if appended_any and cops.count("all") == 0 and cops.count("none") == 0:
        cops.append("none")

    return _resolve_pace_custom_op_groups()


class PacePlatform(CpuPlatform):
    """PACE CPU platform.

    `_enum = PlatformEnum.CPU` stays so `CustomOp` dispatch continues to route
    to `forward_cpu`.
    """

    @classmethod
    def is_available(cls) -> bool:
        return _cpu_supports_avx512_bf16()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """Return the qualname of `PaceAttentionBackend`. vLLM's selector
        imports it and uses `get_impl_cls()` / `get_builder_cls()` from there."""
        if selected_backend is not None:
            logger.debug(
                "pace-vllm: ignoring selected_backend=%s; forcing PACE_SLAB.",
                selected_backend,
            )
        return _PACE_BACKEND_CLS

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        if vllm_config.model_config is None:
            return
        # Pace's slab attention C++ kernel is bf16-only. Catch a non-bf16
        # model dtype here so the user gets an actionable startup error.
        model_dtype = vllm_config.model_config.dtype
        if model_dtype != torch.bfloat16:
            raise ValueError(
                f"pace-vllm: requires dtype=torch.bfloat16, got {model_dtype}. "
                "Pass `dtype='bfloat16'` to vLLM "
                "(LLM/AsyncLLMEngine `dtype=` argument or `--dtype bfloat16` "
                "on the CLI), or unset VLLM_PLUGINS to use stock vLLM CPU."
            )

        # vLLM 0.20+ CpuPlatform rewrites distributed_executor_backend
        # 'uni' -> 'mp' for its OMP shim. PACE sets OMP via the launch
        # script, so we restore 'uni' for TP=1 (saves ~20% IPC on BS=1).
        parallel_config = vllm_config.parallel_config
        prev_executor = parallel_config.distributed_executor_backend
        keep_uniproc = parallel_config.world_size == 1 and prev_executor in (
            None,
            "uni",
        )

        super().check_and_update_config(vllm_config)

        if keep_uniproc and parallel_config.distributed_executor_backend == "mp":
            parallel_config.distributed_executor_backend = "uni"
            logger.info(
                "pace-vllm: keeping distributed_executor_backend='uni' "
                "for TP=1 (CpuPlatform forced 'mp'; OMP env is set by "
                "the launch script)."
            )

        prev = parallel_config.worker_cls
        parallel_config.worker_cls = _PACE_WORKER_CLS

        cache_config = vllm_config.cache_config

        auto_enabled = _enable_pace_custom_ops(vllm_config)

        pace_prefix_caching = cache_config.enable_prefix_caching
        if pace_prefix_caching and _is_hybrid_mamba(cache_config):
            # vLLM 0.28+ auto-enables prefix cache and mamba_cache_mode='align'
            # for hybrid models such as Qwen3.5. Slab prefix restore does not
            # yet cover SSM state, so drop prefix caching rather than abort.
            logger.warning(
                "pace-vllm: disabling prefix caching for hybrid Mamba/SSM "
                "models (mamba_block_size=%s, mamba_cache_mode=%s).",
                getattr(cache_config, "mamba_block_size", None),
                getattr(cache_config, "mamba_cache_mode", None),
            )
            cache_config.enable_prefix_caching = False
            # Match vLLM's MambaModelConfig path when prefix caching is off:
            # mode 'none' and mamba_block_size = max_model_len.
            cache_config.mamba_cache_mode = "none"
            cache_config.mamba_block_size = vllm_config.model_config.max_model_len
            pace_prefix_caching = False
        elif pace_prefix_caching:
            _inject_slab_connector(vllm_config)

        logger.info(
            "pace-vllm: PacePlatform active (worker_cls=%s, was=%s, "
            "attn_backend=%s, prefix_caching=%s, auto_custom_ops=%s).",
            parallel_config.worker_cls,
            prev,
            _PACE_BACKEND_CLS,
            pace_prefix_caching,
            auto_enabled,
        )
