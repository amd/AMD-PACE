# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""TPP weight-packing helpers for pace-vLLM overrides."""

from __future__ import annotations

import os

import torch


def _resolve_block_size_from_env() -> int:
    """Return the configured libXSMM block size."""
    raw = os.getenv("LIBXSMM_BLOCK_SIZE", "32")
    try:
        block_size = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"pace-vllm: invalid LIBXSMM_BLOCK_SIZE={raw!r}; "
            "must be a positive integer."
        ) from exc
    if block_size <= 0:
        raise ValueError(
            f"pace-vllm: LIBXSMM_BLOCK_SIZE={block_size} must be positive."
        )
    return block_size


def tpp_prepack(
    weight: torch.Tensor,
    *,
    block_size: int | None = None,
) -> torch.Tensor | None:
    """Pack a `[out_features, in_features]` bf16 weight into PACE's TPP layout.

    The TPP / libXSMM path expects weights in the 5D packed layout that
    `pace/ops/backends/tpp.py::TPPLinear.preprocess` produces. Shape is:

        (out // block_size, in // 64, 32, block_size, 2)

    which requires `out % block_size == 0` AND `in % 64 == 0`. We return
    `None` when either condition fails so the caller can fall back to
    vLLM's stock CPU linear dispatch (oneDNN / `F.linear`) for that
    layer instead of silently producing a slow path.

    `block_size` defaults to the `LIBXSMM_BLOCK_SIZE` env if set (32
    otherwise), matching `pace/ops/backends/tpp.py::TPPLinear`.
    """
    # Mirrors stock vLLM's `dispatch_cpu_unquantized_gemm` precondition:
    # weight loading on CPU sometimes leaves a meta tensor behind during
    # init. Packing one would produce a meta-storage 5D tensor that
    # crashes opaquely inside the C++ kernel; fall back instead.
    if weight.is_meta:
        return None
    if weight.dtype != torch.bfloat16:
        # TPP is bf16-only; signal fallback rather than silently
        # upcasting here.
        return None
    if weight.dim() != 2:
        return None

    if block_size is None:
        block_size = _resolve_block_size_from_env()

    out_f, in_f = weight.shape
    if out_f % block_size != 0 or in_f % 64 != 0:
        return None

    # Same reshape + permute as pace/ops/backends/tpp.py::TPPLinear.preprocess.
    # 5D target layout: (out // block_size, in // 64, 32, block_size, 2).
    reshaped = weight.reshape(out_f // block_size, block_size, in_f // 64, 32, 2)
    return reshaped.permute(0, 2, 3, 1, 4).contiguous()


def tpp_prepack_moe(
    weight: torch.Tensor,
    *,
    block_size: int | None = None,
) -> torch.Tensor | None:
    """Pack `[E, out, in]` bf16 weights into PACE's 6D MoE layout."""
    if weight.is_meta:
        return None
    if weight.dtype != torch.bfloat16:
        return None
    if weight.dim() != 3:
        return None

    if block_size is None:
        block_size = _resolve_block_size_from_env()

    e, out_f, in_f = weight.shape
    if out_f % block_size != 0 or in_f % 64 != 0:
        return None

    reshaped = weight.reshape(e, out_f // block_size, block_size, in_f // 64, 32, 2)
    return reshaped.permute(0, 1, 3, 4, 2, 5).contiguous()
