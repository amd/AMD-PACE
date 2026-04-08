# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import math

import torch

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 64,
) -> torch.Tensor:
    """
    Dequantize MXFP4-packed blocks into full-precision weights.

    Args:
        blocks: Packed FP4 blocks, shape (..., num_blocks, 16) with uint8
            dtype.
        scales: Per-block scales, shape (..., num_blocks) with uint8 dtype.
        dtype: Output dtype.
        rows_per_chunk: Chunk size to limit peak memory.
    """

    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(
            "MXFP4 blocks/scales must be uint8, got "
            f"{blocks.dtype} and {scales.dtype}"
        )

    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            "MXFP4 blocks shape " f"{blocks.shape} does not match scales {scales.shape}"
        )

    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, num_blocks, packed = blocks.shape
    rows_total = math.prod(prefix_shape) * num_blocks

    blocks = blocks.reshape(rows_total, packed)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, packed * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)

    out = out.reshape(*prefix_shape, num_blocks, packed * 2).view(
        *prefix_shape, num_blocks * packed * 2
    )

    return out
