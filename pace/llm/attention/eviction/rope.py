# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Tuple

import torch

from pace.llm.attention.eviction.config import EvictionConfig, RoPEMode
from pace.ops.rotary_embedding import RotaryEmbedding


def cache_owns_rope(kv_cache) -> bool:
    if isinstance(kv_cache, (list, tuple)):
        kv_cache = kv_cache[0] if kv_cache else None
    for candidate in (
        kv_cache,
        getattr(kv_cache, "context", None),
    ):
        if getattr(candidate, "owns_rotation", False):
            return True
    return False


class EvictionRoPE:
    def __init__(self, config: EvictionConfig, rotary: RotaryEmbedding):
        self.config = config
        self.rotary = rotary
        self.mode = config.rope_mode
        if self.mode == RoPEMode.SINK_ONLY and rotary.rope_type == "longrope":
            raise ValueError(
                f"rope_mode='{RoPEMode.SINK_ONLY}' assumes angle linear in "
                f"position; 'longrope' breaks that. Use '{RoPEMode.REINDEX}'."
            )

    def cos_sin(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary(x, positions, return_cos_sin=True)

    def rotate(
        self, x: torch.Tensor, positions: torch.Tensor, unsqueeze_dim: int = 1
    ) -> torch.Tensor:
        cos, sin = self.cos_sin(x, positions)
        return self.rotary.apply_rotary_to(x, cos, sin, unsqueeze_dim=unsqueeze_dim)

    def check_position_budget(self, filled_max: int, budget: int) -> None:
        if self.mode != RoPEMode.SINK_ONLY:
            return
        limit = self.rotary.cos_cache.shape[0]
        if filled_max > limit:
            raise ValueError(
                f"rope_mode='{RoPEMode.SINK_ONLY}' cannot generate past the "
                f"RoPE table ({limit}); reached {filled_max}. Use "
                f"rope_mode='{RoPEMode.REINDEX}' (positions in [0, {budget}))."
            )

    def sink_positions(self, total_seen, budget, batch_size, sink_size) -> torch.Tensor:
        shift = (total_seen - budget).clamp(min=0)
        return shift.unsqueeze(1).expand(batch_size, sink_size)

    def reindex_sinks(self, pristine: torch.Tensor, total_seen, budget, batch, sinks):
        return self.rotate(
            pristine, self.sink_positions(total_seen, budget, batch, sinks)
        )
