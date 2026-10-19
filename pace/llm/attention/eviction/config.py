# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass
from typing import Optional, Union

try:
    from enum import auto, StrEnum
except ImportError:
    from backports.strenum import StrEnum
    from enum import auto


class RoPEMode(StrEnum):
    """How retained keys are re-indexed after an eviction write.

    REINDEX rotates the whole window to [0, budget) from cached unrotated keys
    (rotate-on-read). SINK_ONLY keeps recent keys at their write-time rotation
    and only re-rotates the sinks. Both yield identical relative offsets.
    """

    REINDEX = auto()
    SINK_ONLY = auto()


class UpdateMode(StrEnum):
    """How the recent window makes room for a new token.

    INPLACE (EQUIP) overwrites the oldest recent slot. SHIFT is classic
    StreamingLLM: memmove the recent window left, then append.
    """

    INPLACE = auto()
    SHIFT = auto()


@dataclass(frozen=True)
class EvictionConfig:
    sink_size: int = 4
    recent_size: int = 252
    rope_mode: RoPEMode = RoPEMode.REINDEX
    update: UpdateMode = UpdateMode.INPLACE

    def __post_init__(self):
        if self.sink_size < 0:
            raise ValueError(f"sink_size must be >= 0, got {self.sink_size}")
        if self.recent_size <= 0:
            raise ValueError(f"recent_size must be > 0, got {self.recent_size}")

    @property
    def budget(self) -> int:
        return self.sink_size + self.recent_size


def resolve_eviction_config(
    config: Union["EvictionConfig", dict, None],
) -> Optional["EvictionConfig"]:
    if config is None or isinstance(config, EvictionConfig):
        return config
    if isinstance(config, dict):
        return EvictionConfig(**config)
    raise TypeError(
        f"eviction_config must be an EvictionConfig, dict or None, got {type(config)}"
    )
