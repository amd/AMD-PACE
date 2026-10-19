# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Contiguous StreamingLLM / EQUIP KV eviction.

INPLACE (EQUIP) ring-writes permute the sequence axis; attention is invariant
to that permutation. SHIFT is classic StreamingLLM memmove-then-append. RoPE is
re-indexed at read time (REINDEX) or by refreshing sink rows only (SINK_ONLY).

Works on contiguous layouts (``KVCacheType.DYNAMIC`` / ``BMC``). BMC uses the
same segmented capacity growth as ``BMCKVCache``; the keep-set is still capped
at ``sink_size + recent_size``.
"""

import math
import os
from typing import NamedTuple, Optional, Tuple

import torch
from transformers import PretrainedConfig

from pace.llm.attention.base import Cache, CacheContext, KVCacheBase, KVCacheType
from pace.llm.attention.eviction.config import EvictionConfig, RoPEMode, UpdateMode
from pace.llm.attention.eviction.rope import EvictionRoPE


class EvictionState:
    def __init__(self, config: EvictionConfig, batch_size: int = 1):
        self.sink_size = config.sink_size
        self.recent_size = config.recent_size
        self.budget = config.budget
        self.update = config.update
        self.batch_size = batch_size
        self.filled = torch.zeros(batch_size, dtype=torch.long)
        self.write_idx = torch.full((batch_size,), self.sink_size, dtype=torch.long)
        self.total_seen = torch.zeros(batch_size, dtype=torch.long)

    def resize(self, batch_size: int) -> None:
        if batch_size == self.batch_size:
            return
        self.batch_size = batch_size
        self.filled = torch.zeros(batch_size, dtype=torch.long)
        self.write_idx = torch.full((batch_size,), self.sink_size, dtype=torch.long)
        self.total_seen = torch.zeros(batch_size, dtype=torch.long)

    @property
    def is_full(self) -> torch.Tensor:
        return self.filled >= self.budget

    @property
    def any_full(self) -> bool:
        return bool(self.is_full.any())

    @property
    def window(self) -> int:
        return int(self.filled.max())

    def admit(self, num_new: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (write slots, per-row recent-window shift counts)."""
        if self.update == UpdateMode.SHIFT:
            return self._admit_shift(num_new)
        return self._admit_inplace(num_new), torch.zeros_like(self.filled)

    def _admit_inplace(self, num_new: int) -> torch.Tensor:
        offsets = torch.arange(num_new, dtype=torch.long).unsqueeze(0)
        virtual = self.filled.unsqueeze(1) + offsets
        ring_base = torch.where(
            self.is_full,
            self.write_idx - self.sink_size,
            torch.zeros_like(self.write_idx),
        ).unsqueeze(1)
        slots = torch.where(
            virtual < self.budget,
            virtual,
            self.sink_size + (ring_base + virtual - self.budget) % self.recent_size,
        )
        ring_count = (self.filled + num_new - self.budget).clamp(min=0)
        self.filled = (self.filled + num_new).clamp(max=self.budget)
        self.write_idx = torch.where(
            self.is_full,
            self.sink_size + (ring_base.squeeze(1) + ring_count) % self.recent_size,
            torch.full_like(self.write_idx, self.sink_size),
        )
        self.total_seen += num_new
        return slots

    def _admit_shift(self, num_new: int) -> Tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(num_new, dtype=torch.long).unsqueeze(0)
        overflow = (self.filled + num_new - self.budget).clamp(min=0)
        write_start = self.filled - overflow
        slots = write_start.unsqueeze(1) + offsets
        self.filled = (self.filled + num_new).clamp(max=self.budget)
        self.write_idx = torch.full_like(self.write_idx, self.sink_size)
        self.total_seen += num_new
        return slots, overflow

    def logical_positions(self) -> torch.Tensor:
        positions = (
            torch.arange(self.window, dtype=torch.long)
            .unsqueeze(0)
            .expand(self.batch_size, self.window)
            .clone()
        )
        if (
            self.update == UpdateMode.SHIFT
            or self.window > self.budget
            or not self.any_full
        ):
            return positions
        age = torch.arange(self.recent_size, dtype=torch.long).unsqueeze(0)
        slots = (
            self.sink_size
            + (self.write_idx.unsqueeze(1) - self.sink_size + age) % self.recent_size
        )
        ring = positions.clone()
        ring[:, : self.sink_size] = torch.arange(self.sink_size)
        ring.scatter_(
            1, slots, (self.sink_size + age).expand(self.batch_size, self.recent_size)
        )
        return torch.where(self.is_full.unsqueeze(1), ring, positions)


class StepPlan(NamedTuple):
    slots: torch.Tensor
    logical_positions: torch.Tensor
    query_positions: torch.Tensor
    compact_index: Optional[torch.Tensor]
    leading_pad: Optional[torch.Tensor]
    shift_by: Optional[torch.Tensor] = None
    filled_before: Optional[torch.Tensor] = None


class EvictingKVCache(KVCacheBase):
    def __init__(self, context: "EvictingCacheContext", layer_idx: int):
        super().__init__()
        self.context = context
        self.layer_idx = layer_idx
        self.token = context.token
        self.key = self.value = self.sink_pristine = None

    @property
    def seq_len(self) -> int:
        return self.context.window

    @property
    def kv_valid_lens(self) -> Optional[torch.Tensor]:
        return self.context.kv_valid_lens

    def _allocate(self, key_states: torch.Tensor, num_new: int) -> None:
        needed = max(self.context.state.budget, num_new)
        if self.context.capacity < needed:
            self.context.capacity = self.context.round_capacity(needed)
        shape = list(key_states.shape)
        shape[2] = self.context.capacity
        kwargs = dict(dtype=key_states.dtype, device=key_states.device)
        if self.key is None:
            self.key = torch.zeros(shape, **kwargs)
            self.value = torch.zeros(shape, **kwargs)
            return
        # BMC-style grow: copy the live prefix into a larger segment.
        new_key = torch.empty(shape, **kwargs)
        new_value = torch.empty(shape, **kwargs)
        old_len = self.key.size(2)
        new_key.narrow(2, 0, old_len).copy_(self.key)
        new_value.narrow(2, 0, old_len).copy_(self.value)
        if self.context.capacity > old_len:
            new_key.narrow(2, old_len, self.context.capacity - old_len).zero_()
            new_value.narrow(2, old_len, self.context.capacity - old_len).zero_()
        self.key, self.value = new_key, new_value

    def _compact(self, index: torch.Tensor) -> None:
        bsz, heads, _, _ = self.key.shape
        budget = index.shape[1]

        def gathered(src):
            w = src.shape[-1]
            return src.gather(
                2, index[:, None, :, None].expand(bsz, heads, budget, w)
            ).contiguous()

        key, value = gathered(self.key), gathered(self.value)
        capacity = self.context.round_capacity(budget)
        if capacity > budget:
            shape = list(key.shape)
            shape[2] = capacity
            padded_key = key.new_zeros(shape)
            padded_value = value.new_zeros(shape)
            padded_key.narrow(2, 0, budget).copy_(key)
            padded_value.narrow(2, 0, budget).copy_(value)
            key, value = padded_key, padded_value
        self.key, self.value = key, value
        self.context.capacity = capacity
        self.sink_pristine = None

    def _write(self, slots, key_states, value_states) -> None:
        num_new = slots.shape[1]
        if bool((slots == slots[0]).all()):
            first = slots[0]
            self.key.index_copy_(2, first, key_states)
            self.value.index_copy_(2, first, value_states)
            return
        bsz, heads, _, dim = self.key.shape
        idx = slots[:, None, :, None].expand(bsz, heads, num_new, dim)
        self.key.scatter_(2, idx, key_states)
        self.value.scatter_(2, idx, value_states)

    def _shift_recent(
        self, shift_by: torch.Tensor, filled_before: torch.Tensor
    ) -> None:
        if not bool((shift_by > 0).any()):
            return
        sink = self.context.state.sink_size
        # Homogeneous batches (benchmark / typical decode) share one shift.
        if bool((shift_by == shift_by[0]).all()) and bool(
            (filled_before == filled_before[0]).all()
        ):
            o = int(shift_by[0])
            filled = int(filled_before[0])
            if o <= 0 or filled <= sink:
                return
            src_end = filled
            src_start = sink + o
            dst_len = src_end - src_start
            if dst_len <= 0:
                return

            def move(buf):
                src = buf[:, :, src_start:src_end].clone()
                buf[:, :, sink : sink + dst_len].copy_(src)

            move(self.key)
            move(self.value)
            return
        raise NotImplementedError(
            "Per-row SHIFT amounts differ across the batch; "
            "use homogeneous sequence lengths with UpdateMode.SHIFT."
        )

    def _refresh_sinks(self) -> None:
        rope, state = self.context.rope, self.context.state
        if (
            rope is None
            or state.sink_size == 0
            or not state.any_full
            or state.window > state.budget
        ):
            return
        if self.sink_pristine is None:
            self.sink_pristine = self.key.narrow(2, 0, state.sink_size).clone()
        self.key.narrow(2, 0, state.sink_size).copy_(
            rope.reindex_sinks(
                self.sink_pristine,
                state.total_seen,
                state.budget,
                state.batch_size,
                state.sink_size,
            )
        )

    def _keys_for_attention(self, window: int) -> torch.Tensor:
        keys = self.key.narrow(2, 0, window)
        rope = self.context.rope
        if rope is None or rope.mode != RoPEMode.REINDEX:
            return keys
        return rope.rotate(keys, self.context.plan.logical_positions)

    def update_cache(
        self, key_states, value_states, concat_dim, positions=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if concat_dim != 2:
            raise ValueError(f"Eviction requires BNSH dim 2, got {concat_dim}")
        plan = self.context.plan_step(
            key_states.shape[2], self.layer_idx, key_states.shape[0], positions
        )
        needed = max(
            self.context.state.budget,
            key_states.shape[2],
            int(plan.slots.max().item()) + 1 if plan.slots.numel() else 0,
        )
        if (
            self.key is None
            or self.key.size(2) < needed
            or self.context.capacity < needed
        ):
            self._allocate(key_states, needed)
        if plan.compact_index is not None:
            self._compact(plan.compact_index)
        if plan.shift_by is not None and plan.filled_before is not None:
            self._shift_recent(plan.shift_by, plan.filled_before)
        self._write(plan.slots, key_states, value_states)
        if (
            self.context.rope is not None
            and self.context.rope.mode == RoPEMode.SINK_ONLY
        ):
            self._refresh_sinks()
        w = self.context.window
        return self._keys_for_attention(w), self.value.narrow(2, 0, w)

    def remove_cache(self, remove_len: int) -> None:
        self.context.rollback(remove_len)


class EvictingCacheContext(CacheContext):
    def __init__(
        self,
        config: PretrainedConfig,
        max_seq_length: int,
        eviction_config: EvictionConfig,
        token: str = "",
        rotary=None,
        batch_size: int = 1,
        kv_cache_type: KVCacheType = KVCacheType.DYNAMIC,
    ):
        self.token = token
        self.kv_cache_type = KVCacheType.get_kv_cache_type(kv_cache_type)
        self.max_seq_length = max_seq_length
        self.tokens_per_split = None
        if self.kv_cache_type == KVCacheType.BMC:
            num_splits = int(
                os.getenv("PACE_BMC_NUM_SPLITS", int(math.sqrt(max_seq_length)))
            )
            self.tokens_per_split = max(1, max_seq_length // max(num_splits, 1))
        self.state = EvictionState(eviction_config, batch_size)
        self.rope = (
            EvictionRoPE(eviction_config, rotary) if rotary is not None else None
        )
        self.capacity = self.window = 0
        self.left_aligned = False
        self.kv_valid_lens = self.plan = self._leading_pad = None
        self.cache_objects = [
            EvictingKVCache(self, i) for i in range(config.num_hidden_layers)
        ]

    def round_capacity(self, needed: int) -> int:
        if self.kv_cache_type != KVCacheType.BMC or self.tokens_per_split is None:
            return needed
        tps = self.tokens_per_split
        return ((max(needed, 1) + tps - 1) // tps) * tps

    def __len__(self) -> int:
        return self.window

    def __getitem__(self, idx: int) -> EvictingKVCache:
        return self.cache_objects[idx]

    @property
    def owns_rotation(self) -> bool:
        return self.rope is not None and self.rope.mode == RoPEMode.REINDEX

    def plan_step(self, num_new, layer_idx, batch_size, positions) -> StepPlan:
        if layer_idx != 0:
            if self.plan is None:
                raise RuntimeError(
                    "Eviction step plan missing; layer 0 must run first."
                )
            return self.plan

        self.state.resize(batch_size)
        sequential_overlong = (
            not self.left_aligned and self.window == 0 and num_new > self.state.budget
        )
        wraps_ring = self.left_aligned or (
            self.window > 0 and self.window + num_new > self.state.budget
        )
        if num_new > 1 and wraps_ring and not sequential_overlong:
            raise NotImplementedError(
                "KV eviction does not support multi-token updates after the "
                "recent window is permuted."
            )
        leading_pad = None
        if not self.left_aligned:
            leading_pad = (
                torch.zeros(batch_size, dtype=torch.long)
                if positions is None
                else (self.window + num_new) - (positions[:, -1] + 1)
            )
            self._leading_pad = leading_pad.to(torch.long)

        budget = self.state.budget
        compact_index = None
        shift_by = filled_before = None
        if not self.left_aligned and self.window > 0 and self.window + num_new > budget:
            compact_index = self._compaction_index()

        if not self.left_aligned and num_new > budget:
            slots = (
                torch.arange(num_new, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, num_new)
                .contiguous()
            )
            self.state.filled = torch.full((batch_size,), num_new, dtype=torch.long)
            self.state.total_seen = self.state.filled.clone()
            if self._leading_pad is not None:
                self.state.total_seen = self.state.filled - self._leading_pad
        else:
            filled_before = self.state.filled.clone()
            slots, shift_by = self.state.admit(num_new)
            if not bool((shift_by > 0).any()):
                shift_by = filled_before = None

        if self.rope is not None:
            self.rope.check_position_budget(
                int(self.state.total_seen.max()), self.state.budget
            )

        self.window = self.state.window
        self.kv_valid_lens = (
            self.state.filled
            if self.left_aligned
            and not bool((self.state.filled == self.state.filled[0]).all())
            else None
        )
        logical = self.state.logical_positions()
        self.plan = StepPlan(
            slots=slots,
            logical_positions=logical,
            query_positions=logical.gather(1, slots),
            compact_index=compact_index,
            leading_pad=None if self.left_aligned else leading_pad,
            shift_by=shift_by,
            filled_before=filled_before,
        )
        return self.plan

    def _compaction_index(self) -> Optional[torch.Tensor]:
        budget, sink, window = self.state.budget, self.state.sink_size, self.window
        batch = self.state.batch_size
        pads = (
            torch.zeros(batch, dtype=torch.long)
            if self._leading_pad is None
            else self._leading_pad
        )
        rows, keeps = [], []
        for b in range(batch):
            pad, valid = int(pads[b]), window - int(pads[b])
            keep = min(valid, budget)
            if keep >= budget:
                idx = torch.cat(
                    [
                        torch.arange(pad, pad + sink),
                        torch.arange(window - (budget - sink), window),
                    ]
                )
            else:
                idx = torch.arange(pad, window)
                idx = torch.cat([idx, idx.new_full((budget - keep,), max(pad, 0))])
            rows.append(idx)
            keeps.append(keep)

        self.left_aligned = True
        self.state.filled = torch.tensor(keeps, dtype=torch.long)
        self.state.write_idx = torch.full((batch,), sink, dtype=torch.long)
        self.state.total_seen = torch.tensor(
            [window - int(pads[b]) for b in range(batch)], dtype=torch.long
        )
        self.window = int(self.state.filled.max())
        self._leading_pad = None
        index = torch.stack(rows)
        return None if window <= budget and bool((pads == 0).all()) else index

    def rollback(self, remove_len: int) -> None:
        if self.left_aligned and int(self.state.total_seen.max()) > self.window:
            raise NotImplementedError(
                "Cannot roll back an evicting KV cache: in-place updates "
                "overwrite evicted entries. Disable eviction for speculative decoding."
            )
        self.state.filled = (self.state.filled - remove_len).clamp(min=0)
        self.state.total_seen = (self.state.total_seen - remove_len).clamp(min=0)
        self.state.write_idx = torch.full_like(
            self.state.write_idx, self.state.sink_size
        )
        self.window = int(self.state.filled.max())

    def remove_cache(self, remove_len: int) -> None:
        self.rollback(remove_len)


class EvictingCache(Cache):
    def __init__(
        self,
        eviction_config: EvictionConfig,
        rotary=None,
        kv_cache_type: KVCacheType = KVCacheType.DYNAMIC,
    ):
        self.eviction_config = eviction_config
        self.rotary = rotary
        self.kv_cache_type = KVCacheType.get_kv_cache_type(kv_cache_type)

    def bind_rotary(self, rotary) -> None:
        self.rotary = rotary

    def create_context(self, config, max_seq_length, **kwargs) -> EvictingCacheContext:
        return EvictingCacheContext(
            config,
            max_seq_length,
            self.eviction_config,
            token=kwargs.get("token", ""),
            rotary=self.rotary,
            batch_size=kwargs.get("batch_size", 1),
            kv_cache_type=self.kv_cache_type,
        )

    def merge_contexts(self, contexts, query_len=1):
        if len(contexts) != 1:
            raise NotImplementedError(
                "KV eviction cannot batch independent requests: each owns its "
                "own ring, and the per-sequence cache list bypasses "
                "eviction-aware masking and cache-owned RoPE."
            )
        return contexts[0]

    def build_prefill_metadata(self, context, seq_len, past_len=0):
        return None

    def remove_context(self, context):
        pass
