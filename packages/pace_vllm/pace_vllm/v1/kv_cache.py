# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PaceKVCache: per-layer slab-pool KV cache owned by pace-vllm.

Wraps one `torch.classes.pace.SlabPool` per attention layer. Tracks a
`request_id -> int64 sequence_id` registry so vLLM's scheduler (which deals
in string request ids) can drive allocation / teardown without the slab
kernels seeing strings. `PaceModelRunner` registers each layer in
`pace_vllm.v1.attention.layer_registry` so `PaceAttentionImpl.forward`
resolves the pool by `layer.layer_name`.

Sizing: `from_kv_cache_config` assigns equal `num_blocks` per layer
under `num_blocks = budget // sum(page_size_bytes_per_layer)`, where
`block_size` per layer comes from `pace::slab_autotune_block_size`
(not vLLM's `AttentionSpec.block_size`, which is scheduler accounting
only). Heterogeneous geometry across layers is supported because each
`SlabPool` is constructed with its own scalars.

Prefix caching is delegated to PaceSlabConnector (KVConnectorBase_V1)
which manages the hash→block map and slab retain/release lifecycle.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger("pace_vllm.v1.kv_cache")


_PACE_VLLM_SLAB_BLOCK_SIZE_ENV = "PACE_VLLM_SLAB_BLOCK_SIZE"


@dataclass(frozen=True)
class PaceLayerSpec:
    """Geometry for one attention layer's SlabPool. Per-layer scalars
    let one PaceKVCache hold layers with different geometry."""

    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: torch.dtype
    num_blocks: int

    @property
    def bytes_per_block(self) -> int:
        # K + V; bf16 only (enforced in PaceKVCache.__init__) so dtype
        # element size is 2 bytes.
        return 2 * self.num_kv_heads * self.block_size * self.head_dim * 2

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_block * self.num_blocks


@dataclass(frozen=True)
class PaceKVCacheSpec:
    """Per-layer slab specs. Layers are independent; geometry can vary."""

    per_layer: tuple[PaceLayerSpec, ...]

    @classmethod
    def uniform(
        cls,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
    ) -> "PaceKVCacheSpec":
        """Build a spec with identical geometry on every layer.

        Convenience for unit tests and for the pre-hybrid Llama-style
        path where every layer is full-attention with the same demand.
        """
        layer = PaceLayerSpec(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            num_blocks=num_blocks,
        )
        return cls(per_layer=tuple([layer] * num_layers))

    @property
    def num_layers(self) -> int:
        return len(self.per_layer)

    @property
    def total_blocks(self) -> int:
        return sum(layer.num_blocks for layer in self.per_layer)

    @property
    def total_bytes(self) -> int:
        return sum(layer.total_bytes for layer in self.per_layer)

    @property
    def per_layer_num_blocks(self) -> tuple[int, ...]:
        return tuple(layer.num_blocks for layer in self.per_layer)


def _pace_kvcache_budget_bytes(vllm_config: "VllmConfig") -> int | None:
    """Return `cache_config.kv_cache_memory_bytes` if positive, else None.

    Populated either by the user (via `--kv-cache-memory-bytes` or
    `VLLM_CPU_KVCACHE_SPACE`) or by `PaceWorker.determine_available_memory`
    stashing vLLM's auto-computed value back onto the config so the
    KV-cache builder reads a consistent field regardless of source.
    """
    cc = getattr(vllm_config, "cache_config", None)
    if cc is None:
        return None
    val = getattr(cc, "kv_cache_memory_bytes", None)
    if val is not None and int(val) > 0:
        return int(val)
    return None


def _prefix_connector_block_size(vllm_config: "VllmConfig") -> int | None:
    transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if transfer_config is None:
        return None
    if transfer_config.kv_connector == "PaceSlabConnector":
        extra_config = transfer_config.kv_connector_extra_config or {}
    else:
        return None
    raw_block_size = extra_config.get("slab_block_size")
    if raw_block_size is None:
        raise ValueError(
            "pace-vllm: PaceSlabConnector requires slab_block_size in "
            "kv_connector_extra_config."
        )
    block_size = int(raw_block_size)
    if block_size <= 0:
        raise ValueError(
            "pace-vllm: PaceSlabConnector slab_block_size must be positive."
        )
    return block_size


def _resolve_slab_block_size_override() -> int | None:
    """Read `PACE_VLLM_SLAB_BLOCK_SIZE`. Returns None when the env var is
    unset or empty (autotuner runs); raises ValueError on parse failure or
    non-positive values, per AGENTS.md env var hygiene.

    Owner-tunable knob to skip the C++ L2 autotuner for tuning experiments.
    """
    raw = os.environ.get(_PACE_VLLM_SLAB_BLOCK_SIZE_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"pace-vllm: invalid {_PACE_VLLM_SLAB_BLOCK_SIZE_ENV}={raw!r}; "
            "must be a positive integer."
        ) from exc
    if value <= 0:
        raise ValueError(
            f"pace-vllm: {_PACE_VLLM_SLAB_BLOCK_SIZE_ENV}={value} must be positive."
        )
    return value


def _autotune_slab_block_size(num_kv_heads: int, head_dim: int) -> int:
    """Call the C++ L2 autotuner for one `(num_kv_heads, head_dim)`.

    Wrapped in a free function so tests can monkey-patch it via
    `pace_vllm.v1.kv_cache._autotune_slab_block_size` without going
    through `torch.ops.pace.*`.
    """
    return int(torch.ops.pace.slab_autotune_block_size(num_kv_heads, head_dim))


class PaceKVCache:
    """Owner of per-layer `torch.classes.pace.SlabPool` instances.

    Thread-compatible with a single worker; not safe across workers.
    """

    def __init__(
        self,
        spec: PaceKVCacheSpec,
        layer_names: list[str] | None = None,
    ) -> None:
        for i, layer in enumerate(spec.per_layer):
            if layer.dtype != torch.bfloat16:
                raise ValueError(
                    f"PaceKVCache layer {i}: requires BF16 KV cache dtype, "
                    f"got {layer.dtype}. See csrc/ops/slab_attention.cpp "
                    "validation wrappers."
                )
        if layer_names is not None and len(layer_names) != spec.num_layers:
            raise ValueError(
                f"PaceKVCache: layer_names length ({len(layer_names)}) does "
                f"not match spec.num_layers ({spec.num_layers})."
            )

        self.spec = spec
        self._pools = [
            torch.classes.pace.SlabPool(
                layer.num_blocks,
                layer.num_kv_heads,
                layer.head_dim,
                layer.block_size,
            )
            for layer in spec.per_layer
        ]
        # Production passes names so attention layers can resolve their pool;
        # direct construction without names is retained for unit tests.
        self._layer_name_to_idx: dict[str, int] = (
            {name: idx for idx, name in enumerate(layer_names)}
            if layer_names is not None
            else {}
        )
        self._request_to_seq: dict[str, int] = {}
        self._seq_id_counter = itertools.count(start=1)
        # Attention resolves request IDs through the runner's persistent batch.
        self._owner: object | None = None
        self._block_size = spec.per_layer[0].block_size
        self._prefix_block_indices_aligned = all(
            layer.block_size == self._block_size for layer in spec.per_layer
        )

        sample = spec.per_layer[0]
        unique_geometries = len(
            {
                (layer.num_kv_heads, layer.head_dim, layer.block_size)
                for layer in spec.per_layer
            }
        )
        logger.info(
            "pace-vllm: PaceKVCache allocated (layers=%d, geometries=%d, "
            "sample block_size=%d num_kv_heads=%d head_dim=%d, "
            "total=%.2f GiB).",
            spec.num_layers,
            unique_geometries,
            sample.block_size,
            sample.num_kv_heads,
            sample.head_dim,
            spec.total_bytes / (1024**3),
        )

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: "KVCacheConfig",
        vllm_config: "VllmConfig",
        shared_kv_cache_layers: dict[str, str] | None = None,
    ) -> "PaceKVCache":
        """Build a PaceKVCache from vLLM's KVCacheConfig.

        Each attention layer gets its own `SlabPool` (geometry can vary
        across layers). `block_size` comes from
        `pace::slab_autotune_block_size` (overridable via
        `PACE_VLLM_SLAB_BLOCK_SIZE`); vLLM's `AttentionSpec.block_size`
        is scheduler-accounting only and intentionally not used for
        SlabPool sizing. PaceSlabConnector supplies one shared block size
        because its hash and physical-block mappings span every layer.
        Sizing uses equal `num_blocks` per layer under `budget //
        sum(page_size_bytes_per_layer)`.

        `shared_kv_cache_layers` (follower -> target) excludes follower
        layers from per-pool allocation; followers are aliased to the
        target's pool in `PaceModelRunner.initialize_kv_cache`.
        """
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            MLAAttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        groups = kv_cache_config.kv_cache_groups
        if not groups:
            raise ValueError("pace-vllm: kv_cache_config has no groups")

        # Walk groups → per-layer (name, AttentionSpec) in registration order.
        # Non-attention specs (Mamba, cross-attn) live in groups we cannot
        # service from a SlabPool; skip them so hybrid model+mamba configs
        # don't crash. MLA is a subclass of AttentionSpec but its K/V
        # projection shape and storage layout aren't something SlabPool
        # can service -- raise loudly so the user reaches for a non-MLA
        # backend or disables the plugin instead of getting silently
        # wrong outputs. The MLA check goes *before* the AttentionSpec
        # check in both branches because MLAAttentionSpec is a subclass.
        per_layer: list[tuple[str, "AttentionSpec"]] = []
        for group in groups:
            raw_spec = group.kv_cache_spec
            if isinstance(raw_spec, UniformTypeKVCacheSpecs):
                for name, layer_spec in raw_spec.kv_cache_specs.items():
                    if isinstance(layer_spec, MLAAttentionSpec):
                        raise NotImplementedError(
                            f"pace-vllm: MLA attention is not supported "
                            f"(layer {name!r}). MLA changes K/V projection "
                            "shape and storage layout in ways SlabPool "
                            "cannot service. Use a non-MLA attention "
                            "backend or disable VLLM_PLUGINS=pace for "
                            "this model."
                        )
                    if isinstance(layer_spec, AttentionSpec):
                        per_layer.append((name, layer_spec))
            elif isinstance(raw_spec, MLAAttentionSpec):
                raise NotImplementedError(
                    "pace-vllm: MLA attention is not supported (layers "
                    f"{list(group.layer_names)}). Use a non-MLA "
                    "attention backend or disable VLLM_PLUGINS=pace."
                )
            elif isinstance(raw_spec, AttentionSpec):
                for name in group.layer_names:
                    per_layer.append((name, raw_spec))

        # KV-sharing: drop follower layers from the per-pool list. Only
        # the canonical target gets a SlabPool; followers share the
        # target's pool via the layer registry. Reject chained sharing
        # (A -> B -> C) and unknown targets up-front so misconfigurations
        # fail fast rather than producing silent K/V divergence.
        shared = shared_kv_cache_layers or {}
        known_layer_names = {name for name, _ in per_layer}
        for follower, target in shared.items():
            if target in shared:
                raise ValueError(
                    "pace-vllm: chained KV sharing not supported "
                    f"({follower} -> {target} -> {shared[target]}). "
                    "Each follower must point directly at a canonical target."
                )
            if follower in known_layer_names and target not in known_layer_names:
                raise ValueError(
                    f"pace-vllm: KV-sharing follower {follower!r} points at "
                    f"unknown target {target!r}; target must be an attention "
                    "layer in kv_cache_config."
                )
        follower_set = set(shared.keys())
        per_layer = [(n, s) for (n, s) in per_layer if n not in follower_set]

        if not per_layer:
            raise ValueError(
                "PaceKVCache.from_kv_cache_config: no attention layer found "
                "in kv_cache_config.kv_cache_groups."
            )

        budget_bytes = _pace_kvcache_budget_bytes(vllm_config)
        if budget_bytes is None or budget_bytes <= 0:
            raise RuntimeError(
                "pace-vllm: no CPU KV cache budget available. Set "
                "--kv-cache-memory-bytes or VLLM_CPU_KVCACHE_SPACE."
            )

        connector_block_size = _prefix_connector_block_size(vllm_config)
        env_override = (
            None
            if connector_block_size is not None
            else _resolve_slab_block_size_override()
        )
        block_size_for: dict[tuple[int, int], int] = {}
        for _, s in per_layer:
            key = (int(s.num_kv_heads), int(s.head_size))
            if key in block_size_for:
                continue
            block_size_for[key] = (
                connector_block_size or env_override or _autotune_slab_block_size(*key)
            )

        # Page size MUST be computed against the autotuned block_size,
        # not vLLM's spec.block_size, since the latter is what vLLM uses
        # for scheduler accounting only and is typically way smaller.
        # bf16 -> 2 bytes/element; K + V doubles that.
        page_sizes = [
            2
            * int(s.num_kv_heads)
            * int(s.head_size)
            * block_size_for[(int(s.num_kv_heads), int(s.head_size))]
            * 2
            for _, s in per_layer
        ]
        total_per_block = sum(page_sizes)
        if int(budget_bytes) < int(total_per_block):
            raise ValueError(
                f"pace-vllm: CPU KV cache budget "
                f"({budget_bytes / 1024**3:.2f} GiB) is smaller than one block "
                f"across all attention layers "
                f"({total_per_block / 1024**3:.2f} GiB). Increase the "
                "budget via --kv-cache-memory-bytes or "
                "VLLM_CPU_KVCACHE_SPACE."
            )
        num_blocks = int(budget_bytes) // int(total_per_block)

        per_layer_specs = tuple(
            PaceLayerSpec(
                num_kv_heads=int(s.num_kv_heads),
                head_dim=int(s.head_size),
                block_size=block_size_for[(int(s.num_kv_heads), int(s.head_size))],
                dtype=s.dtype,
                num_blocks=num_blocks,
            )
            for _, s in per_layer
        )
        layer_names = [name for name, _ in per_layer]

        logger.info(
            "pace-vllm: PaceKVCache sized from budget=%.2f GiB, layers=%d, "
            "geometries=%d, num_blocks=%d per layer (total %.2f GiB).",
            budget_bytes / (1024**3),
            len(per_layer),
            len(block_size_for),
            num_blocks,
            num_blocks * total_per_block / (1024**3),
        )
        for (num_kv_heads, head_dim), block_size in sorted(block_size_for.items()):
            logger.info(
                "pace-vllm:   geometry (num_kv_heads=%d, head_dim=%d) "
                "block_size=%d (%s).",
                num_kv_heads,
                head_dim,
                block_size,
                (
                    "prefix connector"
                    if connector_block_size is not None
                    else "env override" if env_override is not None else "autotuned"
                ),
            )

        return cls(
            PaceKVCacheSpec(per_layer=per_layer_specs),
            layer_names=layer_names,
        )

    def layer_idx_of(self, layer_name: str) -> int:
        """Lookup the slab pool index for a given Attention layer name."""
        return self._layer_name_to_idx[layer_name]

    @property
    def layer_names(self) -> list[str]:
        """Return the ordered list of layer names owned by this PaceKVCache.

        Matches the allocation order of the per-layer `SlabPool`
        instances (first name -> pool_idx 0, and so on). Useful for the
        model runner when it needs to register each layer with the
        attention-backend registry.
        """
        return [
            name
            for name, _ in sorted(self._layer_name_to_idx.items(), key=lambda kv: kv[1])
        ]

    def create_sequence(self, request_id: str, max_seq_len: int) -> int:
        """Register a request; allocates slab block bookkeeping per layer."""
        if request_id in self._request_to_seq:
            return self._request_to_seq[request_id]
        seq_id = next(self._seq_id_counter)
        for pool in self._pools:
            pool.create_sequence(seq_id, max_seq_len)
        self._request_to_seq[request_id] = seq_id
        return seq_id

    def remove_sequence(self, request_id: str) -> None:
        seq_id = self._request_to_seq.pop(request_id, None)
        if seq_id is None:
            return
        for pool in self._pools:
            pool.remove_sequence(seq_id)

    def truncate_sequence(self, request_id: str, remove_len: int) -> None:
        """Drop the last `remove_len` tokens from every per-layer pool."""
        seq_id = self._request_to_seq.get(request_id)
        if seq_id is None or remove_len <= 0:
            return
        for pool in self._pools:
            pool.truncate_sequence(seq_id, remove_len)

    def reset_sequence(self, request_id: str) -> None:
        """Drop ALL K/V for a sequence while keeping the slab seq_id alive.

        Used on vLLM preemption: vLLM frees blocks and will re-prefill from
        token 0 on resumption, so the slab-side K/V must go to length 0.
        """
        seq_id = self._request_to_seq.get(request_id)
        if seq_id is None:
            return
        # All pools share the same per-seq length; querying pool 0 is enough.
        current_len = int(self._pools[0].get_sequence_length(seq_id))
        if current_len <= 0:
            return
        for pool in self._pools:
            pool.truncate_sequence(seq_id, current_len)

    def get_sequence_id(self, request_id: str) -> int | None:
        return self._request_to_seq.get(request_id)

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers

    def pool_for_layer(self, layer_idx: int):  # torch.classes.pace.SlabPool
        return self._pools[layer_idx]

    def _require_aligned_prefix_blocks(self) -> None:
        if not self._prefix_block_indices_aligned:
            raise ValueError(
                "pace-vllm: prefix caching requires uniform block_size across "
                "all SlabPool layers."
            )

    def get_block_indices(self, seq_id: int) -> list[int]:
        """Physical slab block indices for a sequence (pool 0 is representative)."""
        self._require_aligned_prefix_blocks()
        return list(self._pools[0].get_block_indices(seq_id))

    def get_sequence_length_by_id(self, seq_id: int) -> int:
        """Tokens written so far for a sequence (pool 0 is representative)."""
        self._require_aligned_prefix_blocks()
        return int(self._pools[0].get_sequence_length(seq_id))

    def reserve_blocks(self, seq_id: int, num_blocks: int) -> None:
        """Allocate blocks up front, without advancing seq_len."""
        self._require_aligned_prefix_blocks()
        for pool in self._pools:
            pool.reserve_blocks(seq_id, num_blocks)

    def assign_prefix_blocks(
        self, seq_id: int, blocks: list[int], num_tokens: int
    ) -> None:
        """Point a sequence at shared cached prefix blocks on every layer."""
        self._require_aligned_prefix_blocks()
        for pool in self._pools:
            pool.assign_prefix_blocks(seq_id, blocks, num_tokens)

    def retain_blocks(self, blocks: list[int]) -> None:
        """Retain blocks (survive remove_sequence) on every layer."""
        self._require_aligned_prefix_blocks()
        for pool in self._pools:
            pool.retain_blocks(blocks)

    def release_retained_blocks(self, blocks: list[int]) -> None:
        """Release retained blocks back to the free list on every layer."""
        self._require_aligned_prefix_blocks()
        for pool in self._pools:
            pool.release_retained_blocks(blocks)

    def num_free_blocks_per_layer(self) -> int:
        # Layers may have different pool sizes now. Report pool 0 for
        # backwards compat with pre-hybrid call sites; add a new helper
        # if a caller actually needs per-layer numbers.
        return min(int(pool.get_free_block_count()) for pool in self._pools)

    def set_owner(self, runner: object | None) -> None:
        self._owner = runner

    def get_owner(self) -> object | None:
        return self._owner
