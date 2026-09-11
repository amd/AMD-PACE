# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PaceModelRunner: vLLM v1 CPU model runner subclass.

Owns the slab-backed KV state. `_allocate_kv_cache_tensors` ships zero-sized
placeholders for attention layers (real K/V lives in `SlabPool`) while vLLM
allocates and reshapes Mamba / linear-attention state through its native path;
`initialize_kv_cache` builds the `PaceKVCache` and populates the attention
backend's layer registry; `_update_states` fans out request lifecycle events
(create / remove / reset on preempt) to the slab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

from pace_vllm.v1.attention.layer_registry import (
    register_layer,
    set_current_kv_cache,
)
from pace_vllm.v1.kv_cache import PaceKVCache

if TYPE_CHECKING:  # pragma: no cover
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger("pace_vllm.v1.worker.cpu_model_runner")


class PaceModelRunner(CPUModelRunner):
    """PACE-aware CPU model runner."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.pace_kv_cache: PaceKVCache | None = None
        self._pace_connector = None
        logger.info("pace-vllm: PaceModelRunner active.")

    def _init_kv_zero_meta(self) -> None:
        """Opt the slab-backed KV cache out of vLLM's ``KVBlockZeroer``.

        Slab attention layers are zero-sized placeholders -- real K/V lives in
        ``SlabPool``, not in vLLM-managed blocks -- and building a
        ``KVBlockZeroer`` over them aborts engine startup while deriving their
        per-block strides.
        """
        return

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """No-op block zeroing for the slab-backed attention layers.

        vLLM zeroes each FullAttentionSpec layer's KV blocks in place, which
        raises ``IndexError`` on our zero-sized placeholders. Zeroing is
        unnecessary anyway: SlabPool tracks per-sequence length and CPU
        attention masks invalid positions with -INF, so stale KV is never read.
        Mamba/linear-attn state is excluded by vLLM's own FullAttentionSpec
        filter, so this override does not affect it.
        """
        return

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)

        self.pace_kv_cache = PaceKVCache.from_kv_cache_config(
            kv_cache_config,
            self.vllm_config,
            shared_kv_cache_layers=self.shared_kv_cache_layers,
        )
        self.pace_kv_cache.set_owner(self)
        set_current_kv_cache(self.pace_kv_cache)

        for name in self.pace_kv_cache.layer_names:
            pool = self.pace_kv_cache.pool_for_layer(
                self.pace_kv_cache.layer_idx_of(name)
            )
            register_layer(name, self.pace_kv_cache, pool)

        # KV-sharing followers alias the target's pool so
        # `lookup_layer(follower)` returns the shared SlabPool. The
        # factory excluded followers from per-pool allocation.
        for follower, target in self.shared_kv_cache_layers.items():
            # Hybrid models may declare KV-sharing among non-attention
            # layers (Mamba/SSM groups) that SlabPool doesn't service;
            # the factory's existing raises cover the cases that should
            # fail loud (attention follower targeting an unknown layer).
            if target not in self.pace_kv_cache.layer_names:
                continue
            target_pool = self.pace_kv_cache.pool_for_layer(
                self.pace_kv_cache.layer_idx_of(target)
            )
            register_layer(follower, self.pace_kv_cache, target_pool)

        kv_transfer_config = self.vllm_config.kv_transfer_config
        expects_pace_connector = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "PaceSlabConnector"
        )
        if expects_pace_connector:
            from pace_vllm.v1.connector import PaceSlabConnector

            if not has_kv_transfer_group():
                raise RuntimeError(
                    "pace-vllm: PaceSlabConnector worker instance is unavailable."
                )
            connector = get_kv_transfer_group()
            if not isinstance(connector, PaceSlabConnector):
                raise RuntimeError(
                    "pace-vllm: worker initialized an unexpected KV connector."
                )
            connector.register_kv_cache(self.pace_kv_cache)
            self._pace_connector = connector

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        attention_layers: set[str] = set()
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()), None)
            if isinstance(spec, AttentionSpec):
                attention_layers.update(group.layer_names)

        raw_tensors: dict[str, torch.Tensor] = {}
        packed_backing: torch.Tensor | None = None
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            needs_storage = any(
                name not in attention_layers for name in kv_cache_tensor.shared_by
            )
            tensor: torch.Tensor | None = None
            if needs_storage:
                if getattr(kv_cache_tensor, "block_stride", 0) > 0:
                    if packed_backing is None:
                        packed_backing = torch.zeros(
                            kv_cache_tensor.size,
                            dtype=torch.int8,
                            device=self.device,
                        )
                    tensor = packed_backing
                else:
                    tensor = torch.zeros(
                        kv_cache_tensor.size,
                        dtype=torch.int8,
                        device=self.device,
                    )
            for layer_name in kv_cache_tensor.shared_by:
                if layer_name in attention_layers:
                    raw_tensors[layer_name] = torch.empty(
                        0, dtype=torch.int8, device=self.device
                    )
                else:
                    assert tensor is not None
                    raw_tensors[layer_name] = tensor

        return raw_tensors

    def _update_states(self, scheduler_output: "SchedulerOutput"):
        if self.pace_kv_cache is not None:
            max_len = self.model_config.max_model_len
            n_new = len(scheduler_output.scheduled_new_reqs)
            n_finished = len(scheduler_output.finished_req_ids)
            preempted = scheduler_output.preempted_req_ids or ()
            n_preempted = len(preempted)
            if n_new or n_finished or n_preempted:
                logger.info(
                    "pace-vllm: slab lifecycle step: +%d new, -%d finished, "
                    "!%d preempted (reset).",
                    n_new,
                    n_finished,
                    n_preempted,
                )

            connector_metadata = self._get_connector_metadata(scheduler_output)
            if self._pace_connector is not None and connector_metadata is not None:
                self._pace_connector.apply_worker_retains(connector_metadata)

            for preempted_rid in preempted:
                self.pace_kv_cache.reset_sequence(preempted_rid)
            for finished_rid in scheduler_output.finished_req_ids:
                self.pace_kv_cache.remove_sequence(finished_rid)

            if self._pace_connector is not None and connector_metadata is not None:
                self._pace_connector.apply_worker_releases(connector_metadata)

            for new_req in scheduler_output.scheduled_new_reqs:
                self.pace_kv_cache.create_sequence(new_req.req_id, max_len)

            if self._pace_connector is not None and connector_metadata is not None:
                self._pace_connector.apply_worker_assignments(connector_metadata)
        return super()._update_states(scheduler_output)

    def _get_connector_metadata(self, scheduler_output: "SchedulerOutput"):
        from pace_vllm.v1.connector import PaceSlabMetadata

        meta = getattr(scheduler_output, "kv_connector_metadata", None)
        if not isinstance(meta, PaceSlabMetadata):
            return None
        return meta
