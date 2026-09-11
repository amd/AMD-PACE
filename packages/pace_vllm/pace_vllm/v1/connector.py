# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Process-safe vLLM connector for PACE slab prefix caching."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

    from pace_vllm.v1.kv_cache import PaceKVCache

logger = init_logger("pace_vllm.v1.connector")


@dataclass
class PaceSlabMetadata(KVConnectorMetadata):
    prefix_assignments: dict[str, tuple[list[int], int]] = field(default_factory=dict)
    retain_plans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    evicted_cache_ids: list[int] = field(default_factory=list)
    reset_worker_cache: bool = False


@dataclass
class PaceSlabWorkerMetadata(KVConnectorWorkerMetadata):
    retained: dict[int, int] = field(default_factory=dict)
    released: dict[int, int] = field(default_factory=dict)
    failed: dict[int, int] = field(default_factory=dict)
    free_blocks: int | None = None

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        if not isinstance(other, PaceSlabWorkerMetadata):
            raise TypeError(
                "PaceSlabWorkerMetadata can only aggregate its own metadata type."
            )
        return PaceSlabWorkerMetadata(
            retained=_sum_counts(self.retained, other.retained),
            released=_sum_counts(self.released, other.released),
            failed=_sum_counts(self.failed, other.failed),
            free_blocks=_min_optional(self.free_blocks, other.free_blocks),
        )


def _sum_counts(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    result = dict(left)
    for cache_id, count in right.items():
        result[cache_id] = result.get(cache_id, 0) + count
    return result


def _min_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _div_up(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _has_mamba_groups(kv_cache_config: "KVCacheConfig") -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    if kv_cache_config is None:
        return False
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            return True
        if isinstance(spec, UniformTypeKVCacheSpecs) and any(
            isinstance(layer_spec, MambaSpec)
            for layer_spec in spec.kv_cache_specs.values()
        ):
            return True
    return False


class PaceSlabConnector(KVConnectorBase_V1, SupportsHMA):
    """Logical scheduler cache backed by worker-local slab blocks."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        if _has_mamba_groups(kv_cache_config):
            raise ValueError(
                "pace-vllm: prefix caching for hybrid Mamba/SSM models requires "
                "coordinated attention and SSM state restoration."
            )
        self._block_size = int(extra.get("slab_block_size", 64))
        self._evict_headroom = int(extra.get("slab_prefix_evict_headroom", 8))
        self._world_size = int(vllm_config.parallel_config.world_size)
        if self._block_size <= 0:
            raise ValueError("pace-vllm: slab block size must be positive.")
        if self._evict_headroom < 0:
            raise ValueError(
                "pace-vllm: slab prefix eviction headroom cannot be negative."
            )
        if self._world_size <= 0:
            raise ValueError("pace-vllm: connector world size must be positive.")

        self._pace_kv_cache: PaceKVCache | None = None
        self._worker_cache_blocks: dict[int, int] = {}
        self._worker_meta = PaceSlabWorkerMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self._next_cache_id = itertools.count()
            self._requests: dict[str, Request] = {}
            self._hash_map: dict[Any, int] = {}
            self._cache_id_to_hash: dict[int, Any] = {}
            self._pending_hash_to_cache_id: dict[Any, int] = {}
            self._pending_cache_hash: dict[int, Any] = {}
            self._retain_ack_counts: dict[int, int] = {}
            self._aborted_cache_hash: dict[int, Any] = {}
            self._release_ack_counts: dict[int, int] = {}
            self._blocked_hashes: set[Any] = set()
            self._cache_owner: dict[int, str] = {}
            self._aborted_cache_owner: dict[int, str] = {}
            self._deferred_abort_evictions: set[int] = set()
            self._aborted_release_credits: set[int] = set()
            self._ref_count: dict[int, int] = {}
            self._request_refs: dict[str, list[int]] = {}
            self._lru: OrderedDict[int, None] = OrderedDict()
            self._planned_positions: dict[str, set[int]] = {}
            self._pending_assignments: dict[str, tuple[list[int], int]] = {}
            self._request_assigned_cache_ids: dict[str, set[int]] = {}
            self._request_sequence_blocks: dict[str, int] = {}
            self._pending_cleanup_blocks = 0
            self._pending_retain_plans: dict[str, list[tuple[int, int]]] = {}
            self._pending_evictions: list[int] = []
            self._min_free_blocks: int | None = None
            self._reset_worker_cache = False
            self._total_lookups = 0
            self._total_hits = 0
            self._total_blocks_reused = 0
            self._total_inserts = 0
            self._total_evictions = 0

    def register_kv_cache(self, kv_cache: "PaceKVCache") -> None:
        if self.role != KVConnectorRole.WORKER:
            raise RuntimeError(
                "PaceKVCache can only be registered on worker connector."
            )
        self._pace_kv_cache = kv_cache

    def on_new_request(self, request: "Request") -> None:
        if self.role == KVConnectorRole.SCHEDULER:
            self._requests[request.request_id] = request
            self._request_assigned_cache_ids.setdefault(request.request_id, set())
            self._request_sequence_blocks.setdefault(request.request_id, 0)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if (
            self.role != KVConnectorRole.SCHEDULER
            or request.skip_reading_prefix_cache
            or not request.block_hashes
        ):
            return 0, False

        skipped = num_computed_tokens // self._block_size
        matched_blocks = 0
        for block_hash in request.block_hashes[skipped:]:
            if block_hash not in self._hash_map:
                break
            matched_blocks += 1

        if matched_blocks == 0:
            return 0, False

        max_reusable_blocks = (
            max(
                0,
                request.num_tokens - num_computed_tokens - 1,
            )
            // self._block_size
        )
        matched_tokens = min(matched_blocks, max_reusable_blocks) * self._block_size
        return matched_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.role != KVConnectorRole.SCHEDULER or num_external_tokens <= 0:
            return

        skipped = request.num_computed_tokens // self._block_size
        num_cache_ids = _div_up(num_external_tokens, self._block_size)
        cache_ids: list[int] = []
        for block_hash in request.block_hashes[skipped : skipped + num_cache_ids]:
            cache_id = self._hash_map.get(block_hash)
            if cache_id is None:
                break
            cache_ids.append(cache_id)

        if len(cache_ids) != num_cache_ids:
            raise RuntimeError(
                "pace-vllm: prefix cache entry disappeared during allocation."
            )

        for cache_id in cache_ids:
            self._add_ref(request.request_id, cache_id)
        self._request_assigned_cache_ids.setdefault(request.request_id, set()).update(
            cache_ids
        )

        self._pending_assignments[request.request_id] = (
            cache_ids,
            request.num_computed_tokens + num_external_tokens,
        )
        self._total_lookups += 1
        self._total_hits += 1
        self._total_blocks_reused += len(cache_ids)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.role != KVConnectorRole.SCHEDULER:
            return PaceSlabMetadata()

        if self._reset_worker_cache:
            self._record_scheduled_block_counts(scheduler_output)
            metadata = PaceSlabMetadata(reset_worker_cache=True)
            self._reset_worker_cache = False
            self._pending_cleanup_blocks = 0
            return metadata

        for request in self._requests.values():
            self._plan_retain_blocks(request, finishing=False)
        self._queue_safe_abort_evictions()
        cleanup_blocks = self._pending_cleanup_blocks + sum(
            cache_id in self._aborted_release_credits
            and cache_id not in self._aborted_cache_owner
            for cache_id in self._pending_evictions
        )
        for req_id in scheduler_output.preempted_req_ids or ():
            cleanup_blocks += self._take_cleanup_credit(req_id)
        demand = self._scheduled_block_demand(scheduler_output)
        evicted = self._evict_for_target(
            max(0, demand + self._evict_headroom - cleanup_blocks)
        )
        free_blocks = self._min_free_blocks
        effective_free_blocks = (
            None if free_blocks is None else free_blocks + cleanup_blocks
        )
        if effective_free_blocks is not None and effective_free_blocks < demand:
            raise RuntimeError(
                "pace-vllm: insufficient slab blocks for scheduled batch "
                f"(required={demand}, free={effective_free_blocks}, "
                f"evicted={evicted}, cleanup={cleanup_blocks})."
            )

        metadata = PaceSlabMetadata(
            prefix_assignments=dict(self._pending_assignments),
            retain_plans={
                req_id: list(plans)
                for req_id, plans in self._pending_retain_plans.items()
            },
            evicted_cache_ids=list(self._pending_evictions),
            reset_worker_cache=self._reset_worker_cache,
        )
        self._record_scheduled_block_counts(scheduler_output)
        self._pending_assignments.clear()
        self._pending_retain_plans.clear()
        self._pending_evictions.clear()
        self._pending_cleanup_blocks = 0
        self._reset_worker_cache = False
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        if self.role != KVConnectorRole.SCHEDULER:
            return
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        if not isinstance(metadata, PaceSlabWorkerMetadata):
            raise TypeError("pace-vllm: scheduler received unexpected worker metadata.")

        if metadata.free_blocks is not None and metadata.free_blocks < 0:
            raise ValueError("pace-vllm: worker reported negative free block count.")
        self._min_free_blocks = metadata.free_blocks
        for cache_id, count in metadata.failed.items():
            self._validate_worker_count(cache_id, count)
            if (
                cache_id in self._pending_cache_hash
                or cache_id in self._cache_id_to_hash
            ):
                retained_count = self._retain_ack_counts.get(cache_id, 0)
                retained_count += metadata.retained.get(cache_id, 0)
                self._abort_cache_id(
                    cache_id,
                    retained_on_worker=retained_count > 0,
                )

        for cache_id, count in metadata.released.items():
            self._validate_worker_count(cache_id, count)
            if cache_id not in self._aborted_cache_hash:
                continue
            total = self._release_ack_counts.get(cache_id, 0) + count
            if total > self._world_size:
                raise ValueError(
                    f"pace-vllm: cache ID {cache_id} received {total} release "
                    f"acknowledgements for {self._world_size} workers."
                )
            self._release_ack_counts[cache_id] = total
            if total >= self._world_size:
                block_hash = self._aborted_cache_hash.pop(cache_id)
                self._release_ack_counts.pop(cache_id, None)
                self._aborted_cache_owner.pop(cache_id, None)
                self._aborted_release_credits.discard(cache_id)
                self._blocked_hashes.discard(block_hash)

        for cache_id, count in metadata.retained.items():
            self._validate_worker_count(cache_id, count)
            if cache_id not in self._pending_cache_hash:
                continue
            total = self._retain_ack_counts.get(cache_id, 0) + count
            if total > self._world_size:
                raise ValueError(
                    f"pace-vllm: cache ID {cache_id} received {total} retention "
                    f"acknowledgements for {self._world_size} workers."
                )
            self._retain_ack_counts[cache_id] = total
            if total >= self._world_size:
                self._publish_cache_id(cache_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.role != KVConnectorRole.SCHEDULER:
            return False, None
        self._plan_retain_blocks(request, finishing=True)
        self._pending_cleanup_blocks += self._take_cleanup_credit(
            request.request_id,
            fallback_tokens=request.num_computed_tokens,
        )
        self._release_request_refs(request.request_id)
        self._queue_safe_abort_evictions(finishing_req_id=request.request_id)
        self._requests.pop(request.request_id, None)
        self._planned_positions.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids=[])

    def has_pending_push_work(self) -> bool:
        if self.role != KVConnectorRole.SCHEDULER:
            return False
        return bool(
            self._pending_cache_hash
            or self._pending_retain_plans
            or self._pending_evictions
            or self._aborted_cache_hash
            or self._reset_worker_cache
        )

    def reset_cache(self) -> bool | None:
        if self.role != KVConnectorRole.SCHEDULER:
            return None
        from vllm.v1.request import RequestStatus

        active_requests = {
            req_id
            for req_id in self._request_refs
            if (
                (request := self._requests.get(req_id)) is None
                or request.status != RequestStatus.PREEMPTED
            )
        }
        active_requests.update(
            owner
            for owner in self._aborted_cache_owner.values()
            if (
                (request := self._requests.get(owner)) is not None
                and request.status != RequestStatus.PREEMPTED
            )
        )
        if active_requests:
            raise RuntimeError(
                "pace-vllm: prefix cache reset requires active requests to be "
                f"preempted first: {sorted(active_requests)}."
            )
        for req_id in list(self._request_refs):
            self._release_request_refs(req_id)

        self._reset_worker_cache = True
        self._hash_map.clear()
        self._cache_id_to_hash.clear()
        self._pending_hash_to_cache_id.clear()
        self._pending_cache_hash.clear()
        self._retain_ack_counts.clear()
        self._aborted_cache_hash.clear()
        self._release_ack_counts.clear()
        self._blocked_hashes.clear()
        self._cache_owner.clear()
        self._aborted_cache_owner.clear()
        self._deferred_abort_evictions.clear()
        self._aborted_release_credits.clear()
        self._ref_count.clear()
        self._request_refs.clear()
        self._lru.clear()
        self._planned_positions.clear()
        self._pending_assignments.clear()
        self._request_assigned_cache_ids.clear()
        self._request_sequence_blocks.clear()
        self._pending_cleanup_blocks = 0
        self._pending_retain_plans.clear()
        self._pending_evictions.clear()
        return True

    def apply_worker_releases(self, metadata: PaceSlabMetadata) -> None:
        cache = self._require_worker_cache()
        if metadata.reset_worker_cache:
            blocks = list(self._worker_cache_blocks.values())
            if blocks:
                cache.release_retained_blocks(blocks)
            self._worker_cache_blocks.clear()

        release_blocks: list[int] = []
        for cache_id in metadata.evicted_cache_ids:
            block = self._worker_cache_blocks.pop(cache_id, None)
            if block is not None:
                release_blocks.append(block)
            self._worker_meta.released[cache_id] = 1
        if release_blocks:
            cache.release_retained_blocks(release_blocks)

    def apply_worker_retains(self, metadata: PaceSlabMetadata) -> None:
        cache = self._require_worker_cache()
        blocks_to_retain: list[int] = []
        pending_mappings: list[tuple[int, int]] = []

        for req_id, plans in metadata.retain_plans.items():
            seq_id = cache.get_sequence_id(req_id)
            if seq_id is None:
                for _, cache_id in plans:
                    self._worker_meta.failed[cache_id] = 1
                continue
            physical_blocks = cache.get_block_indices(seq_id)
            for block_position, cache_id in plans:
                if cache_id < 0:
                    raise ValueError(
                        "pace-vllm: logical cache IDs must be nonnegative."
                    )
                if cache_id in self._worker_cache_blocks:
                    self._worker_meta.retained[cache_id] = 1
                    continue
                if block_position < 0 or block_position >= len(physical_blocks):
                    self._worker_meta.failed[cache_id] = 1
                    continue
                physical_block = physical_blocks[block_position]
                blocks_to_retain.append(physical_block)
                pending_mappings.append((cache_id, physical_block))

        if blocks_to_retain:
            cache.retain_blocks(blocks_to_retain)
        for cache_id, physical_block in pending_mappings:
            self._worker_cache_blocks[cache_id] = physical_block
            self._worker_meta.retained[cache_id] = 1

    def apply_worker_assignments(self, metadata: PaceSlabMetadata) -> None:
        cache = self._require_worker_cache()
        for req_id, (cache_ids, num_tokens) in metadata.prefix_assignments.items():
            if not cache_ids or num_tokens <= 0:
                raise ValueError("pace-vllm: invalid empty prefix assignment.")
            seq_id = cache.get_sequence_id(req_id)
            if seq_id is None:
                raise RuntimeError(
                    f"pace-vllm: prefix assignment request {req_id!r} is missing."
                )
            try:
                physical_blocks = [
                    self._worker_cache_blocks[cache_id] for cache_id in cache_ids
                ]
            except KeyError as exc:
                raise RuntimeError(
                    f"pace-vllm: worker is missing logical cache ID {exc.args[0]}."
                ) from exc
            cache.assign_prefix_blocks(seq_id, physical_blocks, num_tokens)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if self.role != KVConnectorRole.WORKER or self._pace_kv_cache is None:
            return None
        metadata = self._worker_meta
        metadata.free_blocks = self._pace_kv_cache.num_free_blocks_per_layer()
        self._worker_meta = PaceSlabWorkerMetadata()
        return metadata

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def take_events(self) -> Iterable:
        return ()

    def _plan_retain_blocks(self, request: "Request", finishing: bool) -> None:
        num_full_blocks = min(
            len(request.block_hashes),
            request.num_computed_tokens // self._block_size,
        )
        planned = self._planned_positions.setdefault(request.request_id, set())
        plans = self._pending_retain_plans.setdefault(request.request_id, [])

        for block_position in range(num_full_blocks):
            if block_position in planned:
                continue
            block_hash = request.block_hashes[block_position]
            if block_hash in self._blocked_hashes:
                continue
            planned.add(block_position)
            cache_id = self._hash_map.get(block_hash)
            if cache_id is not None:
                if not finishing:
                    self._add_ref(request.request_id, cache_id)
                continue
            cache_id = self._pending_hash_to_cache_id.get(block_hash)
            if cache_id is not None:
                if not finishing:
                    self._add_ref(request.request_id, cache_id)
                continue

            cache_id = next(self._next_cache_id)
            self._pending_hash_to_cache_id[block_hash] = cache_id
            self._pending_cache_hash[cache_id] = block_hash
            self._cache_owner[cache_id] = request.request_id
            plans.append((block_position, cache_id))
            if not finishing:
                self._add_ref(request.request_id, cache_id)

        if not plans:
            self._pending_retain_plans.pop(request.request_id, None)

    def _publish_cache_id(self, cache_id: int) -> None:
        block_hash = self._pending_cache_hash.pop(cache_id)
        self._pending_hash_to_cache_id.pop(block_hash, None)
        self._retain_ack_counts.pop(cache_id, None)
        self._hash_map[block_hash] = cache_id
        self._cache_id_to_hash[cache_id] = block_hash
        if self._ref_count.get(cache_id, 0) == 0:
            self._lru[cache_id] = None
        self._total_inserts += 1

    def _abort_cache_id(
        self,
        cache_id: int,
        retained_on_worker: bool = False,
    ) -> None:
        owner = self._cache_owner.pop(cache_id, None)
        block_hash = self._pending_cache_hash.pop(cache_id, None)
        if block_hash is None:
            block_hash = self._cache_id_to_hash.pop(cache_id, None)
        if block_hash is not None:
            self._pending_hash_to_cache_id.pop(block_hash, None)
            if self._hash_map.get(block_hash) == cache_id:
                self._hash_map.pop(block_hash, None)
            self._aborted_cache_hash[cache_id] = block_hash
            self._blocked_hashes.add(block_hash)
            if owner is not None:
                self._aborted_cache_owner[cache_id] = owner
            if retained_on_worker:
                self._aborted_release_credits.add(cache_id)
            for req_id, planned in self._planned_positions.items():
                request = self._requests.get(req_id)
                if request is None:
                    continue
                planned.difference_update(
                    position
                    for position in tuple(planned)
                    if position < len(request.block_hashes)
                    and request.block_hashes[position] == block_hash
                )
        self._retain_ack_counts.pop(cache_id, None)
        self._lru.pop(cache_id, None)
        self._ref_count.pop(cache_id, None)
        for req_id, refs in list(self._request_refs.items()):
            self._request_refs[req_id] = [
                existing for existing in refs if existing != cache_id
            ]
            if not self._request_refs[req_id]:
                self._request_refs.pop(req_id)
        if self._abort_can_release(cache_id):
            self._pending_evictions.append(cache_id)
        else:
            self._deferred_abort_evictions.add(cache_id)

    def _add_ref(self, req_id: str, cache_id: int) -> None:
        refs = self._request_refs.setdefault(req_id, [])
        if cache_id in refs:
            return
        refs.append(cache_id)
        self._ref_count[cache_id] = self._ref_count.get(cache_id, 0) + 1
        self._lru.pop(cache_id, None)

    def _release_request_refs(self, req_id: str) -> None:
        for cache_id in self._request_refs.pop(req_id, ()):
            remaining = self._ref_count.get(cache_id, 0) - 1
            if remaining <= 0:
                self._ref_count.pop(cache_id, None)
                if cache_id in self._cache_id_to_hash:
                    self._lru[cache_id] = None
            else:
                self._ref_count[cache_id] = remaining

    def _scheduled_block_demand(self, scheduler_output: SchedulerOutput) -> int:
        demand = 0
        for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self._requests.get(req_id)
            if request is None:
                continue
            assignment = self._pending_assignments.get(req_id)
            before = request.num_computed_tokens
            if assignment is not None:
                before = max(before, assignment[1])
            demand += max(
                0,
                _div_up(before + scheduled_tokens, self._block_size)
                - _div_up(before, self._block_size),
            )
        return demand

    def _take_cleanup_credit(
        self,
        req_id: str,
        fallback_tokens: int = 0,
    ) -> int:
        total_blocks = max(
            self._request_sequence_blocks.pop(req_id, 0),
            _div_up(fallback_tokens, self._block_size),
        )
        retained_cache_ids = self._request_assigned_cache_ids.pop(req_id, set())
        retained_cache_ids.update(
            cache_id for cache_id, owner in self._cache_owner.items() if owner == req_id
        )
        for cache_id in tuple(self._cache_owner):
            if self._cache_owner[cache_id] == req_id:
                self._cache_owner.pop(cache_id)
        return max(0, total_blocks - len(retained_cache_ids))

    def _record_scheduled_block_counts(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self._requests.get(req_id)
            if request is None:
                continue
            assignment = self._pending_assignments.get(req_id)
            before = request.num_computed_tokens
            if assignment is not None:
                before = max(before, assignment[1])
            self._request_sequence_blocks[req_id] = _div_up(
                before + scheduled_tokens,
                self._block_size,
            )

    def _evict_for_target(self, target_free_blocks: int) -> int:
        if self._min_free_blocks is None:
            return 0
        deficit = target_free_blocks - self._min_free_blocks
        evicted = 0
        while deficit > 0 and self._lru:
            cache_id, _ = self._lru.popitem(last=False)
            block_hash = self._cache_id_to_hash.pop(cache_id)
            self._hash_map.pop(block_hash, None)
            self._cache_owner.pop(cache_id, None)
            self._ref_count.pop(cache_id, None)
            self._pending_evictions.append(cache_id)
            self._total_evictions += 1
            deficit -= 1
            self._min_free_blocks += 1
            evicted += 1
        return evicted

    def _abort_can_release(
        self,
        cache_id: int,
        finishing_req_id: str | None = None,
    ) -> bool:
        owner = self._aborted_cache_owner.get(cache_id)
        if owner is None or owner == finishing_req_id:
            return True
        request = self._requests.get(owner)
        if request is None:
            return True
        from vllm.v1.request import RequestStatus

        return request.status == RequestStatus.PREEMPTED

    def _queue_safe_abort_evictions(
        self,
        finishing_req_id: str | None = None,
    ) -> None:
        for cache_id in tuple(self._deferred_abort_evictions):
            if not self._abort_can_release(cache_id, finishing_req_id):
                continue
            self._deferred_abort_evictions.remove(cache_id)
            self._pending_evictions.append(cache_id)

    def _require_worker_cache(self) -> "PaceKVCache":
        if self.role != KVConnectorRole.WORKER or self._pace_kv_cache is None:
            raise RuntimeError("pace-vllm: worker connector has no PaceKVCache.")
        return self._pace_kv_cache

    def _validate_worker_count(self, cache_id: int, count: int) -> None:
        if cache_id < 0 or count <= 0 or count > self._world_size:
            raise ValueError(
                f"pace-vllm: invalid worker acknowledgement cache_id={cache_id}, "
                f"count={count}, world_size={self._world_size}."
            )
