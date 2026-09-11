# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from pace_vllm.v1.connector import (
    PaceSlabConnector,
    PaceSlabMetadata,
    PaceSlabWorkerMetadata,
)


def _connector(
    role: KVConnectorRole = KVConnectorRole.SCHEDULER,
    *,
    world_size: int = 1,
    block_size: int = 16,
    evict_headroom: int = 8,
) -> PaceSlabConnector:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=world_size),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "slab_block_size": block_size,
                "slab_prefix_evict_headroom": evict_headroom,
            }
        ),
    )
    return PaceSlabConnector(config, role, kv_cache_config=None)


def _request(
    request_id: str,
    block_hashes: list[bytes],
    *,
    computed_tokens: int = 0,
    num_tokens: int = 64,
    skip_reading_prefix_cache: bool = False,
):
    return SimpleNamespace(
        request_id=request_id,
        block_hashes=block_hashes,
        num_computed_tokens=computed_tokens,
        num_tokens=num_tokens,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
        status=RequestStatus.RUNNING,
    )


def _scheduler_output(**num_scheduled_tokens: int):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )


class _FakeCache:
    def __init__(
        self,
        request_blocks: dict[str, list[int]],
        *,
        free_blocks: int = 32,
    ) -> None:
        self._seq_by_request = {
            request_id: index + 1 for index, request_id in enumerate(request_blocks)
        }
        self._blocks_by_seq = {
            self._seq_by_request[request_id]: list(blocks)
            for request_id, blocks in request_blocks.items()
        }
        self.free_blocks = free_blocks
        self.retained: list[int] = []
        self.released: list[int] = []
        self.assignments: list[tuple[int, list[int], int]] = []

    def get_sequence_id(self, request_id: str) -> int | None:
        return self._seq_by_request.get(request_id)

    def get_block_indices(self, seq_id: int) -> list[int]:
        return list(self._blocks_by_seq[seq_id])

    def retain_blocks(self, blocks: list[int]) -> None:
        self.retained.extend(blocks)

    def release_retained_blocks(self, blocks: list[int]) -> None:
        self.released.extend(blocks)
        self.free_blocks += len(blocks)

    def assign_prefix_blocks(
        self,
        seq_id: int,
        blocks: list[int],
        num_tokens: int,
    ) -> None:
        self.assignments.append((seq_id, list(blocks), num_tokens))

    def num_free_blocks_per_layer(self) -> int:
        return self.free_blocks


class TestPaceSlabConnectorLookup(unittest.TestCase):
    def test_connector_preserves_native_hybrid_cache_management(self) -> None:
        self.assertIsInstance(_connector(), SupportsHMA)

    def test_lookup_is_side_effect_free(self) -> None:
        connector = _connector()
        connector._hash_map = {b"h0": 10, b"h1": 11}
        connector._lru = OrderedDict(((10, None), (11, None)))
        request = _request("request", [b"h0", b"h1"], num_tokens=33)
        before_lru = OrderedDict(connector._lru)

        first = connector.get_num_new_matched_tokens(request, 0)
        second = connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(first, (32, False))
        self.assertEqual(second, first)
        self.assertEqual(connector._lru, before_lru)
        self.assertEqual(connector._pending_assignments, {})
        self.assertEqual(connector._total_lookups, 0)

    def test_full_prompt_hit_reuses_only_complete_blocks_before_final_token(
        self,
    ) -> None:
        connector = _connector()
        connector._hash_map = {b"h0": 10, b"h1": 11}

        self.assertEqual(
            connector.get_num_new_matched_tokens(
                _request("one-block", [b"h0"], num_tokens=16),
                0,
            ),
            (0, False),
        )
        self.assertEqual(
            connector.get_num_new_matched_tokens(
                _request("two-block", [b"h0", b"h1"], num_tokens=32),
                0,
            ),
            (16, False),
        )

    def test_skip_reading_prefix_cache_returns_miss(self) -> None:
        connector = _connector()
        connector._hash_map = {b"h0": 10}
        request = _request(
            "request",
            [b"h0"],
            skip_reading_prefix_cache=True,
        )

        self.assertEqual(
            connector.get_num_new_matched_tokens(request, 0),
            (0, False),
        )

    def test_partial_hit_pins_only_accepted_cache_ids(self) -> None:
        connector = _connector()
        connector._hash_map = {b"h1": 11, b"h2": 12}
        connector._cache_id_to_hash = {11: b"h1", 12: b"h2"}
        connector._lru = OrderedDict(((11, None), (12, None)))
        request = _request(
            "request",
            [b"h0", b"h1", b"h2"],
            computed_tokens=16,
            num_tokens=49,
        )

        connector.update_state_after_alloc(
            request,
            blocks=None,
            num_external_tokens=32,
        )

        self.assertEqual(
            connector._pending_assignments["request"],
            ([11, 12], 48),
        )
        self.assertEqual(connector._request_refs["request"], [11, 12])
        self.assertEqual(connector._lru, OrderedDict())


class TestPaceSlabConnectorPublication(unittest.TestCase):
    def test_metadata_aggregates_through_vllm_worker_output_path(self) -> None:
        outputs = [
            SimpleNamespace(
                kv_connector_output=KVConnectorOutput(
                    kv_connector_worker_meta=PaceSlabWorkerMetadata(
                        retained={10: 1},
                        free_blocks=8,
                    )
                )
            ),
            SimpleNamespace(
                kv_connector_output=KVConnectorOutput(
                    kv_connector_worker_meta=PaceSlabWorkerMetadata(
                        retained={10: 1, 11: 1},
                        failed={12: 1},
                        free_blocks=5,
                    )
                )
            ),
        ]

        result = KVOutputAggregator(2).aggregate(outputs)
        metadata = result.kv_connector_output.kv_connector_worker_meta

        self.assertEqual(metadata.retained, {10: 2, 11: 1})
        self.assertEqual(metadata.failed, {12: 1})
        self.assertEqual(metadata.free_blocks, 5)

    def test_entry_is_unavailable_until_every_worker_acknowledges(self) -> None:
        connector = _connector(world_size=2, block_size=4, evict_headroom=0)
        producer = _request(
            "producer",
            [b"prefix"],
            computed_tokens=4,
            num_tokens=8,
        )
        connector.on_new_request(producer)
        command = connector.build_connector_meta(_scheduler_output(producer=1))
        cache_id = command.retain_plans["producer"][0][1]
        consumer = _request("consumer", [b"prefix"], num_tokens=8)

        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PaceSlabWorkerMetadata(
                    retained={cache_id: 1},
                    free_blocks=8,
                )
            )
        )
        self.assertEqual(connector.get_num_new_matched_tokens(consumer, 0), (0, False))

        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PaceSlabWorkerMetadata(
                    retained={cache_id: 1},
                    free_blocks=8,
                )
            )
        )
        self.assertEqual(connector.get_num_new_matched_tokens(consumer, 0), (4, False))

    def test_only_newly_completed_full_blocks_are_published(self) -> None:
        connector = _connector(block_size=16, evict_headroom=0)
        producer = _request(
            "producer",
            [b"h0", b"h1", b"h2"],
            computed_tokens=15,
        )
        connector.on_new_request(producer)

        first = connector.build_connector_meta(_scheduler_output())
        self.assertEqual(first.retain_plans, {})

        producer.num_computed_tokens = 20
        second = connector.build_connector_meta(_scheduler_output())
        first_cache_id = second.retain_plans["producer"][0][1]
        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PaceSlabWorkerMetadata(
                    retained={first_cache_id: 1},
                    free_blocks=8,
                )
            )
        )
        self.assertEqual(connector._hash_map, {b"h0": first_cache_id})

        producer.num_computed_tokens = 32
        third = connector.build_connector_meta(_scheduler_output())
        second_cache_id = third.retain_plans["producer"][0][1]
        self.assertEqual(third.retain_plans["producer"], [(1, second_cache_id)])
        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PaceSlabWorkerMetadata(
                    retained={second_cache_id: 1},
                    free_blocks=7,
                )
            )
        )
        self.assertEqual(
            connector._hash_map,
            {b"h0": first_cache_id, b"h1": second_cache_id},
        )

    def test_active_duplicate_pins_canonical_entry_until_finish(self) -> None:
        connector = _connector(block_size=16)
        connector._hash_map = {b"h0": 99}
        connector._cache_id_to_hash = {99: b"h0"}
        connector._lru[99] = None
        producer = _request("producer", [b"h0"], computed_tokens=16)
        connector.on_new_request(producer)

        metadata = connector.build_connector_meta(_scheduler_output())

        self.assertEqual(metadata.retain_plans, {})
        self.assertEqual(connector._request_refs["producer"], [99])
        self.assertNotIn(99, connector._lru)
        connector.request_finished(producer, block_ids=[])
        self.assertIn(99, connector._lru)

    def test_failed_retain_waits_for_owner_preemption_before_release_retry(
        self,
    ) -> None:
        connector = _connector(world_size=2, block_size=4, evict_headroom=0)
        producer = _request(
            "producer",
            [b"prefix"],
            computed_tokens=4,
            num_tokens=8,
        )
        connector.on_new_request(producer)
        command = connector.build_connector_meta(_scheduler_output())
        cache_id = command.retain_plans["producer"][0][1]

        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PaceSlabWorkerMetadata(
                    retained={cache_id: 1},
                    failed={cache_id: 1},
                    free_blocks=8,
                )
            )
        )
        blocked_command = connector.build_connector_meta(_scheduler_output())

        self.assertEqual(blocked_command.evicted_cache_ids, [])
        self.assertEqual(blocked_command.retain_plans, {})
        self.assertNotIn(cache_id, connector._pending_cache_hash)
        self.assertNotIn(b"prefix", connector._hash_map)

        producer.status = RequestStatus.PREEMPTED
        release_command = connector.build_connector_meta(
            SimpleNamespace(
                num_scheduled_tokens={},
                preempted_req_ids={"producer"},
                finished_req_ids=set(),
            )
        )
        self.assertEqual(release_command.evicted_cache_ids, [cache_id])

        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=PaceSlabWorkerMetadata(
                    released={cache_id: 2},
                    free_blocks=8,
                )
            )
        )
        producer.status = RequestStatus.RUNNING
        producer.num_computed_tokens = 4
        retry_command = connector.build_connector_meta(_scheduler_output())
        retried_cache_id = retry_command.retain_plans["producer"][0][1]
        self.assertNotEqual(retried_cache_id, cache_id)

    def test_rejects_impossible_worker_acknowledgement_count(self) -> None:
        connector = _connector(world_size=2)

        with self.assertRaisesRegex(ValueError, "invalid worker acknowledgement"):
            connector.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=PaceSlabWorkerMetadata(
                        retained={10: 3},
                        free_blocks=8,
                    )
                )
            )

    def test_rejects_negative_worker_free_count(self) -> None:
        connector = _connector()

        with self.assertRaisesRegex(ValueError, "negative free block count"):
            connector.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=PaceSlabWorkerMetadata(free_blocks=-1)
                )
            )


class TestPaceSlabConnectorCapacity(unittest.TestCase):
    def test_evicts_for_scheduled_demand_not_only_fixed_headroom(self) -> None:
        connector = _connector(evict_headroom=8)
        connector._min_free_blocks = 8
        for index in range(20):
            block_hash = f"h{index}".encode()
            connector._hash_map[block_hash] = index
            connector._cache_id_to_hash[index] = block_hash
            connector._lru[index] = None
        connector.on_new_request(_request("request", []))

        metadata = connector.build_connector_meta(_scheduler_output(request=9 * 16))

        self.assertEqual(len(metadata.evicted_cache_ids), 9)
        self.assertEqual(connector._min_free_blocks, 17)

    def test_insufficient_capacity_fails_before_forward(self) -> None:
        connector = _connector(evict_headroom=0)
        connector._min_free_blocks = 2
        connector.on_new_request(_request("request", []))

        with self.assertRaisesRegex(RuntimeError, "insufficient slab blocks"):
            connector.build_connector_meta(_scheduler_output(request=3 * 16))

    def test_finished_sequence_cleanup_satisfies_same_step_demand(self) -> None:
        connector = _connector(evict_headroom=0)
        connector._min_free_blocks = 0
        finished = _request("finished", [], computed_tokens=16)
        connector.on_new_request(finished)
        connector._request_sequence_blocks["finished"] = 1
        connector.request_finished(finished, block_ids=[])
        connector.on_new_request(_request("next", []))

        metadata = connector.build_connector_meta(_scheduler_output(next=16))

        self.assertEqual(metadata.evicted_cache_ids, [])

    def test_preempted_sequence_cleanup_satisfies_same_step_demand(self) -> None:
        connector = _connector(evict_headroom=0)
        connector._min_free_blocks = 0
        preempted = _request("preempted", [], computed_tokens=0)
        preempted.status = RequestStatus.PREEMPTED
        connector.on_new_request(preempted)
        connector._request_sequence_blocks["preempted"] = 1
        connector.on_new_request(_request("next", []))
        scheduler_output = _scheduler_output(next=16)
        scheduler_output.preempted_req_ids = {"preempted"}

        metadata = connector.build_connector_meta(scheduler_output)

        self.assertEqual(metadata.evicted_cache_ids, [])


class TestPaceSlabConnectorReset(unittest.TestCase):
    def test_reset_clears_assignments_and_defers_worker_release(self) -> None:
        connector = _connector()
        connector._pending_assignments["request"] = ([10], 16)
        connector._hash_map = {b"h0": 10, b"h1": 11}
        connector._cache_id_to_hash = {10: b"h0", 11: b"h1"}

        self.assertTrue(connector.reset_cache())
        metadata = connector.build_connector_meta(_scheduler_output())

        self.assertEqual(metadata.prefix_assignments, {})
        self.assertTrue(metadata.reset_worker_cache)
        self.assertEqual(connector._hash_map, {})

    def test_reset_requires_active_requests_to_be_preempted(self) -> None:
        connector = _connector()
        request = _request("request", [b"h0"])
        connector.on_new_request(request)
        connector._hash_map = {b"h0": 10}
        connector._cache_id_to_hash = {10: b"h0"}
        connector._add_ref("request", 10)

        with self.assertRaisesRegex(RuntimeError, "preempted first"):
            connector.reset_cache()

        self.assertEqual(connector._hash_map, {b"h0": 10})
        self.assertEqual(connector._request_refs, {"request": [10]})

    def test_reset_requires_aborted_cache_owners_to_be_preempted(self) -> None:
        connector = _connector()
        request = _request("request", [b"h0"])
        connector.on_new_request(request)
        connector._aborted_cache_owner[10] = "request"

        with self.assertRaisesRegex(RuntimeError, "preempted first"):
            connector.reset_cache()

        self.assertEqual(connector._aborted_cache_owner, {10: "request"})

    def test_preempted_request_reset_releases_refs(self) -> None:
        connector = _connector()
        request = _request("request", [b"h0"])
        request.status = RequestStatus.PREEMPTED
        connector.on_new_request(request)
        connector._hash_map = {b"h0": 10}
        connector._cache_id_to_hash = {10: b"h0"}
        connector._add_ref("request", 10)

        self.assertTrue(connector.reset_cache())

        self.assertEqual(connector._request_refs, {})

    def test_repeated_reset_preserves_pending_worker_reset(self) -> None:
        connector = _connector()

        self.assertTrue(connector.reset_cache())
        self.assertTrue(connector.reset_cache())
        metadata = connector.build_connector_meta(_scheduler_output())

        self.assertTrue(metadata.reset_worker_cache)

    def test_reset_step_records_scheduled_blocks_for_later_cleanup(self) -> None:
        connector = _connector()
        request = _request("request", [], computed_tokens=0)
        request.status = RequestStatus.PREEMPTED
        connector.on_new_request(request)
        connector._reset_worker_cache = True

        connector.build_connector_meta(_scheduler_output(request=16))

        self.assertEqual(connector._request_sequence_blocks["request"], 1)


class TestPaceSlabConnectorWorkers(unittest.TestCase):
    def test_logical_id_resolves_to_different_worker_physical_blocks(self) -> None:
        retain = PaceSlabMetadata(retain_plans={"producer": [(0, 42)]})
        worker_a = _connector(KVConnectorRole.WORKER, world_size=2)
        worker_b = _connector(KVConnectorRole.WORKER, world_size=2)
        cache_a = _FakeCache({"producer": [3], "consumer": []}, free_blocks=7)
        cache_b = _FakeCache({"producer": [9], "consumer": []}, free_blocks=5)
        worker_a.register_kv_cache(cache_a)
        worker_b.register_kv_cache(cache_b)

        worker_a.apply_worker_retains(retain)
        worker_b.apply_worker_retains(retain)
        aggregate = worker_a.build_connector_worker_meta().aggregate(
            worker_b.build_connector_worker_meta()
        )

        self.assertEqual(aggregate.retained, {42: 2})
        self.assertEqual(aggregate.free_blocks, 5)
        self.assertEqual(cache_a.retained, [3])
        self.assertEqual(cache_b.retained, [9])

        assignment = PaceSlabMetadata(prefix_assignments={"consumer": ([42], 4)})
        worker_a.apply_worker_assignments(assignment)
        worker_b.apply_worker_assignments(assignment)
        self.assertEqual(cache_a.assignments[0][1], [3])
        self.assertEqual(cache_b.assignments[0][1], [9])

    def test_reset_releases_every_worker_local_mapping(self) -> None:
        worker = _connector(KVConnectorRole.WORKER)
        cache = _FakeCache({"request": [3, 4]}, free_blocks=5)
        worker.register_kv_cache(cache)
        worker._worker_cache_blocks = {10: 3, 11: 4}

        worker.apply_worker_releases(PaceSlabMetadata(reset_worker_cache=True))

        self.assertEqual(cache.released, [3, 4])
        self.assertEqual(worker._worker_cache_blocks, {})

    def test_missing_sequence_is_reported_as_retain_failure(self) -> None:
        worker = _connector(KVConnectorRole.WORKER)
        cache = _FakeCache({}, free_blocks=5)
        worker.register_kv_cache(cache)

        worker.apply_worker_retains(
            PaceSlabMetadata(retain_plans={"missing": [(0, 10)]})
        )
        metadata = worker.build_connector_worker_meta()

        self.assertEqual(metadata.failed, {10: 1})
        self.assertEqual(cache.retained, [])

    def test_negative_retain_position_is_reported_as_failure(self) -> None:
        worker = _connector(KVConnectorRole.WORKER)
        cache = _FakeCache({"request": [3]}, free_blocks=5)
        worker.register_kv_cache(cache)

        worker.apply_worker_retains(
            PaceSlabMetadata(retain_plans={"request": [(-1, 10)]})
        )
        metadata = worker.build_connector_worker_meta()

        self.assertEqual(metadata.failed, {10: 1})
        self.assertEqual(cache.retained, [])

    def test_assignment_requires_existing_local_mapping(self) -> None:
        worker = _connector(KVConnectorRole.WORKER)
        cache = _FakeCache({"consumer": []}, free_blocks=5)
        worker.register_kv_cache(cache)

        with self.assertRaisesRegex(RuntimeError, "missing logical cache ID"):
            worker.apply_worker_assignments(
                PaceSlabMetadata(prefix_assignments={"consumer": ([10], 16)})
            )


if __name__ == "__main__":
    unittest.main()
