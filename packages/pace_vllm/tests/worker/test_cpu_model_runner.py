# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch  # noqa: F401
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

from pace_vllm.v1.connector import PaceSlabMetadata
from pace_vllm.v1.worker.cpu_model_runner import PaceModelRunner


class _FakeCache:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sequence_ids = {"resumed": 1, "finished": 2, "reused": 3}
        self.next_sequence_id = 4

    def create_sequence(self, request_id: str, max_seq_len: int) -> int:
        self.events.append(f"create:{request_id}")
        if request_id not in self.sequence_ids:
            self.sequence_ids[request_id] = self.next_sequence_id
            self.next_sequence_id += 1
        return self.sequence_ids[request_id]

    def get_sequence_id(self, request_id: str) -> int | None:
        return self.sequence_ids.get(request_id)

    def reset_sequence(self, request_id: str) -> None:
        self.events.append(f"reset:{request_id}")

    def remove_sequence(self, request_id: str) -> None:
        self.events.append(f"remove:{request_id}")
        self.sequence_ids.pop(request_id, None)


class _FakeConnector:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def apply_worker_retains(self, metadata: PaceSlabMetadata) -> None:
        self.events.append("retain")

    def apply_worker_releases(self, metadata: PaceSlabMetadata) -> None:
        self.events.append("release")

    def apply_worker_assignments(self, metadata: PaceSlabMetadata) -> None:
        self.events.append("assign")


def _runner(events: list[str]) -> PaceModelRunner:
    runner = PaceModelRunner.__new__(PaceModelRunner)
    runner.pace_kv_cache = _FakeCache(events)
    runner._pace_connector = _FakeConnector(events)
    runner.model_config = SimpleNamespace(max_model_len=128)
    return runner


class TestPaceModelRunnerPrefixLifecycle(unittest.TestCase):
    def test_resumed_assignment_is_applied_after_reset(self) -> None:
        events: list[str] = []
        runner = _runner(events)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[],
            finished_req_ids=set(),
            preempted_req_ids={"resumed"},
            kv_connector_metadata=PaceSlabMetadata(
                prefix_assignments={"resumed": ([10], 16)}
            ),
        )

        with patch.object(CPUModelRunner, "_update_states", return_value=None):
            PaceModelRunner._update_states(runner, scheduler_output)

        self.assertEqual(
            events,
            ["retain", "reset:resumed", "release", "assign"],
        )

    def test_sequences_release_recreate_and_assign_in_safe_order(self) -> None:
        events: list[str] = []
        runner = _runner(events)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id="reused")],
            finished_req_ids={"reused"},
            preempted_req_ids=set(),
            kv_connector_metadata=PaceSlabMetadata(
                prefix_assignments={"reused": ([10], 16)},
                evicted_cache_ids=[20],
            ),
        )

        with patch.object(CPUModelRunner, "_update_states", return_value=None):
            PaceModelRunner._update_states(runner, scheduler_output)

        self.assertEqual(
            events,
            [
                "retain",
                "remove:reused",
                "release",
                "create:reused",
                "assign",
            ],
        )

    def test_retain_happens_before_finished_sequence_removal(self) -> None:
        events: list[str] = []
        runner = _runner(events)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[],
            finished_req_ids={"finished"},
            preempted_req_ids=set(),
            kv_connector_metadata=PaceSlabMetadata(
                retain_plans={"finished": [(0, 10)]}
            ),
        )

        with patch.object(CPUModelRunner, "_update_states", return_value=None):
            PaceModelRunner._update_states(runner, scheduler_output)

        self.assertEqual(
            events,
            ["retain", "remove:finished", "release", "assign"],
        )


if __name__ == "__main__":
    unittest.main()
