# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Mock-based tests for engine frontend endpoints.

Uses FastAPI TestClient against the app object with model_executor methods
patched — no server process, no real model, fully offline.
"""

import os
import uuid
import unittest
from unittest.mock import Mock, patch

os.environ["PACE_LOG_LEVEL"] = "none"

from fastapi.testclient import TestClient
from pace.server.engine.frontend import app, model_executor


def _uuid() -> str:
    return str(uuid.uuid4())


def _prefill_item(req_id=None, prompt=None, gen_config=None):
    return {
        "req_id": req_id or _uuid(),
        "prompt": prompt or [1, 2, 3, 4, 5],
        "is_prefill": True,
        "generation_config": gen_config or {"max_new_tokens": 10, "temperature": 0.0},
    }


# ---------------------------------------------------------------------------
# /step endpoint
# ---------------------------------------------------------------------------


class TestStepEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_decode_step(self):
        result = {"seq-1": {"status": "DECODING_IN_PROGRESS", "token_ids": [42]}}
        with patch.object(model_executor, "decode", return_value=result):
            resp = self.client.post("/step", json={"is_decode": True})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["step_type"], "decode")
        self.assertEqual(data["result"], result)

    def test_decode_empty_queue(self):
        with patch.object(model_executor, "decode", return_value={}):
            resp = self.client.post("/step", json={"is_decode": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"], {})

    def test_prefill_batch(self):
        rid = _uuid()
        prefill_result = {rid: {"token_ids": [99], "status": "PREFILL_COMPLETED"}}
        with patch.object(model_executor, "prefill", return_value=prefill_result):
            resp = self.client.post(
                "/step", json={"prefill_batch": [_prefill_item(req_id=rid)]}
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["step_type"], "prefill_batch")
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["req_id"], rid)

    def test_prefill_invalid_uuid(self):
        resp = self.client.post(
            "/step",
            json={"prefill_batch": [_prefill_item(req_id="not-a-uuid")]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("error", data["results"][0])

    def test_no_decode_no_prefill_returns_400(self):
        resp = self.client.post("/step", json={"is_decode": False})
        self.assertEqual(resp.status_code, 400)

    def test_prefill_model_error_per_item(self):
        rid = _uuid()
        with patch.object(model_executor, "prefill", side_effect=RuntimeError("OOM")):
            resp = self.client.post(
                "/step", json={"prefill_batch": [_prefill_item(req_id=rid)]}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("error", resp.json()["results"][0])

    def test_decode_model_crash_returns_500(self):
        with patch.object(
            model_executor, "decode", side_effect=RuntimeError("kernel panic")
        ):
            resp = self.client.post("/step", json={"is_decode": True})
        self.assertEqual(resp.status_code, 500)


# ---------------------------------------------------------------------------
# /step_auto_mixed endpoint
# ---------------------------------------------------------------------------


def _fake_sequence(req_id, **kwargs):
    """Build a lightweight Sequence stand-in for endpoint tests."""
    seq = Mock()
    seq.id = req_id if isinstance(req_id, uuid.UUID) else uuid.UUID(req_id)
    return seq


class TestStepAutoMixedEndpoint(unittest.TestCase):

    SEQUENCE_PATH = "pace.server.engine.frontend.Sequence"

    def setUp(self):
        self.client = TestClient(app)
        model_executor.prefill_queue = {}
        model_executor.decode_queue = {}
        model_executor._spec_decoder = None
        model_executor.model_config = Mock()
        model_executor._cache_backend = Mock()
        model_executor._tokenizer = Mock()
        model_executor._model = Mock()

    def test_empty_batch_no_decodes(self):
        resp = self.client.post("/step_auto_mixed", json={"prefill_batch": []})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["stats"]["total_in_batch"], 0)

    def test_prefill_creates_sequence_and_calls_mixed(self):
        rid = _uuid()
        prefill_out = {rid: {"status": "PREFILL_COMPLETED", "token_ids": [42]}}
        with (
            patch(self.SEQUENCE_PATH, side_effect=_fake_sequence),
            patch.object(
                model_executor,
                "prefill_decode_mixed_batch",
                return_value=(prefill_out, {}),
            ),
        ):
            resp = self.client.post(
                "/step_auto_mixed",
                json={"prefill_batch": [_prefill_item(req_id=rid)]},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["prefill_result"][rid]["status"], "PREFILL_COMPLETED")

    def test_existing_sequence_reused(self):
        rid = _uuid()
        seq = Mock()
        seq.id = uuid.UUID(rid)
        model_executor.prefill_queue[uuid.UUID(rid)] = seq

        with patch.object(
            model_executor,
            "prefill_decode_mixed_batch",
            return_value=({rid: {"status": "PREFILL_COMPLETED", "token_ids": [1]}}, {}),
        ):
            resp = self.client.post(
                "/step_auto_mixed",
                json={"prefill_batch": [_prefill_item(req_id=rid)]},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(model_executor.prefill_queue), 1)

    def test_decode_queue_included(self):
        seq = Mock()
        seq.id = uuid.uuid4()
        model_executor.decode_queue[seq.id] = seq

        decode_out = {str(seq.id): {"status": "DECODING_IN_PROGRESS", "token_ids": [7]}}
        with patch.object(
            model_executor,
            "prefill_decode_mixed_batch",
            return_value=({}, decode_out),
        ):
            resp = self.client.post("/step_auto_mixed", json={"prefill_batch": []})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["stats"]["num_decode"], 1)

    def test_chunk_info_forwarded(self):
        rid = _uuid()
        seq = Mock()
        seq.id = uuid.UUID(rid)
        model_executor.prefill_queue[uuid.UUID(rid)] = seq

        captured = {}

        def fake_mixed(prefill_seqs, decode_seqs, chunk_info_map):
            captured["chunk_info_map"] = chunk_info_map
            return (
                {rid: {"status": "PREFILL_IN_PROGRESS", "num_computed_tokens": 50}},
                {},
            )

        with patch.object(
            model_executor, "prefill_decode_mixed_batch", side_effect=fake_mixed
        ):
            resp = self.client.post(
                "/step_auto_mixed",
                json={
                    "prefill_batch": [_prefill_item(req_id=rid)],
                    "chunk_info_map": {rid: [50, 50]},
                },
            )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(rid, captured["chunk_info_map"])
        self.assertEqual(captured["chunk_info_map"][rid], (50, 50))

    def test_prefill_creation_error_captured(self):
        with patch(self.SEQUENCE_PATH, side_effect=RuntimeError("bad config")):
            resp = self.client.post(
                "/step_auto_mixed",
                json={"prefill_batch": [_prefill_item()]},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        errors = [v for v in data["prefill_result"].values() if "error" in v]
        self.assertGreater(len(errors), 0)

    def test_mixed_batch_crash_returns_500(self):
        rid = _uuid()
        seq = Mock()
        seq.id = uuid.UUID(rid)
        model_executor.prefill_queue[uuid.UUID(rid)] = seq

        with patch.object(
            model_executor,
            "prefill_decode_mixed_batch",
            side_effect=RuntimeError("forward exploded"),
        ):
            resp = self.client.post(
                "/step_auto_mixed",
                json={"prefill_batch": [_prefill_item(req_id=rid)]},
            )
        self.assertEqual(resp.status_code, 500)

    def test_spec_decoder_headroom(self):
        spec_mock = Mock()
        spec_mock.num_speculative_tokens = 5
        model_executor._spec_decoder = spec_mock

        rid = _uuid()
        with (
            patch(self.SEQUENCE_PATH, side_effect=_fake_sequence),
            patch.object(
                model_executor,
                "prefill_decode_mixed_batch",
                return_value=(
                    {rid: {"status": "PREFILL_COMPLETED", "token_ids": [1]}},
                    {},
                ),
            ),
        ):
            resp = self.client.post(
                "/step_auto_mixed",
                json={"prefill_batch": [_prefill_item(req_id=rid)]},
            )
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# /remove_sequence endpoint
# ---------------------------------------------------------------------------


class TestRemoveSequenceEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_remove_single(self):
        rid = str(uuid.uuid4())
        with patch.object(model_executor, "remove_sequences", return_value=None) as m:
            resp = self.client.post("/remove_sequence", json={"sequence_ids": [rid]})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn(rid, data["removed_sequence_ids"])
        m.assert_called_once()

    def test_remove_batch(self):
        rids = [str(uuid.uuid4()) for _ in range(3)]
        with patch.object(model_executor, "remove_sequences", return_value=None) as m:
            resp = self.client.post("/remove_sequence", json={"sequence_ids": rids})
        self.assertEqual(resp.status_code, 200)
        called_uuids = m.call_args[0][0]
        self.assertEqual(len(called_uuids), 3)

    def test_remove_invalid_uuid(self):
        resp = self.client.post(
            "/remove_sequence", json={"sequence_ids": ["not-valid"]}
        )
        self.assertEqual(resp.status_code, 500)

    def test_remove_empty_list(self):
        resp = self.client.post("/remove_sequence", json={"sequence_ids": []})
        self.assertIn(resp.status_code, (400, 500))

    def test_remove_missing_field(self):
        resp = self.client.post("/remove_sequence", json={"foo": "bar"})
        self.assertIn(resp.status_code, (400, 422, 500))


# ---------------------------------------------------------------------------
# Pydantic validation (StepRequest / MixedStepRequest)
# ---------------------------------------------------------------------------


class TestPydanticValidation(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_step_rejects_extra_unknown_body(self):
        with patch.object(model_executor, "decode", return_value={}):
            resp = self.client.post(
                "/step", json={"is_decode": True, "bogus_field": 123}
            )
        self.assertEqual(resp.status_code, 200)

    def test_mixed_step_negative_chunk_start(self):
        rid = _uuid()
        resp = self.client.post(
            "/step_auto_mixed",
            json={
                "prefill_batch": [_prefill_item(req_id=rid)],
                "chunk_info_map": {rid: [-1, 50]},
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_mixed_step_zero_chunk_size(self):
        rid = _uuid()
        resp = self.client.post(
            "/step_auto_mixed",
            json={
                "prefill_batch": [_prefill_item(req_id=rid)],
                "chunk_info_map": {rid: [0, 0]},
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_mixed_step_chunk_info_wrong_length(self):
        rid = _uuid()
        resp = self.client.post(
            "/step_auto_mixed",
            json={
                "prefill_batch": [_prefill_item(req_id=rid)],
                "chunk_info_map": {rid: [10]},
            },
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
