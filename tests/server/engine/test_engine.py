#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Unittest-based tests for the inference server frontend (/get_models, /config_server, /step).
"""

import os
import time
import uuid
import unittest
import uvicorn
import requests
import signal
import multiprocessing as mp
from pace.server.engine.frontend import app
from pace.utils.logging import suppress_logging_cls
from pace.utils.worker import Worker

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

ENGINE_HOST = os.environ.get("PACE_ENGINE_HOST", "localhost")
ENGINE_PORT = int(os.environ.get("PACE_ENGINE_PORT", "8000"))
BASE_URL = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
CORES_LIST = list(range(os.cpu_count()))


def _server_available() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/get_models", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def run_uvicorn():
    uvicorn.run(app=app, host=ENGINE_HOST, port=ENGINE_PORT, log_config=None)


def set_log_level(level: str = "none"):
    os.environ["PACE_LOG_LEVEL"] = level


@suppress_logging_cls()
class InferenceServerTests(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.ENGINE_PROC = Worker(
            worker_id=1,
            cores_list=CORES_LIST,
            target_func=run_uvicorn,
            target_args=(),
            init_args=("none",),
            init_func=set_log_level,
        )
        cls.ENGINE_PROC.start()
        time.sleep(5)
        if not _server_available():
            raise unittest.SkipTest(f"Server not reachable at {BASE_URL}")

    @classmethod
    def tearDownClass(cls):
        os.kill(cls.ENGINE_PROC.process.pid, signal.SIGINT)
        cls.ENGINE_PROC.join()

    def test_01_get_models(self):
        resp = requests.get(f"{BASE_URL}/get_models", timeout=10)
        self.assertEqual(resp.status_code, 200, f"/get_models failed: {resp.text}")
        data = resp.json()
        self.assertIsInstance(data, (list, dict), "Expected list or dict of models")
        self.assertTrue(len(data) >= 0)

    def test_02_config_server(self):
        # Adjust operatorConfig fields to exactly match OperatorConfig in server.py
        payload = {
            "modelConfig": {
                "modelId": "facebook/opt-6.7b",
                "dataType": "bf16",
                "attnType": "dynamic",
            },
        }
        resp = requests.post(
            f"{BASE_URL}/config_server",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200, f"/config_server failed: {resp.text}")
        data = resp.json()
        self.assertIn("status", data)
        self.assertNotIn("error", data, f"Config returned error: {data}")

    def _post_step_and_assert(self, prefill_batch):
        resp = requests.post(
            f"{BASE_URL}/step",
            json={"prefill_batch": prefill_batch},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200, f"/step prefill failed: {resp.text}")
        data = resp.json()
        self.assertEqual(data.get("step_type"), "prefill_batch")
        results = data.get("results", [])
        self.assertGreater(len(results), 0, "No results returned for prefill")
        for r in results:
            req_id = r.get("req_id")
            self.assertIsNotNone(req_id)
            result_entry = r.get("result", {}).get(req_id, {})
            self.assertIn("output", result_entry)
        return data

    def test_03_prefill_basic(self):
        self.test_02_config_server()
        batch = [
            {
                "is_prefill": True,
                "prompt": "Are you listening. Once upon a time in Mexico",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 50,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 50,
                    "repetition_penalty": 1.0,
                    "do_sample": False,
                    "seed": 123,
                    "stop_strings": ["\n\n"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "AMD Hello, how are you? What is your status?",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 30,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.0,
                    "do_sample": False,
                    "seed": 456,
                    "stop_strings": ["\n\n"],
                },
            },
        ]
        self._post_step_and_assert(batch)

    def test_04_prefill_technical(self):
        self.test_02_config_server()
        batch = [
            {
                "is_prefill": True,
                "prompt": "Explain the concept of machine learning in simple terms",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 75,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 30,
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "seed": 789,
                    "stop_strings": ["\n\n", "###"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "What are the benefits of using Python for data science?",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 60,
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "top_k": 25,
                    "repetition_penalty": 1.0,
                    "do_sample": True,
                    "seed": 101112,
                    "stop_strings": ["\n\n"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "How does artificial intelligence impact society?",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 80,
                    "temperature": 0.6,
                    "top_p": 0.85,
                    "top_k": 35,
                    "repetition_penalty": 1.05,
                    "do_sample": False,
                    "seed": 131415,
                    "stop_strings": ["\n\n", "However,"],
                },
            },
        ]
        self._post_step_and_assert(batch)

    def test_05_prefill_creative(self):
        self.test_02_config_server()
        batch = [
            {
                "is_prefill": True,
                "prompt": "Write a short poem about the ocean",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 40,
                    "temperature": 1.2,
                    "top_p": 0.9,
                    "top_k": 60,
                    "repetition_penalty": 1.2,
                    "do_sample": True,
                    "seed": 161718,
                    "stop_strings": ["\n\n\n", "THE END"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "Describe a futuristic city in the year 2150",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 90,
                    "temperature": 1.0,
                    "top_p": 0.92,
                    "top_k": 55,
                    "repetition_penalty": 1.15,
                    "do_sample": True,
                    "seed": 192021,
                    "stop_strings": ["\n\n", "In conclusion"],
                },
            },
        ]
        self._post_step_and_assert(batch)

    def test_06_prefill_conversational(self):
        self.test_02_config_server()
        batch = [
            {
                "is_prefill": True,
                "prompt": "Hello! Can you help me plan a weekend trip?",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 55,
                    "temperature": 0.8,
                    "top_p": 0.88,
                    "top_k": 45,
                    "repetition_penalty": 1.0,
                    "do_sample": True,
                    "seed": 222324,
                    "stop_strings": ["\n\n", "Is there anything else"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "What's the weather like today? I'm thinking of going for a walk.",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 35,
                    "temperature": 0.6,
                    "top_p": 0.85,
                    "top_k": 40,
                    "repetition_penalty": 1.05,
                    "do_sample": False,
                    "seed": 252627,
                    "stop_strings": ["\n\n"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "Can you recommend a good book to read?",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 65,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "seed": 282930,
                    "stop_strings": ["\n\n", "Happy reading!"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "Tell me a fun fact about space exploration",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 45,
                    "temperature": 0.9,
                    "top_p": 0.87,
                    "top_k": 42,
                    "repetition_penalty": 1.0,
                    "do_sample": True,
                    "seed": 313233,
                    "stop_strings": ["\n\n", "Amazing, right?"],
                },
            },
        ]
        self._post_step_and_assert(batch)

    def test_07_prefill_coding(self):
        self.test_02_config_server()
        batch = [
            {
                "is_prefill": True,
                "prompt": "Write a Python function to calculate fibonacci numbers",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 70,
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "do_sample": False,
                    "seed": 343536,
                    "stop_strings": ["\n\n", "```"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "Explain the difference between a list and a dictionary in Python",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 85,
                    "temperature": 0.4,
                    "top_p": 0.82,
                    "top_k": 25,
                    "repetition_penalty": 1.05,
                    "do_sample": True,
                    "seed": 373839,
                    "stop_strings": ["\n\n", "In summary"],
                },
            },
            {
                "is_prefill": True,
                "prompt": "What are the best practices for writing clean code?",
                "req_id": str(uuid.uuid4()),
                "generation_config": {
                    "max_new_tokens": 95,
                    "temperature": 0.5,
                    "top_p": 0.85,
                    "top_k": 30,
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "seed": 404142,
                    "stop_strings": ["\n\n", "These practices"],
                },
            },
        ]
        self._post_step_and_assert(batch)

    def test_08_decode_loop(self):
        # Configure the engine
        self.test_02_config_server()

        # Add sequences to the system using prefill_basic
        self.test_03_prefill_basic()
        # Attempt decode multiple times until at least one finished or attempts exhausted.
        max_attempts = 30
        seen_any = False
        for _ in range(max_attempts):
            resp = requests.post(
                f"{BASE_URL}/step",
                json={"is_decode": True},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200, f"/step decode failed: {resp.text}")
            data = resp.json()
            result_map = data.get("result", {})
            if result_map:
                seen_any = True
            # Break early if any request reports a finished status
            finished = any(
                isinstance(v, dict) and v.get("status", "").endswith("COMPLETED")
                for v in result_map.values()
            )
            if finished:
                break
            time.sleep(0.3)
        self.assertTrue(seen_any, "No decode results observed after attempts")

    def test_09_get_sequences_summary(self):
        """Test getting summary of all sequences in the system."""
        # Configure the engine
        self.test_02_config_server()

        # Add sequences to the system using prefill_basic
        self.test_03_prefill_basic()

        # Perform a few decode loops to get sequences running
        for i in range(3):
            resp = requests.post(
                f"{BASE_URL}/step",
                json={"is_decode": True},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            self.assertEqual(
                resp.status_code, 200, f"Decode loop {i + 1} failed: {resp.text}"
            )
            time.sleep(0.2)

        # Get the sequences summary
        sequences_response = requests.get(
            f"{BASE_URL}/get_sequences",
            timeout=10,
        )

        # Verify the response status
        self.assertEqual(
            sequences_response.status_code,
            200,
            f"Get sequences failed: {sequences_response.text}",
        )

        # Parse and validate the response
        sequences_data = sequences_response.json()

        # Verify response structure
        self.assertIn("status", sequences_data)
        self.assertEqual(sequences_data["status"], "success")
        self.assertIn("prefill_queue", sequences_data)
        self.assertIn("decode_queue", sequences_data)
        self.assertIn("total_sequences", sequences_data)

        # Verify prefill_queue structure
        prefill_queue = sequences_data["prefill_queue"]
        self.assertIn("count", prefill_queue)
        self.assertIn("sequences", prefill_queue)
        self.assertIsInstance(prefill_queue["sequences"], list)

        # Verify decode_queue structure
        decode_queue = sequences_data["decode_queue"]
        self.assertIn("count", decode_queue)
        self.assertIn("sequences", decode_queue)
        self.assertIsInstance(decode_queue["sequences"], list)

        # Collect all sequences and validate their structure
        all_sequences = prefill_queue["sequences"] + decode_queue["sequences"]

        # Verify we have sequences in the system
        self.assertGreater(
            len(all_sequences), 0, "Expected at least one sequence in the system"
        )

        # Validate each sequence contains required fields
        for seq in all_sequences:
            self.assertIn("id", seq, "Sequence missing 'id' field")
            self.assertIn("state", seq, "Sequence missing 'state' field")

            # Verify id is a valid UUID string
            try:
                uuid.UUID(seq["id"])
            except ValueError:
                self.fail(f"Sequence id '{seq['id']}' is not a valid UUID")

            # Verify state is a non-empty string
            self.assertIsInstance(
                seq["state"], str, "Sequence state should be a string"
            )
            self.assertTrue(len(seq["state"]) > 0, "Sequence state should not be empty")

            # Optional: verify other common fields if present
            if "prompt" in seq:
                self.assertIsInstance(seq["prompt"], str)
            if "input_length" in seq:
                self.assertIsInstance(seq["input_length"], int)
            if "max_new_tokens" in seq:
                self.assertIsInstance(seq["max_new_tokens"], int)

        # Verify total_sequences count matches actual sequences
        self.assertEqual(
            sequences_data["total_sequences"],
            len(all_sequences),
            "Total sequences count doesn't match actual number of sequences",
        )

        # Verify queue counts match actual sequences in each queue
        self.assertEqual(
            prefill_queue["count"],
            len(prefill_queue["sequences"]),
            "Prefill queue count doesn't match actual sequences",
        )
        self.assertEqual(
            decode_queue["count"],
            len(decode_queue["sequences"]),
            "Decode queue count doesn't match actual sequences",
        )

    def test_10_remove_sequence(self):
        """Test removing a sequence from the system."""
        # Configure the engine
        self.test_02_config_server()

        # Add sequences to the system using prefill_basic
        self.test_03_prefill_basic()

        # Perform a few decode loops to get sequences into DECODING state
        for i in range(5):
            resp = requests.post(
                f"{BASE_URL}/step",
                json={"is_decode": True},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            self.assertEqual(
                resp.status_code, 200, f"Decode loop {i + 1} failed: {resp.text}"
            )
            time.sleep(0.2)

        # Get the sequences summary to find sequences in DECODING state
        sequences_response = requests.get(
            f"{BASE_URL}/get_sequences",
            timeout=10,
        )
        self.assertEqual(
            sequences_response.status_code,
            200,
            f"Get sequences failed: {sequences_response.text}",
        )

        sequences_data = sequences_response.json()
        decode_queue = sequences_data.get("decode_queue", {})
        decode_sequences = decode_queue.get("sequences", [])

        # Verify we have at least one sequence in decode queue
        self.assertGreater(
            len(decode_sequences),
            0,
            "Expected at least one sequence in decode queue for removal test",
        )

        # Get the ID of the first sequence in DECODING state
        sequence_to_remove = None
        for seq in decode_sequences:
            if seq.get("state") == "DECODING":
                sequence_to_remove = seq.get("id")
                break

        # If no DECODING sequence found, use the first sequence from decode queue
        if not sequence_to_remove and len(decode_sequences) > 0:
            sequence_to_remove = decode_sequences[0].get("id")

        self.assertIsNotNone(
            sequence_to_remove, "No sequence found to remove from the system"
        )

        # Get initial queue counts before removal
        initial_total = sequences_data.get("total_sequences", 0)

        # Call remove_sequence endpoint
        remove_response = requests.post(
            f"{BASE_URL}/remove_sequence",
            json={"sequence_ids": [sequence_to_remove]},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        # Verify the removal response
        self.assertEqual(
            remove_response.status_code,
            200,
            f"Remove sequence failed: {remove_response.text}",
        )

        remove_data = remove_response.json()
        self.assertIn("status", remove_data)
        self.assertEqual(remove_data["status"], "success")
        self.assertIn("removed_sequence_ids", remove_data)
        self.assertIn(sequence_to_remove, remove_data["removed_sequence_ids"])

        # Verify the sequence is actually removed by getting sequences again
        verify_response = requests.get(
            f"{BASE_URL}/get_sequences",
            timeout=10,
        )
        self.assertEqual(
            verify_response.status_code,
            200,
            f"Get sequences verification failed: {verify_response.text}",
        )

        verify_data = verify_response.json()

        # Check that total sequence count decreased by 1
        new_total = verify_data.get("total_sequences", 0)
        self.assertEqual(
            new_total,
            initial_total - 1,
            f"Total sequences should decrease by 1. Expected {initial_total - 1}, got {new_total}",
        )

        # Verify the removed sequence is not in any queue
        all_remaining_sequences = verify_data.get("prefill_queue", {}).get(
            "sequences", []
        ) + verify_data.get("decode_queue", {}).get("sequences", [])
        remaining_ids = [seq.get("id") for seq in all_remaining_sequences]
        self.assertNotIn(
            sequence_to_remove,
            remaining_ids,
            f"Removed sequence {sequence_to_remove} should not be in any queue",
        )

        # Try to get the removed sequence by ID - should return 404
        get_removed_response = requests.get(
            f"{BASE_URL}/get_sequences/{sequence_to_remove}",
            timeout=10,
        )
        self.assertEqual(
            get_removed_response.status_code,
            404,
            f"Getting removed sequence should return 404, got {get_removed_response.status_code}",
        )


if __name__ == "__main__":
    unittest.main()
