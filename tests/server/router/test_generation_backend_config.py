# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from types import SimpleNamespace
import unittest

from pace.server.router.frontend import build_model_config, resolve_serve_type


class TestGenerationBackendConfig(unittest.TestCase):
    def test_router_forwards_native_backend_config(self):
        args = SimpleNamespace(
            model="model",
            dtype="BF16",
            op_config="{}",
            spec_config="{}",
            generation_backend_config='{"type": "native"}',
            kv_cache_type="SLAB_POOL",
            kv_cache_memory_gb=12.0,
        )

        config = build_model_config(args)

        self.assertEqual(
            config["generation_backend_config"],
            {"type": "native"},
        )
        self.assertEqual(config["kvCacheType"], "SLAB_POOL")
        self.assertEqual(config["kv_cache_memory_gb"], 12.0)

    def test_router_omits_missing_backend_config(self):
        args = SimpleNamespace(
            model="model",
            dtype="bf16",
            op_config="{}",
            spec_config="{}",
            kv_cache_type=None,
            kv_cache_memory_gb=None,
        )

        config = build_model_config(args)

        self.assertNotIn("generation_backend_config", config)

    def test_router_rejects_invalid_backend_config(self):
        for value in ("[]", '{"type": "unknown"}', "not-json"):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                build_model_config(
                    SimpleNamespace(
                        model="model",
                        dtype="bf16",
                        op_config="{}",
                        spec_config="{}",
                        generation_backend_config=value,
                        kv_cache_type=None,
                        kv_cache_memory_gb=None,
                    )
                )

    def test_step_runs_under_the_iterative_scheduler(self):
        for serve_type in ("continuous_batch", "continuous_prefill_first", "iterative"):
            with self.subTest(serve_type=serve_type):
                args = SimpleNamespace(
                    serve_type=serve_type, generation_backend_config='{"type": "step"}'
                )
                self.assertEqual(resolve_serve_type(args), "iterative")

    def test_other_backends_keep_the_requested_scheduler(self):
        for backend in ('{"type": "native"}', "{}"):
            with self.subTest(backend=backend):
                args = SimpleNamespace(
                    serve_type="continuous_batch", generation_backend_config=backend
                )
                self.assertEqual(resolve_serve_type(args), "continuous_batch")


if __name__ == "__main__":
    unittest.main()
