# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Literal
from unittest.mock import Mock, patch

import torch
from pydantic import ValidationError
from torch.testing._internal.common_utils import TestCase

from pace.llm.attention import KVCacheType
from pace.llm.generation.config import (
    GenerationBackendConfig,
    NativeGenerationBackendConfig,
    register_generation_backend_config,
)
from pace.llm.llm import LLMModel
from pace.server.engine.server import ModelExecutor
from pace.server.engine.utils import ModelConfig


class TestBackendConfig(GenerationBackendConfig):
    type: Literal["test"] = "test"
    mode: str

    def initialize_backend(self, model_name_or_path, *, serving=False, **kwargs):
        return {
            "model_name_or_path": model_name_or_path,
            "serving": serving,
            "mode": self.mode,
            **kwargs,
        }


register_generation_backend_config(TestBackendConfig)


class TestGenerationBackendConfig(TestCase):
    def test_config_subclass_requires_type_field(self):
        class MissingTypeConfig(GenerationBackendConfig):
            def initialize_backend(self, *args, **kwargs):
                return None

        with self.assertRaisesRegex(TypeError, "must declare a type field"):
            register_generation_backend_config(MissingTypeConfig)

    def test_config_subclass_rejects_duplicate_type(self):
        class DuplicateTypeConfig(GenerationBackendConfig):
            type: Literal["test"] = "test"

            def initialize_backend(self, *args, **kwargs):
                return None

        with self.assertRaisesRegex(TypeError, "already registered"):
            register_generation_backend_config(DuplicateTypeConfig)

    def test_config_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            TestBackendConfig(mode="strict", unknown=True)

    def test_offline_defaults_directly_to_native_config(self):
        backend = Mock()
        with patch.object(
            NativeGenerationBackendConfig,
            "initialize_backend",
            return_value=backend,
        ) as initialize:
            model = LLMModel("model")

        self.assertIs(model._backend, backend)
        self.assertFalse(initialize.call_args.kwargs["serving"])
        self.assertEqual(
            initialize.call_args.kwargs["kv_cache_type"],
            KVCacheType.SLAB_POOL,
        )

    def test_offline_fp32_defaults_to_paged(self):
        with patch.object(
            NativeGenerationBackendConfig,
            "initialize_backend",
            return_value=Mock(),
        ) as initialize:
            LLMModel("model", dtype=torch.float32)

        self.assertEqual(
            initialize.call_args.kwargs["kv_cache_type"],
            KVCacheType.PAGED,
        )

    def test_offline_uses_supplied_config_directly(self):
        config = TestBackendConfig(mode="offline")

        model = LLMModel("model", generation_backend_config=config)

        self.assertEqual(model._backend["mode"], "offline")
        self.assertFalse(model._backend["serving"])

    def test_non_native_receives_explicit_cache_type(self):
        model = LLMModel(
            "model",
            kv_cache_type=KVCacheType.DYNAMIC,
            generation_backend_config=TestBackendConfig(mode="offline"),
        )

        self.assertEqual(model._backend["kv_cache_type"], KVCacheType.DYNAMIC)

    def test_serving_deserializes_native_config_at_boundary(self):
        model_config = ModelConfig(
            modelId="model",
            generation_backend_config={"type": "native"},
        )

        self.assertIsInstance(
            model_config.generation_backend_config,
            NativeGenerationBackendConfig,
        )

    def test_serving_deserializes_registered_config_at_boundary(self):
        model_config = ModelConfig(
            modelId="model",
            generation_backend_config={"type": "test", "mode": "serving"},
        )

        self.assertEqual(model_config.generation_backend_config.mode, "serving")

    def test_server_uses_config_directly(self):
        backend = Mock()
        backend.model = object()
        backend.model_config = Mock()
        backend.spec_decoder = None
        backend.tokenizer.padding_side = "right"
        backend.tokenizer.pad_token = None
        backend.tokenizer.eos_token = "<eos>"
        executor = ModelExecutor()
        config = NativeGenerationBackendConfig()
        model_config = ModelConfig(
            modelId="model",
            generation_backend_config=config,
        )

        with patch.object(
            NativeGenerationBackendConfig,
            "initialize_backend",
            return_value=backend,
        ) as initialize:
            executor.load_model(model_config)

        self.assertTrue(initialize.call_args.kwargs["serving"])
