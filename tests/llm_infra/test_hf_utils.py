# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import tempfile
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.utils import filter_repo_objects
from torch.testing._internal.common_utils import TestCase
from transformers import LlamaConfig

from pace.llm.models import hf_utils
from pace.llm.models.hf_utils import download_or_cached_weights, resolve_model_path


class TestHubDownload(TestCase):
    """Hub ids resolve to a cached snapshot, checked against the native model list unless told not to."""

    def setUp(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        LlamaConfig(architectures=["Mistral3ForConditionalGeneration"]).save_pretrained(
            f"{cache.name}/config"
        )
        self.snapshot = Path(cache.name) / "snapshot"
        patch.multiple(
            hf_utils.constants, HF_HUB_CACHE=cache.name, HF_HUB_OFFLINE=True
        ).start()
        self.config_download = patch.object(
            hf_utils, "hf_hub_download", return_value=f"{cache.name}/config/config.json"
        ).start()
        self.snapshot_download = patch.object(
            hf_utils, "snapshot_download", return_value=str(self.snapshot)
        ).start()
        self.addCleanup(patch.stopall)

    def test_an_architecture_outside_the_native_list_is_refused_by_default(self):
        """The native backend's model list still gates the default."""
        with self.assertRaisesRegex(ValueError, "not supported"):
            download_or_cached_weights("mistralai/model")
        self.snapshot_download.assert_not_called()

    def test_any_architecture_is_downloaded_when_not_native_only(self):
        """The config is not checked."""
        self.assertEqual(
            resolve_model_path("mistralai/model", native_only=False), self.snapshot
        )
        self.config_download.assert_not_called()

    def test_only_the_files_transformers_loads_are_downloaded(self):
        """Meta's original checkpoint and Mistral's consolidated copy of the weights are skipped."""
        download_or_cached_weights("someone/model", native_only=False)
        files = [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "README.md",
            "original/consolidated.00.pth",
            "original/params.json",
            "consolidated-00001-of-00002.safetensors",
            "consolidated.safetensors.index.json",
        ]
        kept = filter_repo_objects(
            files,
            ignore_patterns=self.snapshot_download.call_args.kwargs["ignore_patterns"],
        )
        self.assertEqual(list(kept), files[:4])
