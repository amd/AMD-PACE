# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import re
from dataclasses import dataclass, field
from typing import ClassVar, List, Literal, Optional

import torch
from pydantic import model_validator

from pace.llm.attention import KVCacheType
from pace.llm.generation.config import (
    GenerationBackendConfig,
    register_generation_backend_config,
)
from pace.llm.step.backend import StepGenerationBackend
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_WARNING

# Below this yield, a speculation round costs more than the plain decode step it replaces.
SPECULATION_AUTO_MIN_TOKENS_PER_STEP = 1.5


@dataclass
class StepCalibrationConfig:
    """
    Configuration for reordering a checkpoint's FFN channels by saliency.

    Args:
        model_name_or_path: HF hub id or local directory holding the checkpoint.
        output_dir: Directory to write the reordered checkpoint into.
        datasets: Calibration corpora. Samples are split evenly across them.
        num_samples: Total calibration windows.
        sequence_length: Tokens per window.
        batch_size: Windows per forward pass. Saliency sums over tokens, so this affects throughput only.
        device: GPU the calibration runs on, "cuda" or "cuda:N". Decoder layers move onto it one at a time, so it needs room for the larger of one layer with the calibration activations and the rest of the model (embeddings, output head and any vision tower), rather than the whole model.
    """

    model_name_or_path: str
    output_dir: str
    datasets: List[str] = field(default_factory=lambda: ["wikitext2"])
    num_samples: int = 1024
    sequence_length: int = 2048
    batch_size: int = 4
    device: str = "cuda"

    def __post_init__(self):
        """Validate the calibration settings."""
        PACE_LLM_ASSERT(
            self.num_samples > 0,
            f"num_samples must be positive, got {self.num_samples}",
        )
        PACE_LLM_ASSERT(
            self.sequence_length > 0,
            f"sequence_length must be positive, got {self.sequence_length}",
        )
        PACE_LLM_ASSERT(
            self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        )
        PACE_LLM_ASSERT(
            len(self.datasets) > 0, "at least one calibration dataset is required"
        )
        PACE_LLM_ASSERT(
            re.fullmatch(r"cuda(:\d+)?", self.device) is not None,
            f"calibration runs on a GPU, so device must be 'cuda' or 'cuda:N', got {self.device!r}",
        )


class StepGenerationBackendConfig(GenerationBackendConfig):
    """
    Configuration for the STeP generation backend.

    The model, tokenizer and dtype come from `LLMModel` or the server's model settings, as for any backend. Sampling is set per request through `SamplingConfig`.

    Args:
        draft_ratio: Fraction of FFN channels held on the GPU. None derives it from `vram_budget_gb`.
        vram_budget_gb: VRAM the runtime may use. Sizes the channel split up front, and eviction keeps usage under it during generation. None uses the memory free on the device at startup minus 1 GiB, which also caps an explicit budget.
        max_seq_length: Sequence length the KV cache and the prefill activations are sized for. Reserved from the budget before the FFN is partitioned, and not enforced during generation: longer sequences are served, and channels move to the CPU to make room.
        tree_width: Candidates per tree level.
        tree_depth: Speculative tree levels.
        tree_mode: "narrow" reduces width by one per level; "full" keeps it.
        gpu_index: CUDA/HIP device ordinal for the prefix.
        cpu_threads: Threads for the residual. None leaves the torch default.
        residual_backend: "pace" uses the libxsmm kernels, "torch" plain ops, "auto" prefers PACE and falls back.
        speculation: "on", "off", or "auto". "off" decodes one token per full-model forward, which is plain tensor parallelism. "auto" stops speculating if the yield falls below `min_tokens_per_step`.
        min_tokens_per_step: Rolling tokens per step below which `speculation="auto"` stops.
        speculation_window: Rounds averaged before the auto decision is taken.
        draft_kv_window: Cap on the context the draft attends to. None lets the draft see everything.
        draft_kv_sink: Leading entries always retained when windowing. Applies only when `draft_kv_window` is set.
        collect_timings: Break decode time into draft, verify and CPU residual. Requires a device synchronize per phase.
        reset_between_requests: Start every request from the planned split with speculation re-enabled. By default channels evicted by one request stay on the CPU for the next, and a `speculation="auto"` fallback persists.
    """

    # Only the server's iterative scheduler sends an engine one request at a time, which is all STeP serves.
    requires_iterative_scheduler: ClassVar[bool] = True

    type: Literal["step"] = "step"
    draft_ratio: Optional[float] = None
    vram_budget_gb: Optional[float] = None
    max_seq_length: int = 4096
    tree_width: int = 3
    tree_depth: int = 3
    tree_mode: Literal["narrow", "full"] = "narrow"
    gpu_index: int = 0
    cpu_threads: Optional[int] = None
    residual_backend: Literal["auto", "pace", "torch"] = "auto"
    speculation: Literal["on", "off", "auto"] = "auto"
    min_tokens_per_step: float = SPECULATION_AUTO_MIN_TOKENS_PER_STEP
    speculation_window: int = 4
    draft_kv_window: Optional[int] = None
    draft_kv_sink: int = 256
    collect_timings: bool = False
    reset_between_requests: bool = False

    @model_validator(mode="after")
    def _validate(self):
        if self.draft_ratio is not None:
            PACE_LLM_ASSERT(
                0.0 < self.draft_ratio <= 1.0,
                f"draft_ratio must be in (0, 1], got {self.draft_ratio}",
            )
        if self.vram_budget_gb is not None:
            PACE_LLM_ASSERT(
                self.vram_budget_gb > 0.0,
                f"vram_budget_gb must be positive, got {self.vram_budget_gb}",
            )
        PACE_LLM_ASSERT(
            self.max_seq_length > 0,
            f"max_seq_length must be positive, got {self.max_seq_length}",
        )
        PACE_LLM_ASSERT(
            self.tree_width >= 1, f"tree_width must be >= 1, got {self.tree_width}"
        )
        PACE_LLM_ASSERT(
            self.tree_depth >= 1, f"tree_depth must be >= 1, got {self.tree_depth}"
        )
        PACE_LLM_ASSERT(
            self.gpu_index >= 0, f"gpu_index must be non-negative, got {self.gpu_index}"
        )
        if self.cpu_threads is not None:
            PACE_LLM_ASSERT(
                self.cpu_threads >= 1,
                f"cpu_threads must be positive, got {self.cpu_threads}",
            )
        PACE_LLM_ASSERT(
            self.min_tokens_per_step >= 1.0,
            f"min_tokens_per_step must be at least 1.0, since every round commits a bonus token, got {self.min_tokens_per_step}",
        )
        PACE_LLM_ASSERT(
            self.speculation_window >= 1,
            f"speculation_window must be positive, got {self.speculation_window}",
        )
        PACE_LLM_ASSERT(
            self.draft_kv_sink >= 0,
            f"draft_kv_sink must be non-negative, got {self.draft_kv_sink}",
        )
        if self.draft_kv_window is not None:
            PACE_LLM_ASSERT(
                self.draft_kv_window > self.draft_kv_sink + self.tree_depth,
                f"draft_kv_window must exceed draft_kv_sink plus tree_depth, got {self.draft_kv_window} against {self.draft_kv_sink} and {self.tree_depth}",
            )
        return self

    def uses_pace_kernels(self, dtype: torch.dtype) -> bool:
        """
        Whether the CPU residual will run on the PACE libxsmm kernels.

        Also determines channel alignment, since the packed kernel constrains the residual width.

        Args:
            dtype: Runtime dtype.

        Returns:
            bool: True when the packed kernel path is reachable.
        """
        return self.residual_backend != "torch" and dtype == torch.bfloat16

    def check_residual_backend(self, dtype: torch.dtype) -> None:
        """
        Refuse a residual backend the dtype cannot run, before any weights load.

        Args:
            dtype: Runtime dtype.
        """
        PACE_LLM_ASSERT(
            self.residual_backend != "pace" or dtype == torch.bfloat16,
            f"the PACE residual kernels are bfloat16 only, got {dtype}; use residual_backend 'auto' or 'torch'",
        )

    def initialize_backend(
        self,
        model_name_or_path,
        *,
        serving=False,
        tokenizer_name_or_path=None,
        dtype=torch.bfloat16,
        kv_cache_type=None,
        spec_config=None,
        opconfig=None,
        disable_tqdm=False,
        cache_kwargs=None,
        eviction_config=None,
    ):
        """
        Construct the STeP backend, rejecting shared settings that do not apply to it.

        STeP keeps its own HuggingFace model and KV caches, so the native cache and operator settings have nothing to act on. Their defaults are accepted and anything else is refused.
        """
        self.check_shared_settings(
            serving=serving,
            dtype=dtype,
            kv_cache_type=kv_cache_type,
            spec_config=spec_config,
            opconfig=opconfig,
            cache_kwargs=cache_kwargs,
            eviction_config=eviction_config,
        )
        return StepGenerationBackend(
            self,
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=dtype,
            disable_tqdm=disable_tqdm,
        )

    @staticmethod
    def check_shared_settings(
        *,
        serving: bool,
        dtype: torch.dtype,
        kv_cache_type,
        spec_config,
        opconfig,
        cache_kwargs,
        eviction_config=None,
    ) -> None:
        """
        Accept or refuse each setting `LLMModel` and the server pass to every backend.

        Args:
            serving: Whether the server is constructing the backend.
            dtype: Model dtype.
            kv_cache_type: Native KV cache layout.
            spec_config: Native speculative decoding settings.
            opconfig: Native operator backends.
            cache_kwargs: Native cache pool settings.
        """
        PACE_LLM_ASSERT(
            spec_config is None,
            "STeP drafts with its own GPU prefix, so spec_config does not apply; leave it unset",
        )
        PACE_LLM_ASSERT(
            eviction_config is None,
            "STeP keeps its own KV caches, so eviction_config does not apply; leave it unset",
        )
        # LLMModel and the server fill in PAGED for float32 and SLAB_POOL otherwise, so any other layout was asked for.
        default_layout = (
            KVCacheType.PAGED if dtype == torch.float32 else KVCacheType.SLAB_POOL
        )
        PACE_LLM_ASSERT(
            kv_cache_type is None
            or KVCacheType.get_kv_cache_type(kv_cache_type) == default_layout,
            f"STeP keeps its own KV caches, so kv_cache_type {kv_cache_type} does not apply; leave it at the default",
        )
        # The server always builds an operator config, defaults included, so it cannot be told apart from a deliberate one.
        if serving:
            if opconfig:
                PACE_LLM_WARNING(
                    "operator backends select native PACE kernels, which STeP does not use; ignoring the server's op_config"
                )
        else:
            PACE_LLM_ASSERT(
                opconfig is None,
                "operator backends select native PACE kernels, which STeP does not use; leave opconfig unset",
            )
        remaining = dict(cache_kwargs or {})
        cache_dtype = remaining.pop("dtype", dtype)
        PACE_LLM_ASSERT(
            cache_dtype == dtype,
            f"STeP's KV caches take the model dtype {dtype}, so a cache dtype of {cache_dtype} does not apply",
        )
        PACE_LLM_ASSERT(
            not remaining,
            f"STeP sizes its KV caches from vram_budget_gb and max_seq_length, so {sorted(remaining)} do not apply",
        )


register_generation_backend_config(StepGenerationBackendConfig)
