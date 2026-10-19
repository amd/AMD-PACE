# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PretrainedConfig

from pace.llm.attention.specs import compute_layer_kv_specs, resolve_text_config
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_WARNING

GIB = 1024**3

# libxsmm packs a [out, in] weight only when out % 32 == 0 and in % 64 == 0. The residual width covers one side of each check; the other side is d_model.
CHANNEL_ALIGNMENT = 64

# Kept free under a budget read from the device, for the allocator's slack and for the copy an eviction makes of the channels that stay on the GPU.
VRAM_RESERVE = GIB


@dataclass
class ModelMemory:
    """
    Per-model memory accounting used to size the GPU channel prefix.

    All figures are bytes at the runtime dtype. The activations are those of a prefill of `max_seq_length` tokens, the largest transient. Their FFN part scales with the GPU share, so `ffn_activations` holds it at the whole FFN's width.
    """

    attention: int
    embedding: int
    norm: int
    ffn: int
    kv_cache: int
    other: int = 0
    activations: int = 0
    ffn_activations: int = 0
    workspace: int = 0

    @property
    def fixed(self) -> int:
        """Bytes that must be resident regardless of the channel split."""
        return (
            self.attention
            + self.embedding
            + self.norm
            + self.kv_cache
            + self.other
            + self.activations
            + self.workspace
        )

    @property
    def ffn_cost(self) -> int:
        """Bytes the whole FFN would take on the GPU, its weights and its part of a prefill's activations."""
        return self.ffn + self.ffn_activations

    @property
    def total(self) -> int:
        return self.fixed + self.ffn_cost

    def installed(self, ratio: float) -> int:
        """Bytes on the GPU once a split at `ratio` is in place, before a request adds its KV cache and activations."""
        return self.fixed - self.kv_cache - self.activations + int(ratio * self.ffn)

    def describe(self, ratio: float = 0.0) -> str:
        """
        List the components for a log line.

        Args:
            ratio: GPU share of the FFN, at which its part of the prefill activations is counted.
        """
        activations = self.activations + int(ratio * self.ffn_activations)
        parts = [
            f"attention {self.attention / GIB:.2f} GiB",
            f"embedding {self.embedding / GIB:.2f} GiB",
            f"norm {self.norm / GIB:.3f} GiB",
            f"kv cache {self.kv_cache / GIB:.2f} GiB",
            f"prefill activations {activations / GIB:.2f} GiB",
            f"ffn {self.ffn / GIB:.2f} GiB",
        ]
        if self.workspace:
            parts.append(f"workspace {self.workspace / GIB:.2f} GiB")
        if self.other:
            parts.append(f"other {self.other / GIB:.2f} GiB")
        return ", ".join(parts)


@dataclass
class ChannelSplit:
    """
    A resolved FFN channel split.

    Args:
        gpu_channels: Channels held on the GPU, per layer.
        total_channels: FFN intermediate size.
        requested_ratio: Ratio asked for, before any alignment.
        aligned: Whether the split point was snapped for the packed CPU kernel.
    """

    gpu_channels: int
    total_channels: int
    requested_ratio: float
    aligned: bool

    @property
    def cpu_channels(self) -> int:
        return self.total_channels - self.gpu_channels

    @property
    def ratio(self) -> float:
        return self.gpu_channels / self.total_channels

    @property
    def packable(self) -> bool:
        """Whether the residual width is aligned for the packed libxsmm kernel. Alignment of d_model is also required and is not checked here."""
        return self.cpu_channels % CHANNEL_ALIGNMENT == 0

    def describe(self) -> str:
        detail = (
            "aligned for the packed CPU kernel" if self.aligned else "exact, unaligned"
        )
        return (
            f"{self.gpu_channels}/{self.total_channels} channels on GPU "
            f"(r={self.ratio:.4f}, requested {self.requested_ratio:.4f}), "
            f"{self.cpu_channels} on CPU, {detail}"
        )


def _model_dims(config: PretrainedConfig):
    """
    Read the dimensions the memory model needs from a checkpoint config.

    Multimodal configs nest the text model's dimensions under `text_config`.

    Args:
        config: Checkpoint config.

    Returns:
        tuple: Hidden size, FFN width, layer count, attention heads, KV heads, head dimension, vocabulary size, and whether the embeddings are tied.
    """
    text_config = resolve_text_config(config)
    hidden = text_config.hidden_size
    heads = text_config.num_attention_heads
    kv_heads = getattr(text_config, "num_key_value_heads", heads) or heads
    head_dim = getattr(text_config, "head_dim", None) or hidden // heads
    return (
        hidden,
        text_config.intermediate_size,
        text_config.num_hidden_layers,
        heads,
        kv_heads,
        head_dim,
        text_config.vocab_size,
        bool(getattr(text_config, "tie_word_embeddings", False)),
    )


def kv_bytes_per_token(config: PretrainedConfig, dtype: torch.dtype) -> int:
    """
    Bytes one token adds to one KV cache, keys and values over every layer.

    Summed layer by layer, since a layer's heads can differ from the config's defaults, as on Gemma 4's full-attention layers, and a layer that reuses another's keys and values stores none.

    Args:
        config: Checkpoint config, multimodal or not.
        dtype: Runtime dtype.

    Returns:
        int: Bytes per token per cache.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    return itemsize * sum(
        2 * spec.num_kv_heads * spec.head_dim
        for spec in compute_layer_kv_specs(config)
        if spec.kv_sharing_target is None
    )


def prefill_activation_bytes(
    config: PretrainedConfig, dtype: torch.dtype, tokens: int, ffn_channels: int
) -> int:
    """
    Transient bytes of one decoder layer's forward over a prompt, which is a prefill's peak since the layers run one after another.

    Counts the hidden states, the attention projections and output of the widest attention layer, and the FFN's gate, up and product over the channels on the GPU.

    Args:
        config: Checkpoint config, multimodal or not.
        dtype: Runtime dtype.
        tokens: Prompt length.
        ffn_channels: FFN channels on the GPU.

    Returns:
        int: Bytes.
    """
    hidden, _, _, heads, _, _, _, _ = _model_dims(config)
    attention = max(
        (heads + spec.num_kv_heads) * spec.head_dim
        for spec in compute_layer_kv_specs(config)
    )
    itemsize = torch.empty((), dtype=dtype).element_size()
    per_token = 7 * hidden + 3 * attention + 3 * ffn_channels
    return tokens * per_token * itemsize


def device_workspace_bytes(device: torch.device, dtype: torch.dtype) -> int:
    """
    Measure what this process holds on a device before the model arrives, chiefly the matrix multiply libraries' scratch space.

    The libraries allocate that space on first use and keep it, so a multiply in the runtime dtype and a batched one in float32, as rotary embeddings use, run first.

    Args:
        device: GPU to measure on.
        dtype: Runtime dtype.

    Returns:
        int: Bytes allocated on the device.
    """
    square = torch.ones(64, 64, device=device, dtype=dtype)
    torch.mm(square, square)
    batch = torch.ones(1, 64, 64, device=device, dtype=torch.float32)
    torch.bmm(batch, batch)
    del square, batch
    return torch.cuda.memory_allocated(device)


def model_memory(
    config: PretrainedConfig,
    max_seq_length: int,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    kv_caches: int = 1,
    workspace: int = 0,
) -> ModelMemory:
    """
    Account for a model's resident bytes, separating FFN from everything else.

    Args:
        config: HF config of the target model.
        max_seq_length: Sequence length the KV cache and the prefill activations are sized for.
        dtype: Runtime dtype.
        batch_size: Concurrent sequences the KV cache must cover.
        kv_caches: Caches to reserve. Speculation keeps a second one for the draft.
        workspace: Bytes already held on the device, from `device_workspace_bytes`.

    Returns:
        ModelMemory: Byte counts per component.
    """
    PACE_LLM_ASSERT(
        max_seq_length > 0,
        f"max_seq_length must be positive, got {max_seq_length}",
    )
    itemsize = torch.empty((), dtype=dtype).element_size()
    hidden, inter, layers, heads, kv_heads, head_dim, vocab, tied = _model_dims(config)

    q_and_o = 2 * hidden * heads * head_dim
    k_and_v = 2 * hidden * kv_heads * head_dim
    attention = layers * (q_and_o + k_and_v) * itemsize

    embedding = vocab * hidden * (1 if tied else 2) * itemsize
    norm = (2 * layers + 1) * hidden * itemsize
    ffn = layers * 3 * hidden * inter * itemsize
    kv_cache = (
        kv_bytes_per_token(config, dtype) * batch_size * kv_caches * max_seq_length
    )
    tokens = batch_size * max_seq_length
    activations = prefill_activation_bytes(config, dtype, tokens, 0)

    return ModelMemory(
        attention=attention,
        embedding=embedding,
        norm=norm,
        ffn=ffn,
        kv_cache=kv_cache,
        activations=activations,
        ffn_activations=prefill_activation_bytes(config, dtype, tokens, inter)
        - activations,
        workspace=workspace,
    )


def ratio_from_budget(memory: ModelMemory, budget_bytes: int) -> float:
    """
    Solve for the FFN fraction that fits a VRAM budget.

    Args:
        memory: Accounting from `model_memory`.
        budget_bytes: VRAM to plan against.

    Returns:
        float: Ratio in [0, 1]. Zero means even the fixed cost does not fit.
    """
    PACE_LLM_ASSERT(memory.ffn > 0, "model has no FFN parameters to split")
    headroom = budget_bytes - memory.fixed
    if headroom <= 0:
        return 0.0
    return min(1.0, headroom / memory.ffn_cost)


def align_channels(channels: int, total: int) -> int:
    """
    Snap a channel count so the CPU residual packs for the libxsmm kernel.

    Rounds down, because the budget is a ceiling. Falls back to no prefix at all only when the model's FFN is too narrow to align.

    Args:
        channels: Desired GPU channel count.
        total: FFN intermediate size.

    Returns:
        int: Aligned GPU channel count in [0, total].
    """
    channels = max(0, min(total, channels))
    if channels == total:
        return total

    residual = total - channels
    aligned_residual = -(-residual // CHANNEL_ALIGNMENT) * CHANNEL_ALIGNMENT
    if aligned_residual >= total:
        return 0
    return total - aligned_residual


def available_vram_bytes(gpu_index: int) -> int:
    """
    Read the memory this process can use on a device, minus `VRAM_RESERVE`.

    That is the memory free on the device plus what this process already holds, so the GPU runtime's own context and other processes, such as a desktop's display, are left out.

    Args:
        gpu_index: Device ordinal.

    Returns:
        int: Byte budget.
    """
    free, _ = torch.cuda.mem_get_info(gpu_index)
    return max(0, free + torch.cuda.memory_reserved(gpu_index) - VRAM_RESERVE)


def resolve_split(
    total_channels: int,
    draft_ratio: Optional[float],
    memory: Optional[ModelMemory] = None,
    budget_bytes: Optional[int] = None,
    align: bool = True,
) -> ChannelSplit:
    """
    Decide how many FFN channels live on the GPU.

    Args:
        total_channels: FFN intermediate size.
        draft_ratio: Explicit ratio, or None to derive one from the budget.
        memory: Accounting from `model_memory`, required when deriving.
        budget_bytes: VRAM budget, required when deriving.
        align: Adjust the split so the CPU residual packs for the fast kernel. Pass False to honour the requested ratio exactly, which applies when the residual runs on plain torch.

    Returns:
        ChannelSplit: Split point plus the ratio that was requested.
    """
    if draft_ratio is None:
        PACE_LLM_ASSERT(
            memory is not None and budget_bytes is not None,
            "deriving draft_ratio needs both a memory accounting and a budget",
        )
        draft_ratio = ratio_from_budget(memory, budget_bytes)
        PACE_LLM_ASSERT(
            draft_ratio > 0.0,
            f"the VRAM budget ({budget_bytes / GIB:.1f} GiB) does not cover this model's fixed cost ({memory.fixed / GIB:.1f} GiB: {memory.describe()}). Raise vram_budget_gb, lower max_seq_length, or use a smaller model.",
        )
    PACE_LLM_ASSERT(
        0.0 < draft_ratio <= 1.0,
        f"draft_ratio must be in (0, 1], got {draft_ratio}",
    )

    exact = int(draft_ratio * total_channels)
    split = ChannelSplit(
        gpu_channels=align_channels(exact, total_channels) if align else exact,
        total_channels=total_channels,
        requested_ratio=draft_ratio,
        aligned=align,
    )

    if split.gpu_channels == 0:
        PACE_LLM_WARNING(
            "no FFN channels fit on the GPU; STeP has no draft subnetwork to speculate with. Consider a larger budget or a smaller model."
        )
    elif split.cpu_channels == 0:
        PACE_LLM_WARNING(
            "the whole FFN fits on the GPU, so there is no CPU residual and no speculation to do. Run the model directly instead of through STeP."
        )
    elif align and not split.packable:
        PACE_LLM_WARNING(
            f"residual width {split.cpu_channels} is not a multiple of {CHANNEL_ALIGNMENT}, so the CPU FFN falls back to the unpacked path and loses most of its speedup."
        )

    return split


@dataclass
class SplitPlan:
    """
    A channel split with the accounting and budget it was solved against.

    Args:
        memory: Resident bytes per component.
        budget_bytes: VRAM budget, which eviction keeps usage under.
        split: The resolved split.
    """

    memory: ModelMemory
    budget_bytes: int
    split: ChannelSplit


def plan_split(
    config: PretrainedConfig,
    budget_bytes: int,
    max_seq_length: int,
    dtype: torch.dtype,
    speculation: str,
    draft_ratio: Optional[float] = None,
    align: bool = True,
    parameter_count: Optional[int] = None,
    workspace: int = 0,
) -> SplitPlan:
    """
    Solve for the split a budget implies, exactly as the runtime will.

    The runtime and the planning tool both call this, so a plan made without loading weights matches the split the runtime chooses. A derived split fills the budget with everything a sequence of `max_seq_length` tokens needs, so nothing is evicted until a sequence grows past it.

    Args:
        config: Checkpoint config, multimodal or not.
        budget_bytes: VRAM budget.
        max_seq_length: Sequence length the KV cache and the prefill activations are sized for.
        dtype: Runtime dtype.
        speculation: "on", "off" or "auto". Anything but "off" reserves a second KV cache for the draft.
        draft_ratio: Explicit ratio, or None to derive one from the budget. A ratio whose weights alone exceed the budget is derived instead.
        align: Align the residual for the packed CPU kernel.
        parameter_count: Parameters of the model, from the loaded model or from `count_parameters`. Weights beyond the named components, such as a vision tower, are then counted as fixed; without it they are not known.
        workspace: Bytes already held on the device, from `device_workspace_bytes`.

    Returns:
        SplitPlan: The split and what it was solved against.
    """
    memory = model_memory(
        config,
        max_seq_length,
        dtype,
        kv_caches=1 if speculation == "off" else 2,
        workspace=workspace,
    )
    if parameter_count is not None:
        itemsize = torch.empty((), dtype=dtype).element_size()
        named = memory.attention + memory.embedding + memory.norm + memory.ffn
        memory.other = max(0, parameter_count * itemsize - named)

    # Even with every channel evicted the fixed cost stays on the GPU.
    PACE_LLM_ASSERT(
        budget_bytes > memory.fixed,
        f"the {budget_bytes / GIB:.1f} GiB VRAM budget does not cover this model's fixed cost ({memory.fixed / GIB:.1f} GiB: {memory.describe()}). Raise vram_budget_gb, lower max_seq_length, or use a smaller model.",
    )
    if draft_ratio is not None:
        installed = memory.installed(draft_ratio)
        planned = memory.fixed + int(draft_ratio * memory.ffn_cost)
        if installed > budget_bytes:
            # Eviction only runs during requests, so it cannot make room while the split is installed.
            PACE_LLM_WARNING(
                f"draft_ratio {draft_ratio} would put about {installed / GIB:.1f} GiB of weights on the GPU, above the {budget_bytes / GIB:.1f} GiB budget, so the ratio is derived from the budget instead."
            )
            draft_ratio = None
        elif planned > budget_bytes:
            PACE_LLM_WARNING(
                f"draft_ratio {draft_ratio} needs about {planned / GIB:.1f} GiB to hold {max_seq_length} tokens, above the {budget_bytes / GIB:.1f} GiB budget, so channels will move to the CPU once a request needs the room. Leave draft_ratio unset to derive it from the budget."
            )

    split = resolve_split(
        resolve_text_config(config).intermediate_size,
        draft_ratio,
        memory=memory,
        budget_bytes=budget_bytes,
        align=align,
    )
    return SplitPlan(memory=memory, budget_bytes=budget_bytes, split=split)
