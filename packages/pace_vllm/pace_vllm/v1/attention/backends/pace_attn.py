# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PACE attention backend for vLLM v1.

- `PaceAttentionBackend.get_kv_cache_shape` returns `(0,)`; the real
  K/V lives in `SlabPool` instances owned by `PaceKVCache`.
- `PaceAttentionImpl` calls `SlabPool.cache_update` + `SlabPool.attention`
  on each forward. vLLM 0.20+ passes a pre-allocated `output` buffer
  (see `unified_attention_with_output`) and discards the return value,
  so the impl writes in-place when `output` is non-None and falls back
  to returning a fresh tensor for vLLM 0.19's `output=None` path.
- `PaceAttentionMetadataBuilder` pre-computes `query_lens` and the
  `req_id -> slab seq_id` map once per step so the impl does not
  repeat that Python work per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

from pace_vllm.v1.attention.layer_registry import (
    get_current_kv_cache,
    lookup_layer,
)

if TYPE_CHECKING:  # pragma: no cover
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger("pace_vllm.v1.attention.backends.pace_attn")


class PaceAttentionBackend(AttentionBackend):
    """Attention backend backed by `torch.classes.pace.SlabPool`.

    `get_kv_cache_shape` returns `(0,)` so vLLM's block-table allocator
    ships zero-sized placeholders for every layer; real K/V lives in
    `SlabPool` instances owned by `PaceKVCache`.
    """

    # `accept_output_buffer` is meaningless on vLLM 0.20+ (the field was
    # removed from `AttentionBackend` and `unified_attention_with_output`
    # always passes a pre-allocated `output`). On vLLM 0.19 the default
    # is `False`, which is what we need to keep the legacy "return value
    # is the answer" path. Either way, `forward()` writes into `output`
    # when it is non-None.
    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        # Shadow CPU_ATTN so vLLM's `AttentionBackendEnum[...]` lookup
        # resolves without adding a PACE_SLAB enum value.
        return "CPU_ATTN"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 112, 128, 160, 192, 224, 256]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["PaceAttentionImpl"]:
        return PaceAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["PaceAttentionMetadataBuilder"]:
        return PaceAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (0,)

    @staticmethod
    def get_kv_cache_block_dim(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        return 0

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class PaceAttentionMetadata:
    """Per-step attention metadata consumed by `PaceAttentionImpl.forward`.

    `seq_ids is None` signals warmup / dummy_run (no mapping yet).
    """

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    query_lens: list[int]
    seq_ids: list[int] | None


class PaceAttentionMetadataBuilder(AttentionMetadataBuilder[PaceAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # SlabPool handles mixed batches uniformly, so no decode/prefill split.
        self._init_reorder_batch_threshold(
            reorder_batch_threshold=None,
            supports_spec_as_decode=False,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> PaceAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs

        qsl_py = common_attn_metadata.query_start_loc[: num_reqs + 1].tolist()
        query_lens = [qsl_py[i + 1] - qsl_py[i] for i in range(num_reqs)]

        seq_ids: list[int] | None = None
        kv_cache = get_current_kv_cache()
        if kv_cache is not None and num_reqs > 0:
            owner = kv_cache.get_owner()
            if owner is not None:
                # `input_batch is None` is the legitimate warmup window before
                # PaceModelRunner wires its InputBatch onto the worker. Once
                # input_batch exists, req_ids must match num_reqs (vLLM 0.20's
                # InputBatch.req_ids is a non-Optional list[str] kept in sync
                # by _update_states each step).
                input_batch = getattr(owner, "input_batch", None)
                if input_batch is not None:
                    req_ids = getattr(input_batch, "req_ids", None)
                    if req_ids is None or len(req_ids) < num_reqs:
                        raise RuntimeError(
                            "PaceAttentionMetadataBuilder.build: "
                            "input_batch.req_ids is missing or shorter than "
                            f"num_reqs={num_reqs} (got "
                            f"{None if req_ids is None else len(req_ids)}). "
                            "State divergence between vLLM scheduler and "
                            "PaceModelRunner."
                        )
                    resolved: list[int] = []
                    for rid in req_ids[:num_reqs]:
                        sid = kv_cache.get_sequence_id(rid)
                        if sid is None:
                            raise RuntimeError(
                                "PaceAttentionMetadataBuilder.build: req_id "
                                f"{rid!r} has no slab sequence_id. "
                                "PaceModelRunner._update_states must call "
                                "create_sequence on every scheduled new "
                                "request before the metadata builder runs. "
                                f"Resolved {len(resolved)} of {num_reqs} "
                                "req_ids before the gap."
                            )
                        resolved.append(sid)
                    seq_ids = resolved

        return PaceAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=num_reqs,
            query_lens=query_lens,
            seq_ids=seq_ids,
        )


class PaceAttentionImpl(AttentionImpl[PaceAttentionMetadata]):
    """SlabPool-backed `AttentionImpl`.

    Each forward appends K/V via `SlabPool.cache_update` and runs
    `SlabPool.attention`; per-step invariants (`query_lens`, `seq_ids`)
    come pre-computed from `PaceAttentionMetadataBuilder`. The layer's
    slab pool is bound on the first successful forward and cached on
    the instance.

    Warmup / dummy_run (missing attn_metadata, empty batch, or
    `input_batch is None`) returns the caller-provided `output` buffer
    on vLLM 0.20+ (which pre-allocates and discards return values), or
    a fresh empty tensor shaped like `query` on vLLM 0.19 where
    `output=None` -- matching `CPUAttentionBackendImpl`'s contract on
    each version.
    """

    # Singleton reused when the layer has no sinks -- avoids per-call alloc.
    _SINKS_EMPTY: torch.Tensor = torch.tensor([], dtype=torch.bfloat16)

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        if num_kv_heads is not None:
            if num_kv_heads <= 0:
                raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}.")
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({num_heads}) must be divisible by "
                    f"num_kv_heads ({num_kv_heads})."
                )
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.alibi_slopes = alibi_slopes

        # SlabPool.attention takes an int: 0 => full causal, >0 => window.
        self.sliding_window = 0 if sliding_window is None else int(sliding_window)

        if sinks is not None and sinks.shape[0] != num_heads:
            raise ValueError(
                "Attention sinks must have one entry per head "
                f"(expected {num_heads}, got shape {tuple(sinks.shape)})."
            )
        self.sinks = sinks

        self._pace_pool = None  # type: ignore[assignment]

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: PaceAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "pace-vllm: fused output quantization is not supported by "
                "PaceAttentionImpl."
            )

        # vLLM 0.20+ pre-allocates `output` and discards our return value
        # (see `unified_attention_with_output` in
        # vllm/model_executor/layers/attention/attention.py:771); we MUST
        # write into the buffer in-place. vLLM 0.19 still passed
        # `output=None` and consumed our return; supporting both means
        # writing into `output` when present and returning either way.
        if (
            attn_metadata is None
            or attn_metadata.num_reqs == 0
            or attn_metadata.seq_ids is None
        ):
            return output if output is not None else torch.empty_like(query)

        pool = self._pace_pool
        if pool is None:
            entry = lookup_layer(layer.layer_name)
            if entry is None:
                raise RuntimeError(
                    f"PaceAttentionImpl.forward: layer {layer.layer_name!r} "
                    "is not in the pace layer registry. "
                    "PaceModelRunner.initialize_kv_cache must register every "
                    "attention layer (and KV-sharing followers) before the "
                    "first forward call."
                )
            self._pace_pool = pool = entry[1]

        # CPU v1 ragged-packs; SlabPool consumes ragged inputs natively
        # (cache_update + attention take (sum(query_lens), heads, head_size)
        # 3D tensors). Trip-wire enforces the contract on every forward;
        # `raise` (not `assert`) so it survives `python -O`.
        if query.shape[0] != attn_metadata.num_actual_tokens:
            raise RuntimeError(
                "PaceAttentionImpl expects ragged-packed inputs: "
                f"query.shape[0]={query.shape[0]}, "
                f"num_actual_tokens={attn_metadata.num_actual_tokens}."
            )

        q = query.contiguous().view(-1, self.num_heads, self.head_size)

        seq_ids = attn_metadata.seq_ids
        query_lens = attn_metadata.query_lens

        if key is not None and value is not None:
            k = key.contiguous().view(-1, self.num_kv_heads, self.head_size)
            v = value.contiguous().view(-1, self.num_kv_heads, self.head_size)
            pool.cache_update(seq_ids, k, v, query_lens)
        elif key is not None or value is not None:
            raise RuntimeError("PaceAttentionImpl requires key and value together.")

        sinks_tensor = self.sinks if self.sinks is not None else self._SINKS_EMPTY
        attn_out = pool.attention(
            seq_ids,
            q,
            query_lens,
            [],
            self.scale,
            int(self.sliding_window),
            sinks_tensor,
        )

        if output is not None:
            # SlabPool returns (T, H, D); query/output are (T, H*D). View
            # to match before the in-place copy.
            output.copy_(attn_out.view_as(query))
            return output
        return attn_out.view_as(query)
