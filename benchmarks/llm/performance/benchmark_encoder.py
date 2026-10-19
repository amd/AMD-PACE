# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Micro-benchmark for PACE encoder (BERT/ModernBERT) inference.

Measures forward-pass latency for an encoder under three configurations to
quantify where PACE acceleration comes from:

    1. fp32 + NATIVE      (pure PyTorch baseline)
    2. bf16 + NATIVE      (dtype-only effect, no PACE kernels)
    3. bf16 + accelerated (TPP linear + JIT LayerNorm + JIT attention)

By default it builds a randomly-initialized BERT-base-sized model (no network
/ checkpoint needed) so the numbers reflect kernel throughput, not weight
quality. Pass ``--model <path-or-id>`` to instead load real weights via
``EmbeddingModel`` (requires the checkpoint to be available locally / online).

Example:
    python benchmark_encoder.py --batch-size 32 --seq-len 32 --iters 30
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

import torch
from transformers import BertConfig

from pace.llm.attention import AttentionBackendType
from pace.llm.configs import OperatorConfig
from pace.llm.embedding import EmbeddingModel, default_encoder_opconfig
from pace.llm.models.bert import BertModel
from pace.llm.models.model_utils import no_init_weights, set_default_torch_dtype
from pace.llm.ops import LLMOperatorType


def _native_opconfig() -> OperatorConfig:
    """All-NATIVE operator config (pure PyTorch, no PACE kernels)."""
    op = OperatorConfig()
    op[LLMOperatorType.Attention] = AttentionBackendType.NATIVE
    return op.finalize()


def _accelerated_opconfig() -> OperatorConfig:
    return default_encoder_opconfig().finalize()


def _build_synthetic_model(
    config: BertConfig, opconfig: OperatorConfig, dtype: torch.dtype
) -> BertModel:
    """Build a BertModel with small random weights for the given backend/dtype."""
    with set_default_torch_dtype(dtype):
        with no_init_weights():
            model = BertModel(config, opconfig)
    # Fill parameters with small random values (uninitialized memory could be
    # inf/nan, which would not change timing but pollutes any sanity print).
    with torch.no_grad():
        for param in model.parameters():
            if param.dtype.is_floating_point:
                param.normal_(mean=0.0, std=0.02)
    model.preprocess_weights()
    return model.eval()


def _time_forward(
    model: BertModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iters: int,
    warmup: int,
) -> list[float]:
    """Run *warmup* untimed forwards then collect *iters* latencies in ms."""
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)
        for _ in range(iters):
            start = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def _summarize(name: str, latencies_ms: list[float], baseline_ms: Optional[float]):
    median = statistics.median(latencies_ms)
    p90 = sorted(latencies_ms)[int(0.9 * (len(latencies_ms) - 1))]
    speedup = f"{baseline_ms / median:5.2f}x" if baseline_ms else "  --  "
    print(
        f"  {name:<22} median={median:8.3f} ms   p90={p90:8.3f} ms   "
        f"speedup={speedup}"
    )
    return median


def run_synthetic(args) -> None:
    """Benchmark a randomly-initialised BERT model across fp32/bf16/accelerated configs."""
    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=max(512, args.seq_len),
        type_vocab_size=2,
        hidden_act="gelu",
        pad_token_id=0,
    )

    torch.manual_seed(0)
    input_ids = torch.randint(1, args.vocab_size, (args.batch_size, args.seq_len))
    attention_mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.long)

    print(
        f"\nSynthetic BERT  hidden={args.hidden_size} layers={args.layers} "
        f"heads={args.heads} | batch={args.batch_size} seq_len={args.seq_len} | "
        f"torch_threads={torch.get_num_threads()} iters={args.iters}\n"
    )

    configs = [
        ("fp32 + NATIVE", _native_opconfig(), torch.float32),
        ("bf16 + NATIVE", _native_opconfig(), torch.bfloat16),
        ("bf16 + accelerated", _accelerated_opconfig(), torch.bfloat16),
    ]

    baseline_ms: Optional[float] = None
    for name, opconfig, dtype in configs:
        model = _build_synthetic_model(config, opconfig, dtype)
        latencies = _time_forward(
            model, input_ids, attention_mask, args.iters, args.warmup
        )
        median = _summarize(name, latencies, baseline_ms)
        if baseline_ms is None:
            baseline_ms = median
        del model


def run_real(args) -> None:
    """Load a real checkpoint via EmbeddingModel and benchmark encode() latency."""
    print(f"\nLoading real model '{args.model}' ...")
    texts = ["the quick brown fox jumps over the lazy dog"] * args.batch_size

    print(
        f"batch={args.batch_size} | torch_threads={torch.get_num_threads()} "
        f"iters={args.iters}\n"
    )

    variants = [
        ("fp32 + NATIVE", _native_opconfig(), torch.float32),
        ("bf16 + NATIVE", _native_opconfig(), torch.bfloat16),
        ("bf16 + accelerated", _accelerated_opconfig(), torch.bfloat16),
    ]

    baseline_ms: Optional[float] = None
    for name, opconfig, dtype in variants:
        model = EmbeddingModel(args.model, dtype=dtype, opconfig=opconfig)
        latencies_ms: list[float] = []
        for _ in range(args.warmup):
            model.encode(texts)
        for _ in range(args.iters):
            start = time.perf_counter()
            model.encode(texts)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        median = _summarize(name, latencies_ms, baseline_ms)
        if baseline_ms is None:
            baseline_ms = median
        del model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=None, help="Real checkpoint path/id"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model:
        run_real(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
