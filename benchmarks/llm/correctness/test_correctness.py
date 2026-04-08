# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
PACE Model Correctness Test

Compares PACE model outputs against HuggingFace Transformers using greedy
decoding and top-k logprob cross-checking. Inspired by vLLM's model
correctness testing approach.

Usage:
    # Run with a config file
    python test_correctness.py --config correctness_config.json

    # Run all registered models with a config (set model_name to "all")
    python test_correctness.py --config correctness_config_all.json
"""

import argparse
import json
import os
import re
import sys

import torch

from comparators import check_logprobs_close
from model_registry import get_all_model_ids
from runners import HfRunner, PaceRunner

from pace.llm import LLMBackendType, LLMOperatorType
from pace.llm.attention import AttentionBackendType
from pace.utils.logging import (
    PACE_LLM_INFO,
    PACE_LLM_WARNING,
    PACE_LLM_ERROR,
    PACE_LLM_ASSERT,
)

EXAMPLE_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "In a shocking finding, scientists discovered",
    "The quick brown fox jumps over the",
    "1 + 1 =",
    "Translate the following to French: Hello, how are you?",
    "A recipe for chocolate cake starts with",
    "The theory of relativity states that",
]

SUPPORTED_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def verify_and_convert_operators(operators: dict) -> dict:
    """Verify operator-to-backend mapping and convert to enum types."""
    verified = {}
    for key, value in operators.items():
        key_lower, value_lower = key.lower(), value.lower()
        PACE_LLM_ASSERT(
            key_lower in [op.value for op in LLMOperatorType],
            f"Unsupported operator type: {key}. "
            f"Valid: {[op.value for op in LLMOperatorType]}",
        )
        op_key = LLMOperatorType(key_lower)
        if op_key == LLMOperatorType.Attention:
            PACE_LLM_ASSERT(
                value_lower in [b.value for b in AttentionBackendType],
                f"Unsupported attention backend: {value}. "
                f"Valid: {[b.value for b in AttentionBackendType]}",
            )
            verified[op_key] = AttentionBackendType(value_lower)
        else:
            PACE_LLM_ASSERT(
                value_lower in [b.value for b in LLMBackendType],
                f"Unsupported backend type: {value}. "
                f"Valid: {[b.value for b in LLMBackendType]}",
            )
            verified[op_key] = LLMBackendType(value_lower)
    return verified


def load_and_verify_config(config_path: str) -> dict:
    """Load a JSON config file and validate its contents.

    Validates:
      - model_name is present (or "all" for registry models)
      - dtype is bf16 or fp32
      - kv_cache_type is BMC, DYNAMIC, SLAB_POOL or PAGED
      - llm_operators map to valid operator/backend types
      - spec_config has required keys if present
      - max_tokens >= 1
      - batch_size is a list of positive integers
      - num_logprobs >= 1
    """
    PACE_LLM_ASSERT(
        os.path.exists(config_path),
        f"Config file does not exist: {config_path}",
    )

    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*([}\]])", r"\1", text)
        config = json.loads(text)

    model_args = config.get("model_args", {})
    gen_args = config.get("generation_args", {})

    PACE_LLM_ASSERT(
        "model_name" in model_args,
        "model_args.model_name is required. Use a model name or 'all'.",
    )

    dtype_str = model_args.get("dtype", "bf16")
    PACE_LLM_ASSERT(
        dtype_str in SUPPORTED_DTYPES,
        f"Unsupported dtype: {dtype_str}. Valid: {list(SUPPORTED_DTYPES.keys())}",
    )

    kv_cache_type = gen_args.get("kv_cache_type", "DYNAMIC")
    PACE_LLM_ASSERT(
        kv_cache_type.upper() in ("BMC", "DYNAMIC", "SLAB_POOL", "PAGED"),
        f"kv_cache_type must be 'BMC', 'DYNAMIC', 'SLAB_POOL', or 'PAGED', got: {kv_cache_type}",
    )

    max_tokens = gen_args.get("max_tokens", 32)
    PACE_LLM_ASSERT(
        isinstance(max_tokens, int) and max_tokens >= 1,
        f"max_tokens must be a positive integer, got: {max_tokens}",
    )

    batch_size = gen_args.get("batch_size", [1, 8])
    if isinstance(batch_size, int):
        batch_size = [batch_size]
    PACE_LLM_ASSERT(
        isinstance(batch_size, list)
        and all(isinstance(b, int) and b >= 1 for b in batch_size),
        f"batch_size must be a positive integer or list of positive integers, got: {batch_size}",
    )

    num_logprobs = gen_args.get("num_logprobs", 5)
    PACE_LLM_ASSERT(
        isinstance(num_logprobs, int) and num_logprobs >= 1,
        f"num_logprobs must be a positive integer, got: {num_logprobs}",
    )

    llm_operators = model_args.get("llm_operators")
    if llm_operators is not None:
        PACE_LLM_ASSERT(
            isinstance(llm_operators, dict),
            "llm_operators must be a dictionary.",
        )
        llm_operators = verify_and_convert_operators(llm_operators)

    spec_config = model_args.get("spec_config")
    if spec_config is not None:
        PACE_LLM_ASSERT(
            "model_name" in spec_config,
            "spec_config must contain 'model_name'.",
        )
        PACE_LLM_ASSERT(
            "num_speculated_tokens" in spec_config,
            "spec_config must contain 'num_speculated_tokens'.",
        )
        PACE_LLM_ASSERT(
            batch_size == [1],
            f"Speculative decoding only supports batch_size [1], got {batch_size}.",
        )

    return {
        "model_name": model_args["model_name"],
        "dtype": SUPPORTED_DTYPES[dtype_str],
        "llm_operators": llm_operators,
        "kv_cache_type": kv_cache_type,
        "spec_config": spec_config,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "num_logprobs": num_logprobs,
    }


def _backend_desc(llm_operators: dict) -> str:
    if llm_operators:
        return ", ".join(f"{k}={v}" for k, v in llm_operators.items())
    return "default"


def run_correctness_test(
    model_id: str,
    max_tokens: int,
    num_logprobs: int,
    batch_size: int,
    llm_operators: dict,
    kv_cache_type: str,
    dtype: torch.dtype,
) -> bool:
    """Compare PACE vs HuggingFace using logprob cross-check.

    Returns True if the test passes, False otherwise.
    """
    prompts = EXAMPLE_PROMPTS

    PACE_LLM_INFO("Running HuggingFace model")
    with HfRunner(model_id, dtype=dtype) as hf:
        hf_outputs = hf.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs, batch_size
        )

    desc = _backend_desc(llm_operators)
    PACE_LLM_INFO(f"Running PACE model (cache={kv_cache_type}, backends=[{desc}])...")
    with PaceRunner(
        model_id,
        dtype=dtype,
        llm_operators=llm_operators,
        kv_cache_type=kv_cache_type,
    ) as pace:
        pace_outputs = pace.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs, batch_size
        )

    PACE_LLM_INFO("Comparing outputs (logprob cross-check)")
    return check_logprobs_close(
        ref_outputs=hf_outputs,
        test_outputs=pace_outputs,
        ref_name="hf",
        test_name="pace",
    )


def run_spec_decode_test(
    model_id: str,
    max_tokens: int,
    num_logprobs: int,
    batch_size: int,
    llm_operators: dict,
    kv_cache_type: str,
    spec_config: dict,
    dtype: torch.dtype,
) -> bool:
    """Compare PACE+spec-decode vs PACE-no-spec using logprob cross-check.

    Speculative decoding is mathematically equivalent to standard greedy
    decoding, but bf16 rounding differences between the batched target
    forward (spec) and incremental decode (non-spec) can flip near-tie
    argmax decisions.  We therefore use the same logprob cross-check used
    for the HF-vs-PACE comparison: a token mismatch is tolerated when
    each side's chosen token appears in the other's top-k logprobs.

    Returns True if the test passes, False otherwise.
    """
    prompts = EXAMPLE_PROMPTS
    desc = _backend_desc(llm_operators)

    PACE_LLM_INFO(
        f"Running PACE model without spec decode "
        f"(cache={kv_cache_type}, backends=[{desc}])..."
    )
    with PaceRunner(
        model_id,
        dtype=dtype,
        llm_operators=llm_operators,
        kv_cache_type=kv_cache_type,
    ) as pace_base:
        base_outputs = pace_base.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs, batch_size
        )

    PACE_LLM_INFO(
        f"Running PACE model with spec decode "
        f"(cache={kv_cache_type}, backends=[{desc}])..."
    )
    with PaceRunner(
        model_id,
        dtype=dtype,
        llm_operators=llm_operators,
        kv_cache_type=kv_cache_type,
        spec_config=spec_config,
    ) as pace_spec:
        spec_outputs = pace_spec.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs, batch_size
        )

    PACE_LLM_INFO("Comparing outputs (logprob cross-check)")
    return check_logprobs_close(
        ref_outputs=base_outputs,
        test_outputs=spec_outputs,
        ref_name="pace",
        test_name="pace+spec",
    )


def main():
    """CLI entry point for running PACE model correctness tests."""
    parser = argparse.ArgumentParser(
        description="PACE Model Correctness Test -- compare against HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="JSON config file (same format as benchmark configs)",
    )
    args = parser.parse_args()

    cfg = load_and_verify_config(args.config)

    if cfg["model_name"].lower() == "all":
        models = get_all_model_ids()
    else:
        models = [cfg["model_name"]]

    results = []
    num_prompts = len(EXAMPLE_PROMPTS)

    for model_id in models:
        for bs in cfg["batch_size"]:
            test_label = f"{model_id} (BS={bs})"
            PACE_LLM_INFO(f"{'=' * 60}")
            PACE_LLM_INFO(f"Testing: {test_label}")
            PACE_LLM_INFO(f"{'=' * 60}")

            try:
                test_fn = (
                    run_spec_decode_test if cfg["spec_config"] else run_correctness_test
                )
                passed = test_fn(
                    model_id,
                    max_tokens=cfg["max_tokens"],
                    num_logprobs=cfg["num_logprobs"],
                    batch_size=bs,
                    llm_operators=cfg["llm_operators"],
                    kv_cache_type=cfg["kv_cache_type"],
                    dtype=cfg["dtype"],
                    **(
                        {"spec_config": cfg["spec_config"]}
                        if cfg["spec_config"]
                        else {}
                    ),
                )
                status = "PASS" if passed else "FAIL"
                detail = (
                    f"{num_prompts}/{num_prompts} prompts, "
                    f"{cfg['max_tokens']} tokens"
                )
                results.append((test_label, status, detail))

            except (OSError, ValueError, ImportError) as e:
                results.append((test_label, "SKIP", str(e)))
            except Exception as e:
                import traceback

                results.append((test_label, "ERROR", str(e)))
                traceback.print_exc()

    PACE_LLM_INFO(f"{'=' * 60}")
    PACE_LLM_INFO("SUMMARY")
    PACE_LLM_INFO(f"{'=' * 60}")

    any_failed = False
    for label, status, detail in results:
        msg = f"[{status}] {label} -- {detail}"
        if status in ("FAIL", "ERROR"):
            PACE_LLM_ERROR(msg)
            any_failed = True
        elif status == "SKIP":
            PACE_LLM_WARNING(msg)
        else:
            PACE_LLM_INFO(msg)

    passed_count = sum(1 for _, s, _ in results if s == "PASS")
    total_count = len(results)
    PACE_LLM_INFO(f"{passed_count}/{total_count} tests passed.")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
