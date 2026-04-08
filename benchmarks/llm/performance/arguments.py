# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse
import os
import json
from importlib.util import find_spec
from typing import List

import torch

from utils import PACE_LLM_ASSERT
from datastructs import ModelArgs, GenerationArgs, TokenArgs, BenchmarkArgs


def check_if_lib_available(packages: List[str]):
    """
    Check if the given packages are installed.

    Args:
        packages (List[str]): List of package names to check.

    Raises:
        AssertionError: If any of the packages are not installed.
    """
    for package in packages:
        PACE_LLM_ASSERT(
            find_spec(package),
            f"The library {package} is not installed, and is required for this benchmark.",
        )


def verify_and_convert_operators(operators: dict) -> dict:
    """
    Verify and convert the operators to the required format.

    Args:
        operators (dict): Dictionary of operators to verify and convert.

    Returns:
        dict: Verified and converted operators.
    """
    from pace.llm.ops import LLMOperatorType, LLMBackendType
    from pace.llm.attention import AttentionBackendType

    verified_operators = {}
    for key, value in operators.items():
        key, value = key.lower(), value.lower()
        PACE_LLM_ASSERT(
            key in [op.value for op in LLMOperatorType],
            f"Unsupported operator type: {key}, only {list([value.value for value in LLMOperatorType])} are supported.",
        )
        op_key = LLMOperatorType(key)
        if op_key == LLMOperatorType.Attention:
            PACE_LLM_ASSERT(
                value in [b.value for b in AttentionBackendType],
                f"Unsupported attention backend: {value}, only {list([b.value for b in AttentionBackendType])} are supported.",
            )
            verified_operators[op_key] = AttentionBackendType(value)
        else:
            PACE_LLM_ASSERT(
                value in [backend.value for backend in LLMBackendType],
                f"Unsupported backend type: {value}, only {list([value.value for value in LLMBackendType])} are supported.",
            )
            verified_operators[op_key] = LLMBackendType(value)
    return verified_operators


def verify_args(args) -> dict:

    PACE_LLM_ASSERT(
        os.path.exists(args.config),
        f"Config file does not exist: {args.config}, please provide a valid path.",
    )

    config = {}
    with open(args.config, "r") as f:
        config_args = json.load(f)
        for key, value in config_args.items():
            config[key] = value

    frameworks = config.get("frameworks")
    if not isinstance(frameworks, list):
        config["frameworks"] = [frameworks]

    for framework in config["frameworks"]:
        if framework == "hf":
            check_if_lib_available(["transformers"])
        elif framework == "vllm":
            check_if_lib_available(["vllm"])
        elif framework == "pace":
            check_if_lib_available(["pace"])
        elif framework == "vllm_zentorch":
            check_if_lib_available(["vllm", "zentorch"])
        elif framework == "zentorch":
            check_if_lib_available(["zentorch"])
        else:
            PACE_LLM_ASSERT(
                False,
                f"Unsupported framework: {framework}, only 'hf', 'vllm', 'vllm_zentorch', 'zentorch' and 'pace' are supported.",
            )

    if config["model_args"]["dtype"] == "bf16":
        config["model_args"]["dtype"] = torch.bfloat16
    else:
        config["model_args"]["dtype"] = torch.float32

    # Only check if pace is one of the frameworks
    if "llm_operators" in config["model_args"] and "pace" in config["frameworks"]:
        PACE_LLM_ASSERT(
            isinstance(config["model_args"]["llm_operators"], dict),
            "llm_operators must be a dictionary.",
        )
        config["model_args"]["llm_operators"] = verify_and_convert_operators(
            config["model_args"]["llm_operators"]
        )

    if config["model_args"].get("spec_config") is not None:
        PACE_LLM_ASSERT(
            "model_name" in config["model_args"]["spec_config"],
            "spec_config must contain 'model_name'.",
        )
        PACE_LLM_ASSERT(
            "num_speculated_tokens" in config["model_args"]["spec_config"],
            "spec_config must contain 'num_speculated_tokens'.",
        )

    PACE_LLM_ASSERT(
        config["generation_args"]["input_tokens"] >= 1,
        "input_tokens must be a positive integer.",
    )
    PACE_LLM_ASSERT(
        config["generation_args"]["output_tokens"] >= 1,
        "output_tokens must be a positive integer.",
    )
    PACE_LLM_ASSERT(
        config["generation_args"]["batch_size"] >= 1,
        "batch_size must be a positive integer.",
    )
    PACE_LLM_ASSERT(
        isinstance(config["generation_args"]["manual_seed"], int)
        and config["generation_args"]["manual_seed"] >= 0,
        "manual_seed must be a non-negative integer if provided.",
    )

    PACE_LLM_ASSERT(
        config["generation_args"]["kv_cache_type"]
        in ["BMC", "DYNAMIC", "SLAB_POOL", "PAGED"],
        "kv_cache_type must be 'BMC', 'DYNAMIC', 'SLAB_POOL', or 'PAGED'.",
    )

    PACE_LLM_ASSERT(config["num_runs"] >= 1, "num_runs must be a positive integer.")
    PACE_LLM_ASSERT(
        config["warmup_runs"] >= 0, "warm_up_runs must be a non-negative integer."
    )
    # Check if the path to the file is valid
    if config["output_dir"]:
        PACE_LLM_ASSERT(
            os.path.exists(os.path.dirname(config["output_dir"])),
            f"Invalid path to the output file: {config['output_dir']}",
        )
        os.makedirs(config["output_dir"], exist_ok=True)

    # Either verbose or output_dir should be True
    PACE_LLM_ASSERT(
        config["verbose"] or config["output_dir"],
        f"Either verbose or output_dir should be True, but got verbose={config['verbose']} and output_dir={config['output_dir']}",
    )

    return config


def get_args() -> BenchmarkArgs:

    description = """
        This script benchmarks the offline generation performance
        of LLM models, for the given framework and model.
    """

    epilog = """
        Example usage:
        python benchmark_llm_throughput.py
            --config ./benchmark_config.json
    """

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
    )

    # Example config file:
    # {
    #     "frameworks": [
    #         "hf",
    #         "pace"
    #     ],                                        // List of frameworks to benchmark
    #     "model_args": {
    #         "model_name": "meta-llama/Llama-3.1-8B", // Name of the model to benchmark
    #         "dtype": "bf16",                      // Data type for model inputs and outputs
    #         "llm_operators": {                    // PACE specific operator configurations
    #             "Norm": "NATIVE",
    #             "QKVProjection": "TPP",
    #             "Attention" : "JIT",
    #             "OutProjection": "TPP",
    #             "MLP": "TPP",
    #             "LMHead": "NATIVE"
    #         },
    #        "spec_config": {                       // PACE specific speculated tokens configuration
    #             "model_name": "amd/PARD-Llama-3.2-1B",   // Model name for speculating
    #             "num_speculated_tokens": 12           // Number of speculated tokens
    #         }
    #     },
    #     "use_real_data": true,                    // Whether to use real data for benchmarking
    #     "generation_args": {
    #         "input_tokens": 128,                  // Number of input tokens
    #         "output_tokens": 128,                 // Number of output tokens
    #         "batch_size": 1,                      // Batch size for benchmarking
    #         "kv_cache_type": "BMC",               // Type of KV cache to use (BMC or DYNAMIC)
    #         "do_sample": false,                   // Whether to use sampling for generation
    #         "manual_seed": 0                      // Seed for random number generation
    #     },
    #     "warmup_runs": 2,                         // Number of warmup runs before actual benchmarking
    #     "num_runs": 5,                            // Number of runs to perform for benchmarking
    #     "verbose": true,                          // Whether to print detailed logs of output
    #     "output_dir": "./benchmark_results",      // Directory to save the benchmarking results
    #     "token_metrics": {
    #         "time_to_first_token": true,          // Whether to collect time to first token metrics
    #         "time_per_tokens": false              // Whether to collect time per token metrics
    #     },
    #     "system_metrics": false                   // Whether to collect system-level metrics
    # }

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the config file.",
        required=True,
    )

    args = parser.parse_args()
    config = verify_args(args)

    model_args = ModelArgs(
        model_name=config["model_args"]["model_name"],
        dtype=config["model_args"]["dtype"],
        llm_operators=config["model_args"].get("llm_operators", {}),
        spec_config=config["model_args"].get("spec_config", None),
    )
    generation_args = GenerationArgs(
        input_tokens=config["generation_args"]["input_tokens"],
        output_tokens=config["generation_args"]["output_tokens"],
        batch_size=config["generation_args"]["batch_size"],
        kv_cache_type=config["generation_args"]["kv_cache_type"],
        do_sample=config["generation_args"]["do_sample"],
        manual_seed=config["generation_args"]["manual_seed"],
    )

    token_args = TokenArgs(
        time_to_first_token=config["token_metrics"]["time_to_first_token"],
        time_per_tokens=config["token_metrics"]["time_per_tokens"],
    )

    benchmark_args = BenchmarkArgs(
        frameworks=config["frameworks"],
        model_args=model_args,
        use_real_data=config["use_real_data"],
        generation_args=generation_args,
        num_runs=config["num_runs"],
        warmup_runs=config["warmup_runs"],
        verbose=config["verbose"],
        output_dir=config["output_dir"],
        token_args=token_args,
        system_metrics=config["system_metrics"],
    )

    return benchmark_args
