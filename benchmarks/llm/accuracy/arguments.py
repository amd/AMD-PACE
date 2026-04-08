# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse
import os
import json

import torch
from pace.llm import LLMBackendType, LLMOperatorType
from pace.llm.attention import AttentionBackendType
from pace.utils.logging import PACE_LLM_ASSERT

from datastructs import ModelArgs, GenerationArgs


def verify_and_convert_operators(operators: dict) -> dict:
    """
    Verify and convert the operators to the required format.

    Args:
        operators (dict): Dictionary of operators to verify and convert.

    Returns:
        dict: Verified and converted operators.
    """
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


def verify_args(args):

    PACE_LLM_ASSERT(
        os.path.exists(args.config),
        f"Config file does not exist: {args.config}, please provide a valid path.",
    )

    config = {}
    with open(args.config, "r") as f:
        config_args = json.load(f)
        for key, value in config_args.items():
            config[key] = value

    PACE_LLM_ASSERT(
        config["generation_args"]["batch_size"] >= 1,
        "batch_size must be a positive integer.",
    )
    PACE_LLM_ASSERT(
        config["generation_args"]["kv_cache_type"]
        in ["BMC", "DYNAMIC", "PAGED", "SLAB_POOL"],
        "kv_cache_type must be 'BMC', 'DYNAMIC', 'PAGED', or 'SLAB_POOL'.",
    )

    if "llm_operators" in config["model_args"]:
        PACE_LLM_ASSERT(
            isinstance(config["model_args"]["llm_operators"], dict),
            "llm_operators must be a dictionary.",
        )
        config["model_args"]["llm_operators"] = verify_and_convert_operators(
            config["model_args"]["llm_operators"]
        )

    if "spec_config" in config_args:
        PACE_LLM_ASSERT(
            "model_name" in config_args["spec_config"],
            "spec_config must contain model_name.",
        )
        PACE_LLM_ASSERT(
            "num_speculated_tokens" in config_args["spec_config"],
            "spec_config must contain num_speculated_tokens.",
        )
        config["spec_config"] = config_args["spec_config"]

    # Make sure that num_fewshot and limit are a positive integer or None
    for task in config["tasks"]:
        PACE_LLM_ASSERT(
            task["num_fewshot"] is None or task["num_fewshot"] >= 0,
            "num_fewshot must be a positive integer or None.",
        )
        PACE_LLM_ASSERT(
            task["limit"] is None or task["limit"] >= 1,
            "limit must be a positive integer or None.",
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

    if config["model_args"]["dtype"] == "bf16":
        config["model_args"]["dtype"] = torch.bfloat16
    else:
        config["model_args"]["dtype"] = torch.float32

    return config


def get_args() -> argparse.Namespace:

    description = """
        This script evaluate the accuracy of a language model on a set of tasks.
        The tasks are defined in the config file, which specifies the model, tokenizer,
        batch size, and the tasks to evaluate on.
    """

    epilog = """
        Example usage:
        pyon llm_evaluate.py
            --config ./evaluation_config.json
    """

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
    )

    # Example config file:
    # {
    #     "model_args": {
    #         "model_name": "Qwen/Qwen2.5-7B-Instruct",
    #         "tokenizer_name": "facebook/opt-125m",
    #         "dtype": "bf16",
    #         "llm_operators": {
    #             "Norm": "NATIVE",
    #             "QKVProjection": "TPP",
    #             "Attention" : "JIT",
    #             "OutProjection": "TPP",
    #             "MLP": "TPP",
    #             "LMHead": "TPP"
    #         },
    #         "spec_config": {
    #             "model_name": "amd/PARD-Qwen2.5-0.5B",
    #             "num_speculated_tokens": 12
    #         }
    #     },
    #     "generation_args": {
    #         "batch_size": 1,
    #         "kv_cache_type": "BMC",
    #     },
    #     "tasks": [
    #         {
    #             "task_name": "mmlu",
    #             "num_fewshot": 5,
    #             "limit": 1
    #         },
    #         {
    #             "task_name": "arc_easy",
    #             "num_fewshot": 25,
    #             "limit": 1
    #         },
    #         {
    #             "task_name": "bbh_cot_fewshot ",
    #             "num_fewshot": 3,
    #             "limit": 1
    #         },
    #         {
    #             "task_name": "gsm8k",
    #             "num_fewshot": 8,
    #             "limit": 1
    #         }
    #     ],
    #     "verbose": true,
    #     "output_dir": "./evaluation_results"
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
        tokenizer_name=config["model_args"]["tokenizer_name"],
        dtype=config["model_args"]["dtype"],
        llm_operators=config["model_args"].get("llm_operators", {}),
        spec_config=config["model_args"].get("spec_config", None),
    )

    generation_args = GenerationArgs(
        batch_size=config["generation_args"]["batch_size"],
        kv_cache_type=config["generation_args"]["kv_cache_type"],
    )

    config = {
        "model_args": model_args,
        "generation_args": generation_args,
        "tasks": config["tasks"],
        "verbose": config.get("verbose", False),
        "output_dir": config.get("output_dir", None),
    }

    return config
