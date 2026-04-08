# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import time

import torch
from transformers import AutoTokenizer

import pace  # noqa: F401
from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    LLMOperatorType,
    LLMBackendType,
    OperatorConfig,
)
from pace.llm.attention import AttentionBackendType


def generate_and_time(model, input_encoded, gen_kwargs, pace_model=True):
    """
    Generate output from the model and time the operation.
    """
    start = time.time()
    if pace_model:
        output = model.generate(input_encoded, gen_kwargs)
    else:
        output = model.generate(
            input_encoded.input_ids,
            attention_mask=input_encoded.attention_mask,
            **gen_kwargs,
        )
    print("Time taken: ", time.time() - start)
    return output


def encode_inputs(tokenizer, inputs):
    """
    Encode inputs using the tokenizer.
    """
    return tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding="longest")


def decode_outputs(tokenizer, outputs):
    """
    Decode outputs using the tokenizer and print them.
    """
    for i, out in enumerate(outputs):
        print(f"\nModel output[{i}]: {tokenizer.decode(out, skip_special_tokens=True)}")


def run_pace_llm():
    """
    Run the PACE LLM model.
    """

    model_name = "facebook/opt-6.7b"
    torch_dtype = torch.bfloat16
    inputs_str = [
        "A lone astronaut discovers a hidden message on Mars,",
        "The world's last bookstore receives a mysterious",
        "Suddenly, all clocks in the city stop at the same",
        "2 + 5 =",
        "The American Civil War was fought",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_encoded = encode_inputs(tokenizer, inputs_str)

    opconfig = OperatorConfig(
        **{
            LLMOperatorType.Norm: LLMBackendType.NATIVE,
            LLMOperatorType.QKVProjection: LLMBackendType.TPP,
            LLMOperatorType.Attention: AttentionBackendType.JIT,
            LLMOperatorType.OutProjection: LLMBackendType.TPP,
            LLMOperatorType.MLP: LLMBackendType.TPP,
            LLMOperatorType.LMHead: LLMBackendType.NATIVE,
        }
    )

    pace_model = LLMModel(
        model_name, dtype=torch_dtype, kv_cache_type=KVCacheType.BMC, opconfig=opconfig
    )

    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "random_seed": 123,
        "stop_strings": ["\n\n"],
    }
    sampling_config = SamplingConfig(**gen_kwargs)

    print("\nRunning PACE LLM model")
    print("Model Name: ", model_name)
    print("Input: ", inputs_str)
    print("Encoded Input: ", input_encoded.input_ids)
    pace_output = generate_and_time(pace_model, input_encoded, sampling_config)
    decode_outputs(tokenizer, pace_output.output_token_ids)


if __name__ == "__main__":
    run_pace_llm()
