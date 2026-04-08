# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import time

import torch
from transformers import AutoTokenizer, TextStreamer, PreTrainedTokenizer

from pace.llm import LLMModel, SamplingConfig, KVCacheType, PardSpecDecodeConfig


def generate_and_time(model, input_encoded, gen_kwargs, text_streamer, pace_model=True):
    """
    Generate output from the model and time the operation.
    """
    start = time.time()
    if pace_model:
        output = model.generate(input_encoded, gen_kwargs, text_streamer)
    else:
        output = model.generate(
            input_encoded.input_ids,
            attention_mask=input_encoded.attention_mask,
            streamer=text_streamer,
            **gen_kwargs,
        )
    print("Time taken: ", time.time() - start)
    return output


def encode_inputs(tokenizer: PreTrainedTokenizer, inputs):
    """
    Encode inputs using the tokenizer.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": inputs},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return tokenizer.batch_encode_plus([text], return_tensors="pt", padding="longest")


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

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype = torch.bfloat16
    inputs_str = (
        "The world is going to end and I am the only person who can save it. "
        "I have a plan, but I need your help to execute it. "
        "First, we need to gather resources and allies. "
        "We need to find a way to communicate with the other survivors and "
        "coordinate our efforts."
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    input_encoded = encode_inputs(tokenizer, inputs_str)

    gen_kwargs = {
        "max_new_tokens": 2048,
        "do_sample": False,
        "temperature": 0,
        "random_seed": 123,
    }

    kv_cache_type = KVCacheType.DYNAMIC
    pace_model = LLMModel(
        model_name,
        dtype=torch_dtype,
        spec_config=PardSpecDecodeConfig(
            model_name_or_path="amd/PARD-Qwen2.5-0.5B", num_speculative_tokens=12
        ),
        kv_cache_type=kv_cache_type,
    )
    sampling_config = SamplingConfig(**gen_kwargs)
    text_streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    print("\nRunning PACE LLM model")
    print("Model Name: ", model_name)
    print("Input: ", inputs_str)
    print("Encoded Input: ", input_encoded.input_ids)
    print("Attention Mask: ", input_encoded.attention_mask)

    generate_and_time(pace_model, input_encoded, sampling_config, text_streamer)


if __name__ == "__main__":
    run_pace_llm()
