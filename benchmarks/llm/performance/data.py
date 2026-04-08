# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import random
from collections.abc import Generator
from typing import Optional

import tqdm
import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from utils import PACE_LLM_ASSERT, PACE_LLM_INFO


class BenchMarkDataGenerator(Generator):
    """
    Data generator for the benchmarking of the language model.
    The generator can be used to generate random input data or use real data from the dataset.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        input_tokens (int): Number of tokens to generate.
        batch_size (int): Number of input samples to generate.
        use_real_data (bool, optional): If True, real data from the dataset is used. Defaults to False.

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: Input tensor and attention mask.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        input_tokens: int,
        batch_size: int,
        max_iter: Optional[int] = None,
        use_real_data: Optional[bool] = False,
    ):
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens
        self.batch_size = batch_size

        self.data_in_memory = []
        if not use_real_data:
            self.generate_random_input(tokenizer, input_tokens, batch_size)
        else:
            self.generate_from_real_data(tokenizer, input_tokens, batch_size, max_iter)

        PACE_LLM_INFO(
            f"Data generator initialized with {len(self.data_in_memory)} samples"
        )

    def generate_random_input(
        self, tokenizer: PreTrainedTokenizer, input_tokens: int, batch_size: int
    ) -> torch.Tensor:
        """
        Prepares input text for the model by generating random text of the specified length.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to use.
            input_tokens (int): Number of tokens to generate.
            batch_size (int): Number of input samples to generate.

        Returns:
            torch.Tensor: Input tensor for the model.
        """

        PACE_LLM_INFO(
            f"Generating random input of {input_tokens} tokens for {batch_size} samples"
        )
        input_ids = random.choices(list(tokenizer.get_vocab().values()), k=input_tokens)
        input_ids = torch.tensor([input_ids] * batch_size)

        PACE_LLM_ASSERT(
            input_ids.shape == (batch_size, input_tokens),
            f"Input tensor shape is incorrect, expected: {(batch_size, input_tokens)}, got: {input_ids.shape}",
        )
        self.data_in_memory.append(
            BatchEncoding(
                {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            )
        )

    def generate_from_real_data(
        self,
        tokenizer: PreTrainedTokenizer,
        input_tokens: int,
        batch_size: int,
        max_iter: Optional[int] = None,
    ):

        from datasets import load_dataset

        # @Question: Should we allow the user to specify the dataset name and version?
        dataset_name = "openai/gsm8k"
        dataset_version = "main"

        tokenizer.padding_side = "left"

        PACE_LLM_INFO(
            f"Generating real data input of {input_tokens} tokens for {batch_size} batch size using {dataset_name}: v{dataset_version}."
        )

        dataset = load_dataset(dataset_name, dataset_version)["test"]
        batched_data = dataset.batch(batch_size=batch_size, drop_last_batch=True)
        max_iter = min(max_iter, len(batched_data)) if max_iter else len(batched_data)

        for idx, data in enumerate(
            tqdm.tqdm(batched_data, desc="Tokenizing data", total=max_iter)
        ):
            data_val = data["question"]
            if tokenizer.chat_template:
                data_val = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for q in data_val
                ]

            input_encoded = tokenizer.batch_encode_plus(
                data_val,
                padding="max_length",
                truncation=True,
                max_length=input_tokens,
                return_tensors="pt",
            )
            PACE_LLM_ASSERT(
                input_encoded["input_ids"].shape == (batch_size, input_tokens),
                f"Input tensor shape is incorrect, expected: {(batch_size, input_tokens)}, got: {input_encoded['input_ids'].shape}",
            )
            self.data_in_memory.append(input_encoded)
            if idx == max_iter:
                break

    def send(self, value) -> BatchEncoding:
        """
        Always return a random sample from the data in memory.
        This make sure that the benchmark is not dictated by the number of samples in the dataset.

        Returns:
            BatchEncoding: Input tensor and attention mask.
        """
        random_idx = random.choice(range(len(self.data_in_memory)))
        return self.data_in_memory[random_idx]

    def throw(self, typ, val=None, tb=None):
        pass  # Never raise an exception
