# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import random
from collections.abc import Generator
from itertools import islice
from typing import Optional

import tqdm
import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from utils import PACE_LLM_ASSERT, PACE_LLM_INFO

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_REVISION = "2c868b8af0ba15693ce607396871eb07802efab1"


class BenchMarkDataGenerator(Generator):
    """
    Data generator for the benchmarking of the language model.
    The generator can be used to generate random input data or use real data from the dataset.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        input_tokens (int, optional): Fixed input length. Shorter inputs are
            padded and longer inputs are truncated.
        batch_size (int): Number of input samples to generate.
        use_real_data (bool, optional): If True, real data from the dataset is used. Defaults to False.
        deterministic (bool, optional): Return real-data batches in dataset order.

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: Input tensor and attention mask.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        input_tokens: Optional[int],
        batch_size: int,
        max_iter: Optional[int] = None,
        use_real_data: Optional[bool] = False,
        deterministic: bool = False,
    ):
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.use_fixed_input_length = input_tokens is not None
        self.next_data_index = 0

        self.data_in_memory = []
        PACE_LLM_ASSERT(
            not deterministic or use_real_data,
            "deterministic data selection requires use_real_data=True.",
        )
        PACE_LLM_ASSERT(
            self.use_fixed_input_length or batch_size == 1,
            "Unpadded benchmark inputs require batch_size=1.",
        )
        if not use_real_data:
            PACE_LLM_ASSERT(
                input_tokens is not None,
                "Random benchmark inputs require input_tokens.",
            )
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
        vocab_ids = list(tokenizer.get_vocab().values())
        input_ids = torch.tensor(
            [random.choices(vocab_ids, k=input_tokens) for _ in range(batch_size)]
        )

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
        input_tokens: Optional[int],
        batch_size: int,
        max_iter: Optional[int] = None,
    ):

        from datasets import load_dataset

        tokenizer.padding_side = "left"

        input_description = (
            f"{input_tokens} fixed tokens"
            if self.use_fixed_input_length
            else "unpadded token lengths"
        )
        PACE_LLM_INFO(
            f"Generating real data input with {input_description} for "
            f"{batch_size} batch size using {GSM8K_DATASET}: {GSM8K_REVISION}."
        )

        dataset = load_dataset(
            GSM8K_DATASET,
            GSM8K_CONFIG,
            split="test",
            revision=GSM8K_REVISION,
        )
        batched_data = dataset.batch(batch_size=batch_size, drop_last_batch=True)
        if self.deterministic and max_iter is not None:
            PACE_LLM_ASSERT(
                max_iter <= len(batched_data),
                "GSM8K does not contain enough distinct batches for all runs.",
            )
        max_iter = min(max_iter, len(batched_data)) if max_iter else len(batched_data)

        for data in tqdm.tqdm(
            islice(batched_data, max_iter), desc="Tokenizing data", total=max_iter
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

            if self.use_fixed_input_length:
                input_encoded = tokenizer(
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
            else:
                input_encoded = tokenizer(
                    data_val,
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
                )
            self.data_in_memory.append(input_encoded)

    def send(self, value) -> BatchEncoding:
        """
        Return the next deterministic batch or a random sample from memory.

        Returns:
            BatchEncoding: Input tensor and attention mask.
        """
        if self.deterministic:
            PACE_LLM_ASSERT(
                self.next_data_index < len(self.data_in_memory),
                "No deterministic data batches remain.",
            )
            data = self.data_in_memory[self.next_data_index]
            self.next_data_index += 1
            return data
        random_idx = random.choice(range(len(self.data_in_memory)))
        return self.data_in_memory[random_idx]

    def throw(self, typ, val=None, tb=None):
        pass  # Never raise an exception
