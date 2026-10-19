# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)
from datasets import load_dataset, Dataset
import json
import os
from typing import Optional, Sequence
import random
import numpy as np
import torch
from tqdm import tqdm

from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_INFO

# Short names for calibration corpora: hub id, config, split, text field, and whether to stream it rather than download it whole.
CALIBRATION_DATASETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text", False),
    "starcoder": ("bigcode/starcoderdata", None, "train", "content", True),
    "c4": ("allenai/c4", "en", "train", "text", True),
}

# Where the text is looked for in a corpus without an entry above.
_TEXT_FIELDS = ("text", "content", "code")


class LLMInferenceDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: Optional[str] = None,
        split: str = "test",
        custom_data_path: Optional[str] = None,
        version: str = None,
        max_input_len: int = 1024,
        max_target_len: int = 1024,
        input_field: str = "data",
        target_field: Optional[str] = None,
        prompt_type: Optional[str] = None,
        num_of_samples: Optional[int] = None,
        seed: Optional[int] = 123,
    ):
        """
        Initializes and tokenizes either a Hugging Face dataset, custom list of dicts, or JSON file.

        :param tokenizer: HuggingFace tokenizer instance.
        :param dataset_name: Name of the HuggingFace dataset (e.g., 'cnn_dailymail', 'gsm8k').
        :param custom_data_path: Path to a JSON/JSONL file with list of dicts.
        :param split: Dataset split to load (ignored for custom data).
        :param version: Dataset configuration/version (used for HuggingFace datasets).
        :param max_input_len: Max token length for input.
        :param max_target_len: Max token length for target.
        :param input_field: Field name to use as input text.
        :param target_field: Field name to use as target/label text.
        :param prompt_type: Field Name to be used for chat template.
        """
        self.tokenizer = tokenizer
        self.input_field = input_field
        self.target_field = target_field
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        raw_dataset = self._load_raw(dataset_name, version, split, custom_data_path)

        # Pre-tokenize all samples and store them
        self.samples = []
        for item in raw_dataset:
            input_text = item[input_field]
            if prompt_type == "llama3":
                system = "You are a helpful assistant"
                input_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif prompt_type == "r1":
                messages = [{"role": "user", "content": input_text}]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif prompt_type == "qwen":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            input_enc = tokenizer(
                input_text,
                add_special_tokens=False,
                # padding=False,
                return_tensors=None,
            )
            if target_field and target_field in item:
                target_text = item[target_field]
                target_enc = tokenizer(
                    target_text,
                    max_length=max_target_len,
                    truncation=True,
                    # padding=False,
                    return_tensors=None,
                )
                labels = target_enc["input_ids"]
            else:
                target_text = None
                labels = []

            self.samples.append(
                {
                    "input_ids": input_enc["input_ids"],
                    "attention_mask": input_enc["attention_mask"],
                    "labels": labels,
                    "input_text": input_text,
                    "target_text": target_text,
                }
            )

        if num_of_samples:
            if num_of_samples > len(self.samples):
                raise IndexError(
                    "Number of samples requested are greater than the dataset size."
                )

            self.total = num_of_samples
        else:
            self.total = len(self.samples)
        self.index = 0

    @staticmethod
    def _load_raw(
        dataset_name: Optional[str] = None,
        version: Optional[str] = None,
        split: str = "test",
        custom_data_path: Optional[str] = None,
        streaming: bool = False,
    ):
        """
        Load rows from a HuggingFace dataset, or from a JSON/JSONL file of dicts.

        :param dataset_name: Name of the HuggingFace dataset.
        :param version: Dataset configuration/version (used for HuggingFace datasets).
        :param split: Dataset split to load (ignored for custom data).
        :param custom_data_path: Path to a JSON/JSONL file with list of dicts.
        :param streaming: Stream a HuggingFace dataset rather than download it whole.
        :return: The rows, each a dict.
        """
        if dataset_name:
            if version:
                return load_dataset(
                    dataset_name, version, split=split, streaming=streaming
                )
            return load_dataset(dataset_name, split=split, streaming=streaming)

        if custom_data_path:
            with open(custom_data_path, "r", encoding="utf-8") as f:
                if custom_data_path.endswith(".json"):
                    data = json.load(f)  # a list of dicts
                elif custom_data_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError("Unsupported file format. Use .json or .jsonl")
            return Dataset.from_list(data)

        raise ValueError(
            "You must specify either 'dataset_name', 'custom_data', or 'custom_data_path'."
        )

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.total:
            raise StopIteration
        item = self.samples[self.index]
        self.index += 1
        return item

    def get_item(self, index):
        if index < 0 or index >= self.total:
            raise IndexError("Index out of range")
        return self.samples[index]

    def get_batch(self, batch_size: int, start_index: int = 0):
        """
        Return a batch of tokenized samples with dynamic padding.

        :param batch_size: Number of samples to include in the batch.
        :param start_index: Starting index of the batch.
        :return: A dictionary of batched tensors with padding.
        """
        end_index = min(start_index + batch_size, self.total)
        if start_index >= self.total:
            raise IndexError("Start index exceeds dataset size.")

        batch = self.samples[start_index:end_index]

        # Extract unbatched examples
        input_batch = [
            {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]}
            for s in batch
        ]
        label_batch = [{"input_ids": s["labels"]} for s in batch]

        # Pad inputs and labels dynamically
        input_padded = self.tokenizer.pad(
            input_batch, padding=True, return_tensors="pt"
        )
        label_padded = self.tokenizer.pad(
            label_batch, padding=True, return_tensors="pt"
        )

        return {
            "input_ids": input_padded["input_ids"],
            "attention_mask": input_padded["attention_mask"],
            "labels": label_padded["input_ids"],
            "input_texts": [s["input_text"] for s in batch],
            "target_texts": [s["target_text"] for s in batch],
        }

    def size(self):
        return self.total

    def reset(self):
        self.index = 0


class CalibrationDataset(LLMInferenceDataset):
    """
    Fixed-length token windows cut from text corpora, for calibration.

    The parent tokenizes each row as a prompt and pads a batch to its longest. Calibration wants every position to be real text, so each corpus is concatenated and cut into `sequence_length` windows, strided so they are not all drawn from its start. Windows are split evenly across the corpora, the remainder going to the first. Loading, iteration and size are the parent's; its constructor is not, since what it tokenizes differs.

    Args:
        tokenizer: Tokenizer for the model under calibration.
        datasets: Corpora: aliases from `CALIBRATION_DATASETS`, other hub ids, or JSON/JSONL files of dicts.
        num_samples: Total windows.
        sequence_length: Tokens per window.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        datasets: Sequence[str],
        num_samples: int,
        sequence_length: int,
    ):
        self.tokenizer = tokenizer
        per_source = num_samples // len(datasets)
        remainder = num_samples - per_source * len(datasets)

        windows = []
        for index, name in enumerate(datasets):
            wanted = per_source + (remainder if index == 0 else 0)
            if wanted <= 0:
                continue
            # Roughly four characters per token, doubled for slack.
            text = self._corpus_text(name, wanted * sequence_length * 8)
            # The corpus is tokenized whole and then cut into windows, so the model's length limit does not apply.
            tokens = tokenizer(
                text, return_tensors="pt", add_special_tokens=True, verbose=False
            ).input_ids[0]
            PACE_LLM_ASSERT(
                tokens.numel() >= sequence_length,
                f"dataset {name} tokenized to {tokens.numel()} tokens, fewer than the {sequence_length}-token window",
            )
            stride = max(1, (tokens.numel() - sequence_length) // max(1, wanted - 1))
            for step in range(wanted):
                start = min(step * stride, tokens.numel() - sequence_length)
                windows.append(tokens[start : start + sequence_length])
            PACE_LLM_INFO(f"calibration source {name}: {wanted} windows")

        self.samples = torch.stack(windows)
        self.total = len(windows)
        self.index = 0

    def _corpus_text(self, name: str, characters: int) -> str:
        """
        Read rows from a corpus until they hold at least `characters` of text.

        Args:
            name: Alias from `CALIBRATION_DATASETS`, a hub id, or a JSON/JSONL file.
            characters: Text to read.

        Returns:
            str: The rows' text, separated by blank lines.
        """
        if os.path.isfile(name):
            rows, fields = self._load_raw(custom_data_path=name), _TEXT_FIELDS
        else:
            # An unknown corpus could be any size, and only its first `characters` are read.
            hub_id, config, split, field, streaming = CALIBRATION_DATASETS.get(
                name, (name, None, "train", None, True)
            )
            rows = self._load_raw(hub_id, config, split, streaming=streaming)
            fields = (field,) if field else _TEXT_FIELDS

        chunks, size = [], 0
        with tqdm(
            total=characters, desc=f"Reading {name}", unit="char", unit_scale=True
        ) as progress:
            for row in rows:
                for field in fields:
                    value = row.get(field)
                    if isinstance(value, str) and len(value) > 50:
                        chunks.append(value)
                        progress.update(min(len(value), characters - size))
                        size += len(value)
                        break
                if size >= characters:
                    break

        PACE_LLM_ASSERT(
            size > 0, f"dataset {name} yielded no usable text in fields {fields}"
        )
        return "\n\n".join(chunks)

    def get_batch(self, batch_size: int, start_index: int = 0):
        """
        Return windows `start_index` to `start_index + batch_size`.

        Args:
            batch_size: Windows in the batch; the last batch may hold fewer.
            start_index: Index of the first window.

        Returns:
            dict: input_ids and attention_mask, each [batch, sequence_length].
        """
        if start_index >= self.total:
            raise IndexError("Start index exceeds dataset size.")
        input_ids = self.samples[start_index : start_index + batch_size]
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
