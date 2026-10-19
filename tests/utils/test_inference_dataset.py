# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import os
import shutil
import tempfile
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.testing._internal.common_utils import TestCase
from transformers import PreTrainedTokenizerFast

from pace.utils.inference_dataset import (
    CALIBRATION_DATASETS,
    CalibrationDataset,
    LLMInferenceDataset,
)
from pace.utils.logging import suppress_logging_cls

VOCAB = 50


def word_tokenizer():
    vocab = {"[UNK]": 0, "[PAD]": 1}
    vocab.update({f"w{index}": index + 2 for index in range(VOCAB)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )


def words(count, first):
    """Text drawing on 25 words from `first`, so each corpus has its own token ids."""
    return " ".join(f"w{first + index % 25}" for index in range(count))


def token_range(first):
    return range(first + 2, first + 27)


@suppress_logging_cls()
class TestCalibrationDataset(TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.tokenizer = word_tokenizer()

    def corpus(self, name, text, field="text"):
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({field: text}) + "\n")
        return path

    def test_windows_are_full_and_split_across_corpora(self):
        first = self.corpus("first.jsonl", words(400, 0))
        second = self.corpus("second.jsonl", words(400, 25), field="content")
        dataset = CalibrationDataset(self.tokenizer, [first, second], 5, 16)
        self.assertEqual(tuple(dataset.samples.shape), (5, 16))
        from_first = [
            all(token in token_range(0) for token in window.tolist())
            for window in dataset.samples
        ]
        self.assertEqual(from_first, [True, True, True, False, False])

    def test_batches_are_consecutive_windows(self):
        dataset = CalibrationDataset(
            self.tokenizer, [self.corpus("text.jsonl", words(400, 0))], 5, 16
        )
        batch = dataset.get_batch(2, 2)
        self.assertTrue(torch.equal(batch["input_ids"], dataset.samples[2:4]))
        self.assertTrue(bool(batch["attention_mask"].all()))
        self.assertEqual(dataset.get_batch(2, 4)["input_ids"].shape[0], 1)
        with self.assertRaises(IndexError):
            dataset.get_batch(2, 5)

    def test_a_corpus_shorter_than_a_window_is_refused(self):
        with self.assertRaises(AssertionError):
            CalibrationDataset(
                self.tokenizer, [self.corpus("short.jsonl", words(30, 0))], 1, 64
            )

    def test_an_alias_streams_and_reads_its_own_field(self):
        rows = Dataset.from_list([{"text": words(400, 0), "content": words(400, 25)}])
        with patch(
            "pace.utils.inference_dataset.load_dataset", return_value=rows
        ) as load:
            dataset = CalibrationDataset(self.tokenizer, ["starcoder"], 2, 16)
        load.assert_called_once_with(
            "bigcode/starcoderdata", split="train", streaming=True
        )
        for window in dataset.samples:
            self.assertTrue(all(token in token_range(25) for token in window.tolist()))

    def test_every_alias_names_a_namespaced_repository(self):
        """Recent versions of huggingface_hub refuse single-segment ids such as "wikitext"."""
        for alias, (hub_id, *_) in CALIBRATION_DATASETS.items():
            with self.subTest(alias=alias):
                self.assertRegex(hub_id, r"^[^/]+/[^/]+$")

    def test_another_hub_id_streams(self):
        rows = Dataset.from_list([{"text": words(400, 0)}])
        with patch(
            "pace.utils.inference_dataset.load_dataset", return_value=rows
        ) as load:
            CalibrationDataset(self.tokenizer, ["someone/corpus"], 2, 16)
        load.assert_called_once_with("someone/corpus", split="train", streaming=True)


@suppress_logging_cls()
class TestLLMInferenceDatasetLoading(TestCase):

    def test_a_hub_dataset_loads_whole_with_its_version(self):
        rows = Dataset.from_list([{"data": "w1 w2 w3"}, {"data": "w4 w5"}])
        with patch(
            "pace.utils.inference_dataset.load_dataset", return_value=rows
        ) as load:
            dataset = LLMInferenceDataset(
                word_tokenizer(), dataset_name="corpus", version="v1"
            )
        load.assert_called_once_with("corpus", "v1", split="test", streaming=False)
        self.assertEqual(dataset.size(), 2)

    def test_a_local_file_still_loads(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "prompts.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for text in ("w1 w2 w3", "w4 w5"):
                f.write(json.dumps({"data": text}) + "\n")
        dataset = LLMInferenceDataset(word_tokenizer(), custom_data_path=path)
        self.assertEqual(dataset.size(), 2)
        self.assertEqual(dataset.get_item(0)["input_ids"], [3, 4, 5])
