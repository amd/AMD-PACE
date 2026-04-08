# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content.
# ******************************************************************************

import logging
import os

import numpy as np

log = logging.getLogger(__name__)


class Dataset:
    def __init__(
        self,
        model_name=None,
        total_sample_count=13368,
        perf_count_override=None,
        dataset_path=None,
        dtype="bfloat16",
    ):
        self.model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"
        self.dataset_path = dataset_path
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_processed_dataset(self):
        if not self.dataset_path or not os.path.isfile(self.dataset_path):
            log.error("Dataset file not found: %s", self.dataset_path)
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}. Check --dataset-path."
            )
        import pandas as pd

        self.processed_data = pd.read_json(self.dataset_path)
        self.input = self.processed_data.input.tolist()
        self.input_ids = self.processed_data.tok_input.tolist()
        self.input_lens = [len(x) for x in self.input_ids]
        self.targets = self.processed_data.output.tolist()

        instruction_template = None
        if (
            "instruction" in self.processed_data.columns
            and len(self.processed_data) > 0
        ):
            instruction = self.processed_data["instruction"].iloc[0]
            if isinstance(instruction, dict) and "llama" in instruction:
                instruction_template = instruction["llama"]

        if instruction_template:
            self.source_encoded_input_ids = [
                instruction_template.replace("{input}", inp) for inp in self.input
            ]
        else:
            self.source_encoded_input_ids = self.input

        del self.processed_data

    def postProcess(
        self,
        out_tokens,
        query_id_list=None,
        sample_index_list=None,
    ):
        output_seq = out_tokens
        assert len(query_id_list) == len(output_seq)
        return [np.asarray(out, dtype=np.int32) for out in output_seq]

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass
