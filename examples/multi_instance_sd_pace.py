# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import multiprocessing as mp
from transformers import (
    AutoTokenizer,
)
import torch
from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    LLMOperatorType,
    LLMBackendType,
    OperatorConfig,
    PardSpecDecodeConfig,
)
from pace.llm.attention import AttentionBackendType
import time
from pace.utils.worker import Worker, MultipleProcesses
from pace.utils.inference_dataset import LLMInferenceDataset
import psutil
import argparse
from pace.utils.logging import PACE_INFO, PACE_DEBUG, suppress_logging
import random
import numpy as np


seed = 123


class LLMWorker(Worker):
    """
    A specialized Worker for handling LLM (Large Language Models) tasks.

    Attributes:
        model: The language model used for inference.
        tokenizer: Tokenizer for processing input text.
    """

    def __init__(
        self, w_id, cores_list, init_func, init_args, target_func, target_args
    ):
        """
        self._initialize(
            w_id,
            cores_list,
            self.init_model,
            init_args,
            self.run_model,
            target_args,
        )
        """
        super().__init__(
            w_id,
            cores_list,
            self.init_model,
            init_args,
            self.run_model,
            target_args,
        )

    def init_model(
        self,
        model_name,
        pard_model_name,
        dataset_path,
        dataset_version,
        dataset_input_field,
        num_of_samples,
    ):
        """
        Initializes the language model and tokenizer.
        """
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

        model_type = "llama3"
        if "llama3" in model_name or "Llama-3" in model_name:
            model_type = "llama3"
        elif ("r1" in model_name) or ("R1" in model_name):
            model_type = "r1"
        elif "qwen" in model_name or "Qwen" in model_name:
            model_type = "qwen"

        kv_cache_type = KVCacheType.BMC

        self.model = LLMModel(
            model_name,
            dtype=torch.bfloat16,
            spec_config=PardSpecDecodeConfig(
                model_name_or_path=pard_model_name,
                num_speculative_tokens=12,
            ),
            kv_cache_type=kv_cache_type,
            opconfig=opconfig,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, model_max_length=2048, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # create the dataset
        self.dataset = LLMInferenceDataset(
            dataset_name=dataset_path,
            version=dataset_version,
            tokenizer=self.tokenizer,
            input_field=dataset_input_field,
            num_of_samples=num_of_samples,
            prompt_type=model_type,
            seed=seed,
        )

    def run_model(self, input_queue, output_queue, max_seq_output, min_seq_output):
        """
        Executes the language model on a given prompt and prints the result.
        """
        batch_size = 1

        gen_kwargs = {
            "max_new_tokens": max_seq_output,
            "min_new_tokens": min_seq_output,
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 50,
        }
        sampling_config = SamplingConfig(**gen_kwargs)
        while True:
            PACE_DEBUG(f"Fetching in {self.worker_id}")
            item = input_queue.get()
            if item is None:
                PACE_DEBUG(f"Queue is empty in {self.worker_id}")
                input_queue.task_done()  # mark task as complete
                break
            PACE_DEBUG(f"Processing {item} by {self.worker_id}")
            input_ids = self.dataset.get_batch(batch_size, item * batch_size)[
                "input_ids"
            ]
            input_queue.task_done()  # mark task as complete
            start_time = time.time_ns()
            with suppress_logging():
                outputs = self.model.generate(input_ids, sampling_config)
            end_time = time.time_ns()
            total_time = (end_time - start_time) / (1000 * 1000 * 1000)
            time_tuple = (outputs.speculative_stats.total_speculated_tokens, total_time)
            output_queue.put(time_tuple)


class LLMMultipleProcesses(MultipleProcesses):
    """
    Manages multiple LLMWorker instances across specified processes and cores for
    handling large language models.

    Attributes:
        workers: List of LLMWorker instances.
    """

    def __init__(
        self,
        workers: list,
    ):
        super().__init__(workers)


def main():

    parser = argparse.ArgumentParser(description="Process arguments for this example.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model",
    )
    parser.add_argument(
        "--pard_model_name",
        type=str,
        default="amd/PARD-Llama-3.2-1B",
        help="Path to the pard model",
    )

    parser.add_argument(
        "--dataset_name", type=str, default="gsm8k", help="Path to the dataset"
    )
    parser.add_argument(
        "--instance_count", type=str, default="1", help="Number of Instances"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=str,
        default="256",
        help="Maximum number of tokens to be generated",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=str,
        default="256",
        help="Minimum number of tokens to be generated",
    )
    parser.add_argument(
        "--sample_size",
        type=str,
        default="1",
        help="number of samples to be processed",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="main",
        help="version of the dataset",
    )
    parser.add_argument(
        "--dataset_input_field",
        type=str,
        default="question",
        help="input field of the dataset",
    )

    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    pard_model_name = args.pard_model_name
    num_of_instances = int(args.instance_count)
    max_new_tokens = int(args.max_new_tokens)
    min_new_tokens = int(args.min_new_tokens)
    num_of_samples = int(args.sample_size)
    dataset_version = args.dataset_version
    dataset_input_field = args.dataset_input_field

    # Get the number of physical cores
    physical_cores = psutil.cpu_count(logical=False)

    ip_queue = mp.JoinableQueue()
    op_queue = mp.Queue()
    for i in range(num_of_samples):
        ip_queue.put(i)

    for i in range(num_of_instances):
        ip_queue.put(None)

    init_arg_list = []
    target_arg_list = []

    for i in range(num_of_instances):
        init_arg_list.append(
            (
                model_name,
                pard_model_name,
                dataset_name,
                dataset_version,
                dataset_input_field,
                num_of_samples,
            )
        )
        target_arg_list.append((ip_queue, op_queue, max_new_tokens, min_new_tokens))

    cores_per_instance = physical_cores // num_of_instances
    start_core = 0
    workers = []

    for i in range(num_of_instances):
        end_core: int = start_core + cores_per_instance
        cores_list = list(range(int(start_core), int(end_core)))
        start_core = end_core
        workers.append(
            LLMWorker(
                i,
                cores_list,
                None,
                init_arg_list[i],
                None,
                target_arg_list[i],
            )
        )

    mul_proc = LLMMultipleProcesses(workers)

    dt_start = time.time_ns()
    mul_proc.run()
    mul_proc.join()
    dt_end = time.time_ns()
    PACE_INFO(f"Time taken in sec: {(dt_end - dt_start) / (1000 * 1000 * 1000)}")

    op_queue.put(None)

    total_tokens_gen = 0
    total_time_taken = 0.0

    while True:
        item = op_queue.get()
        if item is None:
            break
        total_tokens_gen += item[0]
        total_time_taken += item[1]

    throughput = (total_tokens_gen / total_time_taken) * num_of_instances

    PACE_INFO(f"Total tokens:  {total_tokens_gen}")
    PACE_INFO(f"Throughput: {throughput} tokens/sec")

    ip_queue.join()


if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mp.set_start_method("spawn")
    main()
