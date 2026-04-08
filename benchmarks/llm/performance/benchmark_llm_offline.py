# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************


import os
import gc
import time
from abc import ABC, abstractmethod
from datetime import datetime
from statistics import fmean
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding

from arguments import get_args
from datastructs import (
    ModelArgs,
    GenerationArgs,
    TokenArgs,
    BenchmarkArgs,
    BenchmarkResults,
    BenchmarkResultsList,
    GeneratorOutput,
    GeneratorOutputAggregator,
    Metrics,
)
from metrics import SystemMonitor, TokenLatencyStreamer, calculate_metrics
from data import BenchMarkDataGenerator
from utils import PACE_LLM_INFO, PACE_LLM_ASSERT, suppress_logging_fn

# @TODO:
# 1. Add support for multiple instances


class BenchmarkOfflineFramework(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, inputs: BatchEncoding) -> GeneratorOutput:
        pass


class HuggingFaceOfflineFramework(BenchmarkOfflineFramework):

    def __init__(
        self,
        model_args: ModelArgs,
        generation_args: GenerationArgs,
        token_args: TokenArgs,
    ):
        from transformers import AutoModelForCausalLM

        # Init model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name, torch_dtype=model_args.dtype
        )

        if generation_args.manual_seed is not None:
            from transformers import set_seed

            set_seed(generation_args.manual_seed)

        self.token_args = token_args
        self.streamer = None
        if token_args != TokenArgs():
            self.streamer = TokenLatencyStreamer()

        self.gen_kwargs = {
            "max_new_tokens": generation_args.output_tokens,
            "min_new_tokens": generation_args.output_tokens,
            "streamer": self.streamer,
            "do_sample": generation_args.do_sample,
        }

    def generate(self, inputs: torch.Tensor) -> GeneratorOutput:
        """
        Generates text using a Hugging Face Transformer model.

        Args:
            inputs (torch.Tensor): Input tensor for the model.

        Returns:
            Tuple[float, float]: Tuple containing total generation time and time to first token
        """

        inputs, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        self.gen_kwargs["attention_mask"] = attention_mask

        start_time = time.perf_counter()
        out = self.model.generate(inputs, **self.gen_kwargs)
        end_time = time.perf_counter()

        ttft = (
            self.streamer.end_time_first_token - start_time
            if self.token_args.time_to_first_token
            else None
        )
        time_per_tokens = (
            self.streamer.time_per_tokens if self.token_args.time_per_tokens else None
        )
        return GeneratorOutput(
            total_time=(end_time - start_time),
            time_per_tokens=time_per_tokens,
            ttft=ttft,
            input_tokens=inputs.shape[1],
            total_tokens=out.shape[1],
        )


class PACEOfflineFramework(BenchmarkOfflineFramework):

    def __init__(
        self,
        model_args: ModelArgs,
        generation_args: GenerationArgs,
        token_args: TokenArgs,
    ):
        from pace.llm import (
            LLMModel,
            SamplingConfig,
            KVCacheType,
            OperatorConfig,
            PardSpecDecodeConfig,
        )

        cache_type_str = generation_args.kv_cache_type.upper()
        if cache_type_str == "BMC":
            kv_cache_type = KVCacheType.BMC
        elif cache_type_str == "SLAB_POOL":
            kv_cache_type = KVCacheType.SLAB_POOL
        elif cache_type_str == "PAGED":
            kv_cache_type = KVCacheType.PAGED
        else:
            kv_cache_type = KVCacheType.DYNAMIC

        spec_config = None
        if model_args.spec_config is not None:
            # If spec_config is provided, use it to create a PardSpecDecodeConfig
            spec_config = PardSpecDecodeConfig(
                model_name_or_path=model_args.spec_config["model_name"],
                num_speculative_tokens=model_args.spec_config["num_speculated_tokens"],
            )
        # Init model
        self.model = LLMModel(
            model_args.model_name,
            dtype=model_args.dtype,
            kv_cache_type=kv_cache_type,
            opconfig=OperatorConfig(**model_args.llm_operators),
            spec_config=spec_config,
        )

        self.token_args = token_args

        self.streamer = None
        if token_args != TokenArgs():
            self.streamer = TokenLatencyStreamer()

        gen_kwargs = {
            "max_new_tokens": generation_args.output_tokens,
            "min_new_tokens": generation_args.output_tokens,
            "streamer": self.streamer,
            "do_sample": generation_args.do_sample,
            "temperature": (torch.rand(1).item() if generation_args.do_sample else 0),
            "seed": generation_args.manual_seed,
        }

        self.sampling_config = SamplingConfig(**gen_kwargs)

    @suppress_logging_fn
    def generate(self, inputs: BatchEncoding) -> GeneratorOutput:
        """
        Generates text using a PACE LLM model.

        Args:
            inputs (torch.Tensor): Input tensor for the model.

        Returns:
            Tuple[float, float]: Tuple containing total generation time and time to first token
        """

        start_time = time.perf_counter()
        out = self.model.generate(
            inputs, self.sampling_config, text_streamer=self.streamer
        )
        end_time = time.perf_counter()

        ttft = (
            self.streamer.end_time_first_token - start_time
            if self.token_args.time_to_first_token
            else None
        )
        time_per_tokens = (
            self.streamer.time_per_tokens if self.token_args.time_per_tokens else None
        )
        return GeneratorOutput(
            total_time=(end_time - start_time),
            ttft=ttft,
            time_per_tokens=time_per_tokens,
            input_tokens=inputs["input_ids"].shape[1],
            total_tokens=out.output_token_ids.shape[1],
            mean_accepted_tokens=(
                out.speculative_stats.mean_accepted_tokens
                if out.speculative_stats
                else None
            ),
        )


class VLLMOfflineFramework(BenchmarkOfflineFramework):

    def __init__(
        self,
        model_args: ModelArgs,
        generation_args: GenerationArgs,
        token_args: TokenArgs,
    ):

        from vllm import LLM, SamplingParams

        # Init model
        self.model = LLM(
            model_args.model_name,
            dtype=model_args.dtype,
            seed=(
                generation_args.manual_seed if generation_args.manual_seed else 1
            ),  # New vLLM (V1) sets default seed to 1 but cannot pass None
            enable_prefix_caching=False,
        )

        gen_kwargs = {
            "max_tokens": generation_args.output_tokens,
            "min_tokens": generation_args.output_tokens,
            "temperature": torch.rand(1).item() if generation_args.do_sample else 1.0,
        }

        self.sampling_params = SamplingParams(**gen_kwargs)
        self.token_args = token_args

    def generate(self, inputs: BatchEncoding) -> GeneratorOutput:
        """
        Generates text using a VLLM model.

        Args:
            inputs (torch.Tensor): Input tensor for the model.

        Returns:
            Tuple[float, float]: Tuple containing total generation time and time to first token
        """
        from vllm.inputs import TokensPrompt

        inputs = inputs["input_ids"]
        prompts = []
        for prompt_token_ids in inputs:
            prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids.tolist()))

        start_time = time.perf_counter()
        out = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        end_time = time.perf_counter()

        # There is no direct way to get the time to first token
        # w.r.to the start_time. So we calculate it using
        # the metrics returned by the model
        ttft = None
        if self.token_args.time_to_first_token:
            ttft = fmean(
                [response.metrics.first_token_time for response in out]
            ) - fmean([response.metrics.arrival_time for response in out])
        output_tokens = torch.tensor(
            [response.outputs[0].token_ids for response in out]
        ).shape[-1]
        return GeneratorOutput(
            total_time=(end_time - start_time),
            ttft=ttft,
            time_per_tokens=[],  # Not available
            input_tokens=inputs.shape[1],
            total_tokens=inputs.shape[1] + output_tokens,
        )


class VLLMZenTorchOfflineFramework(VLLMOfflineFramework):

    def __init__(
        self,
        model_args: ModelArgs,
        generation_args: GenerationArgs,
        token_args: TokenArgs,
    ):
        import zentorch  # noqa: F401 — must be imported before vLLM init so the platform plugin registers

        super().__init__(model_args, generation_args, token_args)


class ZenTorchOfflineFramework(BenchmarkOfflineFramework):

    def __init__(
        self,
        model_args: ModelArgs,
        generation_args: GenerationArgs,
        token_args: TokenArgs,
    ):
        from transformers import AutoModelForCausalLM
        import zentorch

        # Init model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name, torch_dtype=model_args.dtype
        )

        if generation_args.manual_seed is not None:
            from transformers import set_seed

            set_seed(generation_args.manual_seed)

        self.token_args = token_args
        self.streamer = None
        if token_args != TokenArgs():
            self.streamer = TokenLatencyStreamer()

        self.gen_kwargs = {
            "max_new_tokens": generation_args.output_tokens,
            "min_new_tokens": generation_args.output_tokens,
            "streamer": self.streamer,
            "do_sample": generation_args.do_sample,
        }
        self.model = zentorch.llm.optimize(self.model, self.model.dtype)
        self.model.forward = torch.compile(self.model.forward, backend="zentorch")

    def generate(self, inputs: torch.Tensor) -> GeneratorOutput:
        """
        Generates text using a Hugging Face Transformer model.

        Args:
            inputs (torch.Tensor): Input tensor for the model.

        Returns:
            Tuple[float, float]: Tuple containing total generation time and time to first token
        """

        inputs, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        self.gen_kwargs["attention_mask"] = attention_mask

        start_time = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(inputs, **self.gen_kwargs)
        end_time = time.perf_counter()

        ttft = (
            self.streamer.end_time_first_token - start_time
            if self.token_args.time_to_first_token
            else None
        )
        time_per_tokens = (
            self.streamer.time_per_tokens if self.token_args.time_per_tokens else None
        )
        return GeneratorOutput(
            total_time=(end_time - start_time),
            time_per_tokens=time_per_tokens,
            ttft=ttft,
            input_tokens=inputs.shape[1],
            total_tokens=out.shape[1],
        )


def benchmark(
    framework: str,
    model_args: ModelArgs,
    generation_args: GenerationArgs,
    token_args: TokenArgs,
    use_real_data: bool = False,
    num_runs: Optional[int] = 10,
    warmup_runs: Optional[int] = 2,
    verbose: bool = False,
    system_metrics: bool = False,
):
    PACE_LLM_INFO(
        f"Running benchmark for model: {model_args.model_name} using {framework} framework"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:  # Some models don't have a pad token
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input text
    data_gen = BenchMarkDataGenerator(
        tokenizer,
        generation_args.input_tokens,
        generation_args.batch_size,
        max_iter=warmup_runs + num_runs,
        use_real_data=use_real_data,
    )

    # Start system monitor as soon as possible
    # to get the most accurate metrics
    if system_metrics:
        system_monitor = SystemMonitor()
        system_monitor.start()

    if framework == "hf":
        model = HuggingFaceOfflineFramework(
            model_args,
            generation_args,
            token_args,
        )
    elif framework == "pace":
        model = PACEOfflineFramework(
            model_args,
            generation_args,
            token_args,
        )
    elif framework == "vllm":
        model = VLLMOfflineFramework(
            model_args,
            generation_args,
            token_args,
        )
    elif framework == "vllm_zentorch":
        model = VLLMZenTorchOfflineFramework(
            model_args,
            generation_args,
            token_args,
        )
    elif framework == "zentorch":
        model = ZenTorchOfflineFramework(
            model_args,
            generation_args,
            token_args,
        )
    else:
        PACE_LLM_ASSERT(False, f"Unsupported framework: {framework}")

    # Warm-up runs
    PACE_LLM_INFO("Performing warm-up runs...")
    for _ in tqdm(range(warmup_runs), desc="Warming up"):
        model.generate(next(data_gen))
    PACE_LLM_INFO("Warm-up complete.")

    generator_outputs = GeneratorOutputAggregator(token_args)
    PACE_LLM_INFO("Starting timed runs...")
    for i in tqdm(range(num_runs), desc="Benchmarking"):
        inputs = next(data_gen)
        generation_output: GeneratorOutput = model.generate(inputs)
        generator_outputs.append(generation_output)
    PACE_LLM_INFO("Timed runs complete.")

    if system_metrics:
        system_monitor.stop()

    # Calculate metrics
    metrics: Metrics = calculate_metrics(
        batch_size=generation_args.batch_size,
        generator_outputs=generator_outputs,
        num_runs=num_runs,
    )

    benchmark_results = BenchmarkResults(
        framework=framework,
        model_args=model_args,
        generation_args=generation_args,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        metrics=metrics,
        system_metrics=system_monitor.get_history() if system_metrics else None,
    )

    if verbose:
        PACE_LLM_INFO(f"Benchmark results: {benchmark_results}")

    del model
    gc.collect()

    time.sleep(5)  # To allow garbage collection to complete
    return benchmark_results


def main():

    args: BenchmarkArgs = get_args()

    # Add more when needed
    benchmark_results_list = BenchmarkResultsList()
    for framework in args.frameworks:

        benchmark_results: BenchmarkResults = benchmark(
            framework,
            args.model_args,
            args.generation_args,
            args.token_args,
            args.use_real_data,
            args.num_runs,
            args.warmup_runs,
            args.verbose,
            args.system_metrics,
        )
        benchmark_results_list.append(benchmark_results)

    if args.output_dir:
        frameworks_tag = "-".join(args.frameworks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_prefix = os.path.join(
            args.output_dir,
            f"{frameworks_tag}_{args.model_args.model_name.replace('/', '--')}"
            f"_{args.model_args.dtype}_bs{args.generation_args.batch_size}"
            f"_it{args.generation_args.input_tokens}_nt{args.generation_args.output_tokens}"
            f"_{timestamp}",
        )
        output_file = f"{output_file_prefix}_results.json"
        benchmark_results_list.dump(output_file)
        PACE_LLM_INFO(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
