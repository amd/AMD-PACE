# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy
import json
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from utils import PACE_LLM_ASSERT


class BaseData:

    def to_dict(self):
        return vars(self)

    def __str__(self):
        # remove None values
        dict_val = self.to_dict()
        dict_val = {k: v for k, v in dict_val.items() if v is not None}

        return json.dumps(dict_val, indent=4)

    def __repr__(self):
        return str(self)

    def dump(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


@dataclass
class ModelArgs(BaseData):
    model_name: str
    dtype: torch.dtype
    llm_operators: Optional[dict] = None
    spec_config: Optional[dict] = None

    def to_dict(self):
        # dtype is not always serializable, convert to str
        dict_val = copy.deepcopy(vars(self))
        dict_val["dtype"] = str(dict_val["dtype"])
        return dict_val


@dataclass
class GenerationArgs(BaseData):
    input_tokens: int
    output_tokens: int
    batch_size: int
    kv_cache_type: str
    do_sample: bool
    manual_seed: int


@dataclass
class TokenArgs(BaseData):
    time_to_first_token: bool = False
    time_per_tokens: bool = False


@dataclass
class BenchmarkArgs(BaseData):
    frameworks: List[str]
    model_args: ModelArgs
    generation_args: GenerationArgs
    use_real_data: bool
    num_runs: int
    warmup_runs: int
    verbose: bool
    output_dir: str
    token_args: TokenArgs
    system_metrics: bool


@dataclass
class GeneratorOutput:
    total_time: float
    input_tokens: int
    total_tokens: int
    ttft: Optional[float]
    time_per_tokens: Optional[List[float]] = None
    mean_accepted_tokens: Optional[float] = None


class GeneratorOutputAggregator:

    def __init__(self, token_args: TokenArgs = TokenArgs()):
        self.total_generation_times = []
        self.generated_tokens_count = 0
        self.total_tokens_count = 0

        # If token_metrics is True, we will store the time taken to generate each token
        self.token_args = token_args
        self.ttft_times = []
        self.time_per_tokens = []
        self.mean_accepted_tokens = []

    def append(self, generation_output: GeneratorOutput):
        assert isinstance(
            generation_output, GeneratorOutput
        ), "generation_output must be an instance of GeneratorOutput"

        self.total_generation_times.append(generation_output.total_time)
        # Count newly generated tokens
        self.generated_tokens_count += (
            generation_output.total_tokens - generation_output.input_tokens
        )
        # Count total tokens generated
        self.total_tokens_count += generation_output.total_tokens

        if self.token_args.time_to_first_token:
            self.ttft_times.append(generation_output.ttft)
        if self.token_args.time_per_tokens:
            self.time_per_tokens.append(generation_output.time_per_tokens)
        if generation_output.mean_accepted_tokens is not None:
            self.mean_accepted_tokens.append(generation_output.mean_accepted_tokens)


@dataclass
class Metrics(BaseData):
    total_tokens: int
    total_gen_tokens: int
    average_gen_time: float
    average_latency_per_token: float
    total_tps: float
    output_tps: float
    average_ttft: Optional[float] = None
    time_per_tokens: Optional[List[float]] = None
    mean_accepted_tokens: Optional[float] = None


@dataclass
class SystemMetrics(BaseData):
    interval: float
    cpu_usage: List[float]
    ram_usage: List[float]
    peak_ram_usage: float


@dataclass
class BenchmarkResults(BaseData):
    framework: str
    model_args: ModelArgs
    generation_args: GenerationArgs
    num_runs: int
    warmup_runs: int
    metrics: Metrics
    system_metrics: Optional[SystemMetrics] = None

    def to_dict(self):
        return {
            "framework": self.framework,
            "model_args": self.model_args.to_dict(),
            "generation_args": self.generation_args.to_dict(),
            "num_runs": self.num_runs,
            "warmup_runs": self.warmup_runs,
            "metrics": self.metrics.to_dict(),
            "system_metrics": (
                self.system_metrics.to_dict() if self.system_metrics else None
            ),
        }


@dataclass
class BenchmarkResultsList(BaseData):
    """A class to aggregate and hold a list of BenchmarkResults."""

    results: List[BenchmarkResults] = field(default_factory=list)

    def append(self, benchmark_results: BenchmarkResults):
        if self.results:
            first_result = self.results[0]
            PACE_LLM_ASSERT(
                first_result.model_args == benchmark_results.model_args,
                "Cannot append BenchmarkResults with different model_args",
            )
            PACE_LLM_ASSERT(
                first_result.generation_args == benchmark_results.generation_args,
                "Cannot append BenchmarkResults with different generation_args",
            )
            PACE_LLM_ASSERT(
                first_result.num_runs == benchmark_results.num_runs,
                "Cannot append BenchmarkResults with different num_runs",
            )
            PACE_LLM_ASSERT(
                first_result.warmup_runs == benchmark_results.warmup_runs,
                "Cannot append BenchmarkResults with different warmup_runs",
            )
        self.results.append(benchmark_results)

    def to_dict(self):
        if not self.results:
            return {}

        # For the common cases, only add them in the dict once
        return {
            "model_args": self.results[0].model_args.to_dict(),
            "generation_args": self.results[0].generation_args.to_dict(),
            "num_runs": self.results[0].num_runs,
            "warmup_runs": self.results[0].warmup_runs,
            "benchmark_results": [
                {
                    "framework": res.framework,
                    "metrics": res.metrics.to_dict(),
                    "system_metrics": (
                        res.system_metrics.to_dict() if res.system_metrics else None
                    ),
                }
                for res in self.results
            ],
        }
