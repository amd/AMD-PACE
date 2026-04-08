# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
import time
import torch
from unittest import mock
from typing import Optional

import pace  # noqa: F401
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase
from transformers import AutoTokenizer
from pace.utils.logging import suppress_logging_cls
from pace.llm import KVCacheType

from data import BenchMarkDataGenerator
from datastructs import (
    ModelArgs,
    GenerationArgs,
    TokenArgs,
    GeneratorOutput,
    GeneratorOutputAggregator,
    Metrics,
    SystemMetrics,
    BenchmarkResults,
    BenchmarkResultsList,
)
from metrics import (
    TokenLatencyStreamer,
    FirstTokenTimerStreamer,
    SystemMonitor,
)


@suppress_logging_cls()
class TestBenchMarkDataGenerator(TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    @settings(max_examples=5, deadline=None)
    @given(
        input_tokens=st.integers(min_value=1, max_value=1920),
        batch_size=st.integers(min_value=1, max_value=128),
        use_real_data=st.booleans(),
    )
    def test_data_generator_init(
        self, input_tokens: int, batch_size: int, use_real_data: bool
    ):
        """
        Test that the data generator initializes correctly.
        """

        data_generator = BenchMarkDataGenerator(
            tokenizer=self.tokenizer,
            input_tokens=input_tokens,
            batch_size=batch_size,
            use_real_data=use_real_data,
        )
        self.assertIsNotNone(data_generator)

    @settings(max_examples=5, deadline=None)
    @given(
        input_tokens=st.integers(min_value=1, max_value=1920),
        batch_size=st.integers(min_value=1, max_value=128),
    )
    def test_send_method(self, input_tokens: int, batch_size: int):
        """
        Test the send method of the data generator.
        """

        data_generator = BenchMarkDataGenerator(
            tokenizer=self.tokenizer, input_tokens=input_tokens, batch_size=batch_size
        )
        data = data_generator.send(None)
        self.assertIn("input_ids", data)
        self.assertIn("attention_mask", data)
        self.assertEqual(data["input_ids"].shape, (batch_size, input_tokens))
        self.assertEqual(data["attention_mask"].shape, (batch_size, input_tokens))

    @settings(max_examples=5, deadline=None)
    @given(
        input_tokens=st.integers(min_value=1, max_value=1920),
        batch_size=st.integers(min_value=1, max_value=128),
        use_real_data=st.booleans(),
        max_iter=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    )
    def test_real_data_shape(
        self,
        input_tokens: int,
        batch_size: int,
        use_real_data: bool,
        max_iter: Optional[int],
    ):
        """
        Test that the real data input shape is correct.
        """

        data_generator = BenchMarkDataGenerator(
            tokenizer=self.tokenizer,
            input_tokens=input_tokens,
            batch_size=batch_size,
            use_real_data=use_real_data,
            max_iter=max_iter,
        )
        data = data_generator.send(None)
        self.assertEqual(data["input_ids"].shape, (batch_size, input_tokens))
        self.assertEqual(data["attention_mask"].shape, (batch_size, input_tokens))

    @given(
        input_tokens=st.integers(min_value=1, max_value=1920),
        batch_size=st.integers(min_value=1, max_value=128),
        use_real_data=st.booleans(),
    )
    @settings(max_examples=5, deadline=None)
    def test_data_in_memory_not_empty(
        self, input_tokens: int, batch_size: int, use_real_data: bool
    ):
        """
        Test that the data_in_memory is not empty after initialization.
        """
        data_generator = BenchMarkDataGenerator(
            tokenizer=self.tokenizer,
            input_tokens=input_tokens,
            batch_size=batch_size,
            use_real_data=use_real_data,
        )
        self.assertTrue(len(data_generator.data_in_memory) > 0)


@suppress_logging_cls()
class TestDataStructs(TestCase):

    def setUp(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        pass

    @given(
        total_time=st.floats(min_value=0.0),
        input_tokens=st.integers(min_value=0),
        total_tokens=st.integers(min_value=0),
        ttft=st.one_of(st.none(), st.floats(min_value=0.0)),
        time_per_tokens=st.one_of(st.none(), st.lists(st.floats(min_value=0.0))),
        mean_accepted_tokens=st.one_of(st.none(), st.floats(min_value=0.0)),
    )
    def test_generator_output(
        self,
        total_time: float,
        input_tokens: int,
        total_tokens: int,
        ttft: Optional[float],
        time_per_tokens: Optional[list[float]],
        mean_accepted_tokens: Optional[float],
    ):
        generator_output = GeneratorOutput(
            total_time=total_time,
            input_tokens=input_tokens,
            total_tokens=total_tokens,
            ttft=ttft,
            time_per_tokens=time_per_tokens,
            mean_accepted_tokens=mean_accepted_tokens,
        )

        self.assertEqual(generator_output.total_time, total_time)
        self.assertEqual(generator_output.input_tokens, input_tokens)
        self.assertEqual(generator_output.total_tokens, total_tokens)
        self.assertEqual(generator_output.ttft, ttft)
        self.assertEqual(generator_output.time_per_tokens, time_per_tokens)
        self.assertEqual(generator_output.mean_accepted_tokens, mean_accepted_tokens)

    def test_generator_output_aggregator(self):

        token_args = TokenArgs(time_per_tokens=True, time_to_first_token=True)
        aggregator = GeneratorOutputAggregator(token_args)

        output1 = GeneratorOutput(
            total_time=1.0,
            input_tokens=10,
            total_tokens=20,
            ttft=0.5,
            time_per_tokens=[0.1] * 10,
            mean_accepted_tokens=[0.5] * 10,
        )
        output2 = GeneratorOutput(
            total_time=2.0,
            input_tokens=20,
            total_tokens=30,
            ttft=0.6,
            time_per_tokens=[0.2] * 10,
            mean_accepted_tokens=[0.6] * 10,
        )

        aggregator.append(output1)
        aggregator.append(output2)

        self.assertEqual(aggregator.total_generation_times, [1.0, 2.0])
        self.assertEqual(aggregator.generated_tokens_count, 20)
        self.assertEqual(aggregator.total_tokens_count, 50)
        self.assertEqual(aggregator.ttft_times, [0.5, 0.6])
        self.assertEqual(aggregator.time_per_tokens, [[0.1] * 10, [0.2] * 10])
        self.assertEqual(aggregator.mean_accepted_tokens, [[0.5] * 10, [0.6] * 10])

        token_args = TokenArgs(time_per_tokens=False, time_to_first_token=False)
        aggregator = GeneratorOutputAggregator(token_args)

        output1 = GeneratorOutput(
            total_time=1.0,
            input_tokens=10,
            total_tokens=20,
            ttft=0.5,
            time_per_tokens=[0.1] * 10,
        )
        output2 = GeneratorOutput(
            total_time=2.0,
            input_tokens=20,
            total_tokens=30,
            ttft=0.6,
            time_per_tokens=[0.2] * 10,
        )

        aggregator.append(output1)
        aggregator.append(output2)

        self.assertEqual(aggregator.total_generation_times, [1.0, 2.0])
        self.assertEqual(aggregator.generated_tokens_count, 20)
        self.assertEqual(aggregator.total_tokens_count, 50)
        self.assertEqual(aggregator.ttft_times, [])
        self.assertEqual(aggregator.time_per_tokens, [])
        self.assertEqual(aggregator.mean_accepted_tokens, [])

    @given(
        total_tokens=st.integers(min_value=0),
        total_gen_tokens=st.integers(min_value=0),
        average_gen_time=st.floats(min_value=0.0),
        average_latency_per_token=st.floats(min_value=0.0),
        total_tps=st.floats(min_value=0.0),
        output_tps=st.floats(min_value=0.0),
        average_ttft=st.one_of(st.none(), st.floats(min_value=0.0)),
        time_per_tokens=st.one_of(st.none(), st.lists(st.floats(min_value=0.0))),
        mean_accepted_tokens=st.one_of(st.none(), st.lists(st.floats(min_value=0.0))),
    )
    def test_metrics(
        self,
        total_tokens: int,
        total_gen_tokens: int,
        average_gen_time: float,
        average_latency_per_token: float,
        total_tps: float,
        output_tps: float,
        average_ttft: Optional[float],
        time_per_tokens: Optional[list[float]],
        mean_accepted_tokens: Optional[list[float]],
    ):
        metrics = Metrics(
            total_tokens=total_tokens,
            total_gen_tokens=total_gen_tokens,
            average_gen_time=average_gen_time,
            average_latency_per_token=average_latency_per_token,
            total_tps=total_tps,
            output_tps=output_tps,
            average_ttft=average_ttft,
            time_per_tokens=time_per_tokens,
            mean_accepted_tokens=mean_accepted_tokens,
        )

        self.assertEqual(metrics.total_tokens, total_tokens)
        self.assertEqual(metrics.total_gen_tokens, total_gen_tokens)
        self.assertEqual(metrics.average_gen_time, average_gen_time)
        self.assertEqual(metrics.average_latency_per_token, average_latency_per_token)
        self.assertEqual(metrics.total_tps, total_tps)
        self.assertEqual(metrics.output_tps, output_tps)
        self.assertEqual(metrics.average_ttft, average_ttft)
        self.assertEqual(metrics.time_per_tokens, time_per_tokens)
        self.assertEqual(metrics.mean_accepted_tokens, mean_accepted_tokens)

        metrics_dict = metrics.to_dict()
        self.assertEqual(metrics_dict["total_tokens"], total_tokens)
        self.assertEqual(metrics_dict["total_gen_tokens"], total_gen_tokens)
        self.assertEqual(metrics_dict["average_gen_time"], average_gen_time)
        self.assertEqual(
            metrics_dict["average_latency_per_token"], average_latency_per_token
        )
        self.assertEqual(metrics_dict["total_tps"], total_tps)
        self.assertEqual(metrics_dict["output_tps"], output_tps)
        self.assertEqual(metrics_dict["average_ttft"], average_ttft)
        self.assertEqual(metrics_dict["time_per_tokens"], time_per_tokens)
        self.assertEqual(metrics_dict["mean_accepted_tokens"], mean_accepted_tokens)

    @given(
        interval=st.floats(min_value=0.0),
        cpu_usage=st.lists(st.floats(min_value=0.0)),
        ram_usage=st.lists(st.floats(min_value=0.0)),
        peak_ram_usage=st.floats(min_value=0.0),
    )
    def test_system_metrics(
        self,
        interval: float,
        cpu_usage: list[float],
        ram_usage: list[float],
        peak_ram_usage: float,
    ):
        system_metrics = SystemMetrics(
            interval=interval,
            cpu_usage=cpu_usage,
            ram_usage=ram_usage,
            peak_ram_usage=peak_ram_usage,
        )

        self.assertEqual(system_metrics.interval, interval)
        self.assertEqual(system_metrics.cpu_usage, cpu_usage)
        self.assertEqual(system_metrics.ram_usage, ram_usage)
        self.assertEqual(system_metrics.peak_ram_usage, peak_ram_usage)

        system_metrics_dict = system_metrics.to_dict()
        self.assertEqual(system_metrics_dict["interval"], interval)
        self.assertEqual(system_metrics_dict["cpu_usage"], cpu_usage)
        self.assertEqual(system_metrics_dict["ram_usage"], ram_usage)
        self.assertEqual(system_metrics_dict["peak_ram_usage"], peak_ram_usage)

    @given(
        framework=st.text(),
        model_name=st.text(),
        batch_size=st.integers(min_value=1),
        input_tokens=st.integers(min_value=0),
        output_tokens=st.integers(min_value=0),
        num_runs=st.integers(min_value=1),
        warmup_runs=st.integers(min_value=0),
        total_tokens=st.integers(min_value=0),
        total_gen_tokens=st.integers(min_value=0),
        average_gen_time=st.floats(min_value=0.0),
        average_latency_per_token=st.floats(min_value=0.0),
        total_tps=st.floats(min_value=0.0),
        output_tps=st.floats(min_value=0.0),
        gen_kwargs=st.dictionaries(keys=st.text(), values=st.text()),
        interval=st.floats(min_value=0.0),
        cpu_usage=st.lists(st.floats(min_value=0.0)),
        ram_usage=st.lists(st.floats(min_value=0.0)),
        peak_ram_usage=st.floats(min_value=0.0),
    )
    def test_benchmark_results(
        self,
        framework: str,
        model_name: str,
        batch_size: int,
        input_tokens: int,
        output_tokens: int,
        num_runs: int,
        warmup_runs: int,
        total_tokens: int,
        total_gen_tokens: int,
        average_gen_time: float,
        average_latency_per_token: float,
        total_tps: float,
        output_tps: float,
        gen_kwargs: dict,
        interval: float,
        cpu_usage: list[float],
        ram_usage: list[float],
        peak_ram_usage: float,
    ):
        metrics = Metrics(
            total_tokens=total_tokens,
            total_gen_tokens=total_gen_tokens,
            average_gen_time=average_gen_time,
            average_latency_per_token=average_latency_per_token,
            total_tps=total_tps,
            output_tps=output_tps,
        )

        system_metrics = SystemMetrics(
            interval=interval,
            cpu_usage=cpu_usage,
            ram_usage=ram_usage,
            peak_ram_usage=peak_ram_usage,
        )

        benchmark_results = BenchmarkResults(
            framework=framework,
            model_args=ModelArgs(model_name=model_name, dtype=torch.bfloat16),
            generation_args=GenerationArgs(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_size=batch_size,
                do_sample=False,
                manual_seed=42,
                kv_cache_type=KVCacheType.BMC,
            ),
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            metrics=metrics,
            system_metrics=system_metrics,
        )

        self.assertEqual(benchmark_results.framework, framework)
        self.assertEqual(benchmark_results.model_args.model_name, model_name)
        self.assertEqual(benchmark_results.model_args.dtype, torch.bfloat16)
        self.assertEqual(benchmark_results.generation_args.input_tokens, input_tokens)
        self.assertEqual(benchmark_results.generation_args.output_tokens, output_tokens)
        self.assertEqual(benchmark_results.generation_args.batch_size, batch_size)
        self.assertEqual(benchmark_results.generation_args.do_sample, False)
        self.assertEqual(benchmark_results.generation_args.manual_seed, 42)
        self.assertEqual(benchmark_results.num_runs, num_runs)
        self.assertEqual(benchmark_results.warmup_runs, warmup_runs)
        self.assertEqual(benchmark_results.metrics, metrics)
        self.assertEqual(benchmark_results.system_metrics, system_metrics)

        benchmark_results_dict = benchmark_results.to_dict()
        self.assertEqual(benchmark_results_dict["framework"], framework)
        self.assertEqual(benchmark_results_dict["model_args"]["model_name"], model_name)
        self.assertEqual(
            benchmark_results_dict["model_args"]["dtype"], str(torch.bfloat16)
        )
        self.assertEqual(
            benchmark_results_dict["generation_args"]["input_tokens"], input_tokens
        )
        self.assertEqual(
            benchmark_results_dict["generation_args"]["output_tokens"], output_tokens
        )
        self.assertEqual(
            benchmark_results_dict["generation_args"]["batch_size"], batch_size
        )
        self.assertEqual(benchmark_results_dict["generation_args"]["do_sample"], False)
        self.assertEqual(benchmark_results_dict["generation_args"]["manual_seed"], 42)
        self.assertEqual(benchmark_results_dict["num_runs"], num_runs)
        self.assertEqual(benchmark_results_dict["warmup_runs"], warmup_runs)
        self.assertEqual(benchmark_results_dict["metrics"], metrics.to_dict())
        self.assertEqual(
            benchmark_results_dict["system_metrics"], system_metrics.to_dict()
        )

    def test_benchmark_results_list(self):
        benchmark_results_list = BenchmarkResultsList()

        model_args = ModelArgs(model_name="test_model", dtype=torch.bfloat16)
        generation_args = GenerationArgs(
            input_tokens=50,
            output_tokens=50,
            batch_size=1,
            do_sample=False,
            manual_seed=42,
            kv_cache_type=KVCacheType.BMC,
        )

        # Create some dummy data for BenchmarkResults
        metrics1 = Metrics(
            total_tokens=100,
            total_gen_tokens=50,
            average_gen_time=0.1,
            average_latency_per_token=0.2,
            total_tps=1000.0,
            output_tps=500.0,
        )
        system_metrics1 = SystemMetrics(
            interval=1.0, cpu_usage=[50.0], ram_usage=[2048.0], peak_ram_usage=4096.0
        )

        benchmark_results1 = BenchmarkResults(
            framework="hf",
            model_args=model_args,
            generation_args=generation_args,
            num_runs=10,
            warmup_runs=2,
            metrics=metrics1,
            system_metrics=system_metrics1,
        )

        metrics2 = Metrics(
            total_tokens=200,
            total_gen_tokens=100,
            average_gen_time=0.2,
            average_latency_per_token=0.4,
            total_tps=2000.0,
            output_tps=1000.0,
        )
        system_metrics2 = SystemMetrics(
            interval=2.0, cpu_usage=[60.0], ram_usage=[3072.0], peak_ram_usage=6144.0
        )

        benchmark_results2 = BenchmarkResults(
            framework="pace",
            model_args=model_args,
            generation_args=generation_args,
            num_runs=10,
            warmup_runs=2,
            metrics=metrics2,
            system_metrics=system_metrics2,
        )

        # Append the BenchmarkResults objects
        benchmark_results_list.append(benchmark_results1)
        benchmark_results_list.append(benchmark_results2)

        # Assert that the lists are populated correctly
        self.assertEqual(len(benchmark_results_list.results), 2)
        self.assertEqual(benchmark_results_list.results[0], benchmark_results1)
        self.assertEqual(benchmark_results_list.results[1], benchmark_results2)

        # Convert to dictionary and check
        results_dict = benchmark_results_list.to_dict()
        self.assertEqual(results_dict["model_args"]["model_name"], "test_model")
        self.assertEqual(results_dict["model_args"]["dtype"], str(torch.bfloat16))
        self.assertEqual(results_dict["generation_args"]["input_tokens"], 50)
        self.assertEqual(results_dict["generation_args"]["output_tokens"], 50)
        self.assertEqual(results_dict["generation_args"]["batch_size"], 1)
        self.assertEqual(results_dict["generation_args"]["do_sample"], False)
        self.assertEqual(results_dict["generation_args"]["manual_seed"], 42)
        self.assertEqual(results_dict["num_runs"], 10)
        self.assertEqual(results_dict["warmup_runs"], 2)

    def test_benchmark_results_list_append_error(self):
        benchmark_results_list = BenchmarkResultsList()

        model_args1 = ModelArgs(model_name="test_model", dtype=torch.bfloat16)
        generation_args1 = GenerationArgs(
            input_tokens=50,
            output_tokens=50,
            batch_size=1,
            do_sample=False,
            manual_seed=42,
            kv_cache_type=KVCacheType.BMC,
        )

        # Create some dummy data for BenchmarkResults
        metrics1 = Metrics(
            total_tokens=100,
            total_gen_tokens=50,
            average_gen_time=0.1,
            average_latency_per_token=0.2,
            total_tps=1000.0,
            output_tps=500.0,
        )
        system_metrics1 = SystemMetrics(
            interval=1.0, cpu_usage=[50.0], ram_usage=[2048.0], peak_ram_usage=4096.0
        )

        benchmark_results1 = BenchmarkResults(
            framework="hf",
            model_args=model_args1,
            generation_args=generation_args1,
            num_runs=10,
            warmup_runs=2,
            metrics=metrics1,
            system_metrics=system_metrics1,
        )

        # Create a second BenchmarkResults object with different framework
        # and model_args to trigger the assertion error
        model_args2 = ModelArgs(model_name="different_model", dtype=torch.bfloat16)
        generation_args2 = GenerationArgs(
            input_tokens=50,
            output_tokens=50,
            batch_size=1,
            do_sample=False,
            manual_seed=42,
            kv_cache_type=KVCacheType.BMC,
        )

        # Create some dummy data for the second BenchmarkResults
        metrics2 = Metrics(
            total_tokens=200,
            total_gen_tokens=100,
            average_gen_time=0.2,
            average_latency_per_token=0.4,
            total_tps=2000.0,
            output_tps=1000.0,
        )
        system_metrics2 = SystemMetrics(
            interval=2.0, cpu_usage=[60.0], ram_usage=[3072.0], peak_ram_usage=6144.0
        )
        benchmark_results2 = BenchmarkResults(
            framework="pace",
            model_args=model_args2,
            generation_args=generation_args2,
            num_runs=10,
            warmup_runs=2,
            metrics=metrics2,
            system_metrics=system_metrics2,
        )

        # Append the BenchmarkResults objects
        benchmark_results_list.append(benchmark_results1)
        with self.assertRaises(AssertionError):
            benchmark_results_list.append(benchmark_results2)


@suppress_logging_cls()
class TestMetrics(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    def test_first_token_timer_streamer(self):
        streamer = FirstTokenTimerStreamer(tokenizer=self.tokenizer)
        streamer.put([1])
        self.assertIsNone(getattr(streamer, "end_time_first_token", None))
        streamer.put([2])
        self.assertIsNotNone(getattr(streamer, "end_time_first_token", None))
        streamer.end()
        self.assertEqual(streamer.has_generated_tokens, 2)

    def test_token_latency_streamer(self):
        streamer = TokenLatencyStreamer(tokenizer=self.tokenizer)
        streamer.put([1])
        self.assertIsNone(streamer.end_time_first_token)
        streamer.put([2])
        self.assertIsNotNone(streamer.end_time_first_token)
        self.assertEqual(len(streamer.time_per_tokens), 1)
        streamer.end()
        self.assertIsNone(streamer.start_time)

    @mock.patch("metrics.psutil.virtual_memory")
    @mock.patch("metrics.psutil.cpu_percent")
    def test_system_monitor(self, mock_cpu_percent, mock_virtual_memory):
        # Mock psutil functions
        mock_virtual_memory.return_value.used = 1024 * 1024 * 100  # 100 MB
        mock_cpu_percent.return_value = 50.0

        monitor = SystemMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        self.assertTrue(len(monitor.ram_data_history) > 0)
        self.assertTrue(len(monitor.cpu_data_history) > 0)
        self.assertIsInstance(monitor.get_history(), SystemMetrics)
        self.assertEqual(monitor.get_history().interval, 0.1)
        self.assertEqual(
            monitor.get_history().peak_ram_usage, max(monitor.ram_data_history)
        )

        # Test start and stop when not running
        monitor.stop()
        monitor.start()
        monitor.start()  # start when already running
        time.sleep(0.3)
        monitor.stop()
        monitor.stop()  # stop when already stopped
