# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import threading
import time
from typing import Dict, List, Optional

import psutil
import torch
from transformers.generation.streamers import BaseStreamer

from datastructs import GeneratorOutputAggregator, Metrics, SystemMetrics
from utils import PACE_LLM_ASSERT, PACE_LLM_INFO, PACE_LLM_WARNING


class FirstTokenTimerStreamer(BaseStreamer):
    """
    Streamer to calculate the time taken for the first token to be generated.

    When the put method is called for the first time, the prompt is being put, but
    the second time the put method is called, the first token is being put. So, we can
    calculate the time taken for the first token to be generated.
    """

    def __init__(self, tokenizer=None):
        super().__init__()

        self.has_generated_tokens = 2

    def put(self, value):
        if not self.has_generated_tokens:  # escape if the tokens are already generated
            return

        if self.has_generated_tokens == 1:
            self.end_time_first_token = time.perf_counter()
        self.has_generated_tokens -= 1

    def end(self):
        self.has_generated_tokens = 2

    def on_finalized_text(self, text, stream_end=False):
        pass  # We are not interested in the text itself in this streamer


class TokenLatencyStreamer(BaseStreamer):
    """
    Streamer to calculate the time taken for each token to be generated.

    Each time the put method is called, the time taken for the token to be generated is calculated
    using the time difference between the current time and the time when the previous token was generated.
    """

    def __init__(self, tokenizer=None):
        super().__init__()
        PACE_LLM_WARNING(
            "Using TokenLatencyStreamer for monitoring token generation times, this might affect the performance,"
            + "as it uses some system resources to monitor the token generation times."
            + " PLEASE DISABLE FOR ABSOLUTE PERFORMANCE BENCHMARKING"
        )
        self.start_time = None
        self.end_time = None
        self.time_per_tokens = []
        self.end_time_first_token = None  # special case for the first token

    def put(self, value):
        if self.start_time is None:
            self.time_per_tokens = []
            self.start_time = time.perf_counter()
            return

        if len(self.time_per_tokens) == 0:
            self.end_time_first_token = time.perf_counter()

        self.end_time = time.perf_counter()
        self.time_per_tokens.append(self.end_time - self.start_time)
        self.start_time = self.end_time

    def end(self):
        self.start_time = None

    def on_finalized_text(self, text, stream_end=False):
        pass  # We are not interested in the text itself in this streamer


def calculate_metrics(
    batch_size: int,
    generator_outputs: GeneratorOutputAggregator,
    num_runs: int,
) -> Metrics:

    ttft_times = generator_outputs.ttft_times
    total_generation_times = generator_outputs.total_generation_times
    time_per_tokens = generator_outputs.time_per_tokens
    total_tokens_count = generator_outputs.total_tokens_count
    generated_tokens_count = generator_outputs.generated_tokens_count
    mean_accepted_tokens = generator_outputs.mean_accepted_tokens

    PACE_LLM_ASSERT(
        len(ttft_times) == 0 or len(ttft_times) == num_runs,
        "Number of TTFT times does not match the number of runs.",
    )
    PACE_LLM_ASSERT(
        len(total_generation_times) == num_runs,
        "Number of total generation times does not match the number of runs.",
    )

    average_ttft = sum(ttft_times) / len(ttft_times) if ttft_times else None
    average_gen_time = sum(total_generation_times) / len(total_generation_times)

    total_output_tokens = batch_size * total_tokens_count
    total_gen_output_tokens = batch_size * generated_tokens_count

    average_latency_per_token = sum(total_generation_times) / total_output_tokens
    total_tps = total_output_tokens / sum(total_generation_times)
    output_tps = total_gen_output_tokens / sum(total_generation_times)

    # Calculate the per token time by averaging the per token times across all runs
    time_per_tokens = (
        torch.tensor(time_per_tokens).mean(dim=0).tolist() if time_per_tokens else None
    )

    # Calculate the mean accepted tokens
    if mean_accepted_tokens:
        mean_accepted_tokens = (
            torch.tensor(mean_accepted_tokens).mean().item()
            if isinstance(mean_accepted_tokens, list)
            else mean_accepted_tokens
        )
    else:
        mean_accepted_tokens = None

    return Metrics(
        total_tokens=total_output_tokens,
        total_gen_tokens=total_gen_output_tokens,
        average_gen_time=average_gen_time,
        average_latency_per_token=average_latency_per_token,
        total_tps=total_tps,
        output_tps=output_tps,
        average_ttft=average_ttft,
        time_per_tokens=time_per_tokens,
        mean_accepted_tokens=mean_accepted_tokens,
    )


class SystemMonitor:
    """
    Captures real-time CPU and RAM usage in a background thread.
    """

    def __init__(self, interval: Optional[int] = 1):
        """
        Initializes the CPU and RAM monitor.

        Args:
            interval (int): Time interval (in seconds)
        """
        PACE_LLM_WARNING(
            "Using SystemMonitor for monitoring system resources, this might affect the performance,"
            + "as it uses some system resources to monitor the CPU and RAM usage."
            + " PLEASE DISABLE FOR ABSOLUTE PERFORMANCE BENCHMARKING"
        )
        self.interval = interval
        PACE_LLM_INFO(f"SystemMonitor initialized with interval: {interval} seconds.")
        self._stop_event = threading.Event()
        self._thread = None
        self.ram_data_history: List[Dict] = []
        self.cpu_data_history: List[Dict] = []

        self.initial_ram_usage = self.get_ram_usage()

    def get_ram_usage(self) -> float:
        """
        Retrieves current RAM usage statistics using psutil.

        Returns:
            float: The amount of RAM used (in bytes)
        """
        vm = psutil.virtual_memory()
        return vm.used

    def get_cpu_usage(self) -> float:
        """
        Retrieves current CPU usage statistics using psutil.

        Returns:
            float: The CPU usage percentage
        """
        return psutil.cpu_percent()

    def _monitor_loop(self):
        """
        The main monitoring loop that runs in a background thread.
        """
        while not self._stop_event.is_set():
            ram_usage = self.get_ram_usage()
            self.ram_data_history.append(ram_usage)
            cpu_usage = self.get_cpu_usage()
            self.cpu_data_history.append(cpu_usage)

            time.sleep(self.interval)

    def start(self):
        """
        Starts the CPU and RAM monitoring in a background thread.
        """
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()  # Reset stop event
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = (
                True  # Allow main program to exit even if thread is running
            )
            self._thread.start()
            PACE_LLM_INFO("SystemMonitor started in the background.")
        else:
            PACE_LLM_INFO("SystemMonitor is already running.")

    def stop(self):
        """
        Stops the RAM monitoring thread.
        """
        if self._thread and self._thread.is_alive():
            self._stop_event.set()  # Signal the thread to stop
            self._thread.join()  # Wait for the thread to finish
            self._thread = None
            PACE_LLM_INFO("SystemMonitor stopped.")
        else:
            PACE_LLM_INFO("SystemMonitor is not running.")

        # Convert the bytes to MB
        self.ram_data_history = [
            (x - self.initial_ram_usage) / (1024 * 1024) for x in self.ram_data_history
        ]

    def get_history(self) -> SystemMetrics:
        """
        Returns the history of CPU and RAM usage data captured.

        Returns:
            SystemMetrics: A SystemMetrics object containing the history of CPU and RAM usage data.
        """
        return SystemMetrics(
            interval=self.interval,
            cpu_usage=self.cpu_data_history,
            ram_usage=self.ram_data_history,
            peak_ram_usage=max(self.ram_data_history),
        )
