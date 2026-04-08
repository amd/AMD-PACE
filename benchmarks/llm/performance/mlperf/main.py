#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content.
# ******************************************************************************

import argparse
import logging
import os
import signal
import sys

import mlperf_loadgen as lg

sys.path.insert(0, os.getcwd())
from SUT_PACE_async import SUTServer

log = logging.getLogger(__name__)


def construct_sut_compat(issue_cb, flush_cb, process_latencies_cb=None):
    try:
        import inspect

        sig = inspect.signature(lg.ConstructSUT)
        if len(sig.parameters) == 3 and process_latencies_cb is not None:
            return lg.ConstructSUT(issue_cb, flush_cb, process_latencies_cb)
    except Exception:
        pass
    try:
        return lg.ConstructSUT(issue_cb, flush_cb)
    except TypeError:
        return lg.ConstructSUT(issue_cb, flush_cb, lambda lat_ns: None)


def start_test_compat(
    sut, qsl, settings, logs=None, audit_config_filename="audit.config"
):
    audit_path = audit_config_filename
    parent = os.path.dirname(audit_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not os.path.exists(audit_path):
        with open(audit_path, "w", encoding="utf-8") as f:
            f.write("# empty audit config\n")

    if hasattr(lg, "StartTestWithLogSettings") and logs is not None:
        try:
            return lg.StartTestWithLogSettings(sut, qsl, settings, logs, audit_path)
        except TypeError:
            return lg.StartTestWithLogSettings(sut, qsl, settings, logs)

    return lg.StartTest(sut, qsl, settings, audit_path)


# ============================
# Signal handling
# ============================

_global_sut = None


def signal_handler(signum, frame):
    if _global_sut:
        _global_sut.flush_queries()


# ============================
# Argument parsing
# ============================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model name",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset path")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="data type of the model, choose from float16, bfloat16 and float32",
    )
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="LoadGen user config (min_query_count, max_query_count, target_qps, etc.)",
    )
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=13368,
        help="Number of samples to use in benchmark.",
    )
    parser.add_argument(
        "--output-log-dir", type=str, default="output", help="Where logs are saved"
    )
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b",
        choices=["llama3_1-8b", "llama3_1-8b-edge"],
        help="Model name for LoadGen configuration",
    )
    # PACE-specific arguments
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://localhost:8080/v1/completions",
        help="PACE server endpoint URL",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    # Async-specific arguments
    parser.add_argument(
        "--server-type",
        type=str,
        default="pace",
        choices=["pace", "vllm"],
        help="Server type: 'pace' for PACE server or 'vllm' for vLLM server",
    )
    args = parser.parse_args()
    return args


# ============================
# Main function
# ============================

SCENARIO = "Server"


def main():
    global _global_sut

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = get_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("Starting MLPerf test (Server scenario)")
    log.info("Server type: %s", args.server_type)
    log.info("Endpoint: %s, Timeout: %ss", args.endpoint_url, args.timeout_sec)

    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Server
    settings.FromConfig(args.user_conf, args.lg_model_name, SCENARIO)
    settings.mode = (
        lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
    )

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    # SUT: async Server-only implementation
    sut = SUTServer(
        model_path=args.model_path,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        endpoint_url=args.endpoint_url,
        timeout_sec=args.timeout_sec,
        max_tokens=args.max_tokens,
        server_type=args.server_type,
    )

    _global_sut = sut

    # Start SUT
    sut.start()

    # --- LoadGen SUT wiring ---
    lgSUT = construct_sut_compat(sut.issue_queries, sut.flush_queries)

    log.info(
        "Benchmark run started: mode=%s",
        "accuracy" if args.accuracy else "performance",
    )

    try:
        start_test_compat(
            lgSUT,
            sut.qsl,
            settings,
            log_settings,
            audit_config_filename=args.audit_conf,
        )
        log.info("Run completed")

    except Exception as e:
        log.error("Test failed: %s", e)
        raise
    finally:
        log.info("Cleaning up")
        sut.stop()

        try:
            lg.DestroySUT(lgSUT)
        except Exception as e:
            log.error("DestroySUT failed: %s", e)

        try:
            lg.DestroyQSL(sut.qsl)
        except Exception as e:
            log.error("DestroyQSL failed: %s", e)

        log.info("Cleanup complete")


if __name__ == "__main__":
    main()
