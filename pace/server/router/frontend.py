# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""FastAPI application lifecycle and entry-point for the PACE router.

Responsibilities limited to:
  - app creation and route registration
  - startup (engine configuration, scheduler launch)
  - shutdown
  - CLI argument parsing
  - uvicorn launch

API route handlers live in request_handler.py; streaming in streaming.py;
scheduling in scheduler.py; engine communication in engine_client.py.
"""

import argparse
import asyncio
import json

import aiohttp
import uvicorn
from fastapi import FastAPI
from prometheus_client import make_asgi_app

import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from transformers import AutoTokenizer

from pace.server.router.continuous_batch_scheduler import ContinuousBatchScheduler
from pace.server.router.request_handler import router as api_router, set_dependencies
from pace.server.router.scheduler import IterativeScheduler, PrefillFirstScheduler
from pace.server.router.streaming import set_scheduler
from pace.server.router.tokenizer_utils import init_tokenizer
from pace.server.router.utils import http_config
from pace.utils.logging import PACE_DEBUG, PACE_INFO, PACE_WARNING

app = FastAPI()
app.include_router(api_router)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def startup_event_wrapper(args):
    @app.on_event("startup")
    async def startup_event():
        """Configure all backend engine instances and start the scheduler."""
        try:
            op_config = json.loads(args.op_config)
        except json.JSONDecodeError:
            PACE_WARNING(
                "Invalid op_config JSON. Proceeding with default operator backends."
            )
            op_config = {}

        try:
            spec_config = json.loads(args.spec_config)
        except json.JSONDecodeError:
            PACE_WARNING(
                "Invalid spec_config JSON. Proceeding without speculative decoding."
            )
            spec_config = {}

        model_config = {
            "modelId": args.model,
            "dataType": args.dtype.lower(),
            "norm_backend": op_config.get("norm_backend", ""),
            "qkv_projection_backend": op_config.get("qkv_projection_backend", ""),
            "attention_backend": op_config.get("attention_backend", ""),
            "out_projection_backend": op_config.get("out_projection_backend", ""),
            "mlp_backend": op_config.get("mlp_backend", ""),
            "lm_head_backend": op_config.get("lm_head_backend", ""),
        }
        if args.kv_cache_type is not None:
            model_config["kvCacheType"] = args.kv_cache_type
        if spec_config:
            model_config["spec_config"] = spec_config
        kv_cache_memory_gb = getattr(args, "kv_cache_memory_gb", None)
        if kv_cache_memory_gb is not None:
            model_config["kv_cache_memory_gb"] = kv_cache_memory_gb

        config_payload = {"modelConfig": model_config}

        num_instances = args.num_engine_instances
        PACE_INFO(f"Configuring {num_instances} engine instance(s)")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=http_config.total,
                connect=http_config.connect,
                sock_connect=http_config.sock_connect,
                sock_read=http_config.sock_read,
            )
        ) as client:
            for instance_id in range(num_instances):
                instance_port = args.server_port + instance_id
                config_url = f"http://{args.server_host}:{instance_port}/config_server"

                try:
                    PACE_INFO(
                        f"Configuring engine instance {instance_id} at port {instance_port} with model={args.model}, dtype={args.dtype.lower()}, kv_cache_type={args.kv_cache_type}"
                    )
                    async with client.post(config_url, json=config_payload) as response:
                        if response.status == 200:
                            PACE_INFO(
                                f"Engine instance {instance_id} configured successfully: {await response.json()}"
                            )
                        else:
                            error_text = await response.text()
                            error_msg = f"Failed to configure engine instance {instance_id}: {response.status} - {error_text}"
                            PACE_WARNING(error_msg)

                except Exception as e:
                    PACE_DEBUG(f"Error configuring engine instance {instance_id}: {e}")
                    PACE_WARNING(
                        f"Engine instance {instance_id} will start without pre-loading model. You can configure it later via API."
                    )

        PACE_INFO(
            f"All {num_instances} engine instance(s) are running. You can now send requests."
        )

        asyncio.create_task(scheduler.start())
        PACE_INFO("Frontend started and scheduler initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler gracefully."""
    await scheduler.stop()
    PACE_INFO("Frontend shutting down")


def main(args):
    if args.fastapi_log_level == "None":
        PACE_INFO("Disabling FastAPI/uvicorn logging")
        args.fastapi_log_level = None
    else:
        PACE_INFO("Setting FastAPI/uvicorn log level to DEFAULT")
        args.fastapi_log_level = uvicorn.config.LOGGING_CONFIG
    uvicorn.run(
        app,
        host=args.router_host,
        port=args.router_port,
        log_config=args.fastapi_log_level,
    )


def parse_router_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router_host", type=str, help="Host address to bind the server"
    )
    parser.add_argument("--router_port", type=int, help="Port to bind the server")
    parser.add_argument(
        "--server_host",
        type=str,
        help="Host address to bind the backend inference server",
    )
    parser.add_argument(
        "--server_port", type=int, help="Port to bind the backend inference server"
    )
    parser.add_argument("--model", type=str, help="Model name to load")
    parser.add_argument("--dtype", type=str, help="Data type for the model")
    parser.add_argument("--kv_cache_type", type=str, help="KV cache type")
    parser.add_argument(
        "--serve_type",
        type=str,
        default="iterative",
        choices=["iterative", "continuous_prefill_first", "continuous_batch"],
        help="Type of scheduler to use",
    )
    parser.add_argument(
        "--op_config", type=str, help="Operator backend configuration in JSON format"
    )
    parser.add_argument(
        "--scheduler_metrics_enabled",
        type=str,
        help="Enable scheduler metrics collection",
    )
    parser.add_argument(
        "--fastapi_log_level", type=str, help="Log level for FastAPI/uvicorn"
    )
    parser.add_argument(
        "--num_engine_instances",
        type=int,
        default=1,
        help="Number of engine instances running",
    )
    parser.add_argument(
        "--spec_config", type=str, default="{}", help="Speculative decoding config JSON"
    )
    parser.add_argument(
        "--kv_cache_memory_gb",
        type=float,
        default=None,
        help="Memory budget for SLAB KV cache pool in GB",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=2048,
        help="Max tokens per step for continuous_batch scheduler",
    )
    parser.add_argument(
        "--max_num_running_reqs",
        type=int,
        default=128,
        help="Hard cap on concurrent decode sequences per engine for continuous_batch",
    )
    parser.add_argument(
        "--max_prefill_chunk",
        type=int,
        default=0,
        help="Max prefill chunk size (0=no cap) for continuous_batch scheduler",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_router_arguments()
    if args.scheduler_metrics_enabled in ["True", "true", "1"]:
        args.scheduler_metrics_enabled = True
    else:
        args.scheduler_metrics_enabled = False
    PACE_INFO(f"Using {args.serve_type} scheduler")
    PACE_INFO(f"Scheduler-Wide statistics enabled: {args.scheduler_metrics_enabled}")

    engine_urls = []
    for instance_id in range(args.num_engine_instances):
        instance_port = args.server_port + instance_id
        engine_url = f"http://{args.server_host}:{instance_port}"
        engine_urls.append(engine_url)

    PACE_INFO(
        f"Initializing scheduler with {len(engine_urls)} engine instance(s): {engine_urls}"
    )

    try:
        spec_config = json.loads(args.spec_config)
    except json.JSONDecodeError:
        spec_config = {}

    if args.serve_type == "continuous_batch":
        if args.kv_cache_type is not None:
            kv_type = args.kv_cache_type.lower()
            if kv_type not in ("paged", "slab_pool"):
                PACE_WARNING(
                    f"'continuous_batch' scheduler only works with non-contiguous KV "
                    f"cache backends ('PAGED' or 'SLAB_POOL'). Got '{args.kv_cache_type}'. "
                    f"Overriding to 'SLAB_POOL'."
                )
                args.kv_cache_type = "SLAB_POOL"
        spec_tokens = 1
        if spec_config:
            spec_tokens = spec_config.get("num_speculative_tokens", 12) + 1
        scheduler = ContinuousBatchScheduler(
            engine_url=engine_urls,
            scheduler_metrics_enabled=args.scheduler_metrics_enabled,
            max_num_running_reqs=args.max_num_running_reqs,
            token_budget=args.token_budget,
            max_prefill_chunk=args.max_prefill_chunk,
            tokens_per_decode=spec_tokens,
        )
    elif args.serve_type == "iterative":
        scheduler = IterativeScheduler(
            engine_urls,
            args.scheduler_metrics_enabled,
        )
    elif args.serve_type == "continuous_prefill_first":
        scheduler = PrefillFirstScheduler(
            engine_urls,
            args.scheduler_metrics_enabled,
        )
    else:
        raise ValueError(f"Unknown serve_type: {args.serve_type}")

    PACE_INFO(f"Loading tokenizer for model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    PACE_INFO("Tokenizer loaded successfully")

    init_tokenizer(tokenizer)
    set_dependencies(scheduler, args, tokenizer)
    set_scheduler(scheduler)

    startup_event_wrapper(args)
    main(args)
