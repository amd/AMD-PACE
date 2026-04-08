# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse
import asyncio
import json
from typing import AsyncGenerator, Dict, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import make_asgi_app

import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from pace.server.router.scheduler import IterativeScheduler, PrefillFirstScheduler
from pace.server.router.utils import (
    CompletionRequest,
    CompletionResponse,
    Request,
    http_config,
)
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_INFO, PACE_WARNING

app = FastAPI()

# Active SSE streams keyed by request_id
active_streams: Dict[str, asyncio.Queue] = {}


def startup_event_wrapper(args):
    @app.on_event("startup")
    async def startup_event():
        """
        FastAPI startup hook: Configures all backend server instances and starts the scheduler.
        """

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
            "kvCacheType": args.kv_cache_type,
            "norm_backend": op_config.get("norm_backend", ""),
            "qkv_projection_backend": op_config.get("qkv_projection_backend", ""),
            "attention_backend": op_config.get("attention_backend", ""),
            "out_projection_backend": op_config.get("out_projection_backend", ""),
            "mlp_backend": op_config.get("mlp_backend", ""),
            "lm_head_backend": op_config.get("lm_head_backend", ""),
        }
        if spec_config:
            model_config["spec_config"] = spec_config
        kv_cache_memory_gb = getattr(args, "kv_cache_memory_gb", None)
        if kv_cache_memory_gb is not None:
            model_config["kv_cache_memory_gb"] = kv_cache_memory_gb

        config_payload = {"modelConfig": model_config}

        # Configure all engine instances
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

        # Start the scheduler loop
        asyncio.create_task(scheduler.start())
        PACE_INFO("Frontend started and scheduler initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """
    FastAPI shutdown hook: Stops the scheduler gracefully.
    """
    await scheduler.stop()
    PACE_INFO("Frontend shutting down")


async def stream_generator(
    request_id: str, mlperf_mode: bool = False
) -> AsyncGenerator[str, None]:
    """
    Generate stream for a specific request.
    MLPerf mode: Plain text token IDs (no JSON)
    Generation mode: SSE with JSON
    """
    queue: Optional[asyncio.Queue] = active_streams.get(request_id)
    if not queue:
        if mlperf_mode:
            yield "ERROR\n"
        else:
            yield f"data: {json.dumps({'error': 'Stream not found'})}\n\n"
        return
    try:
        while True:
            token_data = await queue.get()
            PACE_DEBUG(f"Received token data for {request_id}: {token_data}")

            if token_data is None:  # End of stream signal
                if mlperf_mode:
                    yield "[DONE]\n"
                else:
                    yield "data: [DONE]\n\n"
                break
            # Error sentinel: in MLPerf mode only int (token_id) or this sentinel are put; no string tokens
            if token_data == "ERROR":
                if mlperf_mode:
                    yield "ERROR\n"
                else:
                    yield f"data: {json.dumps({'error': 'An error occurred during processing'})}\n\n"
                break

            # Handle both integer token_id (MLPerf mode) and string tokens (generation mode)
            if isinstance(token_data, int):
                # MLPerf mode: just send the token_id as plain text
                yield f"{token_data}\n"
            else:
                # Generation mode: SSE with JSON
                sse_data = {
                    "request_id": request_id,
                    "choices": [{"delta": {"content": token_data}}],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"
    except asyncio.CancelledError:
        PACE_INFO(f"Stream for request {request_id} cancelled by client")
        scheduler.cancelled_requests.append(request_id)
        del scheduler.active_requests[request_id]
    except Exception as e:
        PACE_ERROR(f"Error in stream generator for {request_id}: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/completions")
async def chat_completions(request: CompletionRequest, fastapi_request: FastAPIRequest):
    """
    Handle completion requests in streaming and non-streaming modes.
    """
    if args.model.lower() != request.model.lower():
        PACE_WARNING(
            f"Requested model {request.model} differs from configured model {args.model}"
        )

        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model `{request.model}` does not exist or is not available.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    token_queue: asyncio.Queue = asyncio.Queue()
    req = Request(
        req=request,
        req_sampling_params=request.gen_config,
        token_queue=token_queue,
    )
    if request.input_length is not None:
        req.req_stats.input_length = request.input_length
        PACE_INFO(
            f"Received request {req.request_id} (stream={request.stream}, mlperf_mode={request.mlperf_mode}, input_length={request.input_length})"
        )
    else:
        PACE_INFO(
            f"Received request {req.request_id} (stream={request.stream}, mlperf_mode={request.mlperf_mode})"
        )
    if request.stream:
        PACE_INFO(f"Processing streaming request {req.request_id}")

        # Create a queue for this request's tokens
        active_streams[req.request_id] = token_queue

        # Submit request to scheduler
        await scheduler.submit_request(req)

        # Return streaming response
        # MLPerf mode: plain text; Generation mode: SSE
        media_type = "text/plain" if req.mlperf_mode else "text/event-stream"
        return StreamingResponse(
            stream_generator(req.request_id, req.mlperf_mode),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": req.request_id,
            },
        )

    # Non-streaming mode
    tokens = []
    token_ids = []

    # Submit request to scheduler
    await scheduler.submit_request(req)

    # Collect all tokens
    while True:
        # Check if client is still connected
        if await fastapi_request.is_disconnected():
            PACE_INFO(f"Non-streaming request {req.request_id} - client disconnected")
            scheduler.cancelled_requests.append(req.request_id)
            if req.request_id in scheduler.active_requests:
                del scheduler.active_requests[req.request_id]
            raise HTTPException(status_code=499, detail="Client disconnected")

        token = await token_queue.get()
        if token is None:
            break
        # Error sentinel: schedulers put "ERROR" (string) on failure; never model output in MLPerf mode
        if token == "ERROR":
            break

        # Handle both integer token_id (MLPerf mode) and string tokens (generation mode)
        if isinstance(token, int):
            # MLPerf mode: token is the token_id
            token_ids.append(token)
        else:
            # Generation mode: token is text string
            tokens.append(token)

    # Return complete response
    response_data = {
        "request_id": req.request_id,
        "model": request.model,
    }

    # Include token_ids if mlperf_mode is enabled and token_ids were collected
    if req.mlperf_mode and token_ids:
        response_data["choices"] = [
            {
                "token_ids": token_ids,
                "finish_reason": "stop",
            }
        ]
    else:
        # Generation mode: return text content
        response_data["choices"] = [
            {
                "message": {"role": "assistant", "content": "".join(tokens)},
                "finish_reason": "stop",
            }
        ]

    return JSONResponse(response_data)


@app.get("/v1/health")
async def health_check():
    """
    Health check endpoint for the frontend service.
    """
    return JSONResponse(
        {
            "status": "healthy",
            "service": "frontend",
            "scheduler_running": scheduler.is_running,
            "queue_size": scheduler.get_queue_size(),
            "active_requests": scheduler.get_active_requests_count(),
            "server_metrics_enabled": scheduler.scheduler_metrics_enabled,
        }
    )


@app.get("/v1/status/{request_id}")
async def get_request_status(request_id: str):
    """
    Get the status of a specific request by its request_id.
    """
    status = await scheduler.get_request_status(request_id)
    if status:
        return CompletionResponse(
            request_id=request_id,
            status=status["status"],
            message=status.get("message"),
            created_at=status.get("created_at"),
        )

    raise HTTPException(status_code=404, detail="Request not found")


@app.get("/v1/queue/status")
async def get_queue_status():
    """
    Get the current queue status from the scheduler.
    """
    return {
        "queue_size": scheduler.get_queue_size(),
        "active_requests": scheduler.get_active_requests_count(),
    }


@app.get("/v1/server_metrics")
async def get_server_wide_metrics():
    """
    Get aggregate server metrics from the scheduler.
    """
    return scheduler.server_metrics()


@app.get("/v1/models")
async def get_models():
    """
    Proxy endpoint to retrieve available models from the backend.
    Uses configured server_host/server_port and async client.
    """
    url = f"http://{args.server_host}:{args.server_port}/get_models"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=http_config.total,
                connect=http_config.connect,
                sock_connect=http_config.sock_connect,
                sock_read=http_config.sock_read,
            )
        ) as client:
            async with client.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    PACE_ERROR(
                        f"Backend returned error for get_models: {response.status} - {error_text}"
                    )
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Backend error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        PACE_ERROR(f"Failed to fetch models from backend: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch models") from e


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


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
        "--router_host",
        type=str,
        help="Host address to bind the server",
    )
    parser.add_argument(
        "--router_port",
        type=int,
        help="Port to bind the server",
    )
    parser.add_argument(
        "--server_host",
        type=str,
        help="Host address to bind the backend inference server",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        help="Port to bind the backend inference server",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to load",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type for the model",
    )
    parser.add_argument(
        "--kv_cache_type",
        type=str,
        help="KV cache type",
    )
    parser.add_argument(
        "--serve_type",
        type=str,
        choices=["iterative", "continuous_prefill_first"],
        help="Type of scheduler to use",
    )
    parser.add_argument(
        "--op_config",
        type=str,
        help="Operator backend configuration in JSON format",
    )
    parser.add_argument(
        "--scheduler_metrics_enabled",
        type=str,
        help="Enable scheduler metrics collection",
    )
    parser.add_argument(
        "--fastapi_log_level",
        type=str,
        help="Log level for FastAPI/uvicorn",
    )
    parser.add_argument(
        "--num_engine_instances",
        type=int,
        default=1,
        help="Number of engine instances running",
    )
    parser.add_argument(
        "--spec_config",
        type=str,
        default="{}",
        help="Speculative decoding config JSON",
    )
    parser.add_argument(
        "--kv_cache_memory_gb",
        type=float,
        default=None,
        help="Memory budget for SLAB KV cache pool in GB",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_router_arguments()
    if args.serve_type == "iterative":
        Scheduler = IterativeScheduler
    elif args.serve_type == "continuous_prefill_first":
        Scheduler = PrefillFirstScheduler
    else:
        raise ValueError(f"Unknown serve_type: {args.serve_type}")
    if args.scheduler_metrics_enabled in ["True", "true", "1"]:
        args.scheduler_metrics_enabled = True
    else:
        args.scheduler_metrics_enabled = False
    PACE_INFO(f"Using {args.serve_type} scheduler")
    PACE_INFO(f"Scheduler-Wide statistics enabled: {args.scheduler_metrics_enabled}")

    # Build list of all engine URLs
    engine_urls = []
    for instance_id in range(args.num_engine_instances):
        instance_port = args.server_port + instance_id
        engine_url = f"http://{args.server_host}:{instance_port}"
        engine_urls.append(engine_url)

    PACE_INFO(
        f"Initializing scheduler with {len(engine_urls)} engine instance(s): {engine_urls}"
    )

    # Pass all engine URLs to scheduler (always pass as list for consistency)
    scheduler = Scheduler(
        engine_urls,
        args.scheduler_metrics_enabled,
    )
    startup_event_wrapper(args)
    main(args)
