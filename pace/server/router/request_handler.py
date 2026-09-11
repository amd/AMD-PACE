# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""OpenAI-compatible API routes for the PACE router.

Each public function is a FastAPI route handler.  They are registered on the
app by frontend.py at import time via ``APIRouter``.  The scheduler, args,
and tokenizer references are injected at startup through ``set_dependencies``.

Multi-prompt requests (``prompt`` is a list) are fanned out into separate
internal ``Request`` objects, each submitted independently to the scheduler.
Responses are merged back into a single OpenAI ``CompletionResponse`` (or
interleaved SSE stream) with correct ``choices[].index`` values.
"""

import asyncio
import json
import time
import uuid
from typing import List, Optional, Tuple

import aiohttp
import jinja2
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from pace.server.router.protocol import (
    CHAT_ID_PREFIX,
    COMPLETION_ID_PREFIX,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    RequestStatusResponse,
    UsageInfo,
)
from pace.server.router.streaming import (
    active_streams,
    chat_stream_generator,
    merge_stream_generators,
    stream_generator,
)
from pace.server.router.tokenizer_utils import (
    NoChatTemplateError,
    check_stop_strings,
    decode_logprobs_step,
    decode_token_ids,
    init_tokenizer,
    normalize_prompts,
    render_chat_prompt,
    truncate_at_stop_string,
)
from pace.server.router.metrics import fetch_aggregated_cache_stats
from pace.server.router.utils import Request, http_config
from pace.utils.logging import PACE_INFO, PACE_WARNING

router = APIRouter()

_scheduler = None
_args = None
_tokenizer = None


def set_dependencies(scheduler, args, tokenizer=None):
    """Inject runtime dependencies after startup. Called once from frontend.py."""
    global _scheduler, _args, _tokenizer
    _scheduler = scheduler
    _args = args
    _tokenizer = tokenizer
    if tokenizer is not None:
        init_tokenizer(tokenizer)


async def _collect_token_ids(req: Request) -> List[int]:
    """Drain a request's token queue; items are token-ID lists or single ints."""
    token_ids: List[int] = []
    while True:
        item = await req.token_queue.get()
        if item is None:
            break
        if item == "ERROR":
            raise RuntimeError(f"Engine error processing request {req.request_id}")
        if isinstance(item, list):
            token_ids.extend(item)
        elif isinstance(item, int):
            token_ids.append(item)
    return token_ids


def _model_not_found(request_model: str) -> JSONResponse:
    """Build the OpenAI-style 404 error returned for an unknown model."""
    PACE_WARNING(
        f"Requested model {request_model} differs from configured model {_args.model}"
    )
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"The model `{request_model}` does not exist or is not available.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Handle OpenAI-compatible completion requests with multi-prompt support."""
    if _args.model.lower() != request.model.lower():
        return _model_not_found(request.model)

    prompt_token_lists = normalize_prompts(request.prompt, _tokenizer)

    if request.mlperf_mode and len(prompt_token_lists) > 1:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "mlperf_mode only supports single-prompt requests.",
                    "type": "invalid_request_error",
                    "param": "prompt",
                    "code": "invalid_value",
                }
            },
        )

    engine_gen_config = request.to_engine_config()
    group_id = str(uuid.uuid4())

    sub_requests: List[Tuple[int, Request]] = []
    for idx, token_ids in enumerate(prompt_token_lists):
        token_queue: asyncio.Queue = asyncio.Queue()
        sub_req = Request(
            req=request,
            token_queue=token_queue,
            engine_gen_config=engine_gen_config,
            prompt_index=idx,
            group_id=group_id,
        )
        sub_req.req = request.model_copy(update={"prompt": token_ids})
        sub_req.req_stats.input_length = len(token_ids)

        sub_requests.append((idx, sub_req))

    PACE_INFO(
        f"Received request group {group_id} with {len(prompt_token_lists)} prompt(s), "
        f"stream={request.stream}, mlperf_mode={request.mlperf_mode}"
    )

    if request.stream:
        return await _handle_streaming(sub_requests, request, group_id)

    return await _handle_non_streaming(sub_requests, request, group_id)


async def _handle_streaming(
    sub_requests: List[Tuple[int, Request]],
    openai_req: CompletionRequest,
    group_id: str,
):
    """Set up interleaved SSE streams for all sub-requests."""
    completion_id = f"{COMPLETION_ID_PREFIX}{group_id}"
    mlperf_mode = openai_req.mlperf_mode

    if mlperf_mode:
        idx, req = sub_requests[0]
        PACE_INFO(f"Processing MLPerf streaming request {req.request_id}")
        active_streams[req.request_id] = req.token_queue
        await _scheduler.submit_request(req)
        return StreamingResponse(
            stream_generator(
                request_id=req.request_id,
                model=openai_req.model,
                completion_id=completion_id,
                index=0,
                mlperf_mode=True,
                req=req,
                skip_special_tokens=openai_req.skip_special_tokens,
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": req.request_id,
            },
        )

    completion_counter = [0]
    prompt_tokens = sum(req.req_stats.input_length for _, req in sub_requests)

    generators = []
    for idx, req in sub_requests:
        PACE_INFO(f"Processing streaming sub-request {req.request_id} (index={idx})")
        active_streams[req.request_id] = req.token_queue
        await _scheduler.submit_request(req)
        generators.append(
            stream_generator(
                request_id=req.request_id,
                model=openai_req.model,
                completion_id=completion_id,
                index=idx,
                mlperf_mode=False,
                req=req,
                completion_counter=completion_counter,
                skip_special_tokens=openai_req.skip_special_tokens,
            )
        )

    def final_usage_fn():
        if not openai_req.include_usage:
            return None
        chunk = CompletionStreamResponse(
            id=completion_id,
            created=int(time.time()),
            model=openai_req.model,
            choices=[],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_counter[0],
                total_tokens=prompt_tokens + completion_counter[0],
            ),
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    return StreamingResponse(
        merge_stream_generators(generators, final_usage_fn=final_usage_fn),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": group_id,
        },
    )


def _assemble_logprobs(
    logprobs_parts: List[dict],
) -> object:
    """Merge per-step numeric engine logprobs into an OpenAI logprobs object.

    Detokenization is router-native (via ``decode_logprobs_step``) so
    ``tokens``/``top_logprobs`` use the same tokenizer as the response text;
    ``text_offset`` is cumulative decoded-token length. One entry is emitted
    per generated token; the arrays are mutually length-consistent. Returns
    ``None`` when there are no logprobs.
    """
    if not logprobs_parts:
        return None

    tokens: List[str] = []
    token_logprobs: List = []
    top_logprobs: List = []
    for part in logprobs_parts:
        t, tlp, top = decode_logprobs_step(part, tokenizer=_tokenizer)
        tokens.extend(t)
        token_logprobs.extend(tlp)
        top_logprobs.extend(top)

    if not tokens:
        return None

    text_offset: List[int] = []
    offset = 0
    for tok_str in tokens:
        text_offset.append(offset)
        offset += len(tok_str)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


async def _handle_non_streaming(
    sub_requests: List[Tuple[int, Request]],
    openai_req: CompletionRequest,
    group_id: str,
):
    """Submit all sub-requests, collect token IDs, detokenize, build CompletionResponse."""
    for _, req in sub_requests:
        await _scheduler.submit_request(req)

    async def _collect_tokens(
        req: Request,
    ) -> Tuple[List[int], List[dict], Optional[dict]]:
        """Drain the token queue; items are token-ID lists, dicts, or single ints.

        Returns ``(token_ids, sample_logprobs_parts, prompt_logprobs)`` where
        ``sample_logprobs_parts`` is the list of per-step generated-token
        logprobs dicts (may be empty) and ``prompt_logprobs`` is the single
        echo/prompt logprobs dict from the prefill step (or None).
        """
        token_ids: List[int] = []
        sample_logprobs_parts: List[dict] = []
        prompt_logprobs: Optional[dict] = None
        while True:
            item = await req.token_queue.get()
            if item is None:
                break
            if item == "ERROR":
                raise RuntimeError(f"Engine error processing request {req.request_id}")
            if isinstance(item, dict):
                token_ids.extend(item.get("token_ids", []))
                lp = item.get("logprobs")
                if lp is not None:
                    sample_logprobs_parts.append(lp)
                plp = item.get("prompt_logprobs")
                if plp is not None and prompt_logprobs is None:
                    prompt_logprobs = plp
            elif isinstance(item, list):
                token_ids.extend(item)
            elif isinstance(item, int):
                token_ids.append(item)
        return token_ids, sample_logprobs_parts, prompt_logprobs

    tasks = [asyncio.create_task(_collect_tokens(req)) for _, req in sub_requests]

    try:
        results = await asyncio.gather(*tasks)
    except RuntimeError as e:
        PACE_WARNING(f"Request group {group_id} failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "server_error",
                }
            },
        )

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for (idx, req), (token_ids, sample_logprobs_parts, prompt_logprobs) in zip(
        sub_requests, results
    ):
        finish_reason = req.finish_reason
        # max_tokens=0 (OpenAI scoring contract): score the prompt, generate
        # nothing. The engine always samples one forced token, so discard it.
        if openai_req.max_tokens == 0:
            token_ids = []

        if req.mlperf_mode:
            completion_text = " ".join(str(tid) for tid in token_ids)
        else:
            completion_text = decode_token_ids(
                token_ids, openai_req.skip_special_tokens, _tokenizer
            )
            # If a stop string is present the model output is truncated here;
            # report finish_reason="stop" to match the streaming path.
            if check_stop_strings(completion_text, req.stop_strings):
                finish_reason = "stop"
            completion_text = truncate_at_stop_string(completion_text, req.stop_strings)

        if req.echo:
            prompt_ids = req.req.prompt
            if isinstance(prompt_ids, list):
                prompt_text = decode_token_ids(prompt_ids, tokenizer=_tokenizer)
            else:
                prompt_text = str(prompt_ids)
            completion_text = prompt_text + completion_text

        if req.suffix:
            completion_text = completion_text + req.suffix

        # Echoed prompt logprobs first (when set), then one entry per sampled
        # token (omitted for max_tokens=0 scoring).
        logprobs_ordered_parts: List[dict] = []
        if req.echo and prompt_logprobs is not None:
            logprobs_ordered_parts.append(prompt_logprobs)
        if openai_req.max_tokens != 0:
            logprobs_ordered_parts.extend(sample_logprobs_parts)
        logprobs_response = _assemble_logprobs(logprobs_ordered_parts)

        choices.append(
            CompletionResponseChoice(
                index=idx,
                text=completion_text,
                logprobs=logprobs_response,
                finish_reason=finish_reason,
            )
        )
        total_prompt_tokens += req.req_stats.input_length
        total_completion_tokens += len(token_ids)

    response = CompletionResponse(
        id=f"{COMPLETION_ID_PREFIX}{group_id}",
        model=openai_req.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )
    return JSONResponse(response.model_dump())


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle OpenAI-compatible chat completion requests.

    ``messages`` are rendered to prompt token IDs via the tokenizer's chat
    template, then routed through the same scheduling path as
    ``/v1/completions``.  A single completion is generated per request.
    """
    if _args.model.lower() != request.model.lower():
        return _model_not_found(request.model)

    messages = [m.model_dump() for m in request.messages]
    try:
        prompt_token_ids = render_chat_prompt(
            messages,
            add_generation_prompt=request.add_generation_prompt,
            chat_template_kwargs=request.chat_template_kwargs,
            tokenizer=_tokenizer,
        )
    except NoChatTemplateError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "no_chat_template",
                }
            },
        )
    except (ValueError, TypeError, jinja2.TemplateError) as e:
        # Template exists but failed to render on bad input (e.g. invalid
        # role ordering, Jinja syntax/undefined errors).  Only these
        # client-input errors map to 400; anything else (internal bugs,
        # RuntimeError) propagates as a 500 rather than being mislabeled.
        PACE_WARNING(f"Chat template rendering failed for {request.model}: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Failed to render chat template: {e}",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "template_render_error",
                }
            },
        )

    group_id = str(uuid.uuid4())
    token_queue: asyncio.Queue = asyncio.Queue()
    # Chat token paths expect plain token-ID lists, so drop logprobs here.
    engine_request = request.model_copy(
        update={"prompt": prompt_token_ids, "logprobs": None}
    )
    req = Request(
        req=engine_request,
        token_queue=token_queue,
        engine_gen_config=engine_request.to_engine_config(),
        group_id=group_id,
    )
    req.req_stats.input_length = len(prompt_token_ids)

    PACE_INFO(
        f"Received chat request {req.request_id} "
        f"(stream={request.stream}, messages={len(request.messages)})"
    )

    chat_id = f"{CHAT_ID_PREFIX}{group_id}"

    if request.stream:
        active_streams[req.request_id] = token_queue
        await _scheduler.submit_request(req)

        completion_counter = [0]
        prompt_tokens = req.req_stats.input_length
        generator = chat_stream_generator(
            req.request_id,
            request.model,
            chat_id,
            req,
            completion_counter=completion_counter,
            skip_special_tokens=request.skip_special_tokens,
        )

        def final_usage_fn():
            if not request.include_usage:
                return None
            chunk = ChatCompletionStreamResponse(
                id=chat_id,
                created=int(time.time()),
                model=request.model,
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_counter[0],
                    total_tokens=prompt_tokens + completion_counter[0],
                ),
            )
            return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        return StreamingResponse(
            merge_stream_generators([generator], final_usage_fn=final_usage_fn),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": req.request_id,
            },
        )

    await _scheduler.submit_request(req)
    try:
        token_ids = await _collect_token_ids(req)
    except RuntimeError as e:
        PACE_WARNING(f"Chat request {req.request_id} failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "server_error",
                }
            },
        )

    text = decode_token_ids(token_ids, request.skip_special_tokens, _tokenizer)
    finish_reason = (
        "stop" if check_stop_strings(text, req.stop_strings) else req.finish_reason
    )
    text = truncate_at_stop_string(text, req.stop_strings)

    response = ChatCompletionResponse(
        id=chat_id,
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=req.req_stats.input_length,
            completion_tokens=len(token_ids),
            total_tokens=req.req_stats.input_length + len(token_ids),
        ),
    )
    return JSONResponse(response.model_dump())


@router.get("/v1/health")
async def health_check():
    """Health check endpoint for the frontend service."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "frontend",
            "scheduler_running": _scheduler.is_running,
            "queue_size": _scheduler.get_queue_size(),
            "active_requests": _scheduler.get_active_requests_count(),
            "server_metrics_enabled": _scheduler.scheduler_metrics_enabled,
        }
    )


@router.get("/v1/status/{request_id}")
async def get_request_status(request_id: str):
    """Get the status of a specific request by its request_id."""
    status = await _scheduler.get_request_status(request_id)
    if status:
        return RequestStatusResponse(
            request_id=request_id,
            status=status["status"],
            message=status.get("message"),
            created_at=status.get("created_at"),
        )

    gid = request_id.removeprefix(COMPLETION_ID_PREFIX)
    matches = [
        req
        for req in _scheduler.active_requests.values()
        if getattr(req, "group_id", None) == gid
    ]
    if matches:
        return JSONResponse(
            {
                "id": f"{COMPLETION_ID_PREFIX}{gid}",
                "statuses": [
                    {"index": req.prompt_index, "status": str(req.status.name)}
                    for req in sorted(matches, key=lambda r: r.prompt_index)
                ],
            }
        )
    raise HTTPException(status_code=404, detail="Request not found")


@router.get("/v1/queue/status")
async def get_queue_status():
    """Get the current queue status from the scheduler."""
    return {
        "queue_size": _scheduler.get_queue_size(),
        "active_requests": _scheduler.get_active_requests_count(),
    }


@router.get("/v1/server_metrics")
async def get_server_wide_metrics():
    """Get aggregate server metrics, including cache stats merged across all engines."""
    metrics = _scheduler.server_metrics()
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            agg = await fetch_aggregated_cache_stats(session, _scheduler.engine_urls)
        for key, value in (agg.get("pool") or {}).items():
            if isinstance(value, (int, float)):
                metrics[f"kv_cache_pool_{key}"] = value
        for key, value in (agg.get("prefix_cache") or {}).items():
            if isinstance(value, (int, float)):
                metrics[key] = value
    except Exception as e:
        PACE_WARNING(f"Failed to fetch cache_stats for server metrics: {e!r}")
    return metrics


@router.get("/v1/config")
async def get_config():
    """Return model name and key engine configuration for evaluation tooling."""
    # ``op_config`` is a JSON string CLI arg (default "{}") and may be malformed,
    # so parse defensively and fall back to an empty config.
    op_config = {}
    raw_op_config = getattr(_args, "op_config", None)
    if raw_op_config:
        try:
            op_config = json.loads(raw_op_config)
        except (ValueError, TypeError):
            op_config = {}
    return JSONResponse(
        {
            "model": _args.model,
            "kv_cache_type": getattr(_args, "kv_cache_type", "N/A"),
            "attention_backend": op_config.get("attention_backend", "N/A"),
        }
    )


@router.get("/v1/models")
async def get_models():
    """Proxy endpoint to retrieve available models from the backend."""
    url = f"http://{_args.server_host}:{_args.server_port}/get_models"
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
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Backend error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=502, detail="Failed to fetch models") from e
