# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""SSE and plain-text streaming helpers for token delivery to clients.

Owns the per-request token queues (active_streams) and the async generator
that drains them.  Streaming chunks are serialized directly as
OpenAI-compatible ``text_completion`` / ``chat.completion.chunk`` JSON.

The engine now returns only token IDs.  This module detokenizes them
before sending text to the client (unless ``mlperf_mode`` is active).

For multi-prompt requests, ``merge_stream_generators`` multiplexes N
single-prompt generators into one interleaved SSE stream with correct
``choices[].index`` values.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Callable, Dict, List, Optional

from pace.server.router.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
)
from pace.server.router.tokenizer_utils import decode_logprobs_step, decode_token_ids
from pace.server.router.utils import RequestStatus
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_INFO

active_streams: Dict[str, asyncio.Queue] = {}

_scheduler_ref = None


def set_scheduler(scheduler):
    """Called once at startup to wire the scheduler into the streaming layer."""
    global _scheduler_ref
    _scheduler_ref = scheduler


def _handle_stream_cancel(request_id: str, req: Optional[object]) -> None:
    """Record cancellation intent for an in-flight request.

    Used for client disconnects and for stop-string completion; in the latter
    case the caller pre-sets ``status=COMPLETED`` so the worker keeps it.
    """
    PACE_INFO(
        f"Stream for request {request_id} removed ({getattr(req, 'finish_reason', '')})"
    )
    if _scheduler_ref is None:
        return
    if hasattr(_scheduler_ref, "cancelled_requests_per_worker") and req:
        worker_idx = getattr(req, "assigned_engine_index", 0)
        _scheduler_ref.cancelled_requests_per_worker[worker_idx].add(request_id)
    else:
        _scheduler_ref.cancelled_requests.add(request_id)


def _split_on_stop(text: str, stop_strings: Optional[List[str]]):
    """Split buffered *text* around stop strings.

    Returns ``(emit, hold, stopped)``: text safe to send now, a trailing
    ``max(len(s)) - 1`` window held back to avoid splitting a stop string
    across chunks, and whether a complete stop string was found.
    """
    if not stop_strings:
        return text, "", False
    hits = [p for p in (text.find(s) for s in stop_strings) if p != -1]
    if hits:
        return text[: min(hits)], "", True
    hold = max(len(s) for s in stop_strings) - 1
    if hold <= 0:
        return text, "", False
    return text[:-hold], text[-hold:], False


async def chat_stream_generator(
    request_id: str,
    model: str,
    chat_id: str,
    req: Optional[object] = None,
    completion_counter: Optional[List[int]] = None,
    skip_special_tokens: bool = True,
) -> AsyncGenerator[str, None]:
    """Drain the token queue for *request_id* and yield chat SSE chunks.

    Emits the OpenAI ``chat.completion.chunk`` delta format: a role chunk
    relayed with the first engine output (so time-to-first-token reflects real
    prefill latency, not stream-open time), one content chunk per detokenized
    step, then a final chunk carrying ``finish_reason``.

    Like ``stream_generator``, this does not emit the trailing usage chunk or
    ``[DONE]``; ``merge_stream_generators`` owns both so chat and completions
    terminate streams identically.  ``completion_counter`` is an optional
    single-element list accumulating the generated-token count for the usage
    chunk.
    """
    queue: Optional[asyncio.Queue] = active_streams.get(request_id)
    created = int(time.time())

    if not queue:
        yield f"data: {json.dumps({'error': 'Stream not found'})}\n\n"
        return

    buffer = ""
    role_sent = False
    stop_strings = getattr(req, "stop_strings", None) if req else None
    max_tokens = getattr(req.req, "max_tokens", None) if req else None
    try:
        if max_tokens == 0:
            # Scoring-only request: emit role + finish with no content, and drop
            # the forced token the engine samples during prefill.
            if req:
                req.status = RequestStatus.COMPLETED
            _handle_stream_cancel(request_id, req)
            for choice in (
                ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(role="assistant", content="")
                ),
                ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(), finish_reason="stop"
                ),
            ):
                chunk = ChatCompletionStreamResponse(
                    id=chat_id, created=created, model=model, choices=[choice]
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            return
        while True:
            token_data = await queue.get()
            if not role_sent:
                # Relay role delta with the first engine output.
                chunk = ChatCompletionStreamResponse(
                    id=chat_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=""),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                role_sent = True
            if token_data is None:
                finish = getattr(req, "finish_reason", "stop") if req else "stop"
                if buffer:
                    chunk = ChatCompletionStreamResponse(
                        id=chat_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0, delta=DeltaMessage(content=buffer)
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                chunk = ChatCompletionStreamResponse(
                    id=chat_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0, delta=DeltaMessage(), finish_reason=finish
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                break
            if token_data == "ERROR":
                yield f"data: {json.dumps({'error': 'An error occurred during processing'})}\n\n"
                break
            if completion_counter is not None:
                completion_counter[0] += len(token_data)
            buffer += decode_token_ids(
                token_data, skip_special_tokens=skip_special_tokens
            )
            emit, buffer, stopped = _split_on_stop(buffer, stop_strings)
            if emit:
                chunk = ChatCompletionStreamResponse(
                    id=chat_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0, delta=DeltaMessage(content=emit)
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if stopped:
                if req:
                    req.status = RequestStatus.COMPLETED
                _handle_stream_cancel(request_id, req)
                chunk = ChatCompletionStreamResponse(
                    id=chat_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0, delta=DeltaMessage(), finish_reason="stop"
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                break
    except asyncio.CancelledError:
        _handle_stream_cancel(request_id, req)
    except Exception as e:
        PACE_ERROR(f"Error in chat stream generator for {request_id}: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        active_streams.pop(request_id, None)


async def stream_generator(
    request_id: str,
    model: str,
    completion_id: str,
    index: int = 0,
    mlperf_mode: bool = False,
    req: Optional[object] = None,
    completion_counter: Optional[List[int]] = None,
    skip_special_tokens: bool = True,
) -> AsyncGenerator[str, None]:
    """Drain the token queue for *request_id* and yield SSE / plain-text chunks.

    ``index`` is the choice index for this prompt within a multi-prompt batch.
    The engine sends only token IDs (int).  In normal mode the router
    detokenizes them; in MLPerf mode raw IDs are streamed as plain text.

    ``completion_counter`` is an optional single-element mutable list used to
    accumulate the total number of generated tokens across all multiplexed
    prompt streams, so the merger can emit an aggregate usage chunk.
    """
    queue: Optional[asyncio.Queue] = active_streams.get(request_id)
    created = int(time.time())

    if not queue:
        if mlperf_mode:
            yield "ERROR\n"
        else:
            yield f"data: {json.dumps({'error': 'Stream not found'})}\n\n"
        return
    buffer = ""
    stop_strings = getattr(req, "stop_strings", None) if req else None
    max_tokens = getattr(req.req, "max_tokens", None) if req else None
    # Length of text actually emitted so far; anchors logprobs ``text_offset``.
    emitted_len = 0
    try:
        if not mlperf_mode and max_tokens == 0:
            # Scoring-only request: emit just the finish chunk, dropping the
            # forced token the engine samples during prefill.
            if req:
                req.status = RequestStatus.COMPLETED
            _handle_stream_cancel(request_id, req)
            finish = getattr(req, "finish_reason", "stop") if req else "stop"
            chunk = CompletionStreamResponse(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    CompletionResponseStreamChoice(
                        index=index, text="", finish_reason=finish
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            return
        while True:
            token_data = await queue.get()
            PACE_DEBUG(f"Received token data for {request_id}: {token_data}")

            if token_data is None:
                if mlperf_mode:
                    yield "[DONE]\n"
                else:
                    finish = getattr(req, "finish_reason", "stop") if req else "stop"
                    if buffer:
                        chunk = CompletionStreamResponse(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                CompletionResponseStreamChoice(index=index, text=buffer)
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    chunk = CompletionStreamResponse(
                        id=completion_id,
                        created=created,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=index, text="", finish_reason=finish
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                break

            if token_data == "ERROR":
                if mlperf_mode:
                    yield "ERROR\n"
                else:
                    yield f"data: {json.dumps({'error': 'An error occurred during processing'})}\n\n"
                break

            if isinstance(token_data, dict):
                tids = token_data.get("token_ids", [])
                step_logprobs = token_data.get("logprobs")
                # NOTE: prompt/echo logprobs (token_data["prompt_logprobs"]) are
                # intentionally NOT surfaced in the streaming path. Streamed
                # ``choices[].text`` never includes the echoed prompt, so
                # attaching prompt logprobs here would make
                # ``logprobs.tokens``/``text_offset`` reference tokens that are
                # absent from the emitted text, breaking alignment. Echo+logprobs
                # is served correctly via the non-streaming path.
            else:
                tids = token_data
                step_logprobs = None

            if completion_counter is not None:
                completion_counter[0] += len(tids)

            if mlperf_mode:
                yield " ".join(str(tid) for tid in tids) + "\n"
            else:
                buffer += decode_token_ids(
                    tids, skip_special_tokens=skip_special_tokens
                )
                emit, buffer, stopped = _split_on_stop(buffer, stop_strings)

                # Decode this step's numeric logprobs. One entry per generated
                # token, ``text_offset`` anchored at the running emitted length
                # and advancing by each token's decoded length.
                step_lp_out = None
                if step_logprobs is not None:
                    tokens, token_logprobs, top_logprobs = decode_logprobs_step(
                        step_logprobs
                    )
                    offsets = []
                    running = emitted_len
                    for tok_str in tokens:
                        offsets.append(running)
                        running += len(tok_str)
                    step_lp_out = {
                        "tokens": tokens,
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                        "text_offset": offsets,
                    }

                # Emit per step when logprobs are present or text to flush.
                if emit or step_lp_out is not None:
                    chunk = CompletionStreamResponse(
                        id=completion_id,
                        created=created,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=index, text=emit, logprobs=step_lp_out
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                emitted_len += len(emit)

                if stopped:
                    if req:
                        req.status = RequestStatus.COMPLETED
                    _handle_stream_cancel(request_id, req)
                    chunk = CompletionStreamResponse(
                        id=completion_id,
                        created=created,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=index, text="", finish_reason="stop"
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    break
    except asyncio.CancelledError:
        _handle_stream_cancel(request_id, req)
    except Exception as e:
        PACE_ERROR(f"Error in stream generator for {request_id}: {str(e)}")
        if mlperf_mode:
            yield "ERROR\n"
        else:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        active_streams.pop(request_id, None)


async def merge_stream_generators(
    generators: List[AsyncGenerator[str, None]],
    final_usage_fn: Optional[Callable[[], Optional[str]]] = None,
) -> AsyncGenerator[str, None]:
    """Multiplex N SSE stream generators into one interleaved stream.

    Yields chunks from whichever generator produces output first.  When
    ``final_usage_fn`` is provided, its result (a fully-formatted
    ``data: ...\\n\\n`` chunk, or ``None``) is emitted after all generators are
    exhausted and before ``data: [DONE]``.
    """
    if len(generators) == 1:
        async for chunk in generators[0]:
            yield chunk
        if final_usage_fn is not None:
            usage_chunk = final_usage_fn()
            if usage_chunk is not None:
                yield usage_chunk
        yield "data: [DONE]\n\n"
        return

    loop = asyncio.get_running_loop()
    pending = {loop.create_task(_anext_or_none(gen)): gen for gen in generators}

    try:
        while pending:
            done, _ = await asyncio.wait(
                pending.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                gen = pending.pop(task)
                result = task.result()
                if result is None:
                    continue
                yield result
                pending[loop.create_task(_anext_or_none(gen))] = gen
    except asyncio.CancelledError:
        for task in pending:
            task.cancel()
        raise

    if final_usage_fn is not None:
        usage_chunk = final_usage_fn()
        if usage_chunk is not None:
            yield usage_chunk
    yield "data: [DONE]\n\n"


async def _anext_or_none(gen: AsyncGenerator) -> Optional[str]:
    """Advance *gen* by one step; return ``None`` when exhausted."""
    try:
        return await gen.__anext__()
    except StopAsyncIteration:
        return None
