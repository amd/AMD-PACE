# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""HTTP client for communicating with PACE engine backends.

Encapsulates all HTTP calls to the engine /step and /remove_sequence endpoints.
Schedulers call into these helpers rather than managing aiohttp sessions
directly, keeping network I/O out of the scheduling logic.

The engine payload uses its native ``generation_config`` format (matching
``SamplingConfig`` kwargs).  Translation from the OpenAI request shape is
done once at the router boundary (``CompletionRequest.to_engine_config()``)
and stored on ``Request.engine_gen_config``.
"""

import asyncio
import time
import traceback
from typing import AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp

from pace.server.router.utils import (
    Request,
    RequestStatus,
    TokenStepResult,
    http_config,
)
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_WARNING

WorkItem = Tuple[Request, int, int]


def _step_payload(
    token_ids: List[int], req_result: dict
) -> Union[List[int], TokenStepResult]:
    """Build the per-step value handed to the router.

    Returns a plain ``list[int]`` when no logprob data is present (backward
    compatible), or a ``TokenStepResult`` dict carrying ``token_ids`` plus any
    ``logprobs`` (sampled) and/or ``prompt_logprobs`` (echo) from the engine.
    """
    step_logprobs = req_result.get("logprobs")
    prompt_logprobs = req_result.get("prompt_logprobs")
    if step_logprobs is None and prompt_logprobs is None:
        return token_ids
    payload: TokenStepResult = {"token_ids": token_ids}
    if step_logprobs is not None:
        payload["logprobs"] = step_logprobs
    if prompt_logprobs is not None:
        payload["prompt_logprobs"] = prompt_logprobs
    return payload


def _make_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=http_config.total,
        connect=http_config.connect,
        sock_connect=http_config.sock_connect,
        sock_read=http_config.sock_read,
    )


async def clear_concluded_request(
    req_ids: Union[str, List[str]],
    session: aiohttp.ClientSession,
    engine_url: str,
) -> None:
    """Remove finished/cancelled/errored sequences from the engine."""
    ids = [req_ids] if isinstance(req_ids, str) else req_ids
    if not ids:
        return
    try:
        async with session.post(
            f"{engine_url}/remove_sequence", json={"sequence_ids": ids}
        ) as response:
            if response.status != 200:
                PACE_ERROR(
                    f"Failed to remove {len(ids)} sequence(s): "
                    f"status {response.status}"
                )
    except Exception as e:
        PACE_ERROR(f"Error removing sequences from engine: {e}")


async def iterative_generate_tokens(
    req: Request,
    cancelled_requests: Set[str],
    engine_url: str,
) -> AsyncGenerator[Union[List[int], TokenStepResult, str, None], None]:
    """Run prefill-then-decode loop for a single request (iterative scheduler).

    Yields one of:
      - ``TokenStepResult`` dict with ``token_ids`` and ``logprobs`` when
        the engine returns logprob data for the step.
      - ``list[int]`` token-ID batch when no logprobs were requested.
      - ``"ERROR"`` sentinel on engine failure.
      - ``None`` to signal stream end.

    The router handles all detokenization.
    """
    gen_config = req.engine_gen_config
    prompt = req.req.prompt
    engine_req_id = req.request_id
    assigned_engine_url = req.assigned_engine_url or engine_url

    PACE_DEBUG(f"Request {engine_req_id} will use engine: {assigned_engine_url}")
    PACE_DEBUG(
        f"Prompt token count (req={engine_req_id}): {len(prompt) if isinstance(prompt, list) else 'N/A'}"
    )

    async with aiohttp.ClientSession(timeout=_make_timeout()) as session:
        is_prefill = True

        while req.status not in (RequestStatus.COMPLETED, RequestStatus.ERROR):
            if engine_req_id in cancelled_requests:
                PACE_DEBUG(f"Request {engine_req_id} cancelled, stopping generation")
                if req.status != RequestStatus.COMPLETED:
                    req.status = RequestStatus.ERROR
                break

            try:
                worker_id = getattr(req, "assigned_engine_index", 0)
                if is_prefill:
                    PACE_DEBUG(
                        f"[WORKER-{worker_id}] Executing PREFILL step for request {engine_req_id}"
                    )
                    payload = {
                        "prefill_batch": [
                            {
                                "is_prefill": True,
                                "prompt": prompt,
                                "req_id": engine_req_id,
                                "generation_config": gen_config,
                            }
                        ]
                    }
                else:
                    PACE_DEBUG(
                        f"[WORKER-{worker_id}] Executing DECODE step {req.req_stats['generated_tokens_count']} for request {engine_req_id}"
                    )
                    payload = {"is_decode": True}

                async with session.post(
                    f"{assigned_engine_url}/step", json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    PACE_DEBUG(f"Received new gen token at {time.time()}")
                    if result.get("status") != "success":
                        PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                        req.status = RequestStatus.ERROR
                        yield "ERROR"
                        break

                    if is_prefill:
                        req.req_stats["prefill_finished_at"] = time.time()
                        results = result.get("results", [])
                        result_data = results[0].get("result", {}) if results else {}
                        if result_data == {}:
                            result_data = result.get("results")[0]
                        is_prefill = False
                    else:
                        result_data = result.get("result", {})
                    PACE_DEBUG(f"Result data (req={engine_req_id}): {result_data}")

                    if "error" in result_data:
                        PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                        req.status = RequestStatus.ERROR
                        yield "ERROR"
                        break
                    req_result = result_data.get(engine_req_id, {})
                    if "error" in req_result:
                        PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                        req.status = RequestStatus.ERROR
                        yield "ERROR"
                        break
                    status = req_result.get("status")
                    token_ids = req_result.get("token_ids", [])

                    req.req_stats["generated_tokens_count"] += len(token_ids)
                    if token_ids:
                        yield _step_payload(token_ids, req_result)

                    if status == "COMPLETED":
                        req.finish_reason = req_result.get("stop_reason", "stop")
                        break

            except Exception as e:
                PACE_ERROR(
                    f"Error generating token for request {engine_req_id}: {str(e)}"
                )
                req.status = RequestStatus.ERROR
                yield "ERROR"
                break

        await clear_concluded_request(engine_req_id, session, assigned_engine_url)
    yield None


async def perform_prefill(
    req: Request,
    worker_id: int,
    fallback_engine_url: str,
) -> str:
    """Execute the prefill step for a single request. Returns status string.

    Return values: "OK", "COMPLETED", or "ERROR".
    Engine now returns token_id only (no decoded text).
    """
    req.status = RequestStatus.PROCESSING
    PACE_DEBUG(f"[WORKER-{worker_id}] Performing PREFILL for request {req.request_id}")

    try:
        gen_config = req.engine_gen_config
        prompt = req.req.prompt
        engine_req_id = req.request_id
        assigned_engine_url = req.assigned_engine_url or fallback_engine_url

        PACE_DEBUG(
            f"Worker {worker_id} using engine: {assigned_engine_url} for prefill"
        )
        PACE_DEBUG(
            f"Prompt token count for prefill (req={engine_req_id}): {len(prompt) if isinstance(prompt, list) else 'N/A'}"
        )

        async with aiohttp.ClientSession(timeout=_make_timeout()) as session:
            payload = {
                "prefill_batch": [
                    {
                        "is_prefill": True,
                        "prompt": prompt,
                        "req_id": engine_req_id,
                        "generation_config": gen_config,
                    }
                ]
            }

            async with session.post(
                f"{assigned_engine_url}/step", json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                req.req_stats["prefill_finished_at"] = time.time()

                if result.get("status") != "success":
                    PACE_ERROR(f"Server error during prefill: {result}")
                    await clear_concluded_request(
                        req.request_id, session, assigned_engine_url
                    )
                    return "ERROR"

                results = result.get("results", [])
                if not results:
                    PACE_ERROR("Empty results in prefill response")
                    await clear_concluded_request(
                        req.request_id, session, assigned_engine_url
                    )
                    return "ERROR"

                result_data = results[0].get("result", {})
                req_result = result_data.get(engine_req_id, {})

                prefill_status = req_result.get("status", "")
                token_ids = req_result.get("token_ids", [])

                if prefill_status == "COMPLETED":
                    PACE_DEBUG(
                        f"Request {engine_req_id} fully completed during prefill"
                    )
                    req.status = RequestStatus.COMPLETED
                    req.finish_reason = req_result.get("stop_reason", "stop")
                    req.req_stats["finished_at"] = time.time()
                    req.req_stats["generated_tokens_count"] += len(token_ids)
                    if req.token_queue:
                        if token_ids:
                            await req.token_queue.put(
                                _step_payload(token_ids, req_result)
                            )
                        await req.token_queue.put(None)
                    return "COMPLETED"

                if prefill_status != "PREFILL_COMPLETED":
                    PACE_ERROR(f"Prefill failed, status: {prefill_status}")
                    await clear_concluded_request(
                        req.request_id, session, assigned_engine_url
                    )
                    return "ERROR"

                PACE_DEBUG(
                    f"[WORKER-{worker_id}] PREFILL successful for request {engine_req_id}"
                )

                if req.token_queue and token_ids:
                    req.req_stats["generated_tokens_count"] += len(token_ids)
                    await req.token_queue.put(_step_payload(token_ids, req_result))

                return "OK"

    except Exception as e:
        PACE_ERROR(f"Error during prefill for request {req.request_id}: {str(e)}")
        return "ERROR"


async def decode_step(
    session: aiohttp.ClientSession,
    worker_id: int,
    assigned_engine_url: str,
    decode_queue: Dict[str, Request],
    finalize_callback,
) -> bool:
    """Execute a single decode step across all requests in *decode_queue*.

    Returns True on success, False if the entire batch errored.
    *finalize_callback* is called for each completed/errored request.
    """
    num_requests = len(decode_queue)
    PACE_DEBUG(
        f"[WORKER-{worker_id}] Executing DECODE step for {num_requests} request(s)"
    )

    payload = {"is_decode": True}
    async with session.post(f"{assigned_engine_url}/step", json=payload) as response:
        response.raise_for_status()
        result = await response.json()

        if result.get("status") != "success":
            PACE_WARNING(f"[WORKER-{worker_id}] Server error during DECODE: {result}")
            for req_id in list(decode_queue):
                req = decode_queue[req_id]
                await finalize_callback(
                    req, req_id, "ERROR", "ERROR", worker_id, session
                )
            return False

        result_data = result.get("result", {})
        PACE_DEBUG(f"[WORKER-{worker_id}] DECODE result data: {result_data}")

        for req_id, decode_result in result_data.items():
            if not isinstance(decode_result, dict):
                continue
            PACE_DEBUG(
                f"[WORKER-{worker_id}] DECODE step for request {req_id}: {decode_result}"
            )
            status = decode_result.get("status", "DECODING_IN_PROGRESS")
            token_ids = decode_result.get("token_ids", [])

            if req_id in decode_queue:
                req = decode_queue[req_id]
                req.req_stats["generated_tokens_count"] += len(token_ids)
                if token_ids:
                    await req.token_queue.put(_step_payload(token_ids, decode_result))

                if status in ("COMPLETED", "ERROR"):
                    if status == "COMPLETED":
                        req.finish_reason = decode_result.get("stop_reason", "stop")
                    final_token = None if status == "COMPLETED" else "ERROR"
                    await finalize_callback(
                        req, req_id, status, final_token, worker_id, session
                    )
    return True


# ---------------------------------------------------------------------------
# Continuous-batch scheduler helpers (POST /step_auto_mixed)
# ---------------------------------------------------------------------------


def build_mixed_step_payload(
    work_items: List[WorkItem],
) -> Tuple[Dict, Dict[str, Request]]:
    """Build the HTTP payload for ``/step_auto_mixed`` and the prefill request map.

    Returns (payload_dict, prefill_req_map).
    """
    prefill_batch: List[dict] = []
    req_map: Dict[str, Request] = {}
    chunk_info_map: Dict[str, list] = {}

    for req, chunk_start, chunk_size in work_items:
        if req.req_stats.end_wait_time is None:
            req.status = RequestStatus.PROCESSING
            req.req_stats.end_wait_time = time.time()

        is_continuation = chunk_start > 0
        prefill_batch.append(
            {
                "is_prefill": True,
                "prompt": [] if is_continuation else req.req.prompt,
                "req_id": req.request_id,
                "generation_config": req.engine_gen_config,
            }
        )
        req_map[req.request_id] = req
        chunk_info_map[req.request_id] = [chunk_start, chunk_size]

    payload: Dict = {"prefill_batch": prefill_batch}
    if chunk_info_map:
        payload["chunk_info_map"] = chunk_info_map

    return payload, req_map


async def mixed_batch_step(
    session: aiohttp.ClientSession,
    engine_url: str,
    payload: Dict,
) -> Optional[Dict]:
    """POST ``/step_auto_mixed`` and return parsed JSON, or None on failure.

    On HTTP or protocol error, logs a warning and returns None so the
    caller can error-out affected prefills.
    """
    try:
        async with session.post(
            f"{engine_url}/step_auto_mixed", json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()
    except Exception as e:
        PACE_WARNING(f"step_auto_mixed HTTP error: {e}\n{traceback.format_exc()}")
        return None

    if result.get("status") != "success":
        PACE_WARNING(f"step_auto_mixed returned non-success: {result}")
        return None

    return result


async def process_mixed_step_prefills(
    result: Dict,
    prefill_req_map: Dict[str, Request],
    decode_queue: Dict[str, Request],
    chunked_queue: Dict[str, tuple],
    session: aiohttp.ClientSession,
    engine_url: str,
    cleanup_callback: Callable,
) -> None:
    """Process ``prefill_result`` from a mixed step response.

    Handles PREFILL_IN_PROGRESS (update chunked queue), PREFILL_COMPLETED
    (forward first token, move to decode queue), COMPLETED, and ERROR.
    """
    for req_id, entry in result.get("prefill_result", {}).items():
        req = prefill_req_map.get(req_id)
        if req is None or not isinstance(entry, dict):
            continue

        chunked_queue.pop(req_id, None)
        status = entry.get("status", "")

        if "error" in entry:
            PACE_WARNING(f"Prefill error for {req_id}: {entry['error']}")
            await error_out_requests([req], session, engine_url, cleanup_callback)
            continue

        if status == "PREFILL_IN_PROGRESS":
            num_computed = entry.get("num_computed_tokens", 0)
            actual_input_len = entry.get("actual_input_length")
            if actual_input_len is not None:
                req.req_stats.input_length = actual_input_len
            chunked_queue[req_id] = (req, num_computed)
            PACE_DEBUG(
                f"Chunked prefill {req_id[:8]}: "
                f"{num_computed} computed"
                f"{f', actual_len={actual_input_len}' if actual_input_len else ''}"
            )
            continue

        if status == "COMPLETED":
            req.status = RequestStatus.COMPLETED
            req.finish_reason = entry.get("stop_reason", "stop")
            req.req_stats.finished_at = time.time()
            req.req_stats.prefill_finished_at = time.time()
            token_ids = entry.get("token_ids", [])
            if req.token_queue and token_ids:
                await req.token_queue.put(_step_payload(token_ids, entry))
                req.req_stats.generated_tokens_count += len(token_ids)
            if req.token_queue:
                await req.token_queue.put(None)
            await clear_concluded_request(req_id, session, engine_url)
            asyncio.create_task(cleanup_callback(req))
            continue

        if status == "PREFILL_COMPLETED":
            req.req_stats.prefill_finished_at = time.time()
            token_ids = entry.get("token_ids", [])
            if req.token_queue and token_ids:
                await req.token_queue.put(_step_payload(token_ids, entry))
                req.req_stats.generated_tokens_count += len(token_ids)
            decode_queue[req_id] = req
            continue

        PACE_WARNING(f"Prefill failed for {req_id}, status: {status}")
        await error_out_requests([req], session, engine_url, cleanup_callback)


async def process_mixed_step_decodes(
    result: Dict,
    decode_queue: Dict[str, Request],
    session: aiohttp.ClientSession,
    engine_url: str,
    cleanup_callback: Callable,
) -> None:
    """Process ``decode_result`` from a mixed step response.

    Forwards token_ids to client queues and finalizes completed/errored requests.
    """
    concluded_ids = []
    for req_id, entry in result.get("decode_result", {}).items():
        if not isinstance(entry, dict):
            continue
        req = decode_queue.get(req_id)
        if req is None:
            continue
        status = entry.get("status", "DECODING_IN_PROGRESS")
        token_ids = entry.get("token_ids", [])

        if req.token_queue and token_ids:
            # When the engine attached logprobs, forward the step as a single
            # numeric payload so token IDs stay aligned with their logprobs.
            # Otherwise preserve per-token streaming granularity.
            if entry.get("logprobs") is not None:
                await req.token_queue.put(_step_payload(token_ids, entry))
            else:
                for tid in token_ids:
                    await req.token_queue.put([tid])
            req.req_stats.generated_tokens_count += len(token_ids)

        if status in ("COMPLETED", "ERROR") and req_id in decode_queue:
            req = decode_queue.pop(req_id)
            req.status = (
                RequestStatus.COMPLETED
                if status == "COMPLETED"
                else RequestStatus.ERROR
            )
            if status == "COMPLETED":
                req.finish_reason = entry.get("stop_reason", "stop")
            req.req_stats.finished_at = time.time()
            if req.token_queue:
                if status == "COMPLETED":
                    await req.token_queue.put(None)
                else:
                    await req.token_queue.put("ERROR")
                    await req.token_queue.put(None)
            concluded_ids.append(req_id)
            asyncio.create_task(cleanup_callback(req))

    if concluded_ids:
        await clear_concluded_request(concluded_ids, session, engine_url)


async def error_out_requests(
    reqs: List[Request],
    session: aiohttp.ClientSession,
    engine_url: str,
    cleanup_callback: Callable,
) -> None:
    """Transition requests to ERROR, notify clients, and batch engine removal."""
    concluded_ids = []
    for req in reqs:
        req.status = RequestStatus.ERROR
        req.req_stats.finished_at = time.time()
        if req.token_queue:
            await req.token_queue.put("ERROR")
            await req.token_queue.put(None)
        concluded_ids.append(req.request_id)
        asyncio.create_task(cleanup_callback(req))

    if concluded_ids:
        await clear_concluded_request(concluded_ids, session, engine_url)
