# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Continuous-batch scheduler: one ``/step_auto_mixed`` call per engine step.

Each worker combines pending prefills with in-flight decodes into a single
mixed-batch forward pass.  Chunked prefill is always on: prompts exceeding
the remaining token budget are split across steps.

Admission invariant:
  ``max_num_running_reqs * tokens_per_decode <= token_budget``
New prefills are admitted only while ``num_running < max_num_running_reqs``.
"""

import asyncio
import time
import traceback
from typing import Dict, List, Set, Tuple, Union

import aiohttp

from pace.server.router.engine_client import (
    _make_timeout,
    build_mixed_step_payload,
    clear_concluded_request,
    error_out_requests,
    mixed_batch_step,
    process_mixed_step_decodes,
    process_mixed_step_prefills,
)
from pace.server.router.scheduler import Scheduler
from pace.server.router.utils import Request, RequestStatus
from pace.utils.logging import PACE_DEBUG, PACE_INFO, PACE_WARNING

WorkItem = Tuple[Request, int, int]


class ContinuousBatchScheduler(Scheduler):
    """Continuous-batching scheduler using ``/step_auto_mixed``."""

    def __init__(
        self,
        engine_url: Union[str, List[str]] = "http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
        max_num_running_reqs: int = 128,
        token_budget: int = 2048,
        max_prefill_chunk: int = 0,
        tokens_per_decode: int = 1,
    ):
        super().__init__(engine_url, scheduler_metrics_enabled)

        self.decode_queues: List[Dict[str, Request]] = [
            {} for _ in range(self.num_engines)
        ]
        self.chunked_queues: List[Dict[str, tuple]] = [
            {} for _ in range(self.num_engines)
        ]
        self.cancelled_requests_per_worker: List[Set[str]] = [
            set() for _ in range(self.num_engines)
        ]

        self.token_budget = token_budget
        self.max_prefill_chunk = max_prefill_chunk
        self.tokens_per_decode = max(1, tokens_per_decode)

        admission_cap = token_budget // self.tokens_per_decode
        if max_num_running_reqs > admission_cap:
            PACE_WARNING(
                f"max_num_running_reqs={max_num_running_reqs} exceeds "
                f"token_budget // tokens_per_decode={admission_cap}; clamping"
            )
        self.max_num_running_reqs = max(1, min(max_num_running_reqs, admission_cap))

    def _chunk_size(self, tokens: int) -> int:
        """Cap *tokens* to ``max_prefill_chunk`` when set."""
        if self.max_prefill_chunk > 0 and tokens > self.max_prefill_chunk:
            return self.max_prefill_chunk
        return tokens

    def _schedule(self, num_running: int, worker_id: int = 0) -> List[WorkItem]:
        """Schedule prefill work into the remaining token budget for this step."""
        decode_cost = num_running * self.tokens_per_decode
        assert decode_cost <= self.token_budget
        remaining = self.token_budget - decode_cost

        chunked = self.chunked_queues[worker_id]
        waiting = self.request_queues[worker_id]
        scheduled: List[WorkItem] = []
        used = 0
        new_admissions = 0

        # 1. Continue in-flight chunked prefills (already occupy a running slot).
        for rid, (req, num_computed) in list(chunked.items()):
            if used >= remaining:
                break
            prompt_len = req.req_stats.input_length
            left = prompt_len - num_computed
            if left <= 0:
                continue
            chunk = self._chunk_size(min(left, remaining - used))
            if chunk <= 0:
                continue
            scheduled.append((req, num_computed, chunk))
            used += chunk

        # 2. Admit new prefills while a running slot is free.
        # Occupancy = running decodes + in-flight chunks + newly admitted this step.
        cancelled = self.cancelled_requests_per_worker[worker_id]
        while used < remaining and not waiting.empty():
            if num_running + len(chunked) + new_admissions >= self.max_num_running_reqs:
                break
            try:
                req = waiting.get_nowait()
            except asyncio.QueueEmpty:
                break
            if req.request_id in cancelled:
                self._finalize_cancelled_queued(req, cancelled)
                continue
            prompt_len = req.req_stats.input_length
            chunk = self._chunk_size(min(prompt_len, remaining - used))
            if chunk <= 0:
                waiting.put_nowait(req)
                break
            scheduled.append((req, 0, chunk))
            used += chunk
            new_admissions += 1

        PACE_DEBUG(
            f"[WORKER-{worker_id}] token_budget={self.token_budget} "
            f"running={num_running} decode_cost={decode_cost} "
            f"scheduled={len(scheduled)} prefills ({used} tokens)"
        )
        return scheduled

    async def _processing_loop_worker(self, worker_id: int) -> None:
        engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        waiting = self.request_queues[worker_id]
        running = self.decode_queues[worker_id]
        chunked = self.chunked_queues[worker_id]

        PACE_INFO(f"[CBS WORKER-{worker_id}] Started (engine: {engine_url})")

        try:
            async with aiohttp.ClientSession(timeout=_make_timeout()) as session:
                while self.is_running:
                    # Drain cancellations
                    cancelled = self.cancelled_requests_per_worker[worker_id]
                    if cancelled:
                        handled_ids = []
                        for rid in list(cancelled):
                            chunk_entry = chunked.pop(rid, None)
                            req = None
                            if rid in running:
                                req = running.pop(rid)
                            elif chunk_entry is not None:
                                req = chunk_entry[0]
                            if req is not None:
                                # Preserve a stop-string completion; only a real
                                # client cancel becomes ERROR.
                                if req.status != RequestStatus.COMPLETED:
                                    req.status = RequestStatus.ERROR
                                req.req_stats.finished_at = time.time()
                                asyncio.create_task(self._cleanup_request(req))
                                handled_ids.append(rid)
                            elif rid not in self.active_requests:
                                cancelled.discard(rid)
                        for rid in handled_ids:
                            cancelled.discard(rid)
                        if handled_ids:
                            await clear_concluded_request(
                                handled_ids, session, engine_url
                            )

                    num_running = len(running)
                    work_items = self._schedule(num_running, worker_id)

                    # Block when fully idle
                    if not work_items and num_running == 0:
                        first = await waiting.get()
                        cancelled = self.cancelled_requests_per_worker[worker_id]
                        if first.request_id in cancelled:
                            self._finalize_cancelled_queued(first, cancelled)
                            continue
                        prompt_len = first.req_stats.input_length
                        chunk = self._chunk_size(min(prompt_len, self.token_budget))
                        work_items = [(first, 0, max(1, chunk))]

                    # Refresh gauges after waiting->running promotions this step.
                    self._update_gauges()

                    payload, prefill_req_map = build_mixed_step_payload(work_items)
                    PACE_DEBUG(
                        f"[WORKER-{worker_id}] step: "
                        f"prefills={len(work_items)} running={num_running}"
                    )

                    result = await mixed_batch_step(session, engine_url, payload)
                    if result is None:
                        failed_reqs = []
                        for req, _, _ in work_items:
                            chunked.pop(req.request_id, None)
                            failed_reqs.append(req)
                        failed_reqs.extend(running.values())
                        running.clear()
                        await error_out_requests(
                            failed_reqs, session, engine_url, self._cleanup_request
                        )
                        continue

                    await process_mixed_step_prefills(
                        result,
                        prefill_req_map,
                        running,
                        chunked,
                        session,
                        engine_url,
                        self._cleanup_request,
                    )
                    await process_mixed_step_decodes(
                        result,
                        running,
                        session,
                        engine_url,
                        self._cleanup_request,
                    )

        except asyncio.CancelledError:
            PACE_DEBUG(f"[CBS WORKER-{worker_id}] Cancelled")
        except Exception as e:
            PACE_WARNING(
                f"[CBS WORKER-{worker_id}] Fatal: {e}\n{traceback.format_exc()}"
            )
