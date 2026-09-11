# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Core scheduling logic for managing inference request queues and workers.

Engine HTTP communication lives in engine_client.py; per-request and
session-wide statistics live in metrics.py.  This module owns:
  - request queue management and round-robin engine assignment
  - worker lifecycle (start / stop / processing loops)
  - decode-loop orchestration (PrefillFirstScheduler)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

import aiohttp

from pace.server.router.engine_client import (
    clear_concluded_request,
    decode_step,
    iterative_generate_tokens,
    perform_prefill,
    _make_timeout,
)
from pace.server.router.metrics import SchedulerMetrics
from pace.server.router.utils import (
    Request,
    RequestStatus,
    requests_running_gauge,
    requests_waiting_gauge,
)
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_INFO, PACE_WARNING


class Scheduler(ABC):
    """Base scheduler with common queue management and worker lifecycle."""

    def __init__(
        self,
        engine_url="http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
    ):
        if isinstance(engine_url, list):
            self.engine_urls = engine_url
            self.engine_url = engine_url[0]
            self.num_engines = len(engine_url)
            self.current_engine_index = 0
            PACE_INFO(f"Scheduler initialized with {self.num_engines} engine instances")
        else:
            self.engine_urls = [engine_url]
            self.engine_url = engine_url
            self.num_engines = 1
            self.current_engine_index = 0

        self.scheduler_metrics_enabled = scheduler_metrics_enabled
        self.metrics = SchedulerMetrics(enabled=scheduler_metrics_enabled)

        self.request_queues: List[asyncio.Queue[Request]] = [
            asyncio.Queue() for _ in range(self.num_engines)
        ]
        self.request_queue: asyncio.Queue[Request] = self.request_queues[0]
        self.active_requests: Dict[str, Request] = {}
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        self.cancelled_requests: Set[str] = set()
        self.session: aiohttp.ClientSession | None = None
        self._metric_tracking_task: Optional[asyncio.Task] = None
        self._cache_metrics_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the scheduler processing loops (one per engine instance)."""
        self.is_running = True

        for worker_id in range(self.num_engines):
            task = asyncio.create_task(self._processing_loop_worker(worker_id))
            self.processing_tasks.append(task)
            PACE_INFO(f"Started processing worker {worker_id}")

        if self.scheduler_metrics_enabled:
            self._metric_tracking_task = asyncio.create_task(
                self.metrics.server_metrics_loop(
                    lambda: self.is_running,
                    self.get_queue_size,
                )
            )
        self._cache_metrics_task = asyncio.create_task(
            self.metrics.cache_metrics_loop(
                lambda: self.is_running,
                self.engine_urls,
            )
        )
        PACE_INFO(
            f"{self.__class__.__name__} started with {self.num_engines} parallel workers"
        )

    async def stop(self):
        """Stop the scheduler processing loops."""
        self.is_running = False
        for task in self.processing_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for task in (self._metric_tracking_task, self._cache_metrics_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.session:
            await self.session.close()
        PACE_INFO(f"{self.__class__.__name__} stopped")

    def _update_gauges(self):
        """Refresh Prometheus running/waiting gauges from current state."""
        queue_size = self.get_queue_size()
        requests_waiting_gauge.set(queue_size)
        requests_running_gauge.set(
            max(0, self.get_active_requests_count() - queue_size)
        )

    def _finalize_cancelled_queued(self, req: Request, cancelled: Set[str]):
        """Finalize a request cancelled while still queued (never sent to engine)."""
        cancelled.discard(req.request_id)
        if req.req_stats.get("finished_at") is None:
            req.req_stats["finished_at"] = time.time()
        PACE_INFO(
            f"Discarding cancelled queued request {req.request_id} before prefill"
        )
        asyncio.create_task(self._cleanup_request(req))

    async def submit_request(self, req: Request):
        """Submit a request: assign an engine via round-robin, enqueue for its worker."""
        assigned_engine_url = self.get_next_engine_url()
        req.assigned_engine_url = assigned_engine_url
        req.assigned_engine_index = self.engine_urls.index(assigned_engine_url)

        self.active_requests[req.request_id] = req
        await self.request_queues[req.assigned_engine_index].put(req)
        self._update_gauges()
        PACE_INFO(
            f"Request {req.request_id} submitted to queue and assigned to engine: {assigned_engine_url} (index: {req.assigned_engine_index})"
        )

    async def _cleanup_request(self, req: Request):
        """Delegate cleanup and metrics to SchedulerMetrics."""
        await self.metrics.cleanup_request(req, self.active_requests)
        self._update_gauges()

    async def get_request_status(self, request_id: str) -> Optional[Dict]:
        """Get the status of a specific request."""
        req = self.active_requests.get(request_id)
        if req:
            created_at = req.req_stats.get("created_at")
            created_at_iso = (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created_at))
                if isinstance(created_at, (int, float))
                else None
            )
            return {
                "status": req.status,
                "message": f"Request {request_id} is {req.status}",
                "created_at": created_at_iso,
            }
        return None

    def get_queue_size(self) -> int:
        return sum(q.qsize() for q in self.request_queues)

    def get_active_requests_count(self) -> int:
        return len(self.active_requests)

    def get_next_engine_url(self) -> str:
        """Round-robin engine selection."""
        if self.num_engines == 1:
            return self.engine_urls[0]
        selected_index = self.current_engine_index
        url = self.engine_urls[selected_index]
        self.current_engine_index = (self.current_engine_index + 1) % self.num_engines
        PACE_DEBUG(f"Selected engine URL: {url} (index: {selected_index})")
        return url

    def server_metrics(self) -> Dict[str, float]:
        return self.metrics.server_metrics_snapshot()

    @abstractmethod
    async def _processing_loop_worker(self, worker_id: int): ...


class IterativeScheduler(Scheduler):
    """Scheduler that processes one request at a time per engine instance."""

    def __init__(
        self,
        engine_url: str = "http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
    ):
        super().__init__(engine_url, scheduler_metrics_enabled)

    async def _processing_loop_worker(self, worker_id: int):
        assigned_engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        PACE_INFO(
            f"Worker {worker_id} started (bound to engine: {assigned_engine_url})"
        )

        worker_queue = self.request_queues[worker_id]

        while self.is_running:
            try:
                req = await worker_queue.get()
                self._update_gauges()
                if req.request_id in self.cancelled_requests:
                    self._finalize_cancelled_queued(req, self.cancelled_requests)
                    continue
                PACE_INFO(
                    f"Worker {worker_id} picked up request {req.request_id} from dedicated queue"
                )
                await self._process_request(req)
            except asyncio.CancelledError:
                PACE_INFO(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                PACE_WARNING(f"Worker {worker_id} error: {str(e)}")
        PACE_INFO(f"Worker {worker_id} stopped")

    async def _process_request(self, req: Request):
        """Process a single request using the engine_client token generator."""
        req.status = RequestStatus.PROCESSING
        req.req_stats["end_wait_time"] = time.time()
        worker_id = getattr(req, "assigned_engine_index", 0)
        PACE_INFO(f"[WORKER-{worker_id}] Processing request {req.request_id}")

        try:
            async for token_data in iterative_generate_tokens(
                req, self.cancelled_requests, self.engine_url
            ):
                if token_data is not None:
                    await req.token_queue.put(token_data)
            await req.token_queue.put(None)

            if req.status != RequestStatus.ERROR:
                req.status = RequestStatus.COMPLETED
            req.req_stats["finished_at"] = time.time()
            PACE_INFO(
                f"Request {req.request_id} finished with status {req.status.name}"
            )
        except Exception as e:
            PACE_ERROR(f"Error processing request {req.request_id}: {str(e)}")
            req.status = RequestStatus.ERROR
            req.req_stats["finished_at"] = time.time()
            await req.token_queue.put("ERROR")
            await req.token_queue.put(None)

        self.cancelled_requests.discard(req.request_id)
        asyncio.create_task(self._cleanup_request(req))


class PrefillFirstScheduler(Scheduler):
    """Scheduler that prefills each request first, then interleaves decode."""

    def __init__(
        self,
        engine_url: str = "http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
    ):
        super().__init__(engine_url, scheduler_metrics_enabled)
        self.decode_queues: List[Dict[str, Request]] = [
            {} for _ in range(self.num_engines)
        ]
        self._decode_tasks: List[Optional[asyncio.Task]] = [
            None for _ in range(self.num_engines)
        ]
        self._stop_events: List[asyncio.Event] = [
            asyncio.Event() for _ in range(self.num_engines)
        ]

    async def start_decode_loop(self, worker_id: int):
        if (
            self._decode_tasks[worker_id] is None
            or self._decode_tasks[worker_id].done()
        ):
            self._decode_tasks[worker_id] = asyncio.create_task(
                self.decode_loop(worker_id)
            )
            PACE_INFO(f"Started decode loop for worker {worker_id}")

    async def stop_decode_loop(self, worker_id: int):
        if self._decode_tasks[worker_id]:
            PACE_INFO(
                f"Requesting graceful stop (no cancel) for decode loop {worker_id}"
            )
            self._stop_events[worker_id].set()
            await self._decode_tasks[worker_id]
            PACE_INFO(f"Decode loop {worker_id} stopped")

    async def stop(self):
        for worker_id in range(self.num_engines):
            await self.stop_decode_loop(worker_id)
        await super().stop()

    async def _processing_loop_worker(self, worker_id: int):
        assigned_engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        PACE_INFO(
            f"PrefillFirst Worker {worker_id} started (bound to engine: {assigned_engine_url})"
        )

        worker_queue = self.request_queues[worker_id]

        while self.is_running:
            try:
                req = await worker_queue.get()

                if (
                    self._decode_tasks[worker_id]
                    and not self._decode_tasks[worker_id].done()
                ):
                    PACE_DEBUG(f"Worker {worker_id} stopping decode for prefill batch")
                    await self.stop_decode_loop(worker_id)

                pending = [req]
                while not worker_queue.empty():
                    try:
                        pending.append(worker_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                self._update_gauges()

                PACE_DEBUG(f"Worker {worker_id} draining {len(pending)} prefill(s)")

                for r in pending:
                    if r.request_id in self.cancelled_requests:
                        self._finalize_cancelled_queued(r, self.cancelled_requests)
                        continue
                    r.status = RequestStatus.PROCESSING
                    r.req_stats["end_wait_time"] = time.time()
                    result = await perform_prefill(r, worker_id, self.engine_url)
                    if result == "ERROR":
                        r.status = RequestStatus.ERROR
                        r.req_stats["finished_at"] = time.time()
                        await r.token_queue.put("ERROR")
                        await r.token_queue.put(None)
                        asyncio.create_task(self._cleanup_request(r))
                        continue

                    if result == "COMPLETED":
                        asyncio.create_task(self._cleanup_request(r))
                        continue

                    self.decode_queues[worker_id][r.request_id] = r

                if self.decode_queues[worker_id]:
                    self._stop_events[worker_id].clear()
                    await self.start_decode_loop(worker_id)

            except asyncio.CancelledError:
                PACE_INFO(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                PACE_ERROR(f"Worker {worker_id} error: {str(e)}")

        PACE_INFO(f"PrefillFirst Worker {worker_id} stopped")

    async def _finalize_decode_request(
        self,
        req: Request,
        req_id: str,
        status: str,
        final_token: Optional[str],
        worker_id: int,
        session: aiohttp.ClientSession,
    ):
        """Helper to finalize a request in the decode queue."""
        self.decode_queues[worker_id].pop(req_id, None)
        req.status = (
            RequestStatus.COMPLETED if status == "COMPLETED" else RequestStatus.ERROR
        )
        req.req_stats["finished_at"] = time.time()
        if status != "COMPLETED" and final_token is None:
            final_token = "ERROR"
        await req.token_queue.put(final_token)
        await clear_concluded_request(req_id, session, req.assigned_engine_url)
        asyncio.create_task(self._cleanup_request(req))

    async def decode_loop(self, worker_id: int):
        """Interleave decode steps across requests for a specific worker."""
        assigned_engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        PACE_INFO(f"Decode loop {worker_id} started (engine: {assigned_engine_url})")

        try:
            async with aiohttp.ClientSession(timeout=_make_timeout()) as session:
                while self.decode_queues[worker_id]:
                    if self._stop_events[worker_id].is_set():
                        PACE_INFO(
                            f"Worker {worker_id} decode loop: Stop requested; exiting after finishing current step"
                        )
                        break
                    for req_id in list(self.cancelled_requests):
                        if req_id in self.decode_queues[worker_id]:
                            PACE_INFO(
                                f"Worker {worker_id} clearing cancelled request {req_id} from decode loop"
                            )
                            await self._clear_cancelled_request(
                                req_id, session, worker_id
                            )
                            self.cancelled_requests.discard(req_id)
                    PACE_DEBUG(
                        f"Worker {worker_id} decode loop active with {len(self.decode_queues[worker_id])} requests"
                    )
                    ok = await decode_step(
                        session,
                        worker_id,
                        assigned_engine_url,
                        self.decode_queues[worker_id],
                        self._finalize_decode_request,
                    )
                    if not ok:
                        break

        except Exception as e:
            PACE_WARNING(f"Error in worker {worker_id} decode loop: {e}")
            try:
                async with aiohttp.ClientSession() as err_session:
                    for req_id in list(self.decode_queues[worker_id]):
                        req = self.decode_queues[worker_id][req_id]
                        await self._finalize_decode_request(
                            req, req_id, "ERROR", None, worker_id, err_session
                        )
            except Exception:
                PACE_WARNING(
                    f"Failed to finalize remaining requests for worker {worker_id} after decode loop error"
                )
        finally:
            self._decode_tasks[worker_id] = None
            PACE_INFO(f"Decode loop {worker_id} finished")

    async def _clear_cancelled_request(
        self, req_id: str, session: aiohttp.ClientSession, worker_id: int
    ):
        """Remove a cancelled request from decode queue and notify engine."""
        if req_id in self.decode_queues[worker_id]:
            req = self.decode_queues[worker_id][req_id]
            engine_url = req.assigned_engine_url
            del self.decode_queues[worker_id][req_id]
            PACE_INFO(
                f"Worker {worker_id} cleared cancelled request {req_id} from decode queue"
            )
            await clear_concluded_request(req_id, session, engine_url)
            if req.req_stats.get("finished_at") is None:
                req.req_stats["finished_at"] = time.time()
            asyncio.create_task(self._cleanup_request(req))
