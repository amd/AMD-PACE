# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_INFO, PACE_WARNING
from pace.server.router.utils import (
    Request,
    RequestStatus,
    http_config,
    ttft_histogram,
    tpot_histogram,
)
from typing import Dict, Optional, AsyncGenerator, List, Tuple

"""Schedulers for managing inference requests and processing them asynchronously.

All scheduler types (IterativeScheduler, PrefillFirstScheduler, and ContinuousBatchScheduler)
support MLPerf mode: when req.mlperf_mode is True, the engine is told to skip text decoding
and return token_id only; the scheduler forwards token_id (or text in generation mode) to the client.
"""


class Scheduler(ABC):
    """Base scheduler with common functionality."""

    def __init__(
        self,
        engine_url="http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
    ):
        # Support both single URL (string) and multiple URLs (list)
        if isinstance(engine_url, list):
            self.engine_urls = engine_url
            self.engine_url = engine_url[0]  # Keep for backward compatibility
            self.num_engines = len(engine_url)
            self.current_engine_index = 0
            PACE_INFO(f"Scheduler initialized with {self.num_engines} engine instances")
        else:
            self.engine_urls = [engine_url]
            self.engine_url = engine_url
            self.num_engines = 1
            self.current_engine_index = 0

        self.scheduler_metrics_enabled = scheduler_metrics_enabled
        # Create separate queues for each worker to avoid deadlock
        self.request_queues: List[asyncio.Queue[Request]] = [
            asyncio.Queue() for _ in range(self.num_engines)
        ]
        self.request_queue: asyncio.Queue[Request] = self.request_queues[
            0
        ]  # For backward compatibility
        self.active_requests: Dict[str, Request] = {}
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        self.ttft_intervals: List[Tuple[float, float]] = []
        self.processing_intervals: List[Tuple[float, float]] = []
        self.active_ttft_time: float = 0.0
        self.active_processing_time: float = 0.0
        self.session_ttft: float = 0.0
        self.session_tpot: float = 0.0
        self.session_requests_served_per_second: float = 0.0
        self.total_generated_tokens: int = 0
        self.active_time: float = 0.0
        self.ttft_active_time: float = 0.0
        self.total_completed_requests: int = 0
        self.metric_tracking_task: Optional[asyncio.Task] = None
        self.cancelled_requests: List[str] = []
        self.session: aiohttp.ClientSession | None = None

    async def start(self):
        """Start the scheduler processing loops (one per engine instance)."""
        self.is_running = True

        # Start num_engines worker tasks to process requests in parallel
        for worker_id in range(self.num_engines):
            task = asyncio.create_task(self._processing_loop_worker(worker_id))
            self.processing_tasks.append(task)
            PACE_INFO(f"Started processing worker {worker_id}")

        if self.scheduler_metrics_enabled:
            self.metric_tracking_task = asyncio.create_task(self._server_metrics_loop())
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
        if self.metric_tracking_task:
            self.metric_tracking_task.cancel()
            try:
                await self.metric_tracking_task
            except asyncio.CancelledError:
                pass
        if self.session:
            await self.session.close()
        PACE_INFO(f"{self.__class__.__name__} stopped")

    async def submit_request(
        self,
        req: Request = None,
    ):
        """
        Submit a new request to the queue with common request construction.
        Assigns the request to a specific engine instance using round-robin.
        """
        # Assign this request to a specific engine instance
        assigned_engine_url = self.get_next_engine_url()
        req.assigned_engine_url = assigned_engine_url

        # Also store the engine index for worker binding
        req.assigned_engine_index = self.engine_urls.index(assigned_engine_url)

        self.active_requests[req.request_id] = req
        # Put request directly into the assigned worker's queue
        await self.request_queues[req.assigned_engine_index].put(req)
        PACE_INFO(
            f"Request {req.request_id} submitted to queue and assigned to engine: {assigned_engine_url} (index: {req.assigned_engine_index})"
        )

    async def _clear_concluded_requests(
        self,
        req_id: str,
        session: aiohttp.ClientSession,
        engine_url: Optional[str] = None,
    ):
        """Remove a cancelled, finished or errored request from engine queues"""
        try:
            if engine_url is None:
                req = self.active_requests.get(req_id)
                engine_url = (
                    req.assigned_engine_url or self.engine_url
                    if req
                    else self.engine_url
                )
            PACE_INFO(f"Removing concluded sequence {req_id} from engine {engine_url}")
            payload = {"sequence_ids": [req_id]}
            async with session.post(
                f"{engine_url}/remove_sequence", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    PACE_INFO(
                        f"Successfully removed sequence {req_id} from engine: {result}"
                    )
                else:
                    PACE_ERROR(
                        f"Failed to remove sequence {req_id} from engine: {response.status}"
                    )
        except Exception as e:
            PACE_ERROR(f"Error removing sequence {req_id} from engine: {str(e)}")

    async def _cleanup_request(self, req: Request):
        """
        Clean up a completed request.
        Moves it from active_requests to completed_requests and computes stats.
        """

        request_id = req.request_id
        if request_id in self.active_requests:
            # Move to completed
            del self.active_requests[request_id]
            finished_at = req.req_stats.get("finished_at")
            prefill_finished_at = req.req_stats.get("prefill_finished_at")
            created_at = req.req_stats.get("created_at")
            gen_tokens = max(0, req.req_stats.get("generated_tokens_count", 0))

            PACE_INFO(f"Request {request_id} finished at {finished_at:.6f}")

            # Safe computations with guards
            if gen_tokens > 0:
                req.req_stats["TPOT"] = (finished_at - created_at) / gen_tokens
            if prefill_finished_at:
                req.req_stats["TTFT"] = prefill_finished_at - created_at

            if req.req_stats["TTFT"]:
                PACE_INFO(
                    f"Time to first token (TTFT) for request {request_id}: "
                    f"{req.req_stats['TTFT']:.6f}s"
                )
                ttft_histogram.observe(req.req_stats["TTFT"])
            if req.req_stats["TPOT"]:
                PACE_INFO(
                    f"Per-token time (TPOT) for request {request_id}: "
                    f"{req.req_stats['TPOT']:.6f}s"
                )
                tpot_histogram.observe(req.req_stats["TPOT"])
            PACE_INFO(f"scheduler metrics enabled: {self.scheduler_metrics_enabled}")
            if self.scheduler_metrics_enabled:
                PACE_INFO("Updating scheduler metrics...")
                # Update intervals for server metrics calculation
                if req.req_stats["TTFT"] and prefill_finished_at:
                    self.ttft_intervals.append((created_at, prefill_finished_at))
                if gen_tokens > 0:
                    self.processing_intervals.append((created_at, finished_at))
                self.active_ttft_time = self._calculate_active_time(self.ttft_intervals)
                self.active_processing_time = self._calculate_active_time(
                    self.processing_intervals
                )
            self.total_generated_tokens += gen_tokens
            self.total_completed_requests += 1
            PACE_INFO(
                f"Request spent {req.req_stats['end_wait_time'] - req.req_stats['created_at']} seconds Waiting."
            )
            PACE_INFO(f"request generated {gen_tokens} tokens")
            PACE_INFO(f"Cleaned up request {request_id}")

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
        """Get the current queue size."""
        return sum(q.qsize() for q in self.request_queues)

    def get_active_requests_count(self) -> int:
        """Get the number of active requests."""
        return len(self.active_requests)

    @abstractmethod
    async def _processing_loop_worker(self, worker_id: int):
        """Subclasses implement the worker processing loop."""
        ...

    async def _server_metrics_loop(self):
        """Periodically compute server session metrics based on completed requests."""
        while self.is_running:
            try:
                await asyncio.sleep(30)
                if self.get_queue_size() == 0:
                    PACE_INFO("Calculating server metrics...")
                    self._calculate_server_metrics()
                    self.ttft_intervals.clear()
                    self.processing_intervals.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                PACE_WARNING(f"Error in server metrics loop: {str(e)}")

    def _calculate_server_metrics(self):
        """Compute active-time based TTFT and TPOT across completed requests."""
        self.ttft_active_time = self.ttft_active_time + (
            self._calculate_active_time(self.ttft_intervals)
            if self.ttft_intervals
            else 0.0
        )
        self.session_ttft = (
            self.ttft_active_time / self.total_completed_requests
            if self.total_completed_requests
            else 0.0
        )

        self.active_time = self.active_time + (
            self._calculate_active_time(self.processing_intervals)
            if self.processing_intervals
            else 0.0
        )
        self.session_tpot = (
            (self.active_time / self.total_generated_tokens)
            if self.total_generated_tokens > 0
            else 0.0
        )

        self.session_requests_served_per_second = (
            self.total_completed_requests / self.active_time
            if self.active_time > 0
            else 0.0
        )

        PACE_INFO(
            f"Updated server metrics: session_ttft={self.session_ttft:.6f}, "
            f"session_tpot={self.session_tpot:.6f}"
        )

    def _calculate_active_time(self, intervals: List[Tuple[float, float]]) -> float:
        """Merge overlapping intervals and return cumulative active time."""
        if not intervals:
            return 0.0

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        active_time = sum(end - start for start, end in merged)
        return active_time

    def get_next_engine_url(self) -> str:
        """Get the next engine URL using round-robin load balancing."""
        if self.num_engines == 1:
            return self.engine_urls[0]

        selected_index = self.current_engine_index
        url = self.engine_urls[selected_index]
        self.current_engine_index = (self.current_engine_index + 1) % self.num_engines
        PACE_DEBUG(f"Selected engine URL: {url} (index: {selected_index})")
        return url

    def server_metrics(self) -> Dict[str, float]:
        """Return current server session metrics."""
        return {
            "sched_session_ttft": self.session_ttft,
            "sched_session_tpot": self.session_tpot,
            "sched_active_ttft_time": self.active_ttft_time,
            "sched_requests_served_per_second": self.session_requests_served_per_second,
            "sched_total_generated_tokens": self.total_generated_tokens,
        }


class IterativeScheduler(Scheduler):
    """Scheduler that processes multiple requests in parallel (one per engine instance)."""

    def __init__(
        self,
        engine_url: str = "http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
    ):
        super().__init__(engine_url, scheduler_metrics_enabled)

    async def _processing_loop_worker(self, worker_id: int):
        """
        Worker bound to a specific engine that only processes requests for that engine.
        Each worker has its own dedicated queue to avoid deadlock.
        """
        assigned_engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        PACE_INFO(
            f"Worker {worker_id} started (bound to engine: {assigned_engine_url})"
        )

        # Get this worker's dedicated queue
        worker_queue = self.request_queues[worker_id]

        while self.is_running:
            try:
                # Get next request from this worker's dedicated queue
                req = await worker_queue.get()

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
        """Process a single request by generating tokens and sending them to the token queue."""
        req.status = RequestStatus.PROCESSING
        req.req_stats["end_wait_time"] = time.time()
        worker_id = getattr(req, "assigned_engine_index", 0)
        PACE_INFO(f"[WORKER-{worker_id}] Processing request {req.request_id}")

        try:
            async for token_data in self._generate_tokens(req):
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
            await req.token_queue.put(None)

        # Schedule cleanup
        asyncio.create_task(self._cleanup_request(req))

    async def _generate_tokens(
        self, req: Request
    ) -> AsyncGenerator[Optional[str], None]:
        """Generate tokens for a request by calling the assigned engine."""

        gen_config = (
            req.req.gen_config.model_dump(exclude_none=True)
            if req.req.gen_config
            else None
        )
        PACE_DEBUG(f"Gen config: {gen_config}")
        prompt = req.req.prompt  # Expected as a list based on submit_request
        engine_req_id = req.request_id

        # Use the assigned engine URL for this request
        assigned_engine_url = req.assigned_engine_url or self.engine_url
        PACE_INFO(f"Request {engine_req_id} will use engine: {assigned_engine_url}")
        PACE_DEBUG(f"Prompt for generation (req={engine_req_id}): {prompt}")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=http_config.total,
                connect=http_config.connect,
                sock_connect=http_config.sock_connect,
                sock_read=http_config.sock_read,
            )
        ) as session:
            is_prefill = True

            while req.status not in (RequestStatus.COMPLETED, RequestStatus.ERROR):
                # Check if request was cancelled
                if engine_req_id not in self.active_requests:
                    PACE_INFO(f"Request {engine_req_id} cancelled, stopping generation")
                    break

                try:
                    worker_id = getattr(req, "assigned_engine_index", 0)
                    if is_prefill:
                        PACE_INFO(
                            f"[WORKER-{worker_id}] Executing PREFILL step for request {engine_req_id}"
                        )
                        mlperf_mode = getattr(req.req, "mlperf_mode", False)
                        payload = {
                            "prefill_batch": [
                                {
                                    "is_prefill": True,
                                    "prompt": prompt,
                                    "req_id": engine_req_id,
                                    "generation_config": gen_config,
                                    "mlperf_mode": mlperf_mode,
                                }
                            ]
                        }
                    else:
                        PACE_INFO(
                            f"[WORKER-{worker_id}] Executing DECODE step {req.req_stats['generated_tokens_count']} for request {engine_req_id}"
                        )
                        payload = {"is_decode": True}

                    async with session.post(
                        f"{assigned_engine_url}/step", json=payload
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        PACE_INFO(f"Received new gen token at {time.time()}")
                        if result.get("status") != "success":
                            PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                            req.status = RequestStatus.ERROR
                            yield "ERROR"
                            break

                        if is_prefill:
                            req.req_stats["prefill_finished_at"] = time.time()
                            results = result.get("results", [])
                            result_data = (
                                results[0].get("result", {}) if results else {}
                            )
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
                        token = req_result.get("output", "")
                        token_id = req_result.get("token_id")
                        status = req_result.get("status")

                        num_generated = req_result.get("num_tokens_generated", 1)
                        req.req_stats["generated_tokens_count"] += num_generated

                        if req.mlperf_mode and token_id is not None:
                            yield token_id
                        elif token is not None:
                            yield token

                        # Stop if engine indicates completion
                        if status == "COMPLETED":
                            break

                except Exception as e:
                    PACE_WARNING(
                        f"Error generating token for request {engine_req_id}: {str(e)}"
                    )
                    break

            # Clean up engine state and signal completion
            await self._clear_concluded_requests(
                engine_req_id, session, assigned_engine_url
            )
        yield None


class PrefillFirstScheduler(Scheduler):
    """
    Scheduler that performs prefill for each request first,
    then interleaves decode across active requests.
    Supports multiple engine instances with isolated decode loops per worker.
    """

    def __init__(
        self,
        engine_url: str = "http://localhost:8000",
        scheduler_metrics_enabled: bool = False,
    ):
        super().__init__(engine_url, scheduler_metrics_enabled)
        # Per-worker decode queues and state
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
        """Start decode loop for a specific worker."""
        if (
            self._decode_tasks[worker_id] is None
            or self._decode_tasks[worker_id].done()
        ):
            self._decode_tasks[worker_id] = asyncio.create_task(
                self.decode_loop(worker_id)
            )
            PACE_INFO(f"Started decode loop for worker {worker_id}")

    async def stop_decode_loop(self, worker_id: int):
        """Stop decode loop for a specific worker."""
        if self._decode_tasks[worker_id]:
            PACE_INFO(
                f"Requesting graceful stop (no cancel) for decode loop {worker_id}"
            )
            self._stop_events[worker_id].set()
            await self._decode_tasks[worker_id]
            PACE_INFO(f"Decode loop {worker_id} stopped")

    async def stop(self):
        """Stop the scheduler and all decode loops."""
        for worker_id in range(self.num_engines):
            await self.stop_decode_loop(worker_id)
        await super().stop()

    async def _processing_loop_worker(self, worker_id: int):
        """
        Worker bound to a specific engine that only processes requests for that engine.
        Each worker has its own dedicated queue to avoid deadlock.
        """
        assigned_engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        PACE_INFO(
            f"PrefillFirst Worker {worker_id} started (bound to engine: {assigned_engine_url})"
        )

        # Get this worker's dedicated queue
        worker_queue = self.request_queues[worker_id]

        while self.is_running:
            try:
                # Get first request, then drain all pending from queue.
                req = await worker_queue.get()

                # Stop decode once before draining all pending prefills.
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

                PACE_DEBUG(f"Worker {worker_id} draining {len(pending)} prefill(s)")

                for r in pending:
                    r.status = RequestStatus.PROCESSING
                    r.req_stats["end_wait_time"] = time.time()
                    result = await self.perform_prefill(r, worker_id)
                    if result == "ERROR":
                        r.status = RequestStatus.ERROR
                        r.req_stats["finished_at"] = time.time()
                        await r.token_queue.put("ERROR")
                        await r.token_queue.put(None)
                        asyncio.create_task(self._cleanup_request(r))
                        continue

                    if result != "COMPLETED":
                        self.decode_queues[worker_id][r.request_id] = r

                if self.decode_queues[worker_id]:
                    self._stop_events[worker_id].clear()
                    await self.start_decode_loop(worker_id)

            except asyncio.CancelledError:
                PACE_INFO(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                PACE_ERROR(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(0.5)

        PACE_INFO(f"PrefillFirst Worker {worker_id} stopped")

    async def perform_prefill(self, req: Request, worker_id: int) -> str:
        """Execute prefill step for the request using the assigned engine."""
        req.status = RequestStatus.PROCESSING
        PACE_INFO(
            f"[WORKER-{worker_id}] Performing PREFILL for request {req.request_id}"
        )

        try:
            gen_config = (
                req.req.gen_config.model_dump(exclude_none=True)
                if req.req.gen_config
                else None
            )
            prompt = req.req.prompt
            engine_req_id = req.request_id

            # Use the assigned engine URL for this request
            assigned_engine_url = req.assigned_engine_url or self.engine_url
            PACE_INFO(
                f"Worker {worker_id} using engine: {assigned_engine_url} for prefill"
            )
            PACE_DEBUG(f"Prompt for prefill (req={engine_req_id}): {prompt}")

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=http_config.total,
                    connect=http_config.connect,
                    sock_connect=http_config.sock_connect,
                    sock_read=http_config.sock_read,
                )
            ) as session:
                payload = {
                    "prefill_batch": [
                        {
                            "is_prefill": True,
                            "prompt": prompt,
                            "req_id": engine_req_id,
                            "generation_config": gen_config,
                            "mlperf_mode": req.mlperf_mode,
                        }
                    ]
                }

                async with session.post(
                    f"{assigned_engine_url}/step", json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    req.req_stats["prefill_finished_at"] = time.time()

                    # Check prefill batch status.
                    if result.get("status") != "success":
                        PACE_ERROR(f"Server error during prefill: {result}")
                        await self._clear_concluded_requests(
                            req.request_id, session, assigned_engine_url
                        )
                        return "ERROR"

                    # Extract the single result from results list, as batch contains only one prefill.
                    results = result.get("results", [])
                    if not results:
                        PACE_ERROR("Empty results in prefill response")
                        await self._clear_concluded_requests(
                            req.request_id, session, assigned_engine_url
                        )
                        return "ERROR"

                    result_data = results[0].get("result", {})
                    req_result = result_data.get(engine_req_id, {})

                    # Verify prefill status and extract token if successful.
                    prefill_status = req_result.get("status", "")
                    token = req_result.get("output", "")
                    token_id = req_result.get("token_id")

                    if prefill_status == "COMPLETED":
                        PACE_INFO(
                            f"Request {engine_req_id} fully completed during prefill"
                        )
                        req.status = RequestStatus.COMPLETED
                        req.req_stats["finished_at"] = time.time()
                        req.req_stats["generated_tokens_count"] += 1
                        if req.token_queue:
                            first_token = (
                                token_id
                                if (req.mlperf_mode and token_id is not None)
                                else (token if token is not None else None)
                            )
                            if first_token is not None:
                                await req.token_queue.put(first_token)
                            await req.token_queue.put(None)
                        asyncio.create_task(self._cleanup_request(req))
                        return "COMPLETED"

                    if prefill_status != "PREFILL_COMPLETED":
                        PACE_ERROR(f"Prefill failed, status: {prefill_status}")
                        await self._clear_concluded_requests(
                            req.request_id, session, assigned_engine_url
                        )
                        return "ERROR"

                    PACE_INFO(
                        f"[WORKER-{worker_id}] PREFILL successful for request {engine_req_id}"
                    )

                    # Add first token if present (MLPerf: token_id; generation: text).
                    if req.token_queue:
                        first_token = (
                            token_id
                            if (req.mlperf_mode and token_id is not None)
                            else (token if token is not None else None)
                        )
                        if first_token is not None:
                            req.req_stats["generated_tokens_count"] += 1
                            await req.token_queue.put(first_token)

                    return "OK"

        except Exception as e:
            PACE_INFO(f"Error during prefill for request {req.request_id}: {str(e)}")
            return "ERROR"

    async def _finalize_decode_request(
        self,
        req: Request,
        req_id: str,
        status: str,
        final_token: Optional[str],
        worker_id: int,
        session: aiohttp.ClientSession,
    ):
        """Helper method to finalize a request in the decode queue."""
        del self.decode_queues[worker_id][req_id]
        req.status = (
            RequestStatus.COMPLETED if status == "COMPLETED" else RequestStatus.ERROR
        )
        req.req_stats["finished_at"] = time.time()
        await req.token_queue.put(final_token)
        await self._clear_concluded_requests(req_id, session, req.assigned_engine_url)
        asyncio.create_task(self._cleanup_request(req))

    async def _decode_step(self, session, worker_id: int, assigned_engine_url: str):
        """Execute a single decode step for a specific worker's decode queue."""
        num_requests = len(self.decode_queues[worker_id])
        PACE_INFO(
            f"[WORKER-{worker_id}] Executing DECODE step for {num_requests} request(s)"
        )

        payload = {"is_decode": True}
        async with session.post(
            f"{assigned_engine_url}/step", json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()

            if result.get("status") != "success":
                PACE_DEBUG(f"[WORKER-{worker_id}] Server error during DECODE: {result}")
                for req_id in list(self.decode_queues[worker_id]):
                    req = self.decode_queues[worker_id][req_id]
                    await self._finalize_decode_request(
                        req, req_id, "ERROR", None, worker_id, session
                    )
                return False

            result_data = result.get("result", {})
            PACE_DEBUG(f"[WORKER-{worker_id}] DECODE result data: {result_data}")

            for req_id, decode_result in result_data.items():
                PACE_DEBUG(
                    f"[WORKER-{worker_id}] DECODE step for request {req_id}: {decode_result}"
                )
                if not isinstance(decode_result, dict):
                    continue
                PACE_INFO(
                    f"[WORKER-{worker_id}] DECODE step for request {req_id}: {decode_result}"
                )
                token = decode_result.get("output", "")
                token_id = decode_result.get("token_id")
                status = decode_result.get("status", "DECODING_IN_PROGRESS")

                if req_id in self.decode_queues[worker_id]:
                    req = self.decode_queues[worker_id][req_id]
                    to_put = (
                        token_id
                        if (req.mlperf_mode and token_id is not None)
                        else (token if token is not None else None)
                    )
                    if to_put is not None:
                        num_generated = decode_result.get("num_tokens_generated", 1)
                        req.req_stats["generated_tokens_count"] += num_generated
                        await req.token_queue.put(to_put)

                    # Finalize request when completed
                    if status in ("COMPLETED", "ERROR"):
                        final_token = None if status == "COMPLETED" else "ERROR"
                        await self._finalize_decode_request(
                            req, req_id, status, final_token, worker_id, session
                        )
        return True

    async def decode_loop(self, worker_id: int):
        """
        Interleaves decode steps across requests for a specific worker.
        Stops after completing the current iteration once a stop is requested.
        Each worker has its own isolated decode loop.
        """
        assigned_engine_url = (
            self.engine_urls[worker_id]
            if worker_id < len(self.engine_urls)
            else self.engine_urls[0]
        )
        PACE_INFO(f"Decode loop {worker_id} started (engine: {assigned_engine_url})")

        try:
            timeout = aiohttp.ClientTimeout(
                total=http_config.total,
                connect=http_config.connect,
                sock_connect=http_config.sock_connect,
                sock_read=http_config.sock_read,
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                            await self.clear_cancelled_requests(
                                req_id, session, worker_id
                            )
                            if req_id in self.cancelled_requests:
                                self.cancelled_requests.remove(req_id)
                    PACE_DEBUG(
                        f"Worker {worker_id} decode loop active with {len(self.decode_queues[worker_id])} requests"
                    )
                    ok = await self._decode_step(
                        session, worker_id, assigned_engine_url
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

    async def clear_cancelled_requests(
        self, req_id: str, session: aiohttp.ClientSession, worker_id: int
    ):
        """Remove a cancelled request from decode queue if present, and engine queues."""
        if req_id in self.decode_queues[worker_id]:
            engine_url = self.decode_queues[worker_id][req_id].assigned_engine_url
            del self.decode_queues[worker_id][req_id]
            PACE_INFO(
                f"Worker {worker_id} cleared cancelled request {req_id} from decode queue"
            )
            await self._clear_concluded_requests(req_id, session, engine_url)
