# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Request-lifecycle metrics and server-wide statistics for the scheduler.

Owns TTFT/TPOT tracking, interval merging, and the periodic metrics loop.
The Scheduler base class delegates all stats bookkeeping here.
"""

import asyncio
from typing import Dict, List, Tuple

import aiohttp

from pace.server.router.utils import (
    Request,
    RequestStatus,
    ttft_histogram,
    tpot_histogram,
    e2e_latency_histogram,
    queue_time_histogram,
    prompt_tokens_counter,
    generation_tokens_counter,
    requests_total_counter,
    update_cache_gauges,
)
from pace.utils.logging import PACE_INFO, PACE_WARNING


async def fetch_cache_stats(
    session: aiohttp.ClientSession, engine_url: str
) -> Dict[str, object]:
    """GET an engine's ``/cache_stats`` and return its ``cache_stats`` payload.

    Raises on any non-200 so callers can log/degrade as they see fit. Shared by
    the periodic metrics loop and the ``/v1/server_metrics`` handler.
    """
    url = f"{engine_url.rstrip('/')}/cache_stats"
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"cache_stats returned HTTP {resp.status}")
        payload = await resp.json()
    return payload.get("cache_stats", {})


def aggregate_cache_stats(per_engine: List[Dict[str, object]]) -> Dict[str, object]:
    """Merge per-engine ``cache_stats`` dicts into one server-wide view.

    Pools and prefix caches are per-engine; counts are summed and derived
    ratios (``utilization_pct``, ``prefix_cache_hit_rate``) are recomputed from
    the summed totals rather than averaged.
    """
    pools = [s["pool"] for s in per_engine if s.get("pool")]
    prefixes = [s["prefix_cache"] for s in per_engine if s.get("prefix_cache")]
    agg: Dict[str, object] = {}

    if pools:
        total = sum(p.get("total_blocks", 0) for p in pools)
        allocated = sum(p.get("allocated_blocks", 0) for p in pools)
        agg["pool"] = {
            "total_blocks": total,
            "free_blocks": sum(p.get("free_blocks", 0) for p in pools),
            "allocated_blocks": allocated,
            "utilization_pct": (allocated / total * 100) if total > 0 else 0,
        }

    if prefixes:
        queries = sum(p.get("prefix_cache_queries", 0) for p in prefixes)
        hits = sum(p.get("prefix_cache_hits", 0) for p in prefixes)
        agg["prefix_cache"] = {
            "prefix_cache_enabled": any(
                p.get("prefix_cache_enabled") for p in prefixes
            ),
            "prefix_cache_hash_algo": next(
                (
                    p["prefix_cache_hash_algo"]
                    for p in prefixes
                    if "prefix_cache_hash_algo" in p
                ),
                None,
            ),
            "prefix_cache_queries": queries,
            "prefix_cache_hits": hits,
            "prefix_cache_hit_rate": round(hits / queries, 4) if queries > 0 else 0.0,
            "prefix_cache_cached_blocks": sum(
                p.get("prefix_cache_cached_blocks", 0) for p in prefixes
            ),
            "prefix_cache_evictable_blocks": sum(
                p.get("prefix_cache_evictable_blocks", 0) for p in prefixes
            ),
            "prefix_cache_total_evictions": sum(
                p.get("prefix_cache_total_evictions", 0) for p in prefixes
            ),
            "prefix_cache_total_inserts": sum(
                p.get("prefix_cache_total_inserts", 0) for p in prefixes
            ),
            "prefix_cache_dedup_frees": sum(
                p.get("prefix_cache_dedup_frees", 0) for p in prefixes
            ),
        }

    return agg


async def fetch_aggregated_cache_stats(
    session: aiohttp.ClientSession, engine_urls: List[str]
) -> Dict[str, object]:
    """Fetch ``/cache_stats`` from every engine and return the merged view.

    Engines that fail to respond are logged and skipped so one bad instance
    can't blank out the whole server's metrics.
    """
    results = await asyncio.gather(
        *(fetch_cache_stats(session, url) for url in engine_urls),
        return_exceptions=True,
    )
    stats: List[Dict[str, object]] = []
    for url, res in zip(engine_urls, results):
        if isinstance(res, dict):
            stats.append(res)
        else:
            PACE_WARNING(f"cache_stats fetch failed for {url}: {res!r}")
    return aggregate_cache_stats(stats)


class SchedulerMetrics:
    """Tracks per-request and session-wide inference metrics."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
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

    async def cleanup_request(
        self,
        req: Request,
        active_requests: Dict[str, Request],
    ):
        """Finalize a completed request: remove from active set, compute stats."""
        request_id = req.request_id
        if getattr(req, "_metrics_cleaned_up", False) is True:
            return
        req._metrics_cleaned_up = True

        active_requests.pop(request_id)
        finished_at = req.req_stats.get("finished_at")
        prefill_finished_at = req.req_stats.get("prefill_finished_at")
        created_at = req.req_stats.get("created_at")
        gen_tokens = max(0, req.req_stats.get("generated_tokens_count", 0))

        PACE_INFO(f"Request {request_id} finished at {finished_at:.6f}")

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

        if finished_at and created_at:
            e2e_latency_histogram.observe(finished_at - created_at)
        end_wait = req.req_stats.get("end_wait_time")
        if end_wait and created_at:
            queue_time_histogram.observe(end_wait - created_at)

        prompt_tokens_counter.inc(req.req_stats.get("input_length", 0))
        generation_tokens_counter.inc(gen_tokens)
        status_label = "completed" if req.status == RequestStatus.COMPLETED else "error"
        requests_total_counter.labels(status=status_label).inc()

        PACE_INFO(f"scheduler metrics enabled: {self.enabled}")
        if self.enabled:
            PACE_INFO("Updating scheduler metrics...")
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
        if end_wait and created_at:
            PACE_INFO(f"Request spent {end_wait - created_at} seconds Waiting.")
        PACE_INFO(f"request generated {gen_tokens} tokens")
        PACE_INFO(f"Cleaned up request {request_id}")

    async def cache_metrics_loop(self, is_running_fn, engine_urls: List[str]):
        """Periodically refresh prefix-cache Prometheus gauges, aggregated across
        all engine instances.

        Prometheus scrapes ``/metrics`` on its own schedule, so gauges must be
        refreshed out-of-band rather than on request. Failures degrade silently.
        """
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            while is_running_fn():
                try:
                    await asyncio.sleep(15)
                    update_cache_gauges(
                        await fetch_aggregated_cache_stats(session, engine_urls)
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    PACE_WARNING(f"Error in cache metrics loop: {str(e)}")

    async def server_metrics_loop(self, is_running_fn, get_queue_size_fn):
        """Periodically compute server session metrics based on completed requests."""
        while is_running_fn():
            try:
                await asyncio.sleep(30)
                if get_queue_size_fn() == 0:
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

        return sum(end - start for start, end in merged)

    def server_metrics_snapshot(self) -> Dict[str, float]:
        """Return current server session metrics."""
        return {
            "sched_session_ttft": self.session_ttft,
            "sched_session_tpot": self.session_tpot,
            "sched_active_ttft_time": self.active_ttft_time,
            "sched_requests_served_per_second": self.session_requests_served_per_second,
            "sched_total_generated_tokens": self.total_generated_tokens,
        }
