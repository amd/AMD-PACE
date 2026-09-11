# PACE Server Monitoring with Prometheus

Comprehensive guide for monitoring PACE server performance metrics using Prometheus and the built-in JSON metrics endpoints.

## Overview

PACE provides two complementary monitoring approaches:

1. **Prometheus Metrics** — Standard histogram-based metrics exposed at `/metrics` for time-series monitoring, alerting, and dashboarding.
2. **Scheduler Session Metrics** — Aggregate JSON metrics at `/v1/server_metrics` for quick session-level insight.

## Prerequisites

- PACE server with `prometheus_client` (already in `requirements.txt`)
- Router exposes metrics at `http://<router_host>:<router_port>/metrics` (default: `http://localhost:8080/metrics`)

## Setup

### Prometheus (Automated Sidecar)

1. **Start the launcher with Prometheus** by adding `--enable_prometheus`:
   ```bash
   pace-server --enable_prometheus [other launcher args...]
   ```
   If the Prometheus binary is not in the cache yet, the launcher downloads it automatically (once). It is cached under `~/.cache/pace/prometheus/` (override with `PACE_PROMETHEUS_CACHE`).

2. **Open the Prometheus UI** at http://localhost:9090 (or http://YOUR_IP:9090 remotely).

Prometheus starts after the router is ready and stops when you stop the launcher (e.g. Ctrl+C). Config is built into the launcher (see `pace.server.monitoring.prometheus_runner`).

### Scheduler Session Metrics (JSON)

Enable session-level metrics computation with:
```bash
pace-server --scheduler_metrics_enabled True [other launcher args...]
```

The scheduler computes aggregate metrics every 30 seconds when the request queue is empty.

## Full Monitoring Example

```bash
pace-server \
  --server_model meta-llama/Llama-3.1-8B-Instruct \
  --router_port 8080 \
  --scheduler_metrics_enabled True \
  --enable_prometheus \
  --num_engine_instances 2
```

Wait until the log shows "Router is ready." and "Prometheus UI: http://localhost:9090".

**Check the Prometheus metrics endpoint** (router port 8080 by default):
```bash
curl -s http://localhost:8080/metrics | head -50
```

**Check the JSON session metrics:**
```bash
curl -s http://localhost:8080/v1/server_metrics | jq
```

**Verify Prometheus** — open http://localhost:9090 → **Status → Targets** (pace-server should be UP) → **Graph** and run a query such as `pace_ttft_seconds_count`.

## Available Prometheus Metrics

### Gauges (Live Scheduler State)

| Metric | Description |
|--------|-------------|
| `pace_requests_running` | Number of requests currently being processed |
| `pace_requests_waiting` | Number of requests waiting in queue |
| `pace_prefix_cache_hits` | Cumulative prefix-cache hits, measured in cached tokens |
| `pace_prefix_cache_queries` | Cumulative prefix-cache queries, measured in queried tokens |
| `pace_prefix_cache_hit_rate` | Cached/query token ratio from 0 to 1 |
| `pace_kv_cache_usage_perc` | KV-cache pool utilization from 0 to 1 |

### Counters (Cumulative Totals)

| Metric | Description |
|--------|-------------|
| `pace_prompt_tokens_total` | Total prompt tokens processed |
| `pace_generation_tokens_total` | Total generation tokens produced |
| `pace_requests_total{status="completed"}` | Total successfully completed requests |
| `pace_requests_total{status="error"}` | Total errored requests |

### TTFT (Time To First Token)

Measures from request creation through prefill completion, including queue
wait and first-token work.

| Metric | Description |
|--------|-------------|
| `pace_ttft_seconds_count` | Number of recorded TTFT observations |
| `pace_ttft_seconds_sum` | Sum of all TTFT values (seconds) |
| `pace_ttft_seconds_bucket` | Histogram buckets for TTFT distribution |

**Histogram buckets:** 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0 seconds.

### TPOT (Time Per Output Token)

Each observation is total request lifetime divided by generated-token count.

| Metric | Description |
|--------|-------------|
| `pace_tpot_seconds_count` | Total number of TPOT observations |
| `pace_tpot_seconds_sum` | Sum of all TPOT values (seconds) |
| `pace_tpot_seconds_bucket` | Histogram buckets for TPOT distribution |

**Histogram buckets:** 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0 seconds.

### End-to-End Request Latency

| Metric | Description |
|--------|-------------|
| `pace_e2e_request_latency_seconds_count` | Total observations |
| `pace_e2e_request_latency_seconds_sum` | Sum of all e2e latencies |
| `pace_e2e_request_latency_seconds_bucket` | Histogram buckets |

### Queue Wait Time

| Metric | Description |
|--------|-------------|
| `pace_request_queue_time_seconds_count` | Total observations |
| `pace_request_queue_time_seconds_sum` | Sum of all queue wait times |
| `pace_request_queue_time_seconds_bucket` | Histogram buckets |

**Histogram buckets (e2e, queue):** 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 240.0, 480.0 seconds.

Note: TTFT is `prefill_finished_at - created_at`, so it is not an isolated
prefill-kernel measurement.

## Available Scheduler Session Metrics (JSON)

Accessible at `GET /v1/server_metrics` when `--scheduler_metrics_enabled True` is set.
KV-cache pool and prefix-cache fields are included whenever the engine exposes
them, even if scheduler session metrics are disabled.

| Metric Key | Type | Description |
|------------|------|-------------|
| `sched_session_ttft` | `float` | Average TTFT across all completed requests in the session |
| `sched_session_tpot` | `float` | Average TPOT across all generated tokens in the session |
| `sched_active_ttft_time` | `float` | Cumulative wall-clock time spent in prefill (merged overlapping intervals) |
| `sched_requests_served_per_second` | `float` | Throughput: completed requests per second of active time |
| `sched_total_generated_tokens` | `int` | Total tokens generated across all completed requests |
| `kv_cache_pool_total_blocks` | `int` | Total blocks across engine KV-cache pools |
| `kv_cache_pool_free_blocks` | `int` | Free blocks across engine KV-cache pools |
| `kv_cache_pool_allocated_blocks` | `int` | Allocated blocks across engine KV-cache pools |
| `kv_cache_pool_utilization_pct` | `float` | Aggregate pool utilization percentage |
| `prefix_cache_queries` | `int` | Total queried prefix tokens |
| `prefix_cache_hits` | `int` | Total reused prefix tokens |
| `prefix_cache_hit_rate` | `float` | Reused/query token ratio |
| `prefix_cache_enabled` | `bool` | Whether any engine reports prefix caching enabled |
| `prefix_cache_cached_blocks` | `int` | Cached prefix blocks across engines |
| `prefix_cache_evictable_blocks` | `int` | Cached blocks currently eligible for eviction |
| `prefix_cache_total_evictions` | `int` | Cumulative cache evictions |
| `prefix_cache_total_inserts` | `int` | Cumulative cache inserts |
| `prefix_cache_dedup_frees` | `int` | Blocks freed by deduplicating concurrent inserts |

**Example response:**
```json
{
  "sched_session_ttft": 0.245,
  "sched_session_tpot": 0.032,
  "sched_active_ttft_time": 1.47,
  "sched_requests_served_per_second": 2.1,
  "sched_total_generated_tokens": 1580
}
```

## Health and Queue Endpoints

These endpoints are always available (no flags required):

| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Service health, scheduler status, queue size, active requests |
| `GET /v1/queue/status` | Current queue size and active request count |
| `GET /v1/status/{request_id}` | Status of a specific request |

## Useful PromQL Queries

```promql
# Total completed requests
pace_ttft_seconds_count

# Average TTFT over all time
pace_ttft_seconds_sum / pace_ttft_seconds_count

# Average TPOT over all time
pace_tpot_seconds_sum / pace_tpot_seconds_count

# 50th percentile (median) TTFT over last 5 minutes
histogram_quantile(0.50, rate(pace_ttft_seconds_bucket[5m]))

# 95th percentile TTFT over last 5 minutes
histogram_quantile(0.95, rate(pace_ttft_seconds_bucket[5m]))

# 99th percentile TTFT over last 5 minutes
histogram_quantile(0.99, rate(pace_ttft_seconds_bucket[5m]))

# 95th percentile TPOT over last 5 minutes
histogram_quantile(0.95, rate(pace_tpot_seconds_bucket[5m]))

# Requests per second (throughput)
rate(pace_ttft_seconds_count[1m])

# Average TTFT over last 5 minutes (moving average)
rate(pace_ttft_seconds_sum[5m]) / rate(pace_ttft_seconds_count[5m])

# Average TPOT over last 5 minutes
rate(pace_tpot_seconds_sum[5m]) / rate(pace_tpot_seconds_count[5m])

# Token generation rate (tokens per second)
rate(pace_tpot_seconds_count[1m])

# Prompt token throughput (tokens/s)
rate(pace_prompt_tokens_total[1m])

# Generation token throughput (tokens/s)
rate(pace_generation_tokens_total[1m])

# Request success rate
rate(pace_requests_total{status="completed"}[5m])

# Request error rate
rate(pace_requests_total{status="error"}[5m])

# 95th percentile e2e latency
histogram_quantile(0.95, rate(pace_e2e_request_latency_seconds_bucket[5m]))

# 95th percentile queue wait time
histogram_quantile(0.95, rate(pace_request_queue_time_seconds_bucket[5m]))

# Current scheduler utilization
pace_requests_running
pace_requests_waiting
```

## Prometheus Configuration Details

When `--enable_prometheus` is used, the launcher:

1. Downloads Prometheus 3.9.1 (linux-amd64) to `~/.cache/pace/prometheus/` on first run
2. Writes a temporary `prometheus.yml` with scrape configuration:
   - **Self-monitoring** on port 9090
   - **PACE metrics** scraping the router at `<router_host>:<router_port>/metrics` every 5 seconds
3. Starts Prometheus on port **9090**

**Override cache location:**
```bash
export PACE_PROMETHEUS_CACHE=/path/to/custom/cache
pace-server --enable_prometheus ...
```

## Notes

- Prometheus scrapes the router every **5 seconds**.
- Metrics are exposed by the router at `http://<router_host>:<router_port>/metrics` (default port 8080).
- Histogram bucket ranges are configured in `pace/server/router/utils.py`.
- Scheduler session metrics are computed every 30 seconds when the queue is empty, so there may be a delay in updates during continuous high load.
