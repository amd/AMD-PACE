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

### TTFT (Time To First Token)

Measures the time from request submission to the first generated token.

| Metric | Description |
|--------|-------------|
| `pace_ttft_seconds_count` | Total number of completed requests |
| `pace_ttft_seconds_sum` | Sum of all TTFT values (seconds) |
| `pace_ttft_seconds_bucket` | Histogram buckets for TTFT distribution |

**Histogram buckets:** 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0 seconds.

### TPOT (Time Per Output Token)

Measures the average time to generate each output token across the full request.

| Metric | Description |
|--------|-------------|
| `pace_tpot_seconds_count` | Total number of TPOT observations |
| `pace_tpot_seconds_sum` | Sum of all TPOT values (seconds) |
| `pace_tpot_seconds_bucket` | Histogram buckets for TPOT distribution |

**Histogram buckets:** 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0 seconds.

## Available Scheduler Session Metrics (JSON)

Accessible at `GET /v1/server_metrics` when `--scheduler_metrics_enabled True` is set.

| Metric Key | Type | Description |
|------------|------|-------------|
| `sched_session_ttft` | `float` | Average TTFT across all completed requests in the session |
| `sched_session_tpot` | `float` | Average TPOT across all generated tokens in the session |
| `sched_active_ttft_time` | `float` | Cumulative wall-clock time spent in prefill (merged overlapping intervals) |
| `sched_requests_served_per_second` | `float` | Throughput: completed requests per second of active time |
| `sched_total_generated_tokens` | `int` | Total tokens generated across all completed requests |

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
