# AMD PACE Inference Server

## Architecture Overview

The AMD PACE Inference Server uses a **router + engine** architecture:

```
Client Requests
      │
      ▼
┌─────────────┐      ┌──────────────┐
│   Router     │─────►│  Engine 0    │  (NUMA-bound, port 8000)
│  (port 8080) │─────►│  Engine 1    │  (NUMA-bound, port 8001)
│              │─────►│  Engine N    │  (NUMA-bound, port 800N)
└─────────────┘      └──────────────┘
      │
      ▼
 /metrics (Prometheus)
```

- **Router**: API surface for completions. Accepts `/v1/completions` requests, schedules them across engine instances, and streams responses back.
- **Engine(s)**: Each engine instance loads the model, manages KV cache, and executes prefill/decode steps. Engines are pinned to specific NUMA nodes/cores for optimal performance.
- **Launcher**: Orchestrates startup of all engine instances and the router via a single `pace-server` command.
- **Continuous batching**: The `continuous_batch` scheduler combines active decodes with queued and chunked prefills in shared engine steps, subject to the configured token budget.

---

## Using the AMD PACE Inference Server

All components (server and router) are started easily via a single launcher script.

### 1. **Installation & Setup**

Make sure your Python environment is correctly set up, dependencies are installed, and you are in the appropriate directory. **`numactl` must be installed** on the machine (`sudo apt install numactl` or `sudo dnf install numactl`) — the launcher uses it for CPU and memory binding.

### 2. **Launching the Inference Server**

Run:

```bash
pace-server --help
```

You will see the available options (detailed in the table below).

**Usage:**
```bash
pace-server [-h] [--server_host SERVER_HOST] [--server_port SERVER_PORT]
            [--server_model SERVER_MODEL] [--dtype DTYPE]
            [--kv_cache_type KV_CACHE_TYPE] [--serve_type SERVE_TYPE]
            [--op_config OP_CONFIG] [--router_host ROUTER_HOST]
            [--router_port ROUTER_PORT]
            [--scheduler_metrics_enabled SCHEDULER_METRICS_ENABLED]
            [--fastapi_log_level FASTAPI_LOG_LEVEL]
            [--spec_config SPEC_CONFIG]
            [--kv_cache_memory_gb KV_CACHE_MEMORY_GB]
            [--token_budget TOKEN_BUDGET]
            [--max_num_running_reqs MAX_NUM_RUNNING_REQS]
            [--max_prefill_chunk MAX_PREFILL_CHUNK]
            [--enable_prometheus]
            [--numa_physcpubind NUMA_PHYSCPUBIND]
            [--numa_membind NUMA_MEMBIND]
            [--num_engine_instances NUM_ENGINE_INSTANCES]
```

To start with defaults:

```bash
pace-server
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-h, --help` | - | - | Show help message and exit |
| **Server Configuration** | | | |
| `--server_host` | `str` | `"0.0.0.0"` | Host address to bind the engine server(s)<br>• `0.0.0.0` - Listen on all interfaces<br>• `127.0.0.1` - Local-only access |
| `--server_port` | `int` | `8000` | Base TCP port for engine instances. Instance *i* listens on `server_port + i` |
| `--fastapi_log_level` | `str` | `"Default"` | Controls uvicorn access logging. Set to `"None"` to disable uvicorn access logs; any other value (including the default) keeps standard logging. Does not currently map to specific log levels |
| **Router Configuration** | | | |
| `--router_host` | `str` | `"0.0.0.0"` | Host address for the request router component |
| `--router_port` | `int` | `8080` | Port for the router |
| `--serve_type` | `str` | `"iterative"` | Scheduling strategy<br>Options: `iterative`, `continuous_prefill_first`, `continuous_batch` |
| `--scheduler_metrics_enabled` | `str` | `"False"` | Enable scheduler-level session metrics<br>Values: `"True"`, `"False"` |
| `--token_budget` | `int` | `2048` | Maximum total prefill and decode tokens in one `continuous_batch` engine step |
| `--max_num_running_reqs` | `int` | `128` | Maximum active sequences per engine for `continuous_batch`; clamped to fit the token budget |
| `--max_prefill_chunk` | `int` | `0` | Maximum prefill chunk size for `continuous_batch`; `0` means no additional cap beyond the remaining token budget |
| **NUMA / Multi-Instance** | | | |
| `--numa_physcpubind` | `str` | `None` | Per-instance physical CPU binding. Semicolon-separated for multi-instance<br>Example (2 instances): `"0-127;128-255"` |
| `--numa_membind` | `str` | `None` | Per-instance NUMA memory binding. Semicolon-separated<br>Example: `"0;1"` |
| `--num_engine_instances` | `int` | `1` | Number of parallel engine instances. Cores from socket 0 are automatically split evenly if `numa_physcpubind` is not specified |
| **Model Configuration** | | | |
| `--server_model` | `str` | `"facebook/opt-6.7b"` | HuggingFace model name or local path<br>Example: `meta-llama/Llama-3.1-8B-Instruct` |
| `--dtype` | `str` | `"bfloat16"` | Data type for model weights/compute<br>Options: `float32`, `bfloat16` |
| `--kv_cache_type` | `str` | `"BMC"` | KV cache implementation<br>Options: `BMC` (Balancing Memory & Compute), `dynamic`, `SLAB_POOL`, `PAGED` |
| `--kv_cache_memory_gb` | `float` | `None` | Optional per-engine `SLAB_POOL` memory budget in GB. Providing it allocates each engine's slab pool eagerly; otherwise capacity is derived lazily. It does not configure `PAGED` capacity |
| `--op_config` | `str` | `"{}"` | Operator backend configuration (JSON string, see [Operator Backends](#operator-backend-configuration)) |
| `--spec_config` | `str` | `"{}"` | Speculative decoding configuration (JSON string)<br>Example: `'{"model_name": "amd/PARD-Qwen2.5-0.5B", "num_speculative_tokens": 12}'`<br>See [Online Speculative Decoding](#online-speculative-decoding-pard) for details |
| **Monitoring** | | | |
| `--enable_prometheus` | flag | off | Start bundled Prometheus sidecar for metrics scraping. See [Monitoring](InferenceServerMonitoring.md) |

### Scheduling Strategies

| Strategy | Description |
|----------|-------------|
| `iterative` | Base scheduler. Each request gets its own prefill+decode loop on the assigned engine. No concurrency. |
| `continuous_prefill_first` | Drains queued prefills before resuming a shared decode loop. Prefill and decode run in separate engine calls; prompts are not chunked. |
| `continuous_batch` | Combines active decodes with queued or in-progress chunked prefills in one mixed engine step. Supported with `SLAB_POOL` and `PAGED`; other cache types are overridden to `SLAB_POOL`. |

### Operator Backend Configuration

The `--op_config` flag accepts a JSON string to configure operator backends. Six operator types can be configured:

| Operator | Key | Backends | Default |
|----------|-----|----------|---------|
| Normalization | `norm_backend` | `NATIVE`, `JIT` | `NATIVE` |
| QKV Projection | `qkv_projection_backend` | `NATIVE`, `JIT`, `TPP`, `AOCLDLP` | `TPP` |
| Attention | `attention_backend` | `JIT`, `NATIVE`, `PAGED`, `SLAB` | Cache-dependent |
| Output Projection | `out_projection_backend` | `NATIVE`, `JIT`, `TPP`, `AOCLDLP` | `TPP` |
| MLP | `mlp_backend` | `NATIVE`, `JIT`, `TPP`, `IMBPS`, `AOCLDLP` | `TPP` |
| LM Head | `lm_head_backend` | `NATIVE`, `JIT`, `TPP`, `AOCLDLP` | `NATIVE` |

**Example:**
```bash
pace-server \
  --server_model meta-llama/Llama-3.1-8B-Instruct \
  --op_config '{"norm_backend": "JIT", "qkv_projection_backend": "TPP", "attention_backend": "SLAB", "out_projection_backend": "TPP", "mlp_backend": "TPP", "lm_head_backend": "TPP"}' \
  --kv_cache_type SLAB_POOL
```

The attention backend is finalized from the KV cache: `DYNAMIC`/`BMC` use
`JIT` or `NATIVE`, `SLAB_POOL` uses `SLAB`, and `PAGED` uses `PAGED`.

### Automatic Prefix Caching

`SLAB_POOL` and `PAGED` reuse complete prompt-prefix blocks across requests by
default. Set `PACE_PREFIX_CACHE=0` before launching the server to disable this,
or set `PACE_PREFIX_HASH=sha256` to use SHA-256 instead of the default
`xxhash` block hash. `PAGED` capacity is controlled by
`PACE_MAX_CACHE_TOKENS` (default `262144`). Cache hit and pool utilization
statistics are exposed through `/v1/server_metrics` and Prometheus. See the
[Performance Guide](PerformanceGuide.md#environment-variables-reference) for
the remaining cache and backend environment variables.

> For backend selection and benchmarking guidance, refer to the [Performance Guide](PerformanceGuide.md).

---

### 3. Service API Endpoints (Router)

### Completions

Send text generation requests. The route implements an OpenAI-shaped subset of
the [v1 completions API](https://platform.openai.com/docs/api-reference/completions)
plus PACE-specific fields. Both streaming and non-streaming modes are supported.

**POST** `/v1/completions`

**Request Body** (`CompletionRequest`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | *required* | Model name (must match `--server_model`) |
| `prompt` | `str`, `List[str]`, `List[int]`, or `List[List[int]]` | *required* | Input text(s) or token ID(s) for generation |
| `stream` | `bool` | `false` | Enable Server-Sent Events streaming |
| `stream_options` | `dict` | `None` | Set `{"include_usage": true}` to emit a final usage-only streaming chunk |
| `max_tokens` | `int` | `16` | Maximum tokens to generate |
| `temperature` | `float` | `None` | Sampling temperature (0 = greedy, >0 = random) |
| `top_p` | `float` | `None` | Nucleus sampling threshold |
| `top_k` | `int` | `None` | Top-k sampling (PACE extension) |
| `min_p` | `float` | `None` | Minimum probability threshold (PACE extension) |
| `min_tokens` | `int` | `None` | Minimum generated tokens (PACE extension) |
| `seed` | `int` | `None` | Random seed for reproducibility |
| `stop` | `str` or `List[str]` | `None` | Stop generation on these strings |
| `echo` | `bool` | `false` | Prepend prompt text to output |
| `logprobs` | `int` | `None` | Return sampled token logprobs and up to this many top alternatives |
| `suffix` | `str` | `None` | Append this text after generated output |
| `n` | `int` | `1` | Number of completions (only 1 supported) |
| `best_of` | `int` | `None` | Accepted only when omitted or set to `1` |
| `frequency_penalty` | `float` | `0.0` | Penalty based on token frequency |
| `presence_penalty` | `float` | `0.0` | Accepted but not applied |
| `repetition_penalty` | `float` | `None` | Multiplicative repetition penalty (PACE extension) |
| `ignore_eos` | `bool` | `None` | Ignore EOS stopping (PACE extension) |
| `do_sample` | `bool` | `None` | Explicitly select random or greedy sampling (PACE extension) |
| `skip_special_tokens` | `bool` | `true` | Skip special tokens during decoding (PACE extension) |
| `logit_bias` | `dict` | `None` | Accepted but not applied |
| `user` | `str` | `None` | Accepted but not used |
| `mlperf_mode` | `bool` | `false` | Return space-delimited token IDs as text; streaming uses `text/plain` |

> **Note**: When `temperature` is `None` or `0`, greedy decoding is used. When `temperature > 0`, random sampling is enabled automatically.
> `echo` and `suffix` currently affect non-streaming text completions only.
> Prompt logprobs require non-streaming, non-chunked `echo=true`; chat
> completions do not return logprobs.
> `mlperf_mode` accepts one prompt per request.

**Non-streaming example:**

```bash
curl -sS -X POST "http://localhost:8080/v1/completions" \
  -H "Content-Type: application/json" \
  --data-binary '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain quantum computing in simple terms.",
    "stream": false,
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq
```

**Non-streaming response format** (OpenAI-compatible `text_completion`):

```json
{
  "id": "cmpl-a1b2c3d4-...",
  "object": "text_completion",
  "created": 1714400000,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": "Quantum computing uses quantum bits...",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 42,
    "total_tokens": 50
  }
}
```

**Streaming example:**

```bash
curl -sS -X POST "http://localhost:8080/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  --data-binary '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write a haiku about CPU inference.",
    "stream": true,
    "max_tokens": 50,
    "temperature": 0
  }' --no-buffer
```

**Streaming response format** (Server-Sent Events):

```
data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":"Silicon"}]}

data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":" threads"}]}

data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":" weave"}]}

data: [DONE]
```

Every response includes an `id` field (e.g. `cmpl-a1b2c3d4-...`) in the JSON body following the OpenAI format.
Streaming responses also include an `X-Request-ID` HTTP header: for single-prompt requests this is the per-prompt request ID; for multi-prompt requests it is the shared group ID that ties all `choices[].index` entries together.

### Chat Completions

Send OpenAI-compatible chat messages. The router applies the configured
tokenizer's chat template and uses the same sampling and scheduling path as
text completions. Both streaming and non-streaming responses are supported.
Only text content is accepted.

**POST** `/v1/chat/completions`

```bash
curl -sS -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain CPU inference briefly."}],
    "max_completion_tokens": 64,
    "temperature": 0
  }' | jq
```

The tokenizer must provide a chat template. Chat requests currently generate
one completion and do not return token logprobs.

### Health Check

Verifies the service is up and returns scheduler health details.

**GET** `/v1/health`

```bash
curl -s http://localhost:8080/v1/health | jq
```

**Response:**
```json
{
  "status": "healthy",
  "service": "frontend",
  "scheduler_running": true,
  "queue_size": 0,
  "active_requests": 0,
  "server_metrics_enabled": false
}
```

### Check Request Status

Returns the status of an **active** (queued or processing) request by its ID. Completed requests are removed from the scheduler's tracking and will return 404.

**GET** `/v1/status/{request_id}`

```bash
curl -s http://localhost:8080/v1/status/a1b2c3d4-... | jq
```

**Response (active request):**
```json
{
  "request_id": "a1b2c3d4-...",
  "status": "processing",
  "message": "Request a1b2c3d4-... is processing",
  "created_at": "2026-03-30T10:15:00"
}
```

> Returns **404** if the request has already completed or the ID is unknown.

### Queue Status

Returns the current queue size and active request count.

**GET** `/v1/queue/status`

```bash
curl -s http://localhost:8080/v1/queue/status | jq
```

**Response:**
```json
{
  "queue_size": 3,
  "active_requests": 2
}
```

### Server Metrics (JSON)

Returns aggregate scheduler session metrics. Requires `--scheduler_metrics_enabled True`.

**GET** `/v1/server_metrics`

```bash
curl -s http://localhost:8080/v1/server_metrics | jq
```

**Response:**
```json
{
  "sched_session_ttft": 0.245,
  "sched_session_tpot": 0.032,
  "sched_active_ttft_time": 1.47,
  "sched_requests_served_per_second": 2.1,
  "sched_total_generated_tokens": 1580
}
```

### List Available Models

Proxies through to the backend engine to return available models.

**GET** `/v1/models`

```bash
curl -s http://localhost:8080/v1/models | jq
```

### Current Configuration

Returns the configured model, KV-cache type, and explicitly configured
attention backend. If no attention backend was supplied in `--op_config`, the
response reports `"N/A"` rather than the cache-derived effective backend.

**GET** `/v1/config`

```bash
curl -s http://localhost:8080/v1/config | jq
```

### Prometheus Metrics

Raw Prometheus metrics endpoint for scraping.

**GET** `/metrics`

```bash
curl -s http://localhost:8080/metrics
```

See [InferenceServerMonitoring.md](InferenceServerMonitoring.md) for Prometheus setup and available metrics.

---

### 4. NUMA and Multi-Instance Configuration

The launcher uses `numactl` to pin each engine instance to specific CPU cores and NUMA memory nodes.

**Default behavior** (no NUMA flags): The launcher auto-detects socket-0 physical cores and splits them evenly across instances.

**Explicit binding:**

The examples below assume a **2-socket machine with 128 physical cores per socket** (256 total). Each socket corresponds to one NUMA node:
- **NUMA node 0** — Socket 0, cores 0–127
- **NUMA node 1** — Socket 1, cores 128–255

Adjust the core ranges and node IDs to match your actual topology (check with `lscpu` or `numactl --hardware`).

```bash
# Single instance pinned to socket 0 (128 cores, NUMA node 0)
pace-server --numa_physcpubind "0-127" --numa_membind "0"

# 2 instances, one per socket (each gets 128 cores and its local memory)
pace-server \
  --num_engine_instances 2 \
  --numa_physcpubind "0-127;128-255" \
  --numa_membind "0;1"

# 4 instances on socket 0 only (128 cores split into 4 × 32)
pace-server \
  --num_engine_instances 4 \
  --numa_physcpubind "0-31;32-63;64-95;96-127" \
  --numa_membind "0;0;0;0"
```

Each engine instance also sets:
- `OMP_NUM_THREADS` = number of cores in its binding
- `OMP_WAIT_POLICY=active` for optimal thread utilization

---

### 5. Launch Examples

**Basic single-instance server:**
```bash
pace-server --server_model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
```

**Continuous batching with chunked prefill:**
```bash
pace-server \
  --server_model meta-llama/Llama-3.1-8B-Instruct \
  --kv_cache_type SLAB_POOL \
  --serve_type continuous_batch \
  --token_budget 2048 \
  --max_num_running_reqs 128
```

**Multi-instance with custom ports:**
```bash
pace-server \
  --server_model meta-llama/Llama-3.1-8B-Instruct \
  --server_port 9000 \
  --router_port 9080 \
  --num_engine_instances 2 \
  --kv_cache_type SLAB_POOL \
  --serve_type continuous_batch
```

**With speculative decoding (PARD):**
```bash
pace-server \
  --server_model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --kv_cache_type SLAB_POOL \
  --spec_config '{"model_name": "amd/PARD-Qwen2.5-0.5B", "num_speculative_tokens": 12}' \
  --serve_type continuous_batch
```

**Complete setup with monitoring:**
```bash
pace-server \
  --server_model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --num_engine_instances 2 \
  --numa_physcpubind "0-127;128-255" \
  --numa_membind "0;1" \
  --kv_cache_type SLAB_POOL \
  --serve_type continuous_batch \
  --scheduler_metrics_enabled True \
  --enable_prometheus \
  --kv_cache_memory_gb 16
```

---

### 6. [Engine Documentation](../pace/server/engine/README.md)

### 7. Monitoring with Prometheus

Monitor PACE server metrics (TTFT, TPOT, request rates) using Prometheus. See [InferenceServerMonitoring.md](InferenceServerMonitoring.md) for setup instructions.

---

### 8. HTTP Timeout Configuration

The router uses configurable HTTP timeouts for communicating with engine instances. Override via environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HTTP_TIMEOUT_TOTAL` | `300` | Total request timeout in seconds (5 min for long inference) |
| `HTTP_TIMEOUT_CONNECT` | `30` | Connection establishment timeout |
| `HTTP_TIMEOUT_SOCK_CONNECT` | `30` | Socket connection timeout |
| `HTTP_TIMEOUT_SOCK_READ` | `30` | Read timeout between data chunks |

---

## Online Speculative Decoding (PARD)

The inference server supports speculative decoding via PARD (PARallel Draft Model Adaptation). A smaller draft model predicts multiple tokens ahead, which are then verified in parallel by the target model, reducing the number of forward passes needed for generation. For details on PARD's offline/Python API usage and its architecture, see the [Speculative Decoding documentation](SpeculativeDecoding.md).

### Server Configuration

Speculative decoding is configured at launch via `--spec_config`:

```bash
pace-server \
  --server_model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --kv_cache_type SLAB_POOL \
  --spec_config '{"model_name": "amd/PARD-Qwen2.5-0.5B", "num_speculative_tokens": 12}' \
  --serve_type continuous_batch
```

**`spec_config` JSON fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | *required* | HuggingFace name or path to the draft model |
| `num_speculative_tokens` | `int` | `12` | Number of tokens to speculate per step |
| `draft_kv_cache_memory_gb` | `float` | `None` | Memory budget for draft model's KV cache |

> The server currently only supports `type: "pard"` (the default). Speculative decoding applies to all requests served by that instance.

### Compatible Models

For compatible target/draft model pairs, see the [PARD repository](https://github.com/AMD-AGI/PARD).

### Sending Requests

The API is the same as regular completions — speculative decoding is transparent to the client:

```bash
curl -sS -X POST "http://localhost:8080/v1/completions" \
  -H "Content-Type: application/json" \
  --data-binary '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Explain how speculative decoding works:",
    "stream": true,
    "max_tokens": 200,
    "temperature": 0
  }' --no-buffer
```

### PARD Output Behavior

With speculative decoding, each decode step can produce **multiple tokens** (all accepted speculative tokens). In streaming mode, all accepted tokens from a single decode step are decoded together and emitted as a single SSE chunk:

```
data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":"Speculative decoding is a technique"}]}
data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":" that uses a smaller"}]}
data: [DONE]
```

> Each chunk may contain multiple tokens' worth of text when the draft model's predictions are accepted.

### Important Notes

1. PARD Speculative Decoding is **only enabled for Greedy Decoding** (`temperature=0`, `do_sample=false`).
2. The `continuous_batch` scheduler accounts for each speculative decode as
   `num_speculative_tokens + 1` tokens when enforcing its token budget.
3. Speculative decoding is configured **server-wide** at launch time — all requests automatically use it.

For the interactive speculative decoding demo, see the [Server Speculative Demo](../demos/pace_server_speculative_demo.py).

---

## Supported Models

For the complete and up-to-date model support matrix (including dtype and quantization support status), refer to the [LLM documentation — Roadmap: Models](LLM.md#roadmap).

## Examples

- [Server Basic](../examples/pace_server_basic.py)
- [Server Playbook](../examples/playbook_server.ipynb)
- [PARD Playbook](../examples/playbook_server_speculative.ipynb)

## Demos

- [Speculative Decoding (Curses Grid)](../demos/pace_server_speculative_demo.py)
- [Interactive Generation](../demos/server_chat_demo.py)
