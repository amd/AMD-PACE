# Running the tests

PACE Agent tests are plain `unittest`, run from the package root (`packages/pace_agent`). Tests
gated on an optional accelerator library or a real browser **skip cleanly** when those
are absent, so the suite is green on a minimal install and does more as the environment
gets richer.

## Environment

Install with the `[fast]` extra so the optional optimized tiers (`orjson` /
`selectolax` / `google-re2`) are exercised rather than skipped:

```bash
cd packages/pace_agent
pip install -e .[fast]
pace-agent check              # orjson / selectolax / re2 (and browser) preflight
```

Without `[fast]`, the library-backed tiers honestly fall back to `naive` and their tests
still pass — they assert the fallback path instead of the accelerated one.

## Full suite

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Vendored agent suites (`tests_agents/`)

GAIA planner and nsjail agent tests use the same `unittest` runner (no separate
`pytest.ini`). Run them explicitly from `packages/pace_agent/`:

```bash
python -m unittest discover -s tests_agents/gaia_planner_tests -p 'test_*.py' -v
python -m unittest discover -s tests_agents/nsjail_agent_tests -p 'test_*.py' -v
```

LangGraph planner tests skip cleanly when `langgraph` is absent; nsjail integration
tests skip when the `nsjail` binary is not on `PATH`.

Expected: `Ran 123 tests ... OK (skipped=13)`. Skips are environment-gated (see the table
below), never failures.

## Optimized-op tests

`tests/test_optimized_ops.py` covers parity, availability gating, and honest-fallback
behavior for every op with an `optimized` tier — pure (`vector_search`, `bm25`,
`quantize`, `dedup`, `embed`, `rerank`, `ast_parse`) and library-backed
(`json_serialize` / `json_deserialize` / `html_parse` / `regex_extract`). The
`optimized_available()` honesty hook itself is covered in `tests/test_registry.py`.

```bash
python -m unittest tests.test_optimized_ops
```

Expected: `Ran 12 tests ... OK (skipped=1)` (the skip is the timing assertion — see below).

For a per-op parity + timing view across all registered ops, use the bench CLI (see
[benchmark-methodology.md](../docs/benchmark-methodology.md)):

```bash
pace-agent tools --list       # which ops expose a real optimized candidate vs "=naive"
pace-agent bench tools        # per-op naive-vs-optimized timing + parity
```

## End-to-end optimize loop

Both record a tape with an offline `StubLLM`, then run the real `record -> optimize` loop
over it — no browser, no live LLM — asserting the loop is valid, parity holds, and every
op whose optimized tier is genuinely available is covered:

- `tests/test_e2e_optimize.py` — the deterministic `agent_ops` fixture (synthetic op checklist).
- `tests/test_e2e_agent_optimize.py` — the **real example agents** that drive ops as part of
  their work (`large_corpus_rag` → `embed`/`dedup`, `customer_support` → `dedup`,
  `tool_agent` → `json_deserialize`/`regex_extract`). Needs the `langgraph` runtime and
  **skips cleanly** without it.

```bash
python -m unittest tests.test_e2e_optimize tests.test_e2e_agent_optimize
```

## Timing assertions (opt-in)

Speedup thresholds are machine-sensitive, so any "optimized is faster" test is gated
behind `PACE_AGENT_PERF_ASSERT` and skipped by default:

```bash
PACE_AGENT_PERF_ASSERT=1 python -m unittest \
    tests.test_e2e_optimize tests.test_optimized_ops
```

## Browser-gated parity (real Chrome)

`tests/test_webvoyager_agent.py` groups the browser-gated WebVoyager tests; all skip
cleanly without Chrome. Two classes exercise the tool ops on **real** browser output:
`TestWebvoyagerActOpsAB` applies the ops externally to a captured accessibility tree +
HTML, and `TestWebvoyagerOpsE2E` runs the real agent and asserts it drove every wired op
with `optimized == naive` on every recorded input.

```bash
python -m unittest tests.test_webvoyager_agent
```

Expected without a browser: `Ran 9 tests ... OK (skipped=7)`. To run the browser tests,
set up Chrome-for-Testing + `chromedriver` per [browser-setup.md](../docs/browser-setup.md).

### Live-LLM end-to-end (opt-in, real API calls)

`TestWebvoyagerLiveLLME2E` (same file) is the only test that makes **real LLM API calls**:
a live `LLMClient` drives the real WebVoyager agent over a real browser to complete a task,
while the non-LLM tool ops run on genuine browser output. It asserts real LLM calls
happened, every wired op ran, and the agent answered with the real page heading. It is
**triple-gated** (`PACE_AGENT_LIVE_LLM=1` + a reachable endpoint + a working browser) and
skipped by default, so CI never depends on a live model.

The test is **provider-agnostic**: it targets whatever endpoint you point it at, configured
entirely through environment variables. No vendor is baked in — a local server, any
OpenAI-compatible host, or any Anthropic Messages endpoint all work the same way.

| Env var | Purpose | Default |
|---|---|---|
| `PACE_AGENT_LIVE_LLM` | opt in to the live test | unset → skipped |
| `PACE_AGENT_LIVE_LLM_ENDPOINT` | base URL of the LLM server | `http://localhost:8000` |
| `PACE_AGENT_LIVE_LLM_MODEL` | model id to drive | auto-discovered from `/v1/models` or `/models` |
| `PACE_AGENT_LIVE_LLM_API_KEY` | your API key | `none` (fine for a keyless local server) |
| `PACE_AGENT_LIVE_LLM_FORMAT` | wire protocol: `openai` or `anthropic` | `openai` |
| `PACE_AGENT_LIVE_LLM_SUB_HEADER` | custom auth-header name, for gateways that don't use standard bearer auth | unset |

`openai` speaks the OpenAI Chat/Responses API (a local vLLM, or any OpenAI-compatible
server); `anthropic` speaks the native Anthropic Messages API. `LLMClient` also auto-selects
the Anthropic transport when the model id is `claude*` — see the offline backend tests in
`tests/test_llm_session.py`.

**Bring your own key.** The key is read only from the environment — it is never hard-coded
in the tests or committed to the repo. Each person exports their own (and should rotate it
if it is ever exposed). Pick whichever recipe matches your setup:

```bash
# 1) Local OpenAI-compatible server (e.g. vLLM), no key — the default target.
PACE_AGENT_LIVE_LLM=1 python -m unittest \
    tests.test_webvoyager_agent.TestWebvoyagerLiveLLME2E

# 2) Any hosted OpenAI-compatible endpoint, authenticated with a bearer key.
export LLM_API_KEY="<your-key>"                     # never commit it
PACE_AGENT_LIVE_LLM=1 \
PACE_AGENT_LIVE_LLM_ENDPOINT=https://your-host/v1 \
PACE_AGENT_LIVE_LLM_MODEL=<model-id> \
PACE_AGENT_LIVE_LLM_API_KEY="$LLM_API_KEY" \
python -m unittest tests.test_webvoyager_agent.TestWebvoyagerLiveLLME2E

# 3) A Claude model over the Anthropic Messages API. If the endpoint is a gateway that
#    authenticates with a custom header, name it via PACE_AGENT_LIVE_LLM_SUB_HEADER.
export LLM_API_KEY="<your-key>"
PACE_AGENT_LIVE_LLM=1 \
PACE_AGENT_LIVE_LLM_ENDPOINT=<anthropic-messages-endpoint> \
PACE_AGENT_LIVE_LLM_MODEL=<claude-model-id> \
PACE_AGENT_LIVE_LLM_FORMAT=anthropic \
PACE_AGENT_LIVE_LLM_SUB_HEADER=<Custom-Auth-Header> \
PACE_AGENT_LIVE_LLM_API_KEY="$LLM_API_KEY" \
python -m unittest tests.test_webvoyager_agent.TestWebvoyagerLiveLLME2E
```

The `anthropic` SDK is a runtime dependency (installed with the package); it is imported
lazily, so it is only needed when a Claude route is actually used.

## What is skipped, and how to enable it

| Skipped tests | Why | Enable by |
|---|---|---|
| Timing / "optimized is faster" | machine-sensitive | `PACE_AGENT_PERF_ASSERT=1` |
| Browser + WebVoyager agent | need real Chrome + `chromedriver` | [browser-setup.md](../docs/browser-setup.md) |
| Live-LLM WebVoyager E2E | opt-in; needs real API calls + a reachable model + a browser | `PACE_AGENT_LIVE_LLM=1` |
| Real-agent E2E optimize | needs `langgraph` | `pip install -e .` (runtime dep) |
| Telemetry | need CPU perf counters / uProf | lower `perf_event_paranoid`; install uProf |
| Library-backed optimized tiers | optional accelerator absent | `pip install -e .[fast]` |

A skip is always an honest "not exercised here", never a silent pass — the same principle
`optimized_available()` applies to the tiers themselves.
