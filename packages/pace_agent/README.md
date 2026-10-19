# PACE Agent

**PACE Agent** is a CPU performance benchmark and optimizer for the **non-model layer** of
agentic AI — the retrieval, serialization, context-assembly, tool-I/O, and orchestration
work that runs on the CPU around the model.

It answers three questions:

> **Where did measured time go? → What gain does an available candidate project? → Does candidate parity hold?**

## How it works

A **tool** is any non-LLM operation an agent performs — embed, vector search, BM25, rerank,
quantize, tokenize, (de)serialize, parse, chunk, dedup, summarize, and so on. Tools are
grouped into five CPU **buckets**: retrieval, serialization, context, orchestration, and
tool-I/O.

The built-ins are synthetic CPU patterns for profiling and implementation comparison, not
production semantic embedding, tokenization, reranking, or summarization models. Work that
does not use a registered PACE Agent operation or a supported serialization shim remains
unattributed.

Each tool has a readable `naive` reference implementation. Some tools also provide a distinct
`optimized` candidate; tools without one are listed as reference-only. `pace-agent bench tools`
reports candidate timings and parity, but currently does not suppress a speedup or fail the
command when parity is `FAIL`. `pace-agent optimize` does gate its projection on parity.

External packages can register accelerated **provider** tiers through a small public API.
Unavailable provider columns are skipped by `bench tools`; `optimize` treats zero provider
coverage as invalid rather than silently falling back to Python.

## Install

The package is not published on PyPI; install it from this repository. Run every command
from the repository root:

```bash
pip install -e ./packages/pace_agent
```

This pulls the runtime dependencies (numpy, msgpack, openai, langgraph, selenium, GAIA /
nsjail helpers, and attachment readers such as openpyxl and pdfminer.six). For host-side PDF
text extraction (`pdf_text_extract`), also install the optional `pdf` extra:

```bash
pip install -e "./packages/pace_agent[pdf]"
```

Verify:

```bash
pace-agent check   # core imports, tools, and optional capability prerequisites
```

For GAIA / nsjail sandbox / browser setup (ffmpeg, `NSJAIL_BIN`, Chrome, HF token, gateway
env vars), see **[docs/gaia-planner.md](docs/gaia-planner.md)**,
**[docs/nsjail-agent.md](docs/nsjail-agent.md)**, and
**[docs/browser-setup.md](docs/browser-setup.md)**.

## Usage

### Benchmark the tool tiers

```bash
pace-agent check              # check core imports, tools, and optional capability prerequisites
pace-agent bench tools --list # list registered tools and their tiers, grouped by bucket
pace-agent bench tools        # benchmark naive vs optimized candidates
```

`pace-agent bench tools` runs a built-in default set of tool sizes; pass `--config <file.json>`
to benchmark your own sizes. The default methodology is 20 interleaved samples after three
warmups; `--repeat` changes the sample count.

### Profile an agent — the loop

PACE Agent profiles a real agent run and turns it into an actionable report, following the same
three questions:

```bash
pace-agent profile <agent> --config run.json --characterize-runs 10 --out run/  # record + profile a representative run
pace-agent optimize run/ --impl optimized                                       # evaluate a recorded run
pace-agent report run/                                                          # render the latest result
```

`<agent>` is an import spec or Python file path. It may expose `run(llm, task)`, a compiled
[LangGraph](https://github.com/langchain-ai/langgraph) graph, or a `build(llm)`/`graph(llm)`
factory. The package includes `pace_agent.examples.basic_example`.

`profile` records registered PACE Agent operation calls and responses made through the supplied
PACE Agent LLM client in `trajectory.msgpack`, along with `result.json`, `trace.json`, `env.json`,
and the input `config.json`. Command-line `--query` overrides are not written back to that
copy; a run that used them writes `tasks.json` alongside it, holding the tasks that actually ran.

`optimize` validates a full naive replay, independently re-times candidate-covered operations
on their recorded inputs, parity-checks every covered call, and reports an Amdahl-style
projected E2E. It does not measure a full candidate replay. The command writes
`optimize_<timestamp>.json` and refreshes `report.md`. `profile` itself writes no report — run
`pace-agent report <run_dir>` for it, which also lets a run be re-rendered later without re-running it.
A run with no candidate coverage, replay-contract divergence, or failed parity is invalid and
receives no speedup claim.

### Step 1: profile with characterization — record a representative run, not the first one

An agent asked the same question twice does not do the same thing twice, so recording the
first live run makes one arbitrary sample reproducible without making it typical. With
`--characterize-runs N`, `profile` runs the agent N times live, records a representative run
instead, and reports how representative it is. It is off by default (`characterize_runs: 1`).

```bash
pace-agent profile pace_agent.examples.webvoyager --config run.json \
  --characterize-runs 10 --characterize-strategy mode --out run/
```

Strategies, verdicts, config keys and cost are in step 1 of
[docs/deterministic-replay.md](docs/deterministic-replay.md#step-1-profile-with-characterization).

### Example agents

PACE Agent ships several runnable [LangGraph](https://github.com/langchain-ai/langgraph) agents you
can profile with `pace-agent profile pace_agent.examples.<name> --config <config.json>`:

| agent | what it exercises |
|---|---|
| `basic_example` | a minimal reference agent — the smallest end-to-end loop |
| `customer_support` | a multi-step support agent (retrieval + orchestration) |
| `tool_agent` | a tool-calling agent that looks up structured records |
| `large_corpus_rag` | retrieval-augmented generation over a large corpus (vector-search heavy) |
| `testgen` | a cyclic unit-test author that executes generated Python; use trusted inputs in an isolated environment |
| `run_gaia` | GAIA planner integration (`python -m pace_agent.examples.run_gaia`); see [docs/gaia-planner.md](docs/gaia-planner.md) |
| `gaia_agent_example` | nsjail `execute_code` on the local GAIA sample fixtures |
| `nsjail_langgraph_example` | LangGraph ReAct agent using nsjail `execute_code` (needs langchain-openai) |

Ready-to-edit configs for several of these live in `packages/pace_agent/configs/`
(`basic_example.json`, `pace_agent_rag_gptoss.json`, `pace_agent_testgen.json`) — copy one, point its
`llm.endpoint` at your endpoint, and pass it with `--config`. `pace_agent_characterize.json` is the
same idea with characterization enabled; it names no workload, so it works for any of these agents.

## Configuration

`profile` requires a JSON config:

```json
{
  "workload": "basic_example",
  "impl": "naive",
  "llm": {
    "endpoint": "http://localhost:8000",
    "model": "your-model-id",
    "temperature": 0.0,
    "seed": 0,
    "max_tokens": 2048
  },
  "num_runs": 10,
  "warmup": 1,
  "telemetry": "off"
}
```

Recording always dispatches the `naive` tier. Use `pace-agent optimize ... --impl optimized` for
parity-gated candidate evaluation. A provider name is valid only when an installed provider
has registered implementations for the operations in the run.

`workload` labels the output directory; the positional `<agent>` selects executable code.
`tasks` is a list of objects such as `{"query": "..."}` — a whole suite lives in the one config,
with `task_defaults` supplying per-task settings the tasks leave out.
`parallel` and `cores_per_worker` configure independent multi-query profiling.
Multi-query runs store each task under `query_N/`, with the aggregate results at the root.
`pace-agent optimize <run-dir>` takes that root and replays every task into one workload summary; a
single `query_N/` directory can also be optimized on its own.

A task is stopped when it *stops making progress*, not when it takes long: PACE Agent watches the gap
between observable steps — a node entered or finished, a tool call returned, the model answered —
and aborts only once that gap exceeds 600s (set with `PACE_AGENT_MAX_TIMEOUT`, seconds, `0`
disables it). A task that legitimately needs an hour of stepping runs to completion; one wedged on
a hung page load is reported as a failure, with its last observed step named, and the rest of the
suite continues. The agent's own cleanup still runs, so browsers are closed and artifacts kept.

Add a `scoring` block (`{"gold_key": "gold"}`) to grade answers against an expected answer carried
by each task. PACE Agent withholds that field from the agent and records **both** a strict and a loose
verdict per task, counting each in the aggregate: strict is a floor (a right answer inside a sentence
fails it) and loose is a ceiling (containment over-credits), so neither is quoted alone. Tasks with
no expected answer are reported as unscored rather than wrong. Both are normalized string/number
heuristics, not any benchmark's official scorer.

### LLM endpoint and authentication

`llm` points at an **OpenAI-compatible** endpoint. PACE Agent sends an unrecorded capability probe
to detect `/responses` support (richer token accounting), then falls back to
`/chat/completions` when that route is unavailable.

| field | meaning |
|---|---|
| `endpoint` | OpenAI-compatible endpoint, with or without a trailing `/v1` |
| `model` | model id exactly as the endpoint expects it |
| `api_key_env` | preferred: name of the environment variable holding the API key |
| `subscription_header` | optional: also send the key as this header, for API-gateway auth (e.g. `Ocp-Apim-Subscription-Key`) |
| `vision` | declares that the backend accepts images; agent code must include image blocks |
| `image_format` | `openai` (default, `image_url` data-URI) or `anthropic` (`image`/`source`) |
| `temperature`, `max_tokens` | standard sampling controls |
| `seed` | sent to `/chat/completions`; currently not sent to `/responses` |

Prefer `api_key_env`; the key is resolved from the environment at call time:

```bash
export MY_LLM_KEY="..."       # config sets  "api_key_env": "MY_LLM_KEY"
```

A legacy inline `api_key` is accepted but unsafe because PACE Agent copies the source config into
the run directory. For multimodal input, agent code can build image blocks with
`llm.image_content(...)`.

## Task suites

`profile` runs a whole task suite through the same command as a single task — list several `tasks`
in the config and it profiles each one, then summarizes them:

```bash
pace-agent profile pace_agent.examples.webvoyager --config packages/pace_agent/configs/gaia_web_90.json --out run/
```

**WebVoyager** drives a real headless Chrome through live web tasks (observe the page's
accessibility tree → plan → act) and profiles each run. It needs a Chrome/Chromium browser,
and for vision mode a vision-capable endpoint. Details are in their own docs:

- **[docs/webvoyager.md](docs/webvoyager.md)** — browser runs: configuration, task suites, answer
  grading, the stall limit, vision mode, and the report they produce.
- **[docs/browser-setup.md](docs/browser-setup.md)** — installing Chrome and chromedriver for
  browser agents.

## Profile your own agent in one line

Any compiled LangGraph — including one you already have — can be profiled with no
PACE Agent-specific code in the agent:

```python
from pace_agent import profile_invoke
run = profile_invoke(graph, inputs, return_run=True)
run["node_timings"]   # per-node CPU time; unattributed work stays visible
```

See **[docs/agent-integration.md](docs/agent-integration.md)** for `instrument()`, `@pace_agent.op`,
and running `pace-agent profile` on any graph.

## Documentation

- **[docs/agent-integration.md](docs/agent-integration.md)** — profile any LangGraph agent in one line (`profile_invoke`, `instrument`, `@pace_agent.op`).
- **[docs/deterministic-replay.md](docs/deterministic-replay.md)** — the two benchmark steps: profile with characterization (step 1) and replay (step 2), including replay modes, the strict replay contract, and candidate coverage.
- **[docs/providers.md](docs/providers.md)** — the naive / optimized / provider tier model and registering a provider.
- **[docs/benchmark-methodology.md](docs/benchmark-methodology.md)** — measurement profiles, interleaving, bootstrap CIs, and tail-percentile gating.
- **[docs/trajectory-format.md](docs/trajectory-format.md)** — the safe msgpack trajectory format and codec registry.
- **[docs/webvoyager.md](docs/webvoyager.md)** / **[docs/browser-setup.md](docs/browser-setup.md)** — the browser workload and Chrome setup.
- **[docs/gaia-planner.md](docs/gaia-planner.md)** / **[docs/nsjail-agent.md](docs/nsjail-agent.md)** — GAIA planner runs and the nsjail code sandbox.
- **[tests/testing.md](tests/testing.md)** — how to run the optimized-tier test suites (parity, coverage, E2E).

## Telemetry

Telemetry is opt-in through the run config: set `telemetry` to `perf` for cycles, IPC, and
cache metrics, or `uprof` for AMD uProf process metrics. `pace-agent check` reports availability;
an unavailable optional collector is recorded without failing the profile. Install AMD uProf
from the [official guide](https://www.amd.com/en/developer/uprof.html).

Counters cover a separate whole-process naive replay and are not attributed to individual
tools. `pace-agent check` verifies collector prerequisites, not support for every requested event.
