# Deterministic replay

`pace-agent profile` records a run — every tool call (inputs, output, timing) and every LLM
response — into a trajectory. `pace-agent optimize` then **replays** that recording so the CPU work
can be re-measured under a different implementation tier **without re-hitting the model**.
Replay is driven entirely by the recorded data; PACE Agent core never imports a workload or example
to replay it.

A benchmark runs in two steps:

| step | command | what it does |
|---|---|---|
| **1. Profile with characterization** | `pace-agent profile --characterize-runs N` | samples N live runs, records the representative one as a trajectory, and re-times it |
| **2. Replay** | `pace-agent optimize` | re-times that trajectory under a candidate tier |

Characterization makes the recording **representative**; replay makes it **reproducible**.
`--characterize-runs` is off by default but recommended: without it, `profile` records whichever
live run went first.

## Step 1: Profile with characterization

An agent asked the same question twice does not do the same thing twice: one run answers in 3
LLM calls, the next loops for 15. Record the 15-call run and the benchmark replays perfectly while
measuring a workload that almost never happens. Runs differ even at `temperature: 0`: batching and
kernel reduction order on a served endpoint still move outputs, and one differing token early in a
loop changes every tool call after it. So `profile` characterizes first: it runs the agent N times
live and records a representative run instead of the first one.

```bash
pace-agent profile pace_agent.examples.webvoyager \
  --config packages/pace_agent/configs/pace_agent_wv_characterize.json \
  --characterize-runs 10 \
  --characterize-strategy mode \
  --out pace_agent_results/wv/
```

For each task, `profile`:

1. **characterizes** — runs the agent N times live and picks the representative run (see
   [Strategies](#strategies));
2. **records** — saves that run as `trajectory.msgpack`, next to `characterization.json` (the
   report), `result.json` and `trace.json`. Every sampled run was already recorded, so this adds
   no live run;
3. **re-times** — replays the trajectory `num_runs` times (after `warmup_runs` warmups) for its E2E
   and per-tool timings. Browser agents (`REPLAY_SAFE = False`, e.g. WebVoyager) cannot be
   re-timed from a tape, so they keep the live timing.

Characterization runs once per task, so a suite costs `characterize_runs × tasks` live runs. With
`characterize_runs: 1` (the default), `profile` records the first live run, as before.

### Configuration

```json
{
  "characterize_runs": 10,
  "characterize": {
    "strategy": "auto",
    "granularity": "action",
    "min_success": 3,
    "keep_trajectories": false
  },
  "num_runs": 10
}
```

| key | default | meaning |
|---|---|---|
| `characterize_runs` | `1` | live runs per task; `1` turns characterization off and records the first run |
| `characterize.strategy` | `auto` | how the representative is picked (below) |
| `characterize.granularity` | `action` | what makes two paths differ: `action` (step names), `action_arg` (names plus a normalized argument, so `Click [12]` ≠ `Click [40]`), `coarse` (order-insensitive multiset) |
| `characterize.min_success` | `1` | live runs that must succeed; failed runs are logged and skipped, and too few raises `CharacterizationFailed` (exit code 9) |
| `characterize.keep_trajectories` | `false` | keep every sampled run under `raw/characterize/run_<i>/`, so another strategy can be tried without new live runs |

`--characterize-runs` and `--characterize-strategy` override these keys; an invalid value is
rejected before any live run. `characterize_runs` counts **live** runs (deciding what to record);
`num_runs` counts **replays** (re-timing it).

### Strategies

| strategy | picks |
|---|---|
| `auto` (default) | `mode` when the modal path actually repeated; falls back to `medoid` when every path was distinct — a "most frequent" of one is a coin flip dressed up as a decision |
| `mode` | the most frequent path, then the most ordinary run within that cluster |
| `medoid` | the run closest to all others, blending path edit distance and numeric distance (call and token counts) half and half |
| `first` | run 0 — the behaviour without characterization, kept as an explicit opt-out rather than an accident |

Every strategy picks a run that actually happened (an averaged trajectory is not a tape any agent
produced), never selects on wall-clock time (that is what the benchmark measures), and breaks ties
on the lowest run index.

### What a path is

A path is the ordered list of steps one run took, read from what the trajectory already carries,
so it works for any agent. The first source present wins:

| `path_source` | read from | example path | shipped agents |
|---|---|---|---|
| `agent_steps` | the agent's step log, `meta["result"]["per_step"]` | `act:type act:click act:answer` | `webvoyager`, `testgen` |
| `tool_calls` | recorded tool calls | `tool:tokenize tool:bm25 tool:dedup tool:assemble` | `basic_example`, `customer_support`, `large_corpus_rag`, `tool_agent`, `sandbox_tool_agent` (GAIA-style tool loop) |
| `llm_calls` | the LLM tape (fallback) | `llm:call llm:call llm:call` | — |

A custom agent can supply its own with `pace_agent.characterize.register_step_extractor(fn, name, priority=200)`.

### Reading the result

Characterization judges the sample on two axes:

- **path** (`stability`) — did the runs do the same things in the same order? `deterministic`
  (all agreed), `dominant` (a strict majority took one path), `diverse` (no majority, but a path
  repeated), `divergent` (no path repeated).
- **work** (`work_stability`) — did they do the same amount of work? `uniform`; `dispersed` (a
  feature's `(max - min) / median` exceeds 0.5); `outlier` (a run's modified z-score exceeds 3.5).

Fixed-graph agents (`basic_example`, `customer_support`, `large_corpus_rag`, `tool_agent`) walk the
same path every run by construction, so `deterministic` there is not a finding:
`path_is_uninformative: true` flags it, and the work axis decides. The headline `verdict` takes the
weaker axis, so it never oversells:

| verdict | meaning |
|---|---|
| `representative` | the runs agreed on both what they did and what it cost |
| `wide` | the recorded run is the most central available, but the sample around it is broad |
| `no-majority` | the runs never converged on a path; the recording is one of several behaviours |
| `unrepresentative` | the recorded run is itself an outlier in its own sample |
| `unsampled` | fewer than two live runs — characterization has no opinion |

`characterization.json` holds the full record (path clusters, per-feature spread, outliers,
step-to-step transitions). Its summary is also written to `result.json` and the trajectory's
`meta["characterization"]`, and `pace-agent report` renders it above the timings. A
`customer_support` sample in which run 2 ran away:

```json
{
  "n_runs": 10, "n_distinct_paths": 1, "modal_share": 1.0,
  "stability": "deterministic", "path_is_uninformative": true,
  "work_stability": "outlier", "verdict": "wide",
  "selected_run": 0, "strategy": "mode", "applied": "mode",
  "reason": "most frequent path: 10 of 10 live runs (100%) took it",
  "typicality_max_abs_z": 0.0, "selected_is_outlier": false, "outlier_runs": [2]
}
```

Every run walked `tokenize → bm25 → dedup → assemble`, but run 2 spent 1262 output tokens against a
median of 9. The recorded run 0 is ordinary (`typicality_max_abs_z`, its largest modified z-score,
is 0), but the sample is not, so the verdict is `wide` rather than `representative`. Outlier runs
are reported, never silently dropped.

### Cost and limits

Characterization costs N live runs per task, run one after another so they do not contend for the
cores being measured. On the 90-task GAIA web suite, `--characterize-runs 10` is 900 live browser
sessions, so try a few tasks first. Characterization does not make an unstable workload stable; it
makes the instability visible in the verdict instead of freezing it silently into the recording.

## Step 2: Replay

`optimize` replays the trajectory from step 1 under a candidate tier, writing
`optimize_<timestamp>.json` and refreshing `report.md`:

```bash
pace-agent optimize pace_agent_results/wv/
```

For browser agents, `optimize` replays the recorded page snapshots instead (`browser-snapshot`
mode below).

### Replay modes

A trajectory records how it should be replayed in `meta["replay_mode"]`:

- **`deterministic`** (default) — op-replay: recorded tool calls are re-timed on their
  recorded inputs, tier by tier. Registered in core.
- **`browser-snapshot`** — for browser agents: recorded page snapshots are reloaded into a
  frozen `file://` DOM and the accessibility-tree observation is timed naive-vs-pace on the
  identical state (exact parity, reproducible timing). Registered by
  `pace_agent.replay.browser_snapshot`.

New modes register via `pace_agent.replay.modes.register_replay_mode(name, handler)`; `optimize`
looks the mode up rather than importing the workload that produced the trajectory.

### The replay contract (trust)

Replay is only trustworthy if it fails loudly when a trajectory no longer matches the agent.
On the first replay iteration the player validates a cursor over the recorded streams:

- tool-call **count**, and per position the op **name + bucket + argument signature**;
- no extra call (beyond what was recorded) and no missing call;
- the LLM tape is **fully consumed** (no leftover responses).

The first divergence raises a precise `ReplayDivergence` (e.g. *expected `vector_search`,
got `bm25`*) instead of a downstream `IndexError` or a silently-wrong number. `optimize`
catches it and writes the result as **invalid** — an invalid replay can never produce a
speedup.

The result carries a `replay_contract` block:

```json
{ "validated": true, "tool_calls_expected": 3, "tool_calls_consumed": 3,
  "llm_calls_expected": 2, "source_match": true, "valid": true }
```

### Source / config identity

The recorder stamps `meta["identity"]` (PACE Agent + Python versions, agent source digest) so
replay can detect drift. The compatibility policy is `strict | warn | allow` (default `warn`):
under `strict`, a source mismatch marks the replay non-equivalent (`valid: false`).

### Coverage (honest end-to-end)

A candidate tier (e.g. `pace`) rarely covers every op. `optimize` re-times **only** the
covered ops and projects the honest end-to-end: the un-covered ops keep their recorded naive
time (Amdahl-correct — a 100× win on 4% of the runtime is a ~4% end-to-end win, not "100×").

The result reports `coverage {total_calls, candidate_calls, covered_ops, covered_time_fraction}`
and a per-op table (reference vs candidate). **Zero candidate coverage is invalid, never a
phantom speedup**, and each covered op is parity-checked against the reference.
