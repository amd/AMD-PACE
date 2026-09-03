# Browser agent runs (WebVoyager, GAIA web)

`pace-agent profile pace_agent.examples.webvoyager --config <cfg>` runs the **WebVoyager** agent over one
task or a whole suite and profiles each one.

The agent (`pace_agent.examples.webvoyager`) is a strict [LangGraph](https://github.com/langchain-ai/langgraph)
**observe → plan → act** loop that drives a real headless Chrome: it reads the page's
accessibility tree, asks the LLM for the next action, and executes it (click / type / scroll /
navigate / answer), repeating until the task is answered or `max_iter` is reached.

## Prerequisites

- **A browser** — Chrome/Chromium + chromedriver. See **[browser-setup.md](browser-setup.md)**.
  Verify with `pace-agent check` (look for `[ok] selenium` and `[ok] chrome`).
- **An LLM endpoint** — any OpenAI-compatible endpoint (see *Configuration* in the main
  [README](../README.md)). For **vision mode**, a vision-capable model.

## Run

```bash
# one task, or a suite — the same command either way
pace-agent profile pace_agent.examples.webvoyager --config packages/pace_agent/configs/pace_agent_wv_vision.json \
  --out pace_agent_results/wv/

# the checked-in GAIA web suite (90 tasks with their reference answers)
pace-agent profile pace_agent.examples.webvoyager --config packages/pace_agent/configs/gaia_web_90.json --out pace_agent_results/gaia/

# check the plumbing on one task first: --query replaces the config's tasks (ungraded, no start_url)
pace-agent profile pace_agent.examples.webvoyager --config packages/pace_agent/configs/gaia_web_90.json \
  --query "how many studio albums did Mercedes Sosa publish between 2000 and 2009?" \
  --out pace_agent_results/gaia_smoke/

# the report is a separate step, so a run can be re-rendered after optimize without re-running it
pace-agent report pace_agent_results/gaia/
```

A suite writes one `query_<i>/` directory per task (each with its own `result.json`, `trace.json`
and trajectory) plus a run-level `result.json` holding the per-task table and the aggregate.
Progress is printed per task as it starts and finishes, and failures are printed again at the end.

## Configuration

```json
{
  "tasks": [
    {
      "query": "derivative of x^2 when x=5.6",
      "start_url": "https://www.wolframalpha.com/",
      "gold": "11.2",
      "max_iter": 12
    }
  ],
  "scoring": { "gold_key": "gold" },
  "llm_backend": {
    "type": "openai",
    "base_url": "https://your-endpoint/v1",
    "model": "your-vision-model",
    "vision": true,
    "image_format": "openai",
    "api_key_env": "YOUR_LLM_KEY",
    "temperature": 0.0,
    "seed": 0,
    "max_tokens": 2048
  },
  "parallel": 12,
  "cores_per_worker": 4,
  "num_runs": 1,
  "warmup_runs": 0
}
```

### Tasks

Each entry in `tasks` is one task (the older `graphs: [{spec, kwargs}]` shape still loads):

- `query` — the instruction for the agent (required)
- `start_url` — the page to open first (required)
- `max_iter` — caps the observe/plan/act steps
- `max_context_obs` — how many recent observation/action pairs to resend (default `6`; set
  `0` to send the full history). Prefix caching lowers cost on the stable prefix but does
  not raise the context-window limit — clipping prevents long runs from hitting /chat 400.
- `gold` — the expected answer, used only if the config has a `scoring` block (see below)
- `capture_snapshots` — store the page snapshots that `pace-agent optimize` replays (default on)

Run several tasks concurrently with `"parallel": N` (a process pool; each worker is pinned to
`"cores_per_worker"` cores so runs don't contend). `num_runs` is ignored for a browser agent —
its work cannot be re-timed from a tape, so every task is one live run and PACE Agent says so rather
than pretending otherwise.

A whole suite lives in one config: settings first, then every task under `tasks`.
`configs/gaia_web_90.json` is the checked-in GAIA web subset (26 level-1 + 64 level-2 questions
answerable from a browser, each with its reference answer). A setting the whole suite shares belongs
in `task_defaults` rather than repeated on all 90 tasks — a task that names the field itself wins:

```json
{
  "task_defaults": { "max_iter": 25, "capture_snapshots": false }
}
```

### Grading answers

With a `scoring` block, PACE Agent compares the recorded answer to the expected one:

```json
"scoring": { "gold_key": "gold" }
```

`gold_key` names the task field holding the expected answer (`gold` by default). Every graded task
gets **both** verdicts, recorded as `match_strict` and `match_loose` and counted in the aggregate as
`auto_match_strict` / `auto_match_loose`:

- `strict` is normalized equality — casefold, collapse whitespace, drop currency and thousands
  separators, then the whole answer must be the expected value.
- `loose` also accepts the expected answer inside a sentence, or a number in the answer equal to the
  expected one.

Neither is reported alone, because neither is the accuracy: strict is a floor, since an agent that
answers "3 studio albums." fails it while being right, and loose is a ceiling, since containment
credits an expected `17` to an answer of `17000`. The gap between them is the size of the doubt.

The expected answer is **withheld from the agent** — PACE Agent removes that field from the task the
agent receives, whether or not grading is on, so turning grading on cannot change the run.

A task with no expected answer is *unscored* (both verdicts `null`), not wrong, and the aggregate
reports `n_scored` / `n_unscored` separately. A task that has one but produced no answer — it
crashed or stopped progressing — is a **miss**, so failures stay in the accuracy denominator.

> Both are normalized string/number heuristics, not any benchmark's official scorer. Read the
> answers before quoting accuracy.

### Stall limit

A browser task is stopped when it goes quiet, not when it gets long. PACE Agent measures the gap
between observable steps — a graph node entered or finished, a tool call returned, the model
answered, a snapshot observed — and aborts the task only once that gap exceeds 600s:

```bash
PACE_AGENT_MAX_TIMEOUT=900 pace-agent profile ... # seconds of no progress; 0 disables the watchdog
```

A hard per-task deadline would be the wrong instrument here: web tasks vary by an order of
magnitude in how many steps they need, so a deadline low enough to catch a wedged worker also kills
the legitimately long questions. This way a task that keeps stepping runs as long as it needs,
while one hung on a page load, a driver that never returns, or a gateway that accepted the request
and went silent is cut promptly. `max_iter` remains the bound on *steps*.

The abort surfaces as an ordinary exception inside the agent, so its `finally` closes the browser;
the task's artifacts and error are written (naming the last step seen, e.g. `last progress: node
observe started`), and the suite continues with the remaining tasks.

### Vision (Set-of-Marks) mode

With `"vision": true`, the agent also sends a screenshot to the model. PACE Agent draws a
**Set-of-Marks** overlay — numbered boxes aligned to the *same* accessibility-tree elements the
text observation labels — so the model can refer to elements by number. Set `image_format` to
`openai` (data-URI `image_url`, the default) or `anthropic` (`image` / `source`) to match your
model. Gateway auth (`api_key_env` / `subscription_header`) works exactly as in the main
README's LLM section.

## What it reports

Per task: whether it `completed`, the number of `steps`, the final `answer`, the graded `match`
(when scoring is on), `e2e`, `per_node` CPU timings, and an `observe_ab` naive-vs-pace
accessibility-tree observation A/B.

Across the suite (`aggregate`): queries, answered, errors, total E2E, LLM calls, context/cached/
output tokens, non-model CPU, pooled per-node time, the median in-run observe speedup, and the
auto-match counts. `pace-agent report <run_dir>` renders all of it as `report.md`.

> The live run is always the **naive** baseline — a real, faithful WebVoyager run; there is no
> impl toggle. The naive-vs-pace A/B is produced afterwards by `pace-agent optimize <run_dir>`, which
> replays the stored page snapshots under both implementations on identical DOM state.
