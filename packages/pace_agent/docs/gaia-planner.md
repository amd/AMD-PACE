# GAIA Multi-Agent Planner

PACE Agent GAIA integration guide (`packages/pace_agent/docs/gaia-planner.md`). This is the
canonical documentation for running GAIA through `pace_agent.examples.run_gaia` and the
vendored `gaia_planner` / `nsjail_agent` packages (formerly `gaia_planner/README.md`).

This package coordinates the existing WebVoyager browser agent and
`nsjail-agent-tool` executor. A planning model can answer directly, delegate
focused web research, run code against task attachments, or combine both tools
over several turns.

The runner reads tasks from either source:

- the official GAIA dataset on HuggingFace (`gaia-benchmark/GAIA`), selectable by
  difficulty level, or
- local sample tasks under `packages/pace_agent/configs/gaia_sample_tasks.json`
  (attachments in `pace_agent/examples/fixtures/gaia/`).

The local exact-match summary is not the official GAIA metric.

## Installation

Install PACE Agent from this repository (editable recommended so the vendored
`gaia_planner` / `nsjail_agent` packages import):

```bash
pip install -e ./packages/pace_agent
```

Run GAIA through the PACE Agent entry point (same flags as the former `gaia-planner` CLI):

```bash
python -m pace_agent.examples.run_gaia --help
```

Set up WebVoyager-capable browsing via the in-process PACE Agent browser (default) or an
external checkout. See also **[browser-setup.md](browser-setup.md)** (Chrome setup) and
**[webvoyager.md](webvoyager.md)** (browser agent details). For the nsjail binary and
sandbox mounts, see **[nsjail-agent.md](nsjail-agent.md)**.

```bash
export NSJAIL_BIN=/path/to/nsjail
```

Without nsjail, code-tool calls fail closed. `--allow-local-executor` enables
unsandboxed host execution for development only and must not be used with
untrusted model output.

## PACE Agent run config

Prefer a JSON run config for workers, engine, HuggingFace selection, LLM endpoint,
and task selection (same idea as WebVoyager/`pace-agent profile` configs). Worker count is
**config-only** (`parallel` or `workers`); `--workers` / `-j` on the CLI are rejected.
Other CLI flags still override the config:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible /v1 URL
export OPENAI_MODEL=gpt-4o
export OPENAI_API_KEY=...
export HF_TOKEN=hf_...   # the example config selects the HuggingFace dataset

python -m pace_agent.examples.run_gaia --config packages/pace_agent/configs/gaia_run.example.json
```

Any string in the config may reference an environment variable as `${VAR}`, so
`packages/pace_agent/configs/gaia_run.example.json` names `${OPENAI_BASE_URL}` and
`${OPENAI_MODEL}` instead of hardcoding them; an unset name aborts the run with
the variable name rather than a confusing connection error. The API key is never stored
in the config — `llm.api_key_env` names the variable to read it from.

### Run one (or a few) specific tasks

Set `gaia.task_id` for a single id, or `gaia.task_ids` for an ordered list. When either
is set, `max_tasks` from the config is ignored (the id list is the selection):

```json
"gaia": {
  "huggingface": true,
  "level": 1,
  "split": "validation",
  "task_id": "7707f3dd-1aa6-42f5-847a-b66f3eaf2ee4"
}
```

```json
"gaia": {
  "task_file": "packages/pace_agent/configs/gaia_sample_tasks.json",
  "task_ids": ["gaia_015", "gaia_016"]
}
```

Set `gaia.skip_preflight` (or `GAIA_SKIP_PREFLIGHT=1`) when the endpoint health
check is unnecessary (e.g. offline/local servers).

## Model configuration

The planner and WebVoyager talk to any **OpenAI-compatible Chat Completions**
endpoint (`/v1/chat/completions`). Configure it with environment variables, CLI
flags, or the `llm` block in a run config:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

```bash
python -m pace_agent.examples.run_gaia ... \
  --model your-model-id \
  --base-url http://localhost:8000/v1 \
  --api-key your-key
```

The same endpoint is used for the planner and in-process WebVoyager. For a local
server (vLLM, llama.cpp server, PACE router, etc.), point `--base-url` /
`llm.endpoint` at that URL and skip the health check with `GAIA_SKIP_PREFLIGHT=1`
(or `gaia.skip_preflight: true` in a run config) — there is no `--skip-preflight`
CLI flag. Many local servers accept any `--api-key` value (including `EMPTY`).
The planner model must support tool calling unless you use `--no-tools` /
`gaia.no_tools`.

`--local-base-url` sends only the vision/audio tools (`analyze_video`,
`analyze_image`, `analyze_audio`) to a separate endpoint; omit it when everything
should use the same model server.

## Run the GAIA dataset from HuggingFace

`gaia-benchmark/GAIA` is a gated dataset. Accept its terms on the Hub, then
provide an access token via `HF_TOKEN` (or `--hf-token`):

```bash
export HF_TOKEN=hf_...

# All levels, validation split (scored against reference answers)
python -m pace_agent.examples.run_gaia --huggingface

# Only Level 1 tasks (implies --huggingface)
python -m pace_agent.examples.run_gaia --level 1

# Level 2, first 10 tasks
python -m pace_agent.examples.run_gaia --level 2 --max-tasks 10

# Test split (no reference answers; runs unscored)
python -m pace_agent.examples.run_gaia --huggingface --split test
```

Attachments (spreadsheets, PDFs, images, ...) referenced by each GAIA row are
resolved from the HuggingFace cache and mounted into the sandbox automatically.
The run summary reports overall accuracy plus a per-level breakdown
(`summary.json` → `by_level`).

## Run local tasks

Execution/file samples:

```bash
python -m pace_agent.examples.run_gaia \
  packages/pace_agent/configs/gaia_sample_tasks.json \
  --task-id gaia_015 \
  --model gpt-4o
```

Attachments for those local tasks live under `pace_agent/examples/fixtures/gaia/`.

Web samples (in-process PACE Agent WebVoyager by default):

```bash
python -m pace_agent.examples.run_gaia \
  --level 1 --split validation --max-tasks 5 \
  --model gpt-4o
```

Useful options include `--max-turns`, `--web-max-iterations`,
`--web-max-attached-imgs`, `--web-text-only`, `--task-id`, `--max-tasks`, and
`--output-dir`. Run `python -m pace_agent.examples.run_gaia --help` for the complete list.

WebVoyager attaches recent screenshots to each vision turn. If the model
endpoint returns HTTP 400 only after several successful browsing turns, the
accumulated multimodal payload is exceeding a per-request context limit; lower
`--web-max-attached-imgs` (for example to `1`) to shrink each request.

## Outputs

Each invocation creates a timestamped directory:

```text
gaia-results/<timestamp>/
├── results.jsonl
├── summary.json
└── web/
    └── call-<id>/<webvoyager timestamp>/taskplanner-web/
        ├── result.json
        ├── interact_messages.json
        └── screenshots and logs
```

`results.jsonl` includes the final answer, complete planner message trajectory,
tool events, expected local answer, and local exact-match result. A task record
is flushed immediately, so completed work survives a later task failure.

## Current limitations

- WebVoyager launches a fresh browser subprocess for every `browse_web` call.
- WebVoyager still uses its strict `Thought`/`Action` protocol internally.
- Browser access requires Chrome/Chromium and live network connectivity.
- Attachment analysis depends on libraries and command-line tools mounted into
  the nsjail environment.
- The local scorer only case-folds and normalizes whitespace.
