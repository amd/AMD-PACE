# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Run gaia-planner GAIA tasks with PACE Agent WebVoyager and parallel workers.

This is the PACE Agent-side integration point: it reuses gaia-planner's own CLI (adapters,
nsjail mounts, task loading, scoring) *unchanged* and only

  1. uses gaia-planner's native OpenAI-compatible client (same as ``gaia-planner`` CLI),
  2. serves ``browse_web`` with PACE Agent's own in-process LangGraph WebVoyager
     (``pace_agent.examples.webvoyager``) instead of shelling out to an external
     ``WebVoyager/run.py`` — so ``--webvoyager-root`` and an external checkout are no longer
     needed. Set ``GAIA_USE_EXTERNAL_WEBVOYAGER=1`` to restore the subprocess behavior, and
  3. optionally fans selected tasks across N processes (``parallel`` / ``workers`` in
     ``--config``) with a shared work queue.

gaia-planner is left completely untouched; the wiring lives here via lightweight
monkeypatching of a few module-level helpers.

Usage (same arguments as ``python -m gaia_planner.cli``, plus PACE Agent ``--config``)::

    export LLM_GATEWAY_KEY=...  LLM_GATEWAY_BASE_URL=https://.../v1  LLM_GATEWAY_MODEL=gpt-4o

    # Local sample tasks (config + attachments under configs/ and examples/fixtures/gaia/):
    python -m pace_agent.examples.run_gaia \
        packages/pace_agent/configs/gaia_sample_tasks.json --task-id gaia_015 \
        --model "$LLM_GATEWAY_MODEL" --base-url "$LLM_GATEWAY_BASE_URL" --api-key "$LLM_GATEWAY_KEY"

    # PACE Agent run config (workers/engine/HF selection/LLM endpoint/task_id) — see
    # packages/pace_agent/configs/gaia_run.example.json. Worker count is config-only
    # (``parallel`` or ``workers``); CLI ``--workers`` is rejected:
    python -m pace_agent.examples.run_gaia --config packages/pace_agent/configs/gaia_run.example.json

    # Official GAIA dataset from HuggingFace (gated: accept its terms on the Hub and set
    # HF_TOKEN). --level implies --huggingface; --max-tasks caps how many run. Attachments
    # are pulled from the HuggingFace cache, so no local task/attachment files are needed:
    export HF_TOKEN=hf_...
    python -m pace_agent.examples.run_gaia \
        --level 1 --split validation --max-tasks 5 \
        --model "$LLM_GATEWAY_MODEL" --base-url "$LLM_GATEWAY_BASE_URL" --api-key "$LLM_GATEWAY_KEY"

Per-task PACE Agent ``result.json`` summaries are written under ``$PACE_AGENT_GAIA_OUT`` (default
``pace-agent-gaia/<timestamp>/``), one directory per task id. The GAIA answers/scores still land
in gaia-planner's own ``--output-dir`` exactly as before.

Parallelism: set ``"parallel": N`` (or ``"workers": N``) in the PACE Agent run config to fan the
selected tasks across N isolated processes fed by ONE shared work queue. Each worker loads
the dataset and builds its own client/browser/sandbox state once, then pulls the next task
off the queue as soon as it is free — so the workers stay busy end-to-end and finish at
roughly the same time (dynamic load balancing). Each task still runs in exactly one worker
(keyed by a unique id). A single gateway preflight runs up front, and per-task summaries
plus a merged ``results.jsonl``/``summary.json`` are written just like the sequential run.

A single task (or an explicit list) can be selected in the same config via ``gaia.task_id``
or ``gaia.task_ids`` — see ``gaia_run.example.json``.

No-tools mode: pass ``--no-tools`` (or set ``GAIA_NO_TOOLS=1``) to hide every tool from the
model so it answers each question directly from its own knowledge — no ``tools`` schema and no
``tool_choice="auto"`` is ever sent. This also lets models whose deployment lacks tool-calling
config (which otherwise 400 on ``"auto" tool choice requires --enable-auto-tool-choice``) run
GAIA. The planner's ``CLOSED_BOOK_PROMPT`` is used (including inlined attachment text), and the
langgraph verify-by-execution gate is turned off automatically.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import sys
import time
from pathlib import Path

from pace_agent.harness import (
    count_tool_failures as _count_tool_failures,
    preflight_chat_gateway as _preflight_gateway,
    print_task_token_stats as _print_task_token_stats,
    write_profile_like_result,
)

# Shared between the patched _build_client (OpenAI client + optional PACE Agent LLMClient for
# in-process WebVoyager) and the WebVoyager adapter. Reset at the start of every main() call.
_STATE: dict[str, object] = {}

# Set in a parallel worker process (see :func:`_worker_loop`) to the shared task-id queue.
# When present, :func:`main` installs a selector that drains it, so gaia's own task loop pulls
# the next task dynamically instead of running a fixed, pre-assigned shard.
_WORKER_TASK_QUEUE = None


_TRANSIENT_BROWSER_SIGNS = (
    "invalidsessionid",
    "session deleted",
    "invalid session id",
    "disconnected",
    "chrome not reachable",
    "err_connection",
    "err_timed_out",
    "err_name_not_resolved",
    "err_internet_disconnected",
    "connection_timed_out",
    "renderer",
)


def _is_transient_browser_error(exc: BaseException) -> bool:
    """True for Selenium/Chrome crashes that a fresh browser session can recover from
    (session lost, renderer gone, transient network) — as opposed to a gateway/LLM 400,
    which retrying would only repeat."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(sign in blob for sign in _TRANSIENT_BROWSER_SIGNS)


def _make_llm_client(*, model: str, base_url: str, api_key: str):
    """PACE Agent ``LLMClient`` used only by in-process WebVoyager (browse_web).

    The GAIA planner itself uses gaia-planner's native OpenAI SDK client; WebVoyager's
    ``run(llm, task)`` still expects a PACE Agent ``LLMClient``.
    """
    from pace_agent.llm.client import LLMClient

    gateway_key = os.getenv("LLM_GATEWAY_KEY")
    headers = {"Ocp-Apim-Subscription-Key": gateway_key} if gateway_key else None
    return LLMClient(
        endpoint=base_url,
        model=model,
        temperature=0.0,
        seed=0,
        max_tokens=4096,
        api_key=api_key,
        headers=headers,
    )


def _usage_field(usage, *names):
    """Read the first present attribute/key from an OpenAI ``usage`` object or dict."""
    if usage is None:
        return None
    for name in names:
        if isinstance(usage, dict):
            if name in usage and usage[name] is not None:
                return usage[name]
        else:
            val = getattr(usage, name, None)
            if val is not None:
                return val
    return None


_PLANNER_COMPONENT = "planner"

# Which component is making the current model call. The planner's own reasoning turns run
# under the default; :func:`main` re-labels it to ``tool:<name>`` for the duration of each
# tool call, so LLM work done *inside* a tool (vision/audio adapters, WebVoyager) is billed
# to that tool instead of the planner.
_CURRENT_COMPONENT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "pace_agent_gaia_component", default=_PLANNER_COMPONENT
)


def _component_metrics(
    events: list[dict], tools_invoked: list[str] | None = None
) -> dict:
    """Group token usage by the component that spent it (planner vs each tool).

    Ordered planner-first, then the tools by descending total (context + output) tokens so the
    biggest consumer is immediately visible.
    """
    grouped: dict[str, dict] = {}

    def _slot(name: str) -> dict:
        return grouped.setdefault(
            name,
            {
                "llm_calls": 0,
                "context_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "reasoning_tokens": None,
                "backends": set(),
            },
        )

    for name in tools_invoked or []:
        _slot(name)
    for e in events:
        slot = _slot(e.get("component") or _PLANNER_COMPONENT)
        slot["llm_calls"] += 1
        backend = e.get("backend")
        if backend:
            slot["backends"].add(backend)
        for key, field in (
            ("context_tokens", "prompt"),
            ("output_tokens", "completion"),
            ("cached_tokens", "cached"),
            ("reasoning_tokens", "reasoning"),
        ):
            val = e.get(field)
            if val is not None:
                slot[key] = (slot[key] or 0) + val

    def _rank(item):
        name, m = item
        total = (m["context_tokens"] or 0) + (m["output_tokens"] or 0)
        return (name != _PLANNER_COMPONENT, -total, name)

    out: dict[str, dict] = {}
    for name, m in sorted(grouped.items(), key=_rank):
        backends = sorted(m.pop("backends"))
        row = dict(m)
        # Single backend when all calls for this component hit the same endpoint family;
        # "mixed" if somehow both; omitted when the tool spent no model tokens.
        if backends:
            row["backend"] = backends[0] if len(backends) == 1 else "mixed"
        out[name] = row
    return out


class _UsageTracker:
    """Accumulate per-call token usage from OpenAI ``response.usage`` objects."""

    def __init__(self):
        self.events: list[dict] = []
        # (event index at invocation, component) for every tool call, so a tool that spends no
        # model tokens still shows up in the breakdown as a 0-token row.
        self.tool_marks: list[tuple[int, str]] = []

    def note_tool(self, component: str) -> None:
        self.tool_marks.append((len(self.events), component))

    def record(self, response, wall_s: float, *, backend: str = "gateway") -> None:
        usage = getattr(response, "usage", None)
        prompt = _usage_field(usage, "prompt_tokens", "input_tokens")
        completion = _usage_field(usage, "completion_tokens", "output_tokens")
        details = _usage_field(usage, "prompt_tokens_details", "input_tokens_details")
        cached = _usage_field(details, "cached_tokens") if details is not None else None
        if cached is None:
            cached = _usage_field(usage, "cached_tokens")
        cdetails = _usage_field(
            usage, "completion_tokens_details", "output_tokens_details"
        )
        reasoning = (
            _usage_field(cdetails, "reasoning_tokens") if cdetails is not None else None
        )
        if reasoning is None:
            reasoning = _usage_field(usage, "reasoning_tokens")
        self.events.append(
            {
                "prompt": prompt,
                "completion": completion,
                "cached": cached,
                "reasoning": reasoning,
                "wall": wall_s,
                "component": _CURRENT_COMPONENT.get(),
                "backend": backend,
            }
        )

    def absorb_llm_client(
        self, llm, mark: int, component: str, *, backend: str = "gateway"
    ) -> None:
        """Fold model calls made through a PACE Agent ``LLMClient`` into this tracker.

        In-process WebVoyager drives its own ``LLMClient`` rather than the wrapped OpenAI
        client, so its turns are invisible here. Copy the calls appended since ``mark`` in so
        ``browse_web`` shows up in the per-component breakdown with the rest.
        """
        if llm is None:
            return
        calls = list(getattr(llm, "calls", None) or [])[mark:]
        if not calls:
            return

        def _slice(name):
            return list(getattr(llm, name, None) or [])[mark:]

        prompts = _slice("prompt_tokens")
        completions = _slice("completion_tokens")
        cached = _slice("cached_tokens")
        reasoning = _slice("reasoning_tokens")

        def _at(xs, i):
            return xs[i] if i < len(xs) else None

        for i, wall in enumerate(calls):
            self.events.append(
                {
                    "prompt": _at(prompts, i),
                    "completion": _at(completions, i),
                    "cached": _at(cached, i),
                    "reasoning": _at(reasoning, i),
                    "wall": wall,
                    "component": component,
                    "backend": backend,
                }
            )

    def mark(self) -> int:
        return len(self.events)

    def metrics_since(self, mark: int, *, wall_s: float | None = None) -> dict:
        ev = self.events[mark:]

        def _sum(key):
            xs = [e[key] for e in ev if e.get(key) is not None]
            return sum(xs) if xs else None

        def _max(key):
            xs = [e[key] for e in ev if e.get(key) is not None]
            return max(xs) if xs else None

        out = _sum("completion")
        rea = _sum("reasoning")
        work = None if out is None else out - (rea or 0)
        llm_wall = sum(e["wall"] for e in ev) if ev else 0.0
        return {
            "llm_calls": len(ev),
            "api": "chat",
            "context_tokens_total": _sum("prompt"),
            "context_tokens_max": _max("prompt"),
            "cached_tokens_total": _sum("cached"),
            "output_tokens_total": out,
            "reasoning_tokens_total": rea,
            "work_tokens_total": work,
            "e2e_median": float(wall_s if wall_s is not None else llm_wall),
            "by_component": _component_metrics(
                ev, [name for at, name in self.tool_marks if at >= mark]
            ),
        }


class _TrackingCompletions:
    def __init__(self, inner, tracker: _UsageTracker, *, backend: str):
        self._inner = inner
        self._tracker = tracker
        self._backend = backend

    def create(self, *args, **kwargs):
        t0 = time.perf_counter()
        resp = self._inner.create(*args, **kwargs)
        try:
            self._tracker.record(resp, time.perf_counter() - t0, backend=self._backend)
        except Exception:  # noqa: BLE001 - never break the planner over accounting
            pass
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TrackingChat:
    def __init__(self, inner, tracker: _UsageTracker, *, backend: str):
        self._inner = inner
        self.completions = _TrackingCompletions(
            inner.completions, tracker, backend=backend
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _UsageTrackingClient:
    """Thin OpenAI-client wrapper that records ``usage`` from each chat completion.

    Preserves the full SDK surface (``models``, ``chat``, headers, …) via ``__getattr__`` so
    gaia-planner and its vision/audio adapters keep working unchanged — only token accounting
    is added. This is intentionally *not* a PACE Agent LLMClient / trajectory recorder.

    The planner client and the optional ``--local-base-url`` vision/audio client share one
    ``_UsageTracker`` (via ``tracker=``) so Radeon/local multimodal calls land in the same
    per-task breakdown as gateway planner turns.
    """

    def __init__(
        self, client, tracker: _UsageTracker | None = None, *, backend: str = "gateway"
    ):
        self._client = client
        self.usage = tracker if tracker is not None else _UsageTracker()
        self.backend = backend
        self.chat = _TrackingChat(client.chat, self.usage, backend=backend)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _attributed_call_tool(original_call_tool):
    """Wrap ``GaiaPlanner._call_tool`` so model tokens spent *inside* a tool are billed to that
    tool rather than to the planner.

    ``LangGraphGaiaPlanner`` inherits ``_call_tool``, so patching the base class covers both
    engines. Vision/audio adapters share either the planner client or a ``--local-base-url``
    client — both are wrapped onto the same tracker, and the component label attributes their
    calls. WebVoyager drives its own ``LLMClient`` and is folded in from that client's
    counters once the call returns.
    """

    def _call_tool(self, tool_call, task, execution, web_state=None):
        name = getattr(getattr(tool_call, "function", None), "name", None) or "tool"
        component = f"tool:{name}"
        tracker = _STATE.get("usage")
        if not isinstance(tracker, _UsageTracker):
            client = _STATE.get("client")
            tracker = client.usage if isinstance(client, _UsageTrackingClient) else None
        llm = _STATE.get("llm")
        llm_mark = len(getattr(llm, "calls", None) or []) if llm is not None else 0
        if tracker is not None:
            tracker.note_tool(component)
        token = _CURRENT_COMPONENT.set(component)
        try:
            return original_call_tool(self, tool_call, task, execution, web_state)
        finally:
            _CURRENT_COMPONENT.reset(token)
            if tracker is not None:
                tracker.absorb_llm_client(llm, llm_mark, component)

    return _call_tool


class _PaceAgentWebVoyagerAdapter:
    """Serve ``browse_web`` with PACE Agent's in-process LangGraph WebVoyager
    (``pace_agent.examples.webvoyager``) instead of shelling out to an external ``WebVoyager/run.py``.

    Drives an in-process ``SeleniumBrowser`` via a dedicated PACE Agent ``LLMClient`` stored in
    ``_STATE["llm"]`` (the planner uses a separate OpenAI SDK client). Constructed by
    gaia-planner's CLI with a ``WebVoyagerConfig``; only ``max_iterations`` is honored
    (headless text-only browser).
    """

    def __init__(self, config: object) -> None:
        self.config = config

    def call(self, goal: str, start_url: str = "https://duckduckgo.com/") -> str:
        import json as _json

        llm = _STATE.get("llm")
        if llm is None:
            return _json.dumps(
                {
                    "status": "error",
                    "answer": None,
                    "error": "PACE Agent LLM client not initialized for WebVoyager",
                }
            )
        from pace_agent.examples import webvoyager as _wv

        task = {
            "query": goal,
            "start_url": start_url or "https://duckduckgo.com/",
            "max_iter": int(getattr(self.config, "max_iterations", 15) or 15),
            "capture_snapshots": True,
        }
        ctx_obs = os.getenv("PACE_AGENT_WEBVOYAGER_MAX_CTX_OBS")
        if ctx_obs:
            task["max_context_obs"] = int(ctx_obs)
        try:
            max_seconds = float(os.getenv("PACE_AGENT_WEBVOYAGER_MAX_SECONDS", "180"))
        except ValueError:
            max_seconds = 180.0
        if max_seconds > 0:
            task["max_seconds"] = max_seconds
        out = None
        for attempt in range(2):
            try:
                out = _wv.run(llm, task)
                break
            except Exception as exc:  # noqa: BLE001 - must not crash the task
                if attempt == 0 and _is_transient_browser_error(exc):
                    continue
                return _json.dumps(
                    {
                        "status": "error",
                        "answer": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        snaps = [
            s.get("html")
            for s in (out.get("per_step") or [])
            if isinstance(s, dict) and s.get("html")
        ]
        if snaps:
            _STATE.setdefault("snapshots", []).extend(snaps)
        answer = out.get("answer")
        return _json.dumps(
            {
                "status": "success" if out.get("reached_answer") else "failed",
                "answer": answer,
                "final_output": out.get("final_output"),
                "iterations": out.get("iterations"),
                "timed_out": out.get("timed_out", False),
                "impl": "pace-agent-webvoyager",
            },
            ensure_ascii=False,
        )


def _strip_opts(argv: list[str], names: set[str]) -> list[str]:
    """Drop ``--name value`` and ``--name=value`` occurrences for each name in *names*."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        base = a.split("=", 1)[0]
        if base in names:
            if "=" not in a and i + 1 < len(argv):
                i += 2
            else:
                i += 1
            continue
        out.append(a)
        i += 1
    return out


def _reject_cli_workers(argv: list[str]) -> None:
    """Worker count is config-only (``parallel`` / ``workers``); refuse the old CLI flags."""
    for a in argv:
        if a in ("--workers", "-j") or a.startswith("--workers="):
            raise SystemExit(
                "--workers / -j is no longer accepted on the CLI. "
                'Set "parallel" (or "workers") in the PACE Agent --config JSON instead '
                "(see packages/pace_agent/configs/gaia_run.example.json)."
            )


# Explicit task ids from ``gaia.task_ids`` in --config (order preserved). None means no filter.
_CONFIG_TASK_IDS: list[str] | None = None


def _argv_has(argv: list[str], *flags: str) -> bool:
    names = set(flags)
    for a in argv:
        if a in names or any(
            a.startswith(f"{n}=") for n in names if n.startswith("--")
        ):
            return True
    return False


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_refs(obj, missing: set[str]):
    """Substitute ``${VAR}`` references in every string of a loaded config.

    Lets a checked-in config name the endpoint/model/token it needs without carrying the
    values (or a secret) itself. Unset names are collected rather than left in place, so the
    caller can fail with the variable name instead of letting ``${VAR}`` reach the gateway.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_refs(v, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_refs(v, missing) for v in obj]
    if isinstance(obj, str):

        def _sub(m):
            name = m.group(1)
            val = os.getenv(name)
            if val is None or val == "":
                missing.add(name)
                return m.group(0)
            return val

        return _ENV_REF.sub(_sub, obj)
    return obj


def _extract_pace_agent_config(argv: list[str]) -> tuple[dict | None, list[str]]:
    """Pull PACE Agent ``--config PATH`` (run knobs) out of *argv*; not a GAIA task JSON."""
    out: list[str] = []
    path: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--config", "--pace-agent-config"):
            if i + 1 < len(argv):
                path = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if a.startswith("--config=") or a.startswith("--pace-agent-config="):
            path = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if not path:
        return None, out
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config must be a JSON object: {path}")
    missing: set[str] = set()
    cfg = _expand_env_refs(cfg, missing)
    if missing:
        names = ", ".join(sorted(missing))
        raise SystemExit(
            f"{path} references environment variable(s) that are unset or empty: {names}\n"
            "Export them, or replace the ${...} reference in the config with a literal value."
        )
    return cfg, out


def _apply_pace_agent_config(
    cfg: dict | None, argv: list[str]
) -> tuple[list[str], int]:
    """Fold PACE Agent run-config knobs into argv/env. Explicit CLI flags win (except workers).

    Worker count comes only from config (``parallel`` / ``workers``, default 1). Task
    selection may use ``gaia.task_id`` (one id) or ``gaia.task_ids`` (ordered list).
    """
    global _CONFIG_TASK_IDS
    _CONFIG_TASK_IDS = None
    workers = 1
    if not cfg:
        return argv, workers

    out = list(argv)
    llm = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    # llm_backend is the older PACE Agent shape; prefer ``llm`` when both are present.
    if not llm and isinstance(cfg.get("llm_backend"), dict):
        lb = cfg["llm_backend"]
        base = (lb.get("base_url") or "").rstrip("/")
        llm = {
            "endpoint": base[:-3] if base.endswith("/v1") else lb.get("base_url"),
            "model": lb.get("model"),
            "api_key_env": lb.get("api_key_env"),
        }
    gaia = cfg.get("gaia") if isinstance(cfg.get("gaia"), dict) else {}

    sandbox = gaia.get("sandbox") if isinstance(gaia.get("sandbox"), dict) else None
    if sandbox is None and isinstance(cfg.get("sandbox"), dict):
        sandbox = cfg.get("sandbox")
    if sandbox:
        from nsjail_agent.config import configure_sandbox_paths

        configure_sandbox_paths(overrides=sandbox)

    # Workers: PACE Agent profile uses ``parallel``; GAIA also accepts top-level or gaia.workers.
    for key in ("workers", "parallel"):
        raw = gaia.get(key) if key in gaia else cfg.get(key)
        if raw is not None:
            try:
                workers = max(1, int(raw))
            except (TypeError, ValueError):
                workers = 1
            break

    def _add_flag(flag: str, value: object) -> None:
        if value is None or value == "" or _argv_has(out, flag):
            return
        out.extend([flag, str(value)])

    endpoint = llm.get("endpoint") or llm.get("base_url")
    if endpoint and not _argv_has(out, "--base-url"):
        ep = str(endpoint).rstrip("/")
        if not ep.endswith("/v1"):
            ep = ep + "/v1"
        out.extend(["--base-url", ep])
    _add_flag("--model", llm.get("model") or gaia.get("model"))
    if not _argv_has(out, "--api-key"):
        key_env = llm.get("api_key_env")
        if key_env and os.getenv(str(key_env)):
            out.extend(["--api-key", os.environ[str(key_env)]])
        elif llm.get("api_key"):
            out.extend(["--api-key", str(llm["api_key"])])

    engine = gaia.get("engine") or cfg.get("engine")
    if engine and not _argv_has(out, "--engine"):
        out.extend(["--engine", str(engine)])

    if gaia.get("huggingface") and not _argv_has(out, "--huggingface"):
        out.append("--huggingface")
    if gaia.get("level") is not None and not _argv_has(out, "--level"):
        out.extend(["--level", str(gaia["level"])])
    if gaia.get("split") and not _argv_has(out, "--split"):
        out.extend(["--split", str(gaia["split"])])
    # When an explicit task id list is set, skip max_tasks unless the user put it on the CLI.
    explicit_ids = _normalize_task_ids(gaia.get("task_ids"), gaia.get("task_id"))
    if (
        gaia.get("max_tasks") is not None
        and not explicit_ids
        and not _argv_has(out, "--max-tasks")
    ):
        out.extend(["--max-tasks", str(gaia["max_tasks"])])
    if explicit_ids and not _argv_has(out, "--task-id"):
        if len(explicit_ids) == 1:
            out.extend(["--task-id", explicit_ids[0]])
        else:
            _CONFIG_TASK_IDS = explicit_ids
    if gaia.get("hf_config") and not _argv_has(out, "--hf-config"):
        out.extend(["--hf-config", str(gaia["hf_config"])])
    if (gaia.get("no_tools") or cfg.get("no_tools")) and not _argv_has(
        out, "--no-tools"
    ):
        out.append("--no-tools")

    out_dir = cfg.get("out") or gaia.get("out")
    if out_dir and not os.getenv("PACE_AGENT_GAIA_OUT"):
        os.environ["PACE_AGENT_GAIA_OUT"] = str(out_dir)

    if gaia.get("skip_preflight") or cfg.get("skip_preflight"):
        os.environ.setdefault("GAIA_SKIP_PREFLIGHT", "1")
        os.environ.setdefault("PACE_AGENT_SKIP_PREFLIGHT", "1")

    # Optional positional task file from config (local samples).
    task_file = gaia.get("task_file") or cfg.get("task_file")
    if task_file and not any(not a.startswith("-") for a in out):
        out.insert(0, str(task_file))

    return out, workers


def _normalize_task_ids(task_ids, task_id) -> list[str]:
    """Normalize ``gaia.task_ids`` / ``gaia.task_id`` to an ordered unique-preserving list."""
    ids: list[str] = []
    if isinstance(task_ids, str) and task_ids.strip():
        ids.append(task_ids.strip())
    elif isinstance(task_ids, list):
        for item in task_ids:
            if item is None or item == "":
                continue
            ids.append(str(item))
    if task_id is not None and task_id != "":
        tid = str(task_id)
        if tid not in ids:
            ids.append(tid)
    return ids


def _install_config_task_filter(gcli) -> None:
    """Restrict gaia's task selector to ``gaia.task_ids`` from the PACE Agent config (if set)."""
    if not _CONFIG_TASK_IDS:
        return
    wanted_order = list(_CONFIG_TASK_IDS)
    wanted = set(wanted_order)
    original = gcli._select_tasks

    def _filtered(all_tasks, task_id, max_tasks, source=None):
        # Config owns the id list; ignore a conflicting CLI --task-id.
        selected = original(all_tasks, None, None, source=source)
        by_id = {t.task_id: t for t in selected if t.task_id in wanted}
        ordered = [by_id[i] for i in wanted_order if i in by_id]
        missing = [i for i in wanted_order if i not in by_id]
        if missing:
            print(
                f"WARNING: config task_id(s) not found in the loaded set: {', '.join(missing)}",
                file=sys.stderr,
            )
        return ordered[:max_tasks] if max_tasks is not None else ordered

    gcli._select_tasks = _filtered


def _list_task_ids(gcli, args) -> list[str]:
    """Resolve the ordered task ids the run would execute, using gaia's own loaders and
    selection so ``--level/--split/--source/--max-tasks/--task-id`` behave identically to the
    sequential path. Kept read-only (no client/adapter construction)."""
    use_hf = args.huggingface or args.level != "all"
    if use_hf:
        level = None if args.level == "all" else int(args.level)
        all_tasks = gcli.load_gaia_from_hf(
            level=level,
            split=args.split,
            config=args.hf_config,
            token=args.hf_token,
        )
    elif args.task_file is not None:
        all_tasks = gcli.load_tasks(args.task_file)
    else:
        raise SystemExit("Provide a task_file or use --huggingface/--level.")
    tasks = gcli._select_tasks(
        all_tasks, args.task_id, args.max_tasks, source=args.source
    )
    return [t.task_id for t in tasks]


def _worker_loop(
    worker_index: int,
    task_q,
    result_q,
    argv_base: list[str],
    pace_agent_run_dir: str,
    shard_dir: str,
) -> None:
    """One PERSISTENT worker process. Loads the dataset and builds its own
    ``LLMClient``/browser/sandbox/``_STATE`` exactly ONCE (via gaia's ``main``), then drains
    the shared *task_q*, running whichever tasks it pulls until the queue is empty. Because
    every worker keeps pulling the next task the moment it frees up, load is balanced
    dynamically — a worker that draws cheap tasks keeps going instead of sitting idle while
    another chews through an expensive one (the old static round-robin shards caused exactly
    that straggler skew).

    The queue is handed to gaia's own ``for task in tasks`` loop through the
    :data:`_WORKER_TASK_QUEUE`-driven selector installed in :func:`main`. Per-task trajectories
    land in the shared ``pace_agent_run_dir`` (keyed by task id); this worker's gaia answers land
    under its own ``shard_dir``. Results are posted back on *result_q* as a dict with keys
    ``worker``, ``gaia_by_id``, ``pace_agent_by_id``, ``fatal``, ``setup_ok``, ``ran``. A
    ``worker.log`` (with any fatal traceback) is also written under ``shard_dir`` so a crash is
    diagnosable from disk. Module-level so it is picklable for ``spawn``.
    """
    global _WORKER_TASK_QUEUE
    import glob as _glob
    import json as _json

    _WORKER_TASK_QUEUE = task_q
    # Shared run dir (each task writes <dir>/<task_id>); parent already preflighted once;
    # GAIA_WORKER suppresses the per-worker run-level aggregate so workers don't clobber each
    # other's root artifacts. Selection is queue-driven now, so no GAIA_TASK_IDS.
    os.environ["PACE_AGENT_GAIA_RUN_DIR"] = pace_agent_run_dir
    os.environ["GAIA_SKIP_PREFLIGHT"] = "1"
    os.environ["GAIA_WORKER"] = "1"
    os.environ.pop("GAIA_TASK_IDS", None)

    pid = os.getpid()
    t0 = time.time()
    print(
        f"[worker {worker_index}] START pid={pid} at {time.strftime('%H:%M:%S')}",
        flush=True,
    )

    argv = list(argv_base) + ["--output-dir", shard_dir]
    fatal = None
    fatal_tb = ""
    try:
        main(argv)
    except Exception as exc:  # noqa: BLE001 - never let one worker kill the run
        import traceback as _tb

        fatal = f"{type(exc).__name__}: {exc}"
        fatal_tb = _tb.format_exc()
    # setup_ok: True once this worker built its PACE Agent client (i.e. got past construction and
    # into gaia's task loop). Lets the parent distinguish a crash-before-first-task (dead
    # gateway/model, bad flag, browser/sandbox init) from a mid-task failure.
    setup_ok = _STATE.get("client") is not None

    # gaia writes one results.jsonl (one line per task it ran) under <shard_dir>/<ts>/.
    gaia_by_id: dict[str, dict] = {}
    hits = sorted(_glob.glob(os.path.join(shard_dir, "*", "results.jsonl")))
    if hits:
        try:
            with open(hits[-1], encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _json.loads(line)
                    tid = rec.get("task_id")
                    if tid:
                        gaia_by_id[tid] = rec
        except Exception:  # noqa: BLE001
            pass

    pace_agent_by_id: dict[str, dict] = {}
    for rec in _STATE.get("records") or []:
        tid = rec.get("task")
        if tid:
            pace_agent_by_id[tid] = rec

    dt = time.time() - t0
    # Persist this worker's outcome (especially the fatal + traceback) so a crash is
    # diagnosable from disk after the fact — the console output is otherwise the only record
    # and is easily lost. Best-effort: logging must never mask the original failure.
    try:
        os.makedirs(shard_dir, exist_ok=True)
        with open(os.path.join(shard_dir, "worker.log"), "w", encoding="utf-8") as fh:
            fh.write(
                f"worker {worker_index} pid={pid} setup_ok={setup_ok} "
                f"ran={len(gaia_by_id)} dt={dt:.1f}s\n"
            )
            if fatal:
                fh.write(f"\nFATAL: {fatal}\n\n{fatal_tb}\n")
    except Exception:  # noqa: BLE001
        pass
    print(
        f"[worker {worker_index}] DONE pid={pid} at {time.strftime('%H:%M:%S')} "
        f"— ran {len(gaia_by_id)} tasks in {dt:.1f}s"
        + ("" if setup_ok else " [setup did not complete]")
        + (f" (fatal: {fatal})" if fatal else ""),
        flush=True,
    )
    result_q.put(
        {
            "worker": worker_index,
            "gaia_by_id": gaia_by_id,
            "pace_agent_by_id": pace_agent_by_id,
            "fatal": fatal,
            "setup_ok": setup_ok,
            "ran": len(gaia_by_id),
        }
    )


def _run_parallel(gcli, args, argv: list[str], workers: int) -> int:
    """Run the selected tasks across *workers* persistent processes fed by ONE shared work
    queue, then merge their output into one run.

    Uses a GAIA-specific dynamic work queue (not ``pace_agent.parallel``): profile's pool pins
    short-lived workers to CPU blocks for independent profile tasks, while GAIA keeps each
    worker process warm (browser/sandbox/client) and pulls the next task id as it frees up.
    ``args`` is the same Namespace already parsed for the serial path so flags are not
    re-interpreted.

    Each worker loads the dataset and builds its own client/browser/sandbox/``_STATE`` once,
    then pulls the next task off the shared queue. One up-front gateway preflight fails the
    whole run fast (optional via ``GAIA_SKIP_PREFLIGHT=1`` / config ``skip_preflight``);
    workers skip theirs. Per-task trajectories land in one shared ``pace-agent-gaia/<ts>/`` dir,
    and gaia's answers/scores are merged into a single ``results.jsonl`` + ``summary.json``.
    """
    import multiprocessing as _mp
    from queue import Empty as _QueueEmpty

    task_ids = _list_task_ids(gcli, args)
    if not task_ids:
        print("No tasks selected.", file=sys.stderr)
        return 2
    if not args.api_key:
        # Local OpenAI-compatible servers often accept any/empty key; warn rather than abort.
        print(
            "WARNING: no --api-key / OPENAI_API_KEY / LLM_GATEWAY_KEY; continuing "
            "(required for authenticated gateways).",
            file=sys.stderr,
        )

    n_workers = max(1, min(int(workers), len(task_ids)))
    print(
        f"Planner engine: {args.engine} | workers: {n_workers} | tasks: {len(task_ids)} "
        f"| dynamic work queue (each worker pulls the next task as it frees up)"
    )

    # Single up-front preflight (workers skip theirs) so a dead gateway aborts before fan-out.
    if (
        os.getenv("GAIA_SKIP_PREFLIGHT") != "1"
        and os.getenv("PACE_AGENT_SKIP_PREFLIGHT") != "1"
    ):
        client = gcli._build_client(args.api_key or "EMPTY", args.base_url)
        _preflight_gateway(client, args.model)
        del client

    ts = time.strftime("%Y%m%d_%H%M%S")
    pace_agent_run_dir = str(
        Path(os.getenv("PACE_AGENT_GAIA_OUT", "pace-agent-gaia")).resolve() / ts
    )
    gaia_run_dir = args.output_dir.resolve() / ts
    shard_root = gaia_run_dir / "_shards"
    Path(pace_agent_run_dir).mkdir(parents=True, exist_ok=True)
    gaia_run_dir.mkdir(parents=True, exist_ok=True)

    # Selection is queue-driven and each worker owns its own --output-dir, so strip any
    # run-wide selection/output flags from the base argv handed to them.
    argv_base = _strip_opts(
        argv,
        {
            "--task-id",
            "--output-dir",
            "--max-tasks",
            "--config",
            "--pace-agent-config",
        },
    )

    # Shared queue: every task id, then one None sentinel per worker so each worker's queue
    # drain terminates exactly once.
    ctx = _mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    for tid in task_ids:
        task_q.put(tid)
    for _ in range(n_workers):
        task_q.put(None)

    procs = []
    for idx in range(n_workers):
        shard_dir = str(shard_root / f"worker{idx}")
        p = ctx.Process(
            target=_worker_loop,
            args=(idx, task_q, result_q, argv_base, pace_agent_run_dir, shard_dir),
        )
        p.start()
        procs.append(p)

    # Drain results BEFORE joining (a large payload left on the queue would otherwise deadlock
    # the worker's feeder thread at join). Fall through if a worker dies without reporting.
    results_by_id: dict[str, dict] = {}
    pace_agent_recs_by_id: dict[str, dict] = {}
    worker_fatals: list[dict] = []
    reported: set[int] = set()
    while len(reported) < n_workers:
        try:
            msg = result_q.get(timeout=1.0)
        except _QueueEmpty:
            if all(not p.is_alive() for p in procs) and result_q.empty():
                break  # every worker exited; stop waiting for the missing report(s)
            continue
        widx = msg["worker"]
        reported.add(widx)
        results_by_id.update(msg["gaia_by_id"])
        pace_agent_recs_by_id.update(msg["pace_agent_by_id"])
        if msg.get("fatal"):
            worker_fatals.append(
                {
                    "worker": widx,
                    "fatal": msg["fatal"],
                    "setup_ok": bool(msg.get("setup_ok")),
                }
            )
        note = f" (fatal: {msg['fatal']})" if msg.get("fatal") else ""
        if not msg.get("setup_ok", True):
            note += " [setup did not complete]"
        print(
            f"[{len(reported)}/{n_workers} workers reported] worker {widx}: "
            f"{msg.get('ran', 0)} task results merged{note}"
        )

    for p in procs:
        p.join()

    # Any task with no record (a worker crashed before emitting it, or was never pulled)
    # becomes an error so the merged run still accounts for the full task set. If workers died,
    # surface *why* on those tasks: dynamic queueing means we cannot map a specific lost task to
    # a specific crashed worker, so we attach the run-wide fatal summary and point at the
    # per-worker logs under _shards/.
    fatal_note = ""
    if worker_fatals:
        joined = "; ".join(
            f"worker{f['worker']}: {f['fatal']}"
            + ("" if f["setup_ok"] else " [setup failed]")
            for f in worker_fatals
        )
        fatal_note = f" | worker fatals: {joined} (see {shard_root}/worker*/worker.log)"
    for tid in task_ids:
        if tid not in results_by_id:
            results_by_id[tid] = {
                "task_id": tid,
                "status": "error",
                "error": "no result produced (worker crashed or task unclaimed)"
                + fatal_note,
            }

    # Merge in the original task order.
    records = [results_by_id[tid] for tid in task_ids if tid in results_by_id]
    result_path = gaia_run_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = gcli._summary(records)
    (gaia_run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    completed = sum(1 for r in records if r.get("status") == "completed")
    # Fail loudly: a wipeout (nothing completed) or ANY worker fatal must not look like success
    # to a caller/CI. Exit 0 only when every worker finished cleanly AND at least one task
    # completed. (The old path returned 0 unconditionally, so a total 0/4 crash reported OK.)
    exit_code = 0 if (completed > 0 and not worker_fatals) else 1

    pace_agent_recs = [
        pace_agent_recs_by_id[tid] for tid in task_ids if tid in pace_agent_recs_by_id
    ]
    _write_run_artifacts(
        Path(pace_agent_run_dir), args, exit_code, records=pace_agent_recs
    )
    # Persist the worker roster/fatals next to the trajectories so a failed run is diagnosable
    # from disk (the console 'fatal:' lines are otherwise the only record, and easily lost).
    try:
        (Path(pace_agent_run_dir) / "workers.json").write_text(
            json.dumps(
                {
                    "n_workers": n_workers,
                    "n_reported": len(reported),
                    "tasks": len(task_ids),
                    "completed": completed,
                    "exit_code": exit_code,
                    "worker_fatals": worker_fatals,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass

    print(f"Results: {result_path}")
    if summary.get("scored"):
        print(f"Local exact accuracy: {summary['correct']}/{summary['scored']}")
        for level, bucket in summary.get("by_level", {}).items():
            if bucket.get("scored"):
                print(f"  level {level}: {bucket['correct']}/{bucket['scored']}")
    print(f"PACE Agent run dir: {pace_agent_run_dir}")
    if worker_fatals:
        print(
            f"WARNING: {len(worker_fatals)}/{n_workers} worker(s) crashed — "
            f"{completed}/{len(task_ids)} tasks completed. See "
            f"{shard_root}/worker*/worker.log and {Path(pace_agent_run_dir) / 'workers.json'}.",
            file=sys.stderr,
        )
    elif completed == 0:
        print(
            f"WARNING: 0/{len(task_ids)} tasks completed (no worker fatals reported); "
            "workers may have exited before pulling any task.",
            file=sys.stderr,
        )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    from pace_agent.gaia_planner import GaiaPlanner, cli as gcli

    # Default this PACE Agent entry point to the LangGraph planner engine (the loop engine is
    # still available via an explicit `--engine loop` or config ``gaia.engine``).
    argv_list = list(sys.argv[1:] if argv is None else argv)
    _reject_cli_workers(argv_list)
    cfg, argv_list = _extract_pace_agent_config(argv_list)
    # Worker count and task selection knobs come from --config (``parallel`` / ``workers``,
    # ``gaia.task_id`` / ``gaia.task_ids``). Remaining CLI flags still override config.
    argv_list, workers = _apply_pace_agent_config(cfg, argv_list)
    if not _argv_has(argv_list, "--engine"):
        argv_list += ["--engine", "langgraph"]
    # ``--no-tools``: gaia's own CLI accepts this flag and, when set, closes the book at the
    # PLANNER level — it advertises no tools, uses CLOSED_BOOK_PROMPT, AND inlines each
    # attachment's extracted text into the prompt. Pass the flag through to gaia. Also
    # propagate to spawned workers via the environment and turn off the langgraph
    # verify-by-execution gate since there is no sandbox to verify with.
    no_tools = "--no-tools" in argv_list or os.getenv("GAIA_NO_TOOLS") == "1"
    if no_tools:
        os.environ["GAIA_NO_TOOLS"] = "1"
        os.environ.setdefault("GAIA_ENFORCE_VERIFY", "0")
        if "--no-tools" not in argv_list:
            argv_list.append("--no-tools")
    argv = argv_list

    # Parse once and share with the parallel fan-out so serial/parallel cannot diverge on flags.
    args = gcli.parse_args(argv)
    # Apply before the parallel branch so ``_list_task_ids`` sees the same filter as serial.
    _install_config_task_filter(gcli)
    if workers > 1 and os.getenv("GAIA_WORKER") != "1":
        return _run_parallel(gcli, args, argv, workers)

    # A worker process is handed a fixed run dir via PACE_AGENT_GAIA_RUN_DIR so every worker writes
    # its per-task summary (keyed by task id) into the one shared, parent-owned run dir.
    worker_run_dir = os.getenv("PACE_AGENT_GAIA_RUN_DIR")
    if worker_run_dir:
        pace_agent_out = Path(worker_run_dir)
    else:
        pace_agent_out = Path(
            os.getenv("PACE_AGENT_GAIA_OUT", "pace-agent-gaia")
        ).resolve() / time.strftime("%Y%m%d_%H%M%S")
    pace_agent_out.mkdir(parents=True, exist_ok=True)
    print(f"Planner engine: {args.engine}")

    _STATE.clear()

    original_build_client = gcli._build_client
    original_build_local_client = gcli._build_local_client
    original_run_task = gcli._run_task
    original_select = gcli._select_tasks
    original_webvoyager = getattr(gcli, "WebVoyagerAdapter", None)

    original_call_tool = GaiaPlanner._call_tool
    GaiaPlanner._call_tool = _attributed_call_tool(original_call_tool)
    # browse_web is served by PACE Agent's in-process WebVoyager by default. Set
    # GAIA_USE_EXTERNAL_WEBVOYAGER=1 to restore the original external run.py subprocess.
    use_external_wv = os.getenv("GAIA_USE_EXTERNAL_WEBVOYAGER") == "1"

    if _WORKER_TASK_QUEUE is not None:
        _task_queue = _WORKER_TASK_QUEUE

        def _select_from_queue(all_tasks, task_id, max_tasks, source=None):
            by_id = {t.task_id: t for t in all_tasks}

            def _drain():
                while True:
                    try:
                        tid = _task_queue.get()
                    except (EOFError, OSError):
                        return
                    if tid is None:
                        return
                    task = by_id.get(tid)
                    if task is not None:
                        yield task

            return _drain()

        gcli._select_tasks = _select_from_queue

    def _build_client(api_key: str, base_url: str):
        # Planner uses gaia's native OpenAI-compatible client, wrapped only to capture
        # response.usage. Vision/audio share this client when --local-base-url is unset;
        # when set they use a separate local client (see _build_local_client below) that
        # shares the same tracker so Radeon/local multimodal tokens still show up.
        tracker = _STATE.setdefault("usage", _UsageTracker())
        client = _UsageTrackingClient(
            original_build_client(api_key, base_url), tracker, backend="gateway"
        )
        _STATE["client"] = client
        # In-process WebVoyager still needs a PACE Agent LLMClient (separate from the planner).
        if not use_external_wv:
            _STATE["llm"] = _make_llm_client(
                model=args.model, base_url=base_url, api_key=api_key
            )
        if (
            os.getenv("GAIA_SKIP_PREFLIGHT") != "1"
            and os.getenv("PACE_AGENT_SKIP_PREFLIGHT") != "1"
            and not _STATE.get("preflight_done")
        ):
            _preflight_gateway(client, args.model)
            # Drop the preflight call from the first task's token totals.
            tracker.events.clear()
            tracker.tool_marks.clear()
            _STATE["preflight_done"] = True
        return client

    def _build_local_client(api_key: str, base_url: str):
        # Same tracker as the planner client so analyze_video/image/audio calls against a
        # local (e.g. Radeon vLLM) endpoint are attributed under tool:* in the breakdown.
        tracker = _STATE.setdefault("usage", _UsageTracker())
        return _UsageTrackingClient(
            original_build_local_client(api_key, base_url), tracker, backend="local"
        )

    def _run_task(planner, task):
        rec: dict[str, object] = {
            "task": task.task_id,
            "query": task.question,
            "answer": None,
            "completed": False,
            "error": None,
            "nodes_traversed": 0,
            "tool_calls": 0,
        }
        record = None
        _STATE["snapshots"] = []
        client = _STATE.get("client")
        usage_mark = (
            client.usage.mark() if isinstance(client, _UsageTrackingClient) else 0
        )
        t0 = time.perf_counter()
        try:
            record = original_run_task(planner, task)
            return record
        except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
            rec["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            wall = time.perf_counter() - t0
            per_node: dict = {}
            if isinstance(record, dict):
                rec["answer"] = record.get("answer")
                rec["completed"] = record.get("status") == "completed" and bool(
                    record.get("answer")
                )
                if record.get("error"):
                    rec["error"] = record.get("error")
                rec["tool_calls"] = len(record.get("tool_events") or [])
                fail_total, fail_by_tool = _count_tool_failures(
                    record.get("tool_events")
                )
                rec["tool_failures"] = fail_total
                rec["tool_failures_by_tool"] = fail_by_tool
                rec["degraded"] = fail_total > 0
                per_node = record.get("per_node") or {}
                if isinstance(per_node, dict):
                    rec["nodes_traversed"] = sum(
                        int(v.get("count", 0))
                        for v in per_node.values()
                        if isinstance(v, dict)
                    )
                else:
                    per_node = {}
            if isinstance(client, _UsageTrackingClient):
                rec["metrics"] = client.usage.metrics_since(usage_mark, wall_s=wall)
            else:
                rec["metrics"] = {
                    "llm_calls": None,
                    "api": None,
                    "context_tokens_total": None,
                    "context_tokens_max": None,
                    "cached_tokens_total": None,
                    "output_tokens_total": None,
                    "reasoning_tokens_total": None,
                    "work_tokens_total": None,
                    "e2e_median": wall,
                    "by_component": {},
                }
            _print_task_token_stats(rec)
            _STATE.setdefault("records", []).append(rec)
            try:
                from pace_agent.rundir import RunDir
                from pace_agent.stats import summarize

                meta = {
                    "agent": "gaia_planner",
                    "engine": args.engine,
                    "task": task.task_id,
                    "question": task.question,
                    "answer": rec.get("answer"),
                    "completed": rec.get("completed"),
                    "degraded": rec.get("degraded"),
                    "tool_failures": rec.get("tool_failures"),
                    "tool_failures_by_tool": rec.get("tool_failures_by_tool"),
                }
                snaps = _STATE.get("snapshots") or []
                if snaps:
                    meta["replay_mode"] = "browser-snapshot"
                    meta["n_snapshots"] = len(snaps)
                metrics = rec.get("metrics") or {}
                task_dir = pace_agent_out / task.task_id
                rd = RunDir(task_dir)
                rd.write_result(
                    {
                        "cmd": "profile",
                        "agent": "gaia_planner",
                        "impl": "naive",
                        "query": task.question,
                        "e2e": summarize([wall]),
                        "per_node": per_node if isinstance(per_node, dict) else {},
                        "llm_calls": metrics.get("llm_calls"),
                        "context_tokens_total": metrics.get("context_tokens_total"),
                        "output_tokens_total": metrics.get("output_tokens_total"),
                        "reasoning_tokens_total": metrics.get("reasoning_tokens_total"),
                        "tokens_by_component": metrics.get("by_component") or {},
                        **meta,
                    }
                )
                if snaps:
                    from pace_agent.replay.trajectory import Trajectory

                    llm_wall = 0.0
                    if isinstance(client, _UsageTrackingClient):
                        llm_wall = sum(
                            e["wall"] for e in client.usage.events[usage_mark:]
                        )
                    Trajectory(
                        e2e=wall,
                        llm_time=llm_wall,
                        meta={
                            **meta,
                            "replay_mode": "browser-snapshot",
                            "snapshots": list(snaps),
                        },
                    ).save(str(task_dir))
            except Exception:  # noqa: BLE001 - persistence must never fail the task
                pass

    gcli._build_client = _build_client
    gcli._build_local_client = _build_local_client
    gcli._run_task = _run_task
    if not use_external_wv:
        gcli.WebVoyagerAdapter = _PaceAgentWebVoyagerAdapter
    status = None
    try:
        status = gcli.main(argv)
    finally:
        gcli._build_client = original_build_client
        gcli._build_local_client = original_build_local_client
        gcli._run_task = original_run_task
        gcli._select_tasks = original_select
        GaiaPlanner._call_tool = original_call_tool
        if original_webvoyager is not None:
            gcli.WebVoyagerAdapter = original_webvoyager
        if os.getenv("GAIA_WORKER") != "1":
            _write_run_artifacts(pace_agent_out, args, status)

    print(f"PACE Agent run dir: {pace_agent_out}")
    return status


def _merge_component_metrics(per_query: list[dict]) -> dict:
    """Sum each task's planner/tool token breakdown into one run-level breakdown."""
    merged: list[dict] = []
    for q in per_query:
        for name, m in (q.get("tokens_by_component") or {}).items():
            if not isinstance(m, dict):
                continue
            merged.append(
                {
                    "component": name,
                    "prompt": m.get("context_tokens"),
                    "completion": m.get("output_tokens"),
                    "cached": m.get("cached_tokens"),
                    "reasoning": m.get("reasoning_tokens"),
                    "calls": m.get("llm_calls") or 0,
                    "backend": m.get("backend"),
                }
            )
    grouped: dict[str, dict] = {}
    for row in merged:
        slot = grouped.setdefault(
            row["component"],
            {
                "llm_calls": 0,
                "context_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "reasoning_tokens": None,
                "backends": set(),
            },
        )
        slot["llm_calls"] += int(row["calls"] or 0)
        if row.get("backend"):
            slot["backends"].add(row["backend"])
        for key, field in (
            ("context_tokens", "prompt"),
            ("output_tokens", "completion"),
            ("cached_tokens", "cached"),
            ("reasoning_tokens", "reasoning"),
        ):
            if row[field] is not None:
                slot[key] = (slot[key] or 0) + row[field]

    def _rank(item):
        name, m = item
        total = (m["context_tokens"] or 0) + (m["output_tokens"] or 0)
        return (name != _PLANNER_COMPONENT, -total, name)

    out: dict[str, dict] = {}
    for name, m in sorted(grouped.items(), key=_rank):
        backends = sorted(m.pop("backends"))
        row = dict(m)
        if backends:
            row["backend"] = backends[0] if len(backends) == 1 else "mixed"
        out[name] = row
    return out


def _write_run_artifacts(pace_agent_out: Path, args, status, records=None) -> None:
    """Write the run-level ``result.json`` + ``env.json`` via :mod:`pace_agent.harness`.

    ``records`` defaults to the sequential run's ``_STATE`` accumulator; the parallel path
    passes the per-task recs it collected back from the worker processes explicitly.

    Best-effort: a persistence error here must never fail the run.
    """
    try:
        records = (_STATE.get("records") if records is None else records) or []
        per_query = []
        for rec in records:
            metrics = rec.get("metrics") or {}
            per_query.append(
                {
                    "query": rec.get("query"),
                    "answer": rec.get("answer"),
                    "completed": rec.get("completed"),
                    # gaia has no per-node "steps"; use model turns, as browser agents do
                    "steps": metrics.get("llm_calls"),
                    "error": rec.get("error"),
                    "llm_calls": metrics.get("llm_calls"),
                    "api": metrics.get("api"),
                    "context_tokens_total": metrics.get("context_tokens_total"),
                    "context_tokens_max": metrics.get("context_tokens_max"),
                    "cached_tokens_total": metrics.get("cached_tokens_total"),
                    "output_tokens_total": metrics.get("output_tokens_total"),
                    "reasoning_tokens_total": metrics.get("reasoning_tokens_total"),
                    "work_tokens_total": metrics.get("work_tokens_total"),
                    "tokens_by_component": metrics.get("by_component") or {},
                    "nodes_traversed": rec.get("nodes_traversed"),
                    "tool_calls": rec.get("tool_calls"),
                    "tool_failures": rec.get("tool_failures"),
                    "tool_failures_by_tool": rec.get("tool_failures_by_tool"),
                    "degraded": rec.get("degraded"),
                    "e2e_median": metrics.get("e2e_median"),
                }
            )

        write_profile_like_result(
            pace_agent_out,
            cmd="gaia_run",
            agent="pace_agent.examples.run_gaia",
            engine=getattr(args, "engine", None),
            model=getattr(args, "model", None),
            status=status,
            per_query=per_query,
            aggregate_extra={
                "tokens_by_component": _merge_component_metrics(per_query),
            },
        )
    except Exception:  # noqa: BLE001 - artifact writing must never break the run
        pass


if __name__ == "__main__":
    raise SystemExit(main())
