# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
from dataclasses import dataclass, field

from pace_agent import score

# Result/trajectory schema version. Stamped into result.json and trajectories so
# consumers reject unknown major versions instead of mis-reading renamed fields.
SCHEMA_VERSION = 1


@dataclass
class RunConfig:
    workload: str = ""
    impl: str = "pace"
    plugins: list = field(default_factory=list)
    llm: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)  # [{query, start_url?, max_iter?}, ...]
    max_iter: int = 80
    max_llm_calls: int = 200
    num_runs: int = 10
    warmup: int = 1
    pinning: dict = field(default_factory=dict)
    telemetry: str = "off"
    out: str | None = None
    parallel: int = 1  # multi-task run: run this many tasks concurrently (process pool)
    cores_per_worker: int = 2  # CPU cores pinned to each parallel worker (no overlap)
    # {gold_key} — grade answers against the expected answer carried by each task, reporting
    # both a strict and a loose verdict. Absent (or empty) means the suite is ungraded.
    scoring: dict = field(default_factory=dict)
    # per-task settings applied to every task that does not set them itself, so a knob like
    # capture_snapshots can be flipped for a whole suite without editing every task
    task_defaults: dict = field(default_factory=dict)
    # Closed-book mode: the LLMClient advertises no tools, so agents' tool schemas are
    # dropped and the model must answer from its own reasoning. Set at the top level or as
    # `no_tools` inside the `llm`/`llm_backend` block; either place flows to the client.
    no_tools: bool = False


@dataclass
class BenchConfig:
    tools: dict = field(default_factory=dict)


def _tasks_from_graphs(d):
    tasks = []
    for g in d.get("graphs", []):
        kw = dict(g.get("kwargs", {}))
        kw.setdefault("workload", g.get("spec"))
        tasks.append(kw)
    return tasks


def load_config(path):
    with open(path) as f:
        d = json.load(f)
    if "tools" in d and isinstance(d["tools"], dict):
        return BenchConfig(tools=d["tools"])
    d = dict(d)
    # back-compat: old {graphs:[{spec,kwargs}], llm_backend:{...}} shape. Each mapping is
    # independent — a config may carry llm_backend without graphs (e.g. profiling a single
    # agent), so mapping it must NOT be gated on graphs (else llm silently falls back to
    # localhost:8000).
    if "graphs" in d and "tasks" not in d:
        d["tasks"] = _tasks_from_graphs(d)
    lb = d.get("llm_backend") or {}
    if lb and "llm" not in d:
        base_url = lb.get("base_url") or ""
        clean_url = base_url.rstrip("/")
        d["llm"] = {
            "endpoint": clean_url[:-3] if clean_url.endswith("/v1") else base_url,
            "model": lb.get("model", ""),
            "temperature": lb.get("temperature", 0.0),
            "seed": lb.get("seed", 0),
            "max_tokens": lb.get("max_tokens", 1024),
            # gateway auth: key resolved from env at call time (never stored in config)
            "api_key_env": lb.get("api_key_env"),
            "subscription_header": lb.get("subscription_header"),
            "api_key": lb.get("api_key"),
            # "chat" | "responses" pins the API surface; None infers it (/responses first)
            "api": lb.get("api"),
            "vision": lb.get("vision", False),  # backend accepts images (opt-in)
            "image_format": lb.get("image_format", "openai"),  # "anthropic" for Claude
            "reasoning": lb.get("reasoning"),
            "no_tools": lb.get("no_tools", False),  # closed-book mode (opt-in)
        }  # gpt-oss reasoning effort (opt-in)
    if "warmup_runs" in d:
        d["warmup"] = d["warmup_runs"]
    # A top-level `no_tools` is a convenience shortcut; mirror it into the llm block so the
    # LLMClient (built from cfg.llm) honors it regardless of where the user set it.
    if d.get("no_tools") and isinstance(d.get("llm"), dict):
        d["llm"].setdefault("no_tools", True)
    known = RunConfig.__dataclass_fields__
    cfg = RunConfig(**{k: v for k, v in d.items() if k in known})
    if cfg.task_defaults:
        cfg.tasks = [{**cfg.task_defaults, **t} for t in cfg.tasks]
    # fail on a malformed scoring block now, not 90 tasks later
    score.resolve(cfg.scoring)
    return cfg


def load_bench_config(path):
    with open(path) as f:
        return BenchConfig(tools=json.load(f))
