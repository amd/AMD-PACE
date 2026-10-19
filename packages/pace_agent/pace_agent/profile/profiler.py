# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import importlib
import importlib.util
import os
import sys

from pace_agent import score
from pace_agent.llm.client import LLMClient
from pace_agent.replay import Player, Recorder
from pace_agent.rundir import RunDir
from pace_agent.stats import summarize
from pace_agent.profile.attribution import attribute


def import_agent(path):
    if path.endswith(".py") or os.path.sep in path or os.path.isfile(path):
        abspath = os.path.abspath(path)
        name = "pace_agent_agent_" + os.path.splitext(os.path.basename(abspath))[0]
        spec = importlib.util.spec_from_file_location(name, abspath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(path)


def _llm(cfg):
    return LLMClient.from_config(cfg.llm)


def _context_growth(llm):
    """Per-iteration token counts (one entry per LLM call: prompt / output / cached / reasoning)
    for later analysis. Stored only in this run's result.json."""

    def col(name):
        return list(getattr(llm, name, []) or [])

    prompt, output = col("prompt_tokens"), col("completion_tokens")
    cached, reasoning = col("cached_tokens"), col("reasoning_tokens")

    def at(xs, i):
        return xs[i] if i < len(xs) else None

    return [
        {
            "iter": i + 1,
            "prompt": prompt[i],
            "output": at(output, i),
            "cached": at(cached, i),
            "reasoning": at(reasoning, i),
        }
        for i in range(len(prompt))
    ]


def profile_task(args):
    """Top-level (picklable) worker: profile ONE task into its own dir; return its result dict.
    Used by parallel multi-task paths (multi-query profile, workloads) via pace_agent.parallel.
    """
    import copy

    from pace_agent.rundir import RunDir

    if len(args) == 4:
        agent_path, cfg, task, out_dir = args
        index = total = None
    else:
        agent_path, cfg, task, out_dir, index, total = args
    ci = copy.copy(cfg)
    ci.tasks = [task]
    prefix = f"[profile {index + 1}/{total}]" if index is not None else "[profile]"
    label = str((task or {}).get("query") or "(default task)")[:100]
    print(f"{prefix} starting: {label}", flush=True)
    try:
        profile(agent_path, ci, out_dir)
        result = RunDir(out_dir).read_result()
        elapsed = (result.get("e2e") or {}).get("median", 0)
        graded = (
            f" match=strict:{result['match_strict']} loose:{result['match_loose']}"
            if result.get("scoring")
            else ""
        )
        print(
            f"{prefix} completed: answered={bool(result.get('completed'))} "
            f"steps={result.get('steps')} e2e={elapsed:.1f}s{graded}",
            flush=True,
        )
        return result
    except Exception as e:  # noqa: BLE001 - never abort the batch
        print(f"{prefix} failed: {type(e).__name__}: {e}", flush=True)
        return write_failure(agent_path, cfg, task, out_dir, e)


def write_failure(agent_path, cfg, task, out_dir, exc):
    """Persist a failed task like a completed one — a result row naming the error, plus a trace —
    so the run dir stays readable instead of being one only the terminal explains."""
    from pace_agent.rundir import RunDir

    result = {
        "cmd": "profile",
        "agent": agent_path,
        "query": (task or {}).get("query"),
        "completed": False,
        "steps": 0,
        "answer": None,
        "error": f"{type(exc).__name__}: {exc}"[:200],
        "e2e": {"n": 0},
        "per_node": {},
        "run_metrics": {},
    }
    gold_key = score.resolve(getattr(cfg, "scoring", None))
    if gold_key:
        # a failed task with an expected answer is a miss, not an unscored task
        result.update(score.score(None, (task or {}).get(gold_key)))
    rd = RunDir(out_dir)
    rd.write_result(result)
    rd.write_trace(
        {
            "agent": agent_path,
            "query": result["query"],
            "final_output": None,
            "steps": [],
            "error": result["error"],
        }
    )
    return result


def profile(agent_path, cfg, out_dir):
    from pace_agent import node_counter
    from pace_agent.capture import CaptureContext
    from pace_agent.env import capture_env
    from pace_agent.watchdog import watch

    agent = import_agent(agent_path)
    llm = _llm(cfg)

    task = cfg.tasks[0] if getattr(cfg, "tasks", None) else None
    if task is not None and getattr(cfg, "impl", None):
        task = {**task, "impl": cfg.impl}
    replay_safe = getattr(agent, "REPLAY_SAFE", True)
    graded_key = score.resolve(getattr(cfg, "scoring", None))
    gold_key = graded_key or score.DEFAULT_GOLD_KEY
    gold = (task or {}).get(gold_key) if graded_key else None
    # the agent never sees the expected answer — including when grading is off, so that turning
    # grading on cannot change what the agent was given
    if task is not None and gold_key in task:
        task = {k: v for k, v in task.items() if k != gold_key}
    if not replay_safe and cfg.num_runs > 1:
        print(
            f"warning: {agent_path} is not replay-safe (REPLAY_SAFE=False), so its work cannot be "
            f"re-timed from a tape; num_runs={cfg.num_runs} is reduced to 1 live run",
            file=sys.stderr,
            flush=True,
        )

    ctx = CaptureContext()
    characterization, keep_trajectories = None, False
    with watch(label=f"profile task {str((task or {}).get('query'))[:60]!r}"):
        n_live = max(1, int(getattr(cfg, "characterize_runs", 1) or 1))
        if n_live > 1:
            # characterization: record a representative run rather than whichever went first
            characterization, keep_trajectories = _characterize(
                agent, llm, agent_path, task, cfg, n_live
            )
            traj, ctx = characterization.trajectory, characterization.context
        else:
            traj = Recorder().record(agent, llm, agent_path, task=task, ctx=ctx)
        _stamp_config_identity(traj, cfg)
        per_node = node_counter.timings_of(ctx.node_samples, ctx.node_order)
        player = Player()

        if replay_safe:
            if cfg.warmup:
                player.replay(traj, agent, llm, "naive", cfg.warmup)
            naive = player.replay(
                traj,
                agent,
                llm,
                cfg.impl if cfg.impl in ("naive", "optimized") else "naive",
                cfg.num_runs,
            )
            e2e_samples = naive["e2e"]
            e2e = summarize(e2e_samples)
            per_tool = {k: summarize(v) for k, v in naive["per_tool"].items()}
        else:  # live browser agent — no pickle replay
            e2e_samples = [traj.e2e]
            e2e = summarize(e2e_samples)
            per_tool = {}

    res = traj.meta.get("result")
    node_total = sum(v.get("total_ms", 0) / 1e3 for v in per_node.values())
    observe_ab = None
    if isinstance(res, dict) and res.get("observe_pace_ms"):
        observe_ab = {
            "naive_ms": res["observe_naive_ms"],
            "pace_ms": res["observe_pace_ms"],
            "speedup": res.get("observe_speedup"),
            "parity_all": res.get("observe_parity_all"),
        }
    result = {
        "cmd": "profile",
        "agent": agent_path,
        "impl": getattr(cfg, "impl", "pace"),
        "query": (task or {}).get("query"),
        "e2e": e2e,
        "per_node": per_node,
        "observe_ab": observe_ab,
        "attribution": {
            "total": traj.e2e,
            "model": traj.llm_time,
            "non_model_cpu": max(0.0, node_total - traj.llm_time),
            "unattributed": max(0.0, traj.e2e - node_total),
            "buckets": attribute(traj)["buckets"],
        },
        "steps": (
            res.get("iterations")
            if isinstance(res, dict)
            else traj.run_metrics.get("tool_calls")
        ),
        "completed": (
            res.get("reached_answer") if isinstance(res, dict) else res is not None
        ),
        "answer": (res.get("answer") if isinstance(res, dict) else res),
        "run_metrics": traj.run_metrics,
        "context_growth": _context_growth(llm),
        "stats": {"per_tool": per_tool},
    }
    if characterization is not None:
        # inline, so the result answers what the trajectory was chosen out of on its own
        result["characterization"] = characterization.summary()
    if graded_key:
        result["scoring"] = {"gold_key": graded_key}
        result.update(score.score(result["answer"], gold))
    if cfg.telemetry and cfg.telemetry != "off":
        result["telemetry"] = _counter_pass(cfg, out_dir)

    trace = (
        res.get("per_step")
        if isinstance(res, dict) and res.get("per_step")
        else traj.summary()
    )

    rd = RunDir(out_dir)
    if characterization is not None:
        characterization.save(out_dir, keep_trajectories=keep_trajectories)
    if replay_safe:
        traj.save(out_dir)
    elif isinstance(res, dict) and res.get("per_step"):
        # browser agent: carry the page snapshots in the trajectory (msgpack, no pickle) so
        # `pace-agent optimize` can replay the observe A/B via the browser-snapshot mode
        traj.meta["replay_mode"] = "browser-snapshot"
        traj.meta["snapshots"] = [
            s.get("html") for s in res["per_step"] if s.get("html")
        ]
        traj.save(out_dir)
    if getattr(cfg, "write_env", True):
        rd.write_env(capture_env())
    rd.write_result(result)
    # keep bulky page snapshots ("html"/MHTML) out of trace.json — they live in trajectory.msgpack
    trace_out = (
        [{k: v for k, v in s.items() if k != "html"} for s in trace]
        if isinstance(trace, list)
        else trace
    )
    rd.write_trace(
        {
            "agent": agent_path,
            "query": (task or {}).get("query"),
            "final_output": result["answer"],
            "steps": trace_out,
        }
    )
    # no auto-report: report.md is produced only by `pace-agent report <dir>`
    return out_dir


def _characterize(agent, llm, agent_path, task, cfg, n_live):
    """Choose which live run to record. Returns ``(characterization, keep_trajectories)``."""
    from pace_agent.characterize import characterize
    from pace_agent.characterize import options as characterize_options

    opts = characterize_options.resolve(getattr(cfg, "characterize", None))
    keep = opts.pop("keep_trajectories")
    return characterize(agent, llm, agent_path, task=task, n=n_live, **opts), keep


def _stamp_config_identity(traj, cfg):
    """Record the run-config fingerprint + seeds in the trajectory identity (for the replay
    contract's config_match + reproducibility). Secrets are never stored (api_key dropped).
    """
    import hashlib
    import json

    cfg_snap = {k: v for k, v in (cfg.llm or {}).items() if k != "api_key"}
    cfg_repr = {
        "llm": cfg_snap,
        "num_runs": cfg.num_runs,
        "max_iter": cfg.max_iter,
        "max_llm_calls": cfg.max_llm_calls,
        "impl": getattr(cfg, "impl", None),
    }
    digest = hashlib.sha256(json.dumps(cfg_repr, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    traj.meta.setdefault("identity", {}).update(
        {
            "config": cfg_repr,
            "config_digest": digest,
            "seeds": {"llm": (cfg.llm or {}).get("seed", 0), "measurement": 0},
        }
    )


def _counter_pass(cfg, out_dir):
    import sys
    from pace_agent import telemetry

    argv = [sys.executable, "-m", "pace_agent._counterpass", out_dir]
    dur = max(5, int(cfg.num_runs)) + 3
    if cfg.telemetry == "uprof":
        from pace_agent.telemetry import uprof

        return {"mode": "uprof", **uprof.collect(argv, duration=dur)}
    return telemetry.collect(cfg.telemetry, argv)
