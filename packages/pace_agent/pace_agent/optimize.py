# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import warnings

from pace_agent.errors import ReplayDivergence
from pace_agent.llm.client import LLMClient
from pace_agent.profile.profiler import import_agent
from pace_agent.progress import say
from pace_agent.registry import get_tool
from pace_agent.replay import Player, Trajectory
from pace_agent.replay.modes import get_replay_mode, register_replay_mode
from pace_agent.stats import summarize


def _resolve_replay_mode(traj):
    """When a trajectory carries WebVoyager page snapshots, optimize replays ONLY those
    snapshots (the observe naive-vs-pace A/B) — not the incidental @pace_agent.op tool calls
    or LLM tape that may have been recorded alongside them during the live run."""
    if traj.meta.get("snapshots"):
        return "browser-snapshot"
    return traj.meta.get("replay_mode", "deterministic")


def optimize(run_dir, impl="pace", num_runs=10):
    """Replay a recorded run, timing the requested `impl` against the reference, dispatched
    by the trajectory's replay mode (no workload/example import)."""
    import glob

    opts = {"impl": impl, "num_runs": num_runs}
    if not os.path.exists(os.path.join(run_dir, "trajectory.msgpack")):
        # flat bench layout: one trajectory_<impl>_<task>.msgpack per task, each dispatched by
        # its own recorded replay mode (still no workload/example import)
        flats = sorted(glob.glob(os.path.join(run_dir, "trajectory_*.msgpack")))
        if flats:
            from pace_agent.replay import codec

            say("optimize", f"starting {len(flats)} recorded task(s) in {run_dir}")
            tasks = []
            for i, fp in enumerate(flats):
                with open(fp, "rb") as f:
                    t = codec.loads(f.read())
                name = os.path.basename(fp)
                label = f"{i + 1}/{len(flats)}"
                say("optimize", f"{name}: {_plan_line(t, opts)}", label)
                rep = get_replay_mode(_resolve_replay_mode(t))(
                    t, run_dir, {**opts, "label": name}
                )
                rep["trajectory"] = name
                say("optimize", f"{name}: {_outcome_line(rep)}", label)
                tasks.append(rep)
            return _finish(run_dir, _multi_report(tasks))

        # multi-task profile layout: query_<i>/ (one per task) -> aggregate. Enumerated by task
        # directory, not by trajectory: a failed task has none, and dropping it here would let the
        # survivors report a clean speedup for a broken run.
        qdirs = sorted(
            glob.glob(os.path.join(run_dir, "query_*")),
            key=lambda p: int(os.path.basename(p).rsplit("_", 1)[-1]),
        )
        if qdirs:
            return _optimize_multi(run_dir, qdirs, opts)

    _require_trajectory(run_dir)
    traj = Trajectory.load(run_dir)
    handler = get_replay_mode(_resolve_replay_mode(traj))
    say("optimize", f"{run_dir}: {_plan_line(traj, opts)}")
    report = handler(traj, run_dir, opts)
    say("optimize", _outcome_line(report))
    return _finish(run_dir, report)


def _require_trajectory(run_dir):
    """Fail with a helpful message when result.json claims replay data but no trajectory exists."""
    mp = os.path.join(run_dir, "trajectory.msgpack")
    pkl = os.path.join(run_dir, "trajectory.pkl")
    if os.path.exists(mp) or os.path.exists(pkl):
        return
    respath = os.path.join(run_dir, "result.json")
    if os.path.exists(respath):
        import json

        with open(respath) as f:
            res = json.load(f)
        if res.get("replay_mode") == "browser-snapshot":
            n = res.get("n_snapshots", "?")
            raise FileNotFoundError(
                f"no trajectory (.msgpack or .pkl) in {run_dir} — result.json "
                f"records {n} browser snapshot(s) but trajectory.msgpack was never "
                f"written (re-run the GAIA/profile step to record it)"
            )
    raise FileNotFoundError(f"no trajectory (.msgpack or .pkl) in {run_dir}")


def _finish(run_dir, report):
    """Persist the optimize artifact, refresh the rendered report, and say where both went."""
    from pace_agent.report.render import render

    path = write_optimize(run_dir, report)
    render(run_dir)
    say("optimize", f"wrote {os.path.basename(path)} and report.md")
    return report


def _plan_line(traj, opts):
    """Which mode a task will be replayed under, over how much recorded work."""
    mode = _resolve_replay_mode(traj)
    browser = mode == "browser-snapshot"
    n = len(traj.meta.get("snapshots") or []) if browser else len(traj.tools)
    unit = "snapshot" if browser else "recorded call"
    reps = opts.get("num_runs", 10)
    return f"{mode} replay of {n} {unit}(s) x{reps}, candidate={opts.get('impl')}"


def _outcome_line(rep):
    """What a task's replay found, in the terms of its own replay mode."""
    if rep.get("error"):
        return f"failed: {rep['error']}"
    if rep.get("replay_mode") == "browser-snapshot":
        return (
            f"observe {_x(rep.get('observe_speedup'))} over {rep.get('n_snapshots')} "
            f"snapshot(s), parity {_verdict(rep.get('observe_parity_all'))}"
        )
    cov = rep.get("coverage") or {}
    return (
        f"E2E {_x(rep.get('e2e_speedup'))}, coverage {cov.get('candidate_calls', 0)}/"
        f"{cov.get('total_calls', 0)} call(s), valid {rep.get('valid')}"
    )


def _verdict(passed):
    return "ok" if passed else "FAILED"


def _x(speedup):
    return f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "n/a"


def _ms(seconds):
    return f"{seconds * 1e3:.3f}" if isinstance(seconds, (int, float)) else "n/a"


def write_optimize(run_dir, report):
    """Persist an optimize report as its own timestamped artifact, kept OUT of result.json so
    each optimize run is preserved and result.json stays the profile record."""
    import json
    import time

    path = os.path.join(run_dir, f"optimize_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def read_latest_optimize(run_dir):
    """The most recent optimize_<ts>.json in run_dir (None if never optimized)."""
    import glob
    import json

    files = sorted(glob.glob(os.path.join(run_dir, "optimize_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _candidate_available(tool, impl):
    """Whether `impl` is a genuine (non-fallback) candidate for this op."""
    if impl == "naive":
        return True
    if impl == "optimized":
        info = tool.implementations().get("optimized")
        return bool(info and not info.is_fallback)
    from pace_agent.providers import get_provider_impl

    return get_provider_impl(impl, tool.name) is not None


def _deterministic_replay(traj, run_dir, opts):
    """Built-in op-replay. Replays the recorded run under the naive reference (validating the
    replay contract), then re-times ONLY the candidate-covered ops and projects the honest
    end-to-end: un-covered ops keep their naive time (Amdahl-correct). Never a phantom
    speedup — zero coverage or a divergent replay is reported as invalid."""
    impl = opts.get("impl", "pace")
    num_runs = opts.get("num_runs", 10)
    label = opts.get("label")

    agent = import_agent(traj.meta.get("agent"))
    llm = LLMClient(endpoint="", model="")
    player = Player()

    ops = {o.op for o in traj.tools}
    available = {name: _candidate_available(get_tool(name), impl) for name in ops}
    covered = sorted(name for name in ops if available[name])
    say(
        "optimize",
        f"candidate '{impl}' covers {len(covered)}/{len(ops)} op(s): "
        + (", ".join(covered) or "none"),
        label,
    )
    say(
        "optimize",
        f"reference: replaying {len(traj.tools)} recorded call(s) under naive x{num_runs}",
        label,
    )

    # reference: one full naive replay — validates the contract + gives E2E samples/per-op naive
    try:
        naive = player.replay(traj, agent, llm, "naive", num_runs)
    except ReplayDivergence as e:
        say("optimize", f"replay DIVERGED from the recording: {e}", label)
        return {
            "cmd": "optimize",
            "replay_mode": "deterministic",
            "reference_impl": "naive",
            "candidate_impl": impl,
            "impl": impl,
            "replay_contract": {
                "mode": "op-replay",
                "valid": False,
                "divergence": str(e),
            },
            "valid": False,
            "invalid_reason": f"replay diverged from the recorded trajectory: {e}",
            "e2e_speedup": None,
            "parity_all_pass": False,
        }

    # re-time ONLY the covered ops; everything else keeps its recorded naive time
    calls_by_op = {}
    for o in traj.tools:
        calls_by_op.setdefault(o.op, []).append(o)
    per_tool, parity = {}, {}
    recoverable_total, covered_naive_time = 0.0, 0.0
    total_naive_time = sum(o.naive_time for o in traj.tools)
    for name, calls in calls_by_op.items():
        ref = summarize(naive["per_tool"].get(name, []))
        if available.get(name):
            iso = player.isolate(traj, name, tier=impl, repeat=num_runs)
            ncalls = max(1, iso["n_calls"])
            per_tool[name] = {
                "reference": ref,
                "candidate": {"median": iso["candidate_time"] / ncalls},
                "is_candidate": True,
                "speedup": (
                    iso["naive_time"] / iso["candidate_time"]
                    if iso["candidate_time"]
                    else None
                ),
            }
            recoverable_total += iso["recoverable"]
            covered_naive_time += iso["naive_time"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parity_results = [
                    get_tool(name).parity(
                        call.args,
                        call.kw,
                        reference="naive",
                        candidate=impl,
                    )
                    for call in calls
                ]
                parity[name] = all(parity_results)
            say(
                "optimize",
                f"retimed {name}: {len(calls)} call(s), naive {_ms(ref.get('median'))} -> "
                f"{impl} {_ms(per_tool[name]['candidate']['median'])} ms/call, "
                f"parity {_verdict(parity[name])}",
                label,
            )
        else:
            per_tool[name] = {
                "reference": ref,
                "candidate": ref,
                "is_candidate": False,
                "speedup": 1.0,
            }
            say(
                "optimize",
                f"{name}: no '{impl}' candidate, keeping its recorded naive time",
                label,
            )

    candidate_calls = sum(1 for o in traj.tools if available.get(o.op))
    coverage_info = {
        "total_calls": len(traj.tools),
        "candidate_calls": candidate_calls,
        "covered_ops": sorted(n for n in ops if available[n]),
        "covered_time_fraction": (
            covered_naive_time / total_naive_time if total_naive_time else 0.0
        ),
    }

    en = summarize(naive["e2e"])
    # honest projected end-to-end: only the covered ops change; the rest keep naive time
    cand_e2e_median = max(0.0, en["median"] - recoverable_total)

    # replay contract: source + config identity (source drift invalidates under strict compat)
    src_match, cfg_match = _source_match(traj), _config_match(traj)
    compat = opts.get("compat", "warn")
    contract_valid = not (src_match is False and compat == "strict")
    ident = traj.meta.get("identity") or {}
    replay_contract = {
        "mode": "op-replay",
        "validated": naive["contract"]["validated"],
        "tool_calls_expected": naive["contract"]["tool_calls_expected"],
        "tool_calls_consumed": naive["contract"]["tool_calls_consumed"],
        "llm_calls_expected": naive["contract"]["llm_calls_expected"],
        "source_match": src_match,
        "config_match": cfg_match,
        "config_digest": ident.get("config_digest"),
        "seeds": ident.get("seeds"),
        "compat_policy": compat,
        "valid": contract_valid,
    }

    parity_all_pass = bool(parity) and all(parity.values())
    valid = candidate_calls > 0 and contract_valid and parity_all_pass
    speedup = (en["median"] / cand_e2e_median) if (cand_e2e_median and valid) else None
    from pace_agent.providers import provider_meta

    provenance = (
        {"source": "builtin", "version": None}
        if impl in ("naive", "optimized")
        else provider_meta(impl)
    )
    candidate_e2e = {"median": cand_e2e_median}
    report = {
        "cmd": "optimize",
        "replay_mode": "deterministic",
        "reference_impl": "naive",
        "candidate_impl": impl,
        "impl": impl,  # legacy alias (one release)
        "coverage": coverage_info,
        "provider": provenance,
        "replay_contract": replay_contract,
        "e2e": {"reference": en, "candidate": candidate_e2e, "speedup": speedup},
        "e2e_naive": en,  # legacy alias (one release)
        "e2e_pace": candidate_e2e,  # legacy alias (one release)
        "e2e_speedup": speedup,
        "per_tool": per_tool,
        "parity": parity,
        "parity_all_pass": parity_all_pass,
        "valid": valid,
    }
    if not valid:
        if not contract_valid:
            report["invalid_reason"] = (
                f"source drift under strict compat policy (source_match={src_match})"
            )
        elif candidate_calls == 0:
            report["invalid_reason"] = (
                f"zero candidate coverage for impl '{impl}' "
                f"(no op has a non-fallback '{impl}' implementation)"
            )
        else:
            failed = sorted(name for name, passed in parity.items() if not passed)
            report["invalid_reason"] = (
                "candidate output parity failed for: " + ", ".join(failed)
            )
    return report


def _source_match(traj):
    """Whether the agent source still matches what was recorded (None if not determinable)."""
    import hashlib
    import os

    rec = (traj.meta.get("identity") or {}).get("agent_source_digest")
    path = traj.meta.get("agent")
    if not rec or not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16] == rec


def _config_match(traj):
    """Whether the recorded run-config fingerprint is intact — the config snapshot in the
    trajectory still hashes to the recorded digest (None if no config was recorded)."""
    import hashlib
    import json

    ident = traj.meta.get("identity") or {}
    rec, cfg = ident.get("config_digest"), ident.get("config")
    if not rec or cfg is None:
        return None
    return (
        hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16] == rec
    )


def _replay_one(args):
    """Picklable worker: replay ONE task trajectory and return its report (observe A/B + E2E)."""
    import os

    from pace_agent.replay import codec
    from pace_agent.watchdog import watch

    fp, opts = args[0], args[1]
    index, total = args[2:4] if len(args) > 3 else (None, None)
    with open(fp, "rb") as f:
        t = codec.loads(f.read())
    task = os.path.basename(os.path.dirname(fp))
    label = f"{index + 1}/{total}" if index is not None else None
    mode = get_replay_mode(_resolve_replay_mode(t))
    say("optimize", f"{task}: {_plan_line(t, opts)}", label)
    with watch(
        label=f"optimize {task}"
    ):  # a wedged replay must not hold a worker either
        rep = mode(t, os.path.dirname(fp), {**opts, "label": task})
    rep["task"] = task
    say("optimize", f"{task}: {_outcome_line(rep)}", label)
    return rep


def _optimize_multi(run_dir, qdirs, opts):
    """Aggregate optimize over a multi-task profile run (one query_<i>/ per task): each task's
    observe A/B + E2E projection runs in its own worker (parallelism read from the run's persisted
    config.json, same as the profile run), then pooled into a workload summary. A task with no
    trajectory is carried in as an error, so the survivors cannot report a speedup for the run.
    """
    import os

    from pace_agent.parallel import run_parallel

    workers, cores = 1, 2
    cfgp = os.path.join(run_dir, "config.json")
    if os.path.exists(cfgp):
        from pace_agent.config import load_config

        c = load_config(cfgp)
        workers = getattr(c, "parallel", 1) or 1
        cores = getattr(c, "cores_per_worker", 2) or 2
    workers = min(workers, len(qdirs))  # what run_parallel will actually start
    total = len(qdirs)
    rows, jobs = [None] * total, []
    for i, d in enumerate(qdirs):
        fp = os.path.join(d, "trajectory.msgpack")
        if os.path.exists(fp):
            jobs.append((i, (fp, opts, i, total)))
        else:
            rows[i] = _unreplayable(d, i, total)
    say(
        "optimize",
        f"starting {len(jobs)} of {total} task(s): workers={workers} cores/worker={cores} "
        f"candidate={opts.get('impl')} reps={opts.get('num_runs')}",
    )
    replayed = run_parallel(
        _replay_one,
        [job for _, job in jobs],
        workers=workers,
        cores_per_worker=cores,
    )
    for (i, _), rep in zip(jobs, replayed):
        rows[i] = rep
    report = _multi_report(rows)
    agg = report.get("aggregate") or {}
    say(
        "optimize",
        f"finished: {agg.get('n_scored', agg.get('n_tasks'))}/{agg.get('n_tasks')} scored, "
        f"{agg.get('n_parity_fail', agg.get('n_invalid', 0))} parity-fail, "
        f"{agg.get('n_failed', 0)} error",
    )
    return _finish(run_dir, report)


def _unreplayable(task_dir, index, total):
    """An error row for a task with no trajectory, naming the reason its own result row recorded."""
    import json

    task = os.path.basename(task_dir)
    reason = "no trajectory was recorded"
    respath = os.path.join(task_dir, "result.json")
    if os.path.exists(respath):
        with open(respath) as f:
            recorded = json.load(f).get("error")
        if recorded:
            reason = f"the task failed during profile: {recorded}"
    say("optimize", f"{task}: cannot be replayed — {reason}", f"{index + 1}/{total}")
    return {"task": task, "error": f"nothing to replay — {reason}"}


def _multi_report(tasks):
    modes = {task.get("replay_mode") for task in tasks if not task.get("error")}
    if modes == {"browser-snapshot"}:
        return {
            "cmd": "optimize",
            "mode": "browser-snapshot",
            "aggregate": _aggregate_browser_snapshot(tasks),
            "tasks": tasks,
        }
    failed = [task for task in tasks if task.get("error")]
    invalid = [
        task for task in tasks if not task.get("error") and not task.get("valid", True)
    ]
    return {
        "cmd": "optimize",
        "mode": "multi",
        "aggregate": {
            "n_tasks": len(tasks),
            "n_failed": len(failed),
            "n_invalid": len(invalid),
            "valid": not failed and not invalid,
            "replay_modes": {
                mode: sum(task.get("replay_mode") == mode for task in tasks)
                for mode in sorted(modes)
            },
        },
        "tasks": tasks,
    }


def _aggregate_browser_snapshot(tasks):
    """Pool per-task observe A/B + E2E projections into a workload-level summary. ONLY tasks whose
    observe output matched (parity) contribute to the speedup; parity failures and worker errors
    are counted and invalidate the aggregate — a changed output must never claim a speedup.
    """
    import statistics

    scored = [t for t in tasks if t.get("n_snapshots") and t.get("e2e")]
    failed = [t for t in tasks if not (t.get("n_snapshots") and t.get("e2e"))]
    parity_fail = [t for t in scored if not t.get("observe_parity_all")]
    ok = [
        t for t in scored if t.get("observe_parity_all")
    ]  # parity passed -> speedup counts
    valid = not parity_fail and not failed

    def _med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 3) if xs else None

    e_ref = sum(t["e2e"]["reference"]["median"] for t in ok)
    e_cand = sum(t["e2e"]["candidate"]["median"] for t in ok)
    bd = {
        "llm_model": 0.0,
        "observe_naive": 0.0,
        "observe_pace": 0.0,
        "other_browser": 0.0,
    }
    for t in ok:
        for k in bd:
            bd[k] += t["e2e"]["breakdown_s"].get(k, 0.0)
    agg = {
        "n_tasks": len(tasks),
        "n_scored": len(ok),
        "n_parity_fail": len(parity_fail),
        "n_failed": len(failed),
        "valid": valid,
        "total_snapshots": sum(t["n_snapshots"] for t in ok),
        "observe_naive_ms_median": _med(
            [(t.get("observe_naive_ms") or {}).get("median") for t in ok]
        ),
        "observe_pace_ms_median": _med(
            [(t.get("observe_pace_ms") or {}).get("median") for t in ok]
        ),
        "observe_speedup_median": _med([t.get("observe_speedup") for t in ok]),
        "e2e": {
            "reference_total_s": round(e_ref, 1),
            "candidate_total_s": round(e_cand, 1),
            "speedup": round(e_ref / e_cand, 3) if e_cand > 0 else None,
            "pct_faster": round((1 - e_cand / e_ref) * 100, 1) if e_ref else None,
            "breakdown_total_s": {k: round(v, 1) for k, v in bd.items()},
        },
    }
    if not valid:
        agg["invalid_reason"] = (
            f"{len(parity_fail)} parity failure(s) + {len(failed)} task error(s) of "
            f"{len(tasks)}; speedup covers only the {len(ok)} valid task(s)"
        )
    return agg


def _llm_only_replay(traj, run_dir, opts):
    """Replay mode for trajectories that captured only LLM turns and no ``@pace_agent.op`` tool
    calls — e.g. the gaia planner path, where the browser/sandbox tools run out-of-process
    and are never recorded as ops. There are no CPU-bound ops whose implementation could be
    swapped, so optimization is a well-formed no-op: report zero coverage with a clear reason
    instead of crashing. (Trajectories that captured page snapshots use ``browser-snapshot``;
    those with recorded ops use ``deterministic``.)"""
    impl = opts.get("impl", "pace")
    n_llm = len(getattr(traj, "llm_tape", None) or [])
    # Mirror the deterministic handler's *divergence* report shape (no ``e2e`` key) so the
    # renderer skips the speedup section cleanly rather than dereferencing a null E2E.
    return {
        "cmd": "optimize",
        "replay_mode": "llm-only",
        "reference_impl": "naive",
        "candidate_impl": impl,
        "impl": impl,  # legacy alias (one release)
        "replay_contract": {"mode": "llm-only", "valid": True, "llm_calls": n_llm},
        "valid": False,
        "invalid_reason": (
            "llm-only trajectory: only model turns were recorded (no @pace_agent.op tool "
            "calls), so there are no CPU-bound ops to optimize"
        ),
        "e2e_speedup": None,
        "parity_all_pass": False,
    }


register_replay_mode("deterministic", _deterministic_replay)
register_replay_mode("llm-only", _llm_only_replay)
