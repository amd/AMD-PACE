# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter

from pace_agent import score
from pace_agent.bench.tools import DEFAULT_TOOLS, bench_tools
from pace_agent.check import run_checks
from pace_agent.config import BenchConfig, load_bench_config, load_config
from pace_agent.providers import providers_for
from pace_agent.registry import PaceAgentTool, list_tools, load_plugins

_REQUIRED = {"pace_agent", "numpy", "tools registered"}


def _default_bench_config():
    return BenchConfig(tools=DEFAULT_TOOLS)


def cmd_check(args):
    checks = run_checks(llm_endpoint=args.llm)
    failed_required = 0
    for c in checks:
        mark = "ok " if c["ok"] else "MISS"
        req = " (required)" if c["name"] in _REQUIRED else ""
        line = f"[{mark}] {c['name']}{req}: {c['detail']}"
        if not c["ok"] and c["doc"]:
            line += f"  -> see {c['doc']}"
        print(line)
        if not c["ok"] and c["name"] in _REQUIRED:
            failed_required += 1
    return 1 if failed_required else 0


def _tiers(op):
    tiers = ["naive"]
    # "optimized" only when the tool overrides the naive-fallback stub.
    if type(op).optimized is not PaceAgentTool.optimized:
        tiers.append("optimized")
    # Provider tiers (e.g. "pace") that advertise an impl for this op.
    tiers.extend(providers_for(op.name))
    return tiers


def _tier_label(tier, op=None):
    if tier == "naive":
        return "reference"
    if tier == "optimized":
        # honest: an optional-lib optimized tier whose lib is absent is a naive fallback
        if op is not None:
            info = op.implementations().get("optimized")
            if info is not None and info.is_fallback:
                return "reference (accelerator not installed)"
        return "candidate"
    from pace_agent.providers import provider_meta

    m = provider_meta(tier)
    ver = f" {m['version']}" if m.get("version") else ""
    return f"provider: {m.get('source') or tier}{ver}"


def cmd_tools_list(args):
    bucket = None
    for op in list_tools():
        if op.bucket != bucket:
            bucket = op.bucket
            print(f"\n{bucket}")
        cls = "io-pattern" if op.io_pattern else "cpu-compute"
        tiers = ", ".join(f"{t} ({_tier_label(t, op)})" for t in _tiers(op))
        print(f"  {op.name:20s} [{cls}] tiers={tiers}")
    return 0


_EXAMPLES = [
    "basic_example",
    "customer_support",
    "tool_agent",
    "large_corpus_rag",
    "testgen",
    "sandbox_tool_agent",
    "webvoyager",
]


def cmd_examples_list(args):
    print("examples (pace-agent profile <name> --config <cfg>):")
    for name in _EXAMPLES:
        import importlib

        m = importlib.import_module(f"pace_agent.examples.{name}")
        print(f"  {name:20s} [{getattr(m, 'KIND', 'standard')}]")
    return 0


def _resolve_agent(spec):
    if spec.endswith(".py") or "/" in spec or "\\" in spec or os.path.isfile(spec):
        return os.path.abspath(
            spec
        )  # persisted in trajectory; re-imported by absolute path
    if "." in spec:
        return spec
    return f"pace_agent.examples.{spec}"


def _out_dir(args, workload):
    if args.out:
        return args.out
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("pace_agent_results", f"{ts}_{workload}")


def _apply_characterize_overrides(cfg, args):
    """CLI flags win over the config's characterize block, like --query wins over its tasks."""
    if getattr(args, "characterize_runs", None) is not None:
        cfg.characterize_runs = args.characterize_runs
    if getattr(args, "characterize_strategy", None):
        cfg.characterize = {
            **(cfg.characterize or {}),
            "strategy": args.characterize_strategy,
        }
    from pace_agent.characterize import options

    options.resolve(cfg.characterize)  # a bad override must fail before any live run
    return cfg


def cmd_profile(args):
    if not args.agent:
        print("error: agent is required unless --list is used")
        return 2
    if not args.config:
        print(
            "error: profile requires --config "
            "(provide a run config JSON; see packages/pace_agent/configs/ for examples)"
        )
        return 2
    from pace_agent.pinning import apply_pinning
    from pace_agent.profile.profiler import profile

    cfg = load_config(args.config)
    if args.query:
        cfg.tasks = [{"query": q} for q in args.query]
    _apply_characterize_overrides(cfg, args)
    load_plugins(cfg.plugins)
    apply_pinning(cfg.pinning)
    agent_path = _resolve_agent(args.agent)
    out = _out_dir(args, cfg.workload or "profile")
    os.makedirs(out, exist_ok=True)
    # persist the exact run config for reference + so optimize can read its parallelism
    shutil.copyfile(args.config, os.path.join(out, "config.json"))
    tasks = cfg.tasks or [{}]
    # the copied config still holds the tasks it shipped with, so only a --query override needs
    # recording separately: without this the run dir would claim tasks that never ran
    if args.query:
        with open(os.path.join(out, "tasks.json"), "w") as f:
            json.dump({"tasks": tasks}, f, indent=2, default=str)
    if len(tasks) <= 1:
        from pace_agent.errors import PaceAgentError
        from pace_agent.profile.profiler import write_failure

        try:
            profile(agent_path, cfg, out)
        except Exception as e:  # noqa: BLE001 - a failure is recorded, not just printed
            write_failure(agent_path, cfg, tasks[0], out, e)
            print(
                f"error: {e}" if isinstance(e, PaceAgentError) else f"error: {e!r}",
                file=sys.stderr,
            )
            # the typed exit codes (stalled, browser, provider, config) must survive being recorded
            return e.exit_code if isinstance(e, PaceAgentError) else 1
        print(f"wrote {out}  (run `pace-agent report {out}` for the report)")
        return 0
    return _profile_multi(agent_path, cfg, tasks, out)


def _profile_multi(agent_path, cfg, tasks, out):
    from pace_agent.env import capture_env
    from pace_agent.parallel import run_parallel, worker_cpu_plan
    from pace_agent.pinning import cpu_spec
    from pace_agent.profile.profiler import profile_task
    from pace_agent.rundir import RunDir

    # each query is independent -> run them across pinned-core worker processes (or sequentially
    # when parallel<=1). Per-query replay timing still runs sequentially inside each profile.
    # clamped because a pool never starts more workers than tasks, and env.json has to describe
    # the run that happened
    workers = min(getattr(cfg, "parallel", 1) or 1, len(tasks))
    cores = getattr(cfg, "cores_per_worker", 2)

    # env is identical across workers (same host) -> capture once at the run root, not per query,
    # extended with the worker layout the pool will apply (workers pin themselves, and they do
    # not write their own env.json)
    cfg.write_env = False
    env = capture_env()
    if workers > 1:
        env["pinning"] = {
            **(env.get("pinning") or {}),
            "parallel": workers,
            "cores_per_worker": cores,
            "worker_cpus": [cpu_spec(c) for c in worker_cpu_plan(workers, cores)],
        }
    RunDir(out).write_env(env)

    from pace_agent.watchdog import stall_limit_s

    stall = stall_limit_s()
    print(
        f"[profile] starting {len(tasks)} task(s): workers={workers} cores/worker={cores} "
        f"stall limit={f'{stall:g}s with no progress' if stall else 'off'}",
        flush=True,
    )
    args = [
        (agent_path, cfg, task, os.path.join(out, f"query_{i}"), i, len(tasks))
        for i, task in enumerate(tasks)
    ]
    per = run_parallel(profile_task, args, workers=workers, cores_per_worker=cores)

    gold_key = score.resolve(getattr(cfg, "scoring", None))
    # a worker killed outright never reached its own failure path, so the parent's placeholder row
    # arrives unnamed and ungraded; finish it here so every task is named, graded like any other
    # failure, and has a result.json of its own
    for i, (row, task) in enumerate(zip(per, tasks)):
        # any row a worker produced, success or failure, carries its timings
        if "e2e" in row:
            continue
        row.update(
            {
                "cmd": "profile",
                "agent": agent_path,
                "query": (task or {}).get("query"),
                "completed": False,
                "steps": 0,
                "answer": None,
                "e2e": {"n": 0},
                "per_node": {},
                "run_metrics": {},
            }
        )
        if gold_key:
            row.update(score.score(None, (task or {}).get(gold_key)))
        RunDir(os.path.join(out, f"query_{i}")).write_result(row)

    def _s(r, *path):
        for k in path:
            r = (r or {}).get(k, {})
        return r or 0

    observe_speedups = [
        (r.get("observe_ab") or {}).get("speedup")
        for r in per
        if (r.get("observe_ab") or {}).get("speedup")
    ]
    agg = {
        "n_queries": len(per),
        "answered": sum(bool(r.get("completed")) for r in per),
        "n_errors": sum(bool(r.get("error")) for r in per),
        "e2e_median_total_s": sum(_s(r, "e2e", "median") for r in per),
        "total_llm_calls": sum(_s(r, "run_metrics", "llm_calls") for r in per),
        "total_nodes_traversed": sum(
            _s(r, "run_metrics", "nodes_traversed") for r in per
        ),
        "total_context_tokens": sum(
            _s(r, "run_metrics", "context_tokens_total") for r in per
        ),
        "total_cached_tokens": sum(
            _s(r, "run_metrics", "cached_tokens_total") for r in per
        ),
        "total_output_tokens": sum(
            _s(r, "run_metrics", "output_tokens_total") for r in per
        ),
        "total_reasoning_tokens": sum(
            _s(r, "run_metrics", "reasoning_tokens_total") for r in per
        ),
        "total_non_model_cpu_s": sum(
            _s(r, "attribution", "non_model_cpu") for r in per
        ),
        # in-run observe A/B (browser agents); the projected E2E speedup comes from `pace-agent optimize`
        "observe_speedup_median": (
            statistics.median(observe_speedups) if observe_speedups else None
        ),
        "per_node_total_ms": {
            n: sum(
                ((r.get("per_node") or {}).get(n) or {}).get("total_ms", 0) for r in per
            )
            for n in sorted({n for r in per for n in (r.get("per_node") or {})})
        },
    }
    char = _characterization_rollup(per)
    if char:
        agg["characterization"] = char
    if gold_key:
        agg.update(score.aggregate(per))
        agg["scoring"] = {"gold_key": gold_key}
    RunDir(out).write_result(
        {
            "cmd": "profile_multi",
            "agent": agent_path,
            "impl": getattr(cfg, "impl", "pace"),
            "per_query": [
                {
                    "query": r.get("query"),
                    "answer": r.get("answer"),
                    "completed": r.get("completed"),
                    # browser agents make no PaceAgentTool calls; steps = iterations
                    "steps": r.get("steps"),
                    "error": r.get("error"),
                    **{
                        k: r[k]
                        for k in ("gold", "match_strict", "match_loose")
                        if k in r
                    },
                    **r.get("run_metrics", {}),
                    # per-task e2e is at r["e2e"], not r["stats"]
                    "e2e_median": _s(r, "e2e", "median"),
                }
                for r in per
            ],
            "aggregate": agg,
        }
    )
    graded_line = (
        f" match=strict:{agg.get('auto_match_strict')} "
        f"loose:{agg.get('auto_match_loose')}/{agg.get('n_scored')}"
        + (f" (+{agg['n_unscored']} ungraded)" if agg.get("n_unscored") else "")
        if gold_key
        else ""
    )
    print(
        f"[profile] finished: completed={agg['answered']}/{len(per)} "
        f"errors={agg['n_errors']}{graded_line}",
        flush=True,
    )
    # reprint the failures here: in a 90-task log the originals have long scrolled off
    for i, r in enumerate(per):
        if r.get("error"):
            print(
                f"[profile {i + 1}/{len(per)}] failed: {str(r.get('query'))[:60]!r}: {r['error']}",
                file=sys.stderr,
                flush=True,
            )
    print(
        f"wrote {out} ({len(per)} queries)  (run `pace-agent report {out}` for the report)"
    )
    print(json.dumps(agg, indent=2, default=str))
    if agg["n_errors"] == len(per):
        # a task failing is expected (live sites, stalls) and never aborts the batch, but a run in
        # which nothing succeeded produced no measurement and must not look like a good run
        print(
            f"error: every task failed ({len(per)}/{len(per)}); see the failures above",
            file=sys.stderr,
        )
        return 1
    return 0


def _characterization_rollup(per):
    """Characterization across a suite, including how many tasks still had to record an
    outlier because no better run was available."""
    chars = [r["characterization"] for r in per if r.get("characterization")]
    if not chars:
        return None

    def tally(key):
        labels = [c[key] for c in chars if c.get(key) is not None]
        return dict(sorted(Counter(labels).items()))

    def median(key):
        return round(statistics.median(c.get(key, 0) for c in chars), 4)

    return {
        "n_tasks": len(chars),
        "live_runs_total": sum(c.get("n_runs", 0) for c in chars),
        "modal_share_median": median("modal_share"),
        "normalized_entropy_median": median("normalized_entropy"),
        "stability": tally("stability"),
        "work_stability": tally("work_stability"),
        "verdict": tally("verdict"),
        "n_recorded_outliers": sum(bool(c.get("selected_is_outlier")) for c in chars),
    }


def cmd_optimize(args):
    from pace_agent.optimize import optimize

    r = optimize(args.run_dir, impl=args.impl, num_runs=args.num_runs)
    agg = r.get("aggregate")
    if agg and r.get("mode") == "browser-snapshot":
        e = agg.get("e2e") or {}
        print(
            f"observe: naive {agg.get('observe_naive_ms_median', 0):.1f} -> "
            f"pace {agg.get('observe_pace_ms_median', 0):.1f} ms/observe "
            f"({agg.get('observe_speedup_median')}x median)  "
            f"[{agg.get('n_scored')}/{agg.get('n_tasks')} valid, "
            f"{agg.get('n_parity_fail', 0)} parity-fail, {agg.get('n_failed', 0)} error; "
            f"{agg.get('total_snapshots')} snapshots]"
        )
        print(
            f"E2E: {e.get('reference_total_s', 0):.0f}s -> "
            f"{e.get('candidate_total_s', 0):.0f}s  "
            f"({e.get('speedup')}x, {e.get('pct_faster')}% faster)"
        )
        bd = e.get("breakdown_total_s") or {}
        print("  breakdown (s): " + "  ".join(f"{k}={v}" for k, v in bd.items()))
        if not agg.get("valid", True):
            print(f"INVALID: {agg.get('invalid_reason')}", file=sys.stderr)
            return 1
        return 0
    if agg:
        print(
            f"multi-task optimize: {agg.get('n_tasks')} tasks, "
            f"{agg.get('n_failed', 0)} failed, {agg.get('n_invalid', 0)} invalid, "
            f"modes={agg.get('replay_modes', {})}"
        )
        return 0 if agg.get("valid", False) else 1
    tasks = r.get("tasks")
    snapshot_modes = ("html-replay", "browser-snapshot")
    if (
        tasks is not None
        or r.get("mode") in snapshot_modes
        or r.get("replay_mode") in snapshot_modes
    ):
        for t in tasks or [r]:
            on, op = t.get("observe_naive_ms", {}), t.get("observe_pace_ms", {})
            sp = t.get("observe_speedup")
            line = (
                f"[{t.get('task') or t.get('trajectory') or 'trace'}] "
                f"observe naive {on.get('median', 0):.2f} -> pace {op.get('median', 0):.3f} ms"
                + (f" ({sp:.1f}x)" if sp else "")
                + f"  [{t.get('n_snapshots', '?')} snapshots x{t.get('reps', '?')}]"
            )
            e = t.get("e2e") or {}
            if e:
                line += (
                    f"  | E2E {e['reference']['median']:.1f}s -> "
                    f"{e['candidate']['median']:.1f}s ({e.get('pct_faster')}% faster)"
                )
            print(line)
        return 0
    if not r.get("valid", True):
        print(f"optimize INVALID: {r.get('invalid_reason', 'zero candidate coverage')}")
        return 1
    ref, cand = r.get("reference_impl", "naive"), r.get("candidate_impl", args.impl)
    e2e = r.get("e2e") or {"reference": r["e2e_naive"], "candidate": r["e2e_pace"]}
    sp = r.get("e2e_speedup")
    tail = f" ({sp:.2f}x)" if sp else ""
    print(
        f"E2E {ref} {e2e['reference']['median'] * 1e3:.2f} -> "
        f"{cand} {e2e['candidate']['median'] * 1e3:.2f} ms{tail}"
    )
    cov = r.get("coverage")
    if cov:
        print(
            f"candidate coverage: {cov.get('candidate_calls', 0)}/"
            f"{cov.get('total_calls', 0)} calls"
        )
    print(f"parity all pass: {r['parity_all_pass']}")
    return 0 if r["parity_all_pass"] else 1


def cmd_report(args):
    from pace_agent.report.render import render

    path = render(args.run_dir)
    print(open(path).read())
    return 0


def cmd_bench_tools(args):
    if args.repeat is not None and args.repeat < 1:
        print("error: --repeat must be >= 1")
        return 2
    cfg = load_bench_config(args.config) if args.config else _default_bench_config()
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    if not tiers:
        print("error: --tiers must name at least one tier")
        return 2
    rows = bench_tools(cfg, tiers=tiers, repeat=args.repeat)
    _print_bench(rows, tiers)
    return 0


def _print_bench(rows, tiers):
    has_cmp = {"naive", "optimized"} <= set(tiers)
    print(
        f"{'op':22s}{'bucket':16s}{'status':8s}"
        + "".join(f"{t + ' med(ms)':18s}" for t in tiers)
        + ("speedup [95% CI]         n     CV     parity" if has_cmp else "")
    )
    for r in rows:
        if r["status"] == "skip":
            print(f"{r['op']:22s}{r['bucket']:16s}{'skip':8s}{r.get('reason', '')}")
            continue
        line = f"{r['op']:22s}{r['bucket']:16s}{'ok':8s}"
        for t in tiers:
            cell = r["tiers"].get(t)
            if cell is None:
                line += f"{'skip':18s}"
            elif "fallback" in cell:
                line += f"{'=naive':18s}"
            else:
                line += f"{cell['median'] * 1e3:<18.3f}"
        if "speedup" in r:
            sp, ci = r["speedup"], r.get("speedup_ci")
            cand = r["tiers"].get("optimized", {})
            n = cand.get("n") or r["tiers"]["naive"].get("n") or 0
            cv = cand.get("cv")
            ci_s = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
            # a fast candidate is noise-prone; flag runs the user should repeat pinned
            cv_s = (
                (f"{cv * 100:.0f}%" + ("!" if cv > 0.10 else ""))
                if cv is not None
                else "-"
            )
            line += (
                f"{(f'{sp:.2f}x' if sp else '-'):9s}{ci_s:16s}"
                f"{n:<6d}{cv_s:>6s}  {'PASS' if r['parity'] else 'FAIL'}"
            )
        print(line)


def main(argv=None):
    # local import: characterize reaches back into replay, which imports pace_agent.config
    from pace_agent.characterize.select import STRATEGIES

    p = argparse.ArgumentParser(prog="pace-agent")
    p.add_argument("--llm", default=os.environ.get("PACE_AGENT_LLM_ENDPOINT"))
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("check")
    pb = sub.add_parser("bench")
    bsub = pb.add_subparsers(dest="bench_cmd")
    bo = bsub.add_parser("tools")
    bo.add_argument("--list", action="store_true")
    bo.add_argument("--config")
    bo.add_argument("--tiers", default="naive,optimized")
    bo.add_argument("--repeat", type=int, default=None)
    pp = sub.add_parser("profile")
    pp.add_argument("agent", nargs="?")
    pp.add_argument("--list", action="store_true")
    pp.add_argument("--query", action="append")
    pp.add_argument("--config")
    pp.add_argument("--out")
    # LIVE runs sampled before recording; distinct from num_runs, which replays.
    pp.add_argument(
        "--characterize-runs", dest="characterize_runs", type=int, default=None
    )
    pp.add_argument(
        "--characterize-strategy",
        dest="characterize_strategy",
        choices=STRATEGIES,
        default=None,
    )

    popt = sub.add_parser("optimize")
    popt.add_argument("run_dir")
    popt.add_argument("--impl", default="pace")
    popt.add_argument("--num-runs", dest="num_runs", type=int, default=10)

    prep = sub.add_parser("report")
    prep.add_argument("run_dir")

    args = p.parse_args(argv)
    from pace_agent.errors import PaceAgentError

    try:
        if args.cmd == "check":
            return cmd_check(args)
        if args.cmd == "bench" and args.bench_cmd == "tools":
            return cmd_tools_list(args) if args.list else cmd_bench_tools(args)
        if args.cmd == "profile":
            return cmd_examples_list(args) if args.list else cmd_profile(args)
        if args.cmd == "optimize":
            return cmd_optimize(args)
        if args.cmd == "report":
            return cmd_report(args)
    except PaceAgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
