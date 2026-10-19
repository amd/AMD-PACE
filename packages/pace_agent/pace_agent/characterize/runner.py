# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Characterization: N live runs in, one representative trajectory out. Runs are
sequential, since concurrent ones would contend for the cores being measured."""

import json
import os
import time
from dataclasses import dataclass, field

from pace_agent.capture import CaptureContext
from pace_agent.characterize import distribution as dist_mod
from pace_agent.characterize import select as select_mod
from pace_agent.characterize.signature import ACTION, path_signature
from pace_agent.errors import CharacterizationFailed
from pace_agent.replay.recorder import Recorder

# Per-call counters a client accumulates; each live run must see only its own.
_LLM_COLUMNS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "reasoning_tokens",
)


@dataclass
class Characterization:
    """The sample, its shape, and the run chosen from it."""

    trajectories: list
    contexts: list
    distribution: object
    selection: object
    failures: list = field(default_factory=list)
    n_requested: int = 0
    elapsed: float = 0.0
    run_ids: list = field(default_factory=list)  # live-run number of each success

    @property
    def trajectory(self):
        return self.trajectories[self.selection.index]

    @property
    def context(self):
        return self.contexts[self.selection.index]

    def _run_id(self, i):
        # selection indexes successful runs; reports count live runs, as failures do
        return self.run_ids[i] if self.run_ids else i

    def summary(self):
        """Compact provenance block, small enough to travel inside the trajectory."""
        d, sel = self.distribution, self.selection
        outliers = [o["run"] for o in d.outliers()]
        stability, work = d.stability(), d.work_stability()
        return {
            "n_requested": self.n_requested,
            "n_runs": d.n,
            "n_failed": len(self.failures),
            "n_distinct_paths": d.n_distinct,
            "modal_share": round(d.modal_share, 4),
            "normalized_entropy": round(d.normalized_entropy(), 4),
            "stability": stability,
            "work_stability": work,
            "path_is_uninformative": d.path_is_uninformative(),
            "verdict": dist_mod.verdict(stability, work, sel.index in outliers),
            "path_source": d.source,
            "granularity": d.granularity,
            "selected_run": self._run_id(sel.index),
            "selected_path_digest": d.signatures[sel.index].digest,
            "strategy": sel.strategy,
            "applied": sel.applied,
            "reason": sel.reason,
            "typicality_max_abs_z": sel.typicality,
            "selected_is_outlier": sel.index in outliers,
            "outlier_runs": [self._run_id(r) for r in outliers],
            "characterize_s": round(self.elapsed, 3),
        }

    def report(self):
        """The full record written to characterization.json."""
        dist = self.distribution.as_dict()
        for c in dist["clusters"]:
            c["runs"] = [self._run_id(r) for r in c["runs"]]
        for o in dist["outliers"]:
            o["run"] = self._run_id(o["run"])
        selected = self.selection.index
        return {
            "cmd": "characterize",
            "summary": self.summary(),
            "selection": {**self.selection.as_dict(), "run": self._run_id(selected)},
            "distribution": dist,
            "failures": self.failures,
            "runs": [
                {
                    "run": self._run_id(i),
                    "path_digest": s.digest,
                    "n_steps": len(s),
                    "e2e_s": round(t.e2e, 4),
                    "run_metrics": t.run_metrics,
                    "selected": i == selected,
                }
                for i, (s, t) in enumerate(
                    zip(self.distribution.signatures, self.trajectories)
                )
            ],
        }

    def save(self, run_dir, keep_trajectories=False):
        """Write characterization.json; ``keep_trajectories`` also keeps every sampled run
        under ``raw/characterize/run_<i>/`` so another strategy can be tried for free.
        """
        path = os.path.join(run_dir, "characterization.json")
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2, default=str)
        if keep_trajectories:
            from pace_agent.rundir import RunDir

            root = os.path.join(RunDir(run_dir).raw_dir(), "characterize")
            for i, traj in enumerate(self.trajectories):
                traj.save(RunDir(os.path.join(root, f"run_{self._run_id(i)}")).path)
        return path


def characterize(
    agent,
    llm,
    agent_path=None,
    task=None,
    n=10,
    strategy=select_mod.AUTO,
    granularity=ACTION,
    min_success=1,
    structural_weight=select_mod.STRUCTURAL_WEIGHT,
    verbose=True,
):
    """Run ``agent`` live ``n`` times and return the :class:`Characterization`. A failed
    run is recorded and skipped; fewer than ``min_success`` successes is fatal."""
    n = max(1, int(n or 1))
    trajectories, contexts, columns, failures, run_ids = [], [], [], [], []
    t0 = time.perf_counter()
    for i in range(n):
        _announce(verbose, f"[characterize {i + 1}/{n}] live run starting")
        ctx = CaptureContext()
        try:
            label = f"characterization run {i + 1}/{n}"
            traj, cols = _one_run(agent, llm, agent_path, task, ctx, label)
        except Exception as e:  # noqa: BLE001 - a flaky live run must not end the sweep
            failures.append({"run": i, "error": f"{type(e).__name__}: {e}"[:200]})
            _announce(
                verbose, f"[characterize {i + 1}/{n}] failed: {type(e).__name__}: {e}"
            )
            continue
        trajectories.append(traj)
        contexts.append(ctx)
        columns.append(cols)
        run_ids.append(i)
        _announce(
            verbose,
            f"[characterize {i + 1}/{n}] done: "
            f"{traj.run_metrics.get('llm_calls')} llm call(s), "
            f"{traj.run_metrics.get('tool_calls')} op call(s), {traj.e2e:.1f}s",
        )

    need = max(1, int(min_success))
    if len(trajectories) < need:
        raise CharacterizationFailed(
            f"characterization needed {need} successful live run(s) of {n} but got "
            f"{len(trajectories)}"
            + (f"; first failure: {failures[0]['error']}" if failures else "")
        )

    signatures = [path_signature(t, granularity) for t in trajectories]
    distribution = dist_mod.build(signatures, trajectories)
    selection = select_mod.select(distribution, strategy, structural_weight)
    result = Characterization(
        trajectories=trajectories,
        contexts=contexts,
        distribution=distribution,
        selection=selection,
        failures=failures,
        n_requested=n,
        elapsed=time.perf_counter() - t0,
        run_ids=run_ids,
    )
    _set_llm_columns(llm, columns[selection.index])
    summary = result.summary()
    # stamped into the trajectory so the claim survives being read back on its own
    result.trajectory.meta["characterization"] = summary
    _announce(
        verbose,
        f"[characterize] selected run {summary['selected_run']}: {selection.reason} "
        f"({distribution.n_distinct} distinct path(s) in {distribution.n} run(s); "
        f"path {summary['stability']}, work {summary['work_stability']} "
        f"-> {summary['verdict']})",
    )
    return result


def _one_run(agent, llm, agent_path, task, ctx, label):
    """One live run: its trajectory and the per-call counters it produced."""
    from pace_agent.watchdog import beat, watch

    beat("characterization run starting")
    if hasattr(llm, "reset_session"):
        llm.reset_session()  # no conversation state may leak between live runs
    _set_llm_columns(llm, {})
    # a stall disarms the watchdog that caught it; a scope per run restores the enclosing
    # one on exit, so the next run is still guarded after this one is skipped
    with watch(label=label):
        traj = Recorder().record(agent, llm, agent_path, task=task, ctx=ctx)
    return traj, {name: list(getattr(llm, name)) for name in _llm_columns(llm)}


def _llm_columns(llm):
    return [n for n in _LLM_COLUMNS if isinstance(getattr(llm, n, None), list)]


def _set_llm_columns(llm, columns):
    """Make the client's per-call counters hold ``columns`` (empty when absent): ``profile``
    reads token growth off the client, so it must end up describing the selected run."""
    for name in _llm_columns(llm):
        getattr(llm, name)[:] = columns.get(name, [])


def _announce(verbose, message):
    if verbose:
        print(message, flush=True)
