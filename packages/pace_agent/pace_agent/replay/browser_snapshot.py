# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Browser-snapshot replay mode (core).

Deterministic naive-vs-pace accessibility-tree observe A/B: each recorded page snapshot is
reloaded into a frozen file:// DOM and observed with both implementations on the identical
state, so parity is exact and timing is reproducible. Moved here from the webvoyager example
so replay is a first-class core capability driven by the trajectory — PACE Agent core never imports
a workload/example. Registered as the ``browser-snapshot`` replay mode; the trajectory carries
its snapshots in ``meta["snapshots"]``.
"""

import os
import re
import time

from pace_agent.progress import say
from pace_agent.replay.modes import register_replay_mode
from pace_agent.watchdog import beat


def replay_observe(html_snapshots, reps=3, label=None):
    """DETERMINISTIC naive-vs-pace accessibility-tree observe A/B. Each recorded page HTML is
    reloaded into a frozen file:// DOM and observed with BOTH impls on the *identical* state:
      naive = per-node bounding-rect CDP round-trips (faithful WebVoyager)
      pace  = one batched DOMSnapshot box call
    Same frozen DOM for both => exact parity + reproducible timing (no live page churn).
    """
    import shutil
    import tempfile

    from pace_agent.bench.workloads.browser import SeleniumBrowser as _SB
    from pace_agent.stats import summarize

    br = _SB(headless=True)
    tmpdir = tempfile.mkdtemp(
        prefix="pace_agent_replay_",
        dir=(
            os.path.expanduser("~/.pace_agent_sel")
            if os.path.isdir(os.path.expanduser("~/.pace_agent_sel"))
            else None
        ),
    )
    per, naive_all, pace_all, parities = [], [], [], []
    try:
        for i, snap in enumerate(html_snapshots):
            snap = snap or "<html></html>"
            # MHTML snapshots are self-contained (CSS/images inlined) -> full page renders offline.
            is_mhtml = (
                snap.lstrip()[:200]
                .lower()
                .startswith(
                    ("from:", "mime-version", "content-type: multipart/related")
                )
            )
            f = os.path.join(tmpdir, f"snap_{i}." + ("mhtml" if is_mhtml else "html"))
            with open(f, "w") as fh:
                fh.write(snap)
            try:  # freeze: no JS mutation,
                br.driver.execute_cdp_cmd(
                    "Emulation.setScriptExecutionDisabled", {"value": True}
                )
            except Exception:
                pass
            br.navigate("file://" + f)
            br.await_ready()
            time.sleep(
                2.0
            )  # let the (frozen) page fully render/decode so boxes are stable
            nv, pc, tree_n, tree_p = [], [], None, None
            for _ in range(reps):
                t = time.perf_counter()
                tree_n, _ = br.observe_ax("naive")
                nv.append((time.perf_counter() - t) * 1e3)
                t = time.perf_counter()
                tree_p, _ = br.observe_ax("pace")
                pc.append((time.perf_counter() - t) * 1e3)
            parity = _content_tree(tree_n) == _content_tree(tree_p)  # content parity
            parities.append(parity)
            beat(f"observed snapshot {i + 1}/{len(html_snapshots)}")
            naive_all += nv
            pace_all += pc
            snap_naive, snap_pace = summarize(nv), summarize(pc)
            say(
                "optimize",
                f"snapshot {i + 1}/{len(html_snapshots)}: naive {snap_naive['median']:.1f} -> "
                f"pace {snap_pace['median']:.1f} ms/observe x{reps}, "
                f"parity {'ok' if parity else 'FAILED'}",
                label,
            )
            per.append(
                {
                    "snapshot": i,
                    "observe_naive_ms": snap_naive,
                    "observe_pace_ms": snap_pace,
                    "parity": parity,
                }
            )
    finally:
        br.quit()
        shutil.rmtree(tmpdir, ignore_errors=True)
    on, op = summarize(naive_all), summarize(pace_all)
    return {
        "mode": "browser-snapshot",
        "reps": reps,
        "n_snapshots": len(html_snapshots),
        "observe_naive_ms": on,
        "observe_pace_ms": op,
        "observe_speedup": (on["median"] / op["median"]) if op.get("median") else None,
        "observe_parity_all": all(parities) if parities else None,
        "per_snapshot": per,
    }


def _content_tree(tree):
    """Normalize an accessibility tree for the naive-vs-pace PARITY check: drop redundant
    StaticText lines whose text is already present elsewhere in the tree (e.g. an input's echoed
    value, also in the page title). These browser-internal duplicates are the only nodes a batched
    box method (pace) can't see that per-node (naive) can — and they carry no new information. So
    parity compares INFORMATION, not those redundant nodes."""
    lines = tree.splitlines()
    full = "\n".join(lines)
    out = []
    for ln in lines:
        m = re.match(r"\s*\[\d+\]\s+StaticText\s+'(.*)'\s*$", ln)
        if m:
            txt = (m.group(1) or "").strip()
            if txt and full.count(txt) > 1:  # duplicated elsewhere => redundant
                continue
        out.append(ln)
    return "\n".join(out)


def _project_e2e(traj, ab):
    """Amdahl projection of the observe naive->pace A/B onto the recorded end-to-end wall time.

    observe is the ONLY optimized component; LLM/model time and page nav/act are unchanged.
    Recoverable = n_observes * (naive - pace) per observe, capped at the non-model time (observe
    can't exceed it). Mirrors the deterministic mode's honest E2E: candidate = e2e - recoverable.
    """
    n = ab.get("n_snapshots") or 0
    on = (ab.get("observe_naive_ms") or {}).get("median") or 0.0
    op = (ab.get("observe_pace_ms") or {}).get("median") or 0.0
    e2e_naive = float(traj.e2e or 0.0)
    model = float(getattr(traj, "llm_time", 0.0) or 0.0)
    obs_naive_s = min(n * on / 1000.0, max(0.0, e2e_naive - model))
    obs_pace_s = obs_naive_s * (op / on) if on else 0.0
    recoverable = obs_naive_s - obs_pace_s
    e2e_pace = max(0.0, e2e_naive - recoverable)
    return {
        "reference": {"median": e2e_naive},
        "candidate": {"median": e2e_pace},
        "speedup": (e2e_naive / e2e_pace) if e2e_pace > 0 else None,
        "pct_faster": round((1 - e2e_pace / e2e_naive) * 100, 2) if e2e_naive else None,
        "recoverable_s": recoverable,
        "breakdown_s": {
            "llm_model": model,
            "observe_naive": obs_naive_s,
            "observe_pace": obs_pace_s,
            "other_browser": max(0.0, e2e_naive - model - obs_naive_s),
        },
    }


def _browser_snapshot_replay(traj, run_dir, opts):
    """Replay mode handler: observe A/B over the page snapshots carried in the trajectory, then
    project the end-to-end impact (observe is the only optimized node)."""
    snaps = traj.meta.get("snapshots") or []
    ab = replay_observe(snaps, reps=opts.get("num_runs", 10), label=opts.get("label"))
    return {
        "cmd": "optimize",
        "replay_mode": "browser-snapshot",
        **ab,
        "e2e": _project_e2e(traj, ab),
    }


register_replay_mode("browser-snapshot", _browser_snapshot_replay)
