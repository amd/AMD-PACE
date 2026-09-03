# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import random
import statistics
import time


def _pct(sorted_s, q):
    n = len(sorted_s)
    return sorted_s[min(n - 1, int(round(q * (n - 1))))]


def summarize(samples):
    """Summary stats. Tail percentiles are gated by sample size (p95 needs >=20, p99 >=100)
    because a percentile from a handful of samples is not meaningful."""
    s = sorted(samples)
    n = len(s)
    if not n:
        return {"n": 0}
    med = statistics.median(s)
    sd = statistics.stdev(s) if n > 1 else 0.0
    out = {
        "n": n,
        "mean": statistics.fmean(s),
        "median": med,
        "min": s[0],
        "max": s[-1],
        "stdev": sd,
        "cv": (sd / med) if med else None,
    }
    if n >= 20:
        out["p95"] = _pct(s, 0.95)
    if n >= 100:
        out["p99"] = _pct(s, 0.99)
    return out


def bootstrap_ci(samples, stat=statistics.median, reps=2000, conf=0.95, seed=0):
    """Deterministic bootstrap confidence interval for `stat` over `samples`."""
    s = list(samples)
    if len(s) < 2:
        return None
    rng = random.Random(seed)
    n = len(s)
    boots = sorted(stat([s[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    lo = boots[int((1 - conf) / 2 * reps)]
    hi = boots[min(reps - 1, int((1 + conf) / 2 * reps))]
    return [lo, hi]


def speedup_ci(reference, candidate, reps=2000, conf=0.95, seed=0):
    """Deterministic bootstrap CI for the median speedup reference/candidate."""
    r, c = list(reference), list(candidate)
    if len(r) < 2 or len(c) < 2:
        return None
    rng = random.Random(seed)
    nr, nc = len(r), len(c)
    ratios = []
    for _ in range(reps):
        cm = statistics.median([c[rng.randrange(nc)] for _ in range(nc)])
        if cm:
            ratios.append(
                statistics.median([r[rng.randrange(nr)] for _ in range(nr)]) / cm
            )
    if not ratios:
        return None
    ratios.sort()
    lo = ratios[int((1 - conf) / 2 * len(ratios))]
    hi = ratios[min(len(ratios) - 1, int((1 + conf) / 2 * len(ratios)))]
    return [lo, hi]


def time_call(fn, args, kw, repeat, warmup=1):
    for _ in range(warmup):
        fn(*args, **kw)
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn(*args, **kw)
        out.append(time.perf_counter() - t)
    return out


def measure(fns, seed=0, samples=None, warmup=None, interleave=True):
    """Measure several tiers together; `fns` maps tier name -> zero-arg callable. Tiers are
    sampled in randomized interleaved order (unless interleave=False) so time-dependent machine
    state (frequency scaling, cache, background load) does not systematically bias one tier.
    `samples` and `warmup` default to the standard measurement values. Returns {tier: [samples]}.
    """
    n = 20 if samples is None else samples
    w = 3 if warmup is None else warmup
    names = list(fns)
    for fn in fns.values():
        for _ in range(w):
            fn()
    out = {k: [] for k in names}
    rng = random.Random(seed)
    for _ in range(n):
        batch = names[:]
        if interleave:
            rng.shuffle(batch)
        for tier in batch:
            t = time.perf_counter()
            fns[tier]()
            out[tier].append(time.perf_counter() - t)
    return out
