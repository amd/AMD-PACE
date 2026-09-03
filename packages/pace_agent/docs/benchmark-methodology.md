# Benchmark methodology

PACE Agent aims for measurements you can defend. This documents how it times, summarizes, and gates.

## Timing

`pace-agent bench tools` times each tier with a fixed protocol:

- **20 measured samples** and **3 warmups** per tier by default; `--repeat N` overrides the
  sample count.
- **Interleaving** — when comparing tiers, samples are taken in randomized interleaved order
  (seeded, deterministic) rather than one full block per tier, so slow frequency scaling or
  background load doesn't systematically bias one tier.
- A tier that merely falls back to `naive` is reported as `=naive`, not timed twice into a
  noisy ~1x row.

## Summaries and tail gating

`summarize()` reports n, mean, median, min, max, stdev, and CV. **Tail percentiles are gated
by sample size** because a percentile from a handful of samples is not meaningful:

- `n < 20` — no p95/p99;
- `n >= 20` — p95 (exploratory);
- `n >= 100` — p95 and p99.

## Confidence intervals

Speedups are noisy, so PACE Agent bootstraps confidence intervals (deterministic, seeded):

- `bootstrap_ci(samples)` — CI for the median;
- `speedup_ci(reference, candidate)` — CI for the median speedup.

`pace-agent bench tools` displays the speedup 95% CI, the sample count `n`, and the candidate CV
alongside each comparison, and flags an unstable candidate (CV above ~10%) with a `!` so a
one-off anomalous run is visible rather than mistaken for a stable result.

## Gating a change

`pace-agent optimize` answers "did the fix work?" within a single run: it re-times the covered ops
against the naive reference, and a speedup is reported **only** if every covered op passes its
parity check, the replay contract is valid, and candidate coverage is non-zero. Zero coverage
or a divergent replay is written as invalid — never a phantom speedup.
