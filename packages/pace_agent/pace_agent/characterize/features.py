# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Numeric features of a trajectory and robust scoring over a sample of them. Wall-clock
time is never a feature: selecting on it would fold the measured variable back into the
sample. Scoring is median/MAD, since a mean/stdev hides the runaway."""

import statistics

OUTLIER_Z = 3.5  # Iglewicz-Hoaglin; flagged, never dropped

_MAD_TO_SIGMA = 0.6745  # 0.75 quantile of the normal: makes MAD comparable to a stdev
_MEANAD_TO_SIGMA = 1.253314  # same role for the mean-absolute-deviation fallback

_METRIC_FEATURES = (
    "llm_calls",
    "tool_calls",
    "nodes_traversed",
    "context_tokens_total",
    "cached_tokens_total",
    "output_tokens_total",
    "reasoning_tokens_total",
)


def feature_vector(traj):
    """One run as ``{feature: value}``, omitting what the backend did not report."""
    rm = getattr(traj, "run_metrics", None) or {}
    out = {}
    for name in _METRIC_FEATURES:
        v = rm.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[name] = float(v)
    out.setdefault("llm_calls", float(len(getattr(traj, "llm_tape", None) or [])))
    out.setdefault("tool_calls", float(len(getattr(traj, "tools", None) or [])))
    return out


def feature_matrix(trajs):
    """``(names, rows)`` over the features every run reported. Intersected rather than
    zero-filled, since filling a gap with 0 would invent an outlier."""
    vectors = [feature_vector(t) for t in trajs]
    if not vectors:
        return [], []
    names = sorted(set.intersection(*(set(v) for v in vectors)))
    return names, [[v[n] for n in names] for v in vectors]


def e2e_samples(trajs):
    """Live wall-clock per run: reported, never used to choose the representative."""
    return [float(getattr(t, "e2e", 0.0) or 0.0) for t in trajs]


def modified_z(values):
    """Iglewicz-Hoaglin modified z-scores, falling back to the mean-absolute-deviation
    form when MAD is 0 (more than half the sample identical), then to zeros."""
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return [0.0] * len(xs)
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    if mad > 0:
        return [_MAD_TO_SIGMA * (x - med) / mad for x in xs]
    mean_ad = statistics.fmean([abs(x - med) for x in xs])
    if mean_ad > 0:
        return [(x - med) / (_MEANAD_TO_SIGMA * mean_ad) for x in xs]
    return [0.0] * len(xs)


def z_matrix(names, rows):
    """Per-feature modified z-scores, in the same shape as ``rows``."""
    if not names or not rows:
        return [[] for _ in rows]
    columns = [modified_z(col) for col in zip(*rows)]
    return [list(row) for row in zip(*columns)]


def scaled_distance(za, zb):
    """Mean absolute difference of two z-vectors, squashed to [0, 1) by ``d / (1 + d)``
    so it mixes with the edit-distance term on equal footing."""
    if not za or not zb:
        return 0.0
    d = statistics.fmean([abs(a - b) for a, b in zip(za, zb)])
    return d / (1.0 + d)


def outlier_report(names, rows, threshold=OUTLIER_Z):
    """Runs beyond ``threshold`` modified z, as ``[{run, features, max_abs_z}]``."""
    out = []
    for i, zrow in enumerate(z_matrix(names, rows)):
        flagged = {
            names[j]: {"value": rows[i][j], "z": round(z, 3)}
            for j, z in enumerate(zrow)
            if abs(z) > threshold
        }
        if flagged:
            max_z = round(max(abs(z) for z in zrow), 3)
            out.append({"run": i, "features": flagged, "max_abs_z": max_z})
    return out


def typicality(names, rows, index):
    """How ordinary run ``index`` is: its largest absolute modified z. 0 is perfect."""
    zs = z_matrix(names, rows)
    if not zs or index >= len(zs) or not zs[index]:
        return 0.0
    return round(max(abs(z) for z in zs[index]), 3)


def relative_range(values):
    """``(max - min) / |median|``, non-robust on purpose: on ``[9, 9, 1262, 9, 9]`` both MAD
    and the IQR are 0. The denominator floors at 1 since features are counts."""
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return 0.0
    return (max(xs) - min(xs)) / max(abs(statistics.median(xs)), 1.0)


def dispersion_report(names, rows):
    """Per-feature ``{median, min, max, rel_range}``: sizes the family that
    :func:`outlier_report` names the leavers of."""
    return {
        name: {
            "median": round(statistics.median(col), 3),
            "min": min(col),
            "max": max(col),
            "rel_range": round(relative_range(col), 3),
        }
        for name, col in zip(names, zip(*rows))
    }
