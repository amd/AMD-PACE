# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Choosing which live run to record: always a real one, since an averaged trajectory is
not a tape any agent produced. Ties break on the lowest run index."""

from dataclasses import dataclass

from pace_agent.characterize import features as feat

MODE = "mode"
MEDOID = "medoid"
FIRST = "first"
AUTO = "auto"
STRATEGIES = (AUTO, MODE, MEDOID, FIRST)

# Path-vs-work split in the medoid distance. Half each: a matching path that burns eight
# times the tokens is not the same workload, and neither is the reverse.
STRUCTURAL_WEIGHT = 0.5


@dataclass
class Selection:
    """Which run was chosen, and the evidence for it."""

    index: int
    strategy: str  # what the caller asked for
    applied: str  # what decided it: auto resolves to mode or medoid
    reason: str
    typicality: float  # largest |modified z| of the chosen run; 0 is perfectly ordinary
    modal_share: float
    mean_distance: float = None  # medoid only

    def as_dict(self):
        d = {
            "run": self.index,
            "strategy": self.strategy,
            "applied": self.applied,
            "reason": self.reason,
            "typicality_max_abs_z": self.typicality,
            "modal_share": round(self.modal_share, 4),
        }
        if self.mean_distance is not None:
            d["mean_distance"] = round(self.mean_distance, 4)
        return d


def select(distribution, strategy=AUTO, structural_weight=STRUCTURAL_WEIGHT):
    """Pick the representative run of ``distribution`` and explain the pick."""
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown selection strategy {strategy!r}; expected one of {list(STRATEGIES)}"
        )
    n = distribution.n
    if n == 0:
        raise ValueError("cannot select a representative from zero runs")
    if n == 1:
        return _selection(distribution, 0, strategy, "single", "only one live run")
    if strategy == FIRST:
        return _selection(
            distribution, 0, strategy, FIRST, "first run, selection disabled"
        )
    applied = strategy
    if strategy == AUTO:
        # a modal count of 1 means no path repeated, so "most frequent" is a coin flip
        applied = MODE if distribution.modal.count > 1 else MEDOID
    if applied == MODE:
        return _by_mode(distribution, strategy)
    return _by_medoid(distribution, strategy, structural_weight)


def _by_mode(distribution, strategy):
    cluster = distribution.modal
    zs = feat.z_matrix(distribution.feature_names, distribution.feature_rows)
    # the path is already agreed inside the cluster, so choose on work alone
    index = _medoid_of(cluster.runs, lambda a, b: feat.scaled_distance(zs[a], zs[b]))
    reason = (
        f"most frequent path: {cluster.count} of {distribution.n} live runs "
        f"({cluster.share(distribution.n):.0%}) took it"
    )
    return _selection(distribution, index, strategy, MODE, reason)


def _by_medoid(distribution, strategy, structural_weight):
    n = distribution.n
    zs = feat.z_matrix(distribution.feature_names, distribution.feature_rows)
    sigs = distribution.signatures

    def dist(a, b):
        return structural_weight * normalized_edit(sigs[a].steps, sigs[b].steps) + (
            1.0 - structural_weight
        ) * feat.scaled_distance(zs[a], zs[b])

    index = _medoid_of(range(n), dist)
    mean_d = sum(dist(index, j) for j in range(n) if j != index) / (n - 1)
    reason = f"medoid of {n} live runs (mean distance {mean_d:.3f} to the rest)"
    return _selection(distribution, index, strategy, MEDOID, reason, mean_d)


def _medoid_of(indices, dist):
    """The member with the smallest total distance to the others; ties to lowest index."""
    members = sorted(indices)
    return min(members, key=lambda i: sum(dist(i, j) for j in members if j != i))


def _selection(distribution, index, strategy, applied, reason, mean_distance=None):
    return Selection(
        index=index,
        strategy=strategy,
        applied=applied,
        reason=reason,
        typicality=feat.typicality(
            distribution.feature_names, distribution.feature_rows, index
        ),
        modal_share=distribution.modal_share,
        mean_distance=mean_distance,
    )


def normalized_edit(a, b):
    """Levenshtein distance scaled to [0, 1]; order matters, so not set overlap."""
    return _levenshtein(a, b) / max(len(a), len(b)) if (a or b) else 0.0


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
