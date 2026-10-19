# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""The distribution a recorded run is supposed to represent, judged on two axes: ``path``
(same steps in the same order) and ``work`` (same amount of work). A fixed-graph agent
cannot vary its path, so there only ``work`` carries information."""

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

from pace_agent.characterize import features as feat
from pace_agent.stats import summarize

START = "__start__"
END = "__end__"

DOMINANT_SHARE = 0.5  # strict majority: at exactly half, as many disagreed as agreed
# Loose on purpose: real output lengths wander, and flagging a 10% spread would be noise.
WORK_SPREAD_TOLERANCE = 0.5

_PATH_OK = ("deterministic", "dominant")  # path verdicts that support one recording


@dataclass(frozen=True)
class PathCluster:
    """One distinct path and the runs that took it."""

    digest: str
    steps: tuple
    runs: tuple

    @property
    def count(self):
        return len(self.runs)

    def share(self, n):
        return (self.count / n) if n else 0.0

    def as_dict(self, n):
        return {
            "digest": self.digest,
            "count": self.count,
            "share": round(self.share(n), 4),
            "n_steps": len(self.steps),
            "runs": list(self.runs),
            "steps": list(self.steps),
        }


@dataclass
class PathDistribution:
    """Every live run's path, grouped and described. Nothing here selects."""

    signatures: list
    clusters: list  # most frequent first
    feature_names: list = field(default_factory=list)
    feature_rows: list = field(default_factory=list)
    e2e: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.signatures)

    @property
    def n_distinct(self):
        return len(self.clusters)

    @property
    def modal(self):
        return self.clusters[0] if self.clusters else None

    @property
    def modal_share(self):
        return self.modal.share(self.n) if self.modal else 0.0

    @property
    def source(self):
        return self.signatures[0].source if self.signatures else "empty"

    @property
    def granularity(self):
        return self.signatures[0].granularity if self.signatures else None

    def entropy(self):
        """Shannon entropy of the path distribution, in bits."""
        if self.n <= 1:
            return 0.0
        return -sum(
            c.count / self.n * math.log2(c.count / self.n) for c in self.clusters
        )

    def normalized_entropy(self):
        """Entropy scaled to [0, 1]: 0 when every run agreed, 1 when all differed."""
        return (self.entropy() / math.log2(self.n)) if self.n > 1 else 0.0

    def stability(self):
        """Path axis. ``deterministic`` only means no run took another route, which a
        fixed pipeline guarantees however the runs behaved."""
        if self.n == 0:
            return "empty"
        if self.n_distinct == 1:
            return "deterministic"
        if self.n_distinct == self.n:
            return "divergent"  # no path ever repeated
        return "dominant" if self.modal_share > DOMINANT_SHARE else "diverse"

    def work_stability(self):
        """Work axis: ``outlier`` beyond the modified-z cut, ``dispersed`` when nothing is
        extreme but the sample is still too wide, else ``uniform``."""
        if self.n < 2:
            return "single"
        if self.outliers():
            return "outlier"
        spread = self.work_dispersion().values()
        if any(f["rel_range"] > WORK_SPREAD_TOLERANCE for f in spread):
            return "dispersed"
        return "uniform"

    def work_dispersion(self):
        return feat.dispersion_report(self.feature_names, self.feature_rows)

    def path_is_uninformative(self):
        """Every run took one path, so the case rests on the work axis."""
        return self.n > 1 and self.n_distinct == 1

    def markov(self):
        """Step-to-step transitions; a state with split outgoing probabilities is where
        the runs diverged."""
        counts = Counter(
            pair for s in self.signatures for pair in pairwise([START, *s.steps, END])
        )
        totals = Counter()
        for (a, _), c in counts.items():
            totals[a] += c
        return {
            "states": sorted({state for pair in counts for state in pair}),
            "transitions": [
                {"from": a, "to": b, "count": c, "prob": round(c / totals[a], 4)}
                for (a, b), c in sorted(counts.items())
            ],
        }

    def numeric(self):
        """Per-feature spread, plus ``e2e_s``: reported for context, never selected on."""
        out = {
            name: summarize([row[j] for row in self.feature_rows])
            for j, name in enumerate(self.feature_names)
        }
        if self.e2e:
            out["e2e_s"] = summarize(self.e2e)
        return out

    def outliers(self):
        return feat.outlier_report(self.feature_names, self.feature_rows)

    def as_dict(self):
        return {
            "n_runs": self.n,
            "n_distinct_paths": self.n_distinct,
            "path_source": self.source,
            "granularity": self.granularity,
            "modal_share": round(self.modal_share, 4),
            "entropy_bits": round(self.entropy(), 4),
            "normalized_entropy": round(self.normalized_entropy(), 4),
            "stability": self.stability(),
            "work_stability": self.work_stability(),
            "path_is_uninformative": self.path_is_uninformative(),
            "path_length": summarize([len(s) for s in self.signatures]),
            "clusters": [c.as_dict(self.n) for c in self.clusters],
            "numeric": self.numeric(),
            "work_dispersion": self.work_dispersion(),
            "outliers": self.outliers(),
            "markov": self.markov(),
        }


# Worst first; this is the precedence ``verdict`` applies.
UNSAMPLED = "unsampled"
UNREPRESENTATIVE = "unrepresentative"
NO_MAJORITY = "no-majority"
WIDE = "wide"
REPRESENTATIVE = "representative"


def verdict(path_stability, work_stability, selected_is_outlier):
    """How strong "the recorded run is representative" is: the weaker axis decides."""
    if work_stability == "single" or path_stability == "empty":
        return UNSAMPLED
    if selected_is_outlier:
        return UNREPRESENTATIVE
    if path_stability not in _PATH_OK:
        return NO_MAJORITY
    if work_stability != "uniform":
        return WIDE
    return REPRESENTATIVE


def build(signatures, trajectories=None):
    """Group ``signatures`` into a :class:`PathDistribution`. Clusters sort by count, then
    first appearance, then digest, so an identical sample always picks the same modal.
    """
    grouped = {}
    for i, s in enumerate(signatures):
        grouped.setdefault(s.digest, (s, []))[1].append(i)
    clusters = [
        PathCluster(digest=d, steps=s.steps, runs=tuple(runs))
        for d, (s, runs) in grouped.items()
    ]
    clusters.sort(key=lambda c: (-c.count, c.runs[0], c.digest))
    names, rows = feat.feature_matrix(trajectories or [])
    return PathDistribution(
        signatures=list(signatures),
        clusters=clusters,
        feature_names=names,
        feature_rows=rows,
        e2e=feat.e2e_samples(trajectories or []),
    )
