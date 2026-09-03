# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Approximate answer scoring for task suites that ship an expected answer.

Scoring lives in the harness, never in the agent: `profile` reads the expected answer out of
the task, removes it from the copy the agent receives, and compares it against the recorded
answer afterwards. Neither matcher is the official metric of any published benchmark — both are
sanity signals, so the raw answer and expected value are always reported alongside the verdict.

Both matchers always run and both verdicts are always reported, because neither is the answer on
its own: strict is a floor (an agent that answers in a sentence fails it even when it is right)
and loose is a ceiling (containment credits an expected "17" to an answer of "17000"). Quoting
one number hides that spread, so nothing here picks a winner.

A missing or null expected answer scores as `None` ("unavailable"), never as a failure, so a
partially annotated suite cannot be mistaken for a suite the agent got wrong.
"""

import re

from pace_agent.errors import ConfigError

DEFAULT_GOLD_KEY = "gold"


_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


def normalize(s):
    """Casefold and drop the formatting that carries no answer content.

    A thousands separator is deleted so "$1,700" and "1700" are the same answer, while a comma
    between words stays a separator so "a, b" and "a,b" both read as two items rather than one
    run-together token. Public so GAIA (and other suites) can reuse the same harness
    normalization without reimplementing it.
    """
    s = str(s).lower().replace("$", "").replace("%", "")
    return re.sub(r"[,\s]+", " ", _THOUSANDS.sub("", s)).strip()


def _numbers(s):
    return re.findall(r"-?\d+\.?\d*", s)


def match_strict(answer, gold):
    """Normalized equality: insensitive to case, whitespace and thousands/currency separators,
    but the whole answer must be the expected value."""
    return normalize(answer) == normalize(gold)


def match_loose(answer, gold):
    """`match_strict`, plus containment and a first-number comparison — tolerant of an agent
    that answers in a sentence. Containment can over-credit (an expected "11.2" is contained in
    "111.25"), which is why it is never reported on its own."""
    a, g = normalize(answer), normalize(gold)
    if not g:
        return False
    if g == a or g in a:
        return True
    an, gn = _numbers(a), _numbers(g)
    if an and gn:
        try:
            gv = float(gn[0])
            return any(abs(float(x) - gv) < 1e-6 for x in an)
        except ValueError:
            pass
    return False


def resolve(scoring):
    """Read the run config's `scoring` block: `{gold_key}`. Returns the gold key, or None when the
    config asks for no scoring. An unusable gold key is rejected here, since a config that asks
    for grading and silently gets none is worse than one that fails to load."""
    if not scoring:
        return None
    if not isinstance(scoring, dict):
        raise ConfigError(f"scoring must be an object, got {type(scoring).__name__}")
    gold_key = scoring.get("gold_key", DEFAULT_GOLD_KEY)
    if not isinstance(gold_key, str) or not gold_key.strip():
        raise ConfigError(
            f"scoring.gold_key must be a non-empty string naming the task field that holds the "
            f"expected answer (default {DEFAULT_GOLD_KEY!r}), got {gold_key!r}"
        )
    return gold_key


def score(answer, gold):
    """Both verdicts for one answer.

    Only a missing expected answer is unscorable (None). A task that *has* an expected answer but
    produced no answer at all — it crashed, stalled, or lost its worker — scores as a miss, so
    failures stay in the accuracy denominator instead of quietly flattering the run.
    """
    if gold is None or not normalize(gold):
        return {"gold": gold, "match_strict": None, "match_loose": None}
    if answer is None:
        return {"gold": gold, "match_strict": False, "match_loose": False}
    return {
        "gold": gold,
        "match_strict": match_strict(answer, gold),
        "match_loose": match_loose(answer, gold),
    }


def aggregate(rows):
    """Suite-level counts over per-task rows. `n_scored` excludes tasks with no expected answer,
    so each `auto_match_*`/`n_scored` is the accuracy over the annotated subset only."""
    scored = [r for r in rows if r.get("match_strict") is not None]
    return {
        "n_scored": len(scored),
        "n_unscored": len(rows) - len(scored),
        "auto_match_strict": sum(1 for r in scored if r.get("match_strict")),
        "auto_match_loose": sum(1 for r in scored if r.get("match_loose")),
    }
