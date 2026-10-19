# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Path signatures: the ordered steps one live run took, read from whatever the trajectory
carries (agent step record, then recorded tool calls, then the model tape). Extractors run
in descending priority and the first non-empty one wins."""

import hashlib
import re
from collections import Counter, namedtuple
from dataclasses import dataclass

# How much of a step distinguishes two paths.
ACTION = "action"  # step names only
ACTION_ARG = "action_arg"  # names plus a normalized argument
COARSE = "coarse"  # order-insensitive multiset of names
GRANULARITIES = (ACTION, ACTION_ARG, COARSE)

_MAX_ARG_CHARS = 64  # an arg is a discriminator, not a payload
_WS = re.compile(r"\s+")

Step = namedtuple("Step", "kind name arg")


@dataclass(frozen=True)
class PathSignature:
    """One run's path as comparable tokens; frozen so it can key the cluster map."""

    steps: tuple
    source: str
    granularity: str
    digest: str

    def __len__(self):
        return len(self.steps)


_EXTRACTORS = []  # [(priority, name, fn)], highest priority first


def register_step_extractor(fn, name, priority=200):
    """Register ``fn(trajectory) -> [Step, ...]`` (empty when it does not apply). Workload
    extractors belong above the built-ins (>100) to beat the generic fallbacks."""
    _EXTRACTORS.append((priority, name, fn))
    _EXTRACTORS.sort(key=lambda e: -e[0])  # stable: ties keep insertion order
    return fn


def extract_steps(traj):
    """The highest-priority non-empty step sequence, as ``(steps, source_name)``."""
    for _, name, fn in _EXTRACTORS:
        steps = fn(traj) or []
        if steps:
            return list(steps), name
    return [], "empty"


def path_signature(traj, granularity=ACTION):
    """Reduce ``traj`` to a :class:`PathSignature` at the requested granularity."""
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"unknown granularity {granularity!r}; expected one of {list(GRANULARITIES)}"
        )
    steps, source = extract_steps(traj)
    tokens = _tokens(steps, granularity)
    # granularity is hashed in so two granularities can never collide in a shared map
    blob = f"{granularity}\n" + "\n".join(tokens)
    digest = hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]
    return PathSignature(tokens, source, granularity, digest)


def _tokens(steps, granularity):
    if granularity == COARSE:  # sorted, so the multiset is canonical
        counts = Counter(f"{s.kind}:{s.name}" for s in steps)
        return tuple(f"{tok}*{counts[tok]}" for tok in sorted(counts))
    if granularity == ACTION_ARG:
        return tuple(
            f"{s.kind}:{s.name}({s.arg})" if s.arg else f"{s.kind}:{s.name}"
            for s in steps
        )
    return tuple(f"{s.kind}:{s.name}" for s in steps)


# Agents spell their step keys differently; accept the common ones.
_NAME_KEYS = ("action", "op", "tool", "node", "event", "name")
_ARG_KEYS = ("action_arg", "arg", "args", "label", "value", "target")


def _first(d, keys):
    return next((d[k] for k in keys if d.get(k) is not None), None)


def _steps_from_agent_result(traj):
    """A loop agent's own per-step record: the semantic path."""
    res = (getattr(traj, "meta", None) or {}).get("result")
    steps = res.get("per_step") if isinstance(res, dict) else None
    if not isinstance(steps, list):
        return []
    return [
        Step("act", str(name), _literal_token(_first(s, _ARG_KEYS)))
        for s in steps
        if isinstance(s, dict) and (name := _first(s, _NAME_KEYS)) is not None
    ]


def _steps_from_tools(traj):
    """Recorded PACE Agent tool calls, in order."""
    return [
        Step("tool", t.op, _shape_token(t.args, t.kw))
        for t in (getattr(traj, "tools", None) or [])
    ]


def _steps_from_llm_tape(traj):
    """Floor: one step per model call; the count still separates a runaway loop."""
    return [Step("llm", "call", "") for _ in (getattr(traj, "llm_tape", None) or [])]


register_step_extractor(_steps_from_agent_result, "agent_steps", priority=100)
register_step_extractor(_steps_from_tools, "tool_calls", priority=50)
register_step_extractor(_steps_from_llm_tape, "llm_calls", priority=10)


def _literal_token(value):
    """A discrete argument (a click label, a search string), literal but bounded."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(_literal_token(v) for v in value)
    return _WS.sub(" ", str(value)).strip()[:_MAX_ARG_CHARS]


def _shape_token(args, kw):
    """Type and size bucketed to a power of two, so two renders of the same page that
    differ by a few characters stay one path."""
    parts = [_one_shape(a) for a in (args or ())]
    parts += [f"{k}={_one_shape(kw[k])}" for k in sorted(kw or {})]
    return ",".join(parts)[:_MAX_ARG_CHARS]


def _one_shape(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}{tuple(shape)}"
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        n = len(value)
        return f"{type(value).__name__}~{1 << (n.bit_length() - 1) if n else 0}"
    return type(value).__name__
