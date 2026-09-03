# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import tempfile
from dataclasses import dataclass, field

from pace_agent.config import SCHEMA_VERSION
from pace_agent.replay import codec

# Trajectories are confidential (may contain prompts/outputs); result.json/report.md are
# shareable; trace.json is diagnostic. Consumers use this to decide what is safe to share.
ARTIFACT_CLASS = {
    "report.md": "shareable",
    "result.json": "shareable",
    "env.json": "shareable",
    "trace.json": "diagnostic",
    "trajectory.msgpack": "confidential",
    "trajectory.pkl": "confidential",
}

_REDACTORS = []


def register_redactor(fn):
    """Register ``fn(value) -> value`` applied to recorded LLM/tool values before persist
    (default: none). For scrubbing secrets/PII out of a trajectory prior to writing."""
    _REDACTORS.append(fn)


def _redact(v):
    for fn in _REDACTORS:
        v = fn(v)
    return v


@dataclass
class ToolCall:
    op: str
    bucket: str
    io_pattern: bool
    args: tuple
    kw: dict
    output: object
    naive_time: float


@dataclass
class Trajectory:
    tools: list = field(default_factory=list)
    llm_tape: list = field(default_factory=list)
    llm_time: float = 0.0
    e2e: float = 0.0
    shim_buckets: dict = field(default_factory=dict)
    run_metrics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def _for_persist(self):
        """Self, or a redacted copy when redactors are registered."""
        if not _REDACTORS:
            return self
        tools = [
            ToolCall(
                t.op,
                t.bucket,
                t.io_pattern,
                t.args,
                t.kw,
                _redact(t.output),
                t.naive_time,
            )
            for t in self.tools
        ]
        return Trajectory(
            tools=tools,
            llm_tape=[_redact(r) for r in self.llm_tape],
            llm_time=self.llm_time,
            e2e=self.e2e,
            shim_buckets=self.shim_buckets,
            run_metrics=self.run_metrics,
            meta=self.meta,
        )

    def save(self, run_dir):
        self.meta.setdefault("schema_version", SCHEMA_VERSION)
        path = os.path.join(run_dir, "trajectory.msgpack")
        _atomic_write(path, codec.dumps(self._for_persist()), mode=0o600)
        return path

    @staticmethod
    def load(run_dir, trust_pickle=False):
        mp = os.path.join(run_dir, "trajectory.msgpack")
        if os.path.exists(mp):
            with open(mp, "rb") as f:
                return codec.loads(f.read())
        pkl = os.path.join(run_dir, "trajectory.pkl")
        if os.path.exists(pkl):
            if not trust_pickle:
                raise ValueError(
                    f"refusing to load legacy pickle trajectory {pkl}: pickle executes "
                    f"arbitrary code on load (pass trust_pickle=True only if you trust "
                    f"its origin)."
                )
            import pickle  # nosec: explicit opt-in migration path only

            with open(pkl, "rb") as f:
                return pickle.load(f)  # nosec
        raise FileNotFoundError(f"no trajectory (.msgpack or .pkl) in {run_dir}")

    def summary(self):
        def shape(x):
            s = getattr(x, "shape", None)
            return (
                list(s)
                if s is not None
                else (len(x) if hasattr(x, "__len__") else None)
            )

        return {
            "e2e": self.e2e,
            "llm": {
                "n": len(self.llm_tape),
                "time": self.llm_time,
                "responses": [str(r)[:300] for r in self.llm_tape],
            },
            "run_metrics": self.run_metrics,
            "ops": [
                {
                    "op": o.op,
                    "bucket": o.bucket,
                    "io_pattern": o.io_pattern,
                    "naive_time": o.naive_time,
                    "in_shapes": [shape(a) for a in o.args],
                }
                for o in self.tools
            ],
            "meta": {k: v for k, v in self.meta.items() if k != "result"},
        }


def _atomic_write(path, data, mode=0o600):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-traj-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _toolcall_to_dict(tc):
    return {
        "op": tc.op,
        "bucket": tc.bucket,
        "io_pattern": tc.io_pattern,
        "args": list(tc.args),
        "kw": tc.kw,
        "output": tc.output,
        "naive_time": tc.naive_time,
    }


def _toolcall_from_dict(d):
    return ToolCall(
        d["op"],
        d["bucket"],
        d["io_pattern"],
        tuple(d["args"]),
        d["kw"],
        d["output"],
        d["naive_time"],
    )


def _trajectory_to_dict(t):
    return {
        "tools": t.tools,
        "llm_tape": t.llm_tape,
        "llm_time": t.llm_time,
        "e2e": t.e2e,
        "shim_buckets": t.shim_buckets,
        "run_metrics": t.run_metrics,
        "meta": t.meta,
    }


def _trajectory_from_dict(d):
    return Trajectory(
        tools=list(d.get("tools", [])),
        llm_tape=list(d.get("llm_tape", [])),
        llm_time=d.get("llm_time", 0.0),
        e2e=d.get("e2e", 0.0),
        shim_buckets=d.get("shim_buckets", {}),
        run_metrics=d.get("run_metrics", {}),
        meta=d.get("meta", {}),
    )


codec.register_codec(
    ToolCall, "pace_agent.ToolCall", _toolcall_to_dict, _toolcall_from_dict
)
codec.register_codec(
    Trajectory, "pace_agent.Trajectory", _trajectory_to_dict, _trajectory_from_dict
)
