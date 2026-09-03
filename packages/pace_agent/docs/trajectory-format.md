# Trajectory format

A trajectory is the recording of a run — tool calls (inputs, output, timing), the LLM tape,
shim totals, run metrics, and metadata — that makes deterministic replay possible. It is
written as `trajectory.msgpack`.

## Safe by default: msgpack, not pickle

Trajectories are serialized with **msgpack** through a typed codec registry
(`pace_agent.replay.codec`), never pickle:

- native msgpack handles `str/int/float/bool/None/list/dict/bytes`;
- numpy arrays (dtype + shape + raw bytes) and scalars are handled;
- dataclasses like `ToolCall`/`Trajectory` register their own codecs;
- **any other type fails loudly** with `UnsupportedType` (naming the type and how to register
  a codec) — never a silent, unsafe fallback.

Register a codec for your own type:

```python
from pace_agent.replay import codec
codec.register_codec(MyType, "mypkg.MyType", to_dict=..., from_dict=...)
```

Loading a legacy `trajectory.pkl` is **refused** unless you opt in (`trust_pickle=True`),
because unpickling executes arbitrary code.

## On-disk safety

- written **atomically** (temp file + `os.replace`) with **`0600`** permissions;
- a `schema_version` is stamped in `meta`; consumers reject unknown major versions;
- a **redaction hook** (`register_redactor(fn)`) can scrub recorded LLM/tool values before
  persist (default: none);
- **artifact classification** — `report.md` / `result.json` are shareable, `trace.json` is
  diagnostic, `trajectory.msgpack` is confidential.

## Per-value descriptors

The codec can emit a compact descriptor per value (`{type, shape, dtype, digest}`), which the
strict replay contract uses to detect that a trajectory no longer matches the agent (see
[deterministic-replay.md](deterministic-replay.md)).
