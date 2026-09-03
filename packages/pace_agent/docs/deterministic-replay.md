# Deterministic replay

`pace-agent profile` records a run — every tool call (inputs, output, timing) and every LLM
response — into a trajectory. `pace-agent optimize` then **replays** that recording so the CPU work
can be re-measured under a different implementation tier **without re-hitting the model**.
Replay is driven entirely by the recorded data; PACE Agent core never imports a workload or example
to replay it.

## Replay modes

A trajectory records how it should be replayed in `meta["replay_mode"]`:

- **`deterministic`** (default) — op-replay: recorded tool calls are re-timed on their
  recorded inputs, tier by tier. Registered in core.
- **`browser-snapshot`** — for browser agents: recorded page snapshots are reloaded into a
  frozen `file://` DOM and the accessibility-tree observation is timed naive-vs-pace on the
  identical state (exact parity, reproducible timing). Registered by
  `pace_agent.replay.browser_snapshot`.

New modes register via `pace_agent.replay.modes.register_replay_mode(name, handler)`; `optimize`
looks the mode up rather than importing the workload that produced the trajectory.

## The replay contract (trust)

Replay is only trustworthy if it fails loudly when a trajectory no longer matches the agent.
On the first replay iteration the player validates a cursor over the recorded streams:

- tool-call **count**, and per position the op **name + bucket + argument signature**;
- no extra call (beyond what was recorded) and no missing call;
- the LLM tape is **fully consumed** (no leftover responses).

The first divergence raises a precise `ReplayDivergence` (e.g. *expected `vector_search`,
got `bm25`*) instead of a downstream `IndexError` or a silently-wrong number. `optimize`
catches it and writes the result as **invalid** — an invalid replay can never produce a
speedup.

The result carries a `replay_contract` block:

```json
{ "validated": true, "tool_calls_expected": 3, "tool_calls_consumed": 3,
  "llm_calls_expected": 2, "source_match": true, "valid": true }
```

## Source / config identity

The recorder stamps `meta["identity"]` (PACE Agent + Python versions, agent source digest) so
replay can detect drift. The compatibility policy is `strict | warn | allow` (default `warn`):
under `strict`, a source mismatch marks the replay non-equivalent (`valid: false`).

## Coverage (honest end-to-end)

A candidate tier (e.g. `pace`) rarely covers every op. `optimize` re-times **only** the
covered ops and projects the honest end-to-end: the un-covered ops keep their recorded naive
time (Amdahl-correct — a 100× win on 4% of the runtime is a ~4% end-to-end win, not "100×").

The result reports `coverage {total_calls, candidate_calls, covered_ops, covered_time_fraction}`
and a per-op table (reference vs candidate). **Zero candidate coverage is invalid, never a
phantom speedup**, and each covered op is parity-checked against the reference.
