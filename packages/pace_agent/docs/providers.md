# Provider tiers

Every PACE Agent tool has a readable `naive` reference and an idiomatic-NumPy `optimized` tier.
Above those, a tool can expose accelerated **provider** tiers. PACE Agent core carries no
provider-specific knowledge: providers live in their own packages and self-register.

## The tier model

- **`naive`** — the reference implementation (the parity baseline).
- **`optimized`** — a built-in idiomatic-NumPy candidate. When a tool does not provide one it
  transparently falls back to `naive`; PACE Agent reports that honestly as "same as naive" rather
  than a bogus ~1× speedup.
- **provider** (e.g. `pace`) — an accelerated implementation registered by an external
  package. The first provider is `pace`, routing to the amd-pace `torch.ops.pace.agentic_*`
  kernels when amd-pace is installed.

`PaceAgentTool.run(<tier>)` dispatches to the requested tier, or raises `SkipTool` when a provider
is not loaded / has no implementation for that op — **never a silent Python fallback**.

## Registering a provider

```python
from pace_agent import register_provider

def _fast_quantize(x):
    ...

register_provider("pace", "quantize", _fast_quantize, source="amd-pace", version="0.3.1")
```

Providers are discovered lazily via the `pace_agent.providers` entry-point group, so installing the
provider package is enough — no code change in PACE Agent. `source`/`version` are filled in from
the distribution automatically on discovery, or passed explicitly.

## Honest reporting

`pace-agent bench tools --list` labels each tier by kind, including provider provenance:

```
vector_search  [cpu-compute] tiers=naive (reference), optimized (candidate), pace (provider: amd-pace 0.3.1)
```

`PaceAgentTool.implementations()` returns the tier metadata programmatically
(`ImplementationInfo(kind, source, version, is_fallback, side_effect_free)`).

## Using a provider

```bash
pace-agent bench tools --tiers naive,optimized,pace      # microbench; pace column skips if unavailable
pace-agent optimize run/ --impl pace
```

`optimize` re-times only the ops the provider covers and projects the honest end-to-end (the
un-covered ops keep their naive time); it reports candidate coverage, and **zero candidate
coverage is reported as invalid** (never a provider speedup). See
[deterministic-replay.md](deterministic-replay.md) for candidate coverage and the replay
contract.
