# Profiling any LangGraph agent

PACE Agent profiles the non-model (CPU) layer of any compiled [LangGraph](https://github.com/langchain-ai/langgraph)
graph with **one line** and no PACE Agent-specific code in your agent. LangGraph is pinned
(`langgraph==1.2.9`) so the instrumented surface is stable.

## One line

```python
from pace_agent import profile_invoke

result = profile_invoke(graph, {"question": "..."})
```

`profile_invoke` runs `graph.invoke(inputs)` with PACE Agent's node-timing callback injected,
returns the graph's normal result, and records per-node timings + shim (serialization)
attribution for the run. Ask for the full run handle to see the measurements:

```python
run = profile_invoke(graph, inputs, return_run=True)
run["result"]          # the graph output
run["node_timings"]    # {node_name: {median, p95, ..., count, total_ms}}
run["nodes_traversed"] # node count
run["shim_totals"]     # {bucket: seconds} auto-attributed serialization time
```

## Drop-in instrumentation

Wrap a compiled graph once and keep your existing call sites unchanged; every
`invoke`/`stream`/`ainvoke`/`astream`/`batch`/`abatch` is profiled and recorded on the proxy:

```python
from pace_agent import instrument

g = instrument(graph)
g.invoke(inputs)          # unchanged call site
g.pace_agent_runs[-1]["node_timings"]
```

## Enriching an existing config

`with_pace_agent(config)` merges PACE Agent's callback into a LangGraph config you already build
(existing callbacks, tags, thread-id and recursion-limit are preserved, not replaced):

```python
from pace_agent import with_pace_agent

graph.invoke(inputs, config=with_pace_agent(my_config, run_name="checkout"))
```

## Making an op tier-swappable

Node timing works for any graph. To also let PACE Agent **replay** a function under a faster tier
(and parity-check it), register it as an op — one decorator, no subclassing:

```python
from pace_agent import op

@op("rerank", bucket="retrieval")
def rerank(query, candidates):
    ...
```

Calls to a decorated function route through PACE Agent's capture hook, so the recorder captures its
inputs/outputs and the player can dispatch tiers on replay. Work that is *not* a registered op
still shows up as node time and unattributed CPU — visible, never guessed.

## From the CLI

`pace-agent profile <agent> --config run.json` profiles a brand-new graph with zero agent-side
code. The agent module may expose any of:

- `run(llm, task)` — the classic entry point;
- a compiled `graph` attribute;
- a `build(llm)` / `graph(llm)` factory (optionally an `inputs(task)` mapper).

When a graph/factory is given, PACE Agent drives it through `profile_invoke` and injects the
callback itself.

## Isolation

Each `profile_invoke` (or `instrument` call) owns a run-scoped capture context. LangGraph runs
nodes on a worker-thread pool that does not inherit `contextvars`; PACE Agent binds the context to
each node's worker thread, so **concurrent in-process runs never cross-record** each other's
node timings, tool calls, or serialization time.
