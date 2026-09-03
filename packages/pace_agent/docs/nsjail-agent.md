# nsjail-agent-tool

PACE Agent nsjail sandbox guide (`packages/pace_agent/docs/nsjail-agent.md`). This is the
canonical documentation for the vendored `nsjail_agent` package (formerly
`nsjail_agent/README.md`).

> **Secure code execution sandbox for AI agents, powered by [nsjail](https://github.com/google/nsjail).**

The vendored package lets LLM agents write and run code in an isolated Linux
namespace — with enforced CPU, memory, network, and syscall policies — and
exposes the executor as a standard agent tool for LangGraph, OpenAI function
calling, and CAMEL.

**PACE Agent uses the in-process Python SDK** (`CodeExecutor`) directly. GAIA and the
`sandbox_tool_agent` example call it from the same process as the planner; there
is no Docker deployment and no requirement to run the optional REST API.

---

## Features

| Feature | Details |
|---|---|
| **Linux namespace isolation** | PID, mount, network, user, IPC, UTS namespaces via nsjail |
| **Resource limits** | Configurable CPU time, wall time, memory, PIDs, file size |
| **Network policy** | Off by default; loopback-only mode available |
| **Multi-language** | Python, Bash, JavaScript (Node.js), Go |
| **Framework adapters** | LangChain/LangGraph, OpenAI function calling, CAMEL |
| **REST API (optional)** | FastAPI server with `/execute` and `/schema/openai-tool` |

---

## Quick Start (PACE Agent)

Install PACE Agent from this repository (editable install pulls in the vendored
`nsjail_agent` package):

```bash
pip install -e ./packages/pace_agent
```

Build or install the [nsjail](https://github.com/google/nsjail) binary, then
point PACE Agent at it:

```bash
export NSJAIL_BIN=/path/to/nsjail    # or ensure `nsjail` is on PATH
```

Run a smoke test through the PACE Agent example agent:

```bash
python -m pace_agent.examples.gaia_agent_example --task-id gaia_015
```

Or wire the executor into your own agent loop via
[`pace_agent.examples.sandbox_tool_agent`](../pace_agent/examples/sandbox_tool_agent.py).

Without nsjail, code-tool calls fail closed. For development only,
`--allow-local-executor` on `run_gaia` enables unsandboxed host execution and
must not be used with untrusted model output.

### Host packages for attachment analysis

GAIA tasks often need spreadsheet, image, OCR, and PDF tooling inside the
sandbox. Install common host packages — `/usr` is mounted read-only into the
jail by default:

```bash
sudo apt update
sudo apt install \
  unzip zip \
  python3-pandas python3-openpyxl python3-pil python3-numpy \
  python3-opencv python3-skimage python3-pytesseract \
  python3-pypdf python3-pdfminer \
  tesseract-ocr tesseract-ocr-eng imagemagick poppler-utils
```

If your apt repo lacks a Python wrapper, install the system tools with apt and
add wrappers to system Python with pip:

```bash
sudo apt install python3-pip tesseract-ocr tesseract-ocr-eng poppler-utils
/usr/bin/python3 -m pip install --break-system-packages pytesseract pypdf pdfminer.six
```

For packages in a custom Python prefix, point the sandbox at that interpreter
and mount the prefix read-only:

```bash
export NSJAIL_PYTHON_BIN=/path/to/your/python
export NSJAIL_EXTRA_RO_MOUNTS=/path/to/your/prefix
```

Multiple extra mounts are comma-separated. nsjail `source:dest` syntax is
supported inside each entry:

```bash
export NSJAIL_EXTRA_RO_MOUNTS=/opt/conda,/host/data:/data
```

---

## Usage

### Python SDK (primary PACE Agent path)

```python
from nsjail_agent import CodeExecutor, ExecutionRequest, Language

executor = CodeExecutor()
result = executor.execute_sync(
    ExecutionRequest(code="print('hello from sandbox')", language=Language.PYTHON)
)
print(result.stdout)   # hello from sandbox
print(result.status)   # ExecutionStatus.SUCCESS
```

### LangGraph

```python
from nsjail_agent.tool_adapter import make_langchain_tool

execute_code = make_langchain_tool()   # drop-in LangChain Tool
```

See [`pace_agent.examples.nsjail_langgraph_example`](../pace_agent/examples/nsjail_langgraph_example.py) for the full ReAct graph.

### OpenAI function calling

```python
from nsjail_agent.tool_adapter import OPENAI_TOOL_SCHEMA, OpenAIToolAdapter

adapter = OpenAIToolAdapter()
# Pass OPENAI_TOOL_SCHEMA to client.chat.completions.create(tools=[...])
# Call adapter(tool_call.function.arguments) to execute
```

The reusable execution-agent loop lives in `nsjail_agent/agent.py`. By
default it forces every tool-capable assistant turn to call `execute_code`, then
asks for one final no-tool answer after `max_turns`. The GAIA sample runner is
`python -m pace_agent.examples.gaia_agent_example` and loads tasks from
`packages/pace_agent/configs/gaia_sample_tasks.json` (attachments under
`pace_agent/examples/fixtures/gaia/`); pass `--allow-auto-tool-choice`
to let the model decide whether to call tools during intermediate turns.

### CAMEL

```python
from nsjail_agent.tool_adapter import CAMELToolAdapter

tool = CAMELToolAdapter()
agent = ChatAgent(..., tools=[tool])
```

See [`pace_agent.examples.nsjail_camel_example`](../pace_agent/examples/nsjail_camel_example.py).

### REST API (optional)

The vendored package includes a FastAPI server for standalone deployments. PACE Agent
GAIA runs do **not** use this path — they call `CodeExecutor` in-process. To run
the API locally for testing:

```bash
python -m nsjail_agent.api          # listens on :8080
```

```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(sum(range(101)))", "language": "python"}'
```

---

## Architecture

```
ExecutionRequest
      │
      ▼
 CodeExecutor          ← orchestrates everything
      │
      ├── Workspace    ← creates temp dir, writes files
      ├── sandbox.py   ← builds nsjail argv
      └── ResultVerifier ← classifies exit code / OOM / seccomp
```

### Key modules

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic request/response schemas |
| `workspace.py` | Ephemeral temp-dir lifecycle |
| `sandbox.py` | nsjail command builder |
| `executor.py` | Async/sync execution orchestration |
| `agent.py` | Reusable OpenAI-compatible execution-agent loop |
| `verifier.py` | Exit-code classification, OOM/seccomp detection |
| `tool_adapter.py` | Framework-specific tool wrappers |
| `api.py` | FastAPI REST interface (optional) |

Source lives under `packages/pace_agent/nsjail_agent/`.

---

## Configuration

`ExecutionRequest` accepts nested `ResourceLimits` objects. For the REST API,
sandbox policy is server-owned and request-provided sandbox settings are
normalized before execution.

```python
from nsjail_agent.models import ResourceLimits

req = ExecutionRequest(
    code="...",
    resources=ResourceLimits(max_cpu_seconds=5, max_memory_mb=128),
)
```

Environment knobs:

| Variable | Purpose |
|---|---|
| `NSJAIL_SANDBOX_CONFIG` | Path to sandbox path defaults JSON (default: `configs/nsjail_sandbox.json`) |
| `NSJAIL_SANDBOX_WORKDIR` | In-jail workspace mount point (default: `/sandbox`) |
| `NSJAIL_SANDBOX_DEPS_MOUNT` | In-jail pip deps mount point (default: `/sandbox_deps`) |
| `NSJAIL_SANDBOX_STOCKFISH_PATH` | In-jail Stockfish binary path (default: `/opt/stockfish`) |
| `NSJAIL_BIN` | Path to the nsjail binary (default: `nsjail` on PATH, then `/usr/sbin/nsjail`) |
| `NSJAIL_PYTHON_BIN` | Python interpreter inside the jail |
| `NSJAIL_EXTRA_RO_MOUNTS` | Comma-separated extra read-only bind mounts |
| `NSJAIL_AGENT_API_KEY` | If set, REST `/execute` requires `Authorization: Bearer <key>` or `X-API-Key` |
| `NSJAIL_AGENT_MAX_CONCURRENT` | REST API concurrency cap (default `4`; invalid/<1 values clamped) |
| `NSJAIL_AGENT_CORS_ORIGINS` | Comma-separated browser origins allowed for CORS (default: disabled) |

Default in-jail paths live in `configs/nsjail_sandbox.json` and can also be overridden from an
PACE Agent run config under the top-level `"sandbox"` block (see `configs/gaia_run.example.json`).

---

## Development

```bash
pip install -e ./packages/pace_agent
python -m unittest discover -s tests_agents/nsjail_agent_tests -v
```

Integration tests require nsjail on PATH and are skipped when it is absent.

---

## Security Notes

- nsjail requires `CAP_SYS_ADMIN` (or running as root) to create user namespaces.
- Network is disabled by default — enable cautiously.
- Do **not** expose the REST API to untrusted networks without authentication.
- CORS is disabled by default; set `NSJAIL_AGENT_CORS_ORIGINS` only when browser clients need cross-origin access.
- The REST API does not use `LocalExecutor`; if nsjail is unavailable, execution fails closed.
- `LocalExecutor` (dev fallback) runs **Python only** without sandboxing; bash/JS/Go require nsjail.
- Input and returned file paths are restricted to the ephemeral sandbox workspace.
- `--allow-local-executor` bypasses the sandbox entirely — development only.

See also **[gaia-planner.md](gaia-planner.md)** for the full GAIA runner setup.
