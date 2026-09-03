# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import glob
import shutil
import subprocess

from pace_agent.bench.workloads.browser import _chrome_binary
from pace_agent.telemetry.uprof import INSTALL_URL as UPROF_INSTALL_URL


def _import_ok(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _paranoid():
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _uprof_cli():
    if shutil.which("AMDuProfCLI"):
        return "AMDuProfCLI"
    hits = glob.glob("/opt/AMDuProf*/bin/AMDuProfCLI")
    return hits[0] if hits else None


def _run(cmd):
    try:
        return subprocess.run(
            cmd.split(), capture_output=True, text=True, timeout=8
        ).stdout
    except Exception:
        return ""


def _llm_up(endpoint):
    if not endpoint:
        return None
    try:
        import urllib.request

        with urllib.request.urlopen(
            endpoint.rstrip("/") + "/v1/models", timeout=4
        ) as r:
            r.read(0)
        return True
    except Exception:
        return False


def run_checks(llm_endpoint=None):
    """Preflight: verify the install and any optional capabilities present."""
    checks = []

    def add(name, ok, detail, doc=None):
        checks.append({"name": name, "ok": ok, "detail": detail, "doc": doc})

    add("pace_agent", _import_ok("pace_agent"), "core package importable")
    add("numpy", _import_ok("numpy"), "required for tool baselines")
    try:
        import pace_agent.tools  # noqa: F401  (import triggers tool registration)
        from pace_agent.registry import list_tools

        n = len(list_tools())
        detail = f"{n} tools"
    except Exception as e:
        n = 0
        detail = f"registration failed: {e}"
    add("tools registered", n > 0, detail)

    add(
        "torch",
        _import_ok("torch"),
        "optional; required only by PyTorch-based providers",
    )
    add(
        "amd-pace kernels",
        _import_ok("pace"),
        "optional kernel package; no PACE Agent provider is bundled",
    )

    # Optional — accelerator libs for the library-backed optimized tiers. When absent, those
    # tiers honestly fall back to naive (reported as "same as naive"); never a phantom speedup.
    add(
        "orjson",
        _import_ok("orjson"),
        "fast JSON optimized tier (json_serialize/json_deserialize); pip install pace-agent[fast]",
    )
    add(
        "selectolax",
        _import_ok("selectolax"),
        "fast HTML optimized tier (html_parse); pip install pace-agent[fast]",
    )
    add(
        "re2",
        _import_ok("re2"),
        "linear-time regex optimized tier (regex_extract); pip install pace-agent[fast]",
    )

    # Optional — telemetry collection is opt-in; perf counters need perf_event_paranoid<=0.
    par = _paranoid()
    add("perf", bool(shutil.which("perf")), shutil.which("perf") or "not found")
    add("perf_event_paranoid<=0", par is not None and par <= 0, f"paranoid={par}")
    cli = _uprof_cli()
    add("uProf", cli is not None, cli or "AMDuProfCLI not found", UPROF_INSTALL_URL)
    add(
        "amd_uncore", "amd_uncore" in _run("lsmod"), "kernel module for uncore counters"
    )

    # Optional — the webvoyager browser workload needs selenium + a Chrome-for-Testing binary.
    add(
        "selenium", _import_ok("selenium"), "needed for the webvoyager browser workload"
    )
    chrome = _chrome_binary()
    add("chrome", chrome is not None, chrome or "browser binary not found")

    up = _llm_up(llm_endpoint)
    if up is not None:
        add("llm endpoint", up, llm_endpoint)
    return checks
