# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import csv
import glob
import os
import shutil
import subprocess
import tempfile
import time

INSTALL_URL = "https://www.amd.com/en/developer/uprof.html"


def cli():
    return shutil.which("AMDuProfCLI") or next(
        iter(glob.glob("/opt/AMDuProf*/bin/AMDuProfCLI")), None
    )


def available():
    return cli() is not None


def _reap(work):
    """Kill the workload and reap it, so a failed/aborted collect never leaks a zombie."""
    work.kill()
    try:
        work.wait(timeout=5)
    except Exception:
        pass


def collect(argv, config="cpi", duration=15, attach_delay=0.4, timeout=900):
    exe = cli()
    if not exe:
        return {
            "available": False,
            "reason": f"AMDuProfCLI not found; install AMD uProf: {INSTALL_URL}",
        }
    out = os.path.join(tempfile.mkdtemp(prefix="pace_agent_uprof_"), "s")
    work = subprocess.Popen(argv)
    time.sleep(attach_delay)
    if work.poll() is not None:
        return {"available": False, "reason": "workload exited before uProf attached"}
    cmd = [
        exe,
        "collect",
        "--config",
        config,
        "-o",
        out,
        "-p",
        str(work.pid),
        "-d",
        str(duration),
    ]
    try:
        col = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        _reap(work)
        return {"available": False, "reason": str(e)}
    if col.returncode != 0:
        _reap(work)  # collect failed: don't block on / leak the still-running workload
        return {
            "available": False,
            "reason": (col.stderr or col.stdout or "collect failed").strip()[:300],
        }
    try:
        work.wait(timeout=timeout)
    except Exception:
        _reap(work)
    # collect writes to out/AMDuProf-<ts>/; `report -i` needs THAT session dir (not the -o
    # parent) and writes report.csv inside it — pointing at `out` yields no report.
    session_dir = next(iter(glob.glob(os.path.join(out, "AMDuProf-*"))), out)
    rep = subprocess.run(
        [exe, "report", "-i", session_dir],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    report = os.path.join(session_dir, "report.csv")
    metrics = _parse(report)
    if rep.returncode != 0 or not metrics:
        # a successful collect but an empty/failed report must not read as success
        return {
            "available": False,
            "reason": (
                rep.stderr or rep.stdout or "uProf report produced no parseable metrics"
            ).strip()[:300],
            "config": config,
            "session": session_dir,
        }
    return {
        "available": True,
        "config": config,
        "session": session_dir,
        "metrics": metrics,
    }


def _parse(report):
    if not os.path.exists(report):
        return {}
    header, row = None, None
    with open(report, errors="ignore") as f:
        for parts in csv.reader(f):
            if parts and parts[0] == "PROCESS":
                header = [p.strip() for p in parts]
                continue
            if (
                header
                and parts
                and parts[0].startswith('"') is False
                and "(PID:" in parts[0]
            ):
                row = parts
                break
    if not (header and row):
        return {}
    metrics = {}
    for i, col in enumerate(header[1:], 1):
        key = col.strip().strip('"')
        try:
            metrics[key] = float(row[i])
        except (ValueError, IndexError):
            pass
    return metrics
