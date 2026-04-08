# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Download Prometheus binary to cache and run it with built-in config.
"""

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from typing import Optional, Tuple

from pace.utils.logging import PACE_INFO, PACE_WARNING

VERSION = "3.9.1"
PLATFORM = "linux-amd64"
URL = f"https://github.com/prometheus/prometheus/releases/download/v{VERSION}/prometheus-{VERSION}.{PLATFORM}.tar.gz"
# Pinned SHA256 for integrity verification. Source: same release, sha256sums.txt
# https://github.com/prometheus/prometheus/releases/download/v{VERSION}/sha256sums.txt
# Line for linux-amd64: 86a6999...  prometheus-3.9.1.linux-amd64.tar.gz
TAR_SHA256 = "86a6999dd6aacbd994acde93c77cfa314d4be1c8e7b7c58f444355c77b32c584"
PORT = 9090

# Built-in Prometheus config; PACE target "localhost:8080" is replaced at runtime
PROMETHEUS_YML = """
# my global config
global:
  scrape_interval: 5s
  evaluation_interval: 5s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files: []

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
        labels:
          app: "prometheus"
  - job_name: "pace-server"
    metrics_path: '/metrics/'
    scrape_interval: 5s
    static_configs:
      - targets: ["localhost:8080"]
        labels:
          app: "pace-server"
          service: "llm-inference"
"""


# Cache: ~/.cache/pace/prometheus/prometheus-{VERSION}.{PLATFORM}/prometheus (or PACE_PROMETHEUS_CACHE)
def _cache_dir():
    base = os.environ.get("PACE_PROMETHEUS_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "pace", "prometheus"
    )
    return os.path.join(base, f"prometheus-{VERSION}.{PLATFORM}")


def _binary_path():
    return os.path.join(_cache_dir(), "prometheus")


def _ensure_binary() -> str:
    path = _binary_path()
    if os.path.isfile(path) and os.access(path, os.X_OK):
        PACE_INFO(f"Using cached Prometheus at {path}")
        return path
    PACE_INFO(f"Downloading Prometheus for this platform ({PLATFORM})")
    try:
        with urllib.request.urlopen(URL, timeout=120) as resp:
            tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
            try:
                shutil.copyfileobj(resp, tmp)
                tmp.close()
                with open(tmp.name, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                if digest != TAR_SHA256:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"Prometheus tarball checksum mismatch (expected {TAR_SHA256}, got {digest})"
                    )
                os.makedirs(_cache_dir(), exist_ok=True)
                with tarfile.open(tmp.name, "r:gz") as tar:
                    tar.extractall(path=_cache_dir())
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
    except Exception as e:
        PACE_WARNING(f"Download failed: {e}")
        raise
    # Archive may contain a single subdir (e.g. prometheus-3.9.1.linux-amd64/) with prometheus inside
    if not os.path.isfile(path):
        for name in os.listdir(_cache_dir()):
            sub = os.path.join(_cache_dir(), name)
            if os.path.isdir(sub):
                candidate = os.path.join(sub, "prometheus")
                if os.path.isfile(candidate):
                    shutil.move(candidate, path)
                    shutil.rmtree(sub, ignore_errors=True)
                    break
    if not os.path.isfile(path):
        raise RuntimeError("Prometheus binary not found in archive")
    os.chmod(path, 0o755)
    PACE_INFO(f"Prometheus at {path}")
    return path


def _config_path(router_host: str, router_port: int) -> Tuple[str, bool]:
    host = "localhost" if router_host == "0.0.0.0" else router_host
    target = f"{host}:{router_port}"
    content = PROMETHEUS_YML.replace("localhost:8080", target)
    fd, path = tempfile.mkstemp(suffix=".yml", prefix="pace_prometheus_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content.encode())
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path, True


def start_prometheus(
    enable: bool, router_host: str, router_port: int
) -> Tuple[Optional[subprocess.Popen], Optional[str], bool]:
    """If enable: get config, get/download binary, start process. Returns (proc, config_path, cleanup_config)."""
    if not enable:
        return None, None, False
    try:
        config_path, cleanup = _config_path(router_host, router_port)
    except OSError as e:
        PACE_WARNING(f"Prometheus config: {e}")
        return None, None, False
    try:
        binary = _ensure_binary()
    except Exception as e:
        PACE_WARNING(f"Prometheus: {e}")
        if cleanup:
            try:
                os.unlink(config_path)
            except OSError:
                pass
        return None, None, False
    try:
        proc = subprocess.Popen(
            [
                binary,
                "--config.file=" + config_path,
                "--web.listen-address=:" + str(PORT),
            ],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        PACE_INFO("Prometheus UI: http://localhost:9090")
        return proc, config_path, cleanup
    except FileNotFoundError:
        PACE_WARNING("Prometheus binary not found")
        if cleanup:
            try:
                os.unlink(config_path)
            except OSError:
                pass
        return None, None, False
