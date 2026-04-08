# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portion of this file consists of AI-generated code.
# ******************************************************************************


import time
import requests


def wait_for_server_ready(configs, timeout=1000, initial_delay=0.5):
    """
    Poll server health endpoints until all are ready or timeout is reached.

    Args:
        configs: List of dicts with 'host', 'port', 'endpoint' or single dict
        timeout: Maximum wait time in seconds
        initial_delay: Initial delay between checks

    Returns:
        bool: True if all servers ready, False if timeout
    """
    if isinstance(configs, dict):
        configs = [configs]

    for config in configs:
        config.setdefault("endpoint", "get_health")

    start_time = time.time()
    delay = initial_delay

    while time.time() - start_time < timeout:
        if all(_check_endpoint(config) for config in configs):
            return True
        time.sleep(delay)
        delay = min(delay * 1.5, 2.0)

    return False


def _check_endpoint(config):
    """Check if single endpoint is ready."""
    url = f"http://{config['host']}:{config['port']}/{config['endpoint']}"
    try:
        return requests.get(url, timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False
