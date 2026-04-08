# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

import psutil

from pace.utils.logging import PACE_INFO, PACE_WARNING, PACE_ASSERT
from pace.server.utils import wait_for_server_ready
from pace.server.monitoring.prometheus_runner import start_prometheus


def main():
    parser = argparse.ArgumentParser(description="Inference Server for LLMs")
    parser.add_argument(
        "--server_host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server",
    )
    parser.add_argument(
        "--server_port", type=int, default=8000, help="Port to bind the server"
    )
    parser.add_argument(
        "--server_model",
        type=str,
        default="facebook/opt-6.7b",
        help="Model name to load",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type for the model"
    )
    parser.add_argument(
        "--kv_cache_type", type=str, default="BMC", help="KV Cache type"
    )
    parser.add_argument(
        "--serve_type", type=str, default="iterative", help="Serving type"
    )
    parser.add_argument(
        "--op_config", type=str, default="{}", help="operator backend configuration"
    )
    parser.add_argument(
        "--router_host", type=str, default="0.0.0.0", help="Host address for the router"
    )
    parser.add_argument(
        "--router_port", type=int, default=8080, help="Port for the router"
    )
    parser.add_argument(
        "--scheduler_metrics_enabled",
        type=str,
        default="False",
        help="Enable scheduler metrics collection",
    )
    parser.add_argument(
        "--fastapi_log_level",
        type=str,
        default="Default",
        help="Log level for FastAPI",
    )
    parser.add_argument(
        "--spec_config",
        type=str,
        default="{}",
        help='Speculative decoding config JSON, e.g. \'{"model_name": "amd/PARD-Qwen2.5-0.5B", "num_speculative_tokens": 12}\'',
    )
    parser.add_argument(
        "--kv_cache_memory_gb",
        type=float,
        default=None,
        help="Memory budget for SLAB KV cache pool in GB",
    )
    parser.add_argument(
        "--enable_prometheus",
        action="store_true",
        help="Start Prometheus when launcher starts; downloads binary to cache if not present",
    )
    parser.add_argument(
        "--numa_physcpubind",
        type=str,
        default=None,
        metavar="RANGE",
        help="Numactl: physcpubind per instance (semicolon-separated, e.g. '0-95;96-191'). Default: auto-partition socket 0 cores.",
    )
    parser.add_argument(
        "--numa_membind",
        type=str,
        default=None,
        metavar="NODES",
        help="Numactl: membind per instance (semicolon-separated, e.g. '0;0,1'). Default: auto-derived from core range.",
    )
    parser.add_argument(
        "--num_engine_instances",
        type=int,
        default=1,
        help="Number of engine instances to launch with core affinity",
    )
    args = parser.parse_args()

    PACE_ASSERT(
        shutil.which("numactl") is not None,
        "'numactl' is required but not found on this system.",
    )

    # physical_package_id gives true socket count (NPS-independent) to derive per-socket cores.
    total_physical = psutil.cpu_count(logical=False)
    try:
        topo = pathlib.Path("/sys/devices/system/cpu")
        socket_ids = {
            (topo / cpu / "topology/physical_package_id").read_text().strip()
            for cpu in os.listdir(topo)
            if cpu.startswith("cpu") and cpu[3:].isdigit()
        }
        num_sockets = len(socket_ids) if socket_ids else 1
    except (OSError, ValueError):
        num_sockets = 1
    physical_cores = total_physical // num_sockets
    num_instances = args.num_engine_instances

    # Calculate cores per instance
    cores_per_instance = physical_cores // num_instances

    if cores_per_instance == 0:
        PACE_INFO(
            f"Error: Not enough cores ({physical_cores}) for {num_instances} instances"
        )
        return

    PACE_INFO(
        f"Detected {num_sockets} socket(s), {total_physical} total physical cores, {physical_cores} per socket"
    )
    PACE_INFO(f"Number of engine instances: {num_instances}")

    def _split_override(raw: str | None) -> list[str | None]:
        """Split a semicolon-separated override string into per-instance values."""
        if raw is None:
            return [None] * num_instances
        parts = [p.strip() for p in raw.split(";")]
        if len(parts) == 1:
            return parts * num_instances
        if len(parts) != num_instances:
            PACE_WARNING(
                f"Expected 1 or {num_instances} semicolon-separated values, got {len(parts)} "
                f"in '{raw}'. Ignoring override, falling back to auto-detected defaults."
            )
            return [None] * num_instances
        return parts

    def _numa_node_for_cpu(cpu_id: int) -> str:
        """Read the NUMA node for a physical CPU from sysfs."""
        cpu_dir = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
        try:
            for entry in os.listdir(cpu_dir):
                if entry.startswith("node"):
                    return entry[4:]
        except OSError:
            pass
        return "0"

    def _count_cores(core_spec: str) -> int:
        """Count cores in a physcpubind spec ('0-95' or '1,3,4,6,78' or '0-11,13-25')."""
        count = 0
        for part in core_spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                count += int(hi) - int(lo) + 1
            else:
                count += 1
        return count

    def _derive_membind(core_range: str) -> str:
        """Derive NUMA membind from a physcpubind range (e.g. '0-95' -> '0')."""
        nodes = set()
        for part in core_range.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                for cpu_id in range(int(lo), int(hi) + 1):
                    nodes.add(_numa_node_for_cpu(cpu_id))
            else:
                nodes.add(_numa_node_for_cpu(int(part)))
        return ",".join(sorted(nodes, key=int))

    phy_overrides = _split_override(args.numa_physcpubind)
    mem_overrides = _split_override(args.numa_membind)
    if args.numa_physcpubind is None:
        PACE_WARNING(
            f"--numa_physcpubind not set. Defaulting to socket 0 cores, "
            f"{cores_per_instance} cores per instance. Use --numa_physcpubind to override."
        )
    if args.numa_membind is None:
        PACE_WARNING(
            "--numa_membind not set. Auto-deriving from physcpubind core range. "
            "Use --numa_membind to override."
        )

    # Launch engine instances with numactl.
    server_procs = []
    server_configs = []

    for instance_id in range(num_instances):
        start_core = instance_id * cores_per_instance
        end_core = start_core + cores_per_instance - 1
        instance_port = args.server_port + instance_id

        physcpubind = phy_overrides[instance_id] or f"{start_core}-{end_core}"
        membind = mem_overrides[instance_id] or _derive_membind(physcpubind)

        omp_threads = _count_cores(physcpubind)

        PACE_INFO(
            f"Instance {instance_id}: physcpubind={physcpubind}, membind={membind}, "
            f"OMP_NUM_THREADS={omp_threads}, port {instance_port}"
        )

        numactl_prefix = [
            "numactl",
            f"--physcpubind={physcpubind}",
            f"--membind={membind}",
        ]
        numactl_prefix.append("--")

        server_cmd = numactl_prefix + [
            sys.executable,
            "-m",
            "pace.server.engine.frontend",
            "--host",
            str(args.server_host),
            "--port",
            str(instance_port),
            "--fastapi_log_level",
            str(args.fastapi_log_level),
        ]

        instance_env = os.environ.copy()
        instance_env["OMP_NUM_THREADS"] = str(omp_threads)
        instance_env["OMP_WAIT_POLICY"] = "active"

        try:
            PACE_INFO(
                f"Launching SERVER instance {instance_id}: {' '.join(server_cmd)}"
            )
            server_proc = subprocess.Popen(server_cmd, env=instance_env)
            server_procs.append(server_proc)

            server_configs.append(
                {
                    "host": args.server_host,
                    "port": instance_port,
                    "endpoint": "get_models",
                }
            )

        except Exception as e:
            PACE_INFO(f"Failed to launch server instance {instance_id}: {e}")
            for proc in server_procs:
                proc.terminate()
            return

    # Wait for all servers to be ready
    if not wait_for_server_ready(server_configs):
        PACE_INFO("One or more servers failed to start or become ready within timeout")
        for proc in server_procs:
            proc.terminate()
        return
    PACE_INFO("All server instances are ready.")

    # Build router command - router will connect to all server instances
    router_cmd = [
        sys.executable,
        "-m",
        "pace.server.router.frontend",
        "--router_host",
        str(args.router_host),
        "--router_port",
        str(args.router_port),
        "--serve_type",
        str(args.serve_type),
        "--server_host",
        str(args.server_host),
        "--server_port",
        str(args.server_port),
        "--model",
        str(args.server_model),
        "--dtype",
        str(args.dtype),
        "--kv_cache_type",
        str(args.kv_cache_type),
        "--op_config",
        str(args.op_config),
        "--scheduler_metrics_enabled",
        str(args.scheduler_metrics_enabled),
        "--fastapi_log_level",
        str(args.fastapi_log_level),
        "--num_engine_instances",
        str(args.num_engine_instances),
        "--spec_config",
        str(args.spec_config),
    ]
    if args.kv_cache_memory_gb is not None:
        router_cmd.extend(["--kv_cache_memory_gb", str(args.kv_cache_memory_gb)])
    prometheus_proc = None
    prometheus_config_path = None
    prometheus_cleanup_config = False

    try:
        PACE_INFO(f"Launching ROUTER: {' '.join(router_cmd)}")
        router_proc = subprocess.Popen(router_cmd, env=os.environ)

        # Wait briefly to ensure router starts properly
        router_configs = [
            {
                "host": args.router_host,
                "port": args.router_port,
                "endpoint": "v1/health",
            },
        ]
        if not wait_for_server_ready(router_configs):
            PACE_INFO("Router failed to start or become ready within timeout")
            router_proc.terminate()
            for proc in server_procs:
                proc.terminate()
            return
        PACE_INFO("Router is ready.")

        prometheus_proc, prometheus_config_path, prometheus_cleanup_config = (
            start_prometheus(args.enable_prometheus, args.router_host, args.router_port)
        )

    except Exception as e:
        PACE_INFO(f"Failed to launch router: {e}")
        for proc in server_procs:
            proc.terminate()
        return

    try:
        # Wait for all processes
        for proc in server_procs:
            proc.wait()
        router_proc.wait()
        if prometheus_proc is not None:
            prometheus_proc.wait()
    except KeyboardInterrupt:
        PACE_INFO("Stopping all servers and router...")
        for proc in server_procs:
            proc.terminate()
        router_proc.terminate()
        if prometheus_proc is not None:
            prometheus_proc.terminate()
        if prometheus_cleanup_config and prometheus_config_path:
            try:
                os.unlink(prometheus_config_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
