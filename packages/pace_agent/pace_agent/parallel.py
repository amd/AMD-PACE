# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Shared task-parallel runner for PACE Agent.

Runs INDEPENDENT tasks (queries / workload items) concurrently, each in its own process pinned to
its own block of CPU cores (no overlap). Process isolation is required because PACE Agent keeps module
globals (node_counter, shims, the tool-hook) — threads would corrupt them; separate processes each
get clean state and their own browser.

IMPORTANT: this is for *independent tasks* only. The replay loop inside a single `profile`
(the timing measurement) must stay sequential — running those concurrently would make them contend
for CPU and corrupt the numbers.
"""

import os
import sys

from pace_agent.errors import ConfigError


def usable_cpus(cpus=None):
    """The CPU ids this process may run on — the affinity mask, not `cpu_count()`, so a run under
    `taskset` or in a cgroup pins inside its real allowance. Falls back to all CPUs on platforms
    without `sched_getaffinity` (macOS/BSD), matching `pinning.current_affinity()`."""
    if cpus is not None:
        return sorted(cpus)
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def pin_worker(counter, cores_per_worker, cpus):
    """Pool initializer: pin this worker process (and its children, e.g. Chrome) to its own block
    of `cores_per_worker` cores from `cpus`, assigned by worker index."""
    try:
        with counter.get_lock():
            widx = counter.value
            counter.value += 1
        block = cpus[widx * cores_per_worker : (widx + 1) * cores_per_worker]
        if block and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(block))
    except Exception as exc:  # noqa: BLE001 - a worker must still run unpinned
        # Silence here would look like a successful pinned run while the workers
        # actually share cores and contend, which is exactly what invalidates the
        # timings this module exists to protect.
        print(
            f"pace_agent: WARNING failed to pin worker to cores "
            f"{cores_per_worker} per worker: {type(exc).__name__}: {exc}; "
            "measurements may be affected by CPU contention",
            file=sys.stderr,
        )


def worker_cpu_plan(workers, cores_per_worker, cpus=None):
    """The core block each worker will get, assigned exactly as `pin_worker` does — so a run can
    record the layout it intended instead of leaving it to be re-derived later. A layout that does
    not fit is refused rather than wrapped: shared cores contend, which corrupts the timings.
    """
    available = usable_cpus(cpus)
    workers = max(int(workers or 1), 1)
    cores = max(int(cores_per_worker or 1), 1)
    if workers * cores > len(available):
        raise ConfigError(
            f"parallel={workers} x cores_per_worker={cores} needs {workers * cores} CPUs but this "
            f"process may use {len(available)}; lower either value (pinning workers onto shared "
            f"cores would make them contend and invalidate the measurement)"
        )
    return [available[w * cores : (w + 1) * cores] for w in range(workers)]


def _pool_round(fn, work, workers, cores_per_worker, out, cpus):
    """Run one pool over `work` (a list of `(index, item)`), filling `out` in place.

    Returns `(n_settled, pool_died)`. A task that raises settles as an error row; a task whose
    worker was killed outright settles nothing, so the caller can retry it in a fresh pool.
    """
    import multiprocessing as mp
    from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    settled, died = 0, False
    with ProcessPoolExecutor(
        max_workers=min(workers, len(work)),
        mp_context=ctx,
        initializer=pin_worker,
        initargs=(counter, cores_per_worker, cpus),
    ) as ex:
        futs = {ex.submit(fn, it): i for i, it in work}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            # the pool died under it or it never ran: either way not a task error, so leave it
            except (BrokenProcessPool, CancelledError):
                died = True
                continue
            except Exception as e:  # noqa: BLE001 - never abort the batch
                out[i] = {"error": str(e)[:200]}
            settled += 1
    return settled, died


def run_parallel(fn, items, workers=1, cores_per_worker=2):
    """Run `fn(item)` for each item, returning results in input order.

    workers<=1 -> sequential, in-process (no pickling; default, backwards-compatible).
    workers>1  -> process pool (spawn), each worker pinned to its own cores.

    `fn` MUST be a top-level (importable/picklable) function and `items` must be picklable.
    A task that raises does NOT abort the batch — its slot gets {"error": ...}.

    A worker killed outright (OOM, segfault, an operator's `kill`) does not raise in Python: it
    breaks the whole pool, and every task that had not returned yet fails with it — including tasks
    that were never started. On a long sweep that would throw away hours of remaining work for one
    dead process, so results already in hand are kept and the unfinished tasks are re-run in a
    fresh pool, then, if that pool dies too, one task per pool — a task that reliably kills its
    worker can only lose itself, and a kill caused by the concurrency itself (several browsers
    against the memory on the host) survives as a slow but complete run.
    """
    items = list(items)
    workers = min(int(workers or 1), len(items)) if items else 1
    if workers <= 1:
        return [_settle(fn, it) for it in items]

    # refuses an oversubscribed layout before any task starts
    cpus = [
        cpu for block in worker_cpu_plan(workers, cores_per_worker) for cpu in block
    ]
    out = [None] * len(items)

    def pending(todo):
        return [(i, it) for i, it in todo if out[i] is None]

    todo = list(enumerate(items))
    _, died = _pool_round(fn, todo, workers, cores_per_worker, out, cpus)
    todo = pending(todo)
    if todo and died:
        _warn_died(len(todo), "retrying them in a fresh pool")
        _, died = _pool_round(fn, todo, workers, cores_per_worker, out, cpus)
        todo = pending(todo)
    if todo and died:
        _warn_died(len(todo), "retrying them one at a time to isolate the cause")
        for pair in todo:
            _pool_round(fn, [pair], 1, cores_per_worker, out, cpus)
        todo = pending(todo)
    for i, _ in todo:
        out[i] = {
            "error": "worker process died before this task produced a result (OOM or crash, "
            "not a task error); it did not complete even in a pool of its own"
        }
    return out


def _settle(fn, item):
    """One task in-process under the pool's contract: a raising task becomes an error row."""
    try:
        return fn(item)
    except Exception as e:  # noqa: BLE001 - never abort the batch
        return {"error": str(e)[:200]}


def _warn_died(n, plan):
    print(
        f"warning: a worker process died and took {n} unfinished task(s) with it "
        f"(OOM or crash, not a task error); {plan}",
        file=sys.stderr,
        flush=True,
    )
