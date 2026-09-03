# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""The task-parallel runner, including what happens when a worker dies rather than raises."""

import io
import os
import time
import unittest
from contextlib import redirect_stderr

from pace_agent.errors import ConfigError
from pace_agent.parallel import run_parallel, worker_cpu_plan

# module-level so `spawn` can pickle them


def double(i):
    return {"ok": i * 2}


def raise_on_two(i):
    if i == 2:
        raise RuntimeError("task blew up")
    return {"ok": i}


def kill_worker_on_five(i):
    """Item 5 kills its worker outright, as an OOM kill or a segfault would."""
    if i == 5:
        time.sleep(0.6)
        os._exit(137)
    time.sleep(0.1 if i < 5 else 1.2)
    return {"ok": i}


class TestRunParallel(unittest.TestCase):
    def test_pool_results_come_back_in_input_order(self):
        self.assertEqual(
            run_parallel(double, list(range(6)), workers=3, cores_per_worker=1),
            [{"ok": i * 2} for i in range(6)],
        )

    def test_a_task_that_raises_does_not_abort_the_batch(self):
        rows = run_parallel(raise_on_two, list(range(4)), workers=2, cores_per_worker=1)
        self.assertIn("task blew up", rows[2]["error"])
        self.assertEqual([rows[i]["ok"] for i in (0, 1, 3)], [0, 1, 3])

    def test_the_sequential_path_settles_a_raising_task_the_same_way(self):
        rows = run_parallel(raise_on_two, list(range(4)), workers=1)
        self.assertIn("task blew up", rows[2]["error"])
        self.assertEqual([rows[i]["ok"] for i in (0, 1, 3)], [0, 1, 3])

    def test_a_killed_worker_does_not_abandon_the_remaining_tasks(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rows = run_parallel(
                kill_worker_on_five, list(range(12)), workers=3, cores_per_worker=1
            )
        self.assertEqual(len(rows), 12)
        failed = [i for i, r in enumerate(rows) if r.get("ok") is None]
        self.assertEqual(failed, [5])
        self.assertIn("worker process died", rows[5]["error"])
        self.assertEqual(
            [rows[i]["ok"] for i in range(12) if i != 5],
            [i for i in range(12) if i != 5],
        )
        self.assertIn("worker process died", err.getvalue())


class TestWorkerCpuPlan(unittest.TestCase):
    def test_blocks_do_not_overlap(self):
        self.assertEqual(worker_cpu_plan(3, 2, cpus=range(8)), [[0, 1], [2, 3], [4, 5]])

    def test_blocks_are_taken_from_the_cpus_this_process_may_use(self):
        # under `taskset -c 8-15` the usable ids are not 0..n-1
        self.assertEqual(worker_cpu_plan(2, 2, cpus=[8, 9, 10, 11]), [[8, 9], [10, 11]])

    def test_a_layout_that_does_not_fit_is_refused(self):
        with self.assertRaises(ConfigError) as cm:
            worker_cpu_plan(3, 2, cpus=range(4))
        self.assertIn("needs 6 CPUs", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
