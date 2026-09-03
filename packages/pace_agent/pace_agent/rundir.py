# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import os

from pace_agent.config import SCHEMA_VERSION


class RunDir:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def write_result(self, result):
        result.setdefault("schema_version", SCHEMA_VERSION)
        self._dump("result.json", result)

    def write_env(self, env):
        self._dump("env.json", env)

    def write_trajectory(self, traj):
        self._dump("trajectory.json", traj)

    def write_trace(self, trace):
        self._dump("trace.json", trace)

    def write_report(self, md):
        with open(os.path.join(self.path, "report.md"), "w") as f:
            f.write(md)

    def raw_dir(self):
        d = os.path.join(self.path, "raw")
        os.makedirs(d, exist_ok=True)
        return d

    def read_result(self):
        with open(os.path.join(self.path, "result.json")) as f:
            return json.load(f)

    def _dump(self, name, obj):
        with open(os.path.join(self.path, name), "w") as f:
            json.dump(obj, f, indent=2, default=str)
