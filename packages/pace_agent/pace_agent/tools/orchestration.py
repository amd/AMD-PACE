# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy

from pace_agent.registry import PaceAgentTool, register_tool


@register_tool("state_merge", "orchestration")
class PaceAgentStateMerge(PaceAgentTool):
    def naive(self, states):
        out = {}
        for s in states:
            out = {**out, **copy.deepcopy(s)}
        return out

    def bench_inputs(self, spec):
        n = spec.get("n", 500)
        return ([{f"k{i}": {"v": list(range(10))}} for i in range(n)],), {}
