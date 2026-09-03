# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

_BUCKETS = ("tool_io", "retrieval", "serialization", "context", "orchestration")


def attribute(traj):
    buckets = {b: 0.0 for b in _BUCKETS}
    for o in traj.tools:
        buckets[o.bucket] = buckets.get(o.bucket, 0.0) + o.naive_time
    for b, t in getattr(
        traj, "shim_buckets", {}
    ).items():  # 7.13 layer 2: un-instrumented libs
        buckets[b] = buckets.get(b, 0.0) + t
    op_total = sum(buckets.values())
    model = traj.llm_time
    unattributed = max(0.0, traj.e2e - op_total - model)
    total = traj.e2e or (op_total + model)
    return {
        "total": total,
        "model": model,
        "buckets": buckets,
        "non_model_cpu": op_total,
        "unattributed": unattributed,
        "non_model_frac": (op_total / total) if total else 0.0,
        "unattributed_frac": (unattributed / total) if total else 0.0,
    }
