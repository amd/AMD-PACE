# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse

import torch
from transformers import AutoConfig

from pace.llm.step import StepGenerationBackendConfig, plan_split
from pace.llm.step.budget import GIB, available_vram_bytes, device_workspace_bytes
from pace.llm.step.loading import count_parameters
from pace.utils.device import gpu_available
from pace.utils.dtype import TorchDtypeResolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the FFN channel split STeP would choose for a VRAM budget, without loading weights. The split is solved exactly as the runtime solves it, so a deployment can be sized before anything is loaded."
    )
    parser.add_argument("model", help="HF hub id or local checkpoint directory")
    parser.add_argument("--vram-budget-gb", type=float, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--dtype", default="bfloat16", help="bfloat16, float16 or float32"
    )
    parser.add_argument(
        "--residual-backend", default="auto", choices=("auto", "pace", "torch")
    )
    parser.add_argument(
        "--speculation",
        default="auto",
        choices=("on", "off", "auto"),
        help="off reserves one KV cache; drafting reserves a second",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_config = StepGenerationBackendConfig(
        vram_budget_gb=args.vram_budget_gb,
        max_seq_length=args.max_seq_length,
        residual_backend=args.residual_backend,
        speculation=args.speculation,
    )
    dtype = TorchDtypeResolver.resolve(args.dtype)
    step_config.check_residual_backend(dtype)
    model_config = AutoConfig.from_pretrained(args.model)
    on_gpu = gpu_available()
    requested = int(step_config.vram_budget_gb * GIB)
    budget = requested
    if on_gpu:
        budget = min(requested, available_vram_bytes(torch.cuda.current_device()))
    plan = plan_split(
        model_config,
        budget,
        step_config.max_seq_length,
        dtype,
        step_config.speculation,
        align=step_config.uses_pace_kernels(dtype),
        parameter_count=count_parameters(model_config),
        workspace=(
            device_workspace_bytes(torch.device("cuda"), dtype) if on_gpu else 0
        ),
    )
    memory, split = plan.memory, plan.split

    print(f"model      {args.model}")
    capped = (
        f", capped from {args.vram_budget_gb:.1f} GiB at what this GPU can give with 1 GiB kept free"
        if budget < requested
        else ""
    )
    print(f"budget     {budget / GIB:.1f} GiB{capped}")
    print(f"resident   {memory.describe(split.ratio)}")
    if not on_gpu:
        print(
            "no GPU     workspace not measured and budget not capped, so the runtime's split may be smaller"
        )
    print(
        f"fixed cost {memory.fixed / GIB:.2f} GiB, full model {memory.total / GIB:.2f} GiB"
    )
    print(f"split      {split.describe()}")
    print(f"residual aligned for the fast CPU kernel: {split.packable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
