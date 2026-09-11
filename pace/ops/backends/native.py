# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import (
    OperatorType,
    FusedOperatorType,
    BackendType,
    DataType,
)


@backend_registry.register(
    OperatorType.LINEAR, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeLinear(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(input, weight, bias)


@backend_registry.register(
    OperatorType.LAYERNORM, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeLayerNorm(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        normalized_shape: Tuple[int],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return F.layer_norm(input, normalized_shape, weight, bias, eps)


@backend_registry.register(
    OperatorType.RMSNORM, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeRMSNorm(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        normalized_shape: Tuple[int],
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:

        return F.rms_norm(
            input, normalized_shape=normalized_shape, weight=weight, eps=eps
        )


@backend_registry.register(
    OperatorType.L2NORM, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeL2Norm(BackendBase):

    def execute(self, input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + eps)).type_as(
            input
        )


@backend_registry.register(
    FusedOperatorType.FUSED_LAYERNORM_RESIDUAL,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeFusedLayerNormResidual(BackendBase):

    def execute(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        normed = F.layer_norm(
            residual_out,
            normalized_shape=weight.shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        return normed, residual_out


@backend_registry.register(
    FusedOperatorType.FUSED_RMSNORM_RESIDUAL,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeFusedRMSNormResidual(BackendBase):

    def execute(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        normed = F.rms_norm(
            residual_out,
            normalized_shape=weight.shape,
            weight=weight,
            eps=eps,
        )
        return normed, residual_out


@backend_registry.register(
    OperatorType.SOFTMAX, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeSoftmax(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        dim: int = -1,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return F.softmax(input, dim=dim, dtype=dtype)


@backend_registry.register(
    OperatorType.RELU, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeReLU(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        inplace: bool = False,
    ) -> torch.Tensor:
        return F.relu(input, inplace=inplace)


@backend_registry.register(
    OperatorType.GELU, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeGeLU(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        approximate: bool = False,
    ) -> torch.Tensor:
        return F.gelu(input, approximate=approximate)


@backend_registry.register(
    OperatorType.SILU, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeSiLU(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        inplace: bool = False,
    ) -> torch.Tensor:
        return F.silu(input, inplace=inplace)


@backend_registry.register(
    OperatorType.TANH, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeTanh(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(input)


@backend_registry.register(
    OperatorType.SIGMOID, BackendType.NATIVE, [DataType.FLOAT32, DataType.BFLOAT16]
)
class NativeSigmoid(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(input)


@backend_registry.register(
    OperatorType.MOEROUTER_SOFTMAX,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeSoftmaxRouter(BackendBase):
    @staticmethod
    def softmax_topk(
        router_logits: torch.Tensor, topk: int, renormalize: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = router_logits.float()
        topk_logits, topk_indices = torch.topk(
            logits, topk, dim=-1, largest=True, sorted=True
        )
        if renormalize:
            weights = torch.softmax(topk_logits, dim=-1)
        else:
            weights = torch.exp(
                topk_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            )
        return (
            topk_indices.to(torch.int64).contiguous(),
            weights.to(torch.float32).contiguous(),
        )

    def execute(
        self, router_logits: torch.Tensor, topk: int, renormalize: bool = True, **_
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.softmax_topk(router_logits, topk, renormalize)


@backend_registry.register(
    OperatorType.MOEROUTER_SIGMOID,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeSigmoidRouter(BackendBase):
    """Top-k selection by logit; weight = sigmoid(selected logit), optionally
    renormalized. Llama4-style routing."""

    def execute(
        self, router_logits: torch.Tensor, topk: int, renormalize: bool = False, **_
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = router_logits.float()
        tv, ti = torch.topk(logits, topk, dim=-1, largest=True, sorted=True)
        w = torch.sigmoid(tv)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return ti.to(torch.int64).contiguous(), w.to(torch.float32).contiguous()


@backend_registry.register(
    OperatorType.MOEROUTER_GROUPED_SIGMOID,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeGroupedSigmoidRouter(BackendBase):
    """Group-limited top-k with sigmoid scoring + optional expert-bias correction
    (DeepSeek V2/V3). Experts are partitioned into `num_expert_group` groups; only
    the top `topk_group` groups (by their top-2 score sum) are eligible, then a
    global top-k within that set."""

    def execute(
        self,
        router_logits: torch.Tensor,
        topk: int,
        num_expert_group: int = 1,
        topk_group: int = 1,
        renormalize: bool = True,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        **_,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(router_logits.float())
        sel = (
            scores + e_score_correction_bias
            if e_score_correction_bias is not None
            else scores
        )
        T, E = sel.shape
        g = int(num_expert_group or 1)
        tg = int(topk_group or g)
        # This path is reachable via MoERouter directly (no MoEConfig guard), so
        # validate the group config here to fail fast with an actionable message
        # instead of a cryptic view()/topk() error.
        if E % g != 0:
            raise ValueError(
                f"MOEROUTER_GROUPED_SIGMOID: num_expert_group={g} must evenly "
                f"divide the number of experts E={E}."
            )
        if not (1 <= tg <= g):
            raise ValueError(
                f"MOEROUTER_GROUPED_SIGMOID: topk_group={tg} must be in "
                f"[1, num_expert_group={g}]."
            )
        if g > 1:
            grp = sel.view(T, g, E // g)
            # group score = sum of its top-2 experts (DeepSeek-V3 convention)
            gscore = grp.topk(min(2, E // g), dim=-1).values.sum(dim=-1)
            keep = torch.topk(gscore, tg, dim=-1).indices
            mask = torch.zeros(T, g, dtype=torch.bool, device=sel.device)
            mask.scatter_(1, keep, True)
            sel = sel.masked_fill(
                ~mask.unsqueeze(-1).expand_as(grp).reshape(T, E), float("-inf")
            )
        ti = torch.topk(sel, topk, dim=-1).indices
        w = torch.gather(scores, 1, ti)  # weight from raw sigmoid score (no bias)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return ti.to(torch.int64).contiguous(), w.to(torch.float32).contiguous()
