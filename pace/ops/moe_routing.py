# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
MoE routing — selecting the top-k experts and their weights.

Routing is kept separate from the expert computation. `MoERouter` is a pace
operator: given router logits it produces a uniform contract

    (topk_ids: Tensor[T, k] int64, topk_weights: Tensor[T, k] float32)

that the expert computation consumes. Because the experts only consume
(ids, weights), a new routing scheme is added without changing the expert
kernel. The scheme (softmax / sigmoid / grouped-sigmoid) is selected by
`MoEConfig.routing` and computed by the operator's backend, so routing can be
switched between NATIVE and a future fused backend like any other op. The
per-scheme implementations live with the backend (see backends/native.py).
"""

from enum import Enum
from typing import Optional, Tuple

import torch

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, BackendType, DataType


class RoutingType(str, Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    GROUPED_SIGMOID = "grouped_sigmoid"


# Each routing scheme is its own operator type, so its NATIVE/TPP backends are
# selected by the registry (no scheme branching inside a backend).
_ROUTING_OPERATOR = {
    RoutingType.SOFTMAX: OperatorType.MOEROUTER_SOFTMAX,
    RoutingType.SIGMOID: OperatorType.MOEROUTER_SIGMOID,
    RoutingType.GROUPED_SIGMOID: OperatorType.MOEROUTER_GROUPED_SIGMOID,
}


class MoERouter(OperatorBase):
    """Routing operator: router logits -> (topk_ids, topk_weights).

    The scheme is fixed at construction; its operator type selects the backend,
    so routing is swappable between NATIVE and a fused backend like any other
    pace operator (a scheme with no fused backend falls back to NATIVE).
    """

    @property
    def operator_type(self):
        return _ROUTING_OPERATOR[self.routing]

    def __init__(
        self,
        topk: int,
        routing: RoutingType = RoutingType.SOFTMAX,
        renormalize: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        # operator_type reads self.routing, so set it before super().__init__.
        self.routing = routing
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.e_score_correction_bias = e_score_correction_bias
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backend.execute(
            router_logits,
            topk=self.topk,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            e_score_correction_bias=self.e_score_correction_bias,
        )

    def extra_repr(self) -> str:
        return (
            f"topk={self.topk}, routing={self.routing.value}, "
            f"backend_impl={self.backend}"
        )
