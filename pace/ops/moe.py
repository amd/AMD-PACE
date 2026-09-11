# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Universal Mixture-of-Experts operator.

Follows the same operator/backend pattern as MergedMLP:
  - MoEExperts is the model-agnostic frontend operator. It owns the router and
    selects experts via a MoERouter sub-operator (see moe_routing).
  - Like other fused operators, when no fused backend is registered it runs a
    reference path that composes pace sub-operators (the router, a MoERouter,
    and per-expert linears) — each of which carries the requested backend.
  - A fused backend can be registered to take over via the standard backend
    dispatch.

Every matmul goes through a pace operator (Linear / MoERouter), so each piece
is independently wireable to NATIVE or a fused backend.

Usage in any MoE model:
    self.moe = MoEExperts(
        MoEConfig.from_model(config, activation="silu"),
        backend_impl=opconfig.mlp,
    )
    output = self.moe(hidden_states)   # the router lives inside the operator
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pace.ops.base import OperatorBase
from pace.ops.enum import FusedOperatorType, BackendType, DataType
from pace.ops.linear import Linear
from pace.ops.mlp import MergedMLP
from pace.ops.moe_routing import MoERouter, RoutingType

_SUPPORTED_ACTIVATIONS = {"silu", "gelu", "gelu_tanh", "relu", "gptoss_glu"}
_POINTWISE_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}


@dataclass
class MoEConfig:
    """Configuration for a Mixture-of-Experts layer.

    Use from_model() to build from any HuggingFace PretrainedConfig.

    Shared experts (Llama4, Qwen2-MoE, DeepSeek V2/V3): num_shared_experts > 0
    enables experts applied to all tokens regardless of routing. They run as
    dense MLPs.
    """

    hidden_size: int
    intermediate_size: int  # per routed expert intermediate dim
    num_experts: int  # total routed experts
    topk: int = 2  # experts selected per token
    activation: str = "silu"
    gate: bool = True  # gated FFN (SiLU/GELU) vs non-gated
    bias: bool = False
    router_bias: bool = True  # whether the router linear has a bias term
    num_shared_experts: int = 0  # shared experts applied to all tokens
    shared_intermediate_size: int = 0  # 0 means same as intermediate_size
    routed_scaling_factor: float = 1.0  # scales the routed expert output (DeepSeek)
    build_router: bool = True  # build the internal router Linear; set False when
    # the model supplies router logits / routing externally (e.g. Gemma4, whose
    # router applies a norm, input scale, and per-expert scale before selection).
    # Routing configuration — selects the routing strategy (see moe_routing).
    routing: RoutingType = RoutingType.SOFTMAX
    renormalize: bool = True  # renormalize top-k weights (sigmoid/grouped)
    num_expert_group: int = 1  # grouped routing (DeepSeek)
    topk_group: int = 1  # grouped routing: eligible groups
    e_score_correction_bias: Optional[torch.Tensor] = None  # grouped routing bias

    def __post_init__(self):
        if self.hidden_size is None:
            raise ValueError(
                "MoEConfig: hidden_size is required but was not found in the model config. "
                "Pass it explicitly or check the HuggingFace config field name."
            )
        if self.intermediate_size is None:
            raise ValueError(
                "MoEConfig: intermediate_size is required but was not found in the model config. "
                "Pass it explicitly or check the HuggingFace config field name."
            )
        if self.num_experts is None:
            raise ValueError(
                "MoEConfig: num_experts is required but was not found in the model config. "
                "Try field names: num_local_experts, num_experts."
            )
        # Resolve shared_intermediate_size: 0 means same as routed
        if self.shared_intermediate_size == 0:
            self.shared_intermediate_size = self.intermediate_size
        # Validate topk against the routed expert count so a bad value fails with
        # an actionable message rather than a cryptic torch.topk error later.
        if not (0 < self.topk <= self.num_experts):
            raise ValueError(
                f"MoEConfig: topk must be in [1, num_experts]={self.num_experts}, "
                f"got {self.topk}."
            )

    @classmethod
    def from_model(cls, config, **kwargs) -> "MoEConfig":
        """Build MoEConfig from any HuggingFace PretrainedConfig.

        Tries common HF field name variants for each parameter. Pass
        model-specific overrides as kwargs (e.g. activation="gptoss_glu").
        """

        def _get(primary, *fallbacks, default=None):
            for name in (primary, *fallbacks):
                if hasattr(config, name):
                    val = getattr(config, name)
                    if val is not None:
                        return val
            for name in (primary, *fallbacks):
                if name in kwargs:
                    return kwargs.pop(name)
            return default

        return cls(
            hidden_size=_get("hidden_size"),
            intermediate_size=_get("intermediate_size"),
            num_experts=_get("num_local_experts", "num_experts"),
            topk=_get("num_experts_per_tok", "num_selected_experts", default=2),
            activation=kwargs.pop("activation", _get("hidden_act", default="silu")),
            gate=kwargs.pop("gate", True),
            bias=kwargs.pop("bias", False),
            router_bias=kwargs.pop("router_bias", True),
            num_shared_experts=_get("num_shared_experts", default=0),
            shared_intermediate_size=_get("shared_intermediate_size", default=0),
            routed_scaling_factor=_get("routed_scaling_factor", default=1.0),
            routing=kwargs.pop("routing", RoutingType.SOFTMAX),
            renormalize=kwargs.pop("renormalize", _get("norm_topk_prob", default=True)),
            num_expert_group=_get("n_group", "num_expert_group", default=1) or 1,
            topk_group=_get("topk_group", default=1) or 1,
        )


class MoEExperts(OperatorBase):
    """Universal MoE expert operator — model-agnostic, any activation.

    The router and every expert projection are pace Linear operators, and
    expert selection is a MoERouter sub-operator, so the whole layer is wired to
    NATIVE (or a fused backend) through the standard backend dispatch.
    """

    @property
    def operator_type(self):
        return FusedOperatorType.FUSEDMOE

    def __init__(
        self,
        moe_config: MoEConfig,
        backend_impl: BackendType = BackendType.NATIVE,
        dtype: Optional[DataType] = None,
    ):
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        if moe_config.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"MoEExperts: unsupported activation '{moe_config.activation}', "
                f"must be one of: {', '.join(sorted(_SUPPORTED_ACTIVATIONS))}"
            )
        # gptoss_glu is a gated linear unit by construction (it consumes both a
        # gate and an up projection); it has no meaning without a gate.
        if moe_config.activation == "gptoss_glu" and not moe_config.gate:
            raise ValueError(
                "MoEExperts: activation 'gptoss_glu' requires gate=True "
                "(it is a gated linear unit); got gate=False."
            )

        self.moe_config = moe_config
        self.hidden_size = moe_config.hidden_size
        self.intermediate_size = moe_config.intermediate_size
        self.num_experts = moe_config.num_experts
        self.topk = moe_config.topk
        self.bias_available = moe_config.bias
        self.activation = moe_config.activation
        self.gate = moe_config.gate
        self.num_shared_experts = moe_config.num_shared_experts
        self.routed_scaling_factor = moe_config.routed_scaling_factor

        # When a fused engine backend is bound, the per-expert linears only hold
        # weights (the engine consumes the packed copies); without one they run
        # the reference path. The router and shared MLPs always use the requested
        # backend so routing and shared experts stay independently wired.
        expert_backend = backend_impl if self.backend is None else None

        self.build_router = moe_config.build_router
        self.router = (
            Linear(
                in_features=moe_config.hidden_size,
                out_features=moe_config.num_experts,
                bias=moe_config.router_bias,
                dtype=dtype,
                backend_impl=backend_impl,
            )
            if moe_config.build_router
            else None
        )
        self.router_op = MoERouter(
            topk=moe_config.topk,
            routing=moe_config.routing,
            renormalize=moe_config.renormalize,
            num_expert_group=moe_config.num_expert_group,
            topk_group=moe_config.topk_group,
            e_score_correction_bias=moe_config.e_score_correction_bias,
            dtype=dtype,
            backend_impl=backend_impl,
        )

        def _linear(in_f, out_f):
            return Linear(
                in_features=in_f,
                out_features=out_f,
                bias=moe_config.bias,
                dtype=dtype,
                backend_impl=expert_backend,
            )

        if moe_config.gate:
            self.gate_projs = nn.ModuleList(
                [
                    _linear(moe_config.hidden_size, moe_config.intermediate_size)
                    for _ in range(moe_config.num_experts)
                ]
            )
        self.up_projs = nn.ModuleList(
            [
                _linear(moe_config.hidden_size, moe_config.intermediate_size)
                for _ in range(moe_config.num_experts)
            ]
        )
        self.down_projs = nn.ModuleList(
            [
                _linear(moe_config.intermediate_size, moe_config.hidden_size)
                for _ in range(moe_config.num_experts)
            ]
        )

        if moe_config.num_shared_experts > 0:
            if moe_config.activation == "gptoss_glu":
                raise ValueError(
                    "MoEExperts: shared experts with 'gptoss_glu' are not "
                    "supported (no such model exists)."
                )
            # NATIVE so the raw weights stay available for the fused backend to
            # fold them into the expert stack at preprocess time.
            self.shared_mlps = nn.ModuleList(
                [
                    MergedMLP(
                        in_features=moe_config.hidden_size,
                        out_features=moe_config.shared_intermediate_size,
                        bias=moe_config.bias,
                        activation=moe_config.activation,
                        gate=moe_config.gate,
                        dtype=dtype,
                        backend_impl=BackendType.NATIVE,
                    )
                    for _ in range(moe_config.num_shared_experts)
                ]
            )

        # Shared experts with the same intermediate size can be folded into the
        # fused expert stack (decided by the fused backend's preprocess).
        self.foldable_shared = (
            moe_config.num_shared_experts > 0
            and moe_config.shared_intermediate_size == moe_config.intermediate_size
        )
        # Packed 6D weights + fold flag, populated by the fused backend's
        # preprocess(); None/False until then.
        self._fused_wt_gate = None
        self._fused_wt_up = None
        self._fused_wt_down = None
        self._fused_gate_bias = None
        self._fused_up_bias = None
        self._fused_down_bias = None
        self._folded_shared = False

    def load_expert_weights(
        self,
        expert_idx: int,
        gate_up_weight: Optional[torch.Tensor] = None,
        down_weight: Optional[torch.Tensor] = None,
        gate_up_bias: Optional[torch.Tensor] = None,
        down_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Load HF-format interleaved gate_up weight into the gate/up linears.

        GptOss and similar models store gate+up as one interleaved matrix
        [2*N, H] where even rows -> gate and odd rows -> up. This method handles
        that split so model files need no per-expert logic.

        Args:
            expert_idx:     expert index (0 to num_experts-1)
            gate_up_weight: [2*N, H] interleaved gate+up weight
            down_weight:    [H, N] down-projection weight
            gate_up_bias:   [2*N] interleaved gate+up bias (optional)
            down_bias:      [H] down-projection bias (optional)
        """
        if gate_up_weight is not None:
            if self.gate:
                self.gate_projs[expert_idx].weight.data.copy_(gate_up_weight[::2, :])
                self.up_projs[expert_idx].weight.data.copy_(gate_up_weight[1::2, :])
            else:
                self.up_projs[expert_idx].weight.data.copy_(gate_up_weight)

        if down_weight is not None:
            self.down_projs[expert_idx].weight.data.copy_(down_weight)

        if self.bias_available and gate_up_bias is not None:
            if self.gate:
                self.gate_projs[expert_idx].bias.data.copy_(gate_up_bias[::2])
                self.up_projs[expert_idx].bias.data.copy_(gate_up_bias[1::2])
            else:
                self.up_projs[expert_idx].bias.data.copy_(gate_up_bias)

        if self.bias_available and down_bias is not None:
            self.down_projs[expert_idx].bias.data.copy_(down_bias)

    def _expert_ffn(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """Run expert e's gate/up/down pace linears with the configured activation."""
        up = self.up_projs[e](x)
        if self.activation == "gptoss_glu":
            # Clamped SwiGLU (gpt-oss): (up+1) * (gate.clamp(7) * sigmoid(1.702*gate)).
            g = self.gate_projs[e](x).clamp(max=7.0)
            u = up.clamp(min=-7.0, max=7.0)
            inter = (u + 1) * (g * torch.sigmoid(g * 1.702))
        elif self.gate:
            inter = _POINTWISE_ACTIVATIONS[self.activation](self.gate_projs[e](x)) * up
        else:
            inter = _POINTWISE_ACTIVATIONS[self.activation](up)
        return self.down_projs[e](inter)

    def _run_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dense shared experts, summed (added unscaled to the routed output)."""
        output = torch.zeros_like(hidden_states)
        for mlp in self.shared_mlps:
            output = output + mlp(hidden_states)
        return output

    def route(self, router_logits: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """Select experts from router logits -> (topk_indices, routing_weights).

        Applies the configured routing scheme (via MoERouter) and the routed
        scaling factor. Exposed so a model with a bespoke router front-end can
        drive expert selection itself and then call compute().
        """
        topk_indices, routing_weights = self.router_op(router_logits)
        if self.routed_scaling_factor != 1.0:
            routing_weights = routing_weights * self.routed_scaling_factor
        return topk_indices, routing_weights

    def compute(
        self,
        expert_input: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selected experts over ``expert_input`` and scatter-combine.

        Dispatches to the fused backend when bound, else the reference path.
        ``expert_input`` is [num_tokens, hidden]; routing tensors are
        [num_tokens, topk]. Lets a model route on one tensor and run experts on
        another (Gemma4 routes on the residual, runs experts on a normed copy).
        """
        if self.backend is not None:
            return self._compute_backend(expert_input, topk_indices, routing_weights)
        return self._compute_fallback(expert_input, topk_indices, routing_weights)

    def _compute_fallback(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reference path: run each selected expert and scatter."""
        if self.num_shared_experts > 0:
            output = self._run_shared_experts(hidden_states)
        else:
            output = torch.zeros_like(hidden_states)

        # Iterate only over experts that were actually selected; with large
        # expert pools most experts get no tokens, so scanning all of them wastes
        # O(num_experts * num_tokens) work in this reference path.
        for e in torch.unique(topk_indices).tolist():
            mask = (topk_indices == e).any(dim=-1)
            if not mask.any():
                continue
            expert_out = self._expert_ffn(hidden_states[mask], e)
            for k in range(self.topk):
                km = topk_indices[mask, k] == e
                if not km.any():
                    continue
                w = routing_weights[mask, k][km].unsqueeze(-1)
                output[torch.where(mask)[0][km]] += expert_out[km] * w.to(
                    expert_out.dtype
                )

        return output

    def _compute_backend(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused path: run the expert engine over packed weights."""
        num_tokens = hidden_states.size(0)
        num_experts_kernel = self.num_experts
        if self._folded_shared:
            # Folded shared experts are extra stack entries [num_experts,
            # num_experts + num_shared); each token attends them with weight 1.0
            # (unscaled — routed_scaling_factor applies to routed experts only).
            shared_ids = torch.arange(
                self.num_experts,
                self.num_experts + self.num_shared_experts,
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            ).expand(num_tokens, -1)
            shared_w = torch.ones(
                num_tokens,
                self.num_shared_experts,
                dtype=routing_weights.dtype,
                device=routing_weights.device,
            )
            topk_indices = torch.cat([topk_indices, shared_ids], dim=1)
            routing_weights = torch.cat([routing_weights, shared_w], dim=1)
            output = torch.zeros_like(hidden_states)
            num_experts_kernel = self.num_experts + self.num_shared_experts
        elif self.num_shared_experts > 0:
            output = self._run_shared_experts(hidden_states)
        else:
            output = torch.zeros_like(hidden_states)

        return self.backend.execute(
            output,
            hidden_states,
            topk_indices,
            routing_weights,
            self.activation,
            num_experts_kernel,
            self._fused_wt_gate,
            self._fused_wt_up,
            self._fused_wt_down,
            self._fused_gate_bias,
            self._fused_up_bias,
            self._fused_down_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden] or [num_tokens, hidden]
        Returns:
            Same shape as input.
        """
        if self.router is None:
            raise RuntimeError(
                "MoEExperts was built with build_router=False; call route()/"
                "compute() with externally computed logits instead of forward()."
            )
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        topk_indices, routing_weights = self.route(self.router(flat))
        out = self.compute(flat, topk_indices, routing_weights)
        return out.reshape(orig_shape)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, inter={self.intermediate_size}, "
            f"experts={self.num_experts}, topk={self.topk}, "
            f"act={self.activation}, gate={self.gate}, "
            f"shared={self.num_shared_experts}, backend={self.backend}"
        )
