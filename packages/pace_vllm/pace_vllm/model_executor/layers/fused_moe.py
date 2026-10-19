# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PACE FusedMoE OOT for vLLM's CPU path."""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

try:
    from vllm.model_executor.layers.fused_moe.router.cpu_router import select_experts
except ImportError:  # vLLM 0.28
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import select_experts

from pace_vllm.model_executor.layers.utils import tpp_prepack_moe

logger = init_logger("pace_vllm.model_executor.layers.fused_moe")

# Avoid importing pace: pace_vllm already loaded the bundled library, and a
# second load would double-register the Torch operators.
_PACE_SOFTMAX_ROUTER = "pace::moe_route_softmax"

# Values: auto, pace (validated signatures only), or upstream.
_ROUTER_OVERRIDE = os.environ.get("PACE_VLLM_ROUTER", "auto").lower()

_VLLM_TO_PACE_ACTIVATION = {"swigluoai": "gptoss_glu"}

# The C++ kernel hardcodes vLLM's CPU defaults; layer overrides are rejected.
_GPTOSS_ALPHA = 1.702
_GPTOSS_LIMIT = 7.0


def _resolve_activation_str(layer) -> str:
    """Map vLLM's `MoEActivation` enum / string to pace's op-string."""
    raw = getattr(layer, "activation", "silu")
    if hasattr(raw, "value"):
        raw = raw.value
    name = str(raw).lower()
    return _VLLM_TO_PACE_ACTIVATION.get(name, name)


@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class PaceUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Route vLLM's CPU MoE through PACE."""

    supports_pre_processed_weights = False
    _SUPPORTED_ACTIVATIONS = frozenset({"silu", "gelu", "relu", "gptoss_glu"})

    # SwigluOAI checkpoints alternate gate/up rows; other supported activations
    # use contiguous halves.
    _INTERLEAVED_GATE_UP_ACTIVATIONS = frozenset({"gptoss_glu"})

    @classmethod
    def _gate_up_interleaved(cls, act: str) -> bool:
        """Return whether w13's gate/up rows are interleaved for `act`."""
        if act in cls._INTERLEAVED_GATE_UP_ACTIVATIONS:
            return True
        if act in cls._SUPPORTED_ACTIVATIONS:
            return False
        raise NotImplementedError(
            f"PaceFusedMoE cannot determine the w13 gate/up layout for "
            f"activation {act!r}; known activations are "
            f"{sorted(cls._SUPPORTED_ACTIVATIONS)}. Refusing to guess the "
            "split (interleaved vs contiguous). Disable VLLM_PLUGINS=pace for "
            "this model."
        )

    def process_weights_after_loading(self, layer) -> None:
        if not getattr(self.moe, "is_act_and_mul", True):
            super().process_weights_after_loading(layer)
            layer._pace_use_fused_moe = False
            logger.info(
                "pace-vllm: FusedMoE %s falling back to stock CPU dispatch "
                "(non-gated activation; gate/up split not applicable).",
                getattr(layer, "prefix", "?"),
            )
            return

        w13 = layer.w13_weight.data
        e, two_inter, hidden = w13.shape
        inter = two_inter // 2
        act = _resolve_activation_str(layer)
        if self._gate_up_interleaved(act):
            w_gate = w13[:, 0::2, :].contiguous()
            w_up = w13[:, 1::2, :].contiguous()
        else:
            w_gate = w13[:, :inter, :].contiguous()
            w_up = w13[:, inter:, :].contiguous()
        wt_gate = tpp_prepack_moe(w_gate)
        wt_up = tpp_prepack_moe(w_up)
        wt_down = tpp_prepack_moe(layer.w2_weight.data)
        if wt_gate is None or wt_up is None or wt_down is None:
            super().process_weights_after_loading(layer)
            layer._pace_use_fused_moe = False
            logger.info(
                "pace-vllm: FusedMoE %s falling back to stock CPU dispatch "
                "(shape %s not TPP-packable).",
                getattr(layer, "prefix", "?"),
                tuple(w13.shape),
            )
            return
        layer._pace_wt_gate = wt_gate
        layer._pace_wt_up = wt_up
        layer._pace_wt_down = wt_down
        gate_bias = up_bias = down_bias = None
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        if (
            getattr(self.moe, "has_bias", False)
            and w13_bias is not None
            and w2_bias is not None
        ):
            wb = w13_bias.data
            if self._gate_up_interleaved(act):
                gate_bias = wb[:, 0::2]
                up_bias = wb[:, 1::2]
            else:
                gate_bias = wb[:, :inter]
                up_bias = wb[:, inter:]
            gate_bias = gate_bias.to(w13.dtype).contiguous()
            up_bias = up_bias.to(w13.dtype).contiguous()
            down_bias = w2_bias.data.to(w13.dtype).contiguous()
        layer._pace_gate_bias = gate_bias
        layer._pace_up_bias = up_bias
        layer._pace_down_bias = down_bias
        layer._pace_use_fused_moe = True
        # Resolve eagerly because enum inspection causes a Dynamo graph break.
        layer._pace_activation_str = act
        # Keep router selection outside the compiled forward.
        layer._pace_cpp_router = (
            self._decide_router(layer) if self.is_monolithic else None
        )
        # vLLM may still read the original weights after this hook.
        logger.info(
            "pace-vllm: FusedMoE %s packed into TPP layout "
            "(experts=%d, intermediate=%d, hidden=%d, bias=%s, router=%s).",
            getattr(layer, "prefix", "?"),
            e,
            inter,
            hidden,
            gate_bias is not None,
            layer._pace_cpp_router or "vllm:select_experts",
        )

    @staticmethod
    def _has_validated_cpp_router(layer) -> bool:
        """Return whether PACE can preserve this layer's routing semantics."""
        scoring = getattr(layer, "scoring_func", "softmax")
        grouped = getattr(layer, "use_grouped_topk", False)
        renorm = getattr(layer, "renormalize", True)
        custom = getattr(layer, "custom_routing_function", None)
        scale = getattr(layer, "routed_scaling_factor", 1.0)
        return (
            scoring == "softmax"
            and not grouped
            and renorm
            and custom is None
            and float(scale) == 1.0
        )

    def _decide_router(self, layer):
        """Return the PACE router op name, or None to defer to vLLM."""
        cpp = _PACE_SOFTMAX_ROUTER if self._has_validated_cpp_router(layer) else None
        if _ROUTER_OVERRIDE == "upstream":
            return None
        if _ROUTER_OVERRIDE == "pace":
            if cpp is None:
                raise NotImplementedError(
                    "PACE_VLLM_ROUTER=pace forced, but this layer's routing "
                    "signature has no validated PACE C++ router. Unset the "
                    "override or use a softmax-routed MoE."
                )
            return cpp
        return cpp

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        # Only the upstream fallback consumes input_ids.
        if not getattr(layer, "_pace_use_fused_moe", False):
            return super().apply_monolithic(layer, x, router_logits, input_ids)

        x2d = x.reshape(-1, x.shape[-1]).contiguous()

        cpp_router = getattr(layer, "_pace_cpp_router", None)
        if cpp_router == _PACE_SOFTMAX_ROUTER:
            topk_ids, topk_weights = torch.ops.pace.moe_route_softmax(
                router_logits.contiguous(), int(layer.top_k)
            )
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x2d,
                router_logits=router_logits,
                top_k=int(layer.top_k),
                use_grouped_topk=getattr(layer, "use_grouped_topk", False),
                renormalize=getattr(layer, "renormalize", True),
                topk_group=getattr(layer, "topk_group", None),
                num_expert_group=getattr(layer, "num_expert_group", None),
                custom_routing_function=getattr(layer, "custom_routing_function", None),
                scoring_func=getattr(layer, "scoring_func", "softmax"),
                routed_scaling_factor=getattr(layer, "routed_scaling_factor", 1.0),
                e_score_correction_bias=getattr(layer, "e_score_correction_bias", None),
            )
        return self._apply_pace_experts(layer, x, topk_weights, topk_ids)

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        if not getattr(layer, "_pace_use_fused_moe", False):
            return super().apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        return self._apply_pace_experts(layer, x, topk_weights, topk_ids)

    def _apply_pace_experts(self, layer, x, topk_weights, topk_ids):
        act = getattr(layer, "_pace_activation_str", None)
        if act is None:
            act = _resolve_activation_str(layer)
        if act not in self._SUPPORTED_ACTIVATIONS:
            raise NotImplementedError(
                f"PaceFusedMoE supports activations "
                f"{sorted(self._SUPPORTED_ACTIVATIONS)}; got {act!r}."
            )
        if act == "gptoss_glu":
            self._check_gptoss_glu_params(layer)

        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        topk_ids = topk_ids.to(torch.int64).contiguous()
        topk_weights = topk_weights.to(torch.float32).contiguous()
        num_experts = int(
            getattr(layer, "global_num_experts", None) or layer.num_experts
        )
        output = torch.zeros_like(x2d)
        torch.ops.pace.libxsmm_moe_engine(
            output,
            x2d,
            topk_ids,
            topk_weights,
            layer._pace_wt_gate,
            layer._pace_wt_up,
            layer._pace_wt_down,
            getattr(layer, "_pace_gate_bias", None),
            getattr(layer, "_pace_up_bias", None),
            getattr(layer, "_pace_down_bias", None),
            act,
            num_experts,
        )
        return output.reshape(orig_shape)

    def _check_gptoss_glu_params(self, layer) -> None:
        """Reject parameters that differ from the hardcoded kernel values."""
        alpha = getattr(layer, "gemm1_alpha", getattr(layer, "swiglu_alpha", None))
        limit = getattr(
            layer,
            "gemm1_clamp_limit",
            getattr(layer, "swiglu_limit", None),
        )
        if alpha is not None and float(alpha) != _GPTOSS_ALPHA:
            raise NotImplementedError(
                f"PaceFusedMoE gptoss_glu hardcodes alpha={_GPTOSS_ALPHA}; "
                f"layer requests gemm1_alpha={float(alpha)}. Disable "
                "VLLM_PLUGINS=pace for this model."
            )
        if limit is not None and float(limit) != _GPTOSS_LIMIT:
            raise NotImplementedError(
                f"PaceFusedMoE gptoss_glu hardcodes limit={_GPTOSS_LIMIT}; "
                f"layer requests gemm1_clamp_limit={float(limit)}. Disable "
                "VLLM_PLUGINS=pace for this model."
            )


__all__ = [
    "PaceUnquantizedFusedMoEMethod",
    "_resolve_activation_str",
]
