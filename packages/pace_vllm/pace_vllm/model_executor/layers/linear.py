# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Pace Linear OOT subclasses for vLLM (TPP / libXSMM bf16 backend).

Every `PluggableLayer.register_oot`'d subclass swaps its
`self.quant_method` to `PaceUnquantizedLinearMethod` when the layer is
unquantized. The method packs the `[out, in]` bf16 weight into the 5D
libXSMM layout in `process_weights_after_loading` and routes
`.apply(...)` through `torch.ops.pace.libxsmmlinear_plain`. Shapes
that don't satisfy libXSMM's divisibility (`out % 32 == 0`, `in % 64
== 0`) fall through to vLLM's stock CPU dispatch. LoRA-augmented
Linear subclasses are out of scope.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

from pace_vllm.model_executor.layers.utils import tpp_prepack

logger = init_logger("pace_vllm.model_executor.layers.linear")


class PaceUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Route the matmul through `torch.ops.pace.libxsmmlinear_plain` with
    TPP-packed weights, falling back to stock CPU dispatch for shapes
    libXSMM cannot handle."""

    supports_pre_processed_weights = False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        packed = tpp_prepack(layer.weight.data)
        if packed is None:
            super().process_weights_after_loading(layer)
            layer._pace_use_tpp = False
            # info_once would dedupe on (msg, *args) -- since `prefix` is
            # unique per layer, every layer would still log. Use plain info.
            logger.info(
                "pace-vllm: Linear %s falling back to stock CPU dispatch "
                "(shape %s, dtype %s not TPP-packable).",
                getattr(layer, "prefix", "?"),
                tuple(layer.weight.shape),
                layer.weight.dtype,
            )
            return

        # Stash the packed tensor and drop layer.weight to reclaim memory
        # (matches vLLM's remove_weight=True). Linear's only consumer of
        # layer.weight is apply(), and apply() reads _pace_packed_w on the
        # TPP path -- nothing else looks at the original. The Embedding
        # OOTs deliberately keep layer.weight because forward()'s gather
        # and tied LM heads still read it.
        layer._pace_packed_w = packed
        layer.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        layer._pace_use_tpp = True
        logger.info(
            "pace-vllm: Linear %s packed into TPP layout "
            "(out=%d, in=%d, packed_shape=%s).",
            getattr(layer, "prefix", "?"),
            packed.shape[0] * packed.shape[3],
            packed.shape[1] * 64,
            tuple(packed.shape),
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `getattr` default guards layers whose process_weights_after_loading
        # never ran (swap or shared weights); fall through to stock dispatch.
        if not getattr(layer, "_pace_use_tpp", False):
            return super().apply(layer, x, bias)
        # libXSMM op expects 3D `(B, T, K)`. Pass 3D inputs through; flatten
        # any other rank (1D / 2D / 4D+) into `(1, T, K)` and recover via
        # orig_shape on the way out (matches pace's TPPLinear.preprocess).
        orig_shape = x.shape[:-1]
        x3d = x if x.dim() == 3 else x.reshape(-1, x.shape[-1]).unsqueeze(0)
        out = torch.ops.pace.libxsmmlinear_plain(x3d, layer._pace_packed_w, bias)
        return out.view(*orig_shape, -1)


# PluggableLayer.__new__ keys on the instantiated class's __name__, so
# each Linear subclass needs its own registration. The isinstance()
# guard keeps us off quantized paths (AWQ / GPTQ / compressed-tensors
# own their own quant_method).


@QKVParallelLinear.register_oot
class PaceQKVParallelLinear(QKVParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = PaceUnquantizedLinearMethod()


@MergedColumnParallelLinear.register_oot
class PaceMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = PaceUnquantizedLinearMethod()


@RowParallelLinear.register_oot
class PaceRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = PaceUnquantizedLinearMethod()


@ColumnParallelLinear.register_oot
class PaceColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = PaceUnquantizedLinearMethod()


@ReplicatedLinear.register_oot
class PaceReplicatedLinear(ReplicatedLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = PaceUnquantizedLinearMethod()


__all__ = [
    "PaceUnquantizedLinearMethod",
    "PaceQKVParallelLinear",
    "PaceMergedColumnParallelLinear",
    "PaceRowParallelLinear",
    "PaceColumnParallelLinear",
    "PaceReplicatedLinear",
]
