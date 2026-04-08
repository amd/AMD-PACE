# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

import torch
from torch.nn import Parameter

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, BackendType, DataType


class Linear(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.LINEAR

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.bias_available = bias
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        weight = Parameter(
            torch.empty(out_features, in_features, dtype=self.dtype.to_torch_dtype()),
            requires_grad=False,
        )
        if self.bias_available:
            bias = Parameter(
                torch.empty(out_features, dtype=self.dtype.to_torch_dtype()),
                requires_grad=False,
            )
        else:
            bias = None

        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(x, self.weight, self.bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_available}, "
            f"dtype={self.dtype}, "
            f"backend_impl={self.backend}"
        )


class RepeatedKVLinear(Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        num_key_value_heads: int = 8,
        backend_impl: BackendType = BackendType.NATIVE,
    ):

        super().__init__(in_features, out_features, bias, dtype, backend_impl)

        self.num_key_value_heads = num_key_value_heads
        # Setting a new attribute to the weight object.
        # This is a workaround to allow the weight to have a custom load_weights method.
        self.weight.load_weights = self.load_weights
        if self.bias_available:
            self.bias.load_weights = self.load_bias

    # The only difference from the Linear is how weights are loaded.
    # The weights are reshaped and repeated to match the expected shape.
    def load_weights(self, param: Parameter, loaded_weight: torch.Tensor):
        if param.size() != loaded_weight.size():
            out_channels, in_channels = loaded_weight.shape
            n_rep = param.shape[0] // loaded_weight.shape[0]
            loaded_weight = (
                loaded_weight.reshape(
                    self.num_key_value_heads,
                    out_channels // self.num_key_value_heads,
                    in_channels,
                )
                .repeat_interleave(n_rep, 0)
                .reshape(n_rep * out_channels, in_channels)
            )

        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def load_bias(self, param: Parameter, loaded_bias: torch.Tensor):
        if param.size() != loaded_bias.size():
            n_rep = param.shape[0] // loaded_bias.shape[0]
            loaded_bias = loaded_bias.repeat_interleave(n_rep, 0)

        assert param.size() == loaded_bias.size()
        param.data.copy_(loaded_bias)


class FusedQKVLinear(Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        num_key_value_heads: int = 8,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(in_features, out_features, bias, dtype, backend_impl)
        self.num_key_value_heads = num_key_value_heads

    @torch.no_grad()
    def load_from_unfused(self, qkv_tensors: dict[str, dict[str, torch.Tensor]]):

        w = qkv_tensors["weight"]
        b = qkv_tensors.get("bias", {})

        q_w, k_w, v_w = w["q"], w["k"], w["v"]

        q_b = b.get("q")
        k_b = b.get("k")
        v_b = b.get("v")

        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        assert (
            fused_w.shape == self.weight.shape
        ), f"Fused weight mismatch: {fused_w.shape} != {self.weight.shape}"
        self.weight.copy_(fused_w)

        if self.bias is not None:
            dtype = q_w.dtype
            if q_b is None:
                q_b = torch.zeros(q_w.size(0), dtype=dtype, device=self.weight.device)
            if k_b is None:
                k_b = torch.zeros(k_w.size(0), dtype=dtype, device=self.weight.device)
            if v_b is None:
                v_b = torch.zeros(v_w.size(0), dtype=dtype, device=self.weight.device)

            fused_b = torch.cat([q_b, k_b, v_b], dim=0)
            assert (
                fused_b.shape == self.bias.shape
            ), f"Bias mismatch: {fused_b.shape} != {self.bias.shape}"
            self.bias.copy_(fused_b)
