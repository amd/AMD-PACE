# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn

# pace is imported for torch.ops.pace registration (side effect)
import pace  # noqa: F401
from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import OperatorType, FusedOperatorType, BackendType, DataType


@backend_registry.register(
    OperatorType.LINEAR, BackendType.AOCLDLP, [DataType.BFLOAT16]
)
class AOCLDLPLinear(BackendBase):
    """
    AOCL-DLP backend for linear (fully connected) layers.

    Executes matrix multiplication using AMD AOCL-DLP libraries with bfloat16
    support. Weights are preprocessed (transposed and optionally reordered via
    AOCL-DLP reshape); inputs are normalized to 3D [batch, seq_len, K] for
    the underlying ops.
    """

    def preprocess(self, layer):
        """
        Preprocess the layer weights for AOCL-DLP backend.

        Transposes weights from [N, K] to [K, N] format and optionally applies
        AOCL-DLP optimized reordering based on PACE_USE_AOCL_DLP_RESHAPE env var.

        Args:
            layer: nn.Module with a .weight.data attribute of shape [N, K].
                   Weights are modified in place (replaced with preprocessed Parameter).

        Environment Variable:
            PACE_USE_AOCL_DLP_RESHAPE: Default (unset): enabled. "1": enabled.
                                       "0": disabled. Any other value: warn and
                                       enable.
        """
        weight = layer.weight.data

        # Transpose weight from [N, K] to [K, N] format
        weight = torch.transpose(weight, 0, 1).contiguous()

        # Default 1 (enabled). "1" enabled, "0" disabled. Other: warn and enable.
        _raw = os.getenv("PACE_USE_AOCL_DLP_RESHAPE", "1").strip()
        if _raw == "0":
            use_dlp_reshape = False
        elif _raw == "1":
            use_dlp_reshape = True
        else:
            if _raw:
                warnings.warn(
                    f'PACE_USE_AOCL_DLP_RESHAPE has invalid value "{_raw}"; '
                    'expected "0" or "1". Defaulting to enabled (1).',
                    stacklevel=2,
                )
            use_dlp_reshape = True

        # Apply AOCL-DLP reshape if enabled and dtype is bfloat16
        if use_dlp_reshape and weight.dtype == torch.bfloat16:
            weight = torch.ops.pace.aocl_dlp_reshape_weights(weight)

        # Update layer weight with preprocessed weight. No gradient is required
        # for the preprocessed weight, set to False explicitly.
        layer.weight = nn.Parameter(weight, requires_grad=False)

    def preprocess_input(self, input: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input tensor to ensure it has the correct shape.

        AOCL-DLP operations expect 3D input [batch, seq_len, K]. Stores
        original leading dimensions in self.orig_shape for postprocess_output.

        Args:
            input: Tensor with last dimension K (in_features). May be 2D or 3+ D.

        Returns:
            Tensor of shape [batch, seq_len, K] (3D), view/reshape of input.
        """
        # Store original shape for postprocessing
        self.orig_shape = input.shape[:-1]

        if input.dim() < 3:
            # Reshape to 3D by adding dimensions
            for _ in range(3 - input.dim()):
                input = input.unsqueeze(0)
        elif input.dim() > 3:
            # Flatten all dimensions except the last one
            input = input.reshape(-1, input.size(-1))
            # Then add batch dimension
            input = input.unsqueeze(0)

        return input

    def postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the output tensor to restore the original shape.

        Reshapes the 3D output from the AOCL-DLP op back to the original
        input leading dimensions (e.g. [batch, seq_len, N] or flattened).

        Args:
            output: Tensor of shape [batch, seq_len, N] from execute.

        Returns:
            Tensor with shape (*orig_shape, N) matching the original input.
        """
        # Reshape output to match original input shape
        output = output.reshape(*self.orig_shape, -1)
        return output

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the linear layer with AOCL-DLP (plain matmul, no activation).

        Args:
            input: Input tensor; leading dims are preserved after postprocess.
            weight: Preprocessed weight tensor [K, N] from preprocess().
            bias: Optional bias tensor of shape (N,). Defaults to None.

        Returns:
            Output tensor with shape (*orig_input_dims, N).
        """
        input = self.preprocess_input(input)
        output = torch.ops.pace.aocl_dlp_linear_plain(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARRELU, BackendType.AOCLDLP, [DataType.BFLOAT16]
)
class AOCLDLPFusedLinearRelu(AOCLDLPLinear):
    """
    AOCL-DLP backend for fused linear + ReLU.

    Same as AOCLDLPLinear but applies ReLU after the linear transform.
    Registered for FusedOperatorType.FUSEDLINEARRELU with bfloat16.
    """

    def preprocess(self, layer):
        """Preprocess the underlying linear layer's weights (transpose and optional reshape)."""
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run fused linear + ReLU with AOCL-DLP.

        Args:
            input: Input tensor; leading dims preserved after postprocess.
            weight: Preprocessed weight tensor from preprocess().
            bias: Optional bias tensor. Defaults to None.

        Returns:
            Output tensor with shape (*orig_input_dims, N), ReLU applied.
        """
        input = self.preprocess_input(input)
        output = torch.ops.pace.aocl_dlp_linear_relu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARGELU, BackendType.AOCLDLP, [DataType.BFLOAT16]
)
class AOCLDLPFusedLinearGelu(AOCLDLPLinear):
    """
    AOCL-DLP backend for fused linear + GELU.

    Same as AOCLDLPLinear but applies GELU after the linear transform.
    Registered for FusedOperatorType.FUSEDLINEARGELU with bfloat16.
    """

    def preprocess(self, layer):
        """Preprocess the underlying linear layer's weights (transpose and optional reshape)."""
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run fused linear + GELU with AOCL-DLP.

        Args:
            input: Input tensor; leading dims preserved after postprocess.
            weight: Preprocessed weight tensor from preprocess().
            bias: Optional bias tensor. Defaults to None.

        Returns:
            Output tensor with shape (*orig_input_dims, N), GELU applied.
        """
        input = self.preprocess_input(input)
        output = torch.ops.pace.aocl_dlp_linear_gelu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARSILU, BackendType.AOCLDLP, [DataType.BFLOAT16]
)
class AOCLDLPFusedLinearSilU(AOCLDLPLinear):
    """
    AOCL-DLP backend for fused linear + SiLU (Swish).

    Same as AOCLDLPLinear but applies SiLU after the linear transform.
    Registered for FusedOperatorType.FUSEDLINEARSILU with bfloat16.
    """

    def preprocess(self, layer):
        """Preprocess the underlying linear layer's weights (transpose and optional reshape)."""
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run fused linear + SiLU with AOCL-DLP.

        Args:
            input: Input tensor; leading dims preserved after postprocess.
            weight: Preprocessed weight tensor from preprocess().
            bias: Optional bias tensor. Defaults to None.

        Returns:
            Output tensor with shape (*orig_input_dims, N), SiLU applied.
        """
        input = self.preprocess_input(input)
        output = torch.ops.pace.aocl_dlp_linear_silu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARMUL, BackendType.AOCLDLP, [DataType.BFLOAT16]
)
class AOCLDLPFusedLinearMul(AOCLDLPLinear):
    """
    AOCL-DLP backend for fused linear + element-wise multiply.

    Computes (linear(input) * mul) using AOCL-DLP. Both input and mul are
    preprocessed to 3D; the result is postprocessed to match original input shape.
    Registered for FusedOperatorType.FUSEDLINEARMUL with bfloat16.
    """

    def preprocess(self, layer):
        """Preprocess the underlying linear layer's weights (transpose and optional reshape)."""
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        mul: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run fused linear + element-wise multiply with AOCL-DLP.

        Args:
            input: Input tensor; leading dims preserved after postprocess.
            mul: Tensor to multiply with linear output; same shape semantics as input.
            weight: Preprocessed weight tensor from preprocess().
            bias: Optional bias tensor. Defaults to None.

        Returns:
            Output tensor with shape (*orig_input_dims, N), (linear(input) * mul).
        """
        input = self.preprocess_input(input)
        mul = self.preprocess_input(mul)
        output = torch.ops.pace.aocl_dlp_linear_mul(input, mul, weight, bias)
        return self.postprocess_output(output)
