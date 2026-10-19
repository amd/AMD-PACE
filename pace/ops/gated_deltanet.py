# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Gated DeltaNet (linear attention) operators.

Frontend ops for the Gated DeltaNet primitives, following the standard PACE
operator abstraction (OperatorBase + backend registry):

  - gated delta recurrence   (prefill chunked scan / decode single-step)
  - causal conv1d            (prefill full causal conv / decode cache update)

Each op is cache-agnostic from the model's point of view: it dispatches the
prefill vs decode path internally based on the supplied state and sequence
length, and returns the updated state. This mirrors how PACE abstracts the
attention KV-cache update, so the model issues a single call per primitive
regardless of phase.

The backend is selected by ``backend_impl`` (BackendType): NATIVE runs the
PyTorch reference implementations; JIT runs the optimized AVX-512 C++
kernels. A small :class:`GatedDeltaAttention`
container bundles the primitives so a model can call ``gdn.recurrence(...)``
and ``gdn.conv1d_update(...)`` through one object.
"""

from typing import Optional

from torch import nn

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, BackendType, DataType


class GatedDeltaRecurrence(OperatorBase):
    """Gated delta recurrence over a sequence.

    Dispatches internally: chunked scan for prefill (or any multi-token /
    uncached call) and the single-step recurrence for decode. ``query``/``key``
    are passed UNEXPANDED (num_k_heads); grouped-query expansion is handled by
    the backend. Returns ``(output, new_state)``."""

    @property
    def operator_type(self):
        return OperatorType.GATED_DELTANET_RECURRENCE

    def forward(self, query, key, value, g, beta, state, num_groups=1):
        return self.backend.execute(query, key, value, g, beta, state, num_groups)

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}, backend_impl={self.backend}"


class CausalConv1dUpdate(OperatorBase):
    """Depthwise causal conv1d + SiLU.

    Dispatches internally: full causal convolution for prefill and the
    single-step cache update for decode. Returns ``(output, new_conv_state)``."""

    @property
    def operator_type(self):
        return OperatorType.GATED_DELTANET_CONV1D

    def forward(self, hidden_states, conv_state, weight):
        return self.backend.execute(hidden_states, conv_state, weight)

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}, backend_impl={self.backend}"


class GatedDeltaAttention(nn.Module):
    """Container bundling the Gated DeltaNet primitives under one backend.

    This is the linear-attention operator family (parallel to the standard
    ``Attention`` op) -- not the model-level Gated DeltaNet layer, which owns
    the projections, gated norm and output projection around these primitives.

    Usage in a model::

        self.gdn = GatedDeltaAttention(backend_impl=opconfig.gated_deltanet)
        ...
        x, conv_state = self.gdn.conv1d_update(x, conv_state, weight)
        out, state    = self.gdn.recurrence(q, k, v, g, beta, state, num_groups)

    Both calls are phase-agnostic (prefill vs decode is decided inside the op)
    and return the updated cache state for the caller to store. ``q``/``k`` are
    passed UNEXPANDED (num_k_heads); grouped-query expansion is handled by the
    backend. ``num_groups = num_value_heads // num_key_heads``.
    """

    def __init__(
        self,
        backend_impl: BackendType = BackendType.NATIVE,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.recurrence_op = GatedDeltaRecurrence(
            backend_impl=backend_impl, dtype=dtype
        )
        self.conv1d = CausalConv1dUpdate(backend_impl=backend_impl, dtype=dtype)

    def recurrence(self, query, key, value, g, beta, state, num_groups=1):
        return self.recurrence_op(query, key, value, g, beta, state, num_groups)

    def conv1d_update(self, hidden_states, conv_state, weight):
        return self.conv1d(hidden_states, conv_state, weight)
