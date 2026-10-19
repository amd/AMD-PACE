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
    OperatorType.RMSNORM_GATED,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeRMSNormGated(BackendBase):
    """RMSNorm followed by a SiLU gate.

    The fp32 normalization + gating order is kept explicit (rather than
    F.rms_norm) to preserve the exact dtype flow of the reference
    implementation."""

    def execute(
        self,
        input: torch.Tensor,
        gate: torch.Tensor,
        normalized_shape: Tuple[int],
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        input_dtype = input.dtype
        x = input.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


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


@backend_registry.register(
    OperatorType.GATED_DELTANET_RECURRENCE,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeGatedDeltaRecurrence(BackendBase):
    """Gated delta recurrence. Dispatches prefill (chunked scan) vs decode
    (single-step recurrence) internally based on ``state`` and sequence length.
    Returns ``(output, new_state)``."""

    def execute(self, query, key, value, g, beta, state, num_groups=1):
        query, key = self._expand_qk(query, key, num_groups)
        # query: [B, T, H, D] -> decode when continuing a cached sequence.
        if state is not None and query.shape[1] == 1:
            return self._recurrent_rule(query, key, value, g, beta, state)
        return self._chunk_rule(query, key, value, g, beta, state)

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    @staticmethod
    def _expand_qk(query, key, num_groups):
        # Grouped-query expansion: value head h reads key/query head
        # h // num_groups.
        if num_groups > 1:
            query = query.repeat_interleave(num_groups, dim=2)
            key = key.repeat_interleave(num_groups, dim=2)
        return query, key

    @staticmethod
    def _recurrent_rule(query, key, value, g, beta, initial_state):
        """Single-token (decode) gated delta recurrence. Returns
        (output, state)."""
        initial_dtype = query.dtype
        query = NativeGatedDeltaRecurrence._l2norm(query, dim=-1, eps=1e-6)
        key = NativeGatedDeltaRecurrence._l2norm(key, dim=-1, eps=1e-6)

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        core_attn_out = torch.zeros(
            batch_size,
            num_heads,
            sequence_length,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        state = (
            torch.zeros(
                batch_size,
                num_heads,
                k_head_dim,
                v_head_dim,
                dtype=value.dtype,
                device=value.device,
            )
            if initial_state is None
            else initial_state.to(value)
        )

        for i in range(sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, state

    @staticmethod
    def _chunk_rule(query, key, value, g, beta, initial_state, chunk_size=64):
        """Chunked (prefill) gated delta scan. Returns (output, state)."""
        initial_dtype = query.dtype
        query = NativeGatedDeltaRecurrence._l2norm(query, dim=-1, eps=1e-6)
        key = NativeGatedDeltaRecurrence._l2norm(key, dim=-1, eps=1e-6)

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        state = (
            torch.zeros(
                batch_size,
                num_heads,
                k_head_dim,
                v_head_dim,
                dtype=value.dtype,
                device=value.device,
            )
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_c = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = k_cumdecay[:, :, i] @ state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
            core_attn_out[:, :, i] = attn_inter + attn_c @ v_new
            state = (
                state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, state


@backend_registry.register(
    OperatorType.GATED_DELTANET_CONV1D,
    BackendType.NATIVE,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class NativeCausalConv1dUpdate(BackendBase):
    """Depthwise causal conv1d + SiLU. Dispatches prefill (full causal conv)
    vs decode (single-step cache update) internally. Returns
    ``(output, new_conv_state)``."""

    def execute(self, hidden_states, conv_state, weight):
        # hidden_states: [B, C, T] -> decode when continuing a cached sequence.
        if conv_state is not None and hidden_states.shape[-1] == 1:
            out = self._decode(hidden_states, conv_state, weight)
            return out, conv_state
        return self._prefill(hidden_states, conv_state, weight)

    @staticmethod
    def _decode(hidden_states, conv_state, weight):
        """Decode step: advance an existing conv cache by the new token(s).
        Mutates ``conv_state`` in place; returns the activated output."""
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]
        hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(
            weight.dtype
        )
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
        out = F.conv1d(
            hidden_states_new, weight.unsqueeze(1), None, padding=0, groups=hidden_size
        )
        out = F.silu(out[:, :, -seq_len:])
        return out.to(hidden_states.dtype)

    @staticmethod
    def _prefill(hidden_states, conv_state, weight):
        """Prefill (and chunked continuation): full causal conv + SiLU, and
        build the conv cache for the next step. Returns
        (activated_output, new_conv_state).

        When ``conv_state`` is provided (continuation of a previously cached
        sequence) it is prepended and the extra leading outputs are trimmed, so
        the returned output always has the original ``seq_len`` width."""
        kernel_size = weight.shape[-1]
        hidden_size = hidden_states.shape[1]
        had_cache = conv_state is not None
        if had_cache:
            hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
        width = hidden_states.shape[-1]
        new_conv_state = F.pad(hidden_states, (kernel_size - width, 0))[
            :, :, -(kernel_size - 1) :
        ]
        out = F.conv1d(
            hidden_states,
            weight.unsqueeze(1),
            None,
            padding=kernel_size - 1,
            groups=hidden_size,
        )
        out = F.silu(out[:, :, :width])
        if had_cache:
            out = out[:, :, -(width - (kernel_size - 1)) :]
        return out, new_conv_state
