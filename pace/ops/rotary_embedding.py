# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of this file has been adapted from Hugging Face Transformers and vLLM
# https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/llama/modeling_llama.py#L82
# https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/modeling_rope_utils.py
# https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/layers/rotary_embedding.py

import math
from functools import partial
from typing import Union, Dict, Any, Tuple, Optional, List, Callable

import torch

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, BackendType
from pace.utils.logging import PACE_ASSERT


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _compute_inv_freq(
    base: Union[int, float],
    rotary_dim: int,
    rescale_factors: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Computes the inverse frequency for the RoPE implementation.

    Args:
        base (int or float): The base frequency for the RoPE.
        rotary_dim (int): The dimensionality of the RoPE.
    Returns:
        torch.Tensor: The inverse frequency tensor.
    """
    if rescale_factors:
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
    else:
        rescale_factors = 1

    return 1.0 / (
        rescale_factors
        * (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    )


def _compute_cos_sin_cache_from_inv_freq(
    inv_freq: torch.Tensor, max_position_embeddings: int, scaling_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cosine and sine cache from the inverse frequency.

    Args:
        inv_freq (torch.Tensor): The inverse frequency tensor.
        max_position_embeddings (int): The maximum number of position embeddings.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and sine caches.
    """
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos() * scaling_factor
    sin = freqs.sin() * scaling_factor
    return (cos, sin)


def _compute_default_cos_sin_cache(
    base: Union[int, float],
    rotary_dim: int,
    max_position_embeddings: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    inv_freq = _compute_inv_freq(base, rotary_dim)
    return _compute_cos_sin_cache_from_inv_freq(
        inv_freq=inv_freq,
        max_position_embeddings=max_position_embeddings,
    )


def _compute_linear_cos_sin_cache(
    base: Union[int, float],
    rotary_dim: int,
    max_position_embeddings: int,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = _compute_inv_freq(base, rotary_dim) / scaling_factor
    return _compute_cos_sin_cache_from_inv_freq(
        inv_freq=inv_freq,
        max_position_embeddings=max_position_embeddings,
        scaling_factor=1.0,  # No additional scaling needed
    )


def _compute_dynamic_ntk_cos_sin_cache(
    base: Union[int, float],
    rotary_dim: int,
    max_position_embeddings: int,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max_position_embeddings * scaling_factor
    base = base * (
        (scaling_factor * max_len / max_position_embeddings) - (scaling_factor - 1)
    ) ** (rotary_dim / (rotary_dim - 2))

    inv_freq = _compute_inv_freq(base, rotary_dim)
    return _compute_cos_sin_cache_from_inv_freq(
        inv_freq=inv_freq,
        max_position_embeddings=max_len,
    )


def _compute_llama3_cos_sin_cache(
    base: Union[int, float],
    rotary_dim: int,
    max_position_embeddings: int,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    inv_freq = _compute_inv_freq(base, rotary_dim)

    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    wave_len = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wave_len - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = 0
    new_freq = torch.where(
        wave_len < high_freq_wavelen,
        inv_freq,
        torch.where(
            wave_len > low_freq_wavelen,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )

    return _compute_cos_sin_cache_from_inv_freq(
        inv_freq=new_freq,
        max_position_embeddings=max_position_embeddings,
    )


def _compute_longrope_cos_sin_cache(
    base: Union[int, float],
    rotary_dim: int,
    max_position_embeddings: int,
    rescale_factor: List[float],
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = _compute_inv_freq(base, rotary_dim, rescale_factors=rescale_factor)
    return _compute_cos_sin_cache_from_inv_freq(
        inv_freq, max_position_embeddings, scaling_factor
    )


def _compute_yarn_cos_sin_cache(
    base: Union[int, float],
    rotary_dim: int,
    max_position_embeddings: int,
    rope_scaling: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    factor = rope_scaling["factor"]
    original_max_position_embeddings = rope_scaling.get(
        "original_max_position_embeddings", max_position_embeddings
    )
    if "original_max_position_embeddings" in rope_scaling:
        factor = max_position_embeddings / original_max_position_embeddings

    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

    def get_mscale(scale, value=1):
        if scale <= 1:
            return 1.0
        return 0.1 * value * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1

    def find_correction_dim(num_rot, dim, base_val, max_pos):
        return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (
            2 * math.log(base_val)
        )

    def find_correction_range(low_rot, high_rot, dim, base_val, max_pos, truncate):
        low = find_correction_dim(low_rot, dim, base_val, max_pos)
        high = find_correction_dim(high_rot, dim, base_val, max_pos)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        return torch.clamp(linear_func, 0, 1)

    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = rope_scaling.get("truncate", True)
    low, high = find_correction_range(
        beta_fast,
        beta_slow,
        rotary_dim,
        base,
        original_max_position_embeddings,
        truncate,
    )
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, rotary_dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return _compute_cos_sin_cache_from_inv_freq(
        inv_freq=inv_freq,
        max_position_embeddings=max_position_embeddings,
        scaling_factor=attention_factor,
    )


class RotaryEmbedding(OperatorBase):
    """
    RotaryEmbedding operator for applying Rotary Position Embeddings (RoPE) to input tensors.
    This operator computes the cosine and sine values for the RoPE based on the provided
    configuration and applies them to the input tensor.

    The forward method returns a either a callable that applies the RoPE to the input tensor
    or the cosine and sine tensors for the RoPE, depending on the `return_cos_sin` argument.

    1. Apply RoPE to query and key tensors using the callable returned by the forward method.
        The callable takes the query and key tensors as input and applies the RoPE to them.
        The input tensors should be in the shape of [batch_size, num_heads, seq_len, head_dim].
        The output tensors will have the same shape as the input tensors.
    2. Get the cosine and sine tensors for RoPE, and apply RoPE outside the operator.

    Args:
        rope_scaling (Dict[str, Any]): Configuration for RoPE scaling, including type and parameters.
        rotary_dim (int): Dimensionality of the RoPE.
        max_position_embeddings (int): Maximum number of position embeddings.
        rope_theta (int): Base frequency for the RoPE.
        partial_rotary_factor (float): Factor for partial RoPE.
        dtype (Optional[DataType]): Data type for the operator.
        backend_impl (BackendType): Backend implementation type.
        **rope_kwargs: Additional keyword arguments for RoPE configuration.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine tensors for the RoPE applied to the input tensor.

    Example:

        >>> query_states = torch.randn(1, 10, rotary_dim)
        >>> key_states = torch.randn(1, 10, rotary_dim)
        >>> x = torch.randn(1, 10, rotary_dim)
        >>> position_ids = torch.arange(10).unsqueeze(0)
        >>> rope_scaling = {"rope_type": "default", "scaling_factors": 1.0}
        >>> rotary_dim = 64
        >>> max_position_embeddings = 2048
        >>> rope_theta = 10000
        >>> rotary_embedding = RotaryEmbedding(rope_scaling, rotary_dim, max_position_embeddings, rope_theta)

        RotaryEmbedding can be used in two ways:
        >>> # 1. Apply RoPE to query and key tensors
        >>> positional_embedding = rotary_embedding(x, position_ids)
        >>> query_states, key_states = positional_embedding(query=query_states, key=key_states)

        >>> # 2. Get the cosine and sine tensors for RoPE
        >>> positional_embedding = rotary_embedding(x, position_ids, return_cos_sin=True)
        >>> cos, sin = positional_embedding
        >>> query_states = rotary_embedding.apply_rotary_emb(query_states, key_states, cos, sin)  # Apply RoPE outside the operator

    """

    @property
    def operator_type(self) -> OperatorType:
        return OperatorType.ROTARYEMBEDDING

    def __init__(
        self,
        rope_scaling: Dict[str, Any],
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: int,
        partial_rotary_factor: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        backend_impl: Optional[BackendType] = None,  # unused for RoPE
        **rope_kwargs,
    ):

        super().__init__(
            backend_impl=backend_impl, dtype=dtype
        )  # Do not use backend_impl

        if partial_rotary_factor < 1.0:
            rotary_dim = int(rotary_dim * partial_rotary_factor)

        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_kwargs = rope_kwargs

        self.rope_scaling = rope_scaling
        if rope_scaling is not None:  # Compatibility with old configs
            self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        if self.rope_type == "default":
            cache = _compute_default_cos_sin_cache(
                base=self.rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
            )
        elif self.rope_type == "linear":
            # Support both "factor" (HuggingFace convention) and "scaling_factor"
            scaling_factor = rope_scaling.get(
                "scaling_factor", rope_scaling.get("factor", 1.0)
            )
            cache = _compute_linear_cos_sin_cache(
                base=self.rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif self.rope_type == "dynamic":
            # Support both "factor" and "scaling_factor"
            scaling_factor = rope_scaling.get(
                "scaling_factor", rope_scaling.get("factor", 1.0)
            )
            cache = _compute_dynamic_ntk_cos_sin_cache(
                base=self.rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif self.rope_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position_embeddings = rope_scaling[
                "original_max_position_embeddings"
            ]
            cache = _compute_llama3_cos_sin_cache(
                base=self.rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_max_position_embeddings=original_max_position_embeddings,
            )
        elif self.rope_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            attention_factor = rope_scaling.get("attention_factor")
            original_max_position_embeddings = rope_kwargs.get(
                "original_max_position_embeddings", max_position_embeddings
            )
            self.original_max_position_embeddings = original_max_position_embeddings

            # Compute the base frequency for the short and long RoPE
            scale = self.max_position_embeddings / original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(
                    1 + math.log(scale) / math.log(original_max_position_embeddings)
                )
            scaling_factor = (
                attention_factor if attention_factor is not None else scaling_factor
            )
            short_cache = _compute_longrope_cos_sin_cache(
                base=rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=original_max_position_embeddings,
                rescale_factor=short_factor,
                scaling_factor=scaling_factor,
            )
            long_cache = _compute_longrope_cos_sin_cache(
                base=rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                rescale_factor=long_factor,
                scaling_factor=scaling_factor,
            )
            cache = (
                torch.cat([short_cache[0], long_cache[0]], dim=0),
                torch.cat([short_cache[1], long_cache[1]], dim=0),
            )
        elif self.rope_type == "yarn":
            cache = _compute_yarn_cos_sin_cache(
                base=self.rope_theta,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                rope_scaling=rope_scaling,
            )
        else:
            PACE_ASSERT(False, f"Unsupported RoPE type: {self.rope_type}")

        self.register_buffer("cos_cache", cache[0], persistent=False)
        self.register_buffer("sin_cache", cache[1], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        return_cos_sin: bool = False,
        offsets: Optional[torch.Tensor] = None,
    ) -> Union[Callable, Tuple[torch.Tensor, torch.Tensor]]:

        # Since just indexing is done, we do not implement a custom backend
        # @TODO: have a method which applies the rotary embedding to the input tensor
        # and returns the result, instead of just returning the cos and sin tensors
        if self.rope_type == "longrope":
            k = self.original_max_position_embeddings
            # Per-sequence check: only offset rows where any position exceeds k
            is_long = torch.any(position_ids > k, dim=-1, keepdim=True).float()
            long_prompt_offset = (is_long * k).long().expand_as(position_ids)
            idx = torch.add(position_ids, long_prompt_offset)
            idx = torch.add(idx, offsets) if offsets is not None else idx
            idx = idx.flatten()
        else:
            idx = position_ids.flatten()

        cos = self.cos_cache.index_select(0, idx).unflatten(0, position_ids.shape)
        sin = self.sin_cache.index_select(0, idx).unflatten(0, position_ids.shape)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        if return_cos_sin:
            return cos, sin

        return partial(
            self.apply_rotary_emb,
            cos=cos,
            sin=sin,
            unsqueeze_dim=1,
        )

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int,
        is_neox_style: bool,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        if self.rope_type == "longrope":
            # For longrope, we need to repeat the cos and sin tensors
            # to match the input tensor's shape
            cos = cos.repeat(1, 1, 1, 2)
            sin = sin.repeat(1, 1, 1, 2)

            x_rot = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            x_rot = x_rot * cos + _rotate_neox(x_rot) * sin
            return torch.cat((x_rot, x_pass), dim=-1)
        else:
            if is_neox_style:
                x1, x2 = torch.chunk(x, 2, dim=-1)
            else:
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if is_neox_style:
                return torch.cat((o1, o2), dim=-1)
            else:
                return torch.stack((o1, o2), dim=-1).flatten(-2)

    def apply_rotary_emb(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
        is_neox_style: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary embeddings to the query and key tensors.
        NOTE: Make sure to call this method after splitting the query and key
        tensors into their respective heads and transposing them.
            input shape: [batch_size, num_heads, seq_len, head_dim]

        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim].
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim].
            cos: Cosine tensor of shape [seq_len, head_dim // 2].
            sin: Sine tensor of shape [seq_len, head_dim // 2].
            unsqueeze_dim: The dimension to unsqueeze the cosine and sine tensors.
            is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
                positional embeddings.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with
                rotary embeddings applied.
        """
        query = query.contiguous()
        key = key.contiguous()
        if (
            is_neox_style
            and self.rope_type != "longrope"
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and cos.dtype == torch.bfloat16
            and sin.dtype == torch.bfloat16
            and query.dim() == 4
            and key.dim() == 4
            and unsqueeze_dim in (1, 2)
        ):
            try:
                return torch.ops.pace.fused_rope_apply(
                    query, key, cos, sin, unsqueeze_dim
                )
            except Exception:
                pass

        query = self._apply_rotary_emb(query, cos, sin, unsqueeze_dim, is_neox_style)
        key = self._apply_rotary_emb(key, cos, sin, unsqueeze_dim, is_neox_style)
        return query, key

    def extra_repr(self):
        return (
            f"rope_type={self.rope_type}, "
            f"rotary_dim={self.rotary_dim}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"rope_theta={self.rope_theta}, "
            f"extra_rope_kwargs={self.rope_kwargs}, "
            f"dtype={self.dtype}, "
            f"backend={self.backend}"
        )
