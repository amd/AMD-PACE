# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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

# Adapted from HuggingFace Transformers Gemma3 implementation
# Phase 2: Implementation using PACE ops

from typing import Callable, Iterable, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    RotaryEmbedding,
    MergedMLP,
)
from pace.ops import Gemma3RMSNorm, FusedGemma3RMSNormResidual
from pace.llm.attention import Attention
from pace.utils.logging import PACE_LLM_WARNING


def _is_sliding_window_layer(layer_idx: int) -> bool:
    """
    Determine if a layer uses sliding window (local) attention.
    Gemma 3 uses a pattern of 5 local layers followed by 1 global layer.
    Layer indices 0-4, 6-10, 12-16, etc. are local (sliding window).
    Layer indices 5, 11, 17, etc. are global.

    Args:
        layer_idx: The index of the layer (0-based).

    Returns:
        True if the layer uses sliding window attention, False for global attention.
    """
    return (layer_idx + 1) % 6 != 0


class Gemma3Attention(nn.Module):
    """
    Gemma 3 attention module supporting both local (sliding window) and global attention.
    Includes QK-Norm (query and key normalization).
    Uses PACE Linear and Attention ops.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        opconfig: OperatorConfig,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine if this is a sliding window layer
        self.is_sliding_window = _is_sliding_window_layer(layer_idx)
        self.sliding_window = (
            getattr(config, "sliding_window", 512) if self.is_sliding_window else None
        )

        # Get dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Attention bias (Gemma 3 doesn't use bias by default)
        self.attention_bias = getattr(config, "attention_bias", False)

        # Scaling factor
        query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None)
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = FusedQKVLinear(
            in_features=self.hidden_size,
            out_features=(self.num_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
            bias=self.attention_bias,
            num_key_value_heads=self.num_key_value_heads,
            backend_impl=opconfig.qkv_projection,
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_bias,
            backend_impl=opconfig.out_projection,
        )

        # QK-Norm: Query and Key normalization (Gemma 3 specific)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = Gemma3RMSNorm(
            self.head_dim, eps=rms_norm_eps, backend_impl=opconfig.norm
        )
        self.k_norm = Gemma3RMSNorm(
            self.head_dim, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
            sliding_window=self.sliding_window or 0,
            scale=self.scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache,
        positions: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        Q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        K = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        V = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # QK-Norm before RoPE (Gemma3 specific)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q, K = position_embeddings(query=Q, key=K, unsqueeze_dim=2)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        return self.o_proj(attn_output)


class Gemma3DecoderLayer(nn.Module):
    """
    Gemma 3 decoder layer with pre-normalization.
    Uses PACE MergedMLP for the feedforward network.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        opconfig: OperatorConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # RMS normalization epsilon
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        # Pre-attention fused add+norm (Gemma3 RMSNorm uses (1 + weight) scaling)
        self.input_layernorm = FusedGemma3RMSNormResidual(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        # Self attention
        self.self_attn = Gemma3Attention(config, layer_idx, opconfig)

        # Post-attention norm (plain, applied before residual add)
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        # Pre-feedforward fused add+norm (Gemma 3 specific)
        self.pre_feedforward_layernorm = FusedGemma3RMSNormResidual(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        # Post-feedforward norm (plain, applied before residual add)
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        # MLP using PACE MergedMLP
        hidden_act = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
        self.mlp = MergedMLP(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,  # Gemma 3 doesn't use bias in MLP
            activation=hidden_act,
            gate=True,  # Gemma 3 uses gated MLP
            backend_impl=opconfig.mlp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        position_embeddings: Callable,
        kv_cache: Union[KVCacheBase, List[KVCacheBase]],
        positions: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, kv_cache, positions, **kwargs
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states, residual


class Gemma3Model(nn.Module):
    """
    Gemma 3 transformer model (without LM head).
    Uses PACE ops for optimized operations.
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings (standard nn.Embedding)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx, opconfig)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm (Gemma3 RMSNorm uses (1 + weight) scaling)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm = Gemma3RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)

        # Global attention RoPE (uses rope_theta, typically 1000000.0)
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=rope_scaling,
            rotary_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=rope_theta,
            backend_impl=opconfig.rope,
        )

        # Local/sliding window attention RoPE (uses rope_local_base_freq, typically 10000.0)
        rope_local_base_freq = getattr(config, "rope_local_base_freq", 10000.0)
        self.rotary_emb_local = RotaryEmbedding(
            rope_scaling=None,  # Local RoPE doesn't use scaling
            rotary_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=rope_local_base_freq,
            backend_impl=opconfig.rope,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> torch.Tensor:
        input_shape = input_ids.shape

        # Gemma normalizes embeddings by sqrt(hidden_size)
        hidden_states = self.embed_tokens(input_ids) * (self.config.hidden_size**0.5)

        is_kv_cache_list = isinstance(kv_cache, list)

        # Compute position embeddings for both global and local RoPE
        position_embeddings_global = self.rotary_emb(hidden_states, positions)
        position_embeddings_local = self.rotary_emb_local(hidden_states, positions)

        if is_kv_cache_list:
            if len(kv_cache) != input_shape[0]:
                raise ValueError(
                    f"Number of KVCache objects ({len(kv_cache)}) must match "
                    f"batch size ({input_shape[0]})"
                )

        residual = torch.zeros_like(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):
            position_embeddings = (
                position_embeddings_local
                if _is_sliding_window_layer(idx)
                else position_embeddings_global
            )

            if is_kv_cache_list:
                layer_kv_caches = [
                    kv_cache_mgr.cache_objects[idx] for kv_cache_mgr in kv_cache
                ]
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    position_embeddings,
                    layer_kv_caches,
                    positions,
                    **kwargs,
                )
            else:
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    position_embeddings,
                    kv_cache.cache_objects[idx],
                    positions,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states + residual)
        return hidden_states


class Gemma3Base(BaseModelForCausalLM):
    """
    Base class for Gemma3 causal LM variants providing common weight loading logic.

    Subclasses should set the following class attributes:
        _tie_embeddings_config_attr: Name of attribute containing config for tie_word_embeddings
        _lm_head_attr: Dot-separated path to lm_head (e.g., "lm_head" or "language_model.lm_head")
        _embed_tokens_attr: Dot-separated path to embed_tokens
        _skip_vision_components: Whether to skip vision-related weights
    """

    # Weight renaming for MergedMLP compatibility
    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    # Subclasses should override these
    _tie_embeddings_config_attr: str = "config"
    _lm_head_attr: str = "lm_head"
    _embed_tokens_attr: str = "model.embed_tokens"
    _skip_vision_components: bool = False

    def _get_nested_attr(self, attr_path: str):
        """Get a nested attribute by dot-separated path (e.g., 'model.embed_tokens')."""
        obj = self
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load weights from checkpoint.
        Common implementation for all Gemma3 model variants.
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # Get config for tie_word_embeddings check
        config = getattr(self, self._tie_embeddings_config_attr)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if tie_word_embeddings:
            lm_head = self._get_nested_attr(self._lm_head_attr)
            embed_tokens = self._get_nested_attr(self._embed_tokens_attr)
            lm_head.weight = embed_tokens.weight

        qkv_cache = {}

        for name, weight in weight_iterator:
            # Skip vision components if applicable
            if self._skip_vision_components and (
                "vision_tower" in name or "multi_modal_projector" in name
            ):
                continue

            # Rename fused params for MergedMLP
            name = self.rename_fused_params(name)

            # Skip rotary embedding buffers (PACE uses cos_cache/sin_cache)
            if "rotary_emb" in name and any(
                buf in name
                for buf in [
                    "inv_freq",
                    "cos_cache",
                    "sin_cache",
                    "cos_cached",
                    "sin_cached",
                ]
            ):
                continue

            # Skip lm_head if tied
            if tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Skip bias if not present
            if name.endswith(".bias") and name not in params_dict:
                if not any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
                    continue

            # Already fused qkv -> load directly
            if "qkv_proj" in name and name in params_dict:
                params_dict[name].data.copy_(weight)
                continue

            # Collect q_proj / k_proj / v_proj for fusion
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
                parts = name.split(".")
                proj_token = parts[-2]
                proj_type = proj_token.split("_")[0]
                attn_prefix = ".".join(parts[:-2])

                if attn_prefix not in qkv_cache:
                    qkv_cache[attn_prefix] = {"weight": {}, "bias": {}}

                if name.endswith(".weight"):
                    qkv_cache[attn_prefix]["weight"][proj_type] = weight
                else:
                    qkv_cache[attn_prefix]["bias"][proj_type] = weight
                continue

            # Handle parameter loading
            if name in params_dict:
                assert params_dict[name].size() == weight.size()
                params_dict[name].data.copy_(weight)

        # Fuse collected q/k/v into each layer's qkv_proj
        modules = dict(self.named_modules())
        for attn_prefix, tensors in qkv_cache.items():
            if not all(x in tensors["weight"] for x in ("q", "k", "v")):
                continue

            fused_layer = modules.get(f"{attn_prefix}.qkv_proj")
            if fused_layer is None:
                continue

            fused_layer.load_from_unfused(tensors)

        qkv_cache.clear()


class Gemma3ForCausalLM(Gemma3Base):
    """
    Gemma 3 model for causal language modeling.
    For text-only Gemma 3 models (e.g., google/gemma-3-270m, google/gemma-3-1b-pt).
    Uses PACE ops for optimized operations.
    """

    # Weight loading configuration (uses defaults from base class)
    _tie_embeddings_config_attr = "config"
    _lm_head_attr = "lm_head"
    _embed_tokens_attr = "model.embed_tokens"
    _skip_vision_components = False

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        self.model = Gemma3Model(config, opconfig)
        self.vocab_size = config.vocab_size

        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> ModelOutput:
        model_output = self.model(input_ids, positions, kv_cache, **kwargs)
        logits = self.lm_head(model_output)
        return ModelOutput(logits=logits)


class Gemma3TextModelWrapper(nn.Module):
    """
    Wrapper to match HuggingFace's Gemma3ForCausalLM structure within Gemma3ForConditionalGeneration.
    HF structure: language_model.model.embed_tokens, language_model.model.layers, etc.
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.model = Gemma3Model(config, opconfig)
        self.vocab_size = config.vocab_size

        # LM head (tied to embeddings in most cases)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )


class Gemma3ForConditionalGeneration(Gemma3Base):
    """
    Gemma 3 model for conditional generation (multimodal).
    For multimodal Gemma 3 models (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it).

    Note: Currently only supports text inputs. Vision inputs are not yet supported.
    Uses PACE ops for optimized operations.

    The main difference from Gemma3ForCausalLM is that this model uses a composite
    config (Gemma3Config) which contains text_config and vision_config, whereas
    Gemma3ForCausalLM uses Gemma3TextConfig directly.
    """

    # Weight loading configuration
    _tie_embeddings_config_attr = "text_config"
    _lm_head_attr = "language_model.lm_head"
    _embed_tokens_attr = "language_model.model.embed_tokens"
    _skip_vision_components = True

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        # For Gemma3ForConditionalGeneration, the config is Gemma3Config
        # which contains text_config (Gemma3TextConfig)
        # We use text_config for the language model
        text_config = getattr(config, "text_config", config)

        # Initialize base class with text_config so self.config has max_position_embeddings etc.
        super().__init__(text_config)

        # Warn that only text inputs are supported
        PACE_LLM_WARNING(
            "Gemma3ForConditionalGeneration currently only supports text inputs. "
            "Vision/image inputs are not yet implemented."
        )

        # Store the full config separately for reference
        self.full_config = config
        self.text_config = text_config
        self.vocab_size = text_config.vocab_size

        # Build the language model using text_config
        # Using wrapper to match HuggingFace's naming: language_model.model.embed_tokens, etc.
        self.language_model = Gemma3TextModelWrapper(text_config, opconfig)

        # Vision encoder is not implemented yet
        # self.vision_tower = None  # Would be SigLIP encoder
        # self.multi_modal_projector = None  # Would project vision to text space

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> ModelOutput:
        model_output = self.language_model.model(
            input_ids, positions, kv_cache, **kwargs
        )
        logits = self.language_model.lm_head(model_output)
        return ModelOutput(logits=logits)
