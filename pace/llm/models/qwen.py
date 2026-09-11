# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/qwen2/modeling_qwen2.py
# Unified implementation supporting both Qwen2 and Qwen3 dense architectures.
# Qwen3 differs only in: no QKV bias, and QK-Norm (RMSNorm on Q/K heads before RoPE).

from typing import Tuple, Iterable, Callable, Union, List

import torch
from torch import nn
from transformers import PretrainedConfig

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    RMSNorm,
    FusedRMSNormResidual,
    RotaryEmbedding,
    MergedMLP,
)
from pace.llm.attention import Attention


class QwenAttention(nn.Module):

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()

        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        qkv_bias = getattr(config, "attention_bias", True)

        self.qkv_proj = FusedQKVLinear(
            in_features=config.hidden_size,
            out_features=(config.num_attention_heads + 2 * config.num_key_value_heads)
            * self.head_dim,
            bias=qkv_bias,
            num_key_value_heads=config.num_key_value_heads,
            backend_impl=opconfig.qkv_projection,
        )

        # QK-Norm: RMSNorm on head vectors (Qwen3), Identity no-op (Qwen2).
        # Keyed on model_type rather than bias so custom variants are handled.
        has_qk_norm = getattr(config, "model_type", "") in ("qwen3", "qwen3_moe")
        self.q_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps, backend_impl=opconfig.norm)
            if has_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps, backend_impl=opconfig.norm)
            if has_qk_norm
            else nn.Identity()
        )

        self.attn = Attention(
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
        )

        self.o_proj = Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            backend_impl=opconfig.out_projection,
        )

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache,
        positions: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_len = input_shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        Q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        K = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        V = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q, K = position_embeddings(Q, K, unsqueeze_dim=2)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1)

        return self.o_proj(attn_output)


class QwenDecoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.self_attn = QwenAttention(config, opconfig)
        self.mlp = MergedMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            activation=config.hidden_act,
            gate=True,
            backend_impl=opconfig.mlp,
        )
        self.input_layernorm = FusedRMSNormResidual(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.post_attention_layernorm = FusedRMSNormResidual(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
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

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class QwenModel(nn.Module):

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(config, opconfig)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=config.rope_scaling,
            rotary_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_parameters["rope_theta"],
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

        hidden_states = self.embed_tokens(input_ids)

        is_kv_cache_list = isinstance(kv_cache, list)

        position_embeddings: Callable = self.rotary_emb(hidden_states, positions)

        if is_kv_cache_list:
            if len(kv_cache) != input_shape[0]:
                raise ValueError(
                    f"Number of KVCache objects ({len(kv_cache)}) must match "
                    f"batch size ({input_shape[0]})"
                )

        residual = torch.zeros_like(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):
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


class QwenForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        self.model = QwenModel(config, opconfig)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):

        params = dict(self.named_parameters(remove_duplicate=False))

        qkv_cache = {}

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        for name, tensor in weight_iterator:
            name = self.rename_fused_params(name)
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if name.endswith(".bias") and name not in params:
                if not any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
                    continue

            # Already fused qkv -> load directly
            if "qkv_proj" in name and name in params:
                params[name].data.copy_(tensor)
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
                    qkv_cache[attn_prefix]["weight"][proj_type] = tensor
                else:
                    qkv_cache[attn_prefix]["bias"][proj_type] = tensor
                continue

            if name in params:
                if hasattr(params[name], "load_weights"):
                    params[name].load_weights(params[name], tensor)
                else:
                    params[name].data.copy_(tensor)

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


# Aliases so the model registry can map each HF architecture to its class
Qwen2ForCausalLM = QwenForCausalLM
Qwen3ForCausalLM = QwenForCausalLM
