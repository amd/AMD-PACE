# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/gptj/modeling_gptj.py

from typing import Tuple, Iterable, Union, List

import torch
from torch import nn
from transformers import GPTJConfig

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.ops import (
    Linear,
    LayerNorm,
    FusedLayerNormResidual,
    RotaryEmbedding,
    MergedMLP,
)
from pace.llm.attention import Attention


class GPTJAttention(nn.Module):

    def __init__(self, config: GPTJConfig, opconfig: OperatorConfig):

        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads

        self.k_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )
        self.v_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )

        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=(
                config.rope_scaling if hasattr(config, "rope_scaling") else None
            ),
            rotary_dim=pos_embd_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 10000),
            backend_impl=opconfig.rope,
        )
        self.attn = Attention(
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
        )
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            backend_impl=opconfig.out_projection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        kv_cache,
        **kwargs,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        Q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        K = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        V = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )

        position_embeddings = self.rotary_emb(hidden_states, positions)
        if self.rotary_dim is not None:
            q_rot, q_pass = Q[..., : self.rotary_dim], Q[..., self.rotary_dim :]
            k_rot, k_pass = K[..., : self.rotary_dim], K[..., self.rotary_dim :]
            q_rot, k_rot = position_embeddings(
                q_rot, k_rot, unsqueeze_dim=2, is_neox_style=False
            )
            Q = torch.cat([q_rot, q_pass], dim=-1)
            K = torch.cat([k_rot, k_pass], dim=-1)
        else:
            Q, K = position_embeddings(Q, K, unsqueeze_dim=2, is_neox_style=False)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        return self.out_proj(attn_output)


class GPTJBlock(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        opconfig: OperatorConfig,
    ):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = FusedLayerNormResidual(
            config.n_embd, eps=config.layer_norm_epsilon, backend_impl=opconfig.norm
        )
        self.attn = GPTJAttention(config, opconfig)
        self.mlp = MergedMLP(
            config.n_embd,
            inner_dim,
            bias=True,
            activation=config.activation_function,
            backend_impl=opconfig.mlp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheBase, List[KVCacheBase]],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.ln_1(hidden_states, residual)
        attn_output = self.attn(hidden_states, positions, kv_cache, **kwargs)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states

        return hidden_states, residual


class GPTJModel(nn.Module):

    def __init__(self, config: GPTJConfig, opconfig: OperatorConfig):

        super().__init__()

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.h = nn.ModuleList(
            [GPTJBlock(config, opconfig) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon, backend_impl=opconfig.norm
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> torch.Tensor:

        hidden_states = self.wte(input_ids)

        is_kv_cache_list = isinstance(kv_cache, list)

        residual = torch.zeros_like(hidden_states)

        for idx, decoder_layer in enumerate(self.h):
            if is_kv_cache_list:
                layer_kv_caches = [
                    kv_cache_mgr.cache_objects[idx] for kv_cache_mgr in kv_cache
                ]
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    positions,
                    layer_kv_caches,
                    **kwargs,
                )
            else:
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    positions,
                    kv_cache.cache_objects[idx],
                    **kwargs,
                )

        hidden_states = self.ln_f(hidden_states + residual)
        return hidden_states


class GPTJForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "fc_in": "up_proj.linear",
        "fc_out": "down_proj",
    }

    def __init__(self, config: GPTJConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config

        self.transformer = GPTJModel(config, opconfig)
        self.lm_head = Linear(
            config.n_embd, config.vocab_size, backend_impl=opconfig.lm_head
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, weight in weight_iterator:
            name = self.rename_fused_params(name)

            if "attn.bias" in name or "attn.masked_bias" in name:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            assert params_dict[name].size() == weight.size()
            params_dict[name].data.copy_(weight)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> ModelOutput:
        model_output = self.transformer(input_ids, positions, kv_cache, **kwargs)
        logits = self.lm_head(model_output)

        return ModelOutput(logits=logits)
