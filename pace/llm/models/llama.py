# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/llama/modeling_llama.py
# The file contains the implemention of LLAMA models, as well as Phi3/4 models. Since they are the same,
# the implementation is shared between the two models.

from typing import List, Tuple, Iterable, Union, Callable

import torch
from torch import nn
from transformers import LlamaConfig, PhiConfig

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    RMSNorm,
    FusedRMSNormResidual,
    RotaryEmbedding,
    MergedMLP,
)
from pace.llm.attention import Attention


class LlamaAttention(nn.Module):
    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        phi_model = config.architectures[0] == "Phi3ForCausalLM"
        bias = False if phi_model else config.attention_bias

        self.qkv_proj = FusedQKVLinear(
            in_features=self.hidden_size,
            out_features=(self.num_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
            bias=bias,
            num_key_value_heads=self.num_key_value_heads,
            backend_impl=opconfig.qkv_projection,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
        )

        self.o_proj = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
            backend_impl=opconfig.out_projection,
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Callable,
        kv_cache,
        positions: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        assert hidden_states.dim() == 3
        assert hidden_states.shape[2] == self.hidden_size

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        Q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        K = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        V = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        Q, K = position_embeddings(query=Q, key=K, unsqueeze_dim=2)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = FusedRMSNormResidual(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend_impl=opconfig.norm,
        )
        self.self_attn = LlamaAttention(config, opconfig)
        self.mlp = MergedMLP(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=(
                False
                if config.architectures[0] == "Phi3ForCausalLM"
                else config.mlp_bias
            ),
            activation=config.hidden_act,
            gate=True,
            backend_impl=opconfig.mlp,
        )
        self.post_attention_layernorm = FusedRMSNormResidual(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend_impl=opconfig.norm,
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


class LlamaModel(nn.Module):

    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, opconfig)
                for _ in range(config.num_hidden_layers)
            ]
        )

        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=config.rope_scaling,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            backend_impl=opconfig.rope,
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", None
            ),
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend_impl=opconfig.norm,
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


class LlamaForCausalLM(BaseModelForCausalLM):

    target_map = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: Union[LlamaConfig, PhiConfig], opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        self.model = LlamaModel(config, opconfig)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):

        def split_projection_weight_path(input_string, target_map):
            import re

            """
            Splits a combined projection weight path into separate paths based on a target map.

            Args:
                input_string: The input string representing the combined weight path.
                target_map: A dictionary where keys are target names (e.g., "gate_up_proj")
                        and values are lists of corresponding split names (e.g., ["gate_proj", "up_proj"]).

            Returns:
                A tuple containing the split weight paths, or None if the
                input string doesn't match the expected pattern or the target_name is not in target_map.
                Returns an empty tuple if the target name already exists.
            """
            for target_name, split_names in target_map.items():
                match = re.search(rf"(.*{target_name})(.*)", input_string)
                if match:
                    prefix = match.group(1)[: -len(target_name)]
                    suffix = match.group(2)

                    split_paths = [f"{prefix}{name}{suffix}" for name in split_names]
                    return tuple(split_paths)
            return None

        params = dict(self.named_parameters(remove_duplicate=False))

        qkv_cache = {}

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

            # For Phi models, split fused gate_up_proj into gate_proj and up_proj
            if self.target_map:
                split_names = split_projection_weight_path(name, self.target_map)
                if split_names is not None:
                    split_weights = tensor.chunk(len(split_names), dim=0)

                    for split_name, split_weight in zip(split_names, split_weights):
                        if split_name in params:
                            assert params[split_name].size() == split_weight.size()
                            params[split_name].data.copy_(split_weight)
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

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
