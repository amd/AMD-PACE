# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/opt/modeling_opt.py

from typing import List, Tuple, Iterable, Union

import torch
from torch import nn
from transformers import OPTConfig

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    LayerNorm,
    FusedLayerNormResidual,
    MergedMLP,
)
from pace.llm.attention import Attention


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.LongTensor) -> torch.Tensor:
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        opconfig: OperatorConfig,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.enable_bias = config.enable_bias
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = FusedQKVLinear(
            in_features=self.embed_dim,
            out_features=3 * self.embed_dim,
            bias=self.enable_bias,
            num_key_value_heads=self.num_heads,
            backend_impl=opconfig.qkv_projection,
        )
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=self.enable_bias,
            backend_impl=opconfig.out_projection,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache,
        positions: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        assert hidden_states.dim() == 3
        assert hidden_states.shape[2] == self.embed_dim

        batch_size, seq_len, _ = hidden_states.shape

        bshd_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        Q = q.view(*bshd_shape)
        K = k.view(*bshd_shape)
        V = v.view(*bshd_shape)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.self_attn = OPTAttention(config=config, opconfig=opconfig)
        self.do_layer_norm_before = config.do_layer_norm_before

        if self.do_layer_norm_before:
            self.self_attn_layer_norm = FusedLayerNormResidual(
                self.embed_dim,
                elementwise_affine=config.layer_norm_elementwise_affine,
                backend_impl=opconfig.norm,
            )
        else:
            self.self_attn_layer_norm = LayerNorm(
                self.embed_dim,
                elementwise_affine=config.layer_norm_elementwise_affine,
                backend_impl=opconfig.norm,
            )
        self.fc = MergedMLP(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            activation=config.activation_function,
            gate=False,
            backend_impl=opconfig.mlp,
        )
        if self.do_layer_norm_before:
            self.final_layer_norm = FusedLayerNormResidual(
                self.embed_dim,
                elementwise_affine=config.layer_norm_elementwise_affine,
                backend_impl=opconfig.norm,
            )
        else:
            self.final_layer_norm = LayerNorm(
                self.embed_dim,
                elementwise_affine=config.layer_norm_elementwise_affine,
                backend_impl=opconfig.norm,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kv_cache: Union[KVCacheBase, List[KVCacheBase]],
        positions: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.do_layer_norm_before:
            hidden_states, residual = self.self_attn_layer_norm(hidden_states, residual)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                positions=positions,
                **kwargs,
            )
            hidden_states, residual = self.final_layer_norm(hidden_states, residual)
            hidden_states = self.fc(hidden_states)
        else:
            saved = hidden_states + residual
            hidden_states = self.self_attn(
                hidden_states=saved,
                kv_cache=kv_cache,
                positions=positions,
                **kwargs,
            )
            hidden_states = saved + hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)

            saved = hidden_states
            hidden_states = self.fc(hidden_states)
            hidden_states = saved + hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            residual = torch.zeros_like(hidden_states)

        return hidden_states, residual


class OPTDecoder(nn.Module):

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):

        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = Linear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                backend_impl=(
                    opconfig["project_out"] if "project_out" in opconfig else None
                ),
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = Linear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                backend_impl=(
                    opconfig["project_in"] if "project_in" in opconfig else None
                ),
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config, opconfig) for _ in range(config.num_hidden_layers)]
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> torch.Tensor:

        hidden_states = self.embed_tokens(input_ids)

        # Positional embeddings from caller-provided positions
        pos_embeds = self.embed_positions(positions)

        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)

        hidden_states = hidden_states + pos_embeds

        is_kv_cache_list = isinstance(kv_cache, list)

        residual = torch.zeros_like(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):
            if is_kv_cache_list:
                layer_kv_caches = [
                    kv_cache_mgr.cache_objects[idx] for kv_cache_mgr in kv_cache
                ]
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    layer_kv_caches,
                    positions,
                    **kwargs,
                )
            else:
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    kv_cache.cache_objects[idx],
                    positions,
                    **kwargs,
                )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states + residual)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):
        super().__init__()
        self.decoder = OPTDecoder(config, opconfig)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        return self.decoder(input_ids, positions, kv_cache, **kwargs)


class OPTForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "fc1": "fc.up_proj.linear",
        "fc2": "fc.down_proj",
    }

    def __init__(self, config: OPTConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config

        self.model = OPTModel(config, opconfig)

        # Logits
        self.lm_head = Linear(
            config.word_embed_proj_dim,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):

        params = dict(self.named_parameters(remove_duplicate=False))

        qkv_cache = {}
        found_lm_head = False

        for name, tensor in weight_iterator:
            if name.startswith("decoder."):
                name = "model." + name
            name = self.rename_fused_params(name)

            if "lm_head.weight" in name:
                found_lm_head = True

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

        if not found_lm_head:
            self.lm_head.weight = self.model.decoder.embed_tokens.weight

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
