# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team.
# All rights reserved.
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

# Llama 4 model for PACE (text-only, Scout and Maverick variants).

from typing import Callable, Dict, Iterable, List, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from pace.llm.configs import OperatorConfig, MoEConfig
from pace.ops.moe_routing import RoutingType
from pace.llm.outputs import ModelOutput
from pace.llm.attention import KVCacheBase, KVCacheManager, Attention
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    RMSNorm,
    L2Norm,
    FusedRMSNormResidual,
    RotaryEmbedding,
    MergedMLP,
    MoEExperts,
)


class Llama4Attention(nn.Module):
    """Llama4 attention with iRoPE, optional L2 QK-norm, and temperature tuning."""

    def __init__(
        self,
        config: PretrainedConfig,
        opconfig: OperatorConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.layer_idx = layer_idx

        # iRoPE: RoPE is applied only on a subset of layers. ``no_rope_layers``
        # is HF's per-layer flag (1 => RoPE, 0 => NoPE); NoPE layers rely on
        # attention temperature tuning (below) instead.
        self.use_rope = bool(config.no_rope_layers[layer_idx])
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale

        self.qkv_proj = FusedQKVLinear(
            in_features=self.hidden_size,
            out_features=(self.num_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
            bias=config.attention_bias,
            num_key_value_heads=self.num_key_value_heads,
            backend_impl=opconfig.qkv_projection,
        )

        # Chunked (local) attention on RoPE layers only (window =
        # ``attention_chunk_size``); NoPE layers stay full causal.
        chunk = getattr(config, "attention_chunk_size", None) or 0
        self.attention_chunk_size = chunk if self.use_rope else 0
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
            chunk_size=self.attention_chunk_size,
        )

        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            backend_impl=opconfig.out_projection,
        )

        if getattr(config, "use_qk_norm", False) and self.use_rope:
            self.qk_norm = L2Norm(config.rms_norm_eps, backend_impl=opconfig.norm)

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
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        Q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        K = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        V = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.use_rope:
            Q, K = position_embeddings(
                query=Q, key=K, unsqueeze_dim=2, is_neox_style=False
            )

        if hasattr(self, "qk_norm"):
            Q = self.qk_norm(Q)
            K = self.qk_norm(K)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale))
                * self.attn_scale
                + 1.0
            )
            Q = (Q * attn_scales[:, :, None, None]).to(Q.dtype)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        opconfig: OperatorConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4Attention(config, opconfig, layer_idx)

        is_moe_layer = layer_idx in config.moe_layers
        if is_moe_layer:
            # Sigmoid top-k routed experts + one always-on shared expert
            # (MoEExperts' num_shared_experts); Llama4 does not renormalize.
            self.feed_forward = MoEExperts(
                MoEConfig.from_model(
                    config,
                    router_bias=False,
                    routing=RoutingType.SIGMOID,
                    renormalize=False,
                    num_shared_experts=1,
                ),
                backend_impl=opconfig.mlp,
            )
        else:
            self.feed_forward = MergedMLP(
                config.hidden_size,
                config.intermediate_size_mlp,
                bias=False,
                activation=config.hidden_act,
                gate=True,
                backend_impl=opconfig.mlp,
            )
        self.is_moe_layer = is_moe_layer

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
        hidden_states = self.feed_forward(hidden_states)

        return hidden_states, residual


class Llama4Model(nn.Module):
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
                Llama4DecoderLayer(config, opconfig, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
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


class Llama4ForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config)
        self.config = text_config
        self.model = Llama4Model(text_config, opconfig)
        self.lm_head = Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    @staticmethod
    def _remap_moe_weight(name: str) -> str:
        """Remap HF Llama4 MoE checkpoint names to MoEExperts parameter names.

        The layer's ``feed_forward`` is a single MoEExperts operator. Its router
        already matches HF's ``feed_forward.router.weight``; the always-on shared
        expert is MoEExperts' shared MLP at ``feed_forward.shared_mlps.0``:

          feed_forward.shared_expert.* -> feed_forward.shared_mlps.0.*
        """
        if ".feed_forward.shared_expert." in name:
            return name.replace(
                ".feed_forward.shared_expert.", ".feed_forward.shared_mlps.0."
            )
        return name

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))
        modules = dict(self.named_modules())
        qkv_cache: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        for name, tensor in weight_iterator:
            if "vision_model" in name or "multi_modal_projector" in name:
                continue
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            if "rotary_emb" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # HF 3D expert weights: split into per-expert 2D and copy into the
            # MoEExperts per-expert pace Linear ModuleLists (gate_projs/up_projs/
            # down_projs), whose weights live at plain ``.weight`` paths.
            # gate_up_proj: (E, H, 2*I) -> per-expert gate (I, H) + up (I, H)
            # down_proj:    (E, I, H)    -> per-expert down (H, I)
            if ".experts.gate_up_proj" in name and ".experts.router" not in name:
                # ``...feed_forward.experts.gate_up_proj`` -> the MoEExperts op,
                # which is the layer's ``feed_forward`` module itself.
                layer_prefix = name.rsplit(".experts.gate_up_proj", 1)[0]
                moe_mod = modules.get(layer_prefix)
                if moe_mod is not None:
                    num_e = tensor.size(0)
                    inter_size = tensor.size(2) // 2
                    for e in range(num_e):
                        # HF shape: (H, 2*I); chunk splits into contiguous halves
                        gate_w = tensor[e, :, :inter_size].T.contiguous()  # (I, H)
                        up_w = tensor[e, :, inter_size:].T.contiguous()  # (I, H)
                        moe_mod.gate_projs[e].weight.data.copy_(gate_w)
                        moe_mod.up_projs[e].weight.data.copy_(up_w)
                continue

            if ".experts.down_proj" in name and ".experts.router" not in name:
                layer_prefix = name.rsplit(".experts.down_proj", 1)[0]
                moe_mod = modules.get(layer_prefix)
                if moe_mod is not None:
                    num_e = tensor.size(0)
                    for e in range(num_e):
                        moe_mod.down_projs[e].weight.data.copy_(
                            tensor[e].T.contiguous()
                        )
                continue

            name = self._remap_moe_weight(name)
            name = self.rename_fused_params(name)

            if name.endswith(".bias") and name not in params:
                if not any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
                    continue

            if "qkv_proj" in name and name in params:
                params[name].data.copy_(tensor)
                continue

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
