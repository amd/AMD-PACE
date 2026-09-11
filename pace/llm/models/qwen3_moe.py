# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
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

# Qwen3 Mixture-of-Experts model for PACE.
# Reuses QwenAttention (with QK-Norm, no bias) from the unified dense model.
# MoE layer uses the MoEExperts operator: the router (a MoERouter sub-op) and
# each expert projection are pace ops, so routing and the expert matmuls are
# wired to NATIVE or a fused backend through the standard backend dispatch.

from typing import Callable, Dict, Iterable, List, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from pace.llm.configs import OperatorConfig, MoEConfig
from pace.ops.moe_routing import RoutingType
from pace.llm.outputs import ModelOutput
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.models.qwen import QwenAttention
from pace.llm.ops import (
    Linear,
    RMSNorm,
    FusedRMSNormResidual,
    RotaryEmbedding,
    MergedMLP,
    MoEExperts,
)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, opconfig: OperatorConfig, layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QwenAttention(config, opconfig)

        is_sparse = (layer_idx + 1) % config.decoder_sparse_step == 0
        if is_sparse:
            # Qwen3 MoE routes with softmax over all experts + top-k, then
            # renormalizes the top-k weights when norm_topk_prob is set. The
            # SOFTMAX MoERouter (softmax over the top-k logits) is identical to
            # that when renormalize=True, since the full-softmax partition
            # cancels in the top-k normalization.
            self.mlp = MoEExperts(
                MoEConfig(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    num_experts=config.num_experts,
                    topk=config.num_experts_per_tok,
                    activation=config.hidden_act,
                    gate=True,
                    bias=False,
                    router_bias=False,
                    routing=RoutingType.SOFTMAX,
                    renormalize=getattr(config, "norm_topk_prob", True),
                ),
                backend_impl=opconfig.mlp,
            )
        else:
            self.mlp = MergedMLP(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                activation=config.hidden_act,
                gate=True,
                backend_impl=opconfig.mlp,
            )
        self.is_sparse = is_sparse

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


class Qwen3MoeModel(nn.Module):
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
                Qwen3MoeDecoderLayer(config, opconfig, layer_idx)
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


class Qwen3MoeForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        self.model = Qwen3MoeModel(config, opconfig)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    @staticmethod
    def _remap_moe_weight(name: str) -> str:
        """Remap HF MoE weight names to MoEExperts parameter names.

        MoEExperts holds the router and each expert projection as pace Linear
        ops (router / {gate,up,down}_projs ModuleLists), so their weights live
        at plain ``.weight`` paths:

        HF:   mlp.gate.weight                    -> mlp.router.weight
        HF:   mlp.experts.N.gate_proj.weight     -> mlp.gate_projs.N.weight
        HF:   mlp.experts.N.up_proj.weight       -> mlp.up_projs.N.weight
        HF:   mlp.experts.N.down_proj.weight     -> mlp.down_projs.N.weight
        """
        import re

        name = name.replace(".mlp.gate.", ".mlp.router.")
        m = re.match(
            r"(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name
        )
        if m:
            prefix, idx, proj = m.group(1), m.group(2), m.group(3)
            return f"{prefix}.{proj}s.{idx}.weight"
        return name

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))

        qkv_cache: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        for name, tensor in weight_iterator:
            remapped = self._remap_moe_weight(name)
            # A changed name means this is a MoE router/expert weight, already
            # mapped to its pace Linear path (router.weight / *_projs.N.weight).
            # rename_fused_params targets dense MergedMLP layers and would mangle
            # these (e.g. "gate_projs" -> "gate_proj.linears"), so skip it.
            if remapped == name:
                name = self.rename_fused_params(name)
            else:
                name = remapped
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

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
