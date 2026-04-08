# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import GptOssConfig

from pace.llm.configs import OperatorConfig
from pace.llm.outputs import ModelOutput
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    RMSNorm,
    FusedRMSNormResidual,
    RotaryEmbedding,
    SoftMax,
    Sigmoid,
    BackendType,
)
from pace.llm.attention import Attention
from pace.utils.mxfp4 import dequantize_mxfp4


class GptOssExperts(nn.Module):
    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_up_linears = nn.ModuleList(
            [
                Linear(
                    self.hidden_size,
                    2 * self.expert_dim,
                    bias=True,
                    backend_impl=opconfig.mlp,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.down_linears = nn.ModuleList(
            [
                Linear(
                    self.expert_dim,
                    self.hidden_size,
                    bias=True,
                    backend_impl=opconfig.mlp,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.sigmoid = Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]

        output = torch.zeros(
            num_tokens,
            self.hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Only compute experts that have tokens routed to them
        active_experts = torch.unique(router_indices)
        for expert_id in active_experts:
            token_mask = (router_indices == expert_id).any(dim=-1)
            expert_input = hidden_states[token_mask]

            gate_up = self.gate_up_linears[expert_id](expert_input)
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * self.sigmoid(gate * self.alpha)
            expert_out = self.down_linears[expert_id]((up + 1) * glu)

            weights = routing_weights[token_mask, expert_id].unsqueeze(-1)
            output[token_mask] += expert_out * weights

        return output.view(batch_size, -1, self.hidden_size)


class GptOssTopKRouter(Linear):
    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig):
        super().__init__(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            backend_impl=opconfig.mlp,
        )
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.softmax = SoftMax(
            dim=1,
            backend_impl=getattr(opconfig, "softmax", BackendType.NATIVE),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = super().forward(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = self.softmax(router_top_value)
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig):
        super().__init__()
        self.router = GptOssTopKRouter(config, opconfig)
        self.experts = GptOssExperts(config, opconfig)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(
            hidden_states, router_indices=router_indices, routing_weights=router_scores
        )
        return routed_out, router_scores


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.qkv_proj = FusedQKVLinear(
            in_features=config.hidden_size,
            out_features=(config.num_attention_heads + 2 * config.num_key_value_heads)
            * self.head_dim,
            bias=config.attention_bias,
            num_key_value_heads=config.num_key_value_heads,
            backend_impl=opconfig.qkv_projection,
        )
        self.o_proj = Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            backend_impl=opconfig.out_projection,
        )

        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        self.sinks.requires_grad_(False)

        self.attn = Attention(
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
            sliding_window=self.sliding_window or 0,
            sinks=self.sinks,
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
        input_shape = hidden_states.shape[:-1]

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        Q = q.view(*input_shape, self.config.num_attention_heads, self.head_dim)
        K = k.view(*input_shape, self.config.num_key_value_heads, self.head_dim)
        V = v.view(*input_shape, self.config.num_key_value_heads, self.head_dim)

        Q, K = position_embeddings(query=Q, key=K, unsqueeze_dim=2)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1)

        return self.o_proj(attn_output)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(
            config=config, opconfig=opconfig, layer_idx=layer_idx
        )
        self.mlp = GptOssMLP(config, opconfig)
        self.input_layernorm = FusedRMSNormResidual(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.post_attention_layernorm = FusedRMSNormResidual(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.attention_type = config.layer_types[layer_idx]

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
        hidden_states, _ = self.mlp(hidden_states)

        return hidden_states, residual


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, opconfig, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.rotary_emb = RotaryEmbedding(
            rope_scaling=config.rope_scaling,
            rotary_dim=self.config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
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

        position_embeddings = self.rotary_emb(hidden_states, positions)

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


class GptOssForCausalLM(BaseModelForCausalLM):
    rename_layers = None

    def __init__(self, config: GptOssConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.config = config
        if getattr(self.config, "_attn_implementation", None) is None:
            self.config._attn_implementation = "eager"
        self.model = GptOssModel(config, opconfig)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    def _get_layer_idx(self, name: str) -> Optional[int]:
        if "model.layers." not in name:
            return None
        suffix = name.split("model.layers.", 1)[1]
        return int(suffix.split(".", 1)[0])

    def _maybe_load_expert_weight(
        self,
        pending: Dict[int, Dict[str, torch.Tensor]],
        layer_idx: int,
        prefix: str,
        dtype: torch.dtype,
    ):
        blocks_key = f"{prefix}_blocks"
        scales_key = f"{prefix}_scales"
        if blocks_key in pending[layer_idx] and scales_key in pending[layer_idx]:
            blocks = pending[layer_idx].pop(blocks_key)
            scales = pending[layer_idx].pop(scales_key)
            dequant = dequantize_mxfp4(blocks, scales, dtype=dtype)
            # dequant shape after transpose:
            #   gate_up_proj: (num_experts, hidden_size, 2*expert_dim)
            #   down_proj:    (num_experts, expert_dim, hidden_size)
            # PACE Linear weight shape: (out_features, in_features) = W^T of the above
            weight_batched = dequant.transpose(-1, -2).contiguous()

            experts = self.model.layers[layer_idx].mlp.experts
            if prefix == "gate_up_proj":
                for e in range(weight_batched.shape[0]):
                    # weight_batched[e] is (hidden_size, 2*expert_dim)
                    # Linear expects (2*expert_dim, hidden_size)
                    experts.gate_up_linears[e].weight.data.copy_(
                        weight_batched[e].T.contiguous()
                    )
            elif prefix == "down_proj":
                for e in range(weight_batched.shape[0]):
                    # weight_batched[e] is (expert_dim, hidden_size)
                    # Linear expects (hidden_size, expert_dim)
                    experts.down_linears[e].weight.data.copy_(
                        weight_batched[e].T.contiguous()
                    )

    def _load_expert_bias(
        self,
        layer_idx: int,
        prefix: str,
        weight: torch.Tensor,
    ):
        """Load per-expert bias tensors into the individual Linear modules."""
        experts = self.model.layers[layer_idx].mlp.experts
        if prefix == "gate_up_proj_bias":
            # weight shape: (num_experts, 2*expert_dim)
            for e in range(weight.shape[0]):
                experts.gate_up_linears[e].bias.data.copy_(weight[e])
        elif prefix == "down_proj_bias":
            # weight shape: (num_experts, hidden_size)
            for e in range(weight.shape[0]):
                experts.down_linears[e].bias.data.copy_(weight[e])

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        pending_expert: Dict[int, Dict[str, torch.Tensor]] = {}
        qkv_cache: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

        for name, weight in weight_iterator:
            name = self.rename_fused_params(name) or name
            if ".mlp.experts." in name:
                layer_idx = self._get_layer_idx(name)
                if layer_idx is None:
                    continue
                if layer_idx not in pending_expert:
                    pending_expert[layer_idx] = {}

                if name.endswith("gate_up_proj_blocks"):
                    pending_expert[layer_idx]["gate_up_proj_blocks"] = weight
                    self._maybe_load_expert_weight(
                        pending_expert,
                        layer_idx,
                        "gate_up_proj",
                        self.model.layers[layer_idx]
                        .mlp.experts.gate_up_linears[0]
                        .weight.dtype,
                    )
                    continue
                if name.endswith("gate_up_proj_scales"):
                    pending_expert[layer_idx]["gate_up_proj_scales"] = weight
                    self._maybe_load_expert_weight(
                        pending_expert,
                        layer_idx,
                        "gate_up_proj",
                        self.model.layers[layer_idx]
                        .mlp.experts.gate_up_linears[0]
                        .weight.dtype,
                    )
                    continue
                if name.endswith("down_proj_blocks"):
                    pending_expert[layer_idx]["down_proj_blocks"] = weight
                    self._maybe_load_expert_weight(
                        pending_expert,
                        layer_idx,
                        "down_proj",
                        self.model.layers[layer_idx]
                        .mlp.experts.down_linears[0]
                        .weight.dtype,
                    )
                    continue
                if name.endswith("down_proj_scales"):
                    pending_expert[layer_idx]["down_proj_scales"] = weight
                    self._maybe_load_expert_weight(
                        pending_expert,
                        layer_idx,
                        "down_proj",
                        self.model.layers[layer_idx]
                        .mlp.experts.down_linears[0]
                        .weight.dtype,
                    )
                    continue

                # Handle bias tensors for experts
                if name.endswith("gate_up_proj_bias"):
                    self._load_expert_bias(layer_idx, "gate_up_proj_bias", weight)
                    continue
                if name.endswith("down_proj_bias"):
                    self._load_expert_bias(layer_idx, "down_proj_bias", weight)
                    continue

                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

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

            if name in params_dict:
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
