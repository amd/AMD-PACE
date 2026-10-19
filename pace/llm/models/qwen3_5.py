# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

from typing import Tuple, Iterable, Callable, Union, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from pace.utils.logging import PACE_LLM_ASSERT
from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig, MoEConfig
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.attention import KVCacheBase, KVCacheManager
from pace.llm.attention.recurrent_state import (
    num_recurrent_layers,
    resolve_recurrent_state,
)
from pace.ops.moe_routing import RoutingType
from pace.llm.ops import (
    Linear,
    FusedQKVLinear,
    RotaryEmbedding,
    MergedMLP,
    MoEExperts,
)
from pace.ops import (
    Gemma3RMSNorm,
    FusedGemma3RMSNormResidual,
    GatedDeltaAttention,
    RMSNormGated,
)
from pace.llm.attention import Attention


class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet (GDN) linear attention for Qwen3.5.

    Implements hybrid linear attention with a short causal 1D convolution on QKV,
    followed by the gated delta rule recurrence. Used for ~75% of layers in the model.
    """

    def __init__(
        self, config: PretrainedConfig, layer_idx: int, opconfig: OperatorConfig
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.num_groups = self.num_v_heads // self.num_k_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.gdn_layer_idx = sum(
            1 for lt in config.layer_types[:layer_idx] if lt == "linear_attention"
        )

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(
            self.head_v_dim, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )

        self.gdn = GatedDeltaAttention(backend_impl=opconfig.gated_deltanet)

        self.out_proj = Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            backend_impl=opconfig.out_projection,
        )

        self.in_proj_split = (
            self.key_dim * 2 + self.value_dim,  # qkv -> mixed_qkv
            self.value_dim,  # z   -> gate for output norm
            self.num_v_heads,  # b   -> beta
            self.num_v_heads,  # a   -> g
        )
        self.in_proj = Linear(
            self.hidden_size,
            sum(self.in_proj_split),
            bias=False,
            backend_impl=opconfig.qkv_projection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state=None,
        **kwargs,
    ) -> torch.Tensor:
        query_lens = kwargs.get("query_lens")
        stores = getattr(recurrent_state, "stores", None)
        if (
            stores is not None
            and query_lens is not None
            and hidden_states.shape[0] == 1
            and len(query_lens) == len(stores)
            and hidden_states.shape[1] == sum(query_lens)
            and len(query_lens) > 1
        ):
            parts = []
            offset = 0
            for store, qlen in zip(stores, query_lens):
                h = hidden_states[:, offset : offset + qlen]
                mask = (
                    attention_mask[:, offset : offset + qlen]
                    if attention_mask is not None
                    else None
                )
                parts.append(self._forward_one(h, mask, store))
                offset += qlen
            return torch.cat(parts, dim=1)

        if (
            recurrent_state is not None
            and stores is None
            and query_lens is not None
            and hidden_states.shape[0] == 1
            and len(query_lens) > 1
            and hidden_states.shape[1] == sum(query_lens)
        ):
            parts, convs, recs = [], [], []
            offset = 0
            for qlen in query_lens:
                h = hidden_states[:, offset : offset + qlen]
                mask = (
                    attention_mask[:, offset : offset + qlen]
                    if attention_mask is not None
                    else None
                )
                out, conv_new, rec_new = self._compute(h, mask, None, None)
                parts.append(out)
                convs.append(conv_new)
                recs.append(rec_new)
                offset += qlen
            recurrent_state.commit(
                self.gdn_layer_idx,
                torch.cat(convs, dim=0),
                torch.cat(recs, dim=0),
            )
            return torch.cat(parts, dim=1)

        conv_prev, rec_prev = (
            recurrent_state.gather(self.gdn_layer_idx)
            if recurrent_state is not None
            else (None, None)
        )
        out, conv_new, rec_new = self._compute(
            hidden_states, attention_mask, conv_prev, rec_prev
        )
        if recurrent_state is not None:
            recurrent_state.commit(self.gdn_layer_idx, conv_new, rec_new)
        return out

    def _forward_one(self, hidden_states, attention_mask, store) -> torch.Tensor:
        conv_prev, rec_prev = (
            store.gather(self.gdn_layer_idx) if store is not None else (None, None)
        )
        out, conv_new, rec_new = self._compute(
            hidden_states, attention_mask, conv_prev, rec_prev
        )
        if store is not None:
            store.commit(self.gdn_layer_idx, conv_new, rec_new)
        return out

    def _compute(self, hidden_states, attention_mask, conv_prev, rec_prev):
        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None and seq_len > 1:
            mask_in = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            hidden_states = hidden_states * mask_in

        projected = self.in_proj(hidden_states)
        mixed_qkv, z, b, a = torch.split(projected, self.in_proj_split, dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        mixed_qkv, conv_new = self.gdn.conv1d_update(
            mixed_qkv,
            conv_prev,
            self.conv1d.weight.squeeze(1),
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if attention_mask is not None and seq_len > 1:
            mask_seq = attention_mask.to(beta.dtype).unsqueeze(-1)
            beta = beta * mask_seq
            g = g * mask_seq

        core_attn_out, rec_new = self.gdn.recurrence(
            query, key, value, g, beta, rec_prev, self.num_groups
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out), conv_new, rec_new


Qwen3_5HeadRMSNorm = Gemma3RMSNorm
Qwen3_5FusedRMSNormResidual = FusedGemma3RMSNormResidual


class Qwen3_5FullAttention(nn.Module):
    """
    Standard multi-head attention with Q/K RMSNorm and gated output,
    as used in Qwen3.5 full_attention layers.

    Key differences from Qwen2:
      - q_proj outputs 2x head_dim (query + gate), split and gate applied to output
      - Per-head Q/K RMSNorm before RoPE
      - Partial rotary factor: only a fraction of head_dim is rotated
      - Output is multiplied by sigmoid(gate) before o_proj
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.attention_bias = getattr(config, "attention_bias", False)
        self.q_size = config.num_attention_heads * self.head_dim * 2
        self.kv_size = config.num_key_value_heads * self.head_dim

        self.qkv_proj = FusedQKVLinear(
            in_features=config.hidden_size,
            out_features=self.q_size + 2 * self.kv_size,
            bias=self.attention_bias,
            num_key_value_heads=config.num_key_value_heads,
            backend_impl=opconfig.qkv_projection,
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
            bias=self.attention_bias,
            backend_impl=opconfig.out_projection,
        )

        self.q_norm = Qwen3_5HeadRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.k_norm = Qwen3_5HeadRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, backend_impl=opconfig.norm
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
        batch_size, seq_len = input_shape

        qkv = self.qkv_proj(hidden_states)
        q_out, k_out, v_out = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        query_states, gate = torch.chunk(
            q_out.view(batch_size, seq_len, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        Q = self.q_norm(
            query_states.reshape(
                batch_size, seq_len, self.num_attention_heads, self.head_dim
            )
        )
        K = self.k_norm(
            k_out.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        )
        V = v_out.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        Q, K = position_embeddings(query=Q, key=K, unsqueeze_dim=2)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1)

        attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self, config: PretrainedConfig, layer_idx: int, opconfig: OperatorConfig
    ):
        super().__init__()
        self.config = config
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, opconfig)
            self._attn_forward = self._linear_attn_forward
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, opconfig)
            self._attn_forward = self._full_attn_forward
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        if getattr(config, "num_experts", 0):
            # Qwen3.5 softmaxes over all experts before taking top-k, which the
            # SOFTMAX MoERouter reproduces exactly when renormalizing, since the
            # full-softmax partition cancels in the top-k normalization. Its
            # shared expert is scaled by a per-token sigmoid gate.
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
                    renormalize=True,
                    num_shared_experts=1,
                    shared_intermediate_size=config.shared_expert_intermediate_size,
                    shared_expert_gate=True,
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
        self.input_layernorm = Qwen3_5FusedRMSNormResidual(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )
        self.post_attention_layernorm = Qwen3_5FusedRMSNormResidual(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )

    def _linear_attn_forward(
        self,
        hidden_states,
        position_embeddings,
        kv_cache,
        positions,
        attention_mask,
        recurrent_state,
        **kwargs,
    ):
        return self.linear_attn(
            hidden_states,
            positions,
            attention_mask=attention_mask,
            recurrent_state=recurrent_state,
            **kwargs,
        )

    def _full_attn_forward(
        self,
        hidden_states,
        position_embeddings,
        kv_cache,
        positions,
        attention_mask,
        recurrent_state,
        **kwargs,
    ):
        return self.self_attn(
            hidden_states, position_embeddings, kv_cache, positions, **kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        position_embeddings: Callable,
        kv_cache: Union[KVCacheBase, List[KVCacheBase], None],
        positions: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self._attn_forward(
            hidden_states,
            position_embeddings,
            kv_cache,
            positions,
            attention_mask,
            recurrent_state,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx, opconfig)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5HeadRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, backend_impl=opconfig.norm
        )

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_params.get(
            "rope_theta", getattr(config, "rope_theta", 10000.0)
        )
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
        )
        rope_scaling = getattr(config, "rope_scaling", None)

        self.rotary_emb = RotaryEmbedding(
            rope_scaling=rope_scaling,
            rotary_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            backend_impl=opconfig.rope,
        )

        self._num_full_attn_layers = sum(
            1 for lt in config.layer_types if lt == "full_attention"
        )
        self._num_gdn_layers = num_recurrent_layers(config)

    def _make_position_embeddings(self, hidden_states, positions):
        """Build the RoPE callable used by the full-attention layers.

        Qwen3.5 uses partial rotary (only the first ``rotary_dim`` head dims are
        rotated). Following the GPT-J pattern, the rotary span is sliced here and
        the rotation is delegated to the shared ``RotaryEmbedding`` op (which
        owns the neox math and the fused C++ kernel fast path); the untouched
        pass-through tail is concatenated back. When ``partial_rotary_factor``
        is 1.0 the rotary span is the full head_dim and the tail is empty.
        """
        cos, sin = self.rotary_emb(hidden_states, positions, return_cos_sin=True)
        rotary_dim = self.rotary_emb.rotary_dim

        def apply(query, key, unsqueeze_dim=2):
            q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            q_rot, k_rot = self.rotary_emb.apply_rotary_emb(
                q_rot, k_rot, cos, sin, unsqueeze_dim
            )
            return (
                torch.cat([q_rot, q_pass], dim=-1),
                torch.cat([k_rot, k_pass], dim=-1),
            )

        return apply

    @staticmethod
    def _padding_mask_from_positions(
        positions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Reconstruct the left-padding mask (True = real token, False = pad)
        from the position ids.
        """
        seq_len = positions.shape[1]
        if seq_len <= 1:
            return None

        valid_lengths = (positions[:, -1] + 1).clamp(max=seq_len)
        left_padding = seq_len - valid_lengths

        if not torch.any(left_padding):
            return None

        token_indices = torch.arange(seq_len, device=positions.device)
        return token_indices.unsqueeze(0) >= left_padding.unsqueeze(1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager], None],
        **kwargs,
    ) -> torch.Tensor:

        input_shape = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        is_kv_cache_list = isinstance(kv_cache, list)

        recurrent_state = resolve_recurrent_state(kv_cache, self._num_gdn_layers)
        query_lens = getattr(kv_cache, "slab_query_lens", None)
        if query_lens is None:
            pam = kwargs.get("paged_attn_metadata")
            qsl = getattr(pam, "query_start_loc", None) if pam is not None else None
            if qsl is not None and qsl.numel() > 2:
                query_lens = (qsl[1:] - qsl[:-1]).tolist()
        if query_lens is not None:
            kwargs = dict(kwargs)
            kwargs["query_lens"] = query_lens

        position_embeddings: Callable = self._make_position_embeddings(
            hidden_states, positions
        )

        padding_mask = (
            None
            if kwargs.get("paged_attn_metadata") is not None
            else self._padding_mask_from_positions(positions)
        )

        if is_kv_cache_list and len(kv_cache) != input_shape[0]:
            raise ValueError(
                f"Number of KVCache objects ({len(kv_cache)}) must match "
                f"batch size ({input_shape[0]})"
            )

        residual = torch.zeros_like(hidden_states)

        full_attn_idx = 0
        for decoder_layer in self.layers:
            if decoder_layer.layer_type == "full_attention":
                if is_kv_cache_list:
                    layer_kv_caches = [
                        kv_cache_mgr.cache_objects[full_attn_idx]
                        for kv_cache_mgr in kv_cache
                    ]
                else:
                    layer_kv_caches = kv_cache.cache_objects[full_attn_idx]
                full_attn_idx += 1
            else:
                layer_kv_caches = None

            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                position_embeddings,
                layer_kv_caches,
                positions,
                attention_mask=padding_mask,
                recurrent_state=recurrent_state,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states + residual)
        return hidden_states


def _extract_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Extract the text config from a multimodal Qwen3_5Config wrapper."""
    if hasattr(config, "text_config"):
        return config.text_config
    return config


class Qwen3_5ForCausalLM(BaseModelForCausalLM):

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        text_config = _extract_text_config(config)
        super().__init__(text_config)
        self.config = text_config
        self._orig_config = config
        self.model = Qwen3_5Model(text_config, opconfig)
        self.lm_head = Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )

    @property
    def num_full_attn_layers(self) -> int:
        return self.model._num_full_attn_layers

    _IN_PROJ_FUSE_ORDER = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))

        tie = getattr(self.config, "tie_word_embeddings", False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight

        qkv_cache = {}

        for name, tensor in weight_iterator:
            if name.startswith("model.visual.") or name.startswith("visual."):
                continue
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]

            name = self.rename_fused_params(name)

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if "rotary_emb.original_inv_freq" in name:
                continue
            if tie and "lm_head.weight" in name:
                continue

            fused_proj = self._maybe_load_fused_in_proj(name, tensor)
            if fused_proj:
                continue

            if ".self_attn." in name and any(
                f".{p}_proj." in name for p in ("q", "k", "v")
            ):
                parts = name.split(".")
                proj_type = parts[-2].split("_")[0]  # q / k / v
                attn_prefix = ".".join(parts[:-2])
                slot = qkv_cache.setdefault(attn_prefix, {"weight": {}, "bias": {}})
                if name.endswith(".weight"):
                    slot["weight"][proj_type] = tensor
                else:
                    slot["bias"][proj_type] = tensor
                continue

            if name.endswith(".bias") and name not in params:
                if not any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
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

    @torch.no_grad()
    def _maybe_load_fused_in_proj(self, name: str, tensor: torch.Tensor) -> bool:
        """If `name` is one of the unfused DeltaNet input projections, copy it
        into the correct row-slice of the layer's fused `in_proj.weight` and
        return True. Otherwise return False.
        """
        if ".linear_attn." not in name or not name.endswith(".weight"):
            return False
        for proj in self._IN_PROJ_FUSE_ORDER:
            suffix = f".linear_attn.{proj}.weight"
            if not name.endswith(suffix):
                continue
            layer_prefix = name[: -len(suffix)]
            module = self.get_submodule(layer_prefix + ".linear_attn")
            fused_weight = module.in_proj.weight
            splits = module.in_proj_split
            idx = self._IN_PROJ_FUSE_ORDER.index(proj)
            start = sum(splits[:idx])
            end = start + splits[idx]
            PACE_LLM_ASSERT(
                tensor.shape == fused_weight[start:end].shape,
                f"{name}: source {tuple(tensor.shape)} does not match fused "
                f"slice {tuple(fused_weight[start:end].shape)}",
            )
            fused_weight[start:end].copy_(tensor)
            return True
        return False

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


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.5 MoE — the dense hybrid stack with a sparse FFN on every layer."""

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        modules = dict(self.named_modules())

        def dense_weights():
            for name, tensor in weight_iterator:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                if not self._load_stacked_experts(modules, name, tensor):
                    yield self._remap_moe_weight(name), tensor

        super().load_weights(dense_weights())

    @staticmethod
    def _remap_moe_weight(name: str) -> str:
        """Map HF's MoE module names onto the MoEExperts sub-modules.

        HF:  mlp.gate.weight           -> mlp.router.weight
        HF:  mlp.shared_expert.*       -> mlp.shared_mlps.0.*
        HF:  mlp.shared_expert_gate.*  -> mlp.shared_gate.*
        """
        name = name.replace(".mlp.gate.weight", ".mlp.router.weight")
        name = name.replace(".mlp.shared_expert_gate.", ".mlp.shared_gate.")
        return name.replace(".mlp.shared_expert.", ".mlp.shared_mlps.0.")

    @torch.no_grad()
    def _load_stacked_experts(
        self, modules: dict, name: str, tensor: torch.Tensor
    ) -> bool:
        """Split HF's stacked expert tensors into the per-expert linears.

        ``experts.gate_up_proj`` is [E, 2I, H] with contiguous gate|up halves --
        not the interleaved layout ``MoEExperts.load_expert_weights`` splits --
        and ``experts.down_proj`` is [E, H, I].
        """
        for suffix in (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"):
            if not name.endswith(suffix):
                continue
            moe = modules.get(name[: -len(suffix)] + ".mlp")
            if moe is not None:
                inter = moe.intermediate_size
                for e in range(moe.num_experts):
                    if suffix.endswith("gate_up_proj"):
                        moe.gate_projs[e].weight.data.copy_(tensor[e][:inter])
                        moe.up_projs[e].weight.data.copy_(tensor[e][inter:])
                    else:
                        moe.down_projs[e].weight.data.copy_(tensor[e])
            return True
        return False
