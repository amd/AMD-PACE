# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2026 Google Inc. HuggingFace Inc. team. All rights reserved.
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

"""Gemma 4 text model for PACE (text-only; vision/audio towers skipped at load).

Relative to Gemma 3: alternating sliding/global attention with dual RoPE
(proportional RoPE + larger head_dim on global layers), QK-norm plus a
no-scale value norm, softmax scale folded into Q, Per-Layer Embeddings, a
shared KV cache for the trailing layers, and an optional MoE block (26B-A4B)
whose router applies a per-expert scale.
"""

from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from pace.llm.outputs import ModelOutput
from pace.llm.configs import OperatorConfig, MoEConfig
from pace.ops.moe_routing import RoutingType
from pace.llm.models.base_model import BaseModelForCausalLM
from pace.llm.attention import KVCacheBase, KVCacheManager, Attention
from pace.llm.attention.specs import LayerKVSpec, compute_layer_kv_specs
from pace.llm.ops import (
    Linear,
    RMSNorm,
    L2Norm,
    RotaryEmbedding,
    MergedMLP,
    MoEExperts,
)
from pace.utils.logging import PACE_LLM_WARNING


class Gemma4Attention(nn.Module):
    """Gemma4 attention: alternating sliding/global, dual RoPE, QK/V norms.

    Softmax scaling is 1.0 in HF; PACE attention backends apply the default
    1/sqrt(head_dim), so Q is pre-scaled by sqrt(head_dim) to net a scale of
    1.0. Shared-KV layers build only the query projection.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        opconfig: OperatorConfig,
        spec: LayerKVSpec,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.is_kv_shared_layer = spec.is_shared
        self.head_dim = spec.head_dim
        self.num_kv_heads = spec.num_kv_heads
        self.sliding_window = spec.sliding_window

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.k_eq_v = getattr(config, "attention_k_eq_v", False) and not is_sliding

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        attention_bias = getattr(config, "attention_bias", False)
        self.q_scale = float(self.head_dim) ** 0.5

        self.q_proj = Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=attention_bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.q_norm = RMSNorm(
            self.head_dim, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        if not self.is_kv_shared_layer:
            self.k_proj = Linear(
                self.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=attention_bias,
                backend_impl=opconfig.qkv_projection,
            )
            self.k_norm = RMSNorm(
                self.head_dim, eps=rms_norm_eps, backend_impl=opconfig.norm
            )
            # No-scale (weightless) RMS on the value; shared L2Norm op.
            self.v_norm = L2Norm(eps=rms_norm_eps, backend_impl=opconfig.norm)
            if not self.k_eq_v:
                self.v_proj = Linear(
                    self.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    bias=attention_bias,
                    backend_impl=opconfig.qkv_projection,
                )

        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            backend_impl=opconfig.out_projection,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
            sliding_window=self.sliding_window or 0,
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

        Q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        Q = self.q_norm(Q)
        Q, _ = position_embeddings(query=Q, key=Q, unsqueeze_dim=2)
        # Fold the HF scaling=1.0 into Q (backends apply 1/sqrt(head_dim)).
        Q = Q * self.q_scale

        if self.is_kv_shared_layer:
            # No K/V of our own: the cache layer serves the source layer's KV.
            K = V = None
        else:
            k_raw = self.k_proj(hidden_states).view(
                bsz, q_len, self.num_kv_heads, self.head_dim
            )
            K = self.k_norm(k_raw)
            _, K = position_embeddings(query=K, key=K, unsqueeze_dim=2)
            if self.k_eq_v:
                # Value shares the key projection (pre-norm, no RoPE).
                V = self.v_norm(k_raw)
            else:
                v_raw = self.v_proj(hidden_states).view(
                    bsz, q_len, self.num_kv_heads, self.head_dim
                )
                V = self.v_norm(v_raw)

        attn_output = self.attn(Q, K, V, kv_cache, positions, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class Gemma4Router(nn.Module):
    """Gemma4 MoE router front-end: produce expert logits from hidden states.

    Applies a no-scale RMS norm, a learned per-hidden input scale and a
    1/sqrt(hidden) factor, then a bias-free projection. Expert selection
    (softmax top-k + renormalize) is delegated to the shared MoEExperts.route,
    and the learned `per_expert_scale` is applied to the routing weights by the
    caller. Matches HF ``Gemma4TextRouter``.
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = L2Norm(
            eps=getattr(config, "rms_norm_eps", 1e-6), backend_impl=opconfig.norm
        )
        self.proj = Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            backend_impl=opconfig.mlp,
        )
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        return self.proj(hidden_states)


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        opconfig: OperatorConfig,
        spec: LayerKVSpec,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.self_attn = Gemma4Attention(config, layer_idx, opconfig, spec)

        use_double_wide = (
            getattr(config, "use_double_wide_mlp", False) and spec.is_shared
        )
        intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)
        hidden_act = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
        self.mlp = MergedMLP(
            in_features=config.hidden_size,
            out_features=intermediate_size,
            bias=False,
            activation=hidden_act,
            gate=True,
            backend_impl=opconfig.mlp,
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

        # Per-Layer Embeddings (PLE) gate.
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[hidden_act]
            self.per_layer_input_gate = Linear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                backend_impl=opconfig.mlp,
            )
            self.per_layer_projection = Linear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                backend_impl=opconfig.mlp,
            )
            self.post_per_layer_input_norm = RMSNorm(
                self.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
            )

        # The MoE block runs parallel to the dense MLP, not instead of it.
        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.router = Gemma4Router(config, opconfig)
            self.experts = MoEExperts(
                MoEConfig(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    num_experts=config.num_experts,
                    topk=config.top_k_experts,
                    activation="gelu_tanh",
                    gate=True,
                    bias=False,
                    routing=RoutingType.SOFTMAX,
                    renormalize=True,
                    build_router=False,
                ),
                backend_impl=opconfig.mlp,
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
            )

    def _moe_block(self, residual: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        """Sparse expert branch, summed with the dense MLP branch (HF layout).

        Routes on the raw residual, runs experts on a normed copy of it, and
        applies the learned per-expert scale to the routing weights.
        """
        hidden_states_1 = self.post_feedforward_layernorm_1(mlp_out)

        flat = residual.reshape(-1, residual.shape[-1])
        logits = self.router(flat)
        topk_index, topk_weights = self.experts.route(logits)
        topk_weights = topk_weights * self.router.per_expert_scale[topk_index].to(
            topk_weights.dtype
        )
        experts_in = self.pre_feedforward_layernorm_2(flat)
        experts_out = self.experts.compute(experts_in, topk_index, topk_weights)
        experts_out = experts_out.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(experts_out)

        return hidden_states_1 + hidden_states_2

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor],
        position_embeddings: Callable,
        kv_cache: Union[KVCacheBase, List[KVCacheBase]],
        positions: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, kv_cache, positions, **kwargs
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.enable_moe_block:
            hidden_states = self._moe_block(residual, mlp_out)
        else:
            hidden_states = mlp_out
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4Model(nn.Module):
    """Gemma4 text transformer (no LM head)."""

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.embed_scale = config.hidden_size**0.5

        self.kv_specs = compute_layer_kv_specs(config)

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config, layer_idx, opconfig, self.kv_specs[layer_idx]
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps, backend_impl=opconfig.norm
        )

        # Per-Layer Embeddings (PLE) pipeline.
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        if self.hidden_size_per_layer_input:
            self.vocab_size_per_layer_input = config.vocab_size_per_layer_input
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
            )
            self.per_layer_embed_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_model_projection = Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
                backend_impl=opconfig.mlp,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_norm = RMSNorm(
                config.hidden_size_per_layer_input,
                eps=rms_norm_eps,
                backend_impl=opconfig.norm,
            )

        global_head_dim = getattr(config, "global_head_dim", None) or getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        local_head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        full_rope = config.rope_parameters["full_attention"]
        sliding_rope = config.rope_parameters["sliding_attention"]
        self.rotary_emb_global = RotaryEmbedding(
            rope_scaling=full_rope,
            rotary_dim=global_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=full_rope["rope_theta"],
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 1.0),
            backend_impl=opconfig.rope,
        )
        self.rotary_emb_local = RotaryEmbedding(
            rope_scaling=sliding_rope,
            rotary_dim=local_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=sliding_rope["rope_theta"],
            backend_impl=opconfig.rope,
        )

    def _is_shared(self, layer_idx: int) -> bool:
        return self.kv_specs[layer_idx].is_shared

    def _layer_cache(self, kv_cache, layer_idx: int):
        """Return the cache object(s) for ``layer_idx``.

        KV sharing is resolved by the cache layer (the cache manager aliases a
        shared layer's slot to its target), so this is a plain per-layer index.
        """
        if isinstance(kv_cache, list):
            return [mgr.cache_objects[layer_idx] for mgr in kv_cache]
        return kv_cache.cache_objects[layer_idx]

    def _per_layer_inputs(self, input_ids, inputs_embeds):
        token_identity = (
            self.embed_tokens_per_layer(input_ids) * self.per_layer_embed_scale
        )
        token_identity = token_identity.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        projection = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        projection = projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        projection = self.per_layer_projection_norm(projection)
        return (projection + token_identity) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self._per_layer_inputs(input_ids, inputs_embeds)

        is_list = isinstance(kv_cache, list)
        if is_list and len(kv_cache) != input_ids.shape[0]:
            raise ValueError(
                f"Number of KVCache objects ({len(kv_cache)}) must match "
                f"batch size ({input_ids.shape[0]})"
            )

        pe_global = self.rotary_emb_global(inputs_embeds, positions)
        pe_local = self.rotary_emb_local(inputs_embeds, positions)

        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            position_embeddings = (
                pe_local
                if self.config.layer_types[idx] == "sliding_attention"
                else pe_global
            )
            per_layer_input = (
                per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
            )
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                position_embeddings,
                self._layer_cache(kv_cache, idx),
                positions,
                **kwargs,
            )

        return self.norm(hidden_states)


class Gemma4Base(BaseModelForCausalLM):
    """Common weight-loading and forward for Gemma4 causal-LM variants.

    Both the text-only ``Gemma4ForCausalLM`` (keys ``model.*`` + ``lm_head``)
    and the multimodal ``Gemma4ForConditionalGeneration`` (text weights nested
    under ``model.language_model.*``) load through the same normalization, which
    maps the multimodal nesting onto our ``model.*`` submodule while leaving
    plain causal-LM keys untouched.
    """

    rename_layers = {
        "up_proj": "up_proj.linear",
        "gate_proj": "gate_proj.linear",
    }

    _lm_head_attr = "lm_head"
    _embed_tokens_attr = "model.embed_tokens"

    _skip_substrings = (
        "vision_tower",
        "audio_tower",
        "multi_modal_projector",
        "embed_vision",
        "embed_audio",
    )

    def rename_fused_params(self, param_name: str) -> str:
        # The dense-MLP rename (up_proj -> up_proj.linear) is a substring
        # replace, which would also corrupt MoE expert tensor names containing
        # "up_proj". Expert tensors are loaded separately, so skip them.
        if ".experts." in param_name:
            return param_name
        return super().rename_fused_params(param_name)

    def _get_nested_attr(self, attr_path: str):
        obj = self
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Map a checkpoint key to a PACE parameter name.

        The multimodal checkpoints nest the text model under
        ``model.language_model.``; rewrite that onto our ``model.`` submodule.
        Text-only ``Gemma4ForCausalLM`` keys are already ``model.*`` / ``lm_head.*``
        and pass through unchanged.
        """
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model.") :]
        return name

    def _is_ignored_shared_kv(self, name: str) -> bool:
        """Shared-KV layers reuse source KV; their checkpoint k/v/k_norm/v_norm
        weights are ignored (matching HF ``_keys_to_ignore_on_load_unexpected``)."""
        if ".layers." not in name or ".self_attn." not in name:
            return False
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return False
        if not self.model._is_shared(layer_idx):
            return False
        return any(
            f".self_attn.{p}" in name for p in ("k_proj", "v_proj", "k_norm", "v_norm")
        )

    def _load_expert_tensor(
        self, modules: dict, name: str, weight: torch.Tensor
    ) -> bool:
        """Split an HF 3D expert tensor into the MoEExperts per-expert linears.

        HF stores ``experts.gate_up_proj`` as [E, 2I, H] (contiguous gate|up
        halves) and ``experts.down_proj`` as [E, H, I]. Returns True if handled.
        """
        for suffix, kind in (
            (".experts.gate_up_proj", "gate_up"),
            (".experts.down_proj", "down"),
        ):
            if name.endswith(suffix):
                moe = modules.get(
                    name[: -len(".gate_up_proj" if kind == "gate_up" else ".down_proj")]
                )
                if moe is None:
                    return True
                inter = moe.intermediate_size
                for e in range(moe.num_experts):
                    if kind == "gate_up":
                        moe.gate_projs[e].weight.data.copy_(weight[e][:inter])
                        moe.up_projs[e].weight.data.copy_(weight[e][inter:])
                    else:
                        moe.down_projs[e].weight.data.copy_(weight[e])
                return True
        return False

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers(remove_duplicate=False))
        modules = dict(self.named_modules())
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

        if tie_word_embeddings:
            lm_head = self._get_nested_attr(self._lm_head_attr)
            embed_tokens = self._get_nested_attr(self._embed_tokens_attr)
            lm_head.weight = embed_tokens.weight

        for name, weight in weight_iterator:
            if any(s in name for s in self._skip_substrings):
                continue
            name = self._normalize_name(name)

            if "rotary_emb" in name and any(
                b in name for b in ("inv_freq", "cos_cache", "sin_cache")
            ):
                continue
            if tie_word_embeddings and name.endswith("lm_head.weight"):
                continue
            if self._is_ignored_shared_kv(name):
                continue
            if ".experts." in name and self._load_expert_tensor(modules, name, weight):
                continue

            name = self.rename_fused_params(name)

            if name in params:
                target = params[name]
                if target.size() != weight.size():
                    raise ValueError(
                        f"Shape mismatch for {name}: {target.size()} vs {weight.size()}"
                    )
                target.data.copy_(weight)
            elif name in buffers:
                buffers[name].data.copy_(weight.to(buffers[name].dtype))
            # Unknown keys (e.g. ignored multimodal extras) are silently skipped.

    def _apply_final_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_cache: Union[KVCacheManager, List[KVCacheManager]],
        **kwargs,
    ) -> ModelOutput:
        model_output = self.model(input_ids, positions, kv_cache, **kwargs)
        logits = self.lm_head(model_output)
        logits = self._apply_final_logit_softcapping(logits)
        return ModelOutput(logits=logits)


class Gemma4ForCausalLM(Gemma4Base):
    """Gemma4 text-only causal LM (``Gemma4ForCausalLM`` checkpoints)."""

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config)
        self.config = text_config
        self.vocab_size = text_config.vocab_size
        self.model = Gemma4Model(text_config, opconfig)
        self.lm_head = Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )


class Gemma4ForConditionalGeneration(Gemma4Base):
    """Gemma4 multimodal checkpoints (text-only execution in PACE).

    The published Gemma 4 checkpoints declare ``Gemma4ForConditionalGeneration``
    and nest the text model under ``model.language_model.``. Vision/audio towers
    are skipped at load time; only text inputs are supported.
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config)

        PACE_LLM_WARNING(
            "Gemma4ForConditionalGeneration currently only supports text inputs. "
            "Vision/audio inputs are not implemented."
        )

        self.full_config = config
        self.config = text_config
        self.vocab_size = text_config.vocab_size
        self.model = Gemma4Model(text_config, opconfig)
        self.lm_head = Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            backend_impl=opconfig.lm_head,
        )
