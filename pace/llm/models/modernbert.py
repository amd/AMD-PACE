# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc.
# team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/modernbert/modeling_modernbert.py

"""ModernBERT encoder for embedding / classification (e.g. mmBERT).

Bidirectional, cacheless encoder built on PACE ops with the ModernBERT
specifics: RoPE (no learned position embeddings), alternating global / local
(sliding-window) attention, pre-norm residual blocks, and a gated-GeLU MLP.
Module names mirror HuggingFace ``ModernBertModel`` so checkpoint weights load
by direct name match (a leading ``model.`` prefix is stripped when present).
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from pace.llm.attention import Attention
from pace.llm.configs import OperatorConfig
from pace.llm.models.base_embedding_model import BaseModelForEmbedding
from pace.llm.outputs import (
    ClassifierOutput,
    EmbeddingOutput,
    TokenClassifierOutput,
)
from pace.llm.ops import (
    Activation,
    LayerNorm,
    Linear,
    MergedMLP,
    RotaryEmbedding,
)

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


def _layer_attention_type(config: PretrainedConfig, layer_id: int) -> str:
    """Attention type of ``layer_id``: global (full) or local (sliding window)."""
    return config.layer_types[layer_id]


def _layer_rope_parameters(config: PretrainedConfig, attn_type: str) -> Dict[str, Any]:
    """RoPE settings for one ModernBERT attention type.

    ``ModernBertConfig.rope_parameters`` is keyed by layer type, so global and
    local layers each carry their own ``rope_type`` / ``rope_theta`` (and, for a
    YaRN-scaled 32K checkpoint, ``factor`` and friends).

    Both this and ``layer_types`` are read directly, with no fallback for the
    pre-v5 ``global_attn_every_n_layers`` / ``global_rope_theta`` spelling,
    because the config class materialises them from those older fields on load
    and rejects any other shape. ``test_transformers_normalises_legacy_config``
    pins that behaviour.
    """
    return dict(config.rope_parameters[attn_type])


def _rotary_embedding(config: PretrainedConfig, attn_type: str) -> RotaryEmbedding:
    """The ``RotaryEmbedding`` every layer of ``attn_type`` shares."""
    rope_params = _layer_rope_parameters(config, attn_type)
    return RotaryEmbedding(
        rope_scaling=rope_params,
        rotary_dim=config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=rope_params["rope_theta"],
    )


def _norm(config: PretrainedConfig, opconfig: OperatorConfig) -> LayerNorm:
    return LayerNorm(
        config.hidden_size,
        eps=config.norm_eps,
        bias=getattr(config, "norm_bias", False),
        backend_impl=opconfig.norm,
    )


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.norm = _norm(config, opconfig)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        opconfig: OperatorConfig,
        layer_id: int,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        attention_bias = getattr(config, "attention_bias", False)

        self.Wqkv = Linear(
            self.embed_dim,
            3 * self.embed_dim,
            bias=attention_bias,
            backend_impl=opconfig.qkv_projection,
        )
        self.Wo = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=attention_bias,
            backend_impl=opconfig.out_projection,
        )

        self.attention_type = _layer_attention_type(config, layer_id)
        self.is_global = self.attention_type == FULL_ATTENTION
        local_attention = getattr(config, "local_attention", 128)
        sliding_window = 0 if self.is_global else int(local_attention)

        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
            sliding_window=sliding_window,
            is_causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        apply_rope: Callable,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q, k = apply_rope(query=q, key=k)

        out = self.attn(q, k, v, None, None, attention_mask=attention_mask)
        out = out.reshape(bsz, seq_len, self.embed_dim)
        return self.Wo(out)


def _mlp(config: PretrainedConfig, opconfig: OperatorConfig) -> MergedMLP:
    """ModernBERT's gated MLP: ``Wo(act(input) * gate)``.

    This is exactly :class:`MergedMLP`'s gated form, so the shared operator is
    used rather than a hand-rolled chunk-and-multiply: in bfloat16 it dispatches
    to the fused MLP kernel, which the Python version cannot reach.

    The checkpoint's single ``Wi`` is split across ``gate_proj`` and ``up_proj``
    at load time -- see :func:`_checkpoint_entries`.
    """
    return MergedMLP(
        in_features=config.hidden_size,
        out_features=config.intermediate_size,
        bias=getattr(config, "mlp_bias", False),
        activation=getattr(config, "hidden_activation", None) or "gelu",
        gate=True,
        backend_impl=opconfig.mlp,
    )


def _checkpoint_entries(
    name: str, tensor: torch.Tensor
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Expand one checkpoint tensor into the parameters that consume it.

    ModernBERT keeps the MLP's two projections in a single ``Wi`` of width
    ``2 * intermediate_size``, which :class:`MergedMLP` holds as two. HF
    computes ``act(input) * gate`` over ``Wi(x).chunk(2)``, so ``Wi``'s first
    half is the activated one and maps to ``gate_proj``, and the second half is
    the multiplier and maps to ``up_proj``. The names read backwards against
    HF's, but the arithmetic is the same.
    """
    if "mlp.Wi." in name:
        prefix, suffix = name.split("mlp.Wi.", 1)
        activated, multiplier = tensor.chunk(2, dim=0)
        yield f"{prefix}mlp.gate_proj.linear.{suffix}", activated
        yield f"{prefix}mlp.up_proj.linear.{suffix}", multiplier
    elif "mlp.Wo." in name:
        yield name.replace("mlp.Wo.", "mlp.down_proj."), tensor
    else:
        yield name, tensor


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        opconfig: OperatorConfig,
        layer_id: int,
    ):
        super().__init__()
        # Layer 0's attention is applied directly on the (already-normed)
        # embedding output, so attn_norm is an identity there.
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = _norm(config, opconfig)
        self.attn = ModernBertAttention(config, opconfig, layer_id)
        self.mlp_norm = _norm(config, opconfig)
        self.mlp = _mlp(config, opconfig)

    def forward(
        self,
        hidden_states: torch.Tensor,
        apply_rope: Callable,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), apply_rope, attention_mask
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertModel(BaseModelForEmbedding):
    """HuggingFace ``ModernBertModel``-compatible encoder."""

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.embeddings = ModernBertEmbeddings(config, opconfig)
        # ModernBERT alternates global and local layers and trains each kind
        # with its own RoPE base, so there are exactly two configurations. They
        # live on the model rather than in each attention block: one cos/sin
        # cache pair per kind instead of per layer (16 MiB rather than 176 MiB
        # on a 32K-context, 22-layer model), and RoPE is evaluated twice per
        # forward instead of once per layer.
        self.rotary_emb_global = _rotary_embedding(config, FULL_ATTENTION)
        self.rotary_emb_local = _rotary_embedding(config, SLIDING_ATTENTION)
        self.layers = nn.ModuleList(
            [
                ModernBertEncoderLayer(config, opconfig, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = _norm(config, opconfig)

    def _encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        hidden_states = self.embeddings(input_ids)
        # Q/K reach RoPE as BSHD, hence unsqueeze_dim=2. Both layouts of cos/sin
        # are gathered once here and reused by every layer of that kind.
        rope = {
            True: self.rotary_emb_global(hidden_states, positions, unsqueeze_dim=2),
            False: self.rotary_emb_local(hidden_states, positions, unsqueeze_dim=2),
        }
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, rope[layer.attn.is_global], attention_mask
            )
        return self.final_norm(hidden_states)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        last_hidden_state = self._encode(input_ids, attention_mask)
        return EmbeddingOutput(
            embeddings=last_hidden_state[:, 0],
            last_hidden_state=last_hidden_state,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in weight_iterator:
            if name.startswith("model."):
                name = name[len("model.") :]
            name = self.rename_fused_params(name)
            for target, value in _checkpoint_entries(name, tensor):
                if target not in params:
                    continue
                param = params[target]
                if hasattr(param, "load_weights"):
                    param.load_weights(param, value)
                else:
                    param.data.copy_(value)


class ModernBertForSequenceClassification(BaseModelForEmbedding):
    """ModernBERT encoder + prediction head + linear classifier (mmBERT classifiers)."""

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier_pooling = getattr(config, "classifier_pooling", "cls")
        self.model = ModernBertModel(config, opconfig)
        self.head_dense = Linear(
            config.hidden_size,
            config.hidden_size,
            bias=getattr(config, "classifier_bias", False),
        )
        self.head_act = Activation(
            getattr(config, "classifier_activation", None) or "gelu"
        )
        self.head_norm = _norm(config, opconfig)
        self.classifier = Linear(
            config.hidden_size,
            config.num_labels,
            bias=getattr(config, "classifier_bias", False),
        )

    def _pool(
        self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.classifier_pooling == "mean":
            if attention_mask is None:
                # transformers treats a missing mask as all-ones and still
                # mean-pools. Falling back to CLS here instead would silently
                # return different logits for the same checkpoint.
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        last_hidden_state = self.model._encode(input_ids, attention_mask)
        pooled = self._pool(last_hidden_state, attention_mask)
        pooled = self.head_norm(self.head_act(self.head_dense(pooled)))
        logits = self.classifier(pooled)
        return ClassifierOutput(logits=logits)

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))
        alias = {
            "head.dense.weight": "head_dense.weight",
            "head.dense.bias": "head_dense.bias",
            "head.norm.weight": "head_norm.weight",
            "head.norm.bias": "head_norm.bias",
        }
        for name, tensor in weight_iterator:
            for target, value in _checkpoint_entries(alias.get(name, name), tensor):
                if target not in params:
                    continue
                param = params[target]
                if hasattr(param, "load_weights"):
                    param.load_weights(param, value)
                else:
                    param.data.copy_(value)


class ModernBertForTokenClassification(BaseModelForEmbedding):
    """ModernBERT encoder + prediction head + per-token classifier (NER / PII).

    Same prediction head (dense -> act -> norm) as the sequence classifier, but
    applied to *every* token rather than a pooled vector, producing logits of
    shape ``[batch, seq_len, num_labels]``. Weight names mirror HF
    ``ModernBertForTokenClassification`` (encoder under ``model.*``, head as
    ``head.dense`` / ``head.norm``, classifier as ``classifier.*``).
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ModernBertModel(config, opconfig)
        self.head_dense = Linear(
            config.hidden_size,
            config.hidden_size,
            bias=getattr(config, "classifier_bias", False),
        )
        self.head_act = Activation(
            getattr(config, "classifier_activation", None) or "gelu"
        )
        self.head_norm = _norm(config, opconfig)
        self.classifier = Linear(
            config.hidden_size,
            config.num_labels,
            bias=getattr(config, "classifier_bias", False),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        last_hidden_state = self.model._encode(input_ids, attention_mask)
        hidden = self.head_norm(self.head_act(self.head_dense(last_hidden_state)))
        bsz, seq_len, dim = hidden.shape
        logits = self.classifier(hidden.reshape(bsz * seq_len, dim)).reshape(
            bsz, seq_len, self.num_labels
        )
        return TokenClassifierOutput(logits=logits)

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))
        alias = {
            "head.dense.weight": "head_dense.weight",
            "head.dense.bias": "head_dense.bias",
            "head.norm.weight": "head_norm.weight",
            "head.norm.bias": "head_norm.bias",
        }
        for name, tensor in weight_iterator:
            for target, value in _checkpoint_entries(alias.get(name, name), tensor):
                if target not in params:
                    continue
                param = params[target]
                if hasattr(param, "load_weights"):
                    param.load_weights(param, value)
                else:
                    param.data.copy_(value)
