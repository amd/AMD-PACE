# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.
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

# Adapted from https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/bert/modeling_bert.py

"""BERT encoder for embedding and sequence-classification (e.g. bge-small-en-v1.5).

Bidirectional, cacheless encoder built on PACE ops: token/position/token-type
embeddings, LayerNorm, fused QKV + bidirectional attention, and an ungated
MergedMLP (GELU) with residual LayerNorm. HuggingFace checkpoint names are
mapped onto those fused modules in ``load_weights``.
"""

from typing import Iterable, Optional, Tuple

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
from pace.llm.ops import FusedQKVLinear, LayerNorm, Linear, MergedMLP

_QKV_PROJ = {"query": "q", "key": "k", "value": "v"}


def _map_mlp_checkpoint_name(name: str) -> str:
    """HF BERT MLP names -> MergedMLP parameter names.

    ``attention.output.*`` is the attention out-projection and stays put.
    The feed-forward block is ``intermediate.dense`` / ``output.dense`` /
    ``output.LayerNorm``.
    """
    if ".intermediate.dense." in name:
        return name.replace(".intermediate.dense.", ".mlp.up_proj.linear.")
    if ".attention.output." in name:
        return name
    if ".output.dense." in name:
        return name.replace(".output.dense.", ".mlp.down_proj.")
    if ".output.LayerNorm." in name:
        return name.replace(".output.LayerNorm.", ".output_LayerNorm.")
    return name


def _collect_qkv(cache: dict, name: str, tensor: torch.Tensor) -> Optional[str]:
    """If ``name`` is a BERT Q/K/V projection, stash it and return the prefix."""
    for hf_name, short in _QKV_PROJ.items():
        token = f".{hf_name}."
        if token not in name or ".attention.self." not in name:
            continue
        prefix, suffix = name.split(token, 1)
        bucket = "weight" if suffix.startswith("weight") else "bias"
        slot = cache.setdefault(prefix, {"weight": {}, "bias": {}})
        slot[bucket][short] = tensor
        return prefix
    return None


def _fuse_qkv(module: nn.Module, cache: dict) -> None:
    modules = dict(module.named_modules())
    for prefix, tensors in cache.items():
        if not all(x in tensors["weight"] for x in ("q", "k", "v")):
            continue
        fused = modules.get(f"{prefix}.qkv_proj")
        if fused is None:
            continue
        fused.load_from_unfused(tensors)


def _load_named_parameters(
    module: nn.Module,
    weight_iterator: Iterable[Tuple[str, torch.Tensor]],
    extra_alias: Optional[dict] = None,
) -> None:
    params = dict(module.named_parameters(remove_duplicate=False))
    alias = extra_alias or {}
    qkv_cache: dict = {}
    for name, tensor in weight_iterator:
        name = alias.get(name, name)
        name = (
            module.rename_fused_params(name)
            if hasattr(module, "rename_fused_params")
            else name
        )
        name = _map_mlp_checkpoint_name(name)
        if _collect_qkv(qkv_cache, name, tensor) is not None:
            continue
        if name not in params:
            continue
        param = params[name]
        if hasattr(param, "load_weights"):
            param.load_weights(param, tensor)
        else:
            param.data.copy_(tensor)
    _fuse_qkv(module, qkv_cache)


class BertEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend_impl=opconfig.norm,
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(positions)
        return self.LayerNorm(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size

        self.qkv_proj = FusedQKVLinear(
            in_features=self.embed_dim,
            out_features=3 * self.embed_dim,
            bias=True,
            num_key_value_heads=self.num_heads,
            backend_impl=opconfig.qkv_projection,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            opconfig=opconfig,
            is_causal=False,
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        bshd = (bsz, seq_len, self.num_heads, self.head_dim)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        Q, K, V = q.view(*bshd), k.view(*bshd), v.view(*bshd)
        out = self.attn(Q, K, V, None, None, attention_mask=attention_mask)
        return out.reshape(bsz, seq_len, self.embed_dim)


class BertSelfOutput(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.dense = Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            backend_impl=opconfig.out_projection,
        )
        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, backend_impl=opconfig.norm
        )

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.self = BertSelfAttention(config, opconfig)
        self.output = BertSelfOutput(config, opconfig)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        attn = self.self(hidden_states, attention_mask)
        return self.output(attn, hidden_states)


class BertLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.attention = BertAttention(config, opconfig)
        act = config.hidden_act if isinstance(config.hidden_act, str) else "gelu"
        self.mlp = MergedMLP(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=True,
            activation=act,
            gate=False,
            backend_impl=opconfig.mlp,
        )
        self.output_LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, backend_impl=opconfig.norm
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        attn_out = self.attention(hidden_states, attention_mask)
        return self.output_LayerNorm(self.mlp(attn_out) + attn_out)


class BertEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config, opconfig) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(BaseModelForEmbedding):
    """HuggingFace ``BertModel``-compatible encoder returning hidden states."""

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config, opconfig)
        self.encoder = BertEncoder(config, opconfig)

    def _encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        hidden_states = self.embeddings(input_ids, positions, token_type_ids)
        return self.encoder(hidden_states, attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        last_hidden_state = self._encode(input_ids, attention_mask, token_type_ids)
        return EmbeddingOutput(
            embeddings=last_hidden_state[:, 0],
            last_hidden_state=last_hidden_state,
        )

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        def _strip_bert_prefix():
            for name, tensor in weight_iterator:
                if name.startswith("bert."):
                    name = name[len("bert.") :]
                yield name, tensor

        _load_named_parameters(self, _strip_bert_prefix())


class BertForSequenceClassification(BaseModelForEmbedding):
    """BERT encoder + pooled linear classification head."""

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, opconfig)
        self.pooler_dense = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.classifier = Linear(config.hidden_size, config.num_labels, bias=True)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        last_hidden_state = self.bert._encode(input_ids, attention_mask, token_type_ids)
        pooled = torch.tanh(self.pooler_dense(last_hidden_state[:, 0]))
        logits = self.classifier(pooled)
        return ClassifierOutput(logits=logits)

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        _load_named_parameters(
            self,
            weight_iterator,
            extra_alias={
                "bert.pooler.dense.weight": "pooler_dense.weight",
                "bert.pooler.dense.bias": "pooler_dense.bias",
            },
        )


class BertForTokenClassification(BaseModelForEmbedding):
    """BERT encoder + per-token linear classification head (NER / PII).

    Unlike :class:`BertForSequenceClassification` there is no pooler: the
    classifier is applied to *every* token's hidden state, yielding logits of
    shape ``[batch, seq_len, num_labels]``. HF ``BertForTokenClassification``
    keeps encoder weights under ``bert.*`` and the head as ``classifier.*``
    (its inference-time dropout is a no-op), so names match directly.
    """

    def __init__(self, config: PretrainedConfig, opconfig: OperatorConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, opconfig)
        self.classifier = Linear(config.hidden_size, config.num_labels, bias=True)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        last_hidden_state = self.bert._encode(input_ids, attention_mask, token_type_ids)
        bsz, seq_len, hidden = last_hidden_state.shape
        # Apply the per-token head on a flattened [B*S, H] view so the Linear
        # backend never sees a 3-D input, then restore [B, S, num_labels].
        logits = self.classifier(
            last_hidden_state.reshape(bsz * seq_len, hidden)
        ).reshape(bsz, seq_len, self.num_labels)
        return TokenClassifierOutput(logits=logits)

    def load_weights(self, weight_iterator: Iterable[Tuple[str, torch.Tensor]]):
        _load_named_parameters(self, weight_iterator)
