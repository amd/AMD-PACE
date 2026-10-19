# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-layer KV cache specifications.

vLLM allocates KV cache *per attention layer* from each layer's spec, and lets
a layer reuse another layer's KV via ``kv_sharing_target_layer_name``. This
module provides the PACE analog: a small :class:`LayerKVSpec` and a
config-driven :func:`compute_layer_kv_specs` that derives per-layer
``(num_kv_heads, head_dim, sliding_window, kv_sharing_target)`` from a standard
HuggingFace config.

For ordinary models every layer has identical dims and no sharing, so the
result is uniform and downstream cache backends behave exactly as before.
Models with per-layer variation (e.g. Gemma4: different head_dim on global vs
sliding layers, and a shared KV cache for the trailing layers) are handled
transparently because the relevant fields already live in their config
(``layer_types``, ``global_head_dim``, ``num_kv_shared_layers``, ...).
"""

from dataclasses import dataclass
from typing import List, Optional

from transformers import PretrainedConfig


def resolve_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the sub-config describing the text decoder.

    KV-cache consumers must size layer counts and dims through this helper so
    they agree with :func:`compute_layer_kv_specs` on multimodal wrappers.
    """
    return getattr(config, "text_config", config)


@dataclass(frozen=True)
class LayerKVSpec:
    """KV-cache shape/sharing spec for a single attention layer.

    Attributes:
        num_kv_heads: Number of key/value heads stored for this layer.
        head_dim: Per-head dimension stored for this layer.
        sliding_window: Sliding-window size (0 == full attention).
        kv_sharing_target: If set, the index of an earlier layer whose KV
            cache this layer reuses. Such a layer allocates no storage of its
            own and performs no cache write (mirrors vLLM's
            ``kv_sharing_target_layer_name``).
    """

    num_kv_heads: int
    head_dim: int
    sliding_window: int = 0
    kv_sharing_target: Optional[int] = None

    @property
    def is_shared(self) -> bool:
        return self.kv_sharing_target is not None


def compute_layer_kv_specs(config: PretrainedConfig) -> List[LayerKVSpec]:
    """Derive per-layer KV specs from a HuggingFace config.

    Uniform models yield identical specs with no sharing. Per-layer variation
    is picked up from optional fields:
      - ``layer_types`` / ``sliding_window``: per-layer sliding window.
      - ``global_head_dim``: larger head_dim on full-attention layers.
      - ``attention_k_eq_v`` / ``num_global_key_value_heads``: KV-head count on
        full-attention layers when keys are reused as values.
      - ``num_kv_shared_layers``: trailing layers reuse the KV of the last
        non-shared layer of the same attention type.

    Raises:
        ValueError: If a trailing shared layer has no earlier layer of its
            attention type to reuse (mirrors vLLM, which also refuses rather
            than falling back to per-layer storage).
    """
    text = resolve_text_config(config)
    num_layers = text.num_hidden_layers
    num_heads = getattr(text, "num_attention_heads", 0) or 0
    default_kv_heads = getattr(text, "num_key_value_heads", None) or num_heads
    default_head_dim = getattr(text, "head_dim", None)
    if not default_head_dim:
        hidden_size = getattr(text, "hidden_size", 0) or 0
        default_head_dim = hidden_size // num_heads if num_heads else 0
    global_head_dim = getattr(text, "global_head_dim", None)
    layer_types = getattr(text, "layer_types", None)
    sliding_window = getattr(text, "sliding_window", 0) or 0
    k_eq_v = getattr(text, "attention_k_eq_v", False)
    global_kv_heads = getattr(text, "num_global_key_value_heads", None)

    num_shared = getattr(text, "num_kv_shared_layers", 0) or 0
    first_shared = max(0, num_layers - num_shared) if num_shared > 0 else num_layers

    def layer_type_of(i: int) -> str:
        return layer_types[i] if layer_types is not None else "full_attention"

    # Last non-shared layer of each type; shared layers of that type reuse it.
    source_by_type = {}
    if num_shared > 0:
        for i in range(first_shared):
            if layer_type_of(i) != "linear_attention":
                source_by_type[layer_type_of(i)] = i

    specs: List[LayerKVSpec] = []
    spec_index_of_layer = {}
    for i in range(num_layers):
        layer_type = layer_type_of(i)
        # Linear-attention layers (e.g. Qwen3.5 Gated DeltaNet) keep their own
        # recurrent state and never read or write KV, so they get no spec and
        # no storage. The returned list is therefore compact over KV-holding
        # layers, which is how cache backends index it.
        if layer_type == "linear_attention":
            continue
        is_sliding = layer_type == "sliding_attention"

        head_dim = (
            default_head_dim if (is_sliding or not global_head_dim) else global_head_dim
        )
        is_k_eq_v = k_eq_v and not is_sliding
        num_kv_heads = (
            global_kv_heads if (is_k_eq_v and global_kv_heads) else default_kv_heads
        )
        sw = sliding_window if is_sliding else 0

        target = None
        if num_shared > 0 and i >= first_shared:
            source_layer = source_by_type.get(layer_type)
            if source_layer is None:
                raise ValueError(
                    f"Layer {i} is among the last num_kv_shared_layers="
                    f"{num_shared} of {num_layers} layers, so it must reuse the "
                    f"KV of an earlier '{layer_type}' layer, but none exists "
                    f"before index {first_shared}."
                )
            target = spec_index_of_layer[source_layer]

        spec_index_of_layer[i] = len(specs)
        specs.append(LayerKVSpec(num_kv_heads, head_dim, sw, target))

    return specs
