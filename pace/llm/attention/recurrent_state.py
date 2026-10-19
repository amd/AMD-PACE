# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Per-request Gated DeltaNet (linear-attention) recurrent state.

Hybrid models such as Qwen3.5 carry two kinds of decode state: the standard
KV cache (full-attention layers) and the Gated DeltaNet conv/recurrent memory
(linear-attention layers). The KV cache is already owned per request and merged
into a batch by the cache backend; this module gives the GDN state the *same*
per-request lifecycle so it works under online/continuous-batched serving.

Two small classes:

- :class:`RecurrentStateStore` holds one ``(conv_state, recurrent_state)`` slot
  per linear-attention layer for a single request (or, offline, for a whole
  fixed batch). ``gather``/``commit`` are pass-through.
- :class:`MergedRecurrentState` is an ephemeral per-step view over an ordered
  list of per-request stores. ``gather`` stacks each request's ``[1, ...]`` slot
  into a ``[B, ...]`` batch (in list order, matching the KV-cache merge order);
  ``commit`` scatters the ``[B, ...]`` result rows back into each request store.

Prefill GDN ops return a new state, but decode conv mutates ``conv_state``
in place. The batched state is gathered before the forward and scattered
back after; scatter clones so per-request stores do not share storage.
"""

from typing import List, Optional, Tuple

import torch

StatePair = Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]


def num_recurrent_layers(config) -> int:
    """Number of linear-attention (Gated DeltaNet) layers in a model config.

    Returns 0 for non-hybrid / full-attention-only models, in which case no
    recurrent state is ever allocated or used."""
    layer_types = getattr(config, "layer_types", None) or []
    return sum(1 for lt in layer_types if lt == "linear_attention")


class RecurrentStateStore:
    """Owns the GDN conv/recurrent state for one request (or one offline batch).

    One ``(conv, recurrent)`` slot per linear-attention layer. Slots start as
    ``(None, None)`` so a fresh request's first forward runs the prefill path of
    the GDN ops, which then populate the slot."""

    def __init__(self, num_layers: int):
        self._conv: List[Optional[torch.Tensor]] = [None] * num_layers
        self._rec: List[Optional[torch.Tensor]] = [None] * num_layers

    @property
    def num_layers(self) -> int:
        return len(self._conv)

    def gather(self, idx: int) -> StatePair:
        return self._conv[idx], self._rec[idx]

    def commit(self, idx: int, conv: torch.Tensor, rec: torch.Tensor) -> None:
        self._conv[idx] = conv
        self._rec[idx] = rec


class MergedRecurrentState:
    """Batched per-step view over an ordered list of per-request stores.

    The list order MUST match the batch order of the merged KV cache / input
    tensors, so GDN state row ``i`` corresponds to request row ``i``."""

    def __init__(self, stores: List[RecurrentStateStore]):
        self._stores = stores

    @property
    def stores(self) -> List[RecurrentStateStore]:
        return self._stores

    @property
    def num_requests(self) -> int:
        return len(self._stores)

    def gather(self, idx: int) -> StatePair:
        convs = [s.gather(idx)[0] for s in self._stores]
        recs = [s.gather(idx)[1] for s in self._stores]
        return self._stack(convs), self._stack(recs)

    def commit(self, idx: int, conv: torch.Tensor, rec: torch.Tensor) -> None:
        conv_rows = self._split(conv)
        rec_rows = self._split(rec)
        for i, store in enumerate(self._stores):
            store.commit(idx, conv_rows[i], rec_rows[i])

    @staticmethod
    def _stack(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        present = [t for t in tensors if t is not None]
        if not present:
            return None
        # Fresh prefills are still (None, None) while decode rows already
        # hold conv/recurrent memory. Zero-fill missing rows so a mixed
        # batch can run; zeros match the None/empty initial state of the
        # GDN ops. Packed mixed forwards should still run per-request.
        template = present[0]
        filled = [t if t is not None else torch.zeros_like(template) for t in tensors]
        return torch.cat(filled, dim=0)

    def _split(self, batched: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        if batched is None:
            return [None] * len(self._stores)
        # torch.split returns views of the batched tensor. Decode conv
        # mutates conv_state in-place, so each request must own a copy.
        return [t.clone() for t in torch.split(batched, 1, dim=0)]


def resolve_recurrent_state(kv_cache, num_layers: int):
    """Resolve the per-forward recurrent-state view from the cache context(s).

    Model-agnostic helper for hybrid / SSM-style models (e.g. Qwen3.5 Gated
    DeltaNet): a list of contexts (batched online decode) becomes a
    :class:`MergedRecurrentState` in batch order; a single context exposes its
    own :class:`RecurrentStateStore`. Falls back to an ephemeral (stateless)
    store when the context carries none (e.g. a bare forward with
    ``kv_cache=None``). Returns ``None`` when the model has no recurrent layers.
    """
    if num_layers == 0:
        return None
    if isinstance(kv_cache, list):
        stores = [getattr(m, "recurrent_state", None) for m in kv_cache]
        if any(s is None for s in stores):
            return RecurrentStateStore(num_layers)
        return MergedRecurrentState(stores)
    store = getattr(kv_cache, "recurrent_state", None)
    return store if store is not None else RecurrentStateStore(num_layers)
