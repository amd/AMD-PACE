# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from pace.llm.attention.specs import resolve_text_config
from pace.llm.sampler import Sampler
from pace.utils.logging import PACE_LLM_ASSERT


@dataclass
class DraftTree:
    """
    A speculative tree, stored as parallel arrays indexed by node.

    Node indices coincide with KV slot order, since nodes are created in the order they are written to the cache.
    """

    tokens: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)

    def add(self, token: int, parent: int, depth: int, log_prob: float) -> int:
        """
        Append a node.

        Args:
            token: Proposed token id.
            parent: Parent node index, or -1 for a root.
            depth: Levels below the accepted context.
            log_prob: Cumulative log probability along the path.

        Returns:
            int: Index of the new node.
        """
        self.tokens.append(token)
        self.parents.append(parent)
        self.depths.append(depth)
        self.log_probs.append(log_prob)
        return len(self.tokens) - 1

    def __len__(self) -> int:
        return len(self.tokens)

    def ancestors(self, node: int) -> List[int]:
        """
        Walk from a node to its root, excluding the node itself.

        Args:
            node: Node index.

        Returns:
            list: Ancestor indices, nearest first.
        """
        chain = []
        current = self.parents[node]
        while current != -1:
            chain.append(current)
            current = self.parents[current]
        return chain

    def children(self) -> Dict[int, List[int]]:
        """
        Group nodes by parent, with key -1 holding the roots.

        Returns:
            dict: Parent index to child indices.
        """
        mapping: Dict[int, List[int]] = {}
        for index, parent in enumerate(self.parents):
            mapping.setdefault(parent, []).append(index)
        return mapping


def level_widths(width: int, depth: int, mode: str) -> List[int]:
    """
    Branching factor per tree level.

    Args:
        width: Base branching factor.
        depth: Number of levels.
        mode: "narrow" tapers by one per level, "full" holds `width`.

    Returns:
        list: Branching factor for each level.
    """
    if mode == "full":
        return [width] * depth
    return [max(1, width - level) for level in range(depth)]


def additive_mask(
    context_length: int,
    visible: Sequence[Sequence[int]],
    total_slots: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build an additive attention mask over context plus speculative slots.

    Every row attends to the whole committed context, and beyond that only to the slots it is given, which keeps sibling branches independent.

    Args:
        context_length: Committed tokens already in the cache.
        visible: Speculative slots visible to each row.
        total_slots: Speculative slots in this forward.
        dtype: Model dtype.
        device: Model device.

    Returns:
        torch.Tensor: [1, 1, rows, context_length + total_slots].
    """
    rows = len(visible)
    mask = torch.full(
        (rows, context_length + total_slots),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask[:, :context_length] = 0.0
    for row, slots in enumerate(visible):
        for slot in slots:
            mask[row, context_length + slot] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


def windowed(
    mask: torch.Tensor,
    window: Optional[int],
    query_entries: torch.Tensor,
    key_entries: torch.Tensor,
) -> torch.Tensor:
    """
    Hide keys outside a sliding window.

    A key is inside when it is fewer than `window` cache entries behind the query, as in transformers' sliding-window masks.

    Args:
        mask: [1, 1, rows, keys] additive mask.
        window: Window size, or None for full attention.
        query_entries: [rows] cache entry of each row.
        key_entries: [keys] cache entry of each key.

    Returns:
        torch.Tensor: The mask, unchanged when `window` is None.
    """
    if window is None:
        return mask
    behind = query_entries.reshape(-1, 1) - key_entries.reshape(1, -1)
    return mask.masked_fill(behind >= window, torch.finfo(mask.dtype).min)


@dataclass(frozen=True)
class AttentionWindows:
    """
    The sliding windows of a model's layers, for the masks STeP builds itself.

    transformers passes a prepared 4D mask to its layers unchanged, so each sliding-window layer's window has to be in the mask. Whether the model takes one mask, or one per layer type, follows `create_masks_for_generate`.

    Windows are measured in cache entries, as transformers measures them. A tree node counts as the entry it would take if committed.

    Args:
        window: Window of the single mask every layer takes, or None for full attention.
        by_type: Window per layer type, None meaning full attention, for models that take one mask per type.
    """

    window: Optional[int] = None
    by_type: Optional[Dict[str, Optional[int]]] = None

    @classmethod
    def of(cls, config) -> "AttentionWindows":
        """
        Read the windows from a model config.

        Args:
            config: Model config. A multimodal config resolves to its text config.

        Returns:
            AttentionWindows: The windows of the model's layers.
        """
        text_config = resolve_text_config(config)
        PACE_LLM_ASSERT(
            getattr(text_config, "attention_chunk_size", None) is None,
            "STeP does not support chunked attention",
        )
        window = getattr(text_config, "sliding_window", None)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            return cls(window=window)
        by_type = {
            kind: window if kind == "sliding_attention" else None
            for kind in dict.fromkeys(layer_types)
        }
        unsupported = sorted(set(by_type) - {"full_attention", "sliding_attention"})
        PACE_LLM_ASSERT(
            not unsupported, f"STeP does not support {', '.join(unsupported)} layers"
        )
        if len(by_type) == 1:
            return cls(window=next(iter(by_type.values())))
        return cls(by_type=by_type)

    @property
    def sliding(self) -> bool:
        """Whether any layer attends through a window."""
        windows = self.by_type.values() if self.by_type is not None else [self.window]
        return any(window is not None for window in windows)

    def apply(
        self,
        mask: torch.Tensor,
        query_entries: torch.Tensor,
        key_entries: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Narrow an additive mask to each layer's window.

        Args:
            mask: [1, 1, rows, keys] additive mask.
            query_entries: [rows] cache entry of each row.
            key_entries: [keys] cache entry of each key.

        Returns:
            The model's `attention_mask`: one mask, or a dict of masks keyed by layer type.
        """
        if self.by_type is None:
            return windowed(mask, self.window, query_entries, key_entries)
        return {
            kind: windowed(mask, window, query_entries, key_entries)
            for kind, window in self.by_type.items()
        }


def draft_keep_indices(
    cached: int, budget: Optional[int], sink: int
) -> Optional[List[int]]:
    """
    Choose which draft cache entries survive when the budget is exceeded.

    The retention policy is isolated here so an importance-based token pruner can replace this function alone. The policy is positional: a leading sink plus the most recent tail. The sink retains instruction tokens that a purely recency-based window would discard.

    Args:
        cached: Committed entries currently held.
        budget: Entries the draft may keep, or None for no cap.
        sink: Leading entries to retain.

    Returns:
        Optional[list]: Entry indices to keep, in order, or None when everything already fits.
    """
    if budget is None or cached <= budget:
        return None
    sink = max(0, min(sink, budget - 1))
    tail = budget - sink
    return list(range(sink)) + list(range(cached - tail, cached))


def gather_cache(cache, indices: Sequence[int]) -> None:
    """
    Reduce a cache to the given entries, in the order given.

    Indices may be arbitrary and scattered, so any retention policy can be expressed.

    Args:
        cache: An HF cache whose layers expose `keys` and `values`.
        indices: Entry indices to retain.
    """
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False) or layer.keys.numel() == 0:
            continue
        chosen = torch.tensor(list(indices), device=layer.keys.device)
        layer.keys = layer.keys.index_select(2, chosen).contiguous()
        layer.values = layer.values.index_select(2, chosen).contiguous()


def compact_cache(cache, context_length: int, keep_slots: Sequence[int]) -> None:
    """
    Drop rejected speculative entries, sliding the accepted ones down.

    Only the speculative tail moves, so cost is proportional to the number of accepted tokens rather than to the context length.

    Args:
        cache: An HF cache whose layers expose `keys` and `values`.
        context_length: Committed tokens, which are left untouched.
        keep_slots: Speculative slots to retain, in commit order.
    """
    new_length = context_length + len(keep_slots)
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False) or layer.keys.numel() == 0:
            continue
        if keep_slots:
            source = torch.tensor(
                [context_length + slot for slot in keep_slots],
                device=layer.keys.device,
            )
            layer.keys[:, :, context_length:new_length] = layer.keys.index_select(
                2, source
            )
            layer.values[:, :, context_length:new_length] = layer.values.index_select(
                2, source
            )
        layer.keys = layer.keys[:, :, :new_length]
        layer.values = layer.values[:, :, :new_length]


class TreeDrafter:
    """
    Grows a speculative tree with the GPU prefix subnetwork.

    One forward per level expands the entire frontier, so cost scales with tree depth rather than with candidate count. Nodes at the same level share a position id, since they are competing alternatives for one sequence position.

    The draft cache is maintained as a contiguous prefix. Instead of gathering accepted slots out of the speculative tail, it is cropped to the committed length and the newly committed tokens are replayed through one GPU-only forward.

    When a window is set, the oldest draft cache entries are discarded. Cache slots and RoPE positions are tracked separately, so trimming changes a token's slot but not its position. Sliding-window layers measure their window in slots, as transformers does when it replays tokens into the trimmed cache.
    """

    def __init__(
        self,
        model,
        width: int,
        depth: int,
        mode: str = "narrow",
        window: Optional[int] = None,
        sink: int = 0,
        attention_windows: AttentionWindows = AttentionWindows(),
    ):
        """
        Args:
            model: The split model. Callers put it in draft mode.
            width: Base branching factor.
            depth: Tree levels.
            mode: "narrow" or "full".
            window: Cap on cached context entries, or None for no cap.
            sink: Leading entries always retained when windowing.
            attention_windows: Sliding windows of the model's layers.
        """
        self.model = model
        self.widths = level_widths(width, depth, mode)
        self.window = window
        self.sink = sink
        self.attention_windows = attention_windows

    @torch.no_grad()
    def grow(
        self,
        pending: Sequence[int],
        cache,
        cached_context: int,
        absolute_position: int,
    ) -> Tuple[DraftTree, int]:
        """
        Build a tree continuing from tokens the draft has not seen yet.

        Args:
            pending: Committed tokens not yet in the draft cache.
            cache: Draft KV cache.
            cached_context: Committed entries currently held in `cache`.
            absolute_position: Sequence position of the first pending token.

        Returns:
            tuple: (tree, committed entries in the cache afterwards). Node `i` occupies speculative slot `i`.
        """
        PACE_LLM_ASSERT(len(pending) > 0, "the draft needs at least one new token")
        device = next(self.model.parameters()).device
        tree = DraftTree()

        cache.crop(cached_context)
        keep = draft_keep_indices(cached_context, self.window, self.sink)
        if keep is not None:
            gather_cache(cache, keep)
            cached_context = len(keep)

        catch_up = torch.tensor([list(pending)], device=device)
        outputs = self.model(
            input_ids=catch_up,
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.tensor(
                [list(range(absolute_position, absolute_position + len(pending)))],
                device=device,
            ),
            cache_position=torch.arange(
                cached_context, cached_context + len(pending), device=device
            ),
        )
        # The replayed tokens are now committed context; the tree sits after them.
        base = cached_context + len(pending)
        base_position = absolute_position + len(pending)

        scores = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
        top = torch.topk(scores, self.widths[0])
        frontier = [
            tree.add(token, -1, 0, score)
            for token, score in zip(top.indices.tolist(), top.values.tolist())
        ]

        for level in range(1, len(self.widths)):
            if not frontier:
                break
            tokens = torch.tensor(
                [[tree.tokens[node] for node in frontier]], device=device
            )
            positions = torch.tensor(
                [[base_position + level - 1] * len(frontier)], device=device
            )
            cache_position = torch.tensor(
                [base + node for node in frontier], device=device
            )
            visible = [[node] + tree.ancestors(node) for node in frontier]
            slots = max(frontier) + 1
            mask = additive_mask(base, visible, slots, self.model.dtype, device)
            if self.attention_windows.sliding:
                entries = torch.tensor(tree.depths[:slots], device=device) + base
                mask = self.attention_windows.apply(
                    mask,
                    entries[frontier],
                    torch.cat([torch.arange(base, device=device), entries]),
                )

            outputs = self.model(
                input_ids=tokens,
                past_key_values=cache,
                use_cache=True,
                position_ids=positions,
                cache_position=cache_position,
                attention_mask=mask,
            )

            scores = torch.log_softmax(outputs.logits[0].float(), dim=-1)
            top = torch.topk(scores, self.widths[level], dim=-1)
            next_frontier = []
            for node, row_tokens, row_scores in zip(
                frontier, top.indices.tolist(), top.values.tolist()
            ):
                for token, score in zip(row_tokens, row_scores):
                    next_frontier.append(
                        tree.add(token, node, level, tree.log_probs[node] + score)
                    )
            frontier = next_frontier

        return tree, base


@dataclass
class VerificationResult:
    """
    Outcome of verifying one tree against the full model.

    Args:
        committed: Tokens committed this step: the accepted path plus the bonus.
        accepted: Draft tokens accepted, excluding the bonus.
        context_length: Committed context length after the cache is compacted.
        proposed: Candidates the tree offered, for acceptance-rate accounting.
        probs: The full model's distribution at each committed position, [committed, vocab], when the sampler computed one.
        logprobs: Log of `probs`, under the same condition.
    """

    committed: List[int]
    accepted: int
    context_length: int
    proposed: int
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


class TreeVerifier:
    """
    Checks a speculative tree against the full model in one forward.

    The whole tree is verified under a mask that keeps branches independent, so verification costs one forward regardless of candidate count. Acceptance descends the tree, admitting a child when it matches the full model's prediction at its parent.

    Committed tokens the full model has not yet processed are prepended to the batch. This yields the prediction for the tree's root position from the same forward, avoiding a second pass over the CPU residual.

    The full model's token at every node is drawn by the request's sampler. A node is accepted only when it equals that draw, so each committed token is a sample from the full model given the committed prefix, whatever the draft proposed.
    """

    def __init__(self, model, attention_windows: AttentionWindows = AttentionWindows()):
        """
        Args:
            model: The split model. Callers put it in verify mode.
            attention_windows: Sliding windows of the model's layers.
        """
        self.model = model
        self.attention_windows = attention_windows

    @torch.no_grad()
    def verify(
        self,
        tree: DraftTree,
        cache,
        context_length: int,
        pending: Sequence[int],
        sampler: Sampler,
        history: torch.Tensor,
    ) -> VerificationResult:
        """
        Verify a tree and commit the longest path the full model agrees with.

        Args:
            tree: Tree to verify.
            cache: Verify KV cache holding `context_length` committed tokens.
            context_length: Committed tokens in `cache`.
            pending: Committed tokens the full model has not processed yet.
            sampler: The request's sampler, which draws the full model's token at every node.
            history: [1, tokens] committed so far, which the sampler takes alongside the logits.

        Returns:
            VerificationResult: Committed tokens and the new context length.
        """
        PACE_LLM_ASSERT(len(tree) > 0, "cannot verify an empty tree")
        PACE_LLM_ASSERT(len(pending) > 0, "verification needs at least one new token")
        device = next(self.model.parameters()).device
        lead = len(pending)

        tokens = torch.tensor([list(pending) + tree.tokens], device=device)
        positions = torch.tensor(
            [
                list(range(context_length, context_length + lead))
                + [context_length + lead + depth for depth in tree.depths]
            ],
            device=device,
        )
        cache_position = torch.arange(
            context_length, context_length + lead + len(tree), device=device
        )
        # Pending tokens are a plain causal run; tree nodes additionally see all of them, since the tree continues from the last one.
        visible = [list(range(index + 1)) for index in range(lead)]
        visible += [
            list(range(lead))
            + [lead + node]
            + [lead + ancestor for ancestor in tree.ancestors(node)]
            for node in range(len(tree))
        ]
        mask = additive_mask(
            context_length, visible, lead + len(tree), self.model.dtype, device
        )
        if self.attention_windows.sliding:
            # The verify cache holds every committed token, so a token's entry is its position.
            mask = self.attention_windows.apply(
                mask,
                positions[0],
                torch.cat([torch.arange(context_length, device=device), positions[0]]),
            )

        outputs = self.model(
            input_ids=tokens,
            past_key_values=cache,
            use_cache=True,
            position_ids=positions,
            cache_position=cache_position,
            attention_mask=mask,
        )
        # Row 0 predicts the tree's root position, and row 1 + i the position after node i.
        sampled = sampler.sample(history, outputs.logits[0, lead - 1 :].float())
        predictions = sampled.next_tokens.reshape(-1).tolist()

        children = tree.children()
        accepted: List[int] = []
        target = predictions[0]
        candidates = children.get(-1, [])
        while True:
            match = next(
                (node for node in candidates if tree.tokens[node] == target), None
            )
            if match is None:
                break
            accepted.append(match)
            target = predictions[1 + match]
            candidates = children.get(match, [])

        keep = list(range(lead)) + [lead + node for node in accepted]
        compact_cache(cache, context_length, keep)

        # Each committed token was drawn from the row of its parent.
        rows = [0] + [1 + node for node in accepted]
        return VerificationResult(
            committed=[tree.tokens[node] for node in accepted] + [target],
            accepted=len(accepted),
            context_length=context_length + len(keep),
            proposed=len(tree),
            probs=sampled.probs[rows] if sampled.probs is not None else None,
            logprobs=sampled.logprobs[rows] if sampled.logprobs is not None else None,
        )
