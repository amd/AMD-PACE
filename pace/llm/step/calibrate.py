# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import itertools
import tempfile
import threading
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, modeling_utils
from transformers.utils import logging as hf_logging

from pace.llm.attention.specs import resolve_text_config
from pace.llm.step.budget import GIB
from pace.llm.step.config import StepCalibrationConfig
from pace.llm.step.hf_modules import find_decoder_layers, find_gated_mlps
from pace.llm.step.loading import load_model, on_device
from pace.llm.step.saliency import SaliencyAccumulator, channel_order
from pace.utils.device import require_gpu
from pace.utils.inference_dataset import CalibrationDataset
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_INFO

# transformers' 50 GB default writes an 8B model as one shard, so the save shows no progress until it ends.
MAX_SHARD_SIZE = "5GB"


def permuted_channels(
    weight: torch.Tensor, order: torch.Tensor, columns: bool
) -> torch.Tensor:
    """
    Copy an FFN projection weight with its intermediate channels in `order`.

    The FFN output sums over channels, so permuting `gate` and `up` rows and `down` columns by a common index leaves the block mathematically unchanged. Only the storage order changes, allowing the runtime to take a contiguous prefix as its GPU shard.

    Args:
        weight: A gate_proj or up_proj weight [channels, d_model], or a down_proj weight [d_model, channels].
        order: Permutation of the intermediate channels.
        columns: Whether the channels are the weight's columns, as in down_proj.

    Returns:
        torch.Tensor: The permuted weight, contiguous.
    """
    # index_select is far slower along columns for bf16 on the CPU.
    return (weight[:, order] if columns else weight[order, :]).contiguous()


_shard_writer = threading.local()
_shard_writer_lock = threading.Lock()
_shard_writer_users = 0
_original_safe_save_file = None


def _write_shard(shard, filename, metadata=None):
    """Write a shard with the calling thread's writer, or the original if it set none."""
    write = getattr(_shard_writer, "write", None) or _original_safe_save_file
    write(shard, filename, metadata=metadata)


@contextmanager
def _shards_written_by(write):
    """
    Write the shards of this thread's `save_pretrained` calls with `write`.

    transformers looks up `modeling_utils.safe_save_file` for every shard. While any thread is inside this context, that is a dispatcher sending each thread's shards to the writer it set and the rest to the original, so saves in other threads are unaffected.

    Args:
        write: Takes a shard's state dict, file name and metadata, as `safe_save_file` does.
    """
    global _shard_writer_users, _original_safe_save_file
    with _shard_writer_lock:
        if _shard_writer_users == 0:
            _original_safe_save_file = modeling_utils.safe_save_file
            modeling_utils.safe_save_file = _write_shard
        _shard_writer_users += 1
    previous = getattr(_shard_writer, "write", None)
    _shard_writer.write = write
    try:
        yield
    finally:
        _shard_writer.write = previous
        with _shard_writer_lock:
            _shard_writer_users -= 1
            if _shard_writer_users == 0:
                modeling_utils.safe_save_file = _original_safe_save_file


def save_reordered(
    model: nn.Module, orders: Dict[int, torch.Tensor], output_dir: str
) -> None:
    """
    Save a model with each gated FFN block's channels permuted, one shard at a time.

    FFN weights are recognised by memory address, which survives the key renaming transformers applies on save. Nothing is written unless a dry run sees every FFN weight reach the shard writer exactly once.

    Args:
        model: Model on the host, its FFN channels in their original order.
        orders: Block index to channel permutation.
        output_dir: Directory to write into.
    """
    blocks = find_gated_mlps(model)
    PACE_LLM_ASSERT(
        len(blocks) == len(orders),
        f"got {len(orders)} permutations for {len(blocks)} FFN blocks",
    )
    weights = {}
    for index, (_, block) in enumerate(blocks):
        for name, columns in (
            ("gate_proj", False),
            ("up_proj", False),
            ("down_proj", True),
        ):
            weights[getattr(block, name).weight.data_ptr()] = (orders[index], columns)

    seen = Counter()

    def record(shard, filename, metadata=None):
        seen.update(t.data_ptr() for t in shard.values() if t.data_ptr() in weights)

    bars = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()
    try:
        with tempfile.TemporaryDirectory() as scratch, _shards_written_by(record):
            model.save_pretrained(scratch, max_shard_size=MAX_SHARD_SIZE)
    finally:
        if bars:
            hf_logging.enable_progress_bar()
    missing = len(weights) - len(seen)
    repeated = sum(count > 1 for count in seen.values())
    PACE_LLM_ASSERT(
        missing == 0 and repeated == 0,
        f"{missing} of {len(weights)} FFN weights would be saved without their channels reordered and {repeated} more than once, so nothing was written",
    )

    def write_permuted(shard, filename, metadata=None):
        permuted = {
            name: (
                permuted_channels(tensor, *weights[tensor.data_ptr()])
                if tensor.data_ptr() in weights
                else tensor
            )
            for name, tensor in shard.items()
        }
        save_file(permuted, filename, metadata=metadata)

    with _shards_written_by(write_permuted):
        model.save_pretrained(output_dir, max_shard_size=MAX_SHARD_SIZE)


class _AllLayersSeen(Exception):
    """Ends a recording forward once the last decoder layer's arguments are known."""


class _LayerStandIn(nn.Module):
    """
    Takes a decoder layer's place to record the arguments the model passes it, passing the hidden states through unchanged.

    The model builds each layer's masks and position embeddings before the layers run, and they differ between attention types, so recording them per layer gives each what a full forward would.
    """

    def __init__(self, layer: nn.Module, record, last: bool):
        super().__init__()
        # Outside the module tree, so moving the model moves none of the real layer's weights.
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "record", record)
        self.last = last

    def __getattr__(self, name):
        # Some model loops read per-layer attributes such as the attention type.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "layer"), name)

    def forward(self, hidden_states, *args, **kwargs):
        self.record(hidden_states, args, kwargs)
        if self.last:
            raise _AllLayersSeen
        return hidden_states


def _same(first, second) -> bool:
    """
    Whether two layer arguments are equal, comparing tensors by value inside tuples, lists and mappings.

    Args:
        first: An argument recorded for one batch.
        second: The same argument recorded for another.

    Returns:
        bool: True if they are equal.
    """
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_same(a, b) for a, b in zip(first, second))
        )
    if isinstance(first, Mapping):
        return (
            isinstance(second, Mapping)
            and first.keys() == second.keys()
            and all(_same(first[key], second[key]) for key in first)
        )
    return first == second


def _resident_bytes(module: nn.Module) -> int:
    """
    Bytes held by a module's parameters and buffers.

    Args:
        module: Any module.

    Returns:
        int: Total size in bytes.
    """
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in itertools.chain(module.parameters(), module.buffers())
    )


@torch.no_grad()
def layerwise_saliency(
    model: nn.Module, batches: List[torch.Tensor], device: torch.device
) -> Dict[int, torch.Tensor]:
    """
    Score every FFN channel, running the calibration batches through one decoder layer at a time on `device`.

    Layers keep their host weights, so a memory-mapped checkpoint is never copied into host memory. The hidden states stay on the device when they fit and on the host otherwise, which is slower.

    Args:
        model: Model in eval mode, on the host.
        batches: Batches of input ids, each [batch, sequence].
        device: Device the layers run on.

    Returns:
        dict: Block index to saliency scores, on the host.
    """
    _, layers = find_decoder_layers(model)
    PACE_LLM_ASSERT(
        layers is not None,
        "calibration runs the text decoder one layer at a time, but found no decoder layer list",
    )
    text_config = resolve_text_config(model.config)
    PACE_LLM_ASSERT(
        not getattr(text_config, "num_kv_shared_layers", 0),
        "this model's later layers reuse keys and values computed by earlier ones, so its layers cannot be calibrated one at a time",
    )
    real = list(layers)

    itemsize = next(model.parameters()).element_size()
    tokens = sum(batch.numel() for batch in batches)
    hidden_bytes = tokens * text_config.hidden_size * itemsize
    hidden_on_device = device.type != "cuda"
    if not hidden_on_device:
        free = torch.cuda.mem_get_info(device)[0]
        layer_bytes = [_resident_bytes(layer) for layer in real]
        others = _resident_bytes(model) - sum(layer_bytes)
        # The FFN's gate, up and product activations, and the attention projections, for the largest batch.
        activations = (
            max(batch.numel() for batch in batches)
            * (3 * text_config.intermediate_size + 6 * text_config.hidden_size)
            * itemsize
        )
        # Beside the hidden states the device holds the embeddings and head while recording, then one layer at a time.
        hidden_on_device = (
            hidden_bytes + max(others, max(layer_bytes) + activations) + GIB <= free
        )
    PACE_LLM_INFO(
        f"calibrating {len(real)} decoder layers one at a time on {device}; hidden states for {len(batches)} batches ({hidden_bytes / GIB:.1f} GiB) kept on the {'device' if hidden_on_device else 'host'}"
    )

    hidden: List[torch.Tensor] = []
    arguments: Dict[int, Dict[int, tuple]] = {}
    checked = set()

    def recorder(index):
        def record(states, args, kwargs):
            size = states.shape[0]
            if index == 0:
                hidden.append(states if hidden_on_device else states.cpu())
            seen = arguments.setdefault(size, {})
            if index not in seen:
                seen[index] = (args, kwargs)
            elif size not in checked:
                # A second batch of the same shape must be given the same arguments, or recording them once per shape is wrong.
                PACE_LLM_ASSERT(
                    _same((args, kwargs), seen[index]),
                    f"decoder layer {index} receives arguments that change from batch to batch, so its layers cannot be calibrated one at a time",
                )
                if index == len(real) - 1:
                    checked.add(size)

        return record

    for index, layer in enumerate(real):
        layers[index] = _LayerStandIn(layer, recorder(index), index == len(real) - 1)
    try:
        with on_device(model, device):
            for batch in tqdm(batches, desc="Preparing layer inputs", unit="batch"):
                try:
                    model(input_ids=batch.to(device), use_cache=False)
                except _AllLayersSeen:
                    pass
    finally:
        for index, layer in enumerate(real):
            layers[index] = layer

    scores = {}
    with SaliencyAccumulator(model) as accumulator:
        PACE_LLM_ASSERT(
            len(accumulator.blocks) == len(real),
            f"found {len(accumulator.blocks)} gated FFN blocks in {len(real)} decoder layers; calibration needs one per layer",
        )
        for index, layer in enumerate(
            tqdm(real, desc="Calibrating decoder layers", unit="layer")
        ):
            PACE_LLM_ASSERT(
                any(
                    module is accumulator.blocks[index][1] for module in layer.modules()
                ),
                f"{accumulator.blocks[index][0]} is not in decoder layer {index}",
            )
            with on_device(layer, device):
                for batch, states in enumerate(hidden):
                    args, kwargs = arguments[states.shape[0]][index]
                    output = layer(states.to(device), *args, **kwargs)
                    output = output[0] if isinstance(output, tuple) else output
                    hidden[batch] = output if hidden_on_device else states.copy_(output)
                scores[index] = accumulator.score(index)
    return scores


def calibrate(config: StepCalibrationConfig) -> str:
    """
    Reorder a checkpoint's FFN channels by saliency and write it back out.

    Takes a HuggingFace checkpoint and writes one with the same architecture and weights, its FFN channels in descending saliency order.

    Args:
        config: Calibration settings.

    Returns:
        str: The output directory.
    """
    require_gpu("STeP calibration")
    device = torch.device(config.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    PACE_LLM_INFO(f"loading {config.model_name_or_path}")

    # The tokenizer saved with the checkpoint keeps the corrected pre-tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, fix_mistral_regex=True
    )
    source = AutoConfig.from_pretrained(config.model_name_or_path)
    dtype = getattr(resolve_text_config(source), "dtype", None) or torch.bfloat16
    model = load_model(config.model_name_or_path, dtype)
    model.eval()

    blocks = find_gated_mlps(model)
    PACE_LLM_ASSERT(
        len(blocks) > 0,
        "found no dense gate/up/down FFN blocks to calibrate. Mixture-of-experts and non-gated FFNs are not supported by STeP calibration.",
    )
    for name, block in blocks:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            PACE_LLM_ASSERT(
                getattr(block, projection).bias is None,
                f"{name}.{projection} has a bias, which STeP cannot split",
            )
    channels = blocks[0][1].up_proj.out_features
    PACE_LLM_INFO(
        f"{len(blocks)} FFN blocks, {channels} channels each, calibrating on {device}"
    )

    dataset = CalibrationDataset(
        tokenizer, config.datasets, config.num_samples, config.sequence_length
    )
    batches = [
        dataset.get_batch(config.batch_size, start)["input_ids"]
        for start in range(0, dataset.size(), config.batch_size)
    ]
    PACE_LLM_INFO(f"{len(batches)} calibration batches")
    scores = layerwise_saliency(model, batches, device)
    torch.cuda.empty_cache()

    PACE_LLM_INFO(f"writing reordered checkpoint to {config.output_dir}")
    save_reordered(
        model,
        {index: channel_order(scores[index]) for index in range(len(blocks))},
        config.output_dir,
    )
    tokenizer.save_pretrained(config.output_dir)

    return config.output_dir
