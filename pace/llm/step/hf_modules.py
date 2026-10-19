# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import List, Tuple

import torch.nn as nn

from pace.utils.logging import PACE_LLM_ASSERT

# Where the text decoder's layer list lives, across the wrappers HF uses for plain causal LMs and for multimodal models.
_DECODER_LAYER_PATHS = (
    "model.language_model.layers",
    "language_model.model.layers",
    "model.text_model.layers",
    "model.decoder.layers",
    "model.layers",
    "layers",
)


def find_decoder_layers(model: nn.Module):
    """
    Locate the text decoder's layer list in a HuggingFace model.

    Multimodal checkpoints carry a vision tower whose MLPs are also gated. Those are excluded, since STeP splits the language model only.

    Args:
        model: A causal LM, possibly multimodal.

    Returns:
        tuple: (dotted path, ModuleList), or (None, None) if no known path matches.
    """
    for path in _DECODER_LAYER_PATHS:
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return path, module
    return None, None


def find_gated_mlps(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Locate the dense gated FFN blocks whose channels can be reordered and split.

    A gated FFN computes `down(act(gate(x)) * up(x))` through `gate_proj`, `up_proj` and `down_proj`, whichever activation it uses. Restricted to the text decoder when one can be identified, so a vision tower's gated MLPs are not swept up.

    Args:
        model: A causal LM.

    Returns:
        list: (qualified name, module) in model definition order.
    """
    path, layers = find_decoder_layers(model)
    root = layers if layers is not None else model
    prefix = f"{path}." if path else ""

    found = []
    for name, module in root.named_modules():
        PACE_LLM_ASSERT(
            not isinstance(getattr(module, "experts", None), nn.Module),
            f"{prefix}{name} routes to experts; STeP does not split mixture-of-experts FFNs",
        )
        projections = [
            getattr(module, attr, None)
            for attr in ("gate_proj", "up_proj", "down_proj")
        ]
        if all(isinstance(p, nn.Linear) for p in projections):
            found.append((f"{prefix}{name}" if name else prefix.rstrip("."), module))
    return found
