# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

from pace.llm.step.hf_modules import find_decoder_layers, find_gated_mlps
from pace.utils.logging import suppress_logging_cls


class GatedMLP(nn.Module):
    def __init__(self, d_model=16, channels=8):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, channels, bias=False)
        self.up_proj = nn.Linear(d_model, channels, bias=False)
        self.down_proj = nn.Linear(channels, d_model, bias=False)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = GatedMLP()


class Decoder(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(count)])


class PlainCausalLM(nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.model = Decoder(count)


class VisionTower(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(count)])


class MultimodalLM(nn.Module):
    """Mirrors the HF layout where a vision tower sits beside the text decoder."""

    def __init__(self, text_layers=4, vision_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = Decoder(text_layers)
        self.model.vision_tower = VisionTower(vision_layers)


@suppress_logging_cls()
class TestDecoderDiscovery(TestCase):

    def test_plain_model_resolves_model_layers(self):
        path, layers = find_decoder_layers(PlainCausalLM(count=3))
        self.assertEqual(path, "model.layers")
        self.assertEqual(len(layers), 3)

    def test_multimodal_model_resolves_the_language_model(self):
        path, layers = find_decoder_layers(MultimodalLM(text_layers=4, vision_layers=2))
        self.assertEqual(path, "model.language_model.layers")
        self.assertEqual(len(layers), 4)

    def test_unknown_layouts_report_no_decoder(self):
        path, layers = find_decoder_layers(nn.Linear(4, 4))
        self.assertIsNone(path)
        self.assertIsNone(layers)


@suppress_logging_cls()
class TestFindGatedMLPs(TestCase):

    def test_vision_tower_mlps_are_not_collected(self):
        found = find_gated_mlps(MultimodalLM(text_layers=4, vision_layers=2))
        self.assertEqual(len(found), 4)
        for name, _ in found:
            self.assertIn("language_model", name)
            self.assertNotIn("vision", name)

    def test_names_are_fully_qualified(self):
        found = find_gated_mlps(PlainCausalLM(count=2))
        self.assertEqual(
            [name for name, _ in found], ["model.layers.0.mlp", "model.layers.1.mlp"]
        )

    def test_a_model_without_a_known_layer_path_is_scanned_whole(self):
        loose = nn.Module()
        loose.block = GatedMLP()
        found = find_gated_mlps(loose)
        self.assertEqual(len(found), 1)

    def test_a_model_without_gated_blocks_yields_nothing(self):
        plain = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.assertEqual(find_gated_mlps(plain), [])

    def test_mixture_of_experts_is_refused(self):
        model = PlainCausalLM(count=2)
        model.model.layers[0].mlp.experts = nn.ModuleList([GatedMLP(), GatedMLP()])
        with self.assertRaises(AssertionError):
            find_gated_mlps(model)
