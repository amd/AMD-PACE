# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_utils import TestCase

from pace.utils.device import (
    describe_device,
    gpu_available,
    gpu_backend,
    gpu_count,
    gpu_memory_bytes,
    gpu_name,
    require_gpu,
)

GIB = 2**30
MI355X = ("AMD Instinct MI355X", 288 * GIB)
H100 = ("NVIDIA H100 80GB HBM3", 80 * GIB)


@contextmanager
def build(hip=None, cuda=None, devices=()):
    """
    Present a chosen PyTorch build and set of visible GPUs.

    Args:
        hip: Value for `torch.version.hip`.
        cuda: Value for `torch.version.cuda`.
        devices: (name, memory bytes) per visible device.
    """
    properties = [SimpleNamespace(total_memory=memory) for _, memory in devices]
    patches = (
        mock.patch.object(torch.version, "hip", hip),
        mock.patch.object(torch.version, "cuda", cuda),
        mock.patch.object(torch.cuda, "is_available", lambda: bool(devices)),
        mock.patch.object(torch.cuda, "device_count", lambda: len(devices)),
        mock.patch.object(
            torch.cuda, "get_device_name", lambda index=0: devices[index][0]
        ),
        mock.patch.object(
            torch.cuda, "get_device_properties", lambda index=0: properties[index]
        ),
    )
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield


class TestBackendDetection(TestCase):

    def test_a_rocm_build_is_named(self):
        with build(hip="7.2", devices=[MI355X]):
            self.assertEqual(gpu_backend(), "ROCm")

    def test_a_cuda_build_is_named(self):
        with build(cuda="12.8", devices=[H100]):
            self.assertEqual(gpu_backend(), "CUDA")

    def test_a_cpu_build_has_no_backend(self):
        with build():
            self.assertIsNone(gpu_backend())


class TestDeviceVisibility(TestCase):

    def test_visible_devices_are_reported(self):
        with build(hip="7.2", devices=[MI355X, MI355X]):
            self.assertTrue(gpu_available())
            self.assertEqual(gpu_count(), 2)
            self.assertEqual(gpu_name(), MI355X[0])
            self.assertEqual(gpu_memory_bytes(), MI355X[1])

    def test_a_build_with_no_visible_device_reports_nothing(self):
        with build(hip="7.2"):
            self.assertFalse(gpu_available())
            self.assertEqual(gpu_count(), 0)
            self.assertIsNone(gpu_name())
            self.assertIsNone(gpu_memory_bytes())


class TestRequireGpu(TestCase):

    def test_a_usable_gpu_is_accepted(self):
        with build(hip="7.2", devices=[MI355X]):
            require_gpu("STeP")

    def test_a_cpu_build_is_told_how_to_install(self):
        with build():
            with self.assertRaises(RuntimeError) as caught:
                require_gpu("STeP")
        message = str(caught.exception)
        self.assertIn("CPU build", message)
        self.assertIn("amd-pace[gpu]", message)

    def test_rocm_with_no_device_names_the_hip_mask(self):
        with build(hip="7.2"):
            with self.assertRaises(RuntimeError) as caught:
                require_gpu("STeP")
        self.assertIn("HIP_VISIBLE_DEVICES", str(caught.exception))

    def test_cuda_with_no_device_names_the_cuda_mask(self):
        with build(cuda="12.8"):
            with self.assertRaises(RuntimeError) as caught:
                require_gpu("STeP")
        self.assertIn("CUDA_VISIBLE_DEVICES", str(caught.exception))


class TestDescribeDevice(TestCase):

    def test_a_cpu_build_is_described(self):
        with build():
            self.assertIn("CPU build", describe_device())

    def test_a_build_with_no_visible_device_is_described(self):
        with build(cuda="12.8"):
            text = describe_device()
        self.assertIn("CUDA build", text)
        self.assertIn("no device visible", text)

    def test_matching_devices_are_summarised_together(self):
        with build(hip="7.2", devices=[MI355X] * 8):
            self.assertIn("8 x AMD Instinct MI355X (288 GiB each)", describe_device())

    def test_a_single_device_does_not_claim_each(self):
        with build(hip="7.2", devices=[MI355X]):
            text = describe_device()
        self.assertIn("AMD Instinct MI355X (288 GiB)", text)
        self.assertNotIn("each", text)

    def test_unlike_devices_are_reported_apart(self):
        with build(cuda="12.8", devices=[H100, H100, MI355X]):
            text = describe_device()
        self.assertIn("2 x NVIDIA H100 80GB HBM3 (80 GiB each)", text)
        self.assertIn("AMD Instinct MI355X (288 GiB)", text)
