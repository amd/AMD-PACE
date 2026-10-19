# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m unittest -v attention/test_recurrent_state.py

from torch.testing._internal.common_utils import TestCase
import torch
from transformers import PretrainedConfig

from pace.llm.attention import KVCacheType, KVCacheManager
from pace.llm.attention.recurrent_state import (
    RecurrentStateStore,
    MergedRecurrentState,
    num_recurrent_layers,
)


class _Cfg(PretrainedConfig):
    def __init__(self, num_hidden_layers=4, layer_types=None):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        if layer_types is not None:
            self.layer_types = layer_types


class TestNumRecurrentLayers(TestCase):
    def test_counts_linear_attention(self):
        cfg = _Cfg(
            num_hidden_layers=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        )
        self.assertEqual(num_recurrent_layers(cfg), 3)

    def test_no_layer_types_is_zero(self):
        self.assertEqual(num_recurrent_layers(_Cfg()), 0)


class TestRecurrentStateStore(TestCase):
    def test_starts_empty_and_roundtrips(self):
        store = RecurrentStateStore(2)
        self.assertEqual(store.num_layers, 2)
        self.assertEqual(store.gather(0), (None, None))
        conv = torch.randn(1, 8, 3)
        rec = torch.randn(1, 4, 16, 16)
        store.commit(0, conv, rec)
        gconv, grec = store.gather(0)
        self.assertTrue(torch.equal(gconv, conv))
        self.assertTrue(torch.equal(grec, rec))
        # Other slot untouched.
        self.assertEqual(store.gather(1), (None, None))


class TestMergedRecurrentState(TestCase):
    def _mk_store(self, val, layers=1):
        s = RecurrentStateStore(layers)
        s.commit(0, torch.full((1, 4), float(val)), torch.full((1, 2, 2), float(val)))
        return s

    def test_gather_stacks_in_list_order(self):
        stores = [self._mk_store(v) for v in (10, 20, 30)]
        merged = MergedRecurrentState(stores)
        conv, rec = merged.gather(0)
        self.assertEqual(conv.shape, torch.Size([3, 4]))
        self.assertEqual(rec.shape, torch.Size([3, 2, 2]))
        # Row i must equal store i (order preserved == batch alignment).
        self.assertEqual(conv[0, 0].item(), 10.0)
        self.assertEqual(conv[1, 0].item(), 20.0)
        self.assertEqual(conv[2, 0].item(), 30.0)

    def test_commit_scatters_rows_back(self):
        stores = [self._mk_store(v) for v in (1, 2)]
        merged = MergedRecurrentState(stores)
        new_conv = torch.stack([torch.full((4,), 7.0), torch.full((4,), 9.0)], dim=0)
        new_rec = torch.stack([torch.full((2, 2), 7.0), torch.full((2, 2), 9.0)], dim=0)
        merged.commit(0, new_conv, new_rec)
        self.assertEqual(stores[0].gather(0)[0][0, 0].item(), 7.0)
        self.assertEqual(stores[1].gather(0)[0][0, 0].item(), 9.0)
        # Shapes scattered back to per-request [1, ...].
        self.assertEqual(stores[0].gather(0)[0].shape, torch.Size([1, 4]))

    def test_commit_clones_rows_off_batched_storage(self):
        # Decode conv mutates conv_state in-place. Split views would share
        # one storage, so an unbatched decode of request 0 could clobber
        # request 1. Scatter must give each store its own copy.
        stores = [self._mk_store(v) for v in (1, 2)]
        merged = MergedRecurrentState(stores)
        new_conv = torch.stack([torch.full((4,), 7.0), torch.full((4,), 9.0)], dim=0)
        new_rec = torch.stack([torch.full((2, 2), 7.0), torch.full((2, 2), 9.0)], dim=0)
        merged.commit(0, new_conv, new_rec)
        conv0, rec0 = stores[0].gather(0)
        conv1, rec1 = stores[1].gather(0)
        self.assertNotEqual(
            conv0.untyped_storage().data_ptr(), conv1.untyped_storage().data_ptr()
        )
        self.assertNotEqual(
            rec0.untyped_storage().data_ptr(), rec1.untyped_storage().data_ptr()
        )
        conv0.fill_(3.0)
        new_conv.fill_(0.0)
        self.assertEqual(conv1[0, 0].item(), 9.0)
        self.assertEqual(stores[0].gather(0)[0][0, 0].item(), 3.0)

    def test_mixed_none_and_populated_zero_fills_missing_rows(self):
        populated = self._mk_store(5)
        fresh = RecurrentStateStore(1)
        merged = MergedRecurrentState([fresh, populated])
        conv, rec = merged.gather(0)
        self.assertEqual(conv.shape, torch.Size([2, 4]))
        self.assertEqual(rec.shape, torch.Size([2, 2, 2]))
        self.assertEqual(conv[0].abs().sum().item(), 0.0)
        self.assertEqual(conv[1, 0].item(), 5.0)
        self.assertEqual(rec[1, 0, 0].item(), 5.0)

    def test_mixed_commit_populates_fresh_store(self):
        populated = self._mk_store(5)
        fresh = RecurrentStateStore(1)
        merged = MergedRecurrentState([fresh, populated])
        new_conv = torch.stack([torch.full((4,), 1.0), torch.full((4,), 2.0)], dim=0)
        new_rec = torch.stack([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)], dim=0)
        merged.commit(0, new_conv, new_rec)
        self.assertEqual(fresh.gather(0)[0][0, 0].item(), 1.0)
        self.assertEqual(populated.gather(0)[0][0, 0].item(), 2.0)

    def test_all_none_gathers_none(self):
        stores = [RecurrentStateStore(1), RecurrentStateStore(1)]
        merged = MergedRecurrentState(stores)
        self.assertEqual(merged.gather(0), (None, None))
        # Commit of None is a no-op (fresh batch prefill returns state later).
        merged.commit(0, None, None)
        self.assertEqual(stores[0].gather(0), (None, None))

    def test_variable_membership_between_steps(self):
        # Membership changes across steps (a request leaves); the view is
        # rebuilt from whatever stores are passed, preserving alignment.
        a, b, c = (self._mk_store(v) for v in (1, 2, 3))
        step1 = MergedRecurrentState([a, b, c])
        conv1, _ = step1.gather(0)
        self.assertEqual(conv1.shape[0], 3)
        step2 = MergedRecurrentState([a, c])  # b completed
        conv2, _ = step2.gather(0)
        self.assertEqual(conv2.shape[0], 2)
        self.assertEqual(conv2[0, 0].item(), 1.0)
        self.assertEqual(conv2[1, 0].item(), 3.0)


class TestKVCacheManagerCarriesStore(TestCase):
    def test_manager_has_recurrent_store(self):
        cfg = _Cfg(
            num_hidden_layers=3,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
        )
        mgr = KVCacheManager(cfg, 128, KVCacheType.DYNAMIC)
        self.assertTrue(hasattr(mgr, "recurrent_state"))
        self.assertEqual(mgr.recurrent_state.num_layers, 2)

    def test_full_attention_only_has_empty_store(self):
        mgr = KVCacheManager(_Cfg(num_hidden_layers=2), 128, KVCacheType.DYNAMIC)
        self.assertEqual(mgr.recurrent_state.num_layers, 0)
