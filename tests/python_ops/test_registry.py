# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase
from unittest.mock import patch

# Assuming these are imported in the actual test file:
from pace.ops.registry import BackendRegistry
from pace.ops.enum import (
    OperatorType,
    FusedOperatorType,
    BackendType,
    DataType,
    FALLBACK_BACKEND,
)


# Dummy classes for testing implementations
class DummyOpImpl:
    pass


class NativeLinear(DummyOpImpl):
    pass


class JITLinear(DummyOpImpl):
    pass


class TPPLinear(DummyOpImpl):
    pass


class TestBackendRegistry(TestCase):
    def setUp(self):
        super().setUp()
        # Use a fresh registry for each test
        self.registry = BackendRegistry()
        # Store original FALLBACK_BACKEND and restore in tearDown if tests modify it
        self.original_fallback_backend = list(FALLBACK_BACKEND)

    def tearDown(self):
        # Restore original FALLBACK_BACKEND after each test
        FALLBACK_BACKEND[:] = self.original_fallback_backend
        super().tearDown()

    @given(
        op_type=st.sampled_from(list(OperatorType)),
        backend=st.sampled_from(list(BackendType)),
        dtype=st.sampled_from(list(DataType)),
    )
    def test_register_and_get_single_operator(self, op_type, backend, dtype):
        class SpecificImpl(DummyOpImpl):
            pass

        self.registry.register(op_type, backend, [dtype])(SpecificImpl)

        retrieved_impl = self.registry.get(op_type, backend, dtype)
        self.assertIs(retrieved_impl, SpecificImpl)

    @given(
        op_type=st.sampled_from(list(OperatorType)),
        backend=st.sampled_from(list(BackendType)),
        dtypes=st.lists(
            st.sampled_from(list(DataType)),
            min_size=1,
            max_size=len(DataType),
            unique=True,
        ),
    )
    def test_register_multiple_dtypes(self, op_type, backend, dtypes):
        class SpecificImpl(DummyOpImpl):
            pass

        self.registry.register(op_type, backend, dtypes)(SpecificImpl)

        for dt in dtypes:
            retrieved_impl = self.registry.get(op_type, backend, dt)
            self.assertIs(retrieved_impl, SpecificImpl)

    def test_get_fallback_backend_logic(self):
        op = OperatorType.LINEAR
        dtype = DataType.BFLOAT16

        # Register JIT implementation
        self.registry.register(op, BackendType.JIT, [dtype])(NativeLinear)

        # Modify FALLBACK_BACKEND for predictable test
        # Original FALLBACK_BACKEND is [BackendType.IMBPS, BackendType.TPP]
        FALLBACK_BACKEND[:] = [BackendType.JIT, BackendType.TPP]

        # Case 1: Request IMBPS, IMBPS not registered, JIT is. Fallback to JIT.
        # get(op, IMBPS, dtype) checks (IMBPS, dtype) -> not found.
        # Fallback checks FALLBACK_BACKEND[0] = IMBPS -> (IMBPS, dtype) -> not found in op_backends.
        # Fallback checks FALLBACK_BACKEND[1] = JIT -> (JIT, dtype) -> found.
        retrieved_impl = self.registry.get(op, BackendType.IMBPS, dtype)
        self.assertIs(retrieved_impl, NativeLinear)

        # Case 2: Register TPP impl. Request TPP. Should get TPP impl (direct match).
        self.registry.register(op, BackendType.TPP, [dtype])(TPPLinear)
        retrieved_impl_rocm = self.registry.get(op, BackendType.TPP, dtype)
        self.assertIs(retrieved_impl_rocm, TPPLinear)

    def test_get_fallback_backend_order_preference(self):
        op = OperatorType.LINEAR
        dtype = DataType.BFLOAT16

        self.registry.register(op, BackendType.NATIVE, [dtype])(NativeLinear)
        self.registry.register(op, BackendType.JIT, [dtype])(JITLinear)

        # Scenario 1: TPP preferred fallback, then CPU
        FALLBACK_BACKEND[:] = [BackendType.JIT, BackendType.NATIVE]
        # Request a non-existent backend (e.g. TPP, assuming it's not registered for this op)
        # It should pick JIT from fallback list first.
        retrieved = self.registry.get(op, BackendType.TPP, dtype)
        self.assertIs(retrieved, JITLinear)

        # Scenario 2: JIT preferred fallback, then NATIVE
        # Need a fresh registry state for op_backends if we don't re-register
        self.registry = BackendRegistry()
        self.registry.register(op, BackendType.JIT, [dtype])(JITLinear)
        self.registry.register(op, BackendType.NATIVE, [dtype])(NativeLinear)
        FALLBACK_BACKEND[:] = [BackendType.NATIVE, BackendType.JIT]

        retrieved = self.registry.get(op, BackendType.TPP, dtype)
        self.assertIs(retrieved, NativeLinear)

    @given(
        op_type=st.sampled_from(list(OperatorType)),
        backend=st.sampled_from(list(BackendType)),
        dtype=st.sampled_from(list(DataType)),
    )
    def test_get_operator_completely_unregistered(self, op_type, backend, dtype):
        # self.registry is fresh, so op_type is not registered at all.
        retrieved_impl = self.registry.get(op_type, backend, dtype)
        self.assertIsNone(retrieved_impl)

    def test_get_fused_operator_behavior(self):
        fused_op = FusedOperatorType.FUSEDLINEARGELU
        backend = BackendType.TPP
        dtype = DataType.BFLOAT16

        class FusedImpl(DummyOpImpl):
            pass

        self.registry.register(fused_op, backend, [dtype])(FusedImpl)

        # 1. Exact match
        retrieved = self.registry.get(fused_op, backend, dtype)
        self.assertIs(retrieved, FusedImpl)

        # 2. Different backend: Fused ops should return None if no exact match, no fallback.
        other_backend = BackendType.JIT
        retrieved_other_backend = self.registry.get(fused_op, other_backend, dtype)
        self.assertIsNone(retrieved_other_backend)

        # 3. Different dtype: Same as above.
        other_dtype = DataType.FLOAT32
        retrieved_other_dtype = self.registry.get(fused_op, backend, other_dtype)
        self.assertIsNone(retrieved_other_dtype)

    @patch("pace.ops.registry.PACE_ASSERT")
    def test_get_no_backend_found_assertion_triggered(self, mock_pace_assert):
        op = OperatorType.LINEAR
        requested_backend = BackendType.JIT
        requested_dtype = DataType.FLOAT32

        # Register something else for the op, to ensure op is in registry
        self.registry.register(op, BackendType.JIT, [DataType.BFLOAT16])(JITLinear)

        # Set fallbacks to something that won't match
        FALLBACK_BACKEND[:] = [BackendType.TPP]

        result = self.registry.get(op, requested_backend, requested_dtype)

        mock_pace_assert.assert_called_once_with(
            False,
            f"No backend found for {op} with {requested_backend}, {requested_dtype}",
        )
        self.assertIsNone(result)  # get() returns None after assert

    def test_repr_method(self):
        self.registry.register(
            OperatorType.LINEAR, BackendType.NATIVE, [DataType.BFLOAT16]
        )(NativeLinear)
        representation = repr(self.registry)
        self.assertIsInstance(representation, str)
        self.assertIn("OperatorType.LINEAR", representation)
        self.assertIn("BackendType.NATIVE", representation)
        self.assertIn("DataType.BFLOAT16", representation)
        self.assertIn(NativeLinear.__name__, representation)

    @patch("pace.ops.registry.PACE_DEBUG")
    def test_get_operator_not_in_registry_debug_message(self, mock_pace_debug):
        op = OperatorType.SOFTMAX
        backend = BackendType.JIT
        dtype = DataType.BFLOAT16

        retrieved = self.registry.get(op, backend, dtype)
        self.assertIsNone(retrieved)
        mock_pace_debug.assert_any_call(
            f"Operator {op} not registered, this operator might be a new one, or one without a backend"
        )

    @patch("pace.ops.registry.PACE_DEBUG")
    def test_get_fused_op_no_direct_match_debug_message(self, mock_pace_debug):
        fused_op = FusedOperatorType.FUSEDLINEARSILU
        registered_backend = BackendType.TPP
        registered_dtype = DataType.BFLOAT16
        requested_backend = BackendType.JIT

        class MyFusedImpl(DummyOpImpl):
            pass

        self.registry.register(fused_op, registered_backend, [registered_dtype])(
            MyFusedImpl
        )

        retrieved = self.registry.get(fused_op, requested_backend, registered_dtype)
        self.assertIsNone(retrieved)
        mock_pace_debug.assert_any_call(
            f"Fused operator {fused_op} with backend {requested_backend} and dtype {registered_dtype} not found in registry."
            "It will be handled by the default forward method."
        )
