# Python Operators

All the Python operators are implemented under the [`pace/ops/`](../pace/ops/) directory. All the Python operators follow the same structure. There are two parts to these operators, a Frontend (the operator itself) and a Backend (the actual computation).

## Design

The Python operator framework within AMD PACE is designed to provide a modular, extensible, and flexible way to define and execute operations. The key goals of this design are:

1.  Separation of Concerns: Clearly separate the operator's public interface (Frontend) from its computational logic (Backend). This allows for multiple backend implementations for a single operator.
2.  Extensibility: Easily add new operators or new backend implementations for existing operators without modifying core framework code.
3.  Flexibility: Allow users to select specific backends (e.g., native PyTorch, JIT, or more) based on availability, performance needs, or hardware capabilities.
4.  Type Safety: Utilize Enums for operator types, backend types, and data types to improve code clarity and reduce errors.
5.  Dynamic Registration: Backends register themselves with a central registry, making them discoverable by operators at runtime.

## Core Components

The framework is built around several key components:

1.  **`OperatorBase` ([`pace.ops.base.OperatorBase`](../pace/ops/base.py))**:
    *   This is an abstract base class inherited from `torch.nn.Module` that all Python operators must inherit from.
    *   It defines the common interface for all operators, including:
        *   An `__init__` method that takes `backend_impl` (specifying the desired [`BackendType`](../pace/ops/enum.py)) and `dtype` ([`DataType`](../pace/ops/enum.py)).
        *   A `forward` method, which is the entry point for executing the operator.
        *   An `operator_type` abstract property that must be implemented by subclasses to return their specific [`OperatorType`](../pace/ops/enum.py).
        *   An `extra_repr` method for a string representation of the operator's configuration.
    *   Crucially, `OperatorBase` handles the dynamic selection and instantiation of the appropriate backend based on the provided `backend_impl`, `dtype`, and the operator's `operator_type`. It uses the [`BackendRegistry`](../pace/ops/registry.py) for this.

2.  **`BackendBase` ([`pace.ops.base.BackendBase`](../pace/ops/base.py))**:
    *   An abstract base class for all backend implementations.
    *   It defines the interface that concrete backends (e.g., a native PyTorch backend for Linear, a JIT backend for Linear) must implement.
    *   Key methods include:
        *   `execute`: The core method where the actual computation of the operator is performed.
        *   `create_extra_params` (optional): Allows backends to prepare or transform specific parameters.
        *   `preprocess` (optional): Allows for preprocessing weights specific to the backend.

3.  **Enums ([`pace.ops.enum`](../pace/ops/enum.py))**:
    *   `OperatorType`: Defines all supported operator types (e.g., `LINEAR`, `MHA`, `RMSNORM`). Each concrete operator class corresponds to one of these types.
    *   `FusedOperatorType`: Defines fused operator types (e.g., `FUSEDLINEARRELU`, `FUSEDLINEARSILU`), which are optimized combinations of multiple operations.
    *   `BackendType`: Defines the different kinds of backend implementations available (e.g., `NATIVE`, `JIT`).
    *   `DataType`: Defines the supported data types for operations (e.g., `FLOAT32`, `BFLOAT16`).
    *   `FALLBACK_BACKEND`: A list specifying the order of backend types to try if the initially requested backend is not available for a given operator and data type. This provides a graceful degradation mechanism.

4.  **`BackendRegistry` ([`pace.ops.registry.BackendRegistry`](../pace/ops/registry.py))**:
    *   A global singleton (`backend_registry`) responsible for storing and providing backend implementations.
    *   Backends are registered using the `@backend_registry.register(operator_type, backend_type, dtype)` decorator on the backend class. This maps a unique tuple of (`OperatorType`, `BackendType`, `DataType`) to a specific backend implementation class.
    *   The `OperatorBase` uses the registry's `get` method to find and instantiate the appropriate backend during its initialization.

## Workflow: Operator Instantiation and Execution

1.  **Instantiation**:
    *   A user or higher-level module instantiates an operator, e.g., `my_linear = pace.ops.Linear(in_features, out_features, backend_impl=BackendType.NATIVE, dtype=DataType.FLOAT32)`.
    *   The `OperatorBase.__init__` method is called.
    *   It retrieves its `operator_type` (e.g., `OperatorType.LINEAR`).
    *   It calls `backend_registry.get(OperatorType.LINEAR, BackendType.NATIVE, DataType.FLOAT32)`.
    *   The registry looks up this combination. If found, it instantiates the corresponding backend class (e.g., `NativeLinear`).
    *   If not found, it iterates through `FALLBACK_BACKEND` (e.g., trying `BackendType.JIT` next if `NATIVE` was the primary and failed).
    *   Logging messages ([`PACE_LLM_DEBUG`](../pace/utils/logging.py)) indicate the backend selection process. If no backend is found after trying fallbacks, an assertion ([`PACE_LLM_ASSERT`](../pace/utils/logging.py)) is raised.
    *   The instantiated backend is stored in the operator instance (e.g., `self.backend`).

2.  **Execution**:
    *   When `my_linear.forward(input_tensor)` is called:
    *   The `Linear.forward` method (or `OperatorBase.forward` if not overridden specifically) typically calls `self.backend.execute(input_tensor, self.weight, self.bias, **self.extra_params)`.
    *   The `execute` method of the chosen backend (e.g., `NativeLinear.execute`) performs the actual computation and returns the result.

## Fused Operators

Fused operators combine multiple primitive operations into a single optimized unit. They inherit from `OperatorBase` just like regular ops, but use `FusedOperatorType` (from [`pace/ops/enum.py`](../pace/ops/enum.py)) instead of `OperatorType`.

The key difference in registry behavior is that **fused ops degrade gracefully**: if no fused backend is registered for the requested type/dtype, the registry returns `None` (rather than raising an error). The fused operator then falls back to composing individual ops in its default `forward` method.

For example, `MergedMLP` (`pace/ops/mlp.py`) is a fused op that wraps gate/up projections, activation, and a down projection. When a TPP backend is available, the entire MLP runs as a single fused kernel. When it isn't, the individual `Linear` and activation ops execute independently — correctness is preserved, only performance differs.

When the fused backend is resolved, sub-operators are created with `backend_impl=None` — the fused op manages execution for its children. If no fused backend is found (`self.backend is None`), the sub-operators receive the original `backend_impl` and execute independently as fallback.

See [`pace/ops/enum.py`](../pace/ops/enum.py) for the full list of `FusedOperatorType` values.

## Backend Fallback

When the requested `BackendType` is not registered for a given `(OperatorType, DataType)`, the `BackendRegistry` walks the `FALLBACK_BACKEND` list (defined in [`pace/ops/enum.py`](../pace/ops/enum.py)) in order — `NATIVE` first, then `JIT`, `TPP`, `IMBPS`, `AOCLDLP` — until it finds a registered implementation. If none is found, an assertion is raised.

This fallback only applies to standard `OperatorType` ops. `FusedOperatorType` ops do not fall back through this list — they simply get `None` and use their composed implementation.

## Adding New Operators and Backends

*   **New Operator**:
    1.  Define a new Enum value in `OperatorType` ([`pace.ops.enum`](../pace/ops/enum.py)).
    2.  Create a new Python file in `pace/ops/` (e.g., `my_new_op.py`).
    3.  Define a class inheriting from `OperatorBase`, implementing `operator_type` and other necessary methods.
    4.  Add at least one backend implementation for this new operator type (see below).
    5.  Export the new operator from [`pace/ops/__init__.py`](../pace/ops/__init__.py).

*   **New Backend for an Existing Operator**:
    1.  Choose a `BackendType` (or define a new one in [`pace.ops.enum`](../pace/ops/enum.py)).
    2.  In the appropriate backend module (e.g., [`pace/ops/backends/native.py`](../pace/ops/backends/native.py) or a new backend file), define a class inheriting from `BackendBase`.
    3.  Implement the `execute` method and any other required backend logic.
    4.  Decorate the class with `@backend_registry.register(operator_type, backend_type, dtype)` for all supported data types.
    5.  Ensure the backend module is imported in [`pace/ops/backends/__init__.py`](../pace/ops/backends/__init__.py).
