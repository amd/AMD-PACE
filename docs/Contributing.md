# How to contribute to AMD PACE

This document provides guidelines and instructions for contributing to the AMD PACE library. It covers various aspects of development, including adding external libraries, creating operators, creating core functions, logging, and basic developer checks.

## Table of Contents
1. [Adding an external library](#adding-an-external-library)
2. [Creating a Operator](#creating-a-operator)
3. [Creating a Python Operator](#creating-a-python-operator)
4. [Creating a core function](#creating-a-core-function)
5. [Logging in AMD PACE](#logging-in-amd-pace)
6. [Code Style](#code-style)
7. [Testing](#testing)
8. [Basic Developer Checks](#basic-developer-checks)
9. [Submitting a PR](#submitting-a-pr)

## Adding an external library
The implementation for building and linking with external libraries are under `cmake/Build{Library}.cmake`. All the libraries are downloaded and build using the `ExternalProject_Add` command in CMake. For more information on `ExternalProject_Add`, please refer to the [CMake documentation](https://cmake.org/cmake/help/latest/module/ExternalProject.html). The external libraries are built as static objects and linked with the AMD PACE library.

Once the library is built, set `{Library}_INCLUDE_DIR` and `{Library}_STATIC_LIB` in the cmake file itself. This is then used to link the library with the AMD PACE library in `csrc/CMakeLists.txt`. If there is any specific order in which the libraries need to be linked, please follow the order in the `csrc/CMakeLists.txt` file.

## Creating a Operator
All the operators are implemented under `csrc/ops/` directory. All the operators follow the same structure and naming conventions. To create a new operator in AMD PACE, follow the steps below:
1. Create a new file under `csrc/ops/` with the name of the operator. For example, create files `csrc/ops/opname.h` `csrc/ops/opname.cpp`, `csrc/ops/kernels/opname_kernel.h` and `csrc/ops/kernels/opname_kernel.cpp`.
The directory structure should look like:
    ```
    csrc
    ├── ops
    ├── kernels
    │   ├── [Optional] opname_kernel_avx512.cpp
    │   ├── opname_kernel.cpp
    │   └── opname_kernel.h
    ├── opname.cpp
    └── opname.h
    ```

2. Define the op and the kernel as follows:
    1. Use the namespace `pace` to define the operator. The operator should be declared in the `opname.h` file.
    2. Use the namespace `pace::kernels` to define the kernel and the kernel should be declared in the `opname_kernel.h` file, should be called from the op implementation in the `opname.cpp` file. This file is ideally used to make any safety checks and to make sure the inputs and outputs are valid and for any pre-processing or post-processing of the inputs and outputs.
    3. Any kernel implementation within AMD PACE should be under the `pace::kernels::impl` namespace. This will help to identify and keep track of the kernel implementations vs the redirections into external libraries.
    4. If the kernel is specifically AVX512 based, then the kernel should be defined in `opname_kernel_avx512.cpp`. The kernel should still be declared in the `opname_kernel.h` file as a declaration. If not, the kernel should be defined in `opname_kernel.cpp` itself.

    For example,

    * `opname.h`:
        ```cpp
        namespace pace {
            // Op declaration
            at::Tensor opname(...);
        }
        ```
    * `opname.cpp`:
        ```cpp
        namespace pace {
            // Op definition
            at::Tensor opname(...) {
                // Op implementation
                opname_kernel(...);
            }
        }
        ```
    * `opname_kernel.h`:
        ```cpp
        namespace pace::kernels {
            // Kernel declaration
            void opname_kernel(...);
        }
        ```
    * `opname_kernel.cpp`:
        ```cpp
        namespace pace::kernels {
            // Kernel definition
            void opname_kernel(...) {
                // Kernel implementation
                // should redirect to AVX512 kernel if available
            }
        }
        ```
    The method names can be anything as long as they make sense and are consistent with the naming conventions.

3. All operators should have a profiling mechanism and logging mechanism enabled. There is a logging and timing module already present in the file: `csrc/core/logging.h` and can be invoked by including the file in the operator file. The logging and timing module should be used to log the input and output shapes, the time taken for the operation, and any other relevant information. The logging and timing module should be used as follows:
    ```cpp
    #include "core/logging.h"

    namespace pace {
        at::Tensor opname(...) {
            // Start the timer
            // The name of the method and the name given to the timer should be the same
            PROFILE_PACE_FUNCTION("opname");
            // Operator implementation
            ...
            // Log the input and output shapes
            PROFILE_ADD_INFO(...)
        }
    }
    ```
    Some macros are available based on the type of operation such as linear, binary etc. to make it easier to log the input and output shapes. The macros are defined in `csrc/core/logging.h` file.

    The timer works on the logic of scoping. When `PROFILE_PACE_FUNCTION("opname")` is called, the timer starts and when the scope ends, the timer stops and logs the time taken for the operation. The timer is thread safe and can be used in multi-threaded environments.

4. Once the op is defined and implementation is complete, within the `opname.cpp` file, you can register the op with the torch library as follows:
    ```cpp
    TORCH_LIBRARY_FRAGMENT(pace, m) {
        m.def("operator_name(operator_schema)");
    }

    TORCH_LIBRARY_IMPL(pace, CPU, m) {
        m.impl("operator_name", pace::opname);
    }
    ```

    **Important Notes:**
    - `TORCH_LIBRARY_FRAGMENT` defines the operator schema (signature)
    - `TORCH_LIBRARY_IMPL` provides the implementation for a specific dispatch key (CPU, QuantizedCPU, etc.)
    - For quantized operations, use `TORCH_LIBRARY_IMPL(pace, QuantizedCPU, m)` instead
    - Multiple operators can be defined in the same fragment
    - Multiple implementations can be registered for different dispatch keys

5. The function can be imported and used in the python code as follows:
    ```python
    # Make sure to import torch before importing pace
    import torch
    import pace

    ret = torch.ops.pace.opname(...)
    ```

6. **Register fake ops for torch.compile support**: To enable your operator to work with `torch.compile`, you need to register a fake implementation in `pace/_register_fake.py`. Fake ops are used by the compiler to infer output shapes and types without executing the actual operation.

    Add a corresponding fake implementation in `pace/_register_fake.py`:
    ```python
    from torch.library import register_fake
    
    @register_fake("pace::operator_name")
    def _fake_operator_name(operator_schema):
        # Compute output shape
        out_shape = ...
        out_dtype = ...
        out_device = ...
        return torch.empty(out_shape, dtype=out_dtype, device=out_device)
    ```
    
    **Key points for fake ops:**
    - The fake function should match the operator signature exactly
    - It only needs to return a tensor with the correct shape, dtype, and device
    - Do not perform actual computation - just shape/type inference
    - Quantized operations typically cannot be registered as fake ops due to qtensor limitations
    - Multiple operators with the same signature can share a fake implementation by stacking `@register_fake` decorators
    - See `pace/_register_fake.py` for examples

7. Once the method is implemented, it needs to be documented in the `docs/Ops.md` file. The documentation should include the method signature, the input and output types, and a brief description of the operator.

> For a complete example of an operator, refer to the binary operator in `csrc/ops/binary.h` and `csrc/ops/binary.cpp` for operator implementation, `csrc/ops/kernels/binary_kernel.h`, `csrc/ops/kernels/binary_kernel.cpp`, and `csrc/ops/kernels/binary_kernel_avx512.cpp` for kernel implementation, and `pace/_register_fake.py` for fake op registration for torch.compile support.

### Note:
1. Make sure that the op is registered outside of the `pace` namespace so that the op can be loaded dynamically by the torch library.

> **Attention backends**: Attention kernel implementations are organized under `csrc/ops/attention/` with per-backend subdirectories (e.g., `csrc/ops/attention/contiguous/` for standard MHA/GQA kernels). New attention backends should create their own subdirectory under `csrc/ops/attention/`. The Python-side attention integration lives in `pace/llm/attention/` with a matching per-backend folder structure.
2. Use `TORCH_LIBRARY_FRAGMENT` to define the operator schema and `TORCH_LIBRARY_IMPL` to provide the implementation for specific dispatch keys (CPU, QuantizedCPU, etc.). Do not mix definition and implementation in a single macro.
3. The AVX512 kernel should go in the file `opname_kernel_avx512.cpp` only as only those files are compiled with the AVX512 flags. Failing to do so might result in errors during compilation.
4. All AVX512 kernels should have a reference implementation in the `opname_kernel.cpp` file. This is required for the fallback mechanism in case the AVX512 kernel is not supported on the target machine and for testing purposes.
5. For torch.compile support, always register a fake op in `pace/_register_fake.py` that matches your operator's signature and returns tensors with correct shapes/types.

### Adding a Fused Op / Optimization

Fused ops combine multiple operations into a single kernel for better performance. They use `FusedOperatorType` (from `pace/ops/enum.py`) instead of `OperatorType` and follow the same registry pattern.

For a concrete example, see `pace/ops/mlp.py` — `MergedMLP` is a fused op that combines gate/up projections with activation and a down projection into a single module. It uses `FusedOperatorType.FUSEDMLPLINEAR` and the backend registry resolves the implementation at runtime. If no fused backend is registered, the op falls back to its default `forward` method (composed of individual ops).

See [PythonOps.md](PythonOps.md) for details on the operator/backend registry pattern.

## Creating a Python Operator
Please refer to [PythonOps.md](PythonOps.md#adding-new-operators-and-backends) for more details on how to create a Python operator.


## Creating a core function
All the core functions are implemented under `csrc/core/` directory. All the core functions follow the same structure and naming conventions. To create a new core function in AMD PACE, follow the steps below:
1. Create a new file under `csrc/core/` with the name of the function. For example, create files `csrc/core/core_method.h` and `csrc/core/core_method.cpp`.
The directory structure should look like:
    ```
    core/
    ├── core_method.cpp
    └── core_method.h
    ```
2. Define the core function as follows:

    1. Use the namespace `pace` to define the function. The function should be declared in the `core_method.h` file.
    2. The function should be defined in the `core_method.cpp` file.
    For example,
    * `core_method.h`:
        ```cpp
        namespace pace {
            // Function declaration
            [return type] core_method(...);
        }
        ```
    * `core_method.cpp`:
        ```cpp
        namespace pace {
            // Function definition
            [return type]  core_method(...) {
                // Function implementation
            }
        }
        ```

3. Once the function is defined and implementation is complete, within the `core_method.cpp` file, you can register the function with the AMD PACE library as follows:
    1. Import the function in the `csrc/torch_extension_bindings.cpp` file.
    2. Register the function within `torch_extension_bindings` method:
        ```cpp
        m.def(
            "core_method",
            &pace::core_method
        );
        ```
4. The function can be imported and used in the python code as follows:
    ```python
    # Make sure to import torch before importing pace
    import torch
    import pace

    ret = pace.core.core_method(...)
    ```
5. Once the method is implemented, it needs to be documented in the `docs/CoreFunctions.md` file. The documentation should include the method signature, the input and output types, and a brief description of the function.


## Logging in AMD PACE
There is a logging module that is available in AMD PACE to be used with both C++ and Python. The logging module is available in the `csrc/core/logging.h` file. There are 6 levels of logging available in AMD PACE -> `DEBUG`, `PROFILE`,  `INFO`, `WARNING`, `ERROR`, `NONE`.

The logging can be controlled by setting the environment variable `PACE_LOG_LEVEL`. Refer to [README](../README.md#verbose) for more details.

To make use of Logger in C++, include the `logging.h` file in the file where you want to log the information, and can be called using the following macros:

```cpp
#include "core/logging.h"
PACE_LOG_DEBUG(...)    // Used to log debug information.
PACE_LOG_PROFILE(...)  // Used to log profiling information.
PACE_LOG_INFO(...)     // Used to log information.
PACE_LOG_WARNING(...)  // Used to log warnings.
PACE_LOG_ERROR(...)    // Used to log errors.
```

The logging module in Python is a wrapper around the C++ logging module. The logging module in Python is available in the `pace.utils.logging` module. The logging module in Python has the same levels as the C++ logging module. The logging module in Python can be used as follows. To make use of the different logging levels, the logger should be initialized as follows:
```python
from pace.utils.logging import pacelogger, logLevel

pacelogger(logLevel.DEBUG, "...") # DEBUG level
pacelogger(logLevel.PROFILE, "...") # PROFILE level
pacelogger(logLevel.INFO, "...") # INFO level
pacelogger(logLevel.WARNING, "...") # WARNING level
pacelogger(logLevel.ERROR, "...") # ERROR level
```

There are also convenience functions available for general use that automatically prefix messages with `pace:`:
```python
from pace.utils.logging import PACE_DEBUG, PACE_INFO, PACE_WARNING, PACE_ERROR, PACE_ASSERT

PACE_DEBUG("...")
PACE_INFO("...")
PACE_WARNING("...")
PACE_ERROR("...")
PACE_ASSERT(CONDITION, "...")
```
`PACE_ASSERT` is a special case, where it will raise an exception if the condition is not met. If condition is not met, it will raise an assertion error with the message logged.

For LLM modules, this is abstracted one more level. This is to capture some extra information(please make sure to use these methods inside any python LLM related implementations) , and can be accessed like such:
```python
from pace.utils.logging import PACE_LLM_DEBUG, PACE_LLM_INFO, PACE_LLM_WARNING, PACE_LLM_ERROR, PACE_LLM_ASSERT

PACE_LLM_DEBUG("...")
PACE_LLM_INFO("...")
PACE_LLM_WARNING("...")
PACE_LLM_ERROR("...")
PACE_LLM_ASSERT(CONDITION, "...")
```
`PACE_LLM_ASSERT` is a special case, where it will raise an exception if the condition is not met. If condition is not met, it will raise an assertion error with the message logged.

## Code Style

### Python
All Python files follow the standard `black` formatting. Linting is enforced via `flake8` (see `.flake8` for the full config).

### C++
C++ formatting follows the PyTorch clang-format style and is applied automatically during the build when `ENABLE_CLANG_FORMAT=ON` is set. No manual formatting step is needed.

### Error Handling
- **Python:** Use `PACE_ASSERT(condition, "message")` for invariant checks (raises `AssertionError`). Inside LLM code (`pace/llm/`), use `PACE_LLM_ASSERT` instead. Use `ValueError` for invalid configuration and `RuntimeError` for state errors.
- **C++:** Use `TORCH_CHECK(condition, "message")` for precondition checks on dtypes, shapes, and tensor properties.

## Testing

AMD PACE uses `unittest` for all tests. Tests live under the `tests/` directory:

```
tests/
├── ops/          # Tests for C++ ops (torch.ops.pace.*)
├── python_ops/   # Tests for Python ops (pace.ops.*)
├── llm_infra/    # Tests for LLM infrastructure
├── server/       # Tests for inference server
└── test_utils.py # Shared test utilities
```

**Required coverage for new/modified ops:**
1. Functional correctness — compare output against a PyTorch reference implementation
2. Negative inputs — verify proper error handling for invalid inputs
3. Edge cases — boundary conditions, empty tensors, single-element tensors

## Basic Developer Checks
Before raising patches for review make sure of the following:

> Currently basic linting/formatting is only available. Later more linting will be enforced.

1. Code styling for C++: All the C++ files within AMD PACE follows the PyTorch format of formatting. The formatting module is integrated into AMD PACE itself. To make use of it while building the extension, use the environment variable `ENABLE_CLANG_FORMAT`.

    ```shell
    ENABLE_CLANG_FORMAT=ON pip install [-v] .
    ```

2. Code styling for Python: All the Python files within AMD PACE follows the standard `black` formatting and can be invoked as follows:
    ```
    black .
    ```
    This will format all the python files within the directory.

    Once you invoke `black`, run `flake8` to check for any linting errors.
    ```
    flake8 .
    ```
    It will check for any linting errors in the code. Make sure to fix all the errors before raising a PR.
    > NOTE: Once you fix the linting errors using `flake8`, make sure to run `black` again to rectify the formatting errors.
3. Make sure that the library builds and runs with some basic examples. Unit tests also need to be added for methods exposed.

## Submitting a PR

### Commit Messages
Follow the existing commit style — short, imperative, descriptive. Always sign off your commits with `-s`. 

### PR Guidelines
- Keep PRs small and focused on a single change. Large PRs are harder to review and more likely to introduce regressions.
- Include tests for any new or modified operators (see [Testing](#testing)).
- Update documentation in `docs/` if you add new ops, change APIs, or modify behavior.
- Make sure all [Basic Developer Checks](#basic-developer-checks) pass before submitting.
