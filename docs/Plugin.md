# AMD Platform Aware Compute Engine (AMD PACE)

This sections explains how the AMD PACE is created and the basic idea behind it.

## How it works
AMD PACE follows the approach specified in [EXTENDING TORCHSCRIPT WITH CUSTOM C++ OPERATORS](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html). All ops and helpers are registered through the PyTorch dispatcher and exposed as `torch.ops.pace.*` and `torch.classes.pace.*`. The wheel ships a single plain C++ shared library (`libpace_cpp.so`) and contains no CPython extension module, so the same wheel runs on any CPython 3.x in the supported range.

There are two parts to the library:
1. The C++ library (`libpace_cpp.so`) built using CMake.
2. The python methods to wrap the dispatcher ops and provide a high level interface to the user.

### C++ library
The C++ library is built using CMake. CMake compiles the sources under `csrc/` into a single shared object linked against `torch_cpu` and `c10` (no Python C API, no `libtorch_python`). The library also links statically against oneDNN, libXSMM, and AOCL-DLP. Ops are registered with the torch dispatcher via `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL`; the process-level helper functions (`thread_bind`, `log`) live in [`csrc/core/core_ops.cpp`](../csrc/core/core_ops.cpp).

### Python methods
The python methods wrap the dispatcher ops and provide a high-level interface to the user. They live under the `pace/` folder. On `import pace`, [`pace/__init__.py`](../pace/__init__.py) calls `torch.ops.load_library(pace/lib/libpace_cpp.so)`, which fires the `TORCH_LIBRARY_FRAGMENT` static initializers and populates `torch.ops.pace.*` and `torch.classes.pace.*`. The `pace.core` module ([`pace/core.py`](../pace/core.py)) is a thin Python shim over the process-level helpers; new ops should be reached directly via `torch.ops.pace.<name>`.
