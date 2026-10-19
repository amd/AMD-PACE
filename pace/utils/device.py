# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

import torch


def gpu_backend() -> Optional[str]:
    """
    Identify the GPU backend the installed PyTorch was built against.

    Returns:
        Optional[str]: "ROCm" for an AMD build, "CUDA" for an NVIDIA build, None for a CPU build.
    """
    if torch.version.hip is not None:
        return "ROCm"
    if torch.version.cuda is not None:
        return "CUDA"
    return None


def gpu_available() -> bool:
    """
    Check whether a GPU is usable through the installed PyTorch.

    Returns:
        bool: True if at least one device is visible, on either ROCm or CUDA.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def gpu_count() -> int:
    """
    Count the GPUs visible to the installed PyTorch.

    Returns:
        int: Number of visible GPUs, 0 if none are usable.
    """
    return torch.cuda.device_count() if gpu_available() else 0


def gpu_name(index: int = 0) -> Optional[str]:
    """
    Get the product name of a visible GPU.

    Args:
        index: Device index.

    Returns:
        Optional[str]: Device name, or None if no GPU is usable.
    """
    if not gpu_available():
        return None
    return torch.cuda.get_device_name(index)


def gpu_memory_bytes(index: int = 0) -> Optional[int]:
    """
    Get the total memory of a visible GPU.

    Args:
        index: Device index.

    Returns:
        Optional[int]: Total memory in bytes, or None if no GPU is usable.
    """
    if not gpu_available():
        return None
    return torch.cuda.get_device_properties(index).total_memory


def require_gpu(feature: str) -> None:
    """
    Raise with an actionable message if `feature` cannot run on a GPU here.

    Args:
        feature: Name of the feature, used in the error message.

    Raises:
        RuntimeError: If PyTorch is a CPU build, or no GPU is visible.
    """
    if gpu_available():
        return
    backend = gpu_backend()
    if backend is None:
        raise RuntimeError(
            f"{feature} requires a GPU, but the installed PyTorch ({torch.__version__}) is a CPU "
            "build. Install PyTorch for your accelerator, then the GPU variant:\n"
            "  AMD:    pip install --extra-index-url https://download.pytorch.org/whl/rocm7.2 torch==2.13.0+rocm7.2\n"
            "  NVIDIA: pip install torch==2.13.0\n"
            '  pip install "amd-pace[gpu]"'
        )
    mask = "HIP_VISIBLE_DEVICES" if backend == "ROCm" else "CUDA_VISIBLE_DEVICES"
    raise RuntimeError(
        f"{feature} requires a GPU. PyTorch {torch.__version__} is a {backend} build but no device "
        f"is visible. Check that a GPU is present, that {backend} is installed, and that {mask} is "
        "not masking it."
    )


def describe_device() -> str:
    """
    Describe the installed PyTorch build and any visible GPUs.

    Returns:
        str: One-line summary for logs and bug reports.
    """
    backend = gpu_backend()
    if backend is None:
        return f"torch {torch.__version__} (CPU build)"
    if not gpu_available():
        return f"torch {torch.__version__} ({backend} build, no device visible)"
    # "each" is only true when the visible devices agree, so they are grouped by model.
    counts: dict = {}
    for index in range(gpu_count()):
        key = (gpu_name(index), gpu_memory_bytes(index))
        counts[key] = counts.get(key, 0) + 1
    groups = []
    for (name, memory), count in counts.items():
        size = f"{memory / 2**30:.0f} GiB"
        groups.append(
            f"{count} x {name} ({size} each)" if count > 1 else f"{name} ({size})"
        )
    return f"torch {torch.__version__}, " + ", ".join(groups)
