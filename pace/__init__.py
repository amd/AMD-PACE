# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from .version import __version__, __version_tuple__

try:
    import torch  # noqa F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("Torch not found, install torch. Refer to README.md.")
from . import _C as core
from . import utils
from . import llm
from . import ops
from ._register_fake import *  # noqa: F401,F403

__all__ = ["__version__", "__version_tuple__", "core", "utils", "llm", "ops"]
