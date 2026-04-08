# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(
        f"Failed to read version information:\n{e}",
        RuntimeWarning,
        stacklevel=2,
    )

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
