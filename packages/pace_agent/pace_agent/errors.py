# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PACE Agent error taxonomy with stable, scriptable CLI exit codes.

The CLI maps an uncaught ``PaceAgentError`` to its ``exit_code`` so failures are
distinguishable in scripts. Exit code 2 is reserved for argparse usage errors.
"""


class PaceAgentError(Exception):
    exit_code = 1


class BrowserUnavailable(PaceAgentError):
    exit_code = 3


class ProviderUnavailable(PaceAgentError):
    exit_code = 4


class ReplayDivergence(PaceAgentError):
    exit_code = 5


class UnsupportedType(PaceAgentError):
    exit_code = 6


class ConfigError(PaceAgentError):
    exit_code = 7


class TaskStalled(PaceAgentError):
    exit_code = 8
