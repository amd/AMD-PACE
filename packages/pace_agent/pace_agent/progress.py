# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Progress lines for the long-running commands.

Lines are flushed because they are read live, and they carry a label so that parallel workers
writing into the same terminal stay legible.
"""


def say(cmd, msg, label=None):
    print(f"[{cmd} {label}] {msg}" if label else f"[{cmd}] {msg}", flush=True)
