# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Characterization: sample N live runs and hand the recorder a representative one."""

from pace_agent.characterize.runner import Characterization, characterize
from pace_agent.characterize.signature import Step, register_step_extractor

__all__ = ["Characterization", "Step", "characterize", "register_step_extractor"]
