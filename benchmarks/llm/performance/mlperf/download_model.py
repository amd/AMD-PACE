#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
from huggingface_hub import snapshot_download

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "model")

# Download the model without symlinks
model_path = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)

print(f"Model downloaded successfully to: {model_path}")
