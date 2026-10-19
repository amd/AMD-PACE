# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
from pathlib import Path
from filelock import FileLock
from typing import Union

import fnmatch
from tqdm import tqdm
from transformers import AutoConfig
from huggingface_hub import (
    constants,
    HfFileSystem,
    snapshot_download,
    hf_hub_download,
)

from pace.llm.models.model_list import models_supported


# Copied from vllm implementation to turn off TQDM, so that
# when the model is cached, the progress bar is not shown.
class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        # huggingface_hub 1.x passes `disable=` to the configured tqdm_class
        # itself (via functools.partial in _snapshot_download). Older
        # versions did not. Write into kwargs unconditionally so the same
        # class works against both.
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def download_or_cached_weights(
    model_name: str,
    native_only: bool = True,
) -> os.PathLike:
    """
    Download the model weights from the Hugging Face Hub
    if not already downloaded and cached.
    NOTE: We are not explicitly checking if the model
    weights are already downloaded and cached, as the
    Hugging Face Hub cache directory is used, which
    automatically checks if the weights are already
    downloaded and cached.

    Args:
        model_name (str): Model name or path
        native_only (bool): Refuse architectures the native backend does not implement

    Returns:
        os.PathLike: Path to the model
    """

    # Use Huggingface Hub cache directory
    cache_dir = constants.HF_HUB_CACHE

    # Prefer to safetensors file format if available
    file_formats = {
        "safetensors": "*.safetensors",
        "bin": "*.bin",
    }

    # Do not download the original files
    ignore_patterns = ["*.md", "original/*", "consolidated*", "flax*", "tf*"]

    # If online access is available, prefer to download the safetensors file format
    if not constants.HF_HUB_OFFLINE:
        fs = HfFileSystem()
        file_list = fs.ls(model_name, detail=False)

        use_format = None
        for format_name, format_files in file_formats.items():
            if len(fnmatch.filter(file_list, format_files)) > 0:
                use_format = format_name
                break

        if use_format is None:
            raise ValueError(
                f"Could not find any model weights for {model_name} in the Hugging Face Hub with safetensors or bin format."
                f"Please check the model name and try again."
            )

        # Do not download bin if safetensors is available
        ignore_patterns += [
            file_formats[formats] for formats in file_formats if formats != use_format
        ]

    # Check if the model is supported by downloading
    # the config file only, if not supported, raise an
    # error, it not download the weights or find them
    # in the cache
    if native_only:
        model_config_file = hf_hub_download(
            model_name, filename=constants.CONFIG_NAME, cache_dir=cache_dir
        )
        model_config = AutoConfig.from_pretrained(model_config_file)
        model_arch = model_config.architectures[0]
        if model_arch not in models_supported():
            raise ValueError(
                f"Model {model_name} is not supported by the library. "
                f"Supported models are: {models_supported()}"
            )

    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    model_folder_name = "models--" + model_name.replace(
        "/", constants.REPO_ID_SEPARATOR
    )
    cached_folder = os.path.join(cache_dir, model_folder_name)
    with FileLock(cached_folder + ".lock"):
        hf_folder = snapshot_download(
            model_name,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            local_files_only=constants.HF_HUB_OFFLINE,
        )

    # Clean up the lock file after download
    lock_file = cached_folder + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception:
            pass

    return Path(hf_folder)


def resolve_model_path(
    model_name_or_path: Union[str, os.PathLike], native_only: bool = True
) -> os.PathLike:
    """
    Resolve the model path by downloading the model weights
    if the model name is provided.

    Args:
        model_name_or_path (Union[str, os.PathLike]): Model name or path
        native_only (bool): Refuse architectures the native backend does not implement

    Returns:
        os.PathLike: Path to the model
    """

    if os.path.isdir(model_name_or_path):
        # If the model_name_or_path is a directory, do not
        # do any further processing, just return the path
        model_path = Path(model_name_or_path)
    else:
        # If it's not a directory, it's a model name,
        # download the model weights
        model_path = download_or_cached_weights(model_name_or_path, native_only)

    return model_path
