# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
import json
import contextlib
from typing import Union, Optional

import tqdm
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import PreTrainedTokenizerBase
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import (
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

from pace.llm.configs import OperatorConfig
from pace.llm.models.model_list import get_encoder_model_class, get_model_class
from pace.llm.models.base_model import BaseModel
from pace.utils.logging import PACE_LLM_INFO


# The set_default_torch_dtype and no_init_weights (slightly modified)
# methods are borrowed from the transformers library. They are reimplemented
# here to ensure that the code is self-contained and does not create external
# dependency issues since they are not exported from the transformers library.
# There are cases where transformers library is importing other libraries such
# such as torchvision, accelerate, etc. which are not required for the pace
# library.
@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}


@contextlib.contextmanager
def no_init_weights():
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """

    def _skip_init(*args, **kwargs):
        pass

    # # Save the original initialization functions
    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        # # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)


def get_weight_iterator(
    hf_weight_filenames: str,
    is_safetensor_archive: bool,
    disable_tqdm: Optional[bool] = False,
):
    """
    Get an iterator over the weights in the given checkpoint files.

    Args:
        hf_weight_filenames: List of checkpoint files.
        is_safetensor_archive: Whether the checkpoint is a safetensor archive.

    Returns:
        Iterator over the weights in the checkpoint files.
    """
    for hf_weight_filename in tqdm.tqdm(
        hf_weight_filenames, "Loading checkpoints", disable=disable_tqdm
    ):
        if is_safetensor_archive:
            with safe_open(hf_weight_filename, framework="pt") as f:
                for name in f.keys():
                    param = f.get_tensor(name)
                    yield name, param
        else:
            state = torch.load(
                hf_weight_filename, map_location="cpu", weights_only=True, mmap=True
            )
            for name, param in state.items():
                yield name, param
            del state


def load_weights(
    model: BaseModel,
    model_path: Union[str, os.PathLike],
    disable_tqdm: Optional[bool] = False,
) -> BaseModel:
    """
    Load weights from a checkpoint file into the model.
    Check for the following conditions to load model weights:
    1. Safetensors checkpoint
    2. Sharded safetensors checkpoint
    3. PyTorch checkpoint
    4. Sharded PyTorch checkpoint

    Args:
        model: Model to load weights into.
        model_path: Path to the model checkpoint.

    Returns:
        Model with weights loaded.
    """

    archive_file = None
    is_sharded = False
    is_safetensor_archive = False
    # Prefer safetensors. Check for these 4 conditions to see if we can load model weights
    if os.path.isfile(os.path.join(model_path, SAFE_WEIGHTS_NAME)):
        # Load from a safetensors checkpoint
        archive_file = os.path.join(model_path, SAFE_WEIGHTS_NAME)
        is_safetensor_archive = True
    elif os.path.isfile(os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)):
        # Load from a sharded safetensors checkpoint
        archive_file = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
        is_sharded = True
        is_safetensor_archive = True
    elif os.path.isfile(os.path.join(model_path, WEIGHTS_NAME)):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(model_path, WEIGHTS_NAME)
    elif os.path.isfile(os.path.join(model_path, WEIGHTS_INDEX_NAME)):
        # Load from a sharded PyTorch checkpoint
        archive_file = os.path.join(model_path, WEIGHTS_INDEX_NAME)
        is_sharded = True
    else:
        raise FileNotFoundError(
            f"Either the model is not available, or is an incorrect type, please recheck {model_path} for models available for PyTorch."
        )

    if is_sharded:
        with open(archive_file, "r") as f:
            index = json.loads(f.read())
        hf_weight_filenames = sorted(set(index["weight_map"].values()))
        hf_weight_filenames = [os.path.join(model_path, f) for f in hf_weight_filenames]
    else:
        hf_weight_filenames = [archive_file]

    weight_iterator = get_weight_iterator(
        hf_weight_filenames,
        is_safetensor_archive=is_safetensor_archive,
        disable_tqdm=disable_tqdm,
    )
    model.load_weights(weight_iterator)
    model.preprocess_weights()
    return model


def init_model(
    model_path: Union[str, os.PathLike],
    dtype: Optional[torch.dtype] = torch.bfloat16,
    opconfig: Optional[OperatorConfig] = None,
    disable_tqdm: Optional[bool] = False,
    encoder: bool = False,
) -> BaseModel:
    """
    Initialize the model from the given model path, with the given dtype.
    Load the model weights from the checkpoint file.

    Args:
        model_path: Path to the model checkpoint.
        dtype: Data type to initialize the model with.
        encoder: If True, resolve ``architectures[0]`` against the encoder
            table (``EmbeddingModel``). If False, against the decoder table
            (``LLMModel`` / generation backend).

    Returns:
        Model with weights loaded.
    """

    model_config = AutoConfig.from_pretrained(model_path)
    model_arch = model_config.architectures[0]
    PACE_LLM_INFO(f"Creating the model: {model_arch}")
    resolver = get_encoder_model_class if encoder else get_model_class
    model_class = resolver(model_arch)

    with set_default_torch_dtype(dtype):
        with no_init_weights():
            model = model_class(model_config, opconfig)
        model = load_weights(model, model_path=model_path, disable_tqdm=disable_tqdm)

    return model.eval()


def get_tokenizer(
    model_path: Union[str, os.PathLike],
    tokenizer_path: Optional[Union[str, os.PathLike]] = None,
) -> PreTrainedTokenizerBase:
    """
    Get the tokenizer from the given model path.
    If tokenizer path is provided, it gets priority.

    Args:
        model_path: Path to the model checkpoint.
        tokenizer_path: Path to the tokenizer checkpoint.

    Returns:
        Tokenizer.
    """

    # fix_mistral_regex only changes Mistral tokenizers saved with the faulty pre-tokenizer regex.
    try:
        # If tokenizer path is specificed, it gets priority
        if tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, fix_mistral_regex=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, fix_mistral_regex=True
            )
    except OSError:
        error_message = "Tokenizer not found at {model_path}"
        if tokenizer_path is not None:
            error_message += f" or {tokenizer_path}"
        error_message += ". Please check the path provided."
        raise FileNotFoundError(error_message)

    return tokenizer
