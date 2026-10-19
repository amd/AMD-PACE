# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence

import torch

from pace.llm.configs import SamplingConfig
from pace.llm.sampler import Sampler
from pace.llm.stopping_criteria import StoppingCriteria


class GenerationStatus(Enum):
    PREFILL_IN_PROGRESS = auto()
    PREFILL_COMPLETED = auto()
    DECODING_IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    ERROR = auto()


class TerminationReason(Enum):
    COMPLETED = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    input_ids: torch.Tensor
    sampling_config: Optional[SamplingConfig] = None
    attention_mask: Optional[torch.Tensor] = None
    infer_padding: bool = True
    accumulate_output_tensors: bool = False
    logprobs_n: int = 0


@dataclass
class GenerationProposal:
    token_ids: torch.Tensor
    status: GenerationStatus
    computed_prompt_tokens: int = 0
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    prompt_logprobs: Optional[Any] = None


@dataclass
class GenerationDelta:
    request_id: str
    token_ids: List[int]
    status: GenerationStatus
    computed_prompt_tokens: int = 0
    finish_reason: Optional[str] = None
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[Any] = None
    prompt_logprobs: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class GenerationBatchResult:
    deltas: Dict[str, GenerationDelta]


@dataclass
class GenerationState:
    request_id: str
    prompt_token_ids: torch.Tensor
    sampling_config: SamplingConfig
    sampler: Sampler
    stopping_criteria: StoppingCriteria
    eos_token_ids: set
    token_buffer: torch.Tensor
    token_count: int
    prompt_length: int
    logprobs_n: int = 0
    attention_mask: Optional[torch.Tensor] = None
    infer_padding: bool = True
    accumulate_output_tensors: bool = False
    allocation_batch_size: int = 1
    num_computed_tokens: int = 0
    status: GenerationStatus = GenerationStatus.PREFILL_IN_PROGRESS
    finish_reason: Optional[str] = None
    resources_released: bool = False
    defer_release: bool = False
    backend_state: Any = None
    accumulated_probs: List[torch.Tensor] = field(default_factory=list)
    accumulated_logprobs: List[torch.Tensor] = field(default_factory=list)
    input_logprobs: Optional[torch.Tensor] = None

    @property
    def input_ids(self) -> torch.Tensor:
        return self.token_buffer[:, : self.token_count]

    @property
    def generated_count(self) -> int:
        return self.token_count - self.prompt_length

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.sampling_config.max_new_tokens - self.generated_count)

    @property
    def finished(self) -> bool:
        return self.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.CANCELLED,
            GenerationStatus.ERROR,
        }

    def append_tokens(self, token_ids: Sequence[int]) -> None:
        count = len(token_ids)
        if count > self.remaining_tokens:
            raise ValueError(
                f"cannot append {count} tokens with {self.remaining_tokens} remaining"
            )
        end = self.token_count + count
        if end > self.token_buffer.shape[1]:
            raise RuntimeError(
                f"token buffer overflow: {end} > {self.token_buffer.shape[1]}"
            )
        self.token_buffer[0, self.token_count : end] = torch.as_tensor(
            token_ids,
            dtype=self.token_buffer.dtype,
            device=self.token_buffer.device,
        )
        self.token_count = end
