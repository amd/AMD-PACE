# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-Generated code.
# ******************************************************************************

import os
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, model_validator
from typing import List, Optional, Union, Any
from prometheus_client import Histogram

TTFT_BUCKETS = [
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    30.0,
    60.0,
]
TPOT_BUCKETS = [
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
]

ttft_histogram = Histogram(
    "pace_ttft_seconds",
    "Time To First Token (seconds)",
    buckets=TTFT_BUCKETS,
)

tpot_histogram = Histogram(
    "pace_tpot_seconds",
    "Time Per Output Token (seconds)",
    buckets=TPOT_BUCKETS,
)


class RequestStatus(str, Enum):
    """Enum for request status values."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class CompletionResponse(BaseModel):
    request_id: str
    status: str
    message: Optional[str] = None
    created_at: Optional[str] = None  # ISO formatted timestamp


class GenerationConfig(BaseModel):
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stop_strings: Optional[Union[str, List[str]]] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data):
        if isinstance(data, dict):
            if "num_beams" in data:
                raise ValueError(
                    "Beam search is no longer supported. "
                    "Remove 'num_beams' from your request."
                )
            if "max_tokens" in data and "max_new_tokens" not in data:
                data["max_new_tokens"] = data.pop("max_tokens")
            if "stop" in data and "stop_strings" not in data:
                data["stop_strings"] = data.pop("stop")
        return data


class CompletionRequest(BaseModel):
    model: str = "facebook/opt-6.7b"
    prompt: Union[List[str], str]
    stream: bool = False
    gen_config: Optional[GenerationConfig] = None
    mlperf_mode: bool = False  # Enable MLPerf mode (skip text decoding)
    input_length: Optional[int] = None
    prompt_tokens: Optional[int] = None  # For MLPerf dataset


@dataclass
class RequestStats:
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    generated_tokens_count: int = 0
    input_length: int = 0
    prefill_finished_at: Optional[float] = None
    TTFT: Optional[float] = None
    TPOT: Optional[float] = None
    end_wait_time: Optional[float] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Mimic dict.get(): safely get attribute or default"""
        return getattr(self, key, default)


class Request:
    def __init__(
        self,
        req: CompletionRequest,
        req_sampling_params: GenerationConfig,
        token_queue: asyncio.Queue,
    ):
        self.req: CompletionRequest = req
        self.request_id: str = str(uuid.uuid4())
        self.req_sampling_params: GenerationConfig = req_sampling_params
        self.token_queue: asyncio.Queue = token_queue
        self.status: RequestStatus = RequestStatus.QUEUED
        # Default to first engine (index 0); scheduler sets these in submit_request()
        self.assigned_engine_index: int = 0
        self.assigned_engine_url: Optional[str] = None
        self.response_queue: asyncio.Queue = asyncio.Queue()
        self.batch_submit_time: Optional[float] = None
        self.priority: str = "normal"
        self.req_stats = RequestStats()
        # Extracted once from completion request; used by schedulers and frontend
        self.mlperf_mode: bool = getattr(req, "mlperf_mode", False)


class HTTPConfig:
    """
    HTTP timeout configuration for engine requests.
    Defaults optimized for LLM inference: total=300s, connect/sock_connect/sock_read=30s.
    Override via HTTP_TIMEOUT_* environment variables.
    """

    def __init__(self):
        self.total = float(
            os.environ.get("HTTP_TIMEOUT_TOTAL", "300")
        )  # Total request timeout (5 min for long inference)
        self.connect = float(
            os.environ.get("HTTP_TIMEOUT_CONNECT", "30")
        )  # Connection establishment timeout in seconds
        self.sock_connect = float(
            os.environ.get("HTTP_TIMEOUT_SOCK_CONNECT", "30")
        )  # Socket connection timeout in seconds
        self.sock_read = float(
            os.environ.get("HTTP_TIMEOUT_SOCK_READ", "30")
        )  # Read timeout between data chunks  in seconds

        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} timeout must be positive, got {value}")


http_config = HTTPConfig()
