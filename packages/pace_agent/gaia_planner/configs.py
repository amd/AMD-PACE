# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""CLI configuration for the GAIA planner benchmark runner."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Vendored into the PACE Agent repo: the package now lives at <pace-agent-repo>/gaia_planner, so the
# repo root (which may hold an optional WebVoyager/ checkout and is a sane default for
# --webvoyager-root/--nsjail-root) is one level up. nsjail_agent is vendored as a top-level
# package, so it is importable regardless of --nsjail-root.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GaiaCliConfig:
    """All knobs for ``python -m gaia_planner.cli`` (and PACE Agent ``run_gaia`` wrappers)."""

    task_file: Path | None
    huggingface: bool
    level: str
    split: str
    hf_config: str | None
    hf_token: str | None
    task_id: str | None
    source: str | None
    max_tasks: int | None
    model: str
    base_url: str
    api_key: str
    local_base_url: str | None
    local_api_key: str
    max_turns: int
    no_tools: bool
    engine: str
    output_dir: Path
    webvoyager_root: Path
    webvoyager_python: str
    web_model: str | None
    web_base_url: str | None
    web_api_key: str | None
    web_max_iterations: int
    web_max_attached_imgs: int
    web_timeout: int
    web_headless: bool
    web_text_only: bool
    nsjail_root: Path
    allow_local_executor: bool
    sandbox_memory_mb: int
    sandbox_cpu_seconds: int
    sandbox_wall_seconds: int
    video_model: str | None
    image_model: str | None
    video_max_frames: int
    video_max_height: int
    video_download_timeout: int
    audio_model: str | None
    audio_whisper_size: str
    audio_download_timeout: int

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> GaiaCliConfig:
        return cls(
            task_file=ns.task_file,
            huggingface=ns.huggingface,
            level=ns.level,
            split=ns.split,
            hf_config=ns.hf_config,
            hf_token=ns.hf_token,
            task_id=ns.task_id,
            source=ns.source,
            max_tasks=ns.max_tasks,
            model=ns.model,
            base_url=ns.base_url,
            api_key=ns.api_key,
            local_base_url=ns.local_base_url,
            local_api_key=ns.local_api_key,
            max_turns=ns.max_turns,
            no_tools=ns.no_tools,
            engine=ns.engine,
            output_dir=ns.output_dir,
            webvoyager_root=ns.webvoyager_root,
            webvoyager_python=ns.webvoyager_python,
            web_model=ns.web_model,
            web_base_url=ns.web_base_url,
            web_api_key=ns.web_api_key,
            web_max_iterations=ns.web_max_iterations,
            web_max_attached_imgs=ns.web_max_attached_imgs,
            web_timeout=ns.web_timeout,
            web_headless=ns.web_headless,
            web_text_only=ns.web_text_only,
            nsjail_root=ns.nsjail_root,
            allow_local_executor=ns.allow_local_executor,
            sandbox_memory_mb=ns.sandbox_memory_mb,
            sandbox_cpu_seconds=ns.sandbox_cpu_seconds,
            sandbox_wall_seconds=ns.sandbox_wall_seconds,
            video_model=ns.video_model,
            image_model=ns.image_model,
            video_max_frames=ns.video_max_frames,
            video_max_height=ns.video_max_height,
            video_download_timeout=ns.video_download_timeout,
            audio_model=ns.audio_model,
            audio_whisper_size=ns.audio_whisper_size,
            audio_download_timeout=ns.audio_download_timeout,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coordinate WebVoyager and nsjail-agent-tool on local GAIA tasks."
    )
    parser.add_argument(
        "task_file",
        type=Path,
        nargs="?",
        help="Local JSON array or JSONL task file. Omit when using --huggingface.",
    )
    hf_group = parser.add_argument_group("HuggingFace GAIA dataset")
    hf_group.add_argument(
        "--huggingface",
        "--hf",
        dest="huggingface",
        action="store_true",
        help="Load the official GAIA dataset from HuggingFace (gaia-benchmark/GAIA) "
        "instead of a local file. The repo is gated; accept its terms and set HF_TOKEN.",
    )
    hf_group.add_argument(
        "--level",
        choices=["1", "2", "3", "all"],
        default="all",
        help="GAIA difficulty level to run (default: all). Implies --huggingface.",
    )
    hf_group.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="GAIA split. 'test' has no reference answers, so it is unscored.",
    )
    hf_group.add_argument(
        "--hf-config",
        default=None,
        help="Override the GAIA dataset config (default: 2023_all).",
    )
    hf_group.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="HuggingFace access token (default: HF_TOKEN env var).",
    )
    parser.add_argument("--task-id", help="Run only this task_id/id.")
    parser.add_argument(
        "--source",
        help="Run only tasks whose 'source' field matches (e.g. nsjail_examples, "
        "webvoyager_gaia_web).",
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--model", default=os.getenv("LLM_GATEWAY_MODEL", "gpt-4o"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_GATEWAY_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_GATEWAY_KEY") or os.getenv("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--local-base-url",
        default=os.getenv("GAIA_LOCAL_BASE_URL"),
        help="OpenAI-compatible endpoint for locally hosted (e.g. Radeon vLLM) models used by "
        "the vision tools (analyze_video/analyze_image) and the analyze_audio transcript "
        "Q&A. When set, those tools use this endpoint while the planner --model keeps running "
        "on --base-url (the gateway). Defaults to the GAIA_LOCAL_BASE_URL env var. The model "
        "names come from --video-model/--image-model/--audio-model or GAIA_VISION_MODEL.",
    )
    parser.add_argument(
        "--local-api-key",
        default=os.getenv("GAIA_LOCAL_KEY", ""),
        help="API key (Bearer token) for --local-base-url. Most local vLLM servers accept "
        "any value; defaults to the GAIA_LOCAL_KEY env var. The gateway subscription header "
        "is never sent to this endpoint.",
    )
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Closed-book mode: never advertise any tools to the model and never open "
        "the sandbox. The model is not told which tools exist and must reason to the "
        "answer from its own knowledge. Useful to measure unaided reasoning.",
    )
    parser.add_argument(
        "--engine",
        choices=["loop", "langgraph"],
        default="loop",
        help="Planner control-flow engine: 'loop' (default, hand-rolled ReAct loop) or "
        "'langgraph' (equivalent LangGraph StateGraph). Both share tools, nudges, budget, "
        "and result shape.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("gaia-results"))
    parser.add_argument(
        "--webvoyager-root",
        type=Path,
        default=WORKSPACE_ROOT / "WebVoyager",
    )
    parser.add_argument("--webvoyager-python", default=sys.executable)
    parser.add_argument("--web-model", default=None)
    parser.add_argument("--web-base-url", default=None)
    parser.add_argument("--web-api-key", default=None)
    parser.add_argument("--web-max-iterations", type=int, default=15)
    parser.add_argument(
        "--web-max-attached-imgs",
        type=int,
        default=3,
        help="Screenshots WebVoyager attaches per turn. Lower (e.g. 1) to avoid "
        "late-turn gateway context/payload limits.",
    )
    parser.add_argument("--web-timeout", type=int, default=600)
    parser.add_argument(
        "--web-headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--web-text-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--nsjail-root",
        type=Path,
        default=WORKSPACE_ROOT / "nsjail-agent-tool",
        help="Ignored: nsjail_agent is vendored next to gaia_planner. Kept so older "
        "invocation scripts that pass --nsjail-root keep working.",
    )
    parser.add_argument(
        "--allow-local-executor",
        action="store_true",
        help="Allow unsafe host execution if nsjail is unavailable (development only).",
    )
    parser.add_argument(
        "--sandbox-memory-mb",
        type=int,
        default=1024,
        help="Memory limit (MB) for each execute_code sandbox call (16-4096). "
        "Raise for pandas/opencv/scikit-image, which OOM at the 256 MB default.",
    )
    parser.add_argument(
        "--sandbox-cpu-seconds",
        type=int,
        default=60,
        help="CPU-time limit (seconds) per execute_code call (1-300).",
    )
    parser.add_argument(
        "--sandbox-wall-seconds",
        type=int,
        default=120,
        help="Wall-clock limit (seconds) per execute_code call (1-600).",
    )
    parser.add_argument(
        "--video-model",
        default=None,
        help="Multimodal model used by analyze_video. Falls back to the GAIA_VISION_MODEL "
        "env var, then --model. Set this (or GAIA_VISION_MODEL) to a stronger vision model "
        "to improve video understanding without changing the planner --model.",
    )
    parser.add_argument(
        "--image-model",
        default=None,
        help="Multimodal model used by analyze_image. Falls back to the GAIA_VISION_MODEL "
        "env var, then --model. Independent of --video-model.",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=32,
        help="Frames sampled per analyze_video call.",
    )
    parser.add_argument(
        "--video-max-height",
        type=int,
        default=360,
        help="Max frame height (px) sampled for analyze_video.",
    )
    parser.add_argument(
        "--video-download-timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for downloading a video in analyze_video.",
    )
    parser.add_argument(
        "--audio-model",
        default=None,
        help="Model used by analyze_audio to answer from the transcript (default: --model).",
    )
    parser.add_argument(
        "--audio-whisper-size",
        default=os.getenv("GAIA_WHISPER_SIZE", "base"),
        help="faster-whisper model size for transcription: tiny, base, small, "
        "medium, large-v3 (default: base).",
    )
    parser.add_argument(
        "--audio-download-timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for downloading an audio URL in analyze_audio.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> GaiaCliConfig:
    """Parse argv into a :class:`GaiaCliConfig`."""
    return GaiaCliConfig.from_namespace(build_parser().parse_args(argv))
