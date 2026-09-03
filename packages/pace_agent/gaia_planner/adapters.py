# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Adapters for the existing WebVoyager and nsjail agents."""

from __future__ import annotations

import ast
import base64
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import GaiaTask

BROWSE_WEB_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "browse_web",
        "description": (
            "Delegate a focused research question to WebVoyager. It controls a real browser and "
            "returns its answer and artifact location. Use a narrow goal and call again if a "
            "different fact needs independent research."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Focused web research question.",
                },
                "start_url": {
                    "type": "string",
                    "description": "Optional page to open first; defaults to DuckDuckGo. "
                    "Prefer authoritative sites directly (e.g. arxiv.org) and avoid "
                    "google.com, which blocks automated browsing with CAPTCHAs.",
                },
            },
            "required": ["goal"],
        },
    },
}


ANALYZE_VIDEO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_video",
        "description": (
            "Watch a video and answer a specific visual question about its footage. "
            "Downloads the video (YouTube or a direct URL), samples frames across a "
            "time window, and reasons over them with a vision model. Use this for "
            "questions that require actually seeing moving-image content - e.g. "
            "counting objects/animals/people visible on screen, reading on-screen "
            "text, or describing what happens - which browse_web cannot do. Narrow "
            "the time window when the question refers to a specific part of the video."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Video URL (YouTube link or direct media URL).",
                },
                "question": {
                    "type": "string",
                    "description": "The precise visual question to answer from the footage.",
                },
                "start_seconds": {
                    "type": "number",
                    "description": "Optional start of the time window to inspect (default 0).",
                },
                "end_seconds": {
                    "type": "number",
                    "description": "Optional end of the time window (default: end of video).",
                },
                "frame_interval_seconds": {
                    "type": "number",
                    "description": "Optional seconds between sampled frames. Smaller catches "
                    "brief moments but costs more; chosen automatically if omitted.",
                },
            },
            "required": ["url", "question"],
        },
    },
}


ANALYZE_AUDIO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_audio",
        "description": (
            "Transcribe spoken audio and answer a question about it. Use this for "
            "any question that requires LISTENING - e.g. an attached .mp3/.wav/.m4a "
            "recording or an audio URL - which execute_code cannot transcribe (no "
            "speech-to-text engine is available inside the sandbox). Provide the "
            "attachment's basename in `file` (or an audio/video `url`). Returns the "
            "full transcript plus an answer derived from it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The precise question to answer from the audio transcript.",
                },
                "file": {
                    "type": "string",
                    "description": "Basename of an attached audio file (e.g. 'Homework.mp3'). "
                    "Provide this OR url.",
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Audio/video URL to download and transcribe. Provide this OR file."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}


ANALYZE_IMAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": (
            "Look at an attached image and answer a specific visual question about it "
            "with a vision model. Use this for any attached image the sandbox cannot "
            "interpret - a chess/board position, a chart or plot, a diagram, a photo, "
            "a scanned page, or handwriting. execute_code (PIL/OpenCV/OCR) can open the "
            "file but cannot 'see' its content, so call analyze_image with the "
            "attachment's basename and a precise question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Basename of an attached image (e.g. 'position.png').",
                },
                "question": {
                    "type": "string",
                    "description": "The precise visual question to answer from the image.",
                },
            },
            "required": ["file", "question"],
        },
    },
}


FETCH_URL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "HTTP GET a URL host-side and save the response into the execute_code "
            "workspace, then read it with execute_code. Use this for online PDFs/"
            "documents/data files that browse_web cannot read (open the saved file with "
            "pypdf/pdfminer, pandas/openpyxl, etc.) AND for REST/JSON API endpoints - "
            "e.g. api.github.com, a MediaWiki/Wikipedia api.php query, or the arXiv "
            "export API - saving the JSON to parse in execute_code. The sandbox has NO "
            "internet, so this (or provision) is the ONLY way to get web data to the "
            "code agent. Returns the saved filename."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Direct URL of the file to download (e.g. a .pdf link).",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional name to save as in the workspace (e.g. 'paper.pdf'). "
                    "Defaults to the URL's basename.",
                },
            },
            "required": ["url"],
        },
    },
}


PROVISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "provision",
        "description": (
            "Set up whatever the sandbox is missing so execute_code can solve the task. "
            "The sandbox itself has no internet; this runs host-side with access limited "
            "to an allowlist of trusted hosts (PyPI, GitHub, Hugging Face, arXiv, and a "
            "few data/science hosts). Two capabilities: (1) `pip` installs Python packages "
            "that then become importable in execute_code; (2) `downloads` fetches files - "
            "data files, documents/PDFs, model weights, or even a CLI/engine BINARY - into "
            "the current workspace so execute_code can open or run them. Use this instead "
            "of giving up when a library, tool, engine, or web file is not present: "
            "provision it, then use it in execute_code (a downloaded binary can be marked "
            "executable and run as ./name)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pip": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Python package specs to install (e.g. ['python-chess', "
                    "'pdfplumber']). Installed for the sandbox interpreter.",
                },
                "downloads": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Direct URL to fetch.",
                            },
                            "filename": {
                                "type": "string",
                                "description": "Optional name to save in the workspace; "
                                "defaults to the URL basename.",
                            },
                            "executable": {
                                "type": "boolean",
                                "description": "Mark the saved file executable so "
                                "execute_code can run it (e.g. a downloaded engine binary).",
                            },
                        },
                        "required": ["url"],
                    },
                    "description": "Files to download into the workspace.",
                },
            },
        },
    },
}


@dataclass(frozen=True)
class AudioTranscriptionConfig:
    """Settings for local speech-to-text transcription via faster-whisper."""

    model: str
    whisper_model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    download_timeout_seconds: int = 300
    use_gateway_headers: bool = False


@dataclass(frozen=True)
class VideoAnalysisConfig:
    """Settings for downloading a video and sampling frames for the vision model."""

    model: str
    max_frames: int = 32
    max_height: int = 360
    default_interval_seconds: float = 2.0
    jpeg_quality: int = 70
    download_timeout_seconds: int = 300
    use_gateway_headers: bool = False


class VideoAnalysisAdapter:
    """Downloads a video, samples frames, and asks a vision model about them."""

    def __init__(self, client: Any, config: VideoAnalysisConfig) -> None:
        self.client = client
        self.config = config

    def call(self, arguments: dict[str, Any]) -> str:
        url = arguments.get("url")
        question = arguments.get("question")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("analyze_video requires a non-empty url")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("analyze_video requires a non-empty question")
        start = _as_float(arguments.get("start_seconds"), 0.0)
        end = _as_float(arguments.get("end_seconds"), None)
        interval = _as_float(arguments.get("frame_interval_seconds"), None)

        with tempfile.TemporaryDirectory(prefix="gaia-video-") as temp_dir:
            work = Path(temp_dir)
            video_path = self._download(url.strip(), work)
            duration = _probe_duration(video_path)
            window_end = end if end is not None else duration
            if window_end is not None:
                window_end = min(window_end, duration) if duration else window_end
            frames = self._extract_frames(video_path, start, window_end, interval, work)
            if not frames:
                return json.dumps(
                    {
                        "status": "error",
                        "answer": None,
                        "error": "No frames could be extracted from the video.",
                    }
                )
            answer = self._ask_model(question.strip(), frames)

        return json.dumps(
            {
                "status": "success",
                "answer": answer,
                "frames_analyzed": len(frames),
                "duration_seconds": duration,
                "window": {"start": start, "end": window_end},
            },
            ensure_ascii=False,
        )

    def _download(self, url: str, work: Path) -> Path:
        output_template = str(work / "video.%(ext)s")
        height = self.config.max_height
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            # YouTube frequently returns HTTP 403 for the default web player client;
            # try the mobile/tv clients, which serve un-throttled progressive formats.
            "--extractor-args",
            "youtube:player_client=android,ios,tv,web_safari,web",
            "-f",
            f"best[height<={height}][ext=mp4]/best[height<={height}]/best",
            "-o",
            output_template,
            url,
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.config.download_timeout_seconds,
            check=False,
        )
        downloads = sorted(work.glob("video.*"))
        if not downloads:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise RuntimeError(f"video download failed: {detail or 'no file produced'}")
        return downloads[0]

    def _extract_frames(
        self,
        video_path: Path,
        start: float,
        end: float | None,
        interval: float | None,
        work: Path,
    ) -> list[tuple[float, bytes]]:
        span = None if end is None else max(0.0, end - start)
        if interval is None or interval <= 0:
            if span:
                interval = max(
                    self.config.default_interval_seconds, span / self.config.max_frames
                )
            else:
                interval = self.config.default_interval_seconds
        frames_dir = work / "frames"
        frames_dir.mkdir(exist_ok=True)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if start > 0:
            command += ["-ss", f"{start}"]
        if end is not None:
            command += ["-to", f"{end}"]
        command += [
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval},scale=-2:{self.config.max_height}",
            "-frames:v",
            str(self.config.max_frames),
            "-q:v",
            "3",
            str(frames_dir / "frame_%04d.jpg"),
        ]
        subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        frames: list[tuple[float, bytes]] = []
        for index, path in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
            timestamp = start + index * interval
            frames.append((timestamp, path.read_bytes()))
        return frames

    def _ask_model(self, question: str, frames: list[tuple[float, bytes]]) -> str:
        intro = (
            f"Question about the video: {question}\n\n"
            f"You are given {len(frames)} frames sampled in chronological order. "
            "Examine each frame carefully. If the question asks for a maximum "
            "'simultaneous' or 'at the same time' count, evaluate each frame "
            "independently and report the highest count seen in any single frame. "
            "Reason briefly, then end with a line exactly:\nANSWER: <answer>"
        )
        labeled = [
            (f"Frame at t={timestamp:.1f}s:", "image/jpeg", data)
            for timestamp, data in frames
        ]
        return _vision_answer(self.client, self.config.model, intro, labeled)


def _as_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probe_duration(video_path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    try:
        return float(completed.stdout.strip())
    except (TypeError, ValueError):
        return None


def _vision_answer(
    client: Any,
    model: str,
    intro: str,
    labeled_images: list[tuple[str, str, bytes]],
) -> str:
    """Send one or more labeled images to a multimodal model and return its answer."""
    content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
    for label, mime, data in labeled_images:
        encoded = base64.b64encode(data).decode("ascii")
        content.append({"type": "text", "text": label})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a meticulous visual analyst who answers only from the "
                "images provided and never invents details.",
            },
            {"role": "user", "content": content},
        ],
    )
    text = response.choices[0].message.content or ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("ANSWER:"):
            return stripped.split(":", 1)[1].strip()
    return text.strip()


_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@dataclass(frozen=True)
class ImageAnalysisConfig:
    """Settings for answering a question about an attached image via a vision model."""

    model: str
    use_gateway_headers: bool = False


class ImageAnalysisAdapter:
    """Sends an attached image to a vision model and returns an answer."""

    def __init__(self, client: Any, config: ImageAnalysisConfig) -> None:
        self.client = client
        self.config = config

    def call(self, arguments: dict[str, Any], files: tuple[Path, ...] = ()) -> str:
        file_arg = arguments.get("file")
        question = arguments.get("question")
        if not isinstance(file_arg, str) or not file_arg.strip():
            raise ValueError("analyze_image requires a non-empty file")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("analyze_image requires a non-empty question")
        wanted = Path(file_arg.strip()).name
        source: Path | None = None
        for path in files:
            if path.name == wanted and path.is_file():
                source = path
                break
        if source is None:
            available = ", ".join(sorted(p.name for p in files)) or "(none)"
            return json.dumps(
                {
                    "status": "error",
                    "answer": None,
                    "error": f"attached image {wanted!r} not found. Available files: {available}",
                }
            )
        mime = _IMAGE_MIME.get(source.suffix.lower(), "image/png")
        intro = (
            f"Question about the attached image: {question.strip()}\n\n"
            "Examine the image carefully and answer only from what is visible. If the "
            "question asks you to extract, transcribe, read out, OCR, or otherwise provide "
            "the TEXT or CODE shown in the image, reproduce it COMPLETELY and VERBATIM - "
            "character for character, preserving every symbol, operator, quote, number, "
            "indentation and line break exactly as shown - as the answer. Do NOT summarize, "
            "paraphrase, describe it, or merely say you have done it; output the actual "
            "literal text. If it is a "
            "chess position, first look at the rank (1-8) and file (a-h) labels printed on "
            "the board's edges and use them as the ground truth for coordinates: the board "
            "may be drawn from BLACK's point of view (ranks running 1 at the top down to 8 "
            "at the bottom, files h..a left to right), so do NOT assume White is at the "
            "bottom. Read each occupied square, note its labelled coordinate and the piece's "
            "colour (light piece = White, dark piece = Black), and if asked for the FEN give "
            "the standard FEN for those exact labelled coordinates plus the side to move. "
            "Reason briefly, then end with a line exactly:\nANSWER: <answer>"
        )
        try:
            answer = _vision_answer(
                self.client,
                self.config.model,
                intro,
                [("Attached image:", mime, source.read_bytes())],
            )
        except Exception as exc:  # vision call is best-effort
            return json.dumps(
                {
                    "status": "error",
                    "answer": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return json.dumps({"status": "success", "answer": answer}, ensure_ascii=False)


# The faster-whisper model is expensive to construct (loads weights, may download
# on first use), so cache one instance per (size, device, compute_type) for the
# lifetime of the process.
_WHISPER_MODELS: dict[tuple[str, str, str], Any] = {}


def _load_whisper_model(size: str, device: str, compute_type: str) -> Any:
    key = (size, device, compute_type)
    model = _WHISPER_MODELS.get(key)
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(size, device=device, compute_type=compute_type)
        _WHISPER_MODELS[key] = model
    return model


class AudioAnalysisAdapter:
    """Transcribes an audio file/URL with faster-whisper and answers a question."""

    def __init__(self, client: Any, config: AudioTranscriptionConfig) -> None:
        self.client = client
        self.config = config

    def call(self, arguments: dict[str, Any], files: tuple[Path, ...] = ()) -> str:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("analyze_audio requires a non-empty question")
        file_arg = arguments.get("file")
        url = arguments.get("url")
        if not (isinstance(file_arg, str) and file_arg.strip()) and not (
            isinstance(url, str) and url.strip()
        ):
            raise ValueError("analyze_audio requires either 'file' or 'url'")

        with tempfile.TemporaryDirectory(prefix="gaia-audio-") as temp_dir:
            work = Path(temp_dir)
            try:
                source = self._resolve_source(file_arg, url, files, work)
            except FileNotFoundError as exc:
                return json.dumps(
                    {"status": "error", "answer": None, "error": str(exc)}
                )
            try:
                transcript = self._transcribe(source)
            except Exception as exc:  # transcription is best-effort
                return json.dumps(
                    {
                        "status": "error",
                        "answer": None,
                        "error": f"transcription failed: {type(exc).__name__}: {exc}",
                    }
                )

        transcript = transcript.strip()
        if not transcript:
            return json.dumps(
                {
                    "status": "error",
                    "answer": None,
                    "error": "transcription produced no text (silent or unsupported audio).",
                }
            )
        answer = self._answer_from_transcript(question.strip(), transcript)
        return json.dumps(
            {"status": "success", "answer": answer, "transcript": transcript},
            ensure_ascii=False,
        )

    def _resolve_source(
        self,
        file_arg: str | None,
        url: str | None,
        files: tuple[Path, ...],
        work: Path,
    ) -> Path:
        if isinstance(file_arg, str) and file_arg.strip():
            wanted = Path(file_arg.strip()).name
            for path in files:
                if path.name == wanted and path.is_file():
                    return path
            available = ", ".join(sorted(p.name for p in files)) or "(none)"
            raise FileNotFoundError(
                f"attached audio file {wanted!r} not found. Available files: {available}"
            )
        # Otherwise download the URL (supports direct media and video hosts).
        return self._download(str(url).strip(), work)

    def _download(self, url: str, work: Path) -> Path:
        output_template = str(work / "audio.%(ext)s")
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-f",
            "bestaudio/best",
            "-o",
            output_template,
            url,
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.config.download_timeout_seconds,
            check=False,
        )
        downloads = sorted(work.glob("audio.*"))
        if not downloads:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise FileNotFoundError(
                f"audio download failed: {detail or 'no file produced'}"
            )
        return downloads[0]

    def _transcribe(self, source: Path) -> str:
        model = _load_whisper_model(
            self.config.whisper_model_size,
            self.config.device,
            self.config.compute_type,
        )
        segments, _info = model.transcribe(str(source))
        return " ".join(segment.text.strip() for segment in segments)

    def _answer_from_transcript(self, question: str, transcript: str) -> str:
        intro = (
            "You are given a verbatim transcript of an audio recording. Answer the "
            "question using ONLY the transcript; do not invent details. Reason "
            "briefly, then end with a line exactly:\nANSWER: <answer>\n\n"
            f"Question: {question}\n\nTranscript:\n{transcript}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract precise answers from audio transcripts.",
                    },
                    {"role": "user", "content": intro},
                ],
            )
        except Exception:
            # If the follow-up model call fails, the transcript alone is still useful.
            return ""
        text = response.choices[0].message.content or ""
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped.upper().startswith("ANSWER:"):
                return stripped.split(":", 1)[1].strip()
        return text.strip()


@dataclass(frozen=True)
class WebVoyagerConfig:
    root: Path
    output_root: Path
    python_executable: str = sys.executable
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    use_gateway_headers: bool = False
    max_iterations: int = 15
    max_attached_imgs: int = 3
    timeout_seconds: int = 600
    headless: bool = True
    text_only: bool = False


class WebVoyagerAdapter:
    def __init__(self, config: WebVoyagerConfig) -> None:
        self.config = config

    def call(self, goal: str, start_url: str = "https://duckduckgo.com/") -> str:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        invocation_root = self.config.output_root / f"call-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="gaia-web-task-") as temporary_dir:
            task_file = Path(temporary_dir) / "task.jsonl"
            download_dir = Path(temporary_dir) / "downloads"
            download_dir.mkdir()
            task_file.write_text(
                json.dumps(
                    {
                        "id": "planner-web",
                        "ques": goal,
                        "web": start_url or "https://duckduckgo.com/",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            command = [
                self.config.python_executable,
                str(self.config.root / "run.py"),
                "--test_file",
                str(task_file),
                "--output_dir",
                str(invocation_root),
                "--download_dir",
                str(download_dir),
                "--api_model",
                self.config.model,
                "--api_base_url",
                self.config.base_url,
                "--max_iter",
                str(self.config.max_iterations),
                "--max_attached_imgs",
                str(self.config.max_attached_imgs),
            ]
            if self.config.headless:
                command.append("--headless")
            if self.config.text_only:
                command.append("--text_only")

            env = os.environ.copy()
            if self.config.use_gateway_headers:
                env["LLM_GATEWAY_KEY"] = self.config.api_key
                env["LLM_GATEWAY_BASE_URL"] = self.config.base_url
                env["LLM_GATEWAY_MODEL"] = self.config.model
            elif self.config.api_key:
                env["OPENAI_API_KEY"] = self.config.api_key
            completed = subprocess.run(
                command,
                cwd=self.config.root,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )

        result_paths = sorted(
            invocation_root.rglob("taskplanner-web/result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not result_paths:
            return json.dumps(
                {
                    "status": "error",
                    "answer": None,
                    "error": _subprocess_error(completed),
                }
            )
        result_path = result_paths[0]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifact_dir"] = str(result_path.parent)
        if completed.returncode != 0:
            result["subprocess_error"] = _subprocess_error(completed)
        return json.dumps(result, ensure_ascii=False)


@dataclass(frozen=True)
class SandboxResources:
    """Resource limits applied to every ``execute_code`` sandbox call.

    The OpenAI tool schema does not expose resource limits, so the planner
    forces these values instead of the executor defaults (256 MB / 10 s CPU),
    which are too low for libraries such as pandas, opencv, and scikit-image.
    """

    memory_mb: int = 1024
    cpu_seconds: int = 60
    wall_seconds: int = 120
    max_pids: int = 128


class _ResourceOverrideExecutor:
    """Wraps an executor and forces resource limits on every request."""

    def __init__(self, wrapped: Any, resources: SandboxResources) -> None:
        self._wrapped = wrapped
        self._resources = resources

    def _apply(self, request: Any) -> Any:
        from nsjail_agent.models import ResourceLimits

        limits = ResourceLimits(
            max_cpu_seconds=self._resources.cpu_seconds,
            max_wall_seconds=self._resources.wall_seconds,
            max_memory_mb=self._resources.memory_mb,
            max_pids=self._resources.max_pids,
        )
        return request.model_copy(update={"resources": limits})

    def execute_sync(self, request: Any) -> Any:
        return self._wrapped.execute_sync(self._apply(request))

    def execute_sync_in_workspace(self, request: Any, workspace: Any) -> Any:
        return self._wrapped.execute_sync_in_workspace(self._apply(request), workspace)

    async def execute(self, request: Any) -> Any:
        return await self._wrapped.execute(self._apply(request))


# Each execute_code call runs in a fresh process, so in-memory state never
# persists between calls. Models frequently forget this and reuse `pd`/`np`
# from a previous call, wasting turns on NameError. Silently making these
# common aliases available for every Python call removes that failure mode
# without changing the semantics of the model's own code.
_PYTHON_IMPORT_PREAMBLE = (
    "try:\n"
    "    import pandas as pd\n"
    "    import numpy as np\n"
    "except Exception:\n"
    "    pass\n"
)


def _echo_last_expression(code: str) -> str:
    """Auto-display the last top-level expression, like a notebook cell.

    execute_code runs each payload as a plain script, so a trailing bare
    expression (e.g. ``os.listdir(d)`` or ``df.head()``) is evaluated but never
    shown. Models write notebook-style code expecting that value to appear, then
    proceed on empty output and hallucinate results. Rewriting the final bare
    expression into an explicit repr print restores the expected behaviour
    without touching the rest of the user's code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code
    # A bare string literal is almost always a trailing docstring/comment, and a
    # print(...) call already emits output - leave both alone to avoid noise.
    value = last.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return code
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "print"
    ):
        return code
    segment = ast.get_source_segment(code, last)
    if segment is None:
        return code
    lines = code.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: last.lineno - 1]) + last.col_offset
    prefix = code[:start]
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    wrapper = (
        "_gaia_last = (\n" + segment + "\n)\n"
        "if _gaia_last is not None:\n"
        "    print(repr(_gaia_last))\n"
    )
    return prefix + wrapper


def _prepare_execute_arguments(arguments: str | dict[str, Any]) -> str | dict[str, Any]:
    """Inject a safe import preamble into Python execute_code payloads."""
    if isinstance(arguments, str):
        try:
            parsed: Any = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    else:
        parsed = dict(arguments)
    if not isinstance(parsed, dict):
        return arguments
    code = parsed.get("code")
    language = parsed.get("language", "python")
    # `from __future__` imports must be the first statement, so never prepend
    # ahead of them; leave that code untouched.
    if language == "python" and isinstance(code, str) and "from __future__" not in code:
        parsed["code"] = _PYTHON_IMPORT_PREAMBLE + _echo_last_expression(code)
    return parsed


_MAX_FETCH_BYTES = 50 * 1024 * 1024  # 50 MB cap keeps a huge download from OOMing
_MAX_PROVISION_BYTES = (
    300 * 1024 * 1024
)  # engine/model binaries can be larger than a doc

# Egress allowlist for the self-provisioning tool. The agent may only download from
# these trusted hosts (suffix-matched on hostname); pip is inherently limited to the
# PyPI index. Extend at runtime via the GAIA_PROVISION_ALLOW_HOSTS env var (comma-sep).
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
    "release-assets.githubusercontent.com",
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "arxiv.org",
    "export.arxiv.org",
    "zenodo.org",
)


def _allowed_hosts() -> tuple[str, ...]:
    extra = os.environ.get("GAIA_PROVISION_ALLOW_HOSTS", "")
    hosts = [h.strip().lower() for h in extra.split(",") if h.strip()]
    return DEFAULT_ALLOWED_HOSTS + tuple(hosts)


def _host_allowed(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == allowed or host.endswith("." + allowed) for allowed in _allowed_hosts()
    )


def _fetch_filename(url: str, filename: str | None, content_type: str) -> str:
    """Choose a safe workspace filename for a downloaded URL."""
    from urllib.parse import urlparse

    if isinstance(filename, str) and filename.strip():
        name = Path(filename.strip()).name
    else:
        name = Path(urlparse(url).path).name or "download"
    if "." not in name:
        if "pdf" in content_type:
            name += ".pdf"
        elif "csv" in content_type:
            name += ".csv"
        elif "json" in content_type:
            name += ".json"
    return name


def _assert_public_http_url(url: str) -> None:
    """Reject anything that is not a public http(s) destination.

    ``fetch_url`` takes a model-supplied URL and runs it on the *host*, outside
    the sandbox, so it must not become a way to read local files or reach the
    private network. Without this check ``file:///etc/passwd`` is served by
    urllib's own FileHandler, and ``http://169.254.169.254/`` reaches the cloud
    instance-metadata service; either lands in the workspace and then in the
    model's context. Unlike ``provision`` we cannot use a host allowlist here -
    fetching arbitrary public pages is the point of the tool - so we constrain
    the scheme and the resolved address family instead.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PermissionError(
            f"fetch_url only supports http/https, got {parsed.scheme or 'no'} "
            f"scheme in {url!r}"
        )
    host = parsed.hostname
    if not host:
        raise PermissionError(f"fetch_url could not parse a hostname from {url!r}")

    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise PermissionError(f"cannot resolve {host!r}: {exc}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        # Covers loopback, RFC1918, link-local (incl. 169.254.169.254), CGNAT,
        # multicast and every other non-internet range in one check.
        if not address.is_global:
            raise PermissionError(
                f"fetch_url refuses non-public address {address} for host "
                f"{host!r} (loopback/private/link-local destinations are blocked)"
            )


class _PublicOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop: a public URL can 302 to a private one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_to_workspace(
    url: str, filename: str | None, workspace: Any
) -> tuple[str, int, str]:
    """Download *url* (host-side) into *workspace*; return (filename, bytes, content_type)."""
    _assert_public_http_url(url)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        },
    )
    # Build an opener with only the handlers this tool needs, so urllib's
    # default file:// and ftp:// handlers are not reachable at all.
    opener = urllib.request.build_opener(
        urllib.request.HTTPHandler,
        urllib.request.HTTPSHandler,
        _PublicOnlyRedirectHandler,
        urllib.request.HTTPErrorProcessor,
    )
    with opener.open(request, timeout=60) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        data = response.read(_MAX_FETCH_BYTES + 1)
    if len(data) > _MAX_FETCH_BYTES:
        raise ValueError(
            f"file exceeds {_MAX_FETCH_BYTES // (1024 * 1024)} MB download cap"
        )
    name = _fetch_filename(url, filename, content_type)
    workspace.write_binary_file(name, data)
    return name, len(data), content_type


def _provision_download(
    url: str, filename: str | None, workspace: Any, *, executable: bool = False
) -> tuple[str, int]:
    """Download an allowlisted *url* into *workspace*, optionally marking it executable."""
    import urllib.request

    if not _host_allowed(url):
        raise PermissionError(
            f"host not in provision allowlist: {url!r}. Allowed hosts: "
            + ", ".join(_allowed_hosts())
        )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(
        request, timeout=120
    ) as response:  # noqa: S310 (allowlisted)
        content_type = (response.headers.get("Content-Type") or "").lower()
        data = response.read(_MAX_PROVISION_BYTES + 1)
    if len(data) > _MAX_PROVISION_BYTES:
        raise ValueError(
            f"file exceeds {_MAX_PROVISION_BYTES // (1024 * 1024)} MB download cap"
        )
    name = _fetch_filename(url, filename, content_type)
    dest = workspace.write_binary_file(name, data)
    if executable:
        os.chmod(dest, 0o755)
    return name, len(data)


def _pip_install(packages: list[str], target: Path) -> tuple[bool, str]:
    """Install *packages* into *target* so execute_code can import them.

    Installs with ``pip --target`` into a per-task directory (owned by the calling
    ExecutionAdapter) instead of a shared user-site. The adapter bind-mounts that
    directory into the sandbox and puts it on PYTHONPATH, so packages provisioned
    for one task never leak into later tasks or later CLI runs. Runs on the host
    (the sandbox has no network); pip only ever contacts the PyPI index.
    """
    # The package list comes from the model, and pip parses any leading-dash
    # entry as a flag - `["--index-url", "https://evil/simple", "numpy"]` would
    # silently retarget the index and run that index's setup.py on the host.
    # Requirement specifiers never start with a dash, so reject those outright.
    bad = [p for p in packages if p.startswith("-")]
    if bad:
        return False, f"refusing pip option(s) in package list: {', '.join(bad)}"

    python_bin = os.environ.get("NSJAIL_PYTHON_BIN", "/usr/bin/python3")
    cmd = [
        python_bin,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "--prefer-binary",
        "--no-input",
        "--disable-pip-version-check",
        # Everything after this is a positional requirement, never an option.
        "--",
        *packages,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return proc.returncode == 0, tail


# A NameError almost always means the model referenced a variable/import created
# in a *previous* execute_code call, which is gone because the sandbox is
# stateless. Detecting it lets us hand the model a precise correction exactly
# when it makes the mistake — far more effective than upfront prose alone.
_NAME_ERROR_RE = re.compile(r"NameError: name '([^']+)' is not defined")


def _stateless_recovery_hint(output: str) -> str:
    """Return a corrective note if *output* shows a lost-cross-call-state NameError."""
    match = _NAME_ERROR_RE.search(output)
    if match is None:
        return ""
    name = match.group(1)
    return (
        f"\n\n[sandbox reminder] '{name}' is undefined because this sandbox is "
        "STATELESS: each execute_code call runs in a fresh process, so variables, "
        "DataFrames, and imports from earlier calls do not carry over. Re-write this "
        "as ONE self-contained script that re-imports modules, re-reads the attached "
        f"file(s), and re-creates '{name}' before using it, then print() the result."
    )


# Path *inside* the sandbox where a task's provisioned pip packages are mounted
# and added to PYTHONPATH. Kept distinct from the workspace cwd (see configs/nsjail_sandbox.json).


class ExecutionAdapter(AbstractContextManager["ExecutionAdapter"]):
    """One persistent sandbox workspace for a planner task."""

    def __init__(
        self,
        task: GaiaTask,
        *,
        allow_local_executor: bool = False,
        executor: Any | None = None,
        resources: SandboxResources | None = None,
    ) -> None:
        self.task = task
        self.allow_local_executor = allow_local_executor
        self._executor = executor
        self._resources = resources
        self._workspace: Any | None = None
        self._tool_adapter: Any | None = None
        self.schema: dict[str, Any] | None = None
        self._active = False
        # Per-task pip target dir + saved env, so provisioned packages are
        # isolated and removed when the task ends (no cross-task leakage).
        self._deps_dir: Path | None = None
        self._env_backup: dict[str, str | None] = {}

    def __enter__(self) -> ExecutionAdapter:
        from nsjail_agent.tool_adapter import OPENAI_TOOL_SCHEMA

        self.schema = OPENAI_TOOL_SCHEMA
        self._active = True
        return self

    def _ensure_deps_dir(self) -> Path:
        """Create (once) this task's private pip target dir and mount it read-only.

        The directory is bind-mounted into the sandbox at the configured deps mount
        (``configs/nsjail_sandbox.json``) and exposed on ``PYTHONPATH`` via
        ``NSJAIL_SANDBOX_PYTHONPATH``. Because
        each ExecutionAdapter owns and tears down its own dir, packages a task
        provisions are invisible to other tasks and to later CLI runs. The
        planner runs tasks sequentially, so mutating these process-level env vars
        for the duration of the task is safe.
        """
        if self._deps_dir is not None:
            return self._deps_dir
        from nsjail_agent.config import get_sandbox_paths

        deps_mount = get_sandbox_paths().deps_mount
        deps_dir = Path(tempfile.mkdtemp(prefix="gaia-task-deps-"))
        # Save current values so __exit__ restores the environment exactly.
        for key in ("NSJAIL_EXTRA_RO_MOUNTS", "NSJAIL_SANDBOX_PYTHONPATH"):
            self._env_backup[key] = os.environ.get(key)
        existing = [
            m
            for m in os.environ.get("NSJAIL_EXTRA_RO_MOUNTS", "").split(",")
            if m.strip()
        ]
        existing.append(f"{deps_dir}:{deps_mount}")
        os.environ["NSJAIL_EXTRA_RO_MOUNTS"] = ",".join(existing)
        os.environ["NSJAIL_SANDBOX_PYTHONPATH"] = deps_mount
        self._deps_dir = deps_dir
        return deps_dir

    def call(self, arguments: str | dict[str, Any]) -> str:
        if not self._active:
            raise RuntimeError("ExecutionAdapter must be used as a context manager")
        if self._tool_adapter is None:
            self._initialize_executor()
        output = self._tool_adapter(_prepare_execute_arguments(arguments))
        return output + _stateless_recovery_hint(output)

    def fetch(self, arguments: str | dict[str, Any]) -> str:
        """Download a URL into the sandbox workspace so execute_code can read it.

        The sandbox has no network, so files that live on the web (e.g. an
        academic PDF holding the answer) must be pulled host-side and dropped
        into the shared workspace. Returns the saved filename; the model then
        opens it with a normal execute_code call.
        """
        if not self._active:
            raise RuntimeError("ExecutionAdapter must be used as a context manager")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return json.dumps(
                    {"status": "error", "error": "fetch_url arguments must be JSON"}
                )
        if not isinstance(arguments, dict):
            return json.dumps(
                {"status": "error", "error": "fetch_url arguments must be an object"}
            )
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            return json.dumps(
                {"status": "error", "error": "fetch_url requires a non-empty url"}
            )
        if self._tool_adapter is None:
            self._initialize_executor()
        filename = arguments.get("filename")
        try:
            saved, size, content_type = _download_to_workspace(
                url.strip(), filename, self._workspace
            )
        except Exception as exc:  # network/host failure - report, do not crash the task
            return json.dumps(
                {
                    "status": "error",
                    "error": f"download failed: {type(exc).__name__}: {exc}",
                }
            )
        return json.dumps(
            {
                "status": "success",
                "filename": saved,
                "bytes": size,
                "content_type": content_type,
                "note": "Saved to the workspace. Open it in execute_code (e.g. pypdf/"
                "pdfminer for PDF, pandas/openpyxl for spreadsheets).",
            }
        )

    def provision(self, arguments: str | dict[str, Any]) -> str:
        """Install pip packages and/or download files the task needs into the workspace.

        Lets the agent set up its own capabilities (libraries, data, even a CLI/engine
        binary) instead of us pre-building every one. pip packages are installed for the
        sandbox interpreter (importable in execute_code); downloads are fetched from an
        allowlist of trusted hosts into the shared workspace.
        """
        if not self._active:
            raise RuntimeError("ExecutionAdapter must be used as a context manager")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return json.dumps(
                    {"status": "error", "error": "provision arguments must be JSON"}
                )
        if not isinstance(arguments, dict):
            return json.dumps(
                {"status": "error", "error": "provision arguments must be an object"}
            )

        pip = arguments.get("pip") or []
        downloads = arguments.get("downloads") or []
        if not isinstance(pip, list) or not all(isinstance(p, str) for p in pip):
            return json.dumps(
                {"status": "error", "error": "'pip' must be a list of strings"}
            )
        if not isinstance(downloads, list):
            return json.dumps(
                {"status": "error", "error": "'downloads' must be a list of objects"}
            )
        if not pip and not downloads:
            return json.dumps(
                {
                    "status": "error",
                    "error": "provision requires 'pip' and/or 'downloads'",
                }
            )

        result: dict[str, Any] = {}
        all_ok = True

        packages = [p.strip() for p in pip if p.strip()]
        if packages:
            ok, log = _pip_install(packages, self._ensure_deps_dir())
            all_ok = all_ok and ok
            result["pip"] = {"packages": packages, "ok": ok, "log": log}

        if downloads:
            if self._tool_adapter is None:
                self._initialize_executor()
            saved: list[dict[str, Any]] = []
            for entry in downloads:
                if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
                    saved.append({"error": "each download needs a string 'url'"})
                    all_ok = False
                    continue
                url = entry["url"].strip()
                try:
                    name, size = _provision_download(
                        url,
                        entry.get("filename"),
                        self._workspace,
                        executable=bool(entry.get("executable")),
                    )
                    saved.append(
                        {
                            "filename": name,
                            "bytes": size,
                            "executable": bool(entry.get("executable")),
                        }
                    )
                except (
                    Exception
                ) as exc:  # allowlist/network failure - report, keep going
                    saved.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                    all_ok = False
            result["downloads"] = saved

        result["status"] = "success" if all_ok else "partial"
        result["note"] = (
            "Installed pip packages are importable in execute_code. Downloaded files are "
            "in the current working directory - open them directly, or run an executable "
            "one as ./<filename>."
        )
        return json.dumps(result)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._workspace is not None:
            self._workspace.__exit__(exc_type, exc, traceback)
        self._workspace = None
        self._tool_adapter = None
        self._active = False
        # Restore the env vars we mutated for this task's provisioned deps, then
        # delete the private package dir so nothing carries into the next task.
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._env_backup = {}
        if self._deps_dir is not None:
            shutil.rmtree(self._deps_dir, ignore_errors=True)
            self._deps_dir = None

    def _initialize_executor(self) -> None:
        from nsjail_agent.agent import WorkspaceExecutor
        from nsjail_agent.local_executor import LocalExecutor, make_executor
        from nsjail_agent.tool_adapter import OpenAIToolAdapter
        from nsjail_agent.workspace import Workspace

        executor = self._executor or make_executor(prefer_sandbox=True)
        if isinstance(executor, LocalExecutor) and not self.allow_local_executor:
            raise RuntimeError(
                "nsjail is unavailable and unsafe LocalExecutor fallback is disabled. "
                "Install/configure nsjail or pass --allow-local-executor for development."
            )
        if self._resources is not None:
            executor = _ResourceOverrideExecutor(executor, self._resources)
        workspace = Workspace()
        workspace.__enter__()
        try:
            _install_toolkit(workspace)
            for path in self.task.files:
                if not path.is_file():
                    raise FileNotFoundError(f"Task attachment not found: {path}")
                workspace.write_binary_file(path.name, path.read_bytes())
            self._workspace = workspace
            self._tool_adapter = OpenAIToolAdapter(
                executor=WorkspaceExecutor(executor, workspace),
            )
        except Exception:
            workspace.__exit__(*sys.exc_info())
            raise


def _install_toolkit(workspace: Any) -> None:
    """Copy the general extraction toolkit into the workspace as gaia_toolkit.py.

    The workspace is the sandbox cwd (on sys.path), so writing the module here
    makes ``import gaia_toolkit`` work in execute_code with no mount changes.
    Best-effort: a failure to stage the helper must not break execution.
    """
    try:
        source = Path(__file__).with_name("sandbox_toolkit.py").read_bytes()
        workspace.write_binary_file("gaia_toolkit.py", source)
    except Exception:
        pass


def _subprocess_error(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "").strip()
    if len(detail) > 4000:
        detail = detail[-4000:]
    return f"WebVoyager exited with code {completed.returncode}: {detail}"
