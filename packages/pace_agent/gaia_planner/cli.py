# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Command-line benchmark runner for local GAIA task files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .adapters import (
    AudioAnalysisAdapter,
    AudioTranscriptionConfig,
    ExecutionAdapter,
    ImageAnalysisAdapter,
    ImageAnalysisConfig,
    SandboxResources,
    VideoAnalysisAdapter,
    VideoAnalysisConfig,
    WebVoyagerAdapter,
    WebVoyagerConfig,
)
from .configs import parse_args
from nsjail_agent.config import get_sandbox_paths
from .datasets import answer_verdicts, load_gaia_from_hf, load_tasks
from .models import GaiaTask, PlannerConfig
from .planner import GaiaPlanner


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    get_sandbox_paths()
    # nsjail_agent is vendored next to gaia_planner and importable without a checkout;
    # --nsjail-root is retained only for backward-compatible CLI scripts.
    _ = args.nsjail_root
    _mount_sandbox_user_site()
    _mount_stockfish()

    use_hf = args.huggingface or args.level != "all"
    if use_hf:
        level = None if args.level == "all" else int(args.level)
        try:
            all_tasks = load_gaia_from_hf(
                level=level,
                split=args.split,
                config=args.hf_config,
                token=args.hf_token,
            )
        except Exception as exc:
            print(f"Failed to load GAIA from HuggingFace: {exc}", file=sys.stderr)
            return 2
    elif args.task_file is not None:
        all_tasks = load_tasks(args.task_file)
    else:
        print("Provide a task_file or use --huggingface/--level.", file=sys.stderr)
        return 2

    tasks = _select_tasks(all_tasks, args.task_id, args.max_tasks, source=args.source)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2
    if not args.api_key:
        print("Set OPENAI_API_KEY/LLM_GATEWAY_KEY or pass --api-key.", file=sys.stderr)
        return 2

    run_dir = args.output_dir.resolve() / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    client = _build_client(args.api_key, args.base_url)
    web_adapter = WebVoyagerAdapter(
        WebVoyagerConfig(
            root=args.webvoyager_root.resolve(),
            output_root=run_dir / "web",
            python_executable=args.webvoyager_python,
            model=args.web_model or args.model,
            base_url=args.web_base_url or args.base_url,
            api_key=args.web_api_key if args.web_api_key is not None else args.api_key,
            use_gateway_headers=bool(os.getenv("LLM_GATEWAY_KEY")),
            max_iterations=args.web_max_iterations,
            max_attached_imgs=args.web_max_attached_imgs,
            timeout_seconds=args.web_timeout,
            headless=args.web_headless,
            text_only=args.web_text_only,
        )
    )
    sandbox_resources = SandboxResources(
        memory_mb=args.sandbox_memory_mb,
        cpu_seconds=args.sandbox_cpu_seconds,
        wall_seconds=args.sandbox_wall_seconds,
    )
    # Vision/audio tools can run on a separate locally hosted (e.g. Radeon vLLM) endpoint so
    # the planner stays on the gateway while multimodal work runs on local GPUs. When
    # --local-base-url is unset, the tools share the planner client (unchanged behavior).
    use_local_tools = bool(args.local_base_url)
    tools_client = (
        _build_local_client(args.local_api_key, args.local_base_url)
        if use_local_tools
        else client
    )
    tools_use_gateway_headers = (
        bool(os.getenv("LLM_GATEWAY_KEY")) and not use_local_tools
    )
    # Vision model resolution: explicit per-modality flag > GAIA_VISION_MODEL env > (local
    # endpoint) the single model the endpoint serves > planner --model. When the tools run on a
    # local endpoint the planner --model (a gateway model) is not a valid fallback, so we try to
    # auto-detect the served model and only then fall back to the explicit-model requirement.
    vision_default = os.getenv("GAIA_VISION_MODEL")
    if not vision_default and use_local_tools:
        vision_default = _discover_local_model(args.local_api_key, args.local_base_url)
    if not vision_default and not use_local_tools:
        vision_default = args.model
    video_model = args.video_model or vision_default
    image_model = args.image_model or vision_default
    audio_model = args.audio_model or vision_default
    if use_local_tools:
        missing = [
            flag
            for flag, value in (
                ("--video-model", video_model),
                ("--image-model", image_model),
                ("--audio-model", audio_model),
            )
            if not value
        ]
        if missing:
            print(
                "When --local-base-url is set, a model must be given for the local tools via "
                f"{', '.join(missing)} (or GAIA_VISION_MODEL), and it could not be "
                "auto-detected from the endpoint's /v1/models.",
                file=sys.stderr,
            )
            return 2
    if use_local_tools:
        print(
            f"Vision/audio tools -> {args.local_base_url} "
            f"(video={video_model}, image={image_model}, audio={audio_model})"
        )
    else:
        print("Vision/audio tools -> gateway (shared planner client)")
    video_adapter = VideoAnalysisAdapter(
        client=tools_client,
        config=VideoAnalysisConfig(
            model=video_model,
            max_frames=args.video_max_frames,
            max_height=args.video_max_height,
            download_timeout_seconds=args.video_download_timeout,
            use_gateway_headers=tools_use_gateway_headers,
        ),
    )
    audio_adapter = AudioAnalysisAdapter(
        client=tools_client,
        config=AudioTranscriptionConfig(
            model=audio_model,
            whisper_model_size=args.audio_whisper_size,
            download_timeout_seconds=args.audio_download_timeout,
            use_gateway_headers=tools_use_gateway_headers,
        ),
    )
    image_adapter = ImageAnalysisAdapter(
        client=tools_client,
        config=ImageAnalysisConfig(
            model=image_model,
            use_gateway_headers=tools_use_gateway_headers,
        ),
    )
    if args.engine == "langgraph":
        from .langgraph_planner import LangGraphGaiaPlanner as PlannerCls
    else:
        PlannerCls = GaiaPlanner
    planner = PlannerCls(
        client=client,
        web_adapter=web_adapter,
        execution_factory=lambda task: ExecutionAdapter(
            task,
            allow_local_executor=args.allow_local_executor,
            resources=sandbox_resources,
        ),
        config=PlannerConfig(
            model=args.model, max_turns=args.max_turns, no_tools=args.no_tools
        ),
        video_adapter=video_adapter,
        audio_adapter=audio_adapter,
        image_adapter=image_adapter,
    )

    result_path = run_dir / "results.jsonl"
    records: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as output:
        for task in tasks:
            record = _run_task(planner, task)
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            level = record.get("level")
            prefix = f"[L{level}] " if level else ""
            print(f"{prefix}{task.task_id}: {record.get('answer', '[error]')}")
            sequence = record.get("agent_sequence") or []
            order = " -> ".join(sequence) if sequence else "(no agents invoked)"
            print(f"    agents: {order}")

    summary = _summary(records)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {result_path}")
    if summary["scored"]:
        print(
            f"Local accuracy: strict {summary['correct']}/{summary['scored']}, "
            f"loose {summary['correct_loose']}/{summary['scored']}"
        )
        for level, bucket in summary["by_level"].items():
            if bucket["scored"]:
                print(f"  level {level}: {bucket['correct']}/{bucket['scored']}")
        if summary["unscored"]:
            print(
                f"  {summary['unscored']} task(s) shipped no reference answer "
                "and were left ungraded"
            )
    else:
        # Say so rather than printing nothing: an ungraded run must not look like a run whose
        # accuracy the reader simply missed.
        hint = (
            " — the test split publishes none, so use --split validation to grade a run"
            if getattr(args, "split", None) == "test"
            else ""
        )
        print(
            f"Not scored: none of the {summary['tasks']} task(s) shipped a "
            f"reference answer{hint}"
        )
    return 0


def _build_client(api_key: str, base_url: str) -> Any:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    gateway_key = os.getenv("LLM_GATEWAY_KEY")
    if gateway_key:
        kwargs["default_headers"] = {"Ocp-Apim-Subscription-Key": gateway_key}
    return OpenAI(**kwargs)


def _build_local_client(api_key: str, base_url: str) -> Any:
    """OpenAI-compatible client for locally hosted models (e.g. Radeon vLLM). Unlike
    ``_build_client`` it never attaches the gateway subscription header, so gateway-only auth
    is not leaked to the local endpoint. Used by the vision/audio tools when
    ``--local-base-url`` is set, keeping the planner on the gateway."""
    from openai import OpenAI

    return OpenAI(api_key=api_key or "none", base_url=base_url)


def _discover_local_model(api_key: str, base_url: str) -> str | None:
    """Best-effort: query an OpenAI-compatible endpoint's ``/v1/models`` and return the single
    served model id, so ``--local-base-url`` alone is enough when the server hosts exactly one
    model. Returns None on any failure or when the choice is ambiguous (0 or >1 models), in
    which case the caller falls back to requiring an explicit --video-model/--image-model.
    """
    try:
        client = _build_local_client(api_key, base_url)
        ids = [m.id for m in client.models.list().data]
    except Exception:  # noqa: BLE001 - discovery is optional; never abort a run over it
        return None
    return ids[0] if len(ids) == 1 else None


def _mount_sandbox_user_site() -> None:
    """Expose the sandbox interpreter's user-site packages inside nsjail.

    The sandbox runs /usr/bin/python3 (or NSJAIL_PYTHON_BIN) and mounts /usr, so
    system dist-packages work, but packages installed with `pip install --user`
    live under ~/.local and are invisible because /home is not mounted. We add
    that user-site directory to NSJAIL_EXTRA_RO_MOUNTS so tools like pytesseract
    and python-chess become importable in execute_code. Existing mounts are
    preserved.

    Reproduce the required user-site packages with (using the sandbox interpreter):
        /usr/bin/python3 -m pip install --user --break-system-packages \
            pytesseract Pillow chess beautifulsoup4 lxml python-docx python-pptx \
            pypdf pdfplumber openpyxl tabulate html2text
    These back the gaia_toolkit helper (staged into each workspace) so the agent can
    read/parse most attachments and pages without provisioning at run time.
    """
    sandbox_python = os.environ.get("NSJAIL_PYTHON_BIN", "/usr/bin/python3")
    try:
        result = subprocess.run(
            [sandbox_python, "-m", "site", "--user-site"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    user_site = result.stdout.strip()
    if not user_site or not Path(user_site).is_dir():
        return
    mounts = [
        m for m in os.environ.get("NSJAIL_EXTRA_RO_MOUNTS", "").split(",") if m.strip()
    ]
    if user_site not in mounts:
        mounts.append(user_site)
        os.environ["NSJAIL_EXTRA_RO_MOUNTS"] = ",".join(mounts)


def _mount_stockfish() -> None:
    """Bind-mount the Stockfish engine binary into the sandbox at a fixed path.

    The sandbox has no network, so `chess.engine` code in execute_code cannot
    download an engine. We expose a host-side static Stockfish binary read-only
    at the configured sandbox path so generated code can call
    ``chess.engine.SimpleEngine.popen_uci(...)``. The host binary is
    resolved from STOCKFISH_HOST_BIN or the bundled tools/stockfish/stockfish-bin.
    """
    stockfish_path = get_sandbox_paths().stockfish_path
    host_bin = os.environ.get("STOCKFISH_HOST_BIN")
    if host_bin:
        binary = Path(host_bin)
    else:
        repo_root = Path(__file__).resolve().parents[3]
        binary = repo_root / "tools" / "stockfish" / "stockfish-bin"
    if not binary.is_file():
        return
    mapping = f"{binary}:{stockfish_path}"
    mounts = [
        m for m in os.environ.get("NSJAIL_EXTRA_RO_MOUNTS", "").split(",") if m.strip()
    ]
    if mapping not in mounts:
        mounts.append(mapping)
        os.environ["NSJAIL_EXTRA_RO_MOUNTS"] = ",".join(mounts)


def _select_tasks(
    tasks: list[GaiaTask],
    task_id: str | None,
    max_tasks: int | None,
    *,
    source: str | None = None,
) -> list[GaiaTask]:
    selected = tasks
    if source:
        selected = [task for task in selected if task.metadata.get("source") == source]
    if task_id:
        selected = [task for task in selected if task.task_id == task_id]
    if len(selected) > 1 and task_id:
        print(
            f"Warning: task_id {task_id} matched {len(selected)} entries; "
            "use --source to disambiguate.",
            file=sys.stderr,
        )
    return selected[:max_tasks] if max_tasks is not None else selected


AGENT_LABELS = {
    "browse_web": "WebVoyager(web)",
    "execute_code": "nsjail(code)",
    "analyze_video": "Vision(video)",
    "analyze_audio": "Whisper(audio)",
    "analyze_image": "Vision(image)",
    "fetch_url": "Fetch(url)",
    "provision": "Provision(env)",
}


def _agent_sequence(tool_events: list[dict[str, Any]]) -> list[str]:
    return [
        AGENT_LABELS.get(event.get("name", ""), event.get("name", "unknown"))
        for event in tool_events
    ]


def _run_task(planner: GaiaPlanner, task: GaiaTask) -> dict[str, Any]:
    level = task.metadata.get("level")
    try:
        result = planner.run(task)
        record = result.to_record()
        verdict = answer_verdicts(result.answer, task.expected)
        record.update(
            {
                "question": task.question,
                "expected": task.expected,
                "level": level,
                "agent_sequence": _agent_sequence(result.tool_events),
                # `match` stays as match_strict for backward compatibility with older
                # results.jsonl consumers; both PACE Agent harness verdicts are also recorded.
                "match": verdict["match_strict"],
                "match_strict": verdict["match_strict"],
                "match_loose": verdict["match_loose"],
                "status": "completed",
            }
        )
        return record
    except Exception as exc:
        return {
            "task_id": task.task_id,
            "question": task.question,
            "expected": task.expected,
            "level": level,
            "agent_sequence": [],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def _strict(record: dict[str, Any]) -> bool | None:
        return record.get("match_strict", record.get("match"))

    scored = [record for record in records if _strict(record) is not None]
    by_level: dict[str, dict[str, int]] = {}
    for record in records:
        level = (
            str(record.get("level")) if record.get("level") is not None else "unknown"
        )
        strict = _strict(record)
        loose = record.get("match_loose")
        bucket = by_level.setdefault(
            level,
            {"tasks": 0, "scored": 0, "correct": 0, "correct_loose": 0},
        )
        bucket["tasks"] += 1
        if strict is not None:
            bucket["scored"] += 1
            if strict is True:
                bucket["correct"] += 1
            if loose is True:
                bucket["correct_loose"] += 1
    return {
        "tasks": len(records),
        "completed": sum(record.get("status") == "completed" for record in records),
        "errors": sum(record.get("status") == "error" for record in records),
        "scored": len(scored),
        "unscored": len(records) - len(scored),
        "correct": sum(_strict(r) is True for r in scored),
        "correct_loose": sum(r.get("match_loose") is True for r in scored),
        "by_level": dict(sorted(by_level.items())),
        "metric": (
            "pace_agent.score match_strict / match_loose "
            "(local references only; not the official GAIA metric)"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
