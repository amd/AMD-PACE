# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Config-driven EQUIP / StreamingLLM / AR compare for contiguous caches.

Measures wall time for one or more EQUIP configurations and compares them to
baselines (default: StreamingLLM SHIFT, and full-cache AR with DYNAMIC +
NATIVE attention). No paper tables — just the configs you pass in.

Usage:
    PYTHONPATH=. python benchmarks/llm/performance/benchmark_equip.py \\
        --config benchmarks/llm/performance/benchmark_equip_config.json

Sweep sinks, batches, or seq lengths by editing the JSON (``equip`` may be a
list). AR defaults to DYNAMIC + NATIVE; override via the ``ar`` block.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoTokenizer, BatchEncoding

from pace.llm import (
    EvictionConfig,
    KVCacheType,
    LLMBackendType,
    LLMModel,
    LLMOperatorType,
    OperatorConfig,
    RoPEMode,
    SamplingConfig,
    UpdateMode,
)
from pace.llm.attention import AttentionBackendType

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "benchmark_equip_config.json"

_DTYPE = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32}
_KV = {
    "DYNAMIC": KVCacheType.DYNAMIC,
    "BMC": KVCacheType.BMC,
    "dynamic": KVCacheType.DYNAMIC,
    "bmc": KVCacheType.BMC,
}


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _as_list(value: Union[dict, list]) -> List[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise TypeError(f"expected dict or list of dicts, got {type(value)}")


def _backend(value: str) -> LLMBackendType:
    return LLMBackendType(str(value).lower())


def _attn_backend(value: str) -> AttentionBackendType:
    return AttentionBackendType(str(value).lower())


def _opconfig(
    operators: dict, attention_override: Optional[str] = None
) -> OperatorConfig:
    ops = {
        LLMOperatorType.Norm: _backend(operators.get("Norm", "JIT")),
        LLMOperatorType.QKVProjection: _backend(operators.get("QKVProjection", "TPP")),
        LLMOperatorType.Attention: _attn_backend(operators.get("Attention", "JIT")),
        LLMOperatorType.OutProjection: _backend(operators.get("OutProjection", "TPP")),
        LLMOperatorType.MLP: _backend(operators.get("MLP", "TPP")),
        LLMOperatorType.LMHead: _backend(operators.get("LMHead", "TPP")),
    }
    if attention_override is not None:
        ops[LLMOperatorType.Attention] = _attn_backend(attention_override)
    return OperatorConfig(**ops)


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)

    model_args = cfg.setdefault("model_args", {})
    if "model_name" not in model_args:
        raise ValueError("model_args.model_name is required")
    model_args.setdefault("dtype", "bf16")
    model_args.setdefault(
        "llm_operators",
        {
            "Norm": "JIT",
            "QKVProjection": "TPP",
            "Attention": "JIT",
            "OutProjection": "TPP",
            "MLP": "TPP",
            "LMHead": "TPP",
        },
    )

    gen = cfg.setdefault("generation_args", {})
    gen.setdefault("prompt_len", 128)
    gen.setdefault("seqs", [1024])
    gen.setdefault("batch_sizes", [1])
    if not isinstance(gen["seqs"], list) or not gen["seqs"]:
        raise ValueError("generation_args.seqs must be a non-empty list")
    if not isinstance(gen["batch_sizes"], list) or not gen["batch_sizes"]:
        raise ValueError("generation_args.batch_sizes must be a non-empty list")

    if "equip" not in cfg:
        raise ValueError("config must define an 'equip' block (dict or list)")
    cfg.setdefault("baselines", ["streaming", "ar"])
    cfg.setdefault(
        "streaming",
        {
            "name": "streaming",
            "rope_mode": "REINDEX",
            "update": "SHIFT",
            "kv_cache_type": "DYNAMIC",
        },
    )
    cfg.setdefault(
        "ar",
        {"name": "ar", "kv_cache_type": "DYNAMIC", "attention": "NATIVE"},
    )
    cfg.setdefault("warmup_runs", 1)
    cfg.setdefault("num_runs", 2)
    cfg.setdefault("verbose", True)
    cfg.setdefault("output_dir", str(HERE / "benchmark_results"))
    return cfg


def _resolve_equip_variants(cfg: dict) -> List[dict]:
    variants = []
    for i, raw in enumerate(_as_list(cfg["equip"])):
        v = {
            "name": raw.get("name", "equip" if i == 0 else f"equip_{i}"),
            "kind": "equip",
            "sink_size": int(raw.get("sink_size", 4)),
            "recent_size": int(raw.get("recent_size", 752)),
            "rope_mode": str(raw.get("rope_mode", "SINK_ONLY")).lower(),
            "update": str(raw.get("update", "INPLACE")).lower(),
            "kv_cache_type": str(raw.get("kv_cache_type", "DYNAMIC")).upper(),
            "attention": raw.get("attention"),
        }
        if v["sink_size"] < 0 or v["recent_size"] <= 0:
            raise ValueError(f"invalid sink/recent in equip variant {v['name']}")
        variants.append(v)
    return variants


def _resolve_jobs(cfg: dict) -> List[dict]:
    """Ordered runs: each EQUIP variant, then enabled baselines."""
    jobs = _resolve_equip_variants(cfg)
    baselines = cfg.get("baselines") or []
    equip0 = jobs[0]

    if "streaming" in baselines:
        s = dict(cfg["streaming"])
        jobs.append(
            {
                "name": s.get("name", "streaming"),
                "kind": "streaming",
                "sink_size": int(s.get("sink_size", equip0["sink_size"])),
                "recent_size": int(s.get("recent_size", equip0["recent_size"])),
                "rope_mode": str(s.get("rope_mode", "REINDEX")).lower(),
                "update": str(s.get("update", "SHIFT")).lower(),
                "kv_cache_type": str(s.get("kv_cache_type", "DYNAMIC")).upper(),
                "attention": s.get("attention"),
            }
        )
    if "ar" in baselines:
        a = dict(cfg["ar"])
        jobs.append(
            {
                "name": a.get("name", "ar"),
                "kind": "ar",
                "kv_cache_type": str(a.get("kv_cache_type", "DYNAMIC")).upper(),
                "attention": str(a.get("attention", "NATIVE")),
            }
        )
    names = [j["name"] for j in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate job names: {names}")
    return jobs


def make_inputs(
    tokenizer, batch_size: int, prompt_len: int, seed: int = 0
) -> BatchEncoding:
    """Random vocab ids + dense mask, same idea as BenchMarkDataGenerator."""
    rng = random.Random(seed)
    vocab_ids = list(tokenizer.get_vocab().values())
    input_ids = torch.tensor(
        [rng.choices(vocab_ids, k=prompt_len) for _ in range(batch_size)],
        dtype=torch.long,
    )
    return BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
    )


def load_model(
    model_name: str, dtype: torch.dtype, operators: dict, job: dict
) -> LLMModel:
    opconfig = _opconfig(operators, attention_override=job.get("attention"))
    kv = _KV[job["kv_cache_type"]]
    if job["kind"] == "ar":
        return LLMModel(
            model_name,
            dtype=dtype,
            kv_cache_type=kv,
            opconfig=opconfig,
        )
    return LLMModel(
        model_name,
        dtype=dtype,
        kv_cache_type=kv,
        opconfig=opconfig,
        eviction_config=EvictionConfig(
            sink_size=job["sink_size"],
            recent_size=job["recent_size"],
            rope_mode=RoPEMode(job["rope_mode"]),
            update=UpdateMode(job["update"]),
        ),
    )


def time_generate(model: LLMModel, inputs: BatchEncoding, new_tokens: int) -> float:
    cfg = SamplingConfig(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        seed=0,
    )
    gc.collect()
    t0 = time.perf_counter()
    model.generate(inputs, cfg)
    return time.perf_counter() - t0


def run_job(
    model: LLMModel,
    tokenizer,
    seqs: Sequence[int],
    batches: Sequence[int],
    prompt_len: int,
    warmup: int,
    runs: int,
) -> Dict[int, Dict[int, dict]]:
    rows: Dict[int, Dict[int, dict]] = {}
    for bs in batches:
        rows[bs] = {}
        for seq in seqs:
            new_tokens = seq - prompt_len
            if new_tokens <= 0:
                raise ValueError(f"seq {seq} must exceed prompt_len {prompt_len}")
            inputs = make_inputs(tokenizer, bs, prompt_len)
            for _ in range(warmup):
                time_generate(model, inputs, new_tokens)
            times = [time_generate(model, inputs, new_tokens) for _ in range(runs)]
            mean_t = sum(times) / len(times)
            rows[bs][seq] = {
                "seconds": mean_t,
                "tok_s": (bs * new_tokens) / mean_t,
                "runs": times,
            }
            print(
                f"    bs={bs} seq={seq}: {mean_t:.3f}s  "
                f"{rows[bs][seq]['tok_s']:.2f} tok/s",
                flush=True,
            )
    return rows


def print_tables(
    results: dict,
    jobs: List[dict],
    seqs: Sequence[int],
    batches: Sequence[int],
) -> None:
    names = [j["name"] for j in jobs]
    equip_names = [j["name"] for j in jobs if j["kind"] == "equip"]
    baseline_names = [j["name"] for j in jobs if j["kind"] != "equip"]

    def grid(title: str, cell) -> None:
        print(f"\n{title}")
        print("        " + "".join(f"{s:>10}" for s in seqs))
        for bs in batches:
            print(f"BS={bs:<4}" + "".join(cell(bs, s) for s in seqs))

    for name in names:

        def sec(bs, seq, _n=name):
            return f"{results[_n][bs][seq]['seconds']:10.3f}"

        grid(f"{name} wall time (s)", sec)

    for equip_name in equip_names:
        for base_name in baseline_names:

            def spd(bs, seq, _e=equip_name, _b=base_name):
                return f"{results[_b][bs][seq]['seconds'] / results[_e][bs][seq]['seconds']:10.3f}"

            grid(f"{equip_name} speedup over {base_name}", spd)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare EQUIP configs against StreamingLLM and AR baselines.",
        epilog=(
            "Example:\n"
            "  PYTHONPATH=. python benchmarks/llm/performance/benchmark_equip.py "
            f"--config {DEFAULT_CONFIG}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--output",
        default=None,
        help="JSON results path (default: <output_dir>/equip_<timestamp>.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    jobs = _resolve_jobs(cfg)

    model_name = cfg["model_args"]["model_name"]
    dtype = _DTYPE[str(cfg["model_args"]["dtype"]).lower()]
    operators = cfg["model_args"]["llm_operators"]
    gen = cfg["generation_args"]
    prompt_len = int(gen["prompt_len"])
    seqs = [int(s) for s in gen["seqs"]]
    batches = [int(b) for b in gen["batch_sizes"]]
    warmup = int(cfg["warmup_runs"])
    runs = int(cfg["num_runs"])

    print(
        f"config={args.config}  model={model_name}  dtype={dtype}  "
        f"prompt={prompt_len}  seqs={seqs}  batches={batches}  "
        f"jobs={[j['name'] for j in jobs]}  cpu={cpu_name()}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    job_meta = {}
    for job in jobs:
        print(f"\n=== {job['name']} ({job['kind']}) ===", flush=True)
        if cfg.get("verbose"):
            print(
                f"    {json.dumps({k: v for k, v in job.items() if k != 'kind'})}",
                flush=True,
            )
        model = load_model(model_name, dtype, operators, job)
        results[job["name"]] = run_job(
            model, tokenizer, seqs, batches, prompt_len, warmup, runs
        )
        job_meta[job["name"]] = copy.deepcopy(job)
        del model
        gc.collect()

    print_tables(results, jobs, seqs, batches)

    out = {
        "config_path": str(args.config.resolve()),
        "model": model_name,
        "dtype": str(dtype),
        "cpu": cpu_name(),
        "prompt_len": prompt_len,
        "seqs": seqs,
        "batch_sizes": batches,
        "warmup_runs": warmup,
        "num_runs": runs,
        "jobs": job_meta,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output)
        if args.output
        else out_dir / (f"equip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
