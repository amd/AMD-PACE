# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from pace_agent.registry import SkipTool, list_tools
from pace_agent.stats import measure, speedup_ci, summarize

# Built-in default sizes for `pace-agent bench tools` when no --config is given.
# The same values ship as configs/bench_tools.json for users to copy and tweak.
DEFAULT_TOOLS = {
    "embed": {"n": 512, "dim": 256, "toklen": 32},
    "vector_search": {"N": 10000, "dim": 256, "top_k": 10},
    "bm25": {"N_docs": 2000, "doclen": 40, "qlen": 5, "top_k": 10},
    "rerank": {"n": 2000, "dim": 256, "top_k": 10},
    "quantize": {"size": 200000},
    "json_serialize": {"n": 2000},
    "json_deserialize": {"n": 2000},
    "tokenize": {"n": 4000},
    "html_parse": {"n": 1500},
    "ast_parse": {"n": 400},
    "regex_extract": {"n": 5000},
    "assemble": {"n": 3000},
    "chunk": {"n": 8000, "size": 128, "overlap": 16},
    "dedup": {"n": 3000},
    "summarize": {"n": 2000, "budget": 500},
    "pdf_text_extract": {"file": "sample.pdf", "pages": "1-1"},
}


def bench_tools(bench_config, tiers=("naive",), repeat=None, warmup=None):
    results = []
    for op in list_tools():
        spec = bench_config.tools.get(op.name)
        if spec is None:
            continue
        try:
            args, kw = op.bench_inputs(spec)
        except SkipTool as e:
            results.append(
                {"op": op.name, "bucket": op.bucket, "status": "skip", "reason": str(e)}
            )
            continue
        info = op.implementations()
        row = {
            "op": op.name,
            "bucket": op.bucket,
            "status": "ok",
            "tiers": {},
            "skipped": {},
        }
        fns = {}
        for tier in tiers:
            # a tier that only falls back to naive is reported as such, not timed twice
            if tier != "naive" and info.get(tier) and info[tier].is_fallback:
                row["tiers"][tier] = {"fallback": "naive"}
                continue
            try:
                op.run(
                    tier, *args, **kw
                )  # probe; SkipTool => provider column unavailable
            except SkipTool as e:
                row["skipped"][tier] = str(e)
                continue
            fns[tier] = lambda t=tier: op.run(t, *args, **kw)
        samples = measure(fns, samples=repeat, warmup=warmup)
        for tier, s in samples.items():
            row["tiers"][tier] = summarize(s)
        # naive is the reference; if it can't run the whole op is skipped
        if "naive" in row["skipped"] or "naive" not in row["tiers"]:
            reason = row["skipped"].get("naive") or next(
                iter(row["skipped"].values()), "no tiers ran"
            )
            results.append(
                {"op": op.name, "bucket": op.bucket, "status": "skip", "reason": reason}
            )
            continue
        if "optimized" in tiers:
            if info.get("optimized") and info["optimized"].is_fallback:
                row["speedup"], row["parity"] = (
                    None,
                    True,
                )  # same implementation as naive
            elif "optimized" in samples:
                base = row["tiers"]["naive"]["median"]
                opt = row["tiers"]["optimized"]["median"]
                row["speedup"] = base / opt if opt else None
                row["speedup_ci"] = speedup_ci(samples["naive"], samples["optimized"])
                row["parity"] = op.parity(args, kw)
        results.append(row)
    return results
