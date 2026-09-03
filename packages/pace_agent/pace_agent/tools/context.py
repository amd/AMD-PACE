# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from pace_agent.registry import PaceAgentTool, register_tool


@register_tool("assemble", "context")
class PaceAgentAssemble(PaceAgentTool):
    def naive(self, messages):
        return "".join(m["role"] + ": " + m["content"] + "\n" for m in messages)

    def bench_inputs(self, spec):
        n = spec.get("n", 3000)
        return (
            [{"role": "user", "content": f"message body {i}"} for i in range(n)],
        ), {}


@register_tool("chunk", "context")
class PaceAgentChunk(PaceAgentTool):
    def naive(self, text, size, overlap):
        toks = text.split()
        out, i = [], 0
        step = max(1, size - overlap)
        while i < len(toks):
            out.append(" ".join(toks[i : i + size]))
            i += step
        return out

    def bench_inputs(self, spec):
        return ("word " * spec.get("n", 8000),), {
            "size": spec.get("size", 128),
            "overlap": spec.get("overlap", 16),
        }


@register_tool("dedup", "context")
class PaceAgentDedup(PaceAgentTool):
    def naive(self, items):
        seen, out = [], []
        for it in items:
            if it not in seen:
                seen.append(it)
                out.append(it)
        return out

    def optimized(self, items):
        # Per-input fallback (not the lib-availability kind): the O(n) set path only works on
        # hashable items; unhashable inputs raise TypeError, so fall back to the naive scan.
        try:
            seen, out = set(), []
            for it in items:
                if it not in seen:
                    seen.add(it)
                    out.append(it)
            return out
        except TypeError:
            return self.naive(items)

    def bench_inputs(self, spec):
        n = spec.get("n", 3000)
        return ([f"line {i % (n // 3 or 1)}" for i in range(n)],), {}


@register_tool("summarize", "context")
class PaceAgentSummarize(PaceAgentTool):
    def naive(self, text, budget):
        sents = text.split(". ")
        out, used = [], 0
        for s in sents:
            if used + len(s) > budget:
                break
            out.append(s)
            used += len(s)
        return ". ".join(out)

    def bench_inputs(self, spec):
        return (
            ". ".join(
                f"sentence number {i} has content" for i in range(spec.get("n", 2000))
            ),
        ), {"budget": spec.get("budget", 500)}
