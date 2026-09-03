# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os

from pace_agent.rundir import RunDir


def _ms(x):
    return f"{x * 1e3:.2f} ms" if isinstance(x, (int, float)) else "-"


def _c(d, key, prec=2):
    """A numeric table cell, or '-' when the value is absent (e.g. tail percentiles under a
    small sample) — so small-sample fields don't render as a misleading 0.00."""
    v = d.get(key)
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "-"


def _text(value, width, dash="-"):
    """A cell holding text PACE Agent did not write — a query, a model answer, a gold answer. A pipe
    would split the row and a newline would tear it in half, so both are neutralized."""
    if value is None:
        return dash
    text = str(value)[:width]
    for ch in ("\r\n", "\r", "\n"):
        text = text.replace(ch, " ")
    return text.replace("|", "\\|").strip() or dash


def _render_multi(L, res, opt=None):
    """One aggregate report for a multi-task profile run: workload totals, the optimize E2E
    projection + breakdown (when present), and a compact per-query table."""
    agg = res.get("aggregate") or {}
    L += [
        "",
        "## Workload aggregate",
        "",
        f"- queries: {agg.get('n_queries')}  ·  total E2E: {agg.get('e2e_median_total_s', 0):.0f} s",
        f"- llm calls: {agg.get('total_llm_calls')}  ·  nodes traversed: {agg.get('total_nodes_traversed')}",
        f"- context tokens: {agg.get('total_context_tokens')} "
        f"(cached {agg.get('total_cached_tokens')})  ·  output: {agg.get('total_output_tokens')}",
        f"- non-model CPU: {agg.get('total_non_model_cpu_s', 0):.0f} s",
    ]
    if agg.get("answered") is not None:
        L.append(
            f"- answered: {agg['answered']}/{agg.get('n_queries')}  ·  "
            f"errors: {agg.get('n_errors', 0)}"
        )
    sc = agg.get("scoring")
    if sc:
        L.append(
            f"- auto-match: strict {agg.get('auto_match_strict')}/{agg.get('n_scored')}  ·  "
            f"loose {agg.get('auto_match_loose')}/{agg.get('n_scored')} of the graded tasks"
            + (
                f"  ·  {agg['n_unscored']} task(s) carry no expected answer"
                if agg.get("n_unscored")
                else ""
            )
        )
        L.append(
            "- both are normalized string/number heuristics, not a benchmark's official metric: "
            "strict is a floor (a right answer in a sentence fails it), loose is a ceiling "
            "(containment over-credits). Read the answers before quoting accuracy."
        )
    if agg.get("observe_speedup_median"):
        L.append(
            f"- in-run observe A/B: {agg['observe_speedup_median']:.1f}x median "
            "(projection onto E2E comes from `pace-agent optimize`)"
        )
    oa = (opt or {}).get("aggregate") or {}
    oe = oa.get("e2e") or {}
    if oe:
        L += [
            "",
            "## Optimize — observe A/B + end-to-end projection",
            "",
            f"- observe: naive {oa.get('observe_naive_ms_median', 0):.1f} -> "
            f"pace {oa.get('observe_pace_ms_median', 0):.1f} ms/observe "
            f"({oa.get('observe_speedup_median')}x median, "
            f"{oa.get('n_scored', oa.get('n_tasks'))}/{oa.get('n_tasks')} valid tasks, "
            f"{oa.get('n_parity_fail', 0)} parity-fail, {oa.get('n_failed', 0)} error / "
            f"{oa.get('total_snapshots')} snapshots)",
            f"- **E2E: {oe.get('reference_total_s', 0):.0f}s -> "
            f"{oe.get('candidate_total_s', 0):.0f}s "
            f"({oe.get('speedup')}x, {oe.get('pct_faster')}% faster)**",
        ]
        if not oa.get("valid", True):
            L.append(
                f"- **INVALID: {oa.get('invalid_reason')}** — speedup not claimed for the run."
            )
        L += ["", "| component | seconds |", "|---|---|"]
        for k, v in (oe.get("breakdown_total_s") or {}).items():
            L.append(f"| {k} | {v} |")
    graded = bool(sc)
    L += [
        "",
        "## Per-query",
        "",
        "| query | answer |"
        + (" expected | strict | loose |" if graded else "")
        + " done | steps | E2E (s) | llm calls |",
        "|---|---|" + ("---|---|---|" if graded else "") + "---|---|---|---|",
    ]
    for q in res.get("per_query") or []:
        row = f"| {_text(q.get('query'), 40)} | {_text(q.get('answer'), 30)} "
        if graded:
            row += (
                f"| {_text(q.get('gold'), 30)} | {q.get('match_strict')} "
                f"| {q.get('match_loose')} "
            )
        L.append(
            row + f"| {q.get('completed')} | {q.get('steps')} "
            f"| {q.get('e2e_median', 0):.1f} | {q.get('llm_calls')} |"
        )
    failed = [q for q in res.get("per_query") or [] if q.get("error")]
    if failed:
        L += ["", "## Failures", ""]
        L += [
            f"- `{_text(q.get('query'), 60)}`: {_text(q['error'], 200)}" for q in failed
        ]


def _headline(L, res, opt):
    """One-line verdict at the very top of the report. This is the line that gets quoted, so an
    invalid run says so here rather than only further down the page."""
    agg = (opt or {}).get("aggregate")
    verdict = agg if agg is not None else opt
    oe = (agg or {}).get("e2e") or (opt or {}).get("e2e")
    if verdict and not verdict.get("valid", True):
        L.append(
            f"- **verdict: INVALID — {verdict.get('invalid_reason', 'the run did not validate')}; "
            "no speedup is claimed**"
        )
    elif oe and oe.get("speedup"):
        pf = oe.get("pct_faster")
        L.append(
            f"- **verdict: projected E2E {oe['speedup']}x"
            + (f" ({pf}% faster)" if pf is not None else "")
            + " (observe A/B)**"
        )
    elif isinstance(res.get("e2e"), dict) and res["e2e"].get("median"):
        L.append(
            f"- **verdict: measured E2E {_ms(res['e2e']['median'])} · "
            f"{res.get('steps')} steps · completed={res.get('completed')}**"
        )


def render(run_dir):
    from pace_agent.optimize import read_latest_optimize

    rd = RunDir(run_dir)
    res = rd.read_result()
    opt = read_latest_optimize(run_dir)
    L = [f"# PACE Agent report — {res.get('cmd', '')}", ""]
    L.append(f"- agent: `{res.get('agent', '-')}` · impl: {res.get('impl', '-')}")
    if res.get("query"):
        L.append(f"- query: {res['query']}")
    _headline(L, res, opt)

    if res.get("cmd") == "profile_multi":
        _render_multi(L, res, opt)

    pq = res.get("per_query")
    if pq and res.get("cmd") != "profile_multi":
        # legacy flat layout (the retired `bench workload` command): still renderable so recorded
        # runs stay readable, but nothing produces it any more
        L += [
            "",
            f"## Workload: {res.get('workload', '-')} · impl: {res.get('impl', '-')}",
            "",
            "| query | answer | done | E2E (ms) | observe naive→pace | trace | trajectory |",
            "|---|---|---|---|---|---|---|",
        ]
        for q in pq:
            ab = q.get("observe_ab") or {}
            sp = ab.get("speedup")
            ab_s = (
                (
                    f"{ab['naive_ms']['median']:.0f}→{ab['pace_ms']['median']:.2f} ms"
                    + (f" ({sp:.0f}×)" if sp else "")
                )
                if ab
                else "-"
            )
            L.append(
                f"| {_text(q.get('query'), 40)} | {_text(q.get('answer'), 40)} | {q.get('completed')} "
                f"| {q.get('e2e_ms', 0):.0f} | {ab_s} | `{q.get('trace', '-')}` | `{q.get('trajectory', '-')}` |"
            )
        cg = res.get("cross_graph") or {}
        if cg:
            L += [
                "",
                f"- answered: {cg.get('answered')}/{cg.get('n_queries')} "
                f"· total E2E: {cg.get('total_e2e_ms', 0):.0f} ms "
                f"· observe median speedup: {_c(cg, 'observe_median_speedup', 0)}×",
            ]

    e2e = res.get("e2e")
    if isinstance(e2e, dict):
        med = f"- E2E median: {_ms(e2e.get('median'))}"
        if e2e.get("p95") is not None:
            med += f" (p95 {_ms(e2e['p95'])})"
        L += [
            "",
            "## End-to-end (measured)",
            "",
            med,
            f"- steps: {res.get('steps')} · completed: {res.get('completed')} · answer: {str(res.get('answer'))[:80]}",
        ]
        if res.get("scoring"):
            L.append(
                f"- expected: {str(res.get('gold'))[:80]} · match: "
                f"strict {res.get('match_strict')}, loose {res.get('match_loose')}"
            )

    pn = res.get("per_node")
    if pn:
        L += [
            "",
            "## Per-node (ms)",
            "",
            "| node | count | median | p95 | p99 | total |",
            "|---|---|---|---|---|---|",
        ]
        for n, d in pn.items():
            L.append(
                f"| {n} | {d.get('count')} | {_c(d, 'median')} | {_c(d, 'p95')} "
                f"| {_c(d, 'p99')} | {_c(d, 'total_ms')} |"
            )

    ab = res.get("observe_ab")
    if ab:
        n_, p_ = ab["naive_ms"], ab["pace_ms"]
        L += [
            "",
            "## observe: naive vs pace (ms)",
            "",
            "| impl | median | p95 | p99 |",
            "|---|---|---|---|",
            f"| naive | {_c(n_, 'median')} | {_c(n_, 'p95')} | {_c(n_, 'p99')} |",
            f"| pace | {_c(p_, 'median', 3)} | {_c(p_, 'p95', 3)} | {_c(p_, 'p99', 3)} |",
            "",
            f"- **speedup (median): {ab['speedup']:.0f}×** · parity: {ab['parity_all']}",
        ]

    rm = res.get("run_metrics")
    if rm:

        def _tok(k):
            v = rm.get(k)
            return "n/a" if v is None else str(v)

        tot = rm.get("context_tokens_total")
        ctx = (
            f"{tot} (max {rm.get('context_tokens_max')})" if tot is not None else "n/a"
        )
        L += [
            "",
            "## Run metrics",
            "",
            f"- llm calls: {rm.get('llm_calls')} ({rm.get('api') or '?'} API) "
            f"· nodes traversed: {rm.get('nodes_traversed')} · op calls: {rm.get('tool_calls')}",
            f"- context tokens: {ctx}  ·  cached: {_tok('cached_tokens_total')}",
            f"- output tokens: {_tok('output_tokens_total')} "
            f"(reasoning {_tok('reasoning_tokens_total')}, work {_tok('work_tokens_total')})",
        ]

    attr = res.get("attribution")
    if attr:
        tot = attr.get("total") or 0
        nm = attr.get("non_model_cpu", 0)
        L += [
            "",
            "## Attribution (model vs non-model CPU)",
            "",
            f"- total: {_ms(attr.get('total'))}",
            f"- model: {_ms(attr.get('model'))}",
            f"- non-model CPU: {_ms(nm)}"
            + (f" ({nm / tot * 100:.1f}%)" if tot else ""),
            f"- unattributed: {_ms(attr.get('unattributed'))}",
        ]
        if attr.get("buckets"):
            L += ["", "| bucket | time |", "|---|---|"]
            for b, t in attr["buckets"].items():
                L.append(f"| {b} | {_ms(t)} |")

    if opt and opt.get("mode") in ("html-replay", "browser-snapshot"):
        rows = opt.get("tasks") or [opt]
        L += [
            "",
            "## Optimize (browser-snapshot replay)",
            "",
            "| task | snapshots | observe naive | observe pace | speedup |",
            "|---|---|---|---|---|",
        ]
        for t in rows:
            name = t.get("task") or t.get("trajectory") or "-"
            if t.get("error"):  # a task that could not be replayed still has to appear
                L.append(f"| {name} | {_text(t['error'], 60)} | - | - | - |")
                continue
            on, op_ = t.get("observe_naive_ms", {}), t.get("observe_pace_ms", {})
            sp = t.get("observe_speedup")
            L.append(
                f"| {name} | {t.get('n_snapshots', '-')} "
                f"| {on.get('median', 0):.2f} ms | {op_.get('median', 0):.3f} ms "
                f"| {f'{sp:.0f}×' if sp else '-'} |"
            )
    elif opt and (opt.get("e2e") or opt.get("e2e_naive")):
        ref = opt.get("reference_impl", "naive")
        cand = opt.get("candidate_impl", opt.get("impl", "optimized"))
        e2e = opt.get("e2e") or {
            "reference": opt.get("e2e_naive"),
            "candidate": opt.get("e2e_pace"),
            "speedup": opt.get("e2e_speedup"),
        }
        valid = opt.get("valid", True)
        L += ["", f"## Optimize (replayed E2E: {ref} vs {cand})", ""]
        if not valid:
            L.append(
                f"- **INVALID: {opt.get('invalid_reason', 'zero candidate coverage')}** "
                "— no speedup is claimed."
            )
        L += [
            f"- E2E {ref}: {_ms(e2e['reference']['median'])}",
            f"- E2E {cand}: {_ms(e2e['candidate']['median'])}",
            (
                f"- **E2E speedup: {e2e['speedup']:.2f}×**"
                if e2e.get("speedup")
                else "- E2E speedup: n/a"
            ),
            f"- parity all pass: {opt['parity_all_pass']}",
        ]
        cov = opt.get("coverage")
        if cov:
            total, candc = cov.get("total_calls", 0), cov.get("candidate_calls", 0)
            pct = (100.0 * candc / total) if total else 0.0
            timef = cov.get("covered_time_fraction")
            tail = f" · {timef * 100:.0f}% of naive time" if timef is not None else ""
            L.append(f"- candidate coverage: {candc}/{total} calls ({pct:.0f}%){tail}")
        prov = opt.get("provider") or {}
        if prov.get("source") not in (None, "builtin"):
            L.append(
                f"- provider: {prov['source']}"
                + (f" {prov['version']}" if prov.get("version") else "")
            )
        L += [
            "",
            f"| op | {ref} | {cand} | speedup | parity |",
            "|---|---|---|---|---|",
        ]
        for name, d in opt["per_tool"].items():
            sp = f"{d['speedup']:.2f}×" if d.get("speedup") else "-"
            refc = d.get("reference") or d.get("naive") or {}
            cdc = d.get("candidate") or d.get("optimized") or {}
            if name in opt.get("parity", {}):
                par = "PASS" if opt["parity"][name] else "FAIL"
            else:
                par = "—"  # no candidate for this op (ran reference)
            L.append(
                f"| {name} | {_ms(refc.get('median', 0))} | {_ms(cdc.get('median', 0))} "
                f"| {sp} | {par} |"
            )

    tel = res.get("telemetry")
    if tel:
        L += ["", "## Telemetry", "", f"- mode: {tel.get('mode', tel.get('source'))}"]
        if tel.get("metrics"):
            L.append(f"- metrics: {tel['metrics']}")
        if tel.get("hw"):
            L.append(f"- hw: {tel['hw']}")

    rd.write_report("\n".join(x for x in L if x is not None))
    return os.path.join(run_dir, "report.md")
