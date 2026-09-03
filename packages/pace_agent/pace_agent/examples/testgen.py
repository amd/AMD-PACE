# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PACE Agent example agent: self-correcting unit-test author.

WARNING: This example executes model-generated Python with `exec()`. Use only trusted inputs
in an isolated environment; it is not a security sandbox.

Reads a Python function and drives it through a *cyclic* LangGraph that actually
runs the tests it writes and refines them when they fail:

    read -> analyze -> plan -> generate -> assemble -> lint -> execute -> route
                                              ^                             |
                                              |     (always >=1 round; more while failing)
                                              +------------- improve <-------+
                                                                            |
                                                            (good enough) -> finalize -> END

`execute` runs each generated test against the real function and captures
pass/fail/errors — that's the feedback signal. `improve` feeds those results
(and a coverage self-review) back to the model to revise the tests. The agent
always runs at least one improve round and keeps looping while tests fail (up to
`_MAX_ROUNDS`), so the looped nodes accumulate several samples (real median/p95/p99).

Replay-safe: the only non-determinism is the LLM calls (captured in the tape);
test execution is deterministic, so the recorded trajectory.msgpack replays exactly
and the control flow (how many refine rounds) reproduces.
"""

import ast
import re
import time
from typing import TypedDict

NAME = "testgen"
KIND = "showcase"
REPLAY_SAFE = True

_DEFAULT_FN = (
    "def is_leap_year(year):\n"
    '    """True if `year` is a leap year in the Gregorian calendar."""\n'
    "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"
)
_MIN_ROUNDS = 1
_MAX_ROUNDS = 3

_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


class _S(TypedDict, total=False):
    source: str
    fn_name: str
    signature: str
    docstring: str
    facts: list
    plan: list
    module: str
    tests: list
    lint_ok: bool
    results: list
    n_fail: int
    round: int
    critique: str
    passed: bool
    per_step: list


def _code(raw):
    m = _CODE_RE.search(raw or "")
    return (m.group(1) if m else (raw or "")).strip()


def _step(state, node, **fields):
    entry = {"step": len(state.get("per_step", [])), "node": node, **fields}
    return {"per_step": state.get("per_step", []) + [entry]}


def _run_tests(source, module, fn_name):
    """Execute each test in `module` against the function defined in `source`."""
    ns = {}
    try:
        exec(compile(source, "<fn>", "exec"), ns)
    except Exception as e:  # noqa: BLE001 - report any setup failure
        return [
            {
                "name": "<setup>",
                "outcome": "error",
                "detail": f"function did not exec: {e}",
            }
        ]
    import unittest

    ns["unittest"] = unittest
    # strip imports that reference the function itself (e.g. `from clamp import clamp`)
    keep = [
        ln
        for ln in module.splitlines()
        if not re.match(
            rf"\s*(from\s+\w+\s+import|import)\b.*\b{re.escape(fn_name)}\b", ln
        )
    ]
    try:
        exec(compile("\n".join(keep), "<tests>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return [
            {
                "name": "<module>",
                "outcome": "error",
                "detail": f"tests did not exec: {type(e).__name__}: {e}",
            }
        ]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for obj in list(ns.values()):
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            suite.addTests(loader.loadTestsFromTestCase(obj))
    results = []
    for test in suite:
        r = unittest.TestResult()
        test.run(r)
        name = getattr(test, "_testMethodName", str(test))
        if r.failures:
            detail = r.failures[0][1].strip().splitlines()[-1]
            results.append({"name": name, "outcome": "fail", "detail": detail})
        elif r.errors:
            detail = r.errors[0][1].strip().splitlines()[-1]
            results.append({"name": name, "outcome": "error", "detail": detail})
        else:
            results.append({"name": name, "outcome": "pass", "detail": ""})
    if not results:
        return [
            {
                "name": "<module>",
                "outcome": "error",
                "detail": "no unittest.TestCase defined",
            }
        ]
    return results


def _read(state):
    src = state["source"]
    fn_name, sig, doc = "<unknown>", "", ""
    try:
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef):
                fn_name = node.name
                sig = f"{node.name}({', '.join(a.arg for a in node.args.args)})"
                doc = ast.get_docstring(node) or ""
                break
    except SyntaxError:
        pass
    return {
        "fn_name": fn_name,
        "signature": sig,
        "docstring": doc,
        "round": 0,
        **_step(state, "read", fn_name=fn_name, signature=sig, source_chars=len(src)),
    }


def _analyze(state):
    facts = []
    try:
        tree = ast.parse(state["source"])
        raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
        branches = [
            n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))
        ]
        if raises:
            facts.append(f"raises {len(raises)} exception(s) — test the error path")
        if branches:
            facts.append(f"{len(branches)} branch(es) — cover both sides")
        if state.get("docstring"):
            facts.append("has a docstring describing intended behavior")
    except SyntaxError:
        facts.append("source did not parse")
    return {"facts": facts, **_step(state, "analyze", n_facts=len(facts), facts=facts)}


def _assemble(state):
    module = "import unittest\n\n\n" + "\n\n\n".join(state.get("tests", [])) + "\n"
    return {
        "module": module,
        **_step(
            state,
            "assemble",
            n_tests=len(state.get("tests", [])),
            module_chars=len(module),
        ),
    }


def _lint(state):
    ok, names = True, []
    try:
        for node in ast.walk(ast.parse(state.get("module", ""))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                names.append(node.name)
    except SyntaxError:
        ok = False
    return {
        "lint_ok": ok and bool(names),
        **_step(state, "lint", parses=ok, test_names=names),
    }


def _execute(state):
    """Run the generated tests against the real function — the feedback signal."""
    results = _run_tests(
        state["source"], state.get("module", ""), state.get("fn_name", "")
    )
    n_fail = sum(1 for r in results if r["outcome"] != "pass")
    return {
        "results": results,
        "n_fail": n_fail,
        **_step(
            state,
            "execute",
            results=results,
            n_pass=len(results) - n_fail,
            n_fail=n_fail,
        ),
    }


def _finalize(state):
    results = state.get("results", [])
    passed = (
        state.get("lint_ok", False)
        and results
        and all(r["outcome"] == "pass" for r in results)
    )
    return {
        "passed": bool(passed),
        **_step(
            state,
            "finalize",
            rounds=state.get("round", 0),
            n_tests=len(results),
            passed=bool(passed),
        ),
    }


def _plan(state, llm):
    prompt = [
        {
            "role": "system",
            "content": "You plan unit tests. Given a function and static-analysis "
            "notes, list exactly 3 concrete scenarios (normal, edge, error), one per line, terse.",
        },
        {
            "role": "user",
            "content": f"Function:\n{state['source']}\n\nNotes:\n- "
            + "\n- ".join(state.get("facts", []))
            + "\n\nList the 3 scenarios.",
        },
    ]
    raw = llm.complete(prompt)
    plan = [ln.strip("-* \t") for ln in (raw or "").splitlines() if ln.strip()][:3]
    return {"plan": plan, **_step(state, "plan", llm_raw=raw, plan=plan)}


def _generate(state, llm):
    prompt = [
        {
            "role": "system",
            "content": "You write unittest tests. Output ONLY a single ```python block "
            "with a single unittest.TestCase subclass named TestTarget with EXACTLY 3 methods "
            "named test_normal, test_edge, test_error. Call the function directly by name (it "
            "is already defined — do NOT import it). Use self.assertRaises for expected errors.",
        },
        {
            "role": "user",
            "content": f"Function under test:\n{state['source']}\n\nScenarios:\n- "
            + "\n- ".join(state.get("plan", []))
            + "\n\nWrite the test case.",
        },
    ]
    raw = llm.complete(prompt)
    code = _code(raw)
    tests = [code] if code.strip() else []
    return {
        "tests": tests,
        **_step(state, "generate", llm_raw=raw, n_parsed=len(tests)),
    }


def _improve(state, llm):
    """Feed execution results + a coverage self-review back to the model to revise the tests."""
    res = state.get("results", [])
    summary = (
        "\n".join(
            f"- {r['name']}: {r['outcome']}"
            + (f" — {r['detail']}" if r["detail"] else "")
            for r in res
        )
        or "- (no tests ran)"
    )
    prompt = [
        {
            "role": "system",
            "content": "You improve unittest tests. You are given the current tests and "
            "the result of running them. Fix any failing test and strengthen coverage. "
            "Output ONLY a ```python block with a single unittest.TestCase subclass named "
            "TestTarget with exactly 3 methods test_normal, test_edge, test_error. Call the "
            "function directly by name; do NOT import it.",
        },
        {
            "role": "user",
            "content": f"Function under test:\n{state['source']}\n\nCurrent tests:\n"
            f"{state.get('module', '')}\n\nExecution results:\n{summary}\n\nReturn the improved tests.",
        },
    ]
    raw = llm.complete(prompt)
    code = _code(raw)
    tests = [code] if code.strip() else state.get("tests", [])
    rnd = state.get("round", 0) + 1
    return {
        "tests": tests,
        "round": rnd,
        **_step(
            state,
            "improve",
            llm_raw=raw,
            round=rnd,
            n_fail_before=state.get("n_fail", 0),
            n_parsed=len(tests),
        ),
    }


def _route(state):
    """After execute: always improve at least once; keep looping while tests fail."""
    r = state.get("round", 0)
    if r < _MIN_ROUNDS or (state.get("n_fail", 0) > 0 and r < _MAX_ROUNDS):
        return "improve"
    return "finalize"


def _build(llm):
    from langgraph.graph import END, StateGraph

    from pace_agent.node_counter import _record

    def _timed(name, fn):
        def wrapped(state):
            t = time.perf_counter()
            try:
                return fn(state)
            finally:
                _record(name, time.perf_counter() - t)

        return wrapped

    plain = [
        ("read", _read),
        ("analyze", _analyze),
        ("assemble", _assemble),
        ("lint", _lint),
        ("execute", _execute),
        ("finalize", _finalize),
    ]
    withllm = [("plan", _plan), ("generate", _generate), ("improve", _improve)]

    g = StateGraph(_S)
    for name, fn in plain:
        g.add_node(name, _timed(name, fn))
    for name, fn in withllm:
        g.add_node(name, _timed(name, (lambda f: lambda s: f(s, llm))(fn)))

    g.set_entry_point("read")
    g.add_edge("read", "analyze")
    g.add_edge("analyze", "plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "assemble")
    g.add_edge("assemble", "lint")
    g.add_edge("lint", "execute")
    g.add_conditional_edges(
        "execute", _route, {"improve": "improve", "finalize": "finalize"}
    )
    g.add_edge("improve", "assemble")
    g.add_edge("finalize", END)
    return g.compile()


def run(llm, task=None):
    task = task or {}
    source = task.get("function") or _DEFAULT_FN
    out = _build(llm).invoke(
        {"source": source, "per_step": []}, config={"recursion_limit": 50}
    )
    return {
        "answer": out.get("module", ""),
        "tests": out.get("tests", []),
        "results": out.get("results", []),
        "rounds": out.get("round", 0),
        "reached_answer": bool(out.get("passed")),
        "iterations": len(out.get("per_step") or []),
        "per_step": out.get("per_step"),
    }
