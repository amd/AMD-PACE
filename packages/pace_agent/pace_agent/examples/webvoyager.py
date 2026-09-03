# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

"""PACE Agent example: WebVoyager, adapted from the original (PhaniAgentCPU/AgentCPU/WebVoyager)
as a strict LangGraph agent — text-only mode, using the EXACT WebVoyager observation
(full WebArena accessibility tree via CDP) and the EXACT coordinate-resolved actions
(exec_action_click/type/scroll). Wrapped for PACE Agent per-node timing. Live browser (not replayed).

(The observe naive-vs-pace speedup A/B is intentionally NOT applied here yet — first make the
faithful naive WebVoyager pipeline work; the CPU-observe A/B is layered on afterwards.)
"""

import operator
import re
import time
from typing import Annotated, TypedDict

from pace_agent.bench.workloads.browser import SeleniumBrowser
from pace_agent.examples.common import op
from pace_agent.node_counter import timings as _node_timings

NAME = "webvoyager"
KIND = "showcase"
REPLAY_SAFE = False
QUESTION = "What is the first heading on example.com?"
START_URL = "https://example.com"

# Verbatim WebVoyager SYSTEM_PROMPT_TEXT_ONLY (injected from the original prompts.py).
# Kept byte-for-byte in sync with the upstream WebVoyager repo's prompts.py — the source of
# truth for the prompt. If you update the browsing guidelines, update them upstream too.
_SYS = "Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Accessibility Tree with numerical label representing information about the page, then follow the guidelines and choose one of the following actions:\n1. Click a Web Element.\n2. Delete existing content in a textbox and then type content. \n3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.\n4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.\n5. Go back, returning to the previous webpage.\n6. Google, directly jump to the DuckDuckGo search page (the \"Google\" action now opens DuckDuckGo, which avoids Google's automated-traffic CAPTCHAs). When you can't find information in some websites, try starting over with this search action.\n7. Answer. This action should only be chosen when all questions in the task have been solved.\n\nCorrespondingly, Action should STRICTLY follow the format:\n- Click [Numerical_Label]\n- Type [Numerical_Label]; [Content]\n- Scroll [Numerical_Label or WINDOW]; [up or down]\n- Wait\n- GoBack\n- Google\n- ANSWER; [content]\n\nKey Guidelines You MUST follow:\n* Action guidelines *\n1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  \n2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. \n3) Execute only one action per iteration. \n4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.\n5) When a complex Task involves multiple questions or steps, select \"ANSWER\" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER.\n6) If you can already answer the question from the information currently visible in the accessibility tree, use ANSWER immediately — do NOT click or navigate further just to confirm what you already know.\n7) If a page shows a CAPTCHA or \"unusual/automated traffic\" block, do NOT keep retrying it. Prefer authoritative sites directly (for arXiv use arxiv.org or its search at arxiv.org/search) and use the DuckDuckGo search action instead of returning to a blocked page.\n* Web Browsing Guidelines *\n1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.\n2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.\n3) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.\n4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.\n5) The accessibility tree may not show all page content (especially on JavaScript-heavy sites). If scrolling does not reveal new content after 2 attempts, switch strategy: use the DuckDuckGo search action or navigate directly to a relevant sub-page.\n\nYour reply should strictly follow the format:\nThought: {Your brief thoughts (briefly summarize the info that will help ANSWER)}\nAction: {One Action format you choose}\n\nThen the User will provide:\nObservation: {Accessibility Tree of a web page}"


# The deterministic observe A/B (replay_observe) + its parity normalizer (_content_tree) now
# live in pace_agent.replay.browser_snapshot as the core "browser-snapshot" replay mode. This agent
# only records page snapshots into the trajectory; `pace-agent optimize <run>` replays the A/B.


def _parse_action(text):
    """WebVoyager extract_information: Click/Type/Scroll/Wait/GoBack/Google/ANSWER.
    Labels are the accessibility-tree node ids (numeric strings)."""
    seg = text
    m = re.search(r"Action\s*:\s*(.+)", text, re.I | re.S)
    if m:
        seg = m.group(1)
    for key, pat in [
        ("answer", r"ANSWER\s*[;:]\s*(.+)"),
        ("type", r"Type\s*\[?(\d+)\]?\s*[;:]?\s*(.+)"),
        ("scroll", r"Scroll\s*\[?(\d+|WINDOW)\]?\s*[;:]?\s*(up|down)"),
        ("click", r"Click\s*\[?(\d+)\]?"),
        ("goback", r"\bGoBack\b"),
        ("google", r"\bGoogle\b"),
        ("wait", r"\bWait\b"),
    ]:
        mm = re.search(pat, seg, re.I | re.S)
        if mm:
            if key == "answer":
                return ("answer", mm.group(1).strip())
            if key == "type":
                return (
                    "type",
                    (mm.group(1), mm.group(2).strip().splitlines()[0].strip("[]")),
                )
            if key == "scroll":
                t = mm.group(1)
                return (
                    "scroll",
                    (t if t.upper() == "WINDOW" else t, mm.group(2).lower()),
                )
            if key == "click":
                return ("click", mm.group(1))
            return (key, None)
    return ("stop", None)


def _complete_retry(llm, messages, tries=3):
    for _ in range(tries):
        r = llm.complete(messages)
        if r and r.strip():
            return r
    return ""


# Per-call context bound. WebVoyager resends the full conversation every iteration; each
# observation carries a full accessibility tree, so an unclipped run grows without bound.
# Prefix caching (``LLMClient`` usage ``cached`` tokens) only avoids re-processing the
# stable prefix — it does not shrink the request or raise the context-window limit. A long
# run was seen resending millions of tokens and tripping the gateway's /chat 400. Clipping
# keeps system + the first user turn (task + first page) + the last N observation/action
# pairs. Set ``max_context_obs`` to 0 (or ``PACE_AGENT_WEBVOYAGER_MAX_CTX_OBS=0`` from GAIA) to
# send the full history when your gateway's window is large enough.
#
# Trade-offs: clipping lowers cost and prevents context-limit failures; dropping middle
# turns can hurt tasks that need recall of earlier pages. Prefix cache helps either way on
# the retained prefix but does not replace this bound.
_DEFAULT_CTX_OBS = 6


def _clip_msgs(messages, keep_obs):
    """Bound resent history to system + first turn + last ``keep_obs`` observation/action pairs.

    ``keep_obs <= 0`` disables clipping (full history — use when prefix cache + a large
    context window make the bound unnecessary).
    """
    if keep_obs <= 0 or len(messages) <= 2:
        return messages
    head = messages[:2]  # system prompt + first user turn (task + first observation)
    tail = messages[2:]
    max_tail = 2 * keep_obs
    if len(tail) > max_tail:
        tail = tail[-max_tail:]
    return head + tail


# _content_tree moved to pace_agent.replay.browser_snapshot with the browser-snapshot replay mode.


class _S(TypedDict):
    step: int
    messages: Annotated[list, operator.add]
    per_step: Annotated[list, operator.add]
    obs: dict
    obs_info: dict
    plan: dict
    init: str
    warn: str
    answer: object
    done: bool


def _build(
    llm,
    br,
    query,
    start_url,
    max_iter,
    capture=True,
    keep_obs=_DEFAULT_CTX_OBS,
    deadline=None,
):
    from langgraph.graph import END, StateGraph

    def load(state):
        br.navigate(start_url)
        init = f"Now given a task: {query}  Please interact with {start_url} and get the answer. \n"
        reasoning = getattr(llm, "reasoning", None)
        sys = f"Reasoning: {reasoning}\n\n{_SYS}" if reasoning else _SYS
        if getattr(llm, "vision", False):
            sys += (
                "\n\n* Screenshot *\nYou are ALSO given a screenshot of the current page. "
                "The screenshot has the SAME Numerical_Labels drawn as red tags on the "
                "corresponding elements (red boxes). Use the screenshot together with the "
                "Accessibility Tree to pick the correct Numerical_Label for your Action."
            )
        return {
            "step": 0,
            "messages": [{"role": "system", "content": sys}],
            "init": init,
            "warn": "",
            "obs_info": {},
        }

    def observe(state):
        from pace_agent.node_counter import _record

        # LIVE run uses the naive (faithful WebVoyager) observe — the baseline. The naive-vs-pace
        # A/B is produced afterwards, DETERMINISTICALLY, by replay_observe on the stored snapshot.
        t = time.perf_counter()
        tree, obs_info = br.observe_ax()
        _record("observe", time.perf_counter() - t)
        # PACE Agent CPU op on the REAL observation: drop exact-duplicate accessibility-tree
        # lines (repeated nav/menu carry no new info) to trim redundant tokens sent to the
        # model. Recorded tool call; its optimized tier is benchmarked by `pace-agent optimize`.
        tree = "\n".join(op("dedup", tree.splitlines()))
        # vision mode: Set-of-Marks screenshot whose red labels match the accessibility-tree
        # [labels] the model also reads (union_bound boxes). Timed as its own node so it does not
        # pollute the observe naive-vs-pace metric. Screenshots are runtime-only (not stored).
        shot = ""
        if getattr(llm, "vision", False):
            t = time.perf_counter()
            boxes = {
                lbl: e.get("union_bound")
                for lbl, e in obs_info.items()
                if e.get("union_bound")
            }
            shot = br.som_from_boxes(boxes)
            _record("screenshot", time.perf_counter() - t)
        # self-contained snapshot (full page, CSS/images inlined) for the deterministic replay A/B;
        # skipped for large benchmark runs (capture=False) where only task success matters.
        page_html = br.driver.page_source
        html = (br.snapshot_mhtml() or page_html) if capture else ""
        # PACE Agent CPU op on the REAL rendered HTML: visible-text extract kept on the step
        # record for the trace (recorded tool call; optimized tier = selectolax/lexbor).
        page_text = op("html_parse", page_html) if capture else ""
        warn = state.get("warn") or ""
        if state["step"] == 0:
            content = state["init"] + "\n" + tree
        else:
            content = (
                f"Observation:{(' ' + warn) if warn else ''} please analyze the accessibility "
                f"tree and give the Thought and Action.\n{tree}"
            )
        return {
            "messages": [{"role": "user", "content": content}],
            "obs": {
                "url": br.driver.current_url,
                "tree": tree,
                "n": len(obs_info),
                "html": html,
                "page_text": page_text[:2000],
                "shot": shot,
            },
            "obs_info": obs_info,
        }

    def plan(state):
        t = time.perf_counter()
        # Clip resent history (see _clip_msgs — prefix cache does not replace this bound).
        msgs = _clip_msgs(list(state["messages"]), keep_obs)
        # Vision: attach the screenshot to the latest observation only (one image/call).
        # State stays text-only so context accounting + history stay clean.
        shot = (state.get("obs") or {}).get("shot")
        if getattr(llm, "vision", False) and shot:
            msgs = list(msgs)
            last = dict(msgs[-1])
            last["content"] = [
                {"type": "text", "text": last.get("content", "")},
                llm.image_content(shot),
            ]
            msgs[-1] = last
        resp = _complete_retry(llm, msgs)
        reason_ms = (time.perf_counter() - t) * 1e3
        act_, arg = _parse_action(resp)
        return {
            "messages": [{"role": "assistant", "content": resp or "(empty)"}],
            "plan": {"act": act_, "arg": arg, "raw": resp, "reason_ms": reason_ms},
        }

    def act(state):
        o, p, oi = state["obs"], state["plan"], state["obs_info"]
        act_, arg = p["act"], p["arg"]
        step = state["step"] + 1
        entry = {
            "step": step,
            "url": o["url"],
            "n_elements": o["n"],
            "observation": o["tree"][:4000],
            "html": o["html"],
            "llm_raw": p["raw"],
            "action": act_,
            "action_arg": arg,
            "parsed_ok": act_ != "stop",
            "impl": "naive",
            "reason_ms": p["reason_ms"],
            "shot_chars": len(o.get("shot") or ""),
        }

        def box(label):
            e = oi.get(str(label)) or oi.get(label)
            return e["union_bound"] if (e and e.get("union_bound")) else None

        answer, done, warn = state.get("answer"), False, ""
        if act_ == "answer":
            answer, done = arg, True
        elif act_ == "click":
            ub = box(arg)
            if ub is None or not br.ax_click(ub):
                warn = f"Click [{arg}] failed (not resolvable)."
        elif act_ == "type":
            ub = box(arg[0])
            warn = (
                br.ax_type(ub, arg[1])
                if ub is not None
                else f"Type target [{arg[0]}] not in tree."
            )
        elif act_ == "scroll":
            tgt, dr = arg
            if str(tgt).upper() == "WINDOW":
                br.ax_scroll("WINDOW", dr)
            else:
                br.ax_scroll(tgt, dr, box(tgt))
        elif act_ == "wait":
            time.sleep(5)
        elif act_ == "goback":
            br.go_back()
        elif act_ == "google":
            br.navigate(
                start_url
            )  # "start over" -> task's start_url (hardcoded google.com CAPTCHAs headless)
        else:
            warn = "Invalid action; use Click/Type/Scroll/Wait/GoBack/Google/ANSWER."
        prev = [s["action"] for s in state["per_step"][-1:]]
        if [act_] == prev and not warn:
            warn = "You repeated the last action and the page didn't change — try a different element."
        # PACE Agent serialization ops on the compact step record (recorded tool calls; optimized
        # tier = orjson). Round-trip so both serialize and deserialize are exercised end-to-end.
        arg_json = list(arg) if isinstance(arg, tuple) else arg
        entry["record"] = op(
            "json_serialize",
            {
                "step": step,
                "url": o["url"],
                "n": o["n"],
                "action": act_,
                "arg": arg_json,
            },
        )
        op("json_deserialize", entry["record"])
        return {
            "per_step": [entry],
            "step": step,
            "answer": answer,
            "done": done,
            "warn": warn,
        }

    def finish(state):
        return {}

    def route(state):
        if state.get("done") or state["step"] >= max_iter:
            return "finish"
        # Wall-clock budget: stop between iterations once the deadline passes. Routing to finish
        # (rather than raising) lets the graph return normally so the `finally: br.quit()` in run()
        # always reaps the browser — a single hard task can no longer run unbounded.
        if deadline is not None and time.perf_counter() >= deadline:
            return "finish"
        return "observe"

    from pace_agent.node_counter import _record

    def _timed(name, fn):
        def wrapped(state):
            t = time.perf_counter()
            try:
                return fn(state)
            finally:
                _record(name, time.perf_counter() - t)

        return wrapped

    g = StateGraph(_S)
    g.add_node("observe", observe)  # observe records its own node time
    for name, fn in [("load", load), ("plan", plan), ("act", act), ("finish", finish)]:
        g.add_node(name, _timed(name, fn))
    g.set_entry_point("load")
    g.add_edge("load", "observe")
    g.add_edge("observe", "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", route, {"observe": "observe", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile()


def run(llm, task=None):
    from pace_agent.node_counter import PaceAgentNodeCounter, reset

    task = dict(task or {})
    query = task.get("query") or QUESTION
    start_url = task.get("start_url") or START_URL
    max_iter = int(task.get("max_iter", 15))
    capture = bool(task.get("capture_snapshots", True))
    keep_obs = int(task.get("max_context_obs", _DEFAULT_CTX_OBS))
    try:
        max_seconds = float(task.get("max_seconds") or 0) or None
    except (TypeError, ValueError):
        max_seconds = None
    deadline = (time.perf_counter() + max_seconds) if max_seconds else None
    c0, x0 = len(llm.calls), len(llm.prompt_tokens)
    reset()

    br = SeleniumBrowser(headless=True)
    try:
        app = _build(llm, br, query, start_url, max_iter, capture, keep_obs, deadline)
        out = app.invoke(
            {
                "step": 0,
                "messages": [],
                "per_step": [],
                "obs": {},
                "obs_info": {},
                "plan": {},
                "init": "",
                "warn": "",
                "answer": None,
                "done": False,
            },
            config={
                "callbacks": [PaceAgentNodeCounter()],
                "recursion_limit": max_iter * 4 + 20,
            },
        )
    finally:
        br.quit()

    per_step = out.get("per_step", [])
    answer = out.get("answer")
    timed_out = bool(
        deadline is not None and answer is None and time.perf_counter() >= deadline
    )
    ctx = [t for t in llm.prompt_tokens[x0:] if t is not None]
    # The live agent records naive observe; optimize replays snapshots for the pace A/B.
    return {
        "impl": "naive",
        "modality": "vision" if getattr(llm, "vision", False) else "text",
        "inputs": {"query": query, "start_url": start_url, "max_iter": max_iter},
        "final_output": (
            answer
            if answer is not None
            else (
                f"(wall-clock timeout after {max_seconds:.0f}s without ANSWER)"
                if timed_out
                else "(max_iter reached without ANSWER)"
            )
        ),
        "reached_answer": answer is not None,
        "timed_out": timed_out,
        "query": query,
        "answer": answer,
        "iterations": len(per_step),
        "llm_calls": len(llm.calls) - c0,
        "context_tokens_total": sum(ctx) if ctx else None,
        "per_node": _node_timings(),
        "per_step": per_step,
    }
