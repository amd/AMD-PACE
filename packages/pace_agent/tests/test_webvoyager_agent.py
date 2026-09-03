# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Browser-gated WebVoyager suite.

Groups three concerns, each in its own TestCase, all sharing the same browser gate and
temp-page helper and all skipping cleanly when no Chrome is available:

  * TestWebvoyagerAgent      -- agent + browser primitives (flags, vision, observe replay,
                                a stubbed live run, click/read, screenshot).
  * TestWebvoyagerActOpsAB   -- the PACE Agent tool ops applied *externally* to a captured
                                accessibility tree + rendered HTML; optimized == naive.
  * TestWebvoyagerOpsE2E     -- the real agent drives the ops itself; the recorded tool
                                calls are parity-checked (optimized == naive) end to end.
  * TestWebvoyagerLiveLLME2E -- opt-in: a REAL LLM (real API calls) drives the real agent
                                over a real browser to complete a task; skipped by default.
"""

import getpass
import json
import os
import shutil
import socket
import tempfile
import unittest
import urllib.parse
import urllib.request
import warnings

from pace_agent.bench.workloads.browser import (
    BrowserUnavailable,
    SeleniumBrowser,
    _browser_flags,
)
from pace_agent.capture import CaptureContext
from pace_agent.registry import get_tool
from pace_agent.replay import Recorder

WIRED_OPS = {"dedup", "html_parse", "json_serialize", "json_deserialize"}

_WEBVOYAGER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pace_agent",
    "examples",
    "webvoyager.py",
)

_NAV = ["Home", "Products", "Deals", "Cart", "Sign in", "Support", "About"]


class StubLLM:
    """Offline LLM double with every attribute the WebVoyager agent + Recorder read:
    token-accounting lists (agent) and record_into/replay_from (Recorder tape)."""

    def __init__(self, replies):
        self._replies = list(replies)
        self._i = 0
        self._tape = None
        self._recording = False
        self.calls = []
        self.prompt_tokens = []
        self.completion_tokens = []
        self.cached_tokens = []
        self.reasoning_tokens = []

    def record_into(self, tape):
        self._tape, self._recording = tape, True

    def replay_from(self, tape):
        self._tape, self._recording = list(tape), False

    def complete(self, msgs):
        self.calls.append(0.0)
        r = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        if self._recording and self._tape is not None:
            self._tape.append(r)
        return r


def _realistic_page(n_elements=120):
    """A realistic page: recurring nav/menu (so dedup has exact-duplicate lines to drop) +
    n interactive elements with visible text + filler paragraphs -- enough nodes that the
    accessibility tree does real work."""
    nav = "".join(f"<a href='#'>{x}</a>" for x in _NAV)
    header = f"<header><nav>{nav}</nav></header>" * 3
    rows = "".join(
        f"<div class='row'><span>Laptop model {i} - 16GB RAM, ${1000 + i}.{i % 100:02d}</span>"
        f"<a href='/item/{i}'>view item {i}</a>"
        f"<button data-id='{i}'>add {i} to cart</button>"
        f"<input type='text' aria-label='qty {i}' value='1'></div>"
        for i in range(n_elements)
    )
    filler = "".join(
        f"<p>Product paragraph {i} with a fair amount of descriptive text about specs.</p>"
        for i in range(40)
    )
    return (
        "<html><body>"
        + header
        + "<h1>Laptop search results</h1><div class='list'>"
        + rows
        + "</div>"
        + filler
        + "<footer><nav>"
        + nav
        + "</nav></footer></body></html>"
    )


def _norm(text):
    return " ".join(text.split())


_BROWSER_OK = None


def _require_browser():
    global _BROWSER_OK
    if _BROWSER_OK is None:
        try:
            SeleniumBrowser().quit()
            _BROWSER_OK = True
        except BrowserUnavailable:
            _BROWSER_OK = False
    if not _BROWSER_OK:
        raise unittest.SkipTest("no working browser")


def _live_endpoint():
    return os.environ.get("PACE_AGENT_LIVE_LLM_ENDPOINT", "http://localhost:8000")


def _endpoint_reachable(endpoint):
    """Protocol-agnostic host reachability: a TCP connect to the endpoint host:port. Avoids
    guessing the models route (vLLM `/v1/models` vs the gateway `/models`) or needing auth.
    """
    u = urllib.parse.urlparse(endpoint)
    host = u.hostname
    port = u.port or (443 if u.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=4):
            return True
    except OSError:
        return False


def _require_live_llm():
    """Opt-in gate for the live-LLM E2E: needs PACE_AGENT_LIVE_LLM=1 AND a reachable endpoint.
    Returns the endpoint base URL."""
    if not os.environ.get("PACE_AGENT_LIVE_LLM"):
        raise unittest.SkipTest("live-LLM E2E is opt-in; set PACE_AGENT_LIVE_LLM=1")
    endpoint = _live_endpoint()
    if not _endpoint_reachable(endpoint):
        raise unittest.SkipTest(f"no LLM endpoint reachable at {endpoint}")
    return endpoint


def _live_headers(api_key):
    """Optional gateway-auth header carrying the key, named by PACE_AGENT_LIVE_LLM_SUB_HEADER
    (e.g. AMD's `Ocp-Apim-Subscription-Key`), plus a `user` tag some gateways require.
    """
    name = os.environ.get("PACE_AGENT_LIVE_LLM_SUB_HEADER")
    if not name or not api_key or api_key == "none":
        return None
    try:
        who = getpass.getuser()
    except Exception:
        who = os.environ.get("USER", "pace_agent")
    return {name: api_key, "user": who}


def _served_model(endpoint, headers=None):
    """The model id to drive: PACE_AGENT_LIVE_LLM_MODEL if set, else the first id the endpoint
    advertises (tries `/v1/models` then `/models` so it works for vLLM and the gateway).
    """
    m = os.environ.get("PACE_AGENT_LIVE_LLM_MODEL")
    if m:
        return m
    base = endpoint.rstrip("/")
    for url in (base + "/v1/models", base + "/models"):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.load(r)
            ids = [row.get("id") for row in data.get("data", []) if row.get("id")]
            if ids:
                return ids[0]
        except Exception:
            continue
    return None


class _BrowserPageTest(unittest.TestCase):
    """Shared temp-page helper: write `html` to a temp file and return its file:// URL."""

    def _page(self, html, name="page.html"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            f.write(html)
        return f"file://{path}"


class TestWebvoyagerAgent(_BrowserPageTest):
    def test_headless_flag_is_configurable_without_dropping_anti_bot_flags(self):
        visible = _browser_flags(False)
        headless = _browser_flags(True)
        self.assertNotIn("--headless=new", visible)
        self.assertIn("--headless=new", headless)
        for flags in (visible, headless):
            self.assertIn("--disable-blink-features=AutomationControlled", flags)
            self.assertTrue(any(flag.startswith("--user-agent=") for flag in flags))

    def test_vision_is_config_driven(self):
        # vision is an explicit backend capability flag (default off), not model-name guessing
        from pace_agent.llm.client import LLMClient

        self.assertIs(
            LLMClient.from_config({"model": "openai/gpt-oss-20b"}).vision, False
        )
        c = LLMClient.from_config({"model": "Claude-Sonnet-4.5", "vision": True})
        self.assertIs(c.vision, True)
        # default image format is OpenAI image_url
        self.assertEqual(c.image_content("QUJD")["type"], "image_url")
        # anthropic image block only when image_format="anthropic"
        a = LLMClient.from_config(
            {"model": "Claude-Sonnet-5", "vision": True, "image_format": "anthropic"}
        )
        blk = a.image_content("QUJD")
        self.assertEqual(blk["type"], "image")
        self.assertEqual(blk["source"]["data"], "QUJD")

    def test_replay_observe_html_snapshots(self):
        _require_browser()
        from pace_agent.replay.browser_snapshot import replay_observe

        html = (
            "<html><body>"
            + "".join(f"<a href='#'>l{i}</a><button>b{i}</button>" for i in range(60))
            + "</body></html>"
        )
        r = replay_observe([html, html], reps=2)
        self.assertEqual(r["n_snapshots"], 2)
        self.assertGreaterEqual(r["observe_naive_ms"]["n"], 2)
        self.assertGreaterEqual(r["observe_pace_ms"]["n"], 2)
        self.assertTrue(r["observe_speedup"] and r["observe_speedup"] > 1)

    def test_webvoyager_live_stub(self):
        _require_browser()
        from pace_agent.examples import webvoyager

        page = self._page(
            "<html><body><h1>Example Domain</h1></body></html>", "example.html"
        )
        r = webvoyager.run(
            StubLLM(["ANSWER: Example Domain"]),
            {"query": "heading?", "start_url": page, "max_iter": 4},
        )
        self.assertTrue(r["reached_answer"])
        self.assertEqual(r["iterations"], 1)
        self.assertEqual(r["llm_calls"], 1)
        self.assertEqual(r["answer"], "Example Domain")
        self.assertEqual(r["modality"], "text")
        self.assertIn("median", r["per_node"]["observe"])

    def test_browser_click_and_read(self):
        _require_browser()
        br = SeleniumBrowser()
        try:
            page = self._page(
                "<html><body><button>Go</button><a href='#'>Link</a></body></html>",
                "clicktest.html",
            )
            br.navigate(page)
            self.assertGreaterEqual(len(br.observe("naive")), 1)
            self.assertIs(br.click(0), True)
            self.assertIn("Go", br.read_text())
        finally:
            br.quit()

    def test_observe_vision_screenshot(self):
        _require_browser()
        br = SeleniumBrowser()
        try:
            page = self._page(
                "<html><body><button>Go</button><a href='#'>L</a></body></html>",
                "vis.html",
            )
            br.navigate(page)
            v = br.observe_vision()
            self.assertTrue(v["elements"])
            self.assertIsInstance(v["screenshot_b64"], str)
            self.assertGreater(len(v["screenshot_b64"]), 100)
        finally:
            br.quit()


class TestWebvoyagerActOpsAB(_BrowserPageTest):
    """The PACE Agent `optimized` tiers (html_parse / dedup / json round-trip) agree with the
    naive reference on real browser output. Parity only -- timing lives in the bench
    harness, not the test suite."""

    def test_act_tool_ops_parity_on_real_browser_output(self):
        _require_browser()
        url = self._page(_realistic_page(120))
        br = SeleniumBrowser(headless=True)
        try:
            br.navigate(url)
            br.await_ready()
            tree, _info = br.observe_ax("naive")
            html = br.driver.page_source
        finally:
            br.quit()

        self.assertTrue(tree, "empty accessibility tree")
        lines = tree.splitlines()
        step = {"step": 1, "action": "scroll", "n": len(lines), "obs": lines[:50]}

        hp = get_tool("html_parse")
        dd = get_tool("dedup")
        js = get_tool("json_serialize")
        jd = get_tool("json_deserialize")

        # dedup: exact, order-preserving parity on the real accessibility-tree lines
        self.assertEqual(dd.dispatch("naive", lines), dd.dispatch("optimized", lines))

        # json round-trip: both tiers preserve the real step record; optimized == naive
        self.assertEqual(jd.dispatch("optimized", js.dispatch("optimized", step)), step)
        self.assertEqual(jd.dispatch("naive", js.dispatch("naive", step)), step)

        # html_parse: whitespace-normalized text parity on the real rendered HTML
        self.assertEqual(
            _norm(hp.dispatch("naive", html)), _norm(hp.dispatch("optimized", html))
        )


class TestWebvoyagerOpsE2E(_BrowserPageTest):
    """The real WebVoyager agent drives the ops itself: record the tool calls it makes over
    a frozen page, then assert it drove every wired op AND that optimized == naive on those
    real recorded inputs (a genuine end-to-end parity check, not a synthetic fixture).
    """

    def test_agent_drives_ops_and_optimized_matches_naive(self):
        _require_browser()
        from pace_agent.profile.profiler import import_agent

        agent = import_agent(_WEBVOYAGER)
        url = self._page(_realistic_page())
        # answer on the first step -> one observe + one act, exercising every wired op once
        stub = StubLLM(["Thought: found it.\nAction: ANSWER; Laptop search results"])
        traj = Recorder().record(
            agent,
            stub,
            _WEBVOYAGER,
            task={
                "query": "What is the page heading?",
                "start_url": url,
                "max_iter": 2,
                "capture_snapshots": True,
            },
            ctx=CaptureContext(),
        )

        recorded = {o.op for o in traj.tools}
        self.assertLessEqual(
            WIRED_OPS,
            recorded,
            f"agent did not drive all wired ops; recorded={sorted(recorded)}",
        )

        # parity of optimized vs naive on the REAL browser-derived inputs the agent produced
        for call in traj.tools:
            if call.op in WIRED_OPS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ok = get_tool(call.op).parity(
                        call.args, call.kw, reference="naive", candidate="optimized"
                    )
                self.assertTrue(ok, f"{call.op}: optimized != naive on real input")


class TestWebvoyagerLiveLLME2E(_BrowserPageTest):
    """Opt-in, genuine end-to-end run: a REAL LLM (real API calls) drives the real
    WebVoyager agent over a real headless browser to complete a task, while the non-LLM
    PACE Agent tool ops run on the real browser output -- and optimized == naive is asserted on
    those live-driven inputs. Prints a human-readable readout (model/backend, answer, flow,
    token metrics, ops + per-op parity). Triple-gated (PACE_AGENT_LIVE_LLM=1 + a reachable
    endpoint + a working browser); skipped by default so CI never depends on a live model.

    Config via env (all optional except the gate):
      PACE_AGENT_LIVE_LLM=1            opt in
      PACE_AGENT_LIVE_LLM_ENDPOINT    default http://localhost:8000
      PACE_AGENT_LIVE_LLM_MODEL       else auto-discovered from /v1/models or /models
      PACE_AGENT_LIVE_LLM_API_KEY     default "none" (local vLLM); the key for a gateway
      PACE_AGENT_LIVE_LLM_SUB_HEADER  gateway auth header carrying the key (e.g.
                                 Ocp-Apim-Subscription-Key)
      PACE_AGENT_LIVE_LLM_FORMAT      "openai" (default) or "anthropic" (native Messages API)

    The API key is read ONLY from the environment (PACE_AGENT_LIVE_LLM_API_KEY) — it is never
    hard-coded here or committed. Each person supplies their own key via `export`.
    """

    def test_real_llm_drives_agent_and_completes_task(self):
        _require_browser()
        endpoint = _require_live_llm()
        from pace_agent.llm.client import LLMClient
        from pace_agent.profile.profiler import import_agent

        api_key = os.environ.get("PACE_AGENT_LIVE_LLM_API_KEY", "none")
        headers = _live_headers(api_key)
        model = _served_model(endpoint, headers)
        self.assertTrue(
            model, "no model id (set PACE_AGENT_LIVE_LLM_MODEL or expose /models)"
        )

        client = LLMClient(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=0.0,
            seed=0,
            max_tokens=1024,
            headers=headers,
        )
        # "anthropic" routes to the native Messages API (also auto-detected for claude* ids).
        client.image_format = os.environ.get("PACE_AGENT_LIVE_LLM_FORMAT", "openai")

        heading = "Laptop search results"
        url = self._page(_realistic_page(20))
        agent = import_agent(_WEBVOYAGER)
        traj = Recorder().record(
            agent,
            client,
            _WEBVOYAGER,
            task={
                "query": "What is the main page heading? It is the large H1 text at the top.",
                "start_url": url,
                "max_iter": 4,
                "capture_snapshots": True,
            },
            ctx=CaptureContext(),
        )
        result = traj.meta["result"]
        recorded = {c.op for c in traj.tools}

        # opt-in test: print the run's evidence, which a green pass would otherwise hide.
        backend = client._api.kind if client._api else "?"
        print("\nMODEL      :", model, "| backend:", backend)
        print("ANSWER     :", result["answer"])
        print(
            "FLOW       : reached_answer=%s iterations=%s llm_calls=%s"
            % (result["reached_answer"], result["iterations"], result["llm_calls"])
        )
        print("METRICS    :", client.metrics())
        print("OPS DRIVEN :", sorted(recorded))
        print("WIRED OK   :", WIRED_OPS <= recorded, "(need:", sorted(WIRED_OPS), ")")

        self.assertGreaterEqual(len(client.calls), 1, "no live LLM calls were made")
        self.assertTrue(
            traj.llm_tape and any(s and s.strip() for s in traj.llm_tape),
            "empty LLM tape",
        )
        self.assertLessEqual(
            WIRED_OPS,
            recorded,
            f"agent did not drive all wired ops; recorded={sorted(recorded)}",
        )

        # the heading in the answer proves the model actually read the page, not just replied.
        self.assertTrue(result["reached_answer"], f"agent never answered: {result}")
        self.assertIn(heading.lower(), str(result["answer"]).lower())

        # optimized == naive on the REAL, live-driven inputs: a live analogue of
        # TestWebvoyagerOpsE2E's parity check (which drives a deterministic stub LLM).
        for call in traj.tools:
            if call.op in WIRED_OPS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ok = get_tool(call.op).parity(
                        call.args, call.kw, reference="naive", candidate="optimized"
                    )
                print(f"  parity {call.op:16s}: optimized == naive -> {ok}")
                self.assertTrue(
                    ok, f"{call.op}: optimized != naive on live-driven input"
                )


if __name__ == "__main__":
    unittest.main()
