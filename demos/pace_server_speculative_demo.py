# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portion of this file may consist of AI-generated code.
# ******************************************************************************

"""
PACE Speculative Decoding -- Live Curses Grid

Streams multiple prompts concurrently and shows tokens arriving in a live grid.

    pace-server --server_model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
      --spec_config '{"model_name":"amd/PARD-Qwen2.5-0.5B","num_speculative_tokens":12}' \
      --serve_type continuous_prefill_first \
      --scheduler_metrics_enabled True --enable_prometheus

    python pace_server_speculative_demo.py
"""

import asyncio
import curses
import json
import textwrap
import time
from typing import List, Optional

import httpx
from transformers import AutoTokenizer

from pace.utils.logging import PACE_INFO, PACE_ERROR

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFT = "amd/PARD-Qwen2.5-0.5B"
ROUTER_URL = "http://localhost:8080"
MAX_NEW_TOKENS = 200
DISPATCH_INTERVAL = 0.5

PROMPTS = [
    "Write a Python function that checks whether a string is a valid palindrome, ignoring spaces and punctuation:",
    "Solve step by step: A train travels 120 km in 2 hours then increases speed by 20 km/h for the next 3 hours. Total distance?",
    "Q: What are the three laws of thermodynamics?\nA:",
    "Summarize in one paragraph: The Human Genome Project determined the base pairs that make up human DNA and mapped all genes of the human genome.",
    "Explain how a CPU cache works and why it matters for performance:",
    'Translate to French: "The quick brown fox jumps over the lazy dog near the riverbank."',
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain step by step.",
    "List exactly 5 countries in Africa and their capital cities in alphabetical order.",
]

SPEC_CFG = json.dumps({"model_name": DRAFT, "num_speculative_tokens": 12})
LAUNCH_CMD = (
    f"pace-server --server_model {MODEL} --dtype bfloat16 \\\n"
    f"  --spec_config '{SPEC_CFG}' \\\n"
    f"  --serve_type continuous_prefill_first \\\n"
    f"  --scheduler_metrics_enabled True --enable_prometheus"
)

# ── Server check ─────────────────────────────────────────────────────────────


async def check_server() -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = (await client.get(f"{ROUTER_URL}/v1/health")).json()
    except Exception as e:
        PACE_ERROR(f"Cannot reach server: {e}")
        PACE_ERROR(f"Start the server with:\n\n    {LAUNCH_CMD}\n")
        return False
    if not health.get("scheduler_running"):
        PACE_ERROR("Scheduler not running.")
        PACE_ERROR(f"Start the server with:\n\n    {LAUNCH_CMD}\n")
        return False
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            probe = await client.post(
                f"{ROUTER_URL}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": "probe",
                    "stream": False,
                    "gen_config": {"max_new_tokens": 1},
                },
            )
            if (
                probe.status_code == 404
                and probe.json().get("error", {}).get("code") == "model_not_found"
            ):
                PACE_ERROR(f"Model '{MODEL}' not loaded on server.")
                PACE_ERROR(f"Start the server with:\n\n    {LAUNCH_CMD}\n")
                return False
    except Exception:
        pass
    PACE_INFO(f"Server healthy (queue={health.get('queue_size', 0)})")
    return True


# ── Tile state ───────────────────────────────────────────────────────────────


class Tile:
    __slots__ = (
        "label",
        "chunks",
        "tokens",
        "text",
        "start",
        "first_token_time",
        "elapsed",
        "done",
        "error",
    )

    def __init__(self, label: str):
        self.label = label
        self.chunks: int = 0
        self.tokens: int = 0
        self.text: str = ""
        self.start: float = 0.0
        self.first_token_time: Optional[float] = None
        self.elapsed: float = 0.0
        self.done: bool = False
        self.error: Optional[str] = None

    @property
    def ttft(self) -> float:
        return (self.first_token_time - self.start) if self.first_token_time else 0.0

    @property
    def tps(self) -> float:
        return self.tokens / self.elapsed if self.elapsed > 0 else 0.0


# ── Streaming worker ─────────────────────────────────────────────────────────


async def stream_to_tile(prompt: str, tile: Tile, tokenizer):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "gen_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "do_sample": False,
        },
    }
    hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    tile.start = time.time()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{ROUTER_URL}/v1/completions", json=payload, headers=hdrs
            ) as resp:
                resp.raise_for_status()
                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            break
                        try:
                            obj = json.loads(line[6:])
                            content = (
                                obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if content:
                                if tile.first_token_time is None:
                                    tile.first_token_time = time.time()
                                tile.chunks += 1
                                tile.tokens += len(
                                    tokenizer.encode(content, add_special_tokens=False)
                                )
                                tile.text += content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        tile.error = str(e)
    finally:
        tile.elapsed = time.time() - tile.start
        tile.done = True


# ── Curses rendering ─────────────────────────────────────────────────────────

TILE_COLORS = [
    curses.COLOR_GREEN,
    curses.COLOR_BLUE,
    curses.COLOR_YELLOW,
    curses.COLOR_MAGENTA,
    curses.COLOR_CYAN,
    curses.COLOR_RED,
    curses.COLOR_WHITE,
    curses.COLOR_GREEN,
]


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    for i, fg in enumerate(TILE_COLORS):
        curses.init_pair(i + 1, fg, -1)
    curses.init_pair(len(TILE_COLORS) + 1, curses.COLOR_WHITE, -1)
    curses.init_pair(len(TILE_COLORS) + 2, curses.COLOR_BLACK, curses.COLOR_WHITE)


def _tile_attr(idx):
    return curses.color_pair((idx % len(TILE_COLORS)) + 1)


def _dim_attr():
    return curses.color_pair(len(TILE_COLORS) + 1) | curses.A_DIM


def _bar_attr():
    return curses.color_pair(len(TILE_COLORS) + 2) | curses.A_BOLD


def _put(win, y, x, text, attr=0):
    h, w = win.getmaxyx()
    avail = w - x - 1
    if avail <= 0 or y < 0 or y >= h:
        return
    try:
        win.addnstr(y, x, text, avail, attr)
    except curses.error:
        pass


def draw_tile(scr, tile: Tile, idx: int, top: int, left: int, h: int, w: int):
    attr, dim, bold = _tile_attr(idx), _dim_attr(), _tile_attr(idx) | curses.A_BOLD
    inner = w - 4

    _put(scr, top, left, "\u250c" + "\u2500" * (w - 2) + "\u2510", dim)
    _put(scr, top + 1, left, "\u2502", dim)
    _put(scr, top + 1, left + 1, f" {idx + 1}. {tile.label} "[: w - 2], bold)
    _put(scr, top + 1, left + w - 1, "\u2502", dim)
    _put(scr, top + 2, left, "\u2502" + "\u2504" * (w - 2) + "\u2502", dim)

    body_top, body_h = top + 3, h - 5
    lines = textwrap.wrap(tile.text, width=inner) if tile.text else []
    if len(lines) > body_h:
        lines = lines[-body_h:]
    for r in range(body_h):
        _put(scr, body_top + r, left, "\u2502", dim)
        _put(scr, body_top + r, left + 1, " " * (w - 2), attr)
        if r < len(lines):
            _put(scr, body_top + r, left + 2, lines[r], attr)
        _put(scr, body_top + r, left + w - 1, "\u2502", dim)

    sy = top + h - 2
    _put(scr, sy, left, "\u2502", dim)
    _put(scr, sy, left + 1, " " * (w - 2), dim)
    if tile.error:
        _put(scr, sy, left + 2, f"ERR: {tile.error}"[:inner], curses.color_pair(6))
    elif tile.done:
        _put(
            scr,
            sy,
            left + 2,
            f"\u2713 {tile.tokens} tok  {tile.chunks} steps  {tile.elapsed:.1f}s  {tile.tps:.1f} t/s  TTFT {tile.ttft:.2f}s"[
                :inner
            ],
            bold,
        )
    else:
        el = time.time() - tile.start if tile.start else 0
        sp = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"[
            int(el * 8) % 10
        ]
        s = f"{sp} {tile.tokens} tok  {tile.chunks} steps  {el:.1f}s"
        if tile.tps > 0:
            s += f"  {tile.tps:.1f} t/s"
        _put(scr, sy, left + 2, s[:inner], attr)
    _put(scr, sy, left + w - 1, "\u2502", dim)
    _put(scr, top + h - 1, left, "\u2514" + "\u2500" * (w - 2) + "\u2518", dim)


def draw_header(scr, n, wall_start):
    _, w = scr.getmaxyx()
    el = time.time() - wall_start if wall_start else 0
    _put(
        scr,
        0,
        0,
        f"  PACE Speculative Decoding  \u2502  {MODEL}  \u2502  {n} prompts  \u2502  {el:.1f}s  ".ljust(
            w
        ),
        _bar_attr(),
    )


def draw_footer(scr, tiles, wall_start):
    h, w = scr.getmaxyx()
    done = sum(1 for t in tiles if t.done)
    tok = sum(t.tokens for t in tiles)
    steps = sum(t.chunks for t in tiles)
    el = time.time() - wall_start if wall_start else 0
    tps = tok / el if el > 0 else 0
    _put(
        scr,
        h - 1,
        0,
        f"  Done: {done}/{len(tiles)}  \u2502  {tok} tok  {steps} steps  \u2502  {tps:.1f} tok/s  ".ljust(
            w
        ),
        _bar_attr(),
    )


# ── Curses main loop ─────────────────────────────────────────────────────────


async def curses_main(stdscr, tokenizer):
    curses.curs_set(0)
    stdscr.nodelay(True)
    _init_colors()

    n = len(PROMPTS)
    cols, rows = 2, (n + 1) // 2
    tiles: List[Tile] = []
    for p in PROMPTS:
        short = p[:60].replace("\n", " ")
        tiles.append(Tile(short + "\u2026" if len(p) > 60 else short))

    wall_start = time.time()
    tasks: List[asyncio.Task] = []
    for i, p in enumerate(PROMPTS):
        tasks.append(asyncio.create_task(stream_to_tile(p, tiles[i], tokenizer)))
        if i < n - 1:
            await asyncio.sleep(DISPATCH_INTERVAL)

    while not all(t.done for t in tiles):
        stdscr.erase()
        sh, sw = stdscr.getmaxyx()
        draw_header(stdscr, n, wall_start)
        tw, th = sw // cols, max(8, (sh - 2) // rows)
        for i, tile in enumerate(tiles):
            draw_tile(stdscr, tile, i, 1 + (i // cols) * th, (i % cols) * tw, th, tw)
        draw_footer(stdscr, tiles, wall_start)
        stdscr.refresh()
        try:
            if stdscr.getch() == ord("q"):
                for t in tasks:
                    t.cancel()
                break
        except curses.error:
            pass
        await asyncio.sleep(0.05)

    stdscr.erase()
    sh, sw = stdscr.getmaxyx()
    draw_header(stdscr, n, wall_start)
    tw, th = sw // cols, max(8, (sh - 2) // rows)
    for i, tile in enumerate(tiles):
        draw_tile(stdscr, tile, i, 1 + (i // cols) * th, (i % cols) * tw, th, tw)
    draw_footer(stdscr, tiles, wall_start)
    stdscr.refresh()
    stdscr.nodelay(False)
    _put(
        stdscr,
        sh - 1,
        0,
        "  Press any key to see summary \u2026".ljust(sw),
        _bar_attr(),
    )
    stdscr.refresh()
    stdscr.getch()

    await asyncio.gather(*tasks, return_exceptions=True)
    return tiles, time.time() - wall_start


def print_summary(tiles: List[Tile], wall: float, tokenizer):
    ok = [t for t in tiles if not t.error]
    total_tok = sum(len(tokenizer.encode(t.text, add_special_tokens=False)) for t in ok)
    total_steps = sum(t.chunks for t in ok)
    avg_ttft = sum(t.ttft for t in ok) / len(ok) if ok else 0
    avg_accept = total_tok / total_steps if total_steps else 0
    PACE_INFO("\u2501" * 50)
    PACE_INFO(
        f"PARD Summary \u2014 {len(ok)}/{len(tiles)} ok, {total_tok} tok, "
        f"{total_steps} steps, {avg_accept:.1f} tok/step, "
        f"{wall:.1f}s wall, {total_tok / wall:.1f} tok/s, avg TTFT {avg_ttft:.3f}s"
    )
    PACE_INFO("\u2501" * 50)


# ── Entry ────────────────────────────────────────────────────────────────────


async def main():
    PACE_INFO(f"Model: {MODEL} | Draft: {DRAFT} | Router: {ROUTER_URL}")
    if not await check_server():
        return

    PACE_INFO("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    PACE_INFO(f"Launching grid with {len(PROMPTS)} prompts (press q to quit)")

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    try:
        tiles, wall = await curses_main(stdscr, tokenizer)
    finally:
        stdscr.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

    if tiles:
        print_summary(tiles, wall, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
