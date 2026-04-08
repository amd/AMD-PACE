# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portion of this file may consist of AI-generated code.
# ******************************************************************************

"""
Interactive streaming generation demo for the AMD PACE inference server.

Maintains conversation context by concatenating prior turns into the prompt.
Uses Llama-3 chat template. Streaming only.

Start the server, then run:

    python server_chat_demo.py
"""

import asyncio
import json
import os
import time
from typing import List

import httpx

os.environ["PACE_LOG_LEVEL"] = "none"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ROUTER_URL = "http://localhost:8080"
MAX_NEW_TOKENS = 4096
SYSTEM_PROMPT = "You are a helpful, accurate, and concise AI assistant."

LAUNCH_CMD = (
    f"pace-server --server_model {MODEL} --dtype bfloat16 \\\n"
    f"  --scheduler_metrics_enabled True --enable_prometheus"
)


# ── Prompt building (Llama-3 chat template) ─────────────────────────────────


def build_prompt(system: str, history: List[dict], user_msg: str) -> str:
    parts = [
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
    ]
    for turn in history:
        parts.append(
            f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n"
            f"{turn['content']}<|eot_id|>"
        )
    parts.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return "".join(parts)


STOP_SEQUENCES = ["<|eot_id|>", "<|end_of_text|>"]


# ── Server check ────────────────────────────────────────────────────────────


async def check_server() -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = (await client.get(f"{ROUTER_URL}/v1/health")).json()
    except Exception as e:
        print(f"\n  \033[91m✗\033[0m  Cannot reach server: {e}\n\n    {LAUNCH_CMD}\n")
        return False
    if not health.get("scheduler_running"):
        print(f"\n  \033[91m✗\033[0m  Scheduler not running.\n\n    {LAUNCH_CMD}\n")
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
                print(
                    f"\n  \033[91m✗\033[0m  Model '{MODEL}' not loaded on server.\n\n    {LAUNCH_CMD}\n"
                )
                return False
    except Exception:
        pass
    return True


# ── Streaming generation ────────────────────────────────────────────────────


async def generate(history: List[dict], user_msg: str) -> str:
    prompt = build_prompt(SYSTEM_PROMPT, history, user_msg)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "gen_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "do_sample": False,
            "stop": STOP_SEQUENCES,
        },
    }
    hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    full_text = ""
    t0 = time.time()
    first_tok = None

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
                            delta = (
                                obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                if first_tok is None:
                                    first_tok = time.time()
                                full_text += delta
                                print(delta, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"\n  \033[91m[ERR] {e}\033[0m")

    elapsed = time.time() - t0
    ttft = (first_tok - t0) if first_tok else elapsed
    print(f"\n\033[2m  [{elapsed:.1f}s, TTFT: {ttft:.2f}s]\033[0m")
    return full_text.strip()


# ── Main loop ───────────────────────────────────────────────────────────────


async def main():
    if not await check_server():
        return

    print("\033[1;36m" + "━" * 50)
    print("  PACE Interactive Generation")
    print("━" * 50 + "\033[0m")
    print(f"  Model : {MODEL}")
    print(f"  Server: {ROUTER_URL}")
    print("  Type 'exit' or Ctrl+C to quit, '/clear' to reset.\n")

    history: List[dict] = []

    while True:
        try:
            user_input = input("\033[1;34m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/quit"):
            print("Goodbye!")
            break
        if user_input == "/clear":
            history.clear()
            print("  \033[2mHistory cleared.\033[0m")
            continue

        reply = await generate(history, user_input)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    asyncio.run(main())
