# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portion of this file may consist of AI-generated code.
# ******************************************************************************

import argparse
import asyncio
import time
import httpx
import os
from typing import Dict, Any


os.environ["PACE_LOG_LEVEL"] = "none"

PROMPT_COLOR = "\033[94m"
RESPONSE_COLOR = "\033[92m"
ERROR_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"


async def call_api_router(
    model_name: str,
    router_url: str,
    prompt: str,
    gen_kwargs: Dict[str, Any],
    request_num: int,
    total_requests: int,
) -> str:
    """
    Call the router asynchronously with a single prompt and generation parameters.
    Prints the prompt when sent and output when received.
    """

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "gen_config": {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 50),
            "temperature": gen_kwargs.get("temperature", 0.7),
            "top_p": gen_kwargs.get("top_p", 1.0),
            "top_k": gen_kwargs.get("top_k", 50),
            "seed": gen_kwargs.get("seed", None),
            "do_sample": gen_kwargs.get("do_sample", True),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.0),
            "frequency_penalty": gen_kwargs.get("frequency_penalty", 0.0),
            "stop": gen_kwargs.get("stop_strings", "\n\n"),
        },
    }

    headers = {"Content-Type": "application/json"}

    print(
        f"\n{PROMPT_COLOR}[{request_num}/{total_requests}] Sending: {prompt}{RESET_COLOR}"
    )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{router_url}/v1/completions",
                headers=headers,
                json=payload,
                timeout=500,
            )
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0]["message"]["content"]
                print(
                    f"{RESPONSE_COLOR}[{request_num}/{total_requests}] Received: {output}{RESET_COLOR}"
                )
                return output
            else:
                print(
                    f"{ERROR_COLOR}[{request_num}/{total_requests}] Warning: Unexpected response format{RESET_COLOR}"
                )
                return ""
        except Exception as e:
            print(
                f"{ERROR_COLOR}[{request_num}/{total_requests}] Error: {type(e).__name__}: {e}{RESET_COLOR}"
            )
            return ""


async def generate_and_time(
    model_name, router_url, input_prompts, gen_kwargs, interval=0.5
):
    start = time.time()

    tasks = []
    for i, prompt in enumerate(input_prompts, 1):
        task = asyncio.create_task(
            call_api_router(
                model_name, router_url, prompt, gen_kwargs, i, len(input_prompts)
            )
        )
        tasks.append(task)
        await asyncio.sleep(interval)

    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"\n{'=' * 80}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"Average time per request: {elapsed / len(input_prompts):.2f} seconds")
    print(f"Successful requests: {sum(1 for r in results if r)} / {len(results)}")
    print(f"{'=' * 80}\n")
    return results


async def run_pace_llm(args):
    """
    Run the PACE model via async API calls to the router.
    """
    router_url = f"http://{args.router_host}:{args.router_port}"

    model_name = args.model
    inputs_str = [
        "A lone astronaut discovers a hidden message on Mars,",
        "The world's last bookstore receives a mysterious",
        "Suddenly, all clocks in the city stop at the same",
        "2 + 5 =",
        "The American Civil War was fought",
        "The cat stared at the empty hallway, waiting for",
        "Rain fell softly on the old wooden roof,",
        "A forgotten letter was discovered",
        "The lighthouse blinked twice before",
        "She found a key taped to the back of",
        "The elevator stopped at a floor that",
        "Every mirror in the house cracked",
        "The radio played a song no one",
        "A single red balloon floated",
        "The library's clock chimed",
        "The garden gate creaked open",
        "He woke up to find his reflection missing",
        "The phone rang, but there was only",
        "A trail of footprints led into the woods, but",
        "The candle flickered even though",
        "She wrote a message in the fogged-up window, but",
        "The old photograph showed someone standing behind them who",
        "The streetlights blinked out one by one as",
        "A strange melody drifted in from.",
        "The shadows on the wall didn't match",
    ]

    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1.0,
        "seed": 123,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "stop_strings": ["\n\n"],
    }

    launch_cmd = (
        f"pace-server --server_model {model_name} --dtype bfloat16 \\\n"
        f"      --serve_type {args.scheduler_type} \\\n"
        f"      --scheduler_metrics_enabled True --enable_prometheus"
    )

    print(f"\n{'=' * 80}")
    print("PACE Model Inference Test")
    print(f"{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"Router URL: {router_url}")
    print(f"Scheduler Type: {args.scheduler_type}")
    print(f"Total Prompts: {len(inputs_str)}")
    print(f"{'=' * 80}\n")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{router_url}/v1/health")
            if response.status_code != 200:
                print(
                    f"{ERROR_COLOR}Warning: Router returned status code {response.status_code}{RESET_COLOR}"
                )

            probe = await client.post(
                f"{router_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": "probe",
                    "stream": False,
                    "gen_config": {"max_new_tokens": 1},
                },
                timeout=60,
            )
            if probe.status_code == 404:
                body = probe.json()
                if body.get("error", {}).get("code") == "model_not_found":
                    print(
                        f"{ERROR_COLOR}Model '{model_name}' is not loaded on the server.{RESET_COLOR}"
                    )
                    print(f"\nStart the server with:\n\n    {launch_cmd}\n")
                    return
            print(
                f"{RESPONSE_COLOR}Router available -- model '{model_name}' confirmed{RESET_COLOR}"
            )
    except Exception as e:
        print(f"{ERROR_COLOR}Cannot reach server: {e}{RESET_COLOR}")
        print(f"\nStart the server with:\n\n    {launch_cmd}\n")
        return

    await generate_and_time(model_name, router_url, inputs_str, gen_kwargs, interval=3)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PACE Server Basic Example - Send requests to PACE router"
    )
    parser.add_argument(
        "--router_host",
        type=str,
        default="localhost",
        help="Router host address (default: localhost)",
    )
    parser.add_argument(
        "--router_port",
        type=int,
        default=8080,
        help="Router port (default: 8080)",
    )
    parser.add_argument(
        "--server_host",
        type=str,
        default="localhost",
        help="Engine server host address (default: localhost)",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8000,
        help="Engine server port (default: 8000)",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="iterative",
        choices=["iterative", "continuous_prefill_first"],
        help="Type of scheduler to use (default: iterative)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-6.7b",
        help="Model name to use (default: facebook/opt-6.7b)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(run_pace_llm(args))
