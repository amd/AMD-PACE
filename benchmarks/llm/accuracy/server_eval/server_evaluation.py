# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
"""
Online accuracy evaluation against a running PACE server using lm-eval.

Uses lm_eval's ``local-completions`` model type which talks to an
OpenAI-compatible ``/v1/completions`` endpoint. This exercises the logprobs
and echo code-paths in the server.

Usage:
    python server_evaluation.py --config server_evaluation_config.json
"""

import argparse
import json
import os

from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import postprocess_generated_text


@register_model("local-completions-think")
class ThinkLocalCompletionsAPI(LocalCompletionsAPI):
    def __init__(self, think_end_token=None, **kw):
        super().__init__(**kw)
        self.think_end_token = think_end_token

    def _create_payload(self, messages, generate=False, gen_kwargs=None, **kw):
        p = super()._create_payload(
            messages, generate=generate, gen_kwargs=gen_kwargs, **kw
        )
        if generate and self.think_end_token:
            p["skip_special_tokens"] = False
            p["stop"] = [s for s in (p.get("stop") or []) if s.strip() and len(s) > 1]
        return p

    def parse_generations(self, outputs, **kw):
        res = LocalCompletionsAPI.parse_generations(outputs, **kw)
        t = self.think_end_token
        return [postprocess_generated_text(x, None, t) for x in res] if t else res


def main():
    parser = argparse.ArgumentParser(description="Online server evaluation via lm-eval")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to evaluation config JSON.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    base_url = config["server"]["base_url"]
    model_name = config["model_args"]["model_name"]
    tasks_cfg = config["tasks"]
    output_dir = config.get("output_dir")
    verbose = config.get("verbose", False)
    num_concurrent = config.get("num_concurrent", 1)
    max_retries = config.get("max_retries", 3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer_backend = config.get("tokenizer_backend", "huggingface")
    tokenized_requests = str(config.get("tokenized_requests", True)).lower()
    add_bos_token = str(config.get("add_bos_token", True)).lower()
    think_end_token = config.get("think_end_token")
    model_type = "local-completions-think" if think_end_token else "local-completions"

    model_args = (
        f"model={model_name}"
        f",base_url={base_url}/v1/completions"
        f",num_concurrent={num_concurrent}"
        f",max_retries={max_retries}"
        f",tokenized_requests={tokenized_requests}"
        f",add_bos_token={add_bos_token}"
        f",tokenizer_backend={tokenizer_backend}"
    )
    if think_end_token:
        model_args += f",think_end_token={think_end_token}"

    task_manager = TaskManager()

    for task_cfg in tasks_cfg:
        task_name = task_cfg["task_name"]
        num_fewshot = task_cfg.get("num_fewshot", 0)
        limit = task_cfg.get("limit")

        task_names = task_manager.match_tasks([task_name])
        if task_name not in task_names:
            print(f"[WARN] Task '{task_name}' not found in lm_eval registry, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating task={task_name}, fewshot={num_fewshot}, limit={limit}")
        print(f"{'=' * 60}\n")

        try:
            results = evaluator.simple_evaluate(
                model=model_type,
                model_args=model_args,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                limit=limit,
                apply_chat_template=config.get("apply_chat_template", False),
                fewshot_as_multiturn=config.get("fewshot_as_multiturn", False),
                system_instruction=config.get("system_instruction"),
            )
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {task_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

        if verbose and results:
            print(json.dumps(results.get("results", {}), indent=2))

        if output_dir and results:
            safe_model = model_name.replace("/", "--")
            fname = f"{safe_model}_task_{task_name}_fewshot_{num_fewshot}_limit_{limit}_eval_results.json"
            output_path = os.path.join(output_dir, fname)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
