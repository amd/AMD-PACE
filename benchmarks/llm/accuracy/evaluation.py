# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import json
from typing import Optional, Union, List

import lm_eval.api.registry
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from pace.utils.logging import PACE_LLM_INFO

from arguments import get_args
from lm_model import PaceLLM
from datastructs import ModelArgs, GenerationArgs


def is_built_in_task_valid(task_name: str) -> bool:
    """Checks if a given task name corresponds to a built-in lm_eval task."""
    return task_name in lm_eval.api.registry.TASK_REGISTRY


def eval(
    model_args: ModelArgs,
    tasks: Union[str, List[str]],
    num_fewshot: int,
    generation_args: GenerationArgs,
    limit: Optional[int] = None,
    verbose: Optional[bool] = False,
    output_file: Optional[str] = None,
):
    """Evaluate the model on the given tasks."""

    # Initialize the PaceLLM model
    PACE_LLM_INFO(f"Initializing model {model_args.model_name}")
    model = PaceLLM(model_args, generation_args)

    # Perform the evaluation using lm_eval's simple_evaluate
    PACE_LLM_INFO(
        f"Starting evaluation for task {tasks} with {num_fewshot} fewshot examples"
    )
    evaluation_results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=generation_args.batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    PACE_LLM_INFO(
        f"Evaluation completed for task {tasks} with {num_fewshot} fewshot examples"
    )

    # Print the evaluation results
    if verbose:
        PACE_LLM_INFO(json.dumps(evaluation_results["results"], indent=4))

    # If output file is provided, write the results to the file using json
    if output_file:
        PACE_LLM_INFO(f"Writing evaluation results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)


def main():

    config = get_args()

    PACE_LLM_INFO(f"Running evaluation for model {config['model_args'].model_name}")

    task_manager = TaskManager()

    for task in config["tasks"]:

        task_name = task["task_name"]
        num_fewshot = task["num_fewshot"]
        limit = task["limit"]

        PACE_LLM_INFO(
            f"Running evaluation for task {task_name} with {num_fewshot} fewshot examples and limit {limit}"
        )

        # Check if the task is a valid task
        task_names = task_manager.match_tasks([task_name])
        if task_name not in task_names:
            PACE_LLM_INFO(
                f"Task {task_name} is not a valid task from lm_eval. Skipping evaluation."
            )
            continue

        output_file = None
        if config["output_dir"]:
            model_args = config["model_args"]
            generation_args = config["generation_args"]
            output_filename = (
                f"{model_args.model_name.replace('/', '--')}_"
                f"{str(model_args.dtype).split('.')[-1]}_"
                f"bs{generation_args.batch_size}_"
                f"task_{task_name}_"
                f"fewshot_{num_fewshot}_"
                f"limit_{limit}"
            )
            output_file_prefix = os.path.join(config["output_dir"], output_filename)
            output_file = f"{output_file_prefix}_eval_results.json"

        eval(
            model_args=config["model_args"],
            tasks=task_name,
            num_fewshot=num_fewshot,
            generation_args=config["generation_args"],
            limit=limit,
            verbose=config["verbose"],
            output_file=output_file,
        )


if __name__ == "__main__":
    main()
