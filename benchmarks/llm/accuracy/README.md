# AMD PACE LLM Accuracy Evaluation

This folder contains scripts and configuration files for evaluating the accuracy of LLM in AMD PACE on various benchmark tasks. The model has been implemented according to the LM Eval Guide, for more details, please refer to the [LM Eval Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md).

> This is only to evaluate the accuracy of LLM models within AMD PACE, but can be extended to evaluate other models if needed.

## Folder Structure

The `benchmarks/llm/accuracy/` directory is organized as follows:

```
benchmarks/llm/accuracy/
├── evaluation_config.json
├── evaluation.py
├── eval.txt
└── evaluation_results/
    └── facebook--opt-125m_bfloat16_bs1_task_mmlu_fewshot_5_limit_1_eval_results.json
```

## Key Files and Directories

*   **[`evaluation_config.json`](evaluation_config.json)**:
    This JSON file is central to configuring an accuracy evaluation run. It specifies various parameters, including:
    *   `model_args`: An object containing model-specific arguments.
        *   `model_name`: The identifier for the LLM to be evaluated (e.g., "Qwen/Qwen2.5-7B-Instruct").
        *   `tokenizer_name`: The identifier for the tokenizer.
        *   `dtype`: The data type for evaluation (e.g., "bf16").
        *   `llm_operators`: Configuration for specific LLM operators.
        *   `spec_config`: Configuration for speculative decoding.
    *   `generation_args`: An object for generation-specific arguments.
        *   `batch_size`: The number of samples to process in a single batch.
        *   `kv_cache_type`: The type of key-value cache to use (e.g., "BMC").
    *   `tasks`: An array defining the benchmark tasks to run. Each task object specifies:
        *   `task_name`: The name of the evaluation task (e.g., "mmlu").
        *   `num_fewshot`: The number of few-shot examples for the task.
        *   `limit`: An optional limit on the number of examples to evaluate.
    *   `verbose`: A boolean flag to control the verbosity of logging.
    *   `output_dir`: The path to the directory where evaluation results will be saved.

*   **[`evaluation.py`](evaluation.py)**:
    This Python script serves as the main entry point for initiating and running the LLM accuracy evaluations. Its primary functions are:
    1.  Parsing the [`evaluation_config.json`](evaluation_config.json) file to load the evaluation settings.
    2.  Loading the specified LLM and its tokenizer.
    3.  Executing the evaluation tasks as defined in the configuration.
    4.  Saving the detailed results of each task to the directory specified by `output_dir` in the configuration.

## Workflow

To run a accuracy evaluation of AMD PACE LLM models using the components in this folder, follow these steps:

1.  **Configure Evaluation**:
    Modify the [`evaluation_config.json`](evaluation_config.json) file to set the desired parameters for your evaluation run. This includes specifying the model, tokenizer, data type, batch size, and the tasks to be evaluated, along with their specific settings (like `num_fewshot` and `limit`).

2.  **Run Evaluation Script**:
    Execute the [`evaluation.py`](evaluation.py) script from the terminal:
    ```bash
    python evaluation.py -c evaluation_config.json
    ```

3.  **Processing**:
    The script will read the configuration, load the necessary model and data, and proceed to evaluate the model on the configured tasks.

4.  **View Results**:
    * Detailed results for each evaluation task will be saved as individual JSON files within the directory specified in `output_dir`.
    * A summary of results might also be printed to the console if `verbose` is set to true in the configuration file.
