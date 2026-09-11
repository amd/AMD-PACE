# AMD PACE LLM Accuracy Evaluation

This folder contains scripts and configuration files for evaluating the accuracy of LLM in AMD PACE on benchmark tasks. The model has been implemented according to the LM Eval Guide, for more details, please refer to the [LM Eval Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md).

> This is only to evaluate the accuracy of LLM models within AMD PACE, but can be extended to evaluate other models if needed.

## Folder Structure

The `benchmarks/llm/accuracy/` directory is organized as follows:

```
benchmarks/llm/accuracy/
├── arguments.py
├── datastructs.py
├── evaluation.py
├── evaluation_config.json
├── evaluation_config_gpt_oss_gsm8k.json
├── lm_model.py
├── README.md
└── evaluation_results/          # created at runtime (output_dir from config)
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
        *   Optional: `apply_chat_template` (default `false`) — format prompts with the tokenizer chat template.
        *   Optional: `think_end_token` (default `null`) — extract text after this marker from generated output (e.g. GPT-OSS `final` channel).
        *   Optional: `system_instruction` (default `null`) — system prompt passed to lm-eval when using a chat template.
        *   Optional: `fewshot_as_multiturn` (default `true`) — format few-shot examples as a multi-turn chat.
    *   `tasks`: An array defining the benchmark tasks to run. Each task object specifies:
        *   `task_name`: The name of the evaluation task (e.g., "gsm8k").
        *   `num_fewshot`: The number of few-shot examples for the task.
        *   `limit`: An optional limit on the number of examples to evaluate.
    *   `verbose`: A boolean flag to control the verbosity of logging.
    *   `output_dir`: The path to the directory where evaluation results will be saved.

*   **[`evaluation_config_gpt_oss_gsm8k.json`](evaluation_config_gpt_oss_gsm8k.json)**:
    GSM8K config for `openai/gpt-oss-20b`. Requires the harmony chat template (`apply_chat_template`), `think_end_token` for the `final` channel, and `SLAB_POOL` + `SLAB` attention.

*   **[`evaluation.py`](evaluation.py)**:
    Main entry point. Parses the config, loads the model via `PaceLLM`, runs lm-eval tasks, and saves results to `output_dir`.

## Workflow

To run an accuracy evaluation of AMD PACE LLM models using the components in this folder, follow these steps:

1.  **Install Accuracy Dependencies**:
    ```bash
    pip install -r benchmarks/llm/accuracy/requirements.txt
    ```

2.  **Configure Evaluation**:
    Modify the [`evaluation_config.json`](evaluation_config.json) file to set the desired parameters for your evaluation run. This includes specifying the model, tokenizer, data type, batch size, and the tasks to be evaluated, along with their specific settings (like `num_fewshot` and `limit`).

3.  **Run Evaluation Script**:
    Execute the [`evaluation.py`](evaluation.py) script from the terminal:
    ```bash
    python evaluation.py -c evaluation_config.json
    ```

    For GPT-OSS on GSM8K (harmony chat template required):
    ```bash
    python evaluation.py -c evaluation_config_gpt_oss_gsm8k.json
    ```

4.  **Processing**:
    The script will read the configuration, load the necessary model and data, and proceed to evaluate the model on the configured tasks.

5.  **View Results**:
    * Detailed results for each evaluation task will be saved as individual JSON files within the directory specified in `output_dir`.
    * A summary of results might also be printed to the console if `verbose` is set to true in the configuration file.
