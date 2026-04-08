# MLPerf LLM Inference — Server Scenario

This directory provides scripts to run **MLPerf Inference** for the **Server scenario** (online latency and throughput). The system under test (SUT) sends prompts to a PACE or vLLM server and reports token streams and latency to LoadGen.

**Supported modes:** Performance only, Accuracy only (with optional ROUGE evaluation).

---

## Prerequisites

- Python 3.12 (recommended)
- Conda
- **A PACE or vLLM inference server** — you must start it yourself. The benchmark only sends requests to the server; it does not start or manage the server.

---

## Setup

**Use a dedicated Conda environment named `mlperf`.** Do not use the main project environment for this benchmark; dependencies and versions differ.

### 1. Create the `mlperf` environment and install dependencies

All dependencies (including `mlcommons-loadgen`, `rouge_score`, etc.) are in `requirements.txt`:

```bash
conda create -n mlperf python=3.12 -y
conda activate mlperf
cd /path/to/amd_pace/benchmarks/llm/performance/mlperf
pip install -r requirements.txt
```

### 2. Inference server

Start a vLLM or PACE server separately (the benchmark does not start one). Point the benchmark at it with `ENDPOINT_URL` / `--endpoint-url` (default: `http://localhost:8080/v1/completions`).

### 3. Dataset

Download the CNN/DailyMail evaluation dataset (e.g. via [MLCommons r2-downloader](https://github.com/mlcommons/r2-downloader)) and place it at `data/cnn_eval.json`, or set `DATASET_PATH` when running.

Example (r2-downloader):

```bash
mkdir -p data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
# Move or symlink the resulting file to data/cnn_eval.json as expected by the benchmark.
```

### 4. Model (for tokenizer and optional local use)

The SUT uses the tokenizer for the configured model; the model itself runs on the server. To download the tokenizer (and optionally weights) locally:

```bash
python download_model.py
```

This downloads the Llama-3.1-8B-Instruct tokenizer (and model) into `model/`. If you use a different `--model-path`, ensure that model/tokenizer is available from the Hugging Face hub.

### 5. LoadGen config

Edit `user.conf` to control the run:

- **Performance:** `*.Server.min_query_count`, `*.Server.max_query_count`, `*.Server.target_qps`
- Run length is query-count based (durations are set so query count is the limit).

Default `user.conf` ships with 32 min/max queries and target QPS 32; change as needed(use all samples for representative runs).

**Adding more keys:** Entries are `key = value`. Keys use the form `model.scenario.key` (e.g. `*.Server.target_qps`); use `*` to match all models or scenarios. The full set of keys and how they are parsed is defined in the LoadGen source: [mlcommons/inference loadgen](https://github.com/mlcommons/inference/tree/master/loadgen) — see `test_settings.h` for the settings structure and `test_settings_internal.cc` (function `TestSettings::FromConfig`) for the exact key names and which scenarios they apply to. The [LoadGen README](https://github.com/mlcommons/inference/blob/master/loadgen/README.md) also points to Test Settings as the reference for scenarios, modes, and knobs.

---

## How to Run

### Option A: Shell script (recommended)

From this directory:

```bash
conda activate mlperf
./run_mlperf.sh
```

Defaults: performance mode; **QSL** is sized by `TOTAL_SAMPLE_COUNT` (13368). **user.conf** min/max_query_count (32) selects how many queries are issued from the QSL. Endpoint `http://localhost:8080/v1/completions`, server type `pace`. Override with environment variables or by editing the variables at the top of `run_mlperf.sh`.

**Examples:**

```bash
# Performance, 32 queries (default; min/max_query_count in user.conf)
./run_mlperf.sh

# Accuracy, full dataset (13368 samples) — use for official/submission accuracy
MODE=accuracy TOTAL_SAMPLE_COUNT=13368 ./run_mlperf.sh

# Accuracy, smaller set (e.g. 32 samples for testing); use same TOTAL_SAMPLE_COUNT in evaluation.py
MODE=accuracy TOTAL_SAMPLE_COUNT=32 LOG_FILE=server_pace.log ./run_mlperf.sh

# Custom endpoint and dataset
ENDPOINT_URL=http://192.168.1.10:8080/v1/completions DATASET_PATH=/path/to/cnn_eval.json ./run_mlperf.sh

# vLLM server
SERVER_TYPE=vllm ./run_mlperf.sh
```

**Script variables (override via env or edit script):**  
`CHECKPOINT_PATH`, `DATASET_PATH`, `ENDPOINT_URL`, `TIMEOUT_SEC`, `MAX_TOKENS`, `SERVER_TYPE`, `MODE` (performance | accuracy), `TOTAL_SAMPLE_COUNT`, `OUTPUT_LOG_DIR`, `USER_CONF`, `LOG_FILE` (optional tee), and others — see `run_mlperf.sh`.

### Option B: Python directly

```bash
conda activate mlperf
python main.py --dataset-path data/cnn_eval.json [options]
```

Use `--accuracy` for accuracy mode; omit for performance. All options: `python main.py --help`.

---

## Sample count vs query count

- **`--total-sample-count`** (and script `TOTAL_SAMPLE_COUNT`) sets the **QSL size**: how many samples are loaded. In **accuracy** mode, LoadGen runs over the whole QSL, so the number of prompts is this value.
- In **performance** mode, the **QSL** holds `--total-sample-count` samples; **user.conf** `*.Server.min_query_count` and `*.Server.max_query_count` select how many queries LoadGen issues from that QSL (e.g. 32 out of 13368). Ensure the QSL has at least that many samples so that the same requests are not repeated.

**Accuracy mode:** For full CNN/DailyMail accuracy evaluation, use **total sample count 13368** (the full dataset). Use the same value when running the evaluation script (see below).

**Smaller sample set:** To run accuracy or evaluation on a subset (e.g. for testing), set the same sample count in both places: when running the benchmark and when running the evaluation script. For example, for 32 samples: run the accuracy benchmark with `TOTAL_SAMPLE_COUNT=32` (or `--total-sample-count 32`), then run `evaluation.py` with `--total-sample-count 32`. The `--total-sample-count` in `evaluation.py` must match the value used in the accuracy run.

---

## Accuracy evaluation (ROUGE)

After an accuracy run, LoadGen writes `output/mlperf_log_accuracy.json` (or the path given by `--output-log-dir`). To compute ROUGE vs references, run `evaluation.py` with **the same `--total-sample-count`** you used for the accuracy run.

**Full dataset (13368 samples):**

```bash
conda activate mlperf
python evaluation.py \
  --mlperf-accuracy-file output/mlperf_log_accuracy.json \
  --dataset-file data/cnn_eval.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --total-sample-count 13368 \
  --dtype int32
```

**Smaller sample set:** If you ran accuracy with a smaller count (e.g. 32), pass the same value to the evaluation script:

```bash
python evaluation.py \
  --mlperf-accuracy-file output/mlperf_log_accuracy.json \
  --dataset-file data/cnn_eval.json \
  --total-sample-count 32 \
  --dtype int32
```

Defaults: `--model-name meta-llama/Llama-3.1-8B-Instruct`, `--total-sample-count 13368`, `--dtype int32`, `--output-folder output`. Results are printed and written to `output/accuracy.log`.

---

## Outputs

- **Performance:** LoadGen writes logs and summary under `--output-log-dir` (default `output/`).
- **Accuracy:** `mlperf_log_accuracy.json` in that directory; then run `evaluation.py` to get ROUGE and `accuracy.log`.

---

## File overview

| File | Purpose |
|------|--------|
| `main.py` | Entry point; builds SUT, QSL, and runs LoadGen. |
| `run_mlperf.sh` | Convenience wrapper with env-based defaults; runs `main.py`. |
| `SUT_PACE_async.py` | SUT implementation (async, streaming; PACE and vLLM). |
| `dataset.py` | Dataset loader and QSL callbacks. |
| `user.conf` | LoadGen Server settings (min/max query count, target QPS). |
| `evaluation.py` | Post-accuracy ROUGE evaluation. |
| `download_model.py` | Downloads tokenizer (and model) into `model/`. |
