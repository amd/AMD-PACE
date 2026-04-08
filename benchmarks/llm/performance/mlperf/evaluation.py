# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content.
# ******************************************************************************
"""
MLPerf Server scenario: accuracy evaluation.
Reads mlperf_log_accuracy.json (token IDs), decodes with tokenizer, computes ROUGE vs references.
"""
import argparse
import json
import logging
import os

import evaluate
import nltk
import numpy as np
from transformers import AutoTokenizer

from dataset import Dataset

log = logging.getLogger(__name__)


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json"
    )
    parser.add_argument("--dataset-file", required=True, help="path to cnn_eval.json")
    parser.add_argument(
        "--dtype",
        default="int32",
        help="dtype of the accuracy log",
        choices=["int32", "int64"],
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name for tokenizer",
    )
    parser.add_argument(
        "--total-sample-count",
        default=13368,
        type=int,
        help="Number of samples in the QSL / accuracy run. Must match the value used when running the accuracy benchmark.",
    )
    parser.add_argument(
        "--output-folder",
        default="output",
        help="path to output folder for accuracy.log",
    )
    args = parser.parse_args()
    return args


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = get_args()
    model_name = args.model_name
    dataset_path = args.dataset_file
    total_sample_count = args.total_sample_count
    metric = evaluate.load("rouge")
    nltk.download("punkt")
    nltk.download("punkt_tab")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_object = Dataset(
        model_name=args.model_name,
        dataset_path=dataset_path,
        total_sample_count=total_sample_count,
        dtype=args.dtype,
    )

    targets = data_object.targets

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    # Deduplicate the results loaded from the json
    dedup_results = []
    seen = set()
    for result in results:
        item = result["qsl_idx"]
        if item not in seen:
            seen.add(item)
            dedup_results.append(result)
    results = dedup_results

    target_required = []
    preds_token_ids = []

    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32

    for pred in results:
        qsl_idx = pred["qsl_idx"]
        target = targets[qsl_idx]
        target_required.append(target)
        preds_token_ids.append(np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype))

    preds_decoded_text = tokenizer.batch_decode(
        preds_token_ids, skip_special_tokens=True
    )

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
    )
    result = {k: f"{round(np.mean(v) * 100, 4)}" for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)

    # Save results to accuracy.log in output folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    accuracy_log_path = os.path.join(output_folder, "accuracy.log")

    with open(accuracy_log_path, "w") as log_file:
        log_file.write("Results\n")
        log_file.write("=" * 50 + "\n")
        for key, value in result.items():
            log_file.write(f"{key}: {value}\n")

    log.info("Results saved to %s", accuracy_log_path)


if __name__ == "__main__":
    main()
