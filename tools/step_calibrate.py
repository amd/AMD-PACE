# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse

from pace.llm.step import StepCalibrationConfig
from pace.llm.step.calibrate import calibrate
from pace.utils.inference_dataset import CALIBRATION_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder a checkpoint's FFN channels by saliency for STeP. Takes an HF checkpoint and writes an HF checkpoint with the same weights, with FFN channels permuted so the most salient come first. The permutation is baked into the weights, so the result loads in any HF-compatible runtime."
    )
    parser.add_argument("model", help="HF hub id or local checkpoint directory")
    parser.add_argument("output", help="directory to write the reordered checkpoint to")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help=f"calibration corpus, repeatable to mix sources: an alias ({', '.join(sorted(CALIBRATION_DATASETS))}), another hub id, or a JSON/JSONL file of dicts. default: wikitext2",
    )
    parser.add_argument("--samples", type=int, default=1024, help="calibration windows")
    parser.add_argument(
        "--sequence-length", type=int, default=2048, help="tokens per window"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="windows per forward pass"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="GPU to calibrate on, cuda or cuda:N. Decoder layers move onto it one at a time, so it needs room for the larger of one layer with the calibration activations and the rest of the model (embeddings, output head, any vision tower), not the whole model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = calibrate(
        StepCalibrationConfig(
            model_name_or_path=args.model,
            output_dir=args.output,
            datasets=args.datasets or ["wikitext2"],
            num_samples=args.samples,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
