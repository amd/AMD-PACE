# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import argparse
from typing import Sequence

from transformers import TextStreamer

from pace.llm import LLMModel, SamplingConfig, StepGenerationBackendConfig
from pace.utils.dtype import TorchDtypeResolver

GREEN = "\033[32m"
GREY = "\033[90m"
RESET = "\033[0m"


class AcceptedTokenStreamer(TextStreamer):
    """
    Streams each round with the draft tokens the full model accepted in green and the token it added itself in grey.

    A round commits the accepted draft tokens followed by one bonus token from the full model, so every token of a round but the last came from the draft. The exception is a final round cut short by the length limit, whose last token shows grey although the draft supplied it.

    Each span is decoded as a suffix of the whole run rather than on its own, since a token is not generally a whole character and some tokenizers carry a word's leading space in the token.
    """

    def put(self, value) -> None:
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        tokens = value.reshape(-1).tolist()
        draft = self._extend(tokens[:-1])
        bonus = self._extend(tokens[-1:])
        self.on_finalized_text(paint(draft, GREEN) + paint(bonus, GREY))

    def _extend(self, token_ids: Sequence[int]) -> str:
        """Append tokens to the run and return the text they add."""
        if not token_ids:
            return ""
        self.token_cache.extend(token_ids)
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        # A trailing replacement character is a multi-byte sequence that a later token completes.
        if text.endswith("\ufffd"):
            return ""
        printable = text[self.print_len :]
        if text.endswith("\n"):
            self.token_cache = []
            self.print_len = 0
        else:
            self.print_len = len(text)
        return printable


def paint(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if text else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate with STeP, drafting with the GPU-resident FFN channels and verifying with the CPU residual. Tokens the draft supplied stream in green, the bonus token the full model produced directly in grey."
    )
    parser.add_argument(
        "model",
        help="HF hub id or local directory; a checkpoint reordered by tools/step_calibrate.py drafts considerably better",
    )
    parser.add_argument(
        "--prompt", default="Write a Python function that merges two sorted lists."
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="send the prompt as written instead of through the model's chat template, for completion-style prompts",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", help="bfloat16, float16 or float32"
    )

    split = parser.add_argument_group("channel split")
    split.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="FFN fraction on the GPU. Omit to derive it from the VRAM budget",
    )
    split.add_argument(
        "--vram-budget-gb",
        type=float,
        default=None,
        help="VRAM the runtime may use. Omit to use what is free on the device minus 1 GiB",
    )
    split.add_argument("--max-seq-length", type=int, default=4096)
    split.add_argument("--gpu", type=int, default=0, help="device ordinal")
    split.add_argument(
        "--cpu-threads", type=int, default=None, help="threads for the CPU residual"
    )
    split.add_argument(
        "--residual-backend",
        default="auto",
        choices=("auto", "pace", "torch"),
        help="pace requires the libxsmm kernels and aligns the split for them; torch honours the requested ratio exactly",
    )

    tree = parser.add_argument_group("speculation")
    tree.add_argument("--tree-width", type=int, default=3)
    tree.add_argument("--tree-depth", type=int, default=3)
    tree.add_argument("--tree-mode", default="narrow", choices=("narrow", "full"))
    tree.add_argument(
        "--speculation",
        default="auto",
        choices=("on", "off", "auto"),
        help="off decodes with plain tensor parallelism and no draft; auto stops speculating if the yield falls below --min-tokens-per-step",
    )
    tree.add_argument("--min-tokens-per-step", type=float, default=None)
    tree.add_argument("--speculation-window", type=int, default=None)
    tree.add_argument(
        "--draft-kv-window",
        type=int,
        default=None,
        help="cap the context the draft attends to, bounding drafting cost at long context",
    )
    tree.add_argument(
        "--draft-kv-sink",
        type=int,
        default=256,
        help="leading tokens the draft always keeps when windowing, so it does not lose the prompt instruction",
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 decodes greedily. Above 0 samples, with top_p and top_k from the checkpoint's generation_config.json unless given",
    )
    sampling.add_argument("--top-p", type=float, default=None)
    sampling.add_argument("--top-k", type=int, default=None)
    sampling.add_argument("--seed", type=int, default=None)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--no-highlight",
        action="store_true",
        help="stream plain text instead of colouring tokens by origin",
    )
    output.add_argument(
        "--timings",
        action="store_true",
        help="break decode time into draft, verify and CPU FFN, at the cost of a device synchronize per phase",
    )
    return parser.parse_args()


def backend_config(args: argparse.Namespace) -> StepGenerationBackendConfig:
    settings = dict(
        draft_ratio=args.ratio,
        vram_budget_gb=args.vram_budget_gb,
        max_seq_length=args.max_seq_length,
        tree_width=args.tree_width,
        tree_depth=args.tree_depth,
        tree_mode=args.tree_mode,
        gpu_index=args.gpu,
        cpu_threads=args.cpu_threads,
        residual_backend=args.residual_backend,
        speculation=args.speculation,
        draft_kv_window=args.draft_kv_window,
        draft_kv_sink=args.draft_kv_sink,
        collect_timings=args.timings,
    )
    if args.min_tokens_per_step is not None:
        settings["min_tokens_per_step"] = args.min_tokens_per_step
    if args.speculation_window is not None:
        settings["speculation_window"] = args.speculation_window
    return StepGenerationBackendConfig(**settings)


def sampling_config(args: argparse.Namespace) -> SamplingConfig:
    # Set explicitly, since a checkpoint whose generation_config.json leaves do_sample off would otherwise decode greedily at any temperature.
    settings = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
    )
    if args.top_p is not None:
        settings["top_p"] = args.top_p
    if args.top_k is not None:
        settings["top_k"] = args.top_k
    if args.seed is not None:
        settings["seed"] = args.seed
    return SamplingConfig(**settings)


def encode(tokenizer, prompt: str, chat: bool):
    """
    Tokenize the prompt, through the chat template when the model has one.

    Instruction-tuned models degenerate on a raw prompt, so the template is the default.
    """
    if not chat or not getattr(tokenizer, "chat_template", None):
        return tokenizer(prompt, return_tensors="pt")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # The template already carries the model's BOS.
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def main() -> None:
    args = parse_args()
    model = LLMModel(
        args.model,
        dtype=TorchDtypeResolver.resolve(args.dtype),
        generation_backend_config=backend_config(args),
    )
    tokenizer = model.get_tokenizer()
    inputs = encode(tokenizer, args.prompt, chat=not args.no_chat)
    streamer_class = TextStreamer if args.no_highlight else AcceptedTokenStreamer
    streamer = streamer_class(tokenizer, skip_prompt=True, skip_special_tokens=True)

    output = model.generate(inputs, sampling_config(args), text_streamer=streamer)

    stats = output.speculative_stats
    print(
        f"\naccepted {stats.accepted} of {stats.generated} tokens from the draft, {stats.proposed} proposed"
    )
    print(stats.describe())


if __name__ == "__main__":
    main()
