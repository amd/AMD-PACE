# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

# References used:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py
# https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/vllm_causallms.py
# https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/neuralmagic.py

from tqdm import tqdm
import copy

from typing import Union, Optional, List, Tuple
import torch
from transformers.tokenization_utils_base import BatchEncoding
from lm_eval.api.registry import register_model
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import chunks, pad_and_concat
from lm_eval.utils import Reorderer

from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    OperatorConfig,
    PardSpecDecodeConfig,
)
from pace.utils.logging import PACE_LLM_WARNING, suppress_logging_fn

from datastructs import ModelArgs, GenerationArgs


@register_model("pace")
class PaceLLM(LM):

    def __init__(
        self,
        model_args: ModelArgs,
        generation_args: GenerationArgs,
        max_length: Optional[int] = None,
        max_gen_toks: Optional[int] = 256,
    ):
        super().__init__()

        kv_upper = generation_args.kv_cache_type.upper()
        if kv_upper == "BMC":
            kv_cache_type = KVCacheType.BMC
        elif kv_upper == "PAGED":
            kv_cache_type = KVCacheType.PAGED
        elif kv_upper == "SLAB_POOL":
            kv_cache_type = KVCacheType.SLAB_POOL
        else:
            kv_cache_type = KVCacheType.DYNAMIC

        batch_size = generation_args.batch_size
        if isinstance(batch_size, str) and not batch_size.isdigit():
            PACE_LLM_WARNING(
                f"batch_size={batch_size} is not valid for deepsparse because it is not an integer. "
                "Ignoring and using the default of 1."
            )
            batch_size = 1

        spec_config = None
        if model_args.spec_config is not None:
            # If spec_config is provided, use it to create a PardSpecDecodeConfig
            spec_config = PardSpecDecodeConfig(
                model_name_or_path=model_args.spec_config["model_name"],
                num_speculative_tokens=model_args.spec_config["num_speculated_tokens"],
            )

        self.batch_size = int(batch_size)
        self.model = LLMModel(
            model_args.model_name,
            model_args.tokenizer_name,
            dtype=model_args.dtype,
            kv_cache_type=kv_cache_type,
            opconfig=OperatorConfig(**model_args.llm_operators),
            spec_config=spec_config,
        )
        self.model_config = self.model.get_config()
        self.tokenizer = self.model.get_tokenizer()

        self._max_length = max_length
        self.max_gen_toks = max_gen_toks

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model_config, attr):
                return getattr(self.model_config, attr)

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:

        special_tokens_kwargs = {}
        if add_special_tokens is not None:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        add_special_tokens = {"add_special_tokens": False}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @suppress_logging_fn
    def _model_generate(
        self,
        input_encoded: List[Union[torch.Tensor, BatchEncoding]],
        max_new_tokens: int,
        stop: Optional[List[str]] = None,  # Add later
        **kwargs,
    ):
        # Set temperature to 0 if not specified
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0
        sampling_config = SamplingConfig(
            max_new_tokens=max_new_tokens, stop_strings=stop, **kwargs
        )
        outputs = self.model.generate(input_encoded, sampling_config)
        return outputs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        The function to generate a certain number of new tokens
        given a context.

        This function is an adapted version of the original function from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/openai_completions.py
        """
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        for chunk, request_args in list(
            sameuntil_chunks(re_ord.get_reordered(), self.batch_size)
        ):
            inps = []

            # make a deepcopy since we are changing arguments
            request_args = copy.deepcopy(request_args)

            self.max_gen_toks = request_args.pop("max_gen_toks", self.max_gen_toks)

            for context, _ in chunk:
                # add context (prompts) to the list
                inps.append(context)

            until = request_args.pop("until", ["<|endoftext|>"])
            request_args.pop("do_sample", None)
            request_args["temperature"] = request_args.get("temperature", 0)

            # run inference (generate max_gen_toks tokens)
            max_ctx_len = self.max_length - self.max_gen_toks
            inps = self.tok_batch_encode(
                inps,
                left_truncate_len=max_ctx_len,
            )
            out = self._model_generate(
                input_encoded=inps,
                max_new_tokens=self.max_gen_toks,
                stop=until,
                **request_args,
            )

            for resp, (context, args_) in zip(out.output_token_ids, chunk):
                resp = resp[inps["input_ids"].shape[1] :]
                text = self.tok_decode(resp)
                until_ = until
                # split the text at the first occurrence of any of the until tokens
                for term in until_:
                    if len(term) > 0:
                        text = text.split(term)[0]

                res.append(text)

                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until_}), text
                )

        return re_ord.get_original(res)

    def loglikelihood(self, requests: List[Instance]) -> List[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise NotImplementedError(
                    "Implementing empty context is not supported yet"
                )
            context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            """Defines the key for the sorted method"""
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Reorderer(requests, _collate)

        for chunk in tqdm(
            list(chunks(re_ord.get_reordered(), self.batch_size)),
        ):
            batch_inp = []
            batch_cache_key = []
            batch_continuation_enc = []
            padding_len_inp = None
            # len(chunk) is the batch_size
            for cache_key, context_enc, continuation_enc in chunk:
                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice # noqa: E501

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
                )
                inplen = len(inp)  # length of the input sequence

                batch_inp.append(inp)
                batch_cache_key.append(cache_key)
                batch_continuation_enc.append(continuation_enc)

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

            # pad the input to the longest sequence in the batch
            # (batch_size, max_len)
            batch_inp = pad_and_concat(padding_len_inp, batch_inp, padding_side="right")
            response = self._model_generate(
                batch_inp,
                max_new_tokens=1,
                stop=None,
                return_input_logprobs=True,
            )

            for multi_logits, continuation_enc, cache_key in zip(
                response.input_logprobs, batch_continuation_enc, batch_cache_key
            ):
                import numpy as np

                # toss out the context half of the sequence
                # (cont_len, vocab_size)
                continuation_multi_logits = multi_logits[-len(continuation_enc) :]

                # pick out the logits for the continuation tokens
                # (cont_len,)
                continuation_logits = continuation_multi_logits[
                    np.arange(len(continuation_enc)), continuation_enc
                ]
                # check if the tokens generated greedly are the same
                # as the expected continuation
                greedy_tokens = continuation_multi_logits.argmax(axis=1)
                max_equal = greedy_tokens.tolist() == continuation_enc

                # Answer: (log prob, is-exact-match)
                answer = (float(continuation_logits.sum()), bool(max_equal))

                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[tuple[float, bool]]:
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for PACE models"
        )
