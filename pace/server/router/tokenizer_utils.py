# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Router-side tokenization and detokenization utilities.

All text <-> token-ID conversion lives here.  The engine operates
exclusively on token IDs; the router owns encode, decode, and
stop-string detection.

Kept as a single utility module since PACE does not expose separate
``/tokenize`` or ``/detokenize`` HTTP endpoints.
"""

from typing import List, Optional, Union

_tokenizer = None


class NoChatTemplateError(ValueError):
    """Raised when the tokenizer has no chat template configured.

    Distinct from generic rendering errors so callers can return a precise
    ``no_chat_template`` response instead of conflating a missing template with
    malformed ``messages`` input.
    """


# Keys that control tokenization and output shape.  Allowing a client to
# override these via ``chat_template_kwargs`` could make ``apply_chat_template``
# return a string or dict (or raise ``TypeError`` for a duplicate keyword),
# instead of the flat list of token IDs the scheduler/detokenizer expect.
_RESERVED_TEMPLATE_KWARGS = frozenset(
    {
        "tokenize",
        "add_generation_prompt",
        "return_dict",
        "return_tensors",
        "return_assistant_tokens_mask",
    }
)


def init_tokenizer(tokenizer) -> None:
    """Set the module-level tokenizer. Called once at startup from frontend.py."""
    global _tokenizer
    _tokenizer = tokenizer


def get_tokenizer():
    """Return the shared tokenizer instance."""
    return _tokenizer


def normalize_prompts(
    prompt: Union[str, List[str], List[int], List[List[int]]],
    tokenizer=None,
) -> List[List[int]]:
    """Normalize the four OpenAI prompt formats into a list of token-ID lists.

    - ``str``              -> ``[tokenizer.encode(s)]``
    - ``list[str]``        -> ``[tokenizer.encode(s) for s in list]``
    - ``list[int]``        -> ``[list]``  (single prompt already tokenized)
    - ``list[list[int]]``  -> pass through
    """
    tok = tokenizer or _tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer not initialized. Call init_tokenizer() first.")

    if isinstance(prompt, str):
        return [tok.encode(prompt)]

    if isinstance(prompt, list) and len(prompt) == 0:
        return [tok.encode("")]

    first = prompt[0]

    if isinstance(first, str):
        return [tok.encode(s) for s in prompt]

    if isinstance(first, int):
        return [list(prompt)]

    if isinstance(first, list):
        return [list(ids) for ids in prompt]

    return [tok.encode(str(prompt))]


def decode_token_ids(
    token_ids: List[int],
    skip_special_tokens: bool = True,
    tokenizer=None,
) -> str:
    """Decode a list of token IDs into text."""
    tok = tokenizer or _tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer not initialized. Call init_tokenizer() first.")
    return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)


def decode_logprobs_step(step: dict, tokenizer=None):
    """Detokenize a *numeric* engine logprobs step into decoded strings.

    The engine emits token IDs and ``[token_id, logprob]`` top-k pairs; this
    performs the router-native detokenization (single source of truth, same
    tokenizer/settings as the response text) and returns parallel lists::

        (tokens, token_logprobs, top_logprobs)

    ``text_offset`` is intentionally left to the caller, since it depends on
    accumulated/emitted text (non-streaming counts from 0; streaming anchors
    to the emitted-text length).
    """
    token_ids = step.get("token_ids", [])
    token_logprobs = list(step.get("token_logprobs", []))
    raw_top = step.get("top_logprobs", [])

    tokens = [decode_token_ids([tid], tokenizer=tokenizer) for tid in token_ids]
    top_logprobs = []
    for top in raw_top:
        top_map = {}
        for cand_id, logprob in top:
            top_map[decode_token_ids([int(cand_id)], tokenizer=tokenizer)] = logprob
        top_logprobs.append(top_map)
    return tokens, token_logprobs, top_logprobs


def render_chat_prompt(
    messages: List[dict],
    add_generation_prompt: bool = True,
    chat_template_kwargs: Optional[dict] = None,
    tokenizer=None,
) -> List[int]:
    """Render chat ``messages`` into prompt token IDs via the model's template.

    ``messages`` is a list of ``{"role", "content"}`` dicts. Returns the
    tokenized prompt, with a generation prompt appended when
    ``add_generation_prompt`` is set so the model continues as the assistant.

    Raises ``NoChatTemplateError`` if the tokenizer has no chat template;
    callers should surface this as a 400 (the client should use
    ``/v1/completions`` instead).  Reserved template controls (``tokenize``,
    ``add_generation_prompt``, etc.) in ``chat_template_kwargs`` are dropped so
    the router always receives token IDs.
    """
    tok = tokenizer or _tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer not initialized. Call init_tokenizer() first.")
    if getattr(tok, "chat_template", None) is None:
        raise NoChatTemplateError("model has no chat template; use /v1/completions")
    extra = {
        k: v
        for k, v in (chat_template_kwargs or {}).items()
        if k not in _RESERVED_TEMPLATE_KWARGS
    }
    rendered = tok.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        **extra,
    )
    input_ids = rendered["input_ids"]
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return [int(t) for t in input_ids]


def check_stop_strings(
    generated_text: str,
    stop_strings: Optional[List[str]],
) -> Optional[str]:
    """Check if any stop string appears in generated text.

    Returns the matched stop string, or None if no match.
    """
    if not stop_strings:
        return None
    for s in stop_strings:
        if s in generated_text:
            return s
    return None


def truncate_at_stop_string(
    text: str,
    stop_strings: Optional[List[str]],
) -> str:
    """Truncate text at the earliest occurrence of any stop string."""
    if not stop_strings:
        return text
    earliest_pos = len(text)
    for s in stop_strings:
        pos = text.find(s)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos]
