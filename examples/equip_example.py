from transformers import AutoTokenizer

from pace.llm import LLMModel, SamplingConfig, KVCacheType, EvictionConfig, RoPEMode

# NousResearch mirror avoids Meta HF gated access; meta-llama/Llama-3.2-1B also works.
model_path = "NousResearch/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LLMModel(
    model_path,
    # EQUIP is contiguous-only; default offline cache is SLAB_POOL.
    kv_cache_type=KVCacheType.DYNAMIC,
    eviction_config=EvictionConfig(
        sink_size=4,
        recent_size=252,
        rope_mode=RoPEMode.REINDEX,  # Llama supports cache-owned RoPE
    ),
)

prompts = ["Once upon a time"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

outputs = model.generate(
    inputs,
    SamplingConfig(max_new_tokens=128, do_sample=False),
)
print(tokenizer.decode(outputs.output_token_ids[0], skip_special_tokens=True))
