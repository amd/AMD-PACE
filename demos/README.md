# PACE Server Demos

Interactive demos for the AMD PACE inference server. Start the server first, then run a demo.

| Demo | Description | Command |
|------|-------------|---------|
| `pace_server_speculative_demo.py` | Live curses grid showing PARD speculative decoding across multiple prompts | `python pace_server_speculative_demo.py` |
| `server_chat_demo.py` | Streaming multi-turn chat with Llama-3 prompt template | `python server_chat_demo.py` |

Both demos auto-detect whether the server is running and print the exact launch command if not.

```bash
# Chat demo
pace-server --server_model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
python server_chat_demo.py

# Speculative decoding demo (requires PARD-enabled server)
pace-server --server_model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
  --spec_config '{"model_name":"amd/PARD-Qwen2.5-0.5B","num_speculative_tokens":12}' \
  --serve_type continuous_prefill_first
python pace_server_speculative_demo.py
```
