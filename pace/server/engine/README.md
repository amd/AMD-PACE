# Inference Server Engine Documentation

## Overview

The Inference Server Engine is a FastAPI-based service that provides text generation capabilities using large language models. It consists of two main components:

1. **Backend Engine (`server.py`)** - Core model execution and sequence management
2. **Frontend API (`frontend.py`)** - REST API endpoints for client interaction

## Architecture Components

### Core Classes

#### ModelExecutor (`server.py`)
- **Purpose**: Manages model loading, prefill operations, and decode steps
- **Key Features**:
  - Model and tokenizer loading
  - Sequence queue management (prefill and decode queues)
  - KV cache management
  - Sampling configuration handling

#### Sequence (`server.py`)
- **Purpose**: Represents individual text generation requests
- **Key Features**:
  - Unique UUID identification
  - State tracking (PREFILL/DECODING)
  - Input tokenization and encoding
  - Attention mask management
  - Stopping criteria handling

## Available Endpoints
Note: install JSON command line processor(jq) for cleaner prints.
### 1. Model Management

#### `GET /get_models`
**Purpose**: List all supported models and their data types
```bash
curl -X GET "http://localhost:8000/get_models" | jq
```

**Response**:
```json
{
  "data": [
    {
      "id": "facebook/opt-6.7b",
      "object": "model",
      "created": 1640995200,
      "owned_by": "local",
      "dtypes": ["bfloat16", "float32"]
    }
  ]
}
```

#### `POST /config_server`
**Purpose**: Load and configure a model for inference
```bash
curl -X POST "http://localhost:8000/config_server" \
  -H "Content-Type: application/json" \
  -d '{
    "modelConfig": {
      "modelId": "facebook/opt-6.7b",
      "dataType": "bf16",
      "kvCacheType": "BMC",
      "norm_backend": "NATIVE",
      "qkv_projection_backend": "TPP",
      "attention_backend": "JIT",
      "out_projection_backend": "TPP",
      "mlp_backend": "TPP",
      "lm_head_backend": "NATIVE"
    }
  }'  | jq
```

### 2. Text Generation

#### `POST /step`
**Purpose**: Execute prefill batches or decode steps

##### Prefill Batch Request

The `prompt` field must be a **pre-tokenized list of integer token IDs**
(e.g. from `tokenizer.encode()`), not a raw string.

```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": [4014, 2581, 263, 931, 297, 263, 29592, 2982],
        "req_id": "12345678-1234-5678-9abc-123456789abc",
        "generation_config": {
          "max_new_tokens": 50,
          "temperature": 0.8,
          "top_p": 0.9,
          "top_k": 40,
          "repetition_penalty": 1.0,
          "frequency_penalty": 0.0,
          "do_sample": true,
          "seed": 123,
          "stop_strings": ["\n\n"]
        }
      }
    ]
  }' | jq
```

**Prefill Response**:
```json
{
  "status": "success",
  "step_type": "prefill_batch",
  "results": [
    {
      "req_id": "12345678-1234-5678-9abc-123456789abc",
      "result": {
        "12345678-1234-5678-9abc-123456789abc": {
          "token_ids": [29892],
          "status": "PREFILL_COMPLETED",
          "num_tokens_generated": 1
        }
      }
    }
  ]
}
```

##### Decode Request
```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{"is_decode": true}'  | jq
```

**Decode Response**:
```json
{
  "status": "success",
  "step_type": "decode",
  "result": {
    "12345678-1234-5678-9abc-123456789abc": {
      "token_ids": [727],
      "status": "DECODING_IN_PROGRESS",
      "num_tokens_generated": 1
    }
  }
}
```

When a sequence finishes, `status` becomes `"COMPLETED"` and includes a `stop_reason` (`"stop"` or `"length"`).

#### `POST /step_auto_mixed`
**Purpose**: Execute active decodes and new or chunked prefills in one mixed
forward pass. This is the internal endpoint used by the router's
`continuous_batch` scheduler with `SLAB_POOL` or `PAGED` KV cache.

The request contains a `prefill_batch` plus a `chunk_info_map` whose values are
`[chunk_start, chunk_size]`. Existing decode sequences are taken from the
engine's decode queue.

### 3. Inspection, Testing and Debugging

#### `GET /get_sequences`
**Purpose**: Get detailed information about all sequences
```bash
curl -X GET "http://localhost:8000/get_sequences"  | jq
```

#### `GET /get_sequences/summary`
**Purpose**: Get summary statistics of sequences
```bash
curl -X GET "http://localhost:8000/get_sequences/summary"  | jq
```

#### `GET /get_sequences/{sequence_id}`
**Purpose**: Get detailed information about a specific sequence
```bash
curl -X GET "http://localhost:8000/get_sequences/12345678-1234-5678-9abc-123456789abc"  | jq
```

#### `POST /remove_sequence`
**Purpose**: Remove sequences from the system
```bash
curl -X POST "http://localhost:8000/remove_sequence" \
  -H "Content-Type: application/json" \
  -d '{"sequence_ids": [0, 1]}'  | jq
```

#### `GET /tokenizer_status`
**Purpose**: Check tokenizer loading status
```bash
curl -X GET "http://localhost:8000/tokenizer_status"  | jq
```

#### `GET /cache_stats`
**Purpose**: Return KV-cache pool utilization and, when enabled, prefix-cache
statistics.

```bash
curl -X GET "http://localhost:8000/cache_stats" | jq
```

## Manual Testing with curl

### Complete Workflow Example

```bash
# 1. Check server health
curl -X GET "http://localhost:8000/get_models" | jq

# 2. Configure server
curl -X POST "http://localhost:8000/config_server" \
  -H "Content-Type: application/json" \
  -d '{
    "modelConfig": {
      "modelId": "facebook/opt-125m",
      "dataType": "float32",
      "kvCacheType": "BMC",
      "norm_backend": "NATIVE",
      "qkv_projection_backend": "TPP",
      "attention_backend": "JIT",
      "out_projection_backend": "TPP",
      "mlp_backend": "TPP",
      "lm_head_backend": "NATIVE"
    }
  }'  | jq

# 3. Test tokenizer
curl -X GET "http://localhost:8000/tokenizer_status"

# 4. Submit prefill request (prompt must be pre-tokenized token IDs)
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": [450, 5434, 310, 319, 29902, 338],
        "req_id": "a0000000-0000-0000-0000-000000000001",
        "generation_config": {
          "max_new_tokens": 30,
          "temperature": 0.7,
          "do_sample": true
        }
      }
    ]
  }'  | jq

# 5. Monitor sequences
curl -X GET "http://localhost:8000/get_sequences/summary"

# 6. Continue generation with decode steps
for i in {1..10}; do
  curl -X POST "http://localhost:8000/step" \
    -H "Content-Type: application/json" \
    -d '{"is_decode": true}'  | jq
  sleep 0.5
done

# 7. Check final results
curl -X GET "http://localhost:8000/get_sequences"  | jq
```

### Testing Different Scenarios

#### High Temperature Creative Writing
```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": [6113, 263, 26576, 1048, 278, 23474],
        "req_id": "a0000000-0000-0000-0000-000000000002",
        "generation_config": {
          "max_new_tokens": 50,
          "temperature": 1.2,
          "top_p": 0.9,
          "do_sample": true
        }
      }
    ]
  }'  | jq
```

#### Low Temperature Technical Writing
```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": [5765, 7420, 5765, 6509, 29257, 297, 2560, 4958],
        "req_id": "a0000000-0000-0000-0000-000000000003",
        "generation_config": {
          "max_new_tokens": 100,
          "temperature": 0.3,
          "top_k": 20,
          "do_sample": false
        }
      }
    ]
  }'  | jq
```

#### Multiple Concurrent Requests
```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": [5765, 29896, 29901, 9038, 2501, 263, 931],
        "req_id": "a0000000-0000-0000-0000-000000000004",
        "generation_config": {"max_new_tokens": 25}
      },
      {
        "is_prefill": true,
        "prompt": [5765, 29906, 29901, 512, 263, 17471, 2215, 3448],
        "req_id": "a0000000-0000-0000-0000-000000000005",
        "generation_config": {"max_new_tokens": 25}
      },
      {
        "is_prefill": true,
        "prompt": [5765, 29941, 29901, 450, 1629, 338, 29871, 29906, 29900, 29945, 29900],
        "req_id": "a0000000-0000-0000-0000-000000000006",
        "generation_config": {"max_new_tokens": 25}
      }
    ]
  }'  | jq
```

## Configuration Parameters

### Model Configuration
- **modelId**: HuggingFace model identifier
- **dataType**: Model precision (`bf16`/`bfloat16` or `fp32`/`float32`)
- **kvCacheType**: `DYNAMIC`, `BMC`, `SLAB_POOL`, or `PAGED`
- **norm_backend**: `NATIVE` or `JIT`
- **qkv_projection_backend**: Linear backend for fused QKV projection
- **attention_backend**: `JIT`, `NATIVE`, `SLAB`, or `PAGED`; finalized to match `kvCacheType`
- **out_projection_backend**: Linear backend for attention output projection
- **mlp_backend**: MLP or FusedMoE backend
- **lm_head_backend**: Linear backend for the language-model head
- **spec_config**: Optional PARD speculative-decoding configuration
- **kv_cache_memory_gb**: Optional KV-cache memory budget

### Generation Configuration
- **max_new_tokens**: Maximum tokens to generate
- **temperature**: Sampling temperature
- **top_p**: Nucleus sampling parameter
- **top_k**: Top-k sampling parameter
- **repetition_penalty**: Multiplicative penalty on tokens in prompt+output (1.0 = disabled; >1.0 discourages repetition)
- **frequency_penalty**: Additive penalty proportional to token count in output only (0.0 = disabled; >0.0 discourages repetition)
- **do_sample**: Enable/disable sampling
- **seed**: Random seed for reproducibility
- **stop_strings**: List of stop sequences

## Error Handling

### Common Error Responses

#### Server Not Configured
```json
{
  "status": "error",
  "error": "Model and tokenizer must be loaded before running prefill."
}
```

#### Invalid Request Format
```json
{
  "detail": "Request must specify either 'is_decode=true' or contain 'prefill_batch'"
}
```

#### Sequence Not Found
```json
{
  "detail": "Sequence not found in any queue."
}
```

## Performance Monitoring

### Sequence States
- **PREFILL**: Initial processing of input prompt
- **DECODING**: Token-by-token generation
- **COMPLETED**: Generation finished
- **ERROR**: Processing failed

### Queue Management
- **Prefill Queue**: Sequences waiting for initial processing
- **Decode Queue**: Sequences in active generation

### Monitoring Commands
```bash
# Quick status check
curl -s "http://localhost:8000/get_sequences/summary" | jq '.summary'

# Detailed queue inspection
curl -s "http://localhost:8000/get_sequences" | jq '.decode_queue.sequences'

# Individual sequence tracking
curl -s "http://localhost:8000/get_sequences/YOUR-UUID-HERE" | jq '.sequence'
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check model ID availability
   - Verify data type compatibility
   - Ensure sufficient memory

2. **Tokenizer Issues**
   - Check tokenizer status with `/tokenizer_status`

3. **Generation Stalls**
   - Monitor sequence queues
   - Check stopping criteria
   - Verify generation parameters

4. **Memory Issues**
   - Reduce batch size
   - Lower max_new_tokens
   - Use smaller model variants

### Debug Mode
Enable detailed logging by checking server console output for debug messages prefixed with `[DEBUG]`, `[Custom]`, or `[ERROR]`.
