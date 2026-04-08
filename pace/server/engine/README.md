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
      "attnType": "dynamic"
    }
  }'  | jq
```

### 2. Text Generation

#### `POST /step`
**Purpose**: Execute prefill batches or decode steps

##### Prefill Batch Request
```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": "Once upon a time in a distant land",
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

##### Decode Request
```bash
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{"is_decode": true}'  | jq
```

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
      "attnType": "dynamic"
    },
    "serveConfig": {
      "serveType": "iterative",
      "serveParams": {}
    }
  }'  | jq

# 3. Test tokenizer
curl -X GET "http://localhost:8000/tokenizer_status"

# 4. Submit prefill request
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_batch": [
      {
        "is_prefill": true,
        "prompt": "The future of AI is",
        "req_id": "test-001",
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
        "prompt": "Write a poem about the ocean",
        "req_id": "creative-001",
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
        "prompt": "Explain machine learning in simple terms",
        "req_id": "technical-001",
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
        "prompt": "Story 1: Once upon a time",
        "req_id": "story-1",
        "generation_config": {"max_new_tokens": 25}
      },
      {
        "is_prefill": true,
        "prompt": "Story 2: In a galaxy far away",
        "req_id": "story-2",
        "generation_config": {"max_new_tokens": 25}
      },
      {
        "is_prefill": true,
        "prompt": "Story 3: The year is 2050",
        "req_id": "story-3",
        "generation_config": {"max_new_tokens": 25}
      }
    ]
  }'  | jq
```

## Configuration Parameters

### Model Configuration
- **modelId**: HuggingFace model identifier
- **dataType**: Model precision (bf16, float32, float16)
- **attnType**: Attention mechanism (dynamic, bmc)

### Generation Configuration
- **max_new_tokens**: Maximum tokens to generate (1-2048)
- **temperature**: Sampling temperature (0.1-2.0)
- **top_p**: Nucleus sampling parameter (0.1-1.0)
- **top_k**: Top-k sampling parameter (1-100)
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
