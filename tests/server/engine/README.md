# Test Engine (`test_engine.py`)

## Purpose
The `test_engine.py` script is a comprehensive testing tool that helps users:

1. **Validate Server Functionality**: Test all major endpoints systematically
2. **Understand API Usage**: Demonstrate proper request formats and parameters
3. **Debug Issues**: Identify configuration or model loading problems
4. **Performance Testing**: Test various prompt types and generation configurations

## Key Test Functions

### `test_get_models()`
- Validates server accessibility
- Lists available models
- Checks model metadata

### `test_config_server()`
- Tests model loading functionality
- Validates configuration parameters
- Confirms server readiness

### `test_prefill_batch_X()` (5 variants)
- **Batch 1**: Basic story generation prompts
- **Batch 2**: Technical question prompts
- **Batch 3**: Creative writing prompts
- **Batch 4**: Conversational prompts
- **Batch 5**: Coding-related prompts

Each batch tests different:
- Generation parameters (temperature, top_p, top_k)
- Prompt complexity and length
- Sampling strategies (deterministic vs. stochastic)
- Stop sequences and constraints

### `test_decode()`
- Tests decode step execution
- Validates sequence state transitions
- Monitors generation progress

## How to Use test_engine.py

1. **Start the inference server**:
   ```bash
   cd amd_pace/pace/server/engine
   python frontend.py
   ```
   ***Expected Output***:
   ```
   @app.on_event("startup")
   INFO:     Started server process [747502]
   INFO:     Waiting for application startup.
   30-Sep-25 10:24:02.682 I frontend.py:35 pace: Inference engine started: FastAPI app is running...
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

2. **Run the test suite**:
   ```bash
   cd amd_pace/
   python -m unittest tests.server.engine.test_engine -v
   ```

   ***Expected Output***:
   ```
   test_01_get_models (tests.server.engine.test_engine.InferenceServerTests.test_01_get_models) ... ok
   test_02_config_server (tests.server.engine.test_engine.InferenceServerTests.test_02_config_server) ...ok
   test_03_prefill_basic (tests.server.engine.test_engine.InferenceServerTests.test_03_prefill_basic) ... ok
   test_04_prefill_technical (tests.server.engine.test_engine.InferenceServerTests.test_04_prefill_technical) ... ok
   test_05_prefill_creative (tests.server.engine.test_engine.InferenceServerTests.test_05_prefill_creative) ... ok
   test_06_prefill_conversational (tests.server.engine.test_engine.InferenceServerTests.test_06_prefill_conversational) ... ok
   test_07_prefill_coding (tests.server.engine.test_engine.InferenceServerTests.test_07_prefill_coding) ... ok
   test_08_decode_loop (tests.server.engine.test_engine.InferenceServerTests.test_08_decode_loop) ... ok

   ----------------------------------------------------------------------
   Ran 8 tests in 12.599s

   OK

   ```