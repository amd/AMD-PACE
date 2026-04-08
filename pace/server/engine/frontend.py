# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portion of this file consist of AI-Generated code.
# ******************************************************************************

import uuid
import time
import uvicorn
import asyncio

from fastapi import FastAPI
from fastapi import HTTPException
from typing import List, Dict, Any
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Configure Pydantic to allow arbitrary types
class PydanticConfig:
    arbitrary_types_allowed = True


from pace.utils.logging import PACE_INFO, PACE_DEBUG
from pace.server.engine.utils import ServerConfig, PrefillRequest
from pace.server.engine.server import model_executor

from pace.server.model_list import SUPPORTED_MODEL_LIST

app = FastAPI()


@app.on_event("startup")
def startup_event():
    PACE_INFO("Inference engine started: FastAPI app is running...")
    return


@app.post("/config_server")
def config_server(config: ServerConfig):
    modelConfig = config.modelConfig
    PACE_INFO("Received model configuration: Starting model setup...")
    PACE_DEBUG(f"Model config: {modelConfig}")

    model_executor.load_model(modelConfig)
    PACE_INFO(f"Model Config: {model_executor.model_config} loaded successfully.")
    return {"status": "Server configured successfully"}


@app.post("/step")
def step(req: Dict[str, Any]):

    try:
        # Handle two types of requests:
        # 1. Batch of prefill requests (list of prefill operations)
        # 2. Single decode request
        PACE_DEBUG(f"[ENGINE/FRONTEND] Entering the step endpoint with request: {req}")
        if req.get("is_decode", False):
            # Handle single decode request
            result = model_executor.decode()
            return {"status": "success", "step_type": "decode", "result": result}

        elif "prefill_batch" in req:
            # Handle batch of prefill requests
            prefill_requests = req["prefill_batch"]
            results = []

            for prefill_req in prefill_requests:
                if not prefill_req.get("is_prefill", False):
                    results.append(
                        {
                            "req_id": prefill_req.get("req_id", "unknown"),
                            "error": "Request must have is_prefill=true",
                        }
                    )
                    continue

                # Validate required fields
                if "prompt" not in prefill_req or "req_id" not in prefill_req:
                    results.append(
                        {
                            "req_id": prefill_req.get("req_id", "unknown"),
                            "error": "Prefill request must include 'prompt' and 'req_id'",
                        }
                    )
                    continue

                prompt = prefill_req["prompt"]
                req_id = prefill_req["req_id"]
                generation_config = prefill_req.get("generation_config", {})
                mlperf_mode = prefill_req.get("mlperf_mode", False)

                gen_config_dict = (
                    {k: v for k, v in generation_config.items() if v is not None}
                    if generation_config
                    else {}
                )

                # Create request tuple for model_executor.prefill()
                # and return gracefully if it does not validate
                try:
                    if not req_id or not isinstance(req_id, str) or len(req_id) != 36:
                        error_msg = (
                            "req_id must be a non-empty string and "
                            "req_id must be a valid UUID string (36 characters)"
                        )
                        results.append(
                            {
                                "req_id": req_id if req_id else "missing",
                                "error": error_msg,
                            }
                        )
                        continue

                    req_uuid = uuid.UUID(req_id)
                except ValueError:
                    results.append(
                        {
                            "req_id": req_id,
                            "error": "req_id must be a valid UUID format",
                        }
                    )
                    continue

                try:
                    # Execute prefill for this request (same as mlperf_scripts: prefill creates Sequence inside)
                    result = model_executor.prefill(
                        PrefillRequest(
                            request_id=req_uuid,
                            prompt=prompt,
                            gen_config=gen_config_dict,
                            mlperf_mode=mlperf_mode,
                        )
                    )
                    results.append({"req_id": req_id, "result": result})
                except Exception as e:
                    results.append({"req_id": req_id, "error": str(e)})
            PACE_DEBUG(f"[ENGINE/FRONTEND] Prefill batch completed. Results: {results}")
            PACE_INFO(f"[ENGINE/FRONTEND] Status: Processed {len(results)} requests")

            return {
                "status": "success",
                "step_type": "prefill_batch",
                "results": results,
            }

        else:
            raise HTTPException(
                status_code=400,
                detail="Request must specify either 'is_decode=true' or contain 'prefill_batch'",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove_sequence")
def remove_sequences(req: Dict[str, List[str]]):
    try:
        sequence_ids = req.get("sequence_ids")
        if not sequence_ids:
            raise HTTPException(
                status_code=400, detail="Missing 'sequence_ids' in request."
            )

        # Convert string IDs to UUID objects
        uuid_list = []
        for seq_id in sequence_ids:
            try:
                uuid_list.append(uuid.UUID(seq_id))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid UUID format for sequence_id: {seq_id}",
                )

        result = model_executor.remove_sequences(uuid_list)

        return {
            "status": "success",
            "removed_sequence_ids": sequence_ids,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_models")
def list_models():
    models = []
    for model_info in SUPPORTED_MODEL_LIST:
        models.append(
            {
                "id": model_info["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "dtypes": model_info.get("dtypes", []),
            }
        )
    return {"data": models, "object": "list"}


@app.get("/get_sequences")
def get_sequences():
    """Get information about all sequences currently in the system."""
    try:
        prefill_sequences = []
        decode_sequences = []

        # Get prefill queue sequences
        for seq_id, sequence in model_executor.prefill_queue.items():
            prefill_sequences.append(
                {
                    "id": str(seq_id),
                    "state": sequence.state.name,
                    "prompt": sequence.input_request,
                    "input_length": (
                        sequence.input_encoded.input_ids.shape[-1]
                        if hasattr(sequence, "input_encoded")
                        else 0
                    ),
                    "max_new_tokens": (
                        sequence.sampling_config.max_new_tokens
                        if hasattr(sequence, "sampling_config")
                        else 0
                    ),
                    "output_tokens_generated": (
                        len(sequence.output_str)
                        if hasattr(sequence, "output_str")
                        else 0
                    ),
                }
            )

        # Get decode queue sequences
        for seq_id, sequence in model_executor.decode_queue.items():
            decode_sequences.append(
                {
                    "id": str(seq_id),
                    "state": sequence.state.name,
                    "prompt": sequence.input_request,
                    "input_length": (
                        sequence.input_encoded.input_ids.shape[-1]
                        if hasattr(sequence, "input_encoded")
                        else 0
                    ),
                    "max_new_tokens": (
                        sequence.sampling_config.max_new_tokens
                        if hasattr(sequence, "sampling_config")
                        else 0
                    ),
                    "output_tokens_generated": (
                        len(sequence.output_str)
                        if hasattr(sequence, "output_str")
                        else 0
                    ),
                    "current_output": (
                        sequence.output_str[-1]
                        if hasattr(sequence, "output_str") and sequence.output_str
                        else ""
                    ),
                }
            )

        return {
            "status": "success",
            "total_sequences": len(prefill_sequences) + len(decode_sequences),
            "prefill_queue": {
                "count": len(prefill_sequences),
                "sequences": prefill_sequences,
            },
            "decode_queue": {
                "count": len(decode_sequences),
                "sequences": decode_sequences,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_sequences/summary")
def get_sequences_summary():
    """Get a summary of sequences currently in the system."""
    try:
        prefill_count = len(model_executor.prefill_queue)
        decode_count = len(model_executor.decode_queue)
        total_count = prefill_count + decode_count

        return {
            "status": "success",
            "summary": {
                "total_sequences": total_count,
                "prefill_queue_count": prefill_count,
                "decode_queue_count": decode_count,
                "server_status": "active" if total_count > 0 else "idle",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_sequences/{sequence_id}")
def get_sequence_by_id(sequence_id: str):
    """Get detailed information about a specific sequence."""
    try:
        # Try to convert string to UUID
        try:
            seq_uuid = uuid.UUID(sequence_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid sequence ID format. Must be a valid UUID.",
            )

        # Check prefill queue
        if seq_uuid in model_executor.prefill_queue:
            sequence = model_executor.prefill_queue[seq_uuid]
            return {
                "status": "success",
                "sequence": {
                    "id": str(seq_uuid),
                    "state": sequence.state.name,
                    "queue": "prefill",
                    "prompt": sequence.input_request,
                    "input_length": (
                        sequence.input_encoded.input_ids.shape[-1]
                        if hasattr(sequence, "input_encoded")
                        else 0
                    ),
                    "max_new_tokens": (
                        sequence.sampling_config.max_new_tokens
                        if hasattr(sequence, "sampling_config")
                        else 0
                    ),
                    "output_tokens_generated": (
                        len(sequence.output_str)
                        if hasattr(sequence, "output_str")
                        else 0
                    ),
                    "sampling_config": (
                        {
                            "temperature": (
                                sequence.sampling_config.temperature
                                if hasattr(sequence, "sampling_config")
                                else None
                            ),
                            "top_p": (
                                sequence.sampling_config.top_p
                                if hasattr(sequence, "sampling_config")
                                else None
                            ),
                            "top_k": (
                                sequence.sampling_config.top_k
                                if hasattr(sequence, "sampling_config")
                                else None
                            ),
                        }
                        if hasattr(sequence, "sampling_config")
                        else {}
                    ),
                },
            }

        # Check decode queue
        elif seq_uuid in model_executor.decode_queue:
            sequence = model_executor.decode_queue[seq_uuid]
            return {
                "status": "success",
                "sequence": {
                    "id": str(seq_uuid),
                    "state": sequence.state.name,
                    "queue": "decode",
                    "prompt": sequence.input_request,
                    "input_length": (
                        sequence.input_encoded.input_ids.shape[-1]
                        if hasattr(sequence, "input_encoded")
                        else 0
                    ),
                    "max_new_tokens": (
                        sequence.sampling_config.max_new_tokens
                        if hasattr(sequence, "sampling_config")
                        else 0
                    ),
                    "output_tokens_generated": (
                        len(sequence.output_str)
                        if hasattr(sequence, "output_str")
                        else 0
                    ),
                    "current_output": (
                        sequence.output_str[-1]
                        if hasattr(sequence, "output_str") and sequence.output_str
                        else ""
                    ),
                    "full_output": (
                        sequence.output_str if hasattr(sequence, "output_str") else []
                    ),
                    "sampling_config": (
                        {
                            "temperature": (
                                sequence.sampling_config.temperature
                                if hasattr(sequence, "sampling_config")
                                else None
                            ),
                            "top_p": (
                                sequence.sampling_config.top_p
                                if hasattr(sequence, "sampling_config")
                                else None
                            ),
                            "top_k": (
                                sequence.sampling_config.top_k
                                if hasattr(sequence, "sampling_config")
                                else None
                            ),
                        }
                        if hasattr(sequence, "sampling_config")
                        else {}
                    ),
                },
            }

        else:
            raise HTTPException(
                status_code=404, detail="Sequence not found in any queue."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to test tokenizer status for debugging issues
@app.get("/tokenizer_status")
def get_tokenizer_status():
    """Get basic tokenizer status information."""
    try:
        if model_executor._tokenizer is None:
            return {
                "status": "not_loaded",
                "message": "Tokenizer not loaded. Please configure the server first.",
            }

        return {
            "status": "loaded",
            "tokenizer_class": str(type(model_executor._tokenizer)),
            "model_name": getattr(model_executor._tokenizer, "name_or_path", "Unknown"),
            "vocab_size": model_executor._tokenizer.vocab_size,
            "special_tokens": {
                "pad_token": model_executor._tokenizer.pad_token,
                "eos_token": model_executor._tokenizer.eos_token,
                "bos_token": getattr(model_executor._tokenizer, "bos_token", None),
                "unk_token": getattr(model_executor._tokenizer, "unk_token", None),
            },
            "settings": {
                "padding_side": model_executor._tokenizer.padding_side,
                "truncation_side": getattr(
                    model_executor._tokenizer, "truncation_side", "Unknown"
                ),
            },
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


import argparse


def main():
    parser = argparse.ArgumentParser(description="Inference Server for LLMs")
    parser.add_argument("--host", type=str, help="Host address to bind the server")
    parser.add_argument("--port", type=int, help="Port to bind the server")
    parser.add_argument(
        "--fastapi_log_level",
        type=str,
        help="Log level for FastAPI",
    )
    args = parser.parse_args()
    if args.fastapi_log_level == "None":
        args.fastapi_log_level = None
    else:
        args.fastapi_log_level = uvicorn.config.LOGGING_CONFIG
    uvicorn.run(app, host=args.host, port=args.port, log_config=args.fastapi_log_level)


if __name__ == "__main__":
    main()
