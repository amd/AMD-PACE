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


from pace.utils.logging import PACE_INFO, PACE_DEBUG, PACE_WARNING
from pace.server.engine.utils import (
    ServerConfig,
    StepRequest,
    MixedStepRequest,
    PrefillRequest,
)
from pace.server.engine.server import model_executor, Sequence

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
def step(req: StepRequest):
    """Iterative scheduler step: either a decode or a batch of prefills."""
    try:
        if req.is_decode:
            result = model_executor.decode()
            return {"status": "success", "step_type": "decode", "result": result}

        if not req.prefill_batch:
            raise HTTPException(
                status_code=400,
                detail="Request must specify either 'is_decode=true' or contain 'prefill_batch'",
            )

        results = []
        for pr in req.prefill_batch:
            rid = pr.req_id
            generation_config = pr.generation_config or {}
            gen_config_dict = (
                {k: v for k, v in generation_config.items() if v is not None}
                if generation_config
                else {}
            )
            try:
                req_uuid = uuid.UUID(rid)
                result = model_executor.prefill(
                    PrefillRequest(
                        request_id=req_uuid,
                        prompt=pr.prompt,
                        gen_config=gen_config_dict,
                    )
                )
                results.append({"req_id": rid, "result": result})
            except Exception as e:
                results.append({"req_id": rid, "error": str(e)})

        return {
            "status": "success",
            "step_type": "prefill_batch",
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step_auto_mixed")
def step_auto_mixed(req: MixedStepRequest):
    """Continuous batching: prefill new requests + decode existing ones in one forward."""
    try:
        prefill_requests = req.prefill_batch
        PACE_DEBUG(
            f"[step_auto_mixed] prefill_count={len(prefill_requests)}, "
            f"decode_queue_size={len(model_executor.decode_queue)}"
        )
        chunk_info_map = (
            {k: tuple(v) for k, v in req.chunk_info_map.items()}
            if req.chunk_info_map
            else None
        )

        decode_sequences = list(model_executor.decode_queue.values())

        prefill_sequences: List = []
        prefill_errors: Dict[str, Any] = {}
        for pr in prefill_requests:
            rid = pr.req_id
            prompt_token_ids = pr.prompt
            generation_config = pr.generation_config or {}
            gen_config_dict = (
                {k: v for k, v in generation_config.items() if v is not None}
                if generation_config
                else {}
            )
            try:
                req_uuid = uuid.UUID(rid)

                existing_seq = model_executor.prefill_queue.get(req_uuid)
                if existing_seq is not None:
                    prefill_sequences.append(existing_seq)
                    continue

                spec_headroom = 0
                if model_executor._spec_decoder is not None:
                    spec_headroom = (
                        model_executor._spec_decoder.num_speculative_tokens + 1
                    )
                seq = Sequence(
                    req_id=req_uuid,
                    input_token_ids=prompt_token_ids,
                    model_config=model_executor.model_config,
                    cache_backend=model_executor._cache_backend,
                    _prepare_sampling_config=model_executor._model.generator._prepare_sampling_config,
                    genConfig=gen_config_dict if gen_config_dict else None,
                    spec_headroom=spec_headroom,
                    tokenizer=model_executor._tokenizer,
                )
                model_executor.prefill_queue[seq.id] = seq
                prefill_sequences.append(seq)
            except Exception as e:
                PACE_WARNING(f"Failed to create sequence for {rid}: {e}")
                prefill_errors[str(rid)] = {
                    "status": "ERROR",
                    "error": str(e),
                }

        if not prefill_sequences and not decode_sequences:
            return {
                "status": "success",
                "step_type": "auto_mixed",
                "prefill_result": prefill_errors,
                "decode_result": {},
                "stats": {"num_prefill": 0, "num_decode": 0, "total_in_batch": 0},
            }

        prefill_result, decode_result = model_executor.prefill_decode_mixed_batch(
            prefill_sequences, decode_sequences, chunk_info_map
        )
        prefill_result.update(prefill_errors)

        return {
            "status": "success",
            "step_type": "auto_mixed",
            "prefill_result": prefill_result,
            "decode_result": decode_result,
            "stats": {
                "num_prefill": len(prefill_result),
                "num_decode": len(decode_result),
                "total_in_batch": len(prefill_result) + len(decode_result),
            },
        }

    except HTTPException:
        raise
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


@app.get("/cache_stats")
def get_cache_stats():
    """Return KV cache pool and prefix cache statistics.

    Internal engine endpoint (no OpenAI equivalent). The router consumes it for
    ``/v1/server_metrics`` and to feed the standard Prometheus ``/metrics``
    gauges (``pace_prefix_cache_*``, ``pace_kv_cache_usage_perc``).
    """
    stats = {}
    backend = getattr(model_executor, "_cache_backend", None)
    if backend is not None:
        # Paged exposes the pool as ``_pool``; slab as ``_manager``.
        pool = getattr(backend, "_pool", None) or getattr(backend, "_manager", None)
        if pool is not None and hasattr(pool, "get_stats"):
            stats["pool"] = pool.get_stats()
        if hasattr(backend, "get_prefix_cache_stats"):
            stats["prefix_cache"] = backend.get_prefix_cache_stats()
    return {"status": "success", "cache_stats": stats}


@app.get("/get_sequences")
def get_sequences():
    """Get information about all sequences currently in the system."""
    try:
        prefill_sequences = []
        decode_sequences = []

        for seq_id, sequence in model_executor.prefill_queue.items():
            prefill_sequences.append(
                {
                    "id": str(seq_id),
                    "state": sequence.state.name,
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
                    "total_tokens": sequence._token_len,
                }
            )

        for seq_id, sequence in model_executor.decode_queue.items():
            decode_sequences.append(
                {
                    "id": str(seq_id),
                    "state": sequence.state.name,
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
                    "total_tokens": sequence._token_len,
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

        if seq_uuid in model_executor.prefill_queue:
            sequence = model_executor.prefill_queue[seq_uuid]
            return {
                "status": "success",
                "sequence": {
                    "id": str(seq_uuid),
                    "state": sequence.state.name,
                    "queue": "prefill",
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
                    "total_tokens": sequence._token_len,
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

        elif seq_uuid in model_executor.decode_queue:
            sequence = model_executor.decode_queue[seq_uuid]
            return {
                "status": "success",
                "sequence": {
                    "id": str(seq_uuid),
                    "state": sequence.state.name,
                    "queue": "decode",
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
                    "total_tokens": sequence._token_len,
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
