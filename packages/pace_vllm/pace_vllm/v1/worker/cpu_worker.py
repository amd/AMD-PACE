# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PaceWorker: vLLM v1 CPU worker that builds PaceModelRunner.

Thin subclass of `CPUWorker`. `init_device` does three things:

1. `_register_pace_oots()` imports the modules that carry our
   `PluggableLayer.register_oot` / `CustomOp.register_oot` decorators.
   Must fire before model construction so the OOT substitution lands
   on every `QKVParallelLinear(...)` / `RMSNorm(...)` / etc. call.
2. `_install_fused_mlp_pass()` stashes the post-grad pattern matcher
   into `compilation_config.inductor_compile_config` in this
   subprocess -- `FusedMLPPass` captures `torch.ops.pace.*` OpOverload
   handles that are not pickleable across the worker spawn.
3. Delegate to `super().init_device()` for libtcmalloc / OMP / thread
   binding / distributed init / seed, then swap the throwaway
   `CPUModelRunner` super() built for `PaceModelRunner`.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker

from pace_vllm.v1.worker.cpu_model_runner import PaceModelRunner

logger = init_logger("pace_vllm.v1.worker")

_PACE_OOTS_REGISTERED = False


def _register_pace_oots() -> None:
    """Install pace-vllm OOTs (TPP Linear + ParallelLMHead + RMSNorm + FusedMoE).

    Idempotent. Must run before model construction so the
    `PluggableLayer.register_oot` / `CustomOp.register_oot` substitutions
    land on every QKVParallelLinear / RMSNorm / etc. in the model graph.
    """
    global _PACE_OOTS_REGISTERED
    if _PACE_OOTS_REGISTERED:
        return
    import pace_vllm.model_executor.layers.fused_moe  # noqa: F401
    import pace_vllm.model_executor.layers.layernorm  # noqa: F401
    import pace_vllm.model_executor.layers.linear  # noqa: F401
    import pace_vllm.model_executor.layers.vocab_parallel_embedding  # noqa: F401

    _PACE_OOTS_REGISTERED = True
    logger.info(
        "pace-vllm: registered OOTs (TPP Linear + ParallelLMHead + JIT RMSNorm "
        "+ FusedMoE)."
    )


def _install_fused_mlp_pass(vllm_config) -> None:
    """Install the pace fused-MLP post-grad pass for this subprocess.

    `FusedMLPPass` captures `torch.ops.pace.*` OpOverload handles that
    are not pickleable; building it in the main process would break
    the worker-spawn pickle. Doing it here is safe because the worker
    has already unpickled `vllm_config`. No-op in eager mode.
    """
    from vllm.config.compilation import CompilationMode

    if vllm_config.compilation_config.mode == CompilationMode.NONE:
        return

    key = "post_grad_custom_post_pass"
    inductor_config = vllm_config.compilation_config.inductor_compile_config
    existing = inductor_config.get(key)
    if existing is not None:
        logger.warning(
            "pace-vllm: post_grad_custom_post_pass already set (%s); "
            "pace FusedMLPPass registration skipped.",
            type(existing).__name__,
        )
        return

    from pace_vllm.compilation.fused_mlp_pass import FusedMLPPass

    inductor_config[key] = FusedMLPPass(vllm_config)
    logger.info("pace-vllm: installed FusedMLPPass into post_grad_custom_post_pass.")


class PaceWorker(CPUWorker):
    """CPU worker that instantiates `PaceModelRunner` instead of `CPUModelRunner`."""

    def init_device(self) -> None:
        logger.info("pace-vllm: PaceWorker.init_device starting.")

        _register_pace_oots()
        _install_fused_mlp_pass(self.vllm_config)

        super().init_device()

        # Swap super()'s throwaway CPUModelRunner for our subclass; weights
        # aren't loaded until load_model, so the discarded instance only
        # costs its in-memory setup.
        self.model_runner: PaceModelRunner = PaceModelRunner(
            self.vllm_config, torch.device("cpu")
        )

        logger.info(
            "pace-vllm: PaceWorker active (worker=%s, model_runner=%s).",
            type(self).__name__,
            type(self.model_runner).__name__,
        )

    def determine_available_memory(self) -> int:
        # The returned share sizes vLLM's own block pool and scheduler
        # admission; the slab share goes to
        # `cache_config.kv_cache_memory_bytes`, where PaceKVCache reads it.
        budget = int(super().determine_available_memory())
        if budget <= 0:
            # CPUWorker raises on `<= 0` in its auto path, but the explicit
            # path doesn't guard the same way (a `--kv-cache-memory-bytes=0`
            # would slip through). Catch here so PaceKVCache never has to
            # reason about a non-positive budget.
            raise RuntimeError(
                f"pace-vllm: vLLM returned a non-positive CPU KV cache "
                f"budget ({budget} bytes). Set --kv-cache-memory-bytes or "
                "VLLM_CPU_KVCACHE_SPACE to a positive value."
            )

        vllm_budget, slab_budget = self._split_kv_budget(budget)
        self.vllm_config.cache_config.kv_cache_memory_bytes = slab_budget
        return vllm_budget

    def _split_kv_budget(self, budget: int) -> tuple[int, int]:
        """Return ``(vllm_budget, slab_budget)``.

        For pure-attention models the two shares intentionally do NOT sum to
        ``budget``: vLLM physically allocates no KV pool (the runner ships
        zero-sized placeholders), so there is no double-count and both shares
        are the full ``budget`` -- the returned value only drives scheduler
        admission, and the slab owns all storage.

        For hybrid models the two shares DO sum to ``budget``, because vLLM
        *does* allocate a Mamba pool and we must avoid double-booking memory.
        Hybrid models share one vLLM block pool across Mamba and attention
        layers; the pool's ``num_blocks`` both sizes the Mamba state *and*
        caps attention admission. We slab the attention storage, so the slab
        must back exactly the attention the pool admits. Solving
        ``vllm_pool + slab == budget`` with the slab sized to back
        ``num_blocks`` attention blocks gives, after page unification
        (attention page == uniform page),

            vllm_budget = budget * G / (G + L_attn)
            slab_budget = budget * L_attn / (G + L_attn)

        where ``G`` is the max layers in any group and ``L_attn`` the number
        of attention layers.
        """
        from copy import deepcopy

        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
        from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs

        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        if not kv_cache_spec:
            return budget, budget

        # get_kv_cache_groups unifies (pads) page sizes in place; copy so the
        # engine's own later call sees a pristine spec.
        groups = get_kv_cache_groups(self.vllm_config, deepcopy(kv_cache_spec))

        def _is_attention(group) -> bool:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()), None)
            return isinstance(spec, AttentionSpec)

        if all(_is_attention(g) for g in groups):
            return budget, budget

        group_size = max(len(g.layer_names) for g in groups)
        num_attn_layers = sum(len(g.layer_names) for g in groups if _is_attention(g))
        vllm_budget = budget * group_size // (group_size + num_attn_layers)
        slab_budget = budget - vllm_budget
        logger.info(
            "pace-vllm: hybrid KV budget split: %.2f GiB total -> %.2f GiB "
            "vLLM Mamba pool + %.2f GiB slab (group_size=%d, attn_layers=%d).",
            budget / (1024**3),
            vllm_budget / (1024**3),
            slab_budget / (1024**3),
            group_size,
            num_attn_layers,
        )
        return vllm_budget, slab_budget
