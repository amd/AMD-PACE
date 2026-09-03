# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import functools
import importlib
from dataclasses import dataclass

import numpy as np

TOOLS = {}

_HOOK = None


def set_tool_hook(hook):
    """Install the per-run tool hook. Bound onto the current CaptureContext when one is
    active (so concurrent runs never cross-record), else a process-global fallback."""
    from pace_agent.capture import current_capture

    ctx = current_capture()
    if ctx is not None:
        ctx.tool_hook = hook
    else:
        global _HOOK
        _HOOK = hook


class SkipTool(Exception):
    pass


@dataclass
class ImplementationInfo:
    name: str
    kind: str  # reference | candidate | provider
    source: str = "builtin"  # builtin | plugin/package name
    version: str | None = None
    is_fallback: bool = False  # True when this tier just delegates to naive
    deterministic: bool = True
    # False => the op has side effects and must NOT be re-executed on replay (its recorded
    # output is reused instead). Every built-in PACE Agent op is pure CPU compute (True).
    side_effect_free: bool = True


class PaceAgentTool:
    name = ""
    bucket = ""
    io_pattern = False
    side_effect_free = True  # pure, re-runnable CPU compute (see ImplementationInfo)

    def naive(self, *args, **kw):
        raise NotImplementedError

    def optimized(self, *args, **kw):
        return self.naive(*args, **kw)

    def optimized_available(self):
        """Whether the ``optimized`` fast-path is usable here. Default True: pure-Python tiers
        need no optional dep, so they're always available. Tiers backed by an optional library
        (orjson, selectolax, re2) override this; when False the tier is honestly reported as a
        naive fallback (``is_fallback``) rather than a phantom candidate."""
        return True

    def bench_inputs(self, spec):
        return (), dict(spec)

    def run(self, tier, *args, **kw):
        from pace_agent.capture import current_capture

        ctx = current_capture()
        hook = ctx.tool_hook if ctx is not None and ctx.tool_hook is not None else _HOOK
        if hook is not None:
            return hook(self, tier, args, kw)
        return self.dispatch(tier, *args, **kw)

    def dispatch(self, tier, *args, **kw):
        if tier == "naive":
            return self.naive(*args, **kw)
        if tier == "optimized":
            # Single fallback point: if this op's optional accelerator is unavailable, the
            # optimized tier is an honest naive fallback (also surfaced via
            # optimized_available() -> is_fallback). Op bodies may therefore assume their
            # accelerator library is importable and skip a per-op availability check.
            if not self.optimized_available():
                return self.naive(*args, **kw)
            return self.optimized(*args, **kw)
        # any other tier names a provider; dispatch to its impl for this op, or skip
        # if the provider isn't loaded / has no impl (no silent Python fallback, no crash)
        from pace_agent.providers import get_provider_impl

        fn = get_provider_impl(tier, self.name)
        if fn is None:
            raise SkipTool(
                f"tier/provider '{tier}' unavailable for op '{self.name}' "
                "(not loaded or no impl)"
            )
        return fn(*args, **kw)

    def implementations(self):
        """Tier metadata for reporting: distinguishes a real built-in candidate from an
        `optimized` that merely falls back to `naive`, and enumerates provider tiers."""
        overridden = type(self).optimized is not PaceAgentTool.optimized
        # a real candidate must be both overridden AND (for optional-lib tiers) available
        available = overridden and self.optimized_available()
        sef = type(self).side_effect_free
        impls = {
            "naive": ImplementationInfo("naive", "reference", side_effect_free=sef),
            "optimized": ImplementationInfo(
                "optimized",
                "candidate" if available else "reference",
                is_fallback=not available,
                side_effect_free=sef,
            ),
        }
        from pace_agent.providers import provider_meta, providers_for

        for prov in providers_for(self.name):
            m = provider_meta(prov)
            impls[prov] = ImplementationInfo(
                prov,
                "provider",
                source=m["source"],
                version=m["version"],
                side_effect_free=sef,
            )
        return impls

    def parity(
        self,
        args,
        kw,
        *,
        reference="naive",
        candidate="optimized",
        tol=1e-6,
        comparator=None,
    ):
        # dispatch() (not run()) so parity checks bypass the record/replay hook
        exp = self.dispatch(reference, *args, **kw)
        act = self.dispatch(candidate, *args, **kw)
        return comparator(exp, act) if comparator else _equal(exp, act, tol)


def register_tool(name, bucket, io_pattern=False):
    def deco(cls):
        cls.name, cls.bucket, cls.io_pattern = name, bucket, io_pattern
        TOOLS[name] = cls()
        return cls

    return deco


def op(name, bucket="custom", io_pattern=False, side_effect_free=True):
    """Decorator: register a plain function as a replayable PaceAgentTool without subclassing.

    Calls to the returned wrapper route through the (context-scoped) registry hook, so the
    recorder captures inputs/outputs and the player can dispatch tiers / replay it. The
    wrapped function is the ``naive`` tier; provider candidates can be added later.

    The op MUST be pure and re-runnable: replay re-executes it to time it. If your function
    has side effects, pass ``side_effect_free=False`` so replay reuses the recorded output
    instead of re-running it.
    """

    def deco(fn):
        @register_tool(name, bucket, io_pattern)
        class _Op(PaceAgentTool):
            def naive(self, *args, **kw):
                return fn(*args, **kw)

        _Op.side_effect_free = side_effect_free
        tool = TOOLS[name]

        @functools.wraps(fn)
        def wrapper(*args, **kw):
            return tool.run("naive", *args, **kw)

        wrapper.pace_agent_tool = tool
        return wrapper

    return deco


def get_tool(name):
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown tool '{name}'; known: {sorted(TOOLS)}") from None


def list_tools():
    return sorted(TOOLS.values(), key=lambda o: (o.bucket, o.name))


def load_plugins(mods):
    for m in mods or ():
        importlib.import_module(m)


def _equal(a, b, tol):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if np.issubdtype(a.dtype, np.number):
            return np.allclose(a, b, atol=tol, rtol=tol)
        return bool((a == b).all())
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol + tol * max(abs(a), abs(b))
    return a == b
