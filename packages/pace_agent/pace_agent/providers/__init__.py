# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Provider registry — the pluggable third tier above naive/optimized.

A provider is any library that registers an accelerated implementation for an op via
``register_provider`` (``PaceAgentTool.run(<provider>, ...)`` dispatches to it). PACE Agent core
carries NO provider-specific knowledge: providers live in their own packages and can
self-register through the ``pace_agent.providers`` entry point, discovered lazily the first time
a provider tier is looked up. A missing provider tier raises ``SkipTool`` — never a silent
Python fallback.
"""

import importlib.metadata as _md

_PROVIDERS = {}
_PROVIDER_META = {}  # provider -> {"source": pkg, "version": str | None}
_discovered = False


def register_provider(provider, op_name, fn, *, source=None, version=None):
    """Register ``fn`` as provider ``provider``'s implementation of op ``op_name``.

    ``source``/``version`` describe the providing package (for honest tier reporting).
    When a provider self-registers via its entry point they are filled in from the
    distribution automatically; a direct caller may pass them explicitly.
    """
    _PROVIDERS[(provider, op_name)] = fn
    meta = _PROVIDER_META.setdefault(provider, {"source": provider, "version": None})
    if source:
        meta["source"] = source
    if version:
        meta["version"] = version


def _discover():
    global _discovered
    if _discovered:
        return
    _discovered = True
    try:
        eps = _md.entry_points(group="pace_agent.providers")
    except Exception:
        return
    for ep in eps:
        try:
            ep.load()()
        except Exception:
            continue
        # fill provenance from the distribution that declares the entry point
        dist = getattr(ep, "dist", None)
        meta = _PROVIDER_META.get(ep.name)
        if dist is not None and meta is not None:
            if meta.get("source") in (None, ep.name):
                meta["source"] = getattr(dist, "name", ep.name)
            if not meta.get("version"):
                meta["version"] = getattr(dist, "version", None)


def get_provider_impl(provider, op_name):
    """Return the registered impl for ``(provider, op_name)``, or ``None``."""
    _discover()
    return _PROVIDERS.get((provider, op_name))


def providers_for(op_name):
    """Sorted provider names that have registered an impl for ``op_name``."""
    _discover()
    return sorted({p for (p, o) in _PROVIDERS if o == op_name})


def provider_meta(provider):
    """Best-effort provenance for ``provider``: ``{"source": pkg, "version": str|None}``."""
    _discover()
    return dict(_PROVIDER_META.get(provider, {"source": provider, "version": None}))
