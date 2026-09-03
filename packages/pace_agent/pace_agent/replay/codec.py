# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Safe trajectory serialization: msgpack + a typed codec registry.

Replaces pickle in the load path. Native msgpack handles str/int/float/bool/None/
list/dict/bytes; numpy arrays and scalars are handled here; any other type must have
a registered codec, otherwise serialization fails loudly with ``UnsupportedType``
(never a silent, unsafe fallback). Dataclasses like ToolCall/Trajectory register their
own codecs (see replay/trajectory.py).
"""

import hashlib

import msgpack
import numpy as np

from pace_agent.errors import UnsupportedType

_TAG = "__pace_agent__"
_ENCODERS = {}  # type -> (tag, to_dict)
_DECODERS = {}  # tag -> from_dict


def register_codec(type_, tag, to_dict, from_dict):
    """Register ``type_`` <-> dict codec under ``tag`` (must be globally unique)."""
    _ENCODERS[type_] = (tag, to_dict)
    _DECODERS[tag] = from_dict


def _default(o):
    if isinstance(o, np.ndarray):
        return {
            _TAG: "ndarray",
            "dtype": str(o.dtype),
            "shape": list(o.shape),
            "data": o.tobytes(),
        }
    if isinstance(o, np.generic):
        return o.item()
    enc = _ENCODERS.get(type(o))
    if enc is None:
        for t, e in _ENCODERS.items():  # allow subclasses of registered types
            if isinstance(o, t):
                enc = e
                break
    if enc is None:
        raise UnsupportedType(
            f"cannot serialize {type(o).__module__}.{type(o).__name__}; register a codec "
            f"via pace_agent.replay.codec.register_codec(type, tag, to_dict, from_dict)"
        )
    tag, to_dict = enc
    d = dict(to_dict(o))
    d[_TAG] = tag
    return d


def _object_hook(d):
    tag = d.get(_TAG)
    if tag == "ndarray":
        return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
    if tag is not None and tag in _DECODERS:
        return _DECODERS[tag]({k: v for k, v in d.items() if k != _TAG})
    return d


def dumps(obj):
    return msgpack.packb(obj, default=_default, use_bin_type=True, strict_types=False)


def loads(b):
    return msgpack.unpackb(b, object_hook=_object_hook, raw=False, strict_map_key=False)


def descriptor(obj):
    """Compact per-value descriptor for the strict replay contract (report #5)."""
    d = {
        "type": type(obj).__name__,
        "digest": hashlib.sha256(dumps(obj)).hexdigest()[:16],
    }
    shape = getattr(obj, "shape", None)
    if shape is not None:
        d["shape"] = list(shape)
        d["dtype"] = str(getattr(obj, "dtype", ""))
    elif hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        d["len"] = len(obj)
    return d
