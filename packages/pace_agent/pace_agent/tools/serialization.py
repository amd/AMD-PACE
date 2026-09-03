# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import ast
import json
import re
from html.parser import HTMLParser

from pace_agent.registry import PaceAgentTool, SkipTool, register_tool

# Optional accelerator libraries. Each optimized tier that needs one reports its
# availability via optimized_available(); when absent it is honestly a naive fallback.
try:
    import orjson as _orjson  # fast C JSON (dumps returns bytes)

    _HAS_ORJSON = True
except ImportError:  # pragma: no cover
    _HAS_ORJSON = False

try:
    from selectolax.parser import HTMLParser as _SelectolaxParser  # lexbor-backed HTML

    _HAS_SELECTOLAX = True
except ImportError:  # pragma: no cover
    _HAS_SELECTOLAX = False

try:
    import re2 as _re2  # google-re2: linear-time regex (worst-case safe)

    _HAS_RE2 = True
except ImportError:  # pragma: no cover
    _HAS_RE2 = False


def _obj(n):
    return {
        "items": [
            {"id": i, "name": f"item{i}", "vals": list(range(5))} for i in range(n)
        ]
    }


@register_tool("json_serialize", "serialization")
class PaceAgentJsonSerialize(PaceAgentTool):
    def naive(self, obj):
        return json.dumps(obj)

    def optimized(self, obj):
        # orjson emits compact bytes; decode to str to keep the naive return contract.
        # dispatch() only routes here when optimized_available() is True (orjson present).
        return _orjson.dumps(obj).decode()

    def optimized_available(self):
        return _HAS_ORJSON

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
        # compare SEMANTICS, not byte-for-byte formatting (orjson drops whitespace)
        return json.loads(self.dispatch(reference, *args, **kw)) == json.loads(
            self.dispatch(candidate, *args, **kw)
        )

    def bench_inputs(self, spec):
        return (_obj(spec.get("n", 2000)),), {}


@register_tool("json_deserialize", "serialization")
class PaceAgentJsonDeserialize(PaceAgentTool):
    def naive(self, s):
        return json.loads(s)

    def optimized(self, s):
        # dispatch() only routes here when optimized_available() is True (orjson present)
        return _orjson.loads(s)

    def optimized_available(self):
        return _HAS_ORJSON

    def bench_inputs(self, spec):
        return (json.dumps(_obj(spec.get("n", 2000))),), {}


@register_tool("tokenize", "serialization")
class PaceAgentTokenize(PaceAgentTool):
    def naive(self, text):
        return re.findall(r"\w+", text.lower())

    def bench_inputs(self, spec):
        return (("the quick brown fox " * spec.get("n", 4000)),), {}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []

    def handle_data(self, d):
        d = d.strip()
        if d:
            self.out.append(d)


@register_tool("html_parse", "serialization")
class PaceAgentHtmlParse(PaceAgentTool):
    def naive(self, html):
        p = _Text()
        p.feed(html)
        return " ".join(p.out)

    def optimized(self, html):
        # lexbor-backed extraction; parity compares whitespace-normalized text (below).
        # dispatch() only routes here when optimized_available() is True (selectolax present).
        return _SelectolaxParser(html).text(separator=" ")

    def optimized_available(self):
        return _HAS_SELECTOLAX

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
        def norm(s):
            return " ".join(str(s).split())  # collapse whitespace differences

        return norm(self.dispatch(reference, *args, **kw)) == norm(
            self.dispatch(candidate, *args, **kw)
        )

    def bench_inputs(self, spec):
        row = "<div class='r'><span>cell {i}</span><a href='#'>link {i}</a></div>"
        body = "".join(row.format(i=i) for i in range(spec.get("n", 1500)))
        return (f"<html><body>{body}</body></html>",), {}


@register_tool("ast_parse", "serialization")
class PaceAgentAstParse(PaceAgentTool):
    def naive(self, source):
        return sum(1 for _ in ast.walk(ast.parse(source)))

    def optimized(self, source):
        # same node count, explicit stack avoids ast.walk's deque + generator overhead
        tree = ast.parse(source)
        count, stack = 0, [tree]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(ast.iter_child_nodes(node))
        return count

    def bench_inputs(self, spec):
        fn = "def f{i}(x):\n    y = x + {i}\n    return y * 2\n"
        return ("".join(fn.format(i=i) for i in range(spec.get("n", 400))),), {}


@register_tool("regex_extract", "serialization")
class PaceAgentRegexExtract(PaceAgentTool):
    def naive(self, text, pattern):
        return re.findall(pattern, text)

    def optimized(self, text, pattern):
        # google-re2: linear-time matching (worst-case safe vs backtracking re).
        # dispatch() only routes here when optimized_available() is True (re2 present).
        return _re2.findall(pattern, text)

    def optimized_available(self):
        return _HAS_RE2

    def bench_inputs(self, spec):
        text = " ".join(f"key{i}=val{i}" for i in range(spec.get("n", 5000)))
        return (text, r"key\d+=(\w+)"), {}


def _page_range(spec, total):
    if not spec:
        return range(total)
    lo, _, hi = str(spec).partition("-")
    try:
        lo = int(lo)
        hi = int(hi) if hi else lo
    except ValueError:
        raise SkipTool(f"invalid pages spec {spec!r} (want '1' or '1-5')") from None
    lo = max(1, lo)  # 1-based: page 0/negative must not wrap to the last page
    hi = min(hi, total)
    return range(lo - 1, hi) if hi >= lo else range(0)


@register_tool("pdf_text_extract", "serialization")
class PaceAgentPdfTextExtract(PaceAgentTool):
    def naive(self, path, pages=None):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SkipTool("pypdf not installed")
        import os

        if not os.path.exists(path):
            raise SkipTool(f"missing fixture {path}")
        reader = PdfReader(path)
        idx = _page_range(pages, len(reader.pages))
        return "\n".join(reader.pages[i].extract_text() or "" for i in idx)

    def bench_inputs(self, spec):
        from pace_agent.bench.fixtures import fixture_path

        return (fixture_path(spec.get("file", "sample.pdf")),), {
            "pages": spec.get("pages")
        }
