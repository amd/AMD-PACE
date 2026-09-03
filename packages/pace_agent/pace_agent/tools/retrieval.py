# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import zlib

import numpy as np

from pace_agent.registry import PaceAgentTool, register_tool


class _RankTool(PaceAgentTool):
    def parity(
        self,
        args,
        kw,
        *,
        reference="naive",
        candidate="optimized",
        tol=1e-4,
        comparator=None,
    ):
        expected = self.dispatch(reference, *args, **kw)
        actual = self.dispatch(candidate, *args, **kw)
        if comparator:
            return comparator(expected, actual)
        _, sn = expected
        _, so = actual
        # Ties may produce different index order while preserving ranking quality.
        sn, so = np.sort(np.asarray(sn)), np.sort(np.asarray(so))
        return sn.shape == so.shape and np.allclose(sn, so, atol=tol, rtol=tol)


_VOCAB = [f"w{i}" for i in range(2000)]


def _rng():
    return np.random.default_rng(0)


def _texts(n, toklen, rng):
    return [" ".join(rng.choice(_VOCAB, size=toklen)) for _ in range(n)]


@register_tool("embed", "retrieval")
class PaceAgentEmbed(PaceAgentTool):
    def naive(self, texts, dim):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.split():
                out[i, zlib.crc32(tok.encode()) % dim] += 1.0
            n = float(np.linalg.norm(out[i])) or 1.0
            out[i] /= n
        return out

    def optimized(self, texts, dim):
        # same crc32 bag-of-words hashing (identical columns => parity)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        cache = {}
        for i, t in enumerate(texts):
            row = out[i]
            for tok in t.split():
                c = cache.get(tok)
                if c is None:
                    c = cache[tok] = zlib.crc32(tok.encode()) % dim
                row[c] += 1.0
        norms = np.linalg.norm(out, axis=1)
        norms[norms == 0] = 1.0
        out /= norms[:, None]
        return out

    def bench_inputs(self, spec):
        rng = _rng()
        n, dim, toklen = (
            spec.get("n", 512),
            spec.get("dim", 256),
            spec.get("toklen", 32),
        )
        return (_texts(n, toklen, rng), dim), {}


@register_tool("vector_search", "retrieval")
class PaceAgentVectorSearch(_RankTool):
    def naive(self, index, query, top_k):
        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scores = np.empty(len(index), dtype=np.float32)
        for i in range(len(index)):
            r = index[i]
            rn = float(np.linalg.norm(r)) or 1.0
            scores[i] = float(np.dot(r, q)) / (rn * qn)
        idx = np.argsort(-scores)[:top_k].astype(np.int64)
        return idx, scores[idx]

    def optimized(self, index, query, top_k):
        index = np.asarray(index, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        rn = np.linalg.norm(index, axis=1)  # cosine for ANY index (parity with naive)
        rn[rn == 0] = 1.0
        scores = (index @ q) / (rn * qn)
        k = max(0, min(int(top_k), scores.shape[0]))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        part = np.argpartition(-scores, k - 1)[:k]
        idx = part[np.argsort(-scores[part])].astype(np.int64)
        return idx, scores[idx].astype(np.float32)

    def bench_inputs(self, spec):
        rng = _rng()
        N, dim, top_k = (
            spec.get("N", 10000),
            spec.get("dim", 256),
            spec.get("top_k", 10),
        )
        index = rng.standard_normal((N, dim), dtype=np.float32)
        query = rng.standard_normal(dim, dtype=np.float32)
        return (index, query), {"top_k": top_k}


@register_tool("bm25", "retrieval")
class PaceAgentBM25(_RankTool):
    def naive(self, corpus, query, top_k, k1=1.5, b=0.75):
        n = len(corpus)
        avgdl = sum(len(d) for d in corpus) / n if n else 0.0
        df = {}
        for d in corpus:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        import math

        scores = np.zeros(n, dtype=np.float32)
        for i, d in enumerate(corpus):
            dl = len(d)
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                f = tf[t]
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / (avgdl or 1.0)))
            scores[i] = s
        idx = np.argsort(-scores)[:top_k].astype(np.int64)
        return idx, scores[idx]

    def optimized(self, corpus, query, top_k, k1=1.5, b=0.75):
        import math
        from collections import Counter

        n = len(corpus)
        dl = np.array([len(d) for d in corpus], dtype=np.float32)
        avgdl = float(dl.mean()) if n else 1.0
        counters = [Counter(d) for d in corpus]
        df = {}
        for c in counters:
            for t in c:
                df[t] = df.get(t, 0) + 1
        scores = np.zeros(n, dtype=np.float32)
        denom_len = k1 * (1 - b + b * dl / (avgdl or 1.0))
        for t, qc in Counter(query).items():
            if t not in df:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            tf = np.array([c.get(t, 0) for c in counters], dtype=np.float32)
            scores += qc * idf * (tf * (k1 + 1) / (tf + denom_len))
        idx = np.argsort(-scores)[:top_k].astype(np.int64)
        return idx, scores[idx]

    def bench_inputs(self, spec):
        rng = _rng()
        nd, dl, top_k = (
            spec.get("N_docs", 2000),
            spec.get("doclen", 40),
            spec.get("top_k", 10),
        )
        corpus = [list(rng.choice(_VOCAB, size=dl)) for _ in range(nd)]
        query = list(rng.choice(_VOCAB, size=spec.get("qlen", 5)))
        return (corpus, query), {"top_k": top_k}


@register_tool("rerank", "retrieval")
class PaceAgentRerank(_RankTool):
    def naive(self, query, cands, top_k):
        q = np.asarray(query, dtype=np.float32)
        scored = []
        for i in range(len(cands)):
            scored.append((float(np.dot(cands[i], q)), i))
        scored.sort(reverse=True)
        idx = np.array([i for _, i in scored[:top_k]], dtype=np.int64)
        return idx, np.array([s for s, _ in scored[:top_k]], dtype=np.float32)

    def optimized(self, query, cands, top_k):
        cands = np.asarray(cands, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        scores = cands @ q
        k = max(0, min(int(top_k), scores.shape[0]))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        part = np.argpartition(-scores, k - 1)[:k]
        idx = part[np.argsort(-scores[part])].astype(np.int64)
        return idx, scores[idx].astype(np.float32)

    def bench_inputs(self, spec):
        rng = _rng()
        n, dim, top_k = spec.get("n", 2000), spec.get("dim", 256), spec.get("top_k", 10)
        return (
            rng.standard_normal(dim, dtype=np.float32),
            rng.standard_normal((n, dim), dtype=np.float32),
        ), {"top_k": top_k}


@register_tool("quantize", "retrieval")
class PaceAgentQuantize(PaceAgentTool):
    def naive(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size == 0:
            return np.empty(0, dtype=np.int8)
        scale = (float(np.max(np.abs(x))) / 127.0) or 1.0
        out = np.empty(x.shape, dtype=np.int8)
        for i in range(x.size):
            v = int(round(x[i] / scale))
            out[i] = -127 if v < -127 else 127 if v > 127 else v
        return out

    def optimized(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        scale = (float(np.max(np.abs(x))) / 127.0) or 1.0
        return np.clip(np.rint(x / scale), -127, 127).astype(np.int8)

    def bench_inputs(self, spec):
        return (_rng().standard_normal(spec.get("size", 200000), dtype=np.float32),), {}
