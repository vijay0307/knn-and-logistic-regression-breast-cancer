"""Chunking + BM25 retrieval over Markdown/text files.

Pure Python, no embedding API or vector DB needed, so it runs offline, in CI and
is fully deterministic (which makes retrieval evaluation reproducible). For a
large corpus, swap `BM25Index` for an embedding store behind the same
`search()` interface.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for with is are was were be been by as at it its this that "
    "these those from into than then so such can do does did how what which who why when "
    "where i you we they my your our their me us them not no if about there here also".split()
)


def tokenize(text: str) -> list[str]:
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
    # light stemming: plurals ("features" -> "feature")
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t for t in toks]


@dataclass(frozen=True)
class Chunk:
    id: str          # stable id, e.g. "dataset.md#3"
    source: str      # file name
    heading: str     # nearest Markdown heading
    text: str


def chunk_markdown(text: str, source: str, max_chars: int = 1200) -> list[Chunk]:
    """Split on headings, then pack paragraphs up to max_chars. Each chunk keeps its heading."""
    sections, heading, buf = [], source, []
    for line in text.splitlines():
        if line.startswith("#"):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))

    chunks: list[Chunk] = []
    for heading, body in sections:
        piece = ""
        for para in re.split(r"\n\s*\n", body):
            if piece and len(piece) + len(para) > max_chars:
                chunks.append(Chunk(f"{source}#{len(chunks)}", source, heading, piece.strip()))
                piece = ""
            piece += para + "\n\n"
        if piece.strip():
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, heading, piece.strip()))
    return chunks


def load_corpus(paths: list[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paths:
        files = sorted(p.glob("*.md")) + sorted(p.glob("*.txt")) if p.is_dir() else [p]
        for f in files:
            chunks.extend(chunk_markdown(f.read_text(encoding="utf-8"), f.name))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("empty corpus")
        self.chunks, self.k1, self.b = chunks, k1, b
        # heading is indexed twice: it is a strong topical signal
        self.docs = [Counter(tokenize(f"{c.heading} {c.heading} {c.text}")) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scores[:k]]
