"""Offline retrieval over the extension guides with BM25.

BM25 needs no model download and almost no memory, which matters on a hub
that already holds a 17 GB model. Its other job is a "do the guides cover
this?" signal: `coverage` is the share of the question's content words found
in the best passage.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from rank_bm25 import BM25Plus

_STOP = set("""
a an and are as at be been but by can could did do does doing for from had has have how i if in into
is it its me my of on or our should so than that the their them then there these they this to was
we were what when where which who why will with would you your about any after also am get got just
like more most much now only other out over same some such too very what's whats help please tell
know want need use using used should shall may might must
""".split())


def _stem(tok: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


# British -> American spelling, so "recognise"/"colour" match "recognize"/"color".
_UK_US = [(re.compile(r"(\w{3,})is(e|ed|es|ing|ation)\b"), r"\1iz\2"),
          (re.compile(r"\b(colo|behavio|favo|labo|neighbo)ur"), r"\1r")]

# Farmer wording -> guide wording. Applied to queries only, as extra terms.
# Tuned on the dev split; keep it short and generic (no answer words).
QUERY_SYNONYMS = {
    "worm": ["larva", "caterpillar"], "worms": ["larvae", "caterpillars"],
    "caterpillar": ["larva"], "caterpillars": ["larvae"],
    "lay": ["laid", "oviposition", "egg"], "lays": ["laid", "oviposition"],
    "bug": ["insect", "pest"], "bugs": ["insects", "pests"],
    "spray": ["pesticide", "insecticide"], "chemical": ["pesticide"], "chemicals": ["pesticides"],
    "kill": ["control"], "rid": ["control", "eliminate"], "stop": ["control", "prevent"],
    "check": ["scout", "inspect", "observe"], "checking": ["scouting"], "visit": ["scout"],
    "marks": ["spots", "stripes", "marking"], "recognize": ["identify", "identification"],
    "throw": ["dispose", "remove"], "pull": ["uproot", "remove"], "sick": ["diseased", "infected"],
    "cutting": ["planting", "material"], "cuttings": ["planting", "material"],
}


def _normalise(text: str) -> str:
    text = (text or "").lower()
    for pat, rep in _UK_US:
        text = pat.sub(rep, text)
    return text


def tokenize(text: str, expand: bool = False) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", _normalise(text))
    if expand:
        toks = toks + [s for t in toks for s in QUERY_SYNONYMS.get(t, [])]
    return [_stem(t) for t in toks if t not in _STOP and len(t) > 1]


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    text: str
    url: str = ""
    licence: str = ""
    score: float = 0.0

    def cite(self) -> dict:
        d = asdict(self)
        d["snippet"] = self.text[:280] + ("…" if len(self.text) > 280 else "")
        d.pop("text")
        return d


@dataclass
class SearchResult:
    passages: list[Passage]
    coverage: float        # share of query content terms present in the top passage
    top_score: float


class GuideIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._tokens = [tokenize(c["text"]) for c in chunks]
        self._sets = [set(t) for t in self._tokens]
        # BM25Plus keeps term weights positive even in a small guide collection.
        self._bm25 = BM25Plus(self._tokens) if chunks else None

    @classmethod
    def load(cls, path: Path) -> "GuideIndex":
        if not Path(path).exists():
            return cls([])
        data = json.loads(Path(path).read_text("utf-8"))
        return cls(data["chunks"])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5, expand: bool = True) -> SearchResult:
        q = tokenize(query)                                  # the farmer's own words: used for coverage
        if not q or self._bm25 is None:
            return SearchResult([], 0.0, 0.0)
        scores = self._bm25.get_scores(tokenize(query, expand=expand))   # plus guide synonyms for ranking
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        passages = []
        for i in order:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            passages.append(Passage(c["chunk_id"], c["doc_id"], c["title"], c.get("page"), c["text"],
                                    c.get("url", ""), c.get("licence", ""), float(scores[i])))
        if not passages:
            return SearchResult([], 0.0, 0.0)
        qset = set(q)
        best = order[0]
        coverage = len(qset & self._sets[best]) / len(qset)
        return SearchResult(passages, coverage, float(scores[best]))


def chunk_pages(pages: list[tuple[int | None, str]], words_per_chunk: int = 220, overlap: int = 40) -> list[tuple[int | None, str]]:
    """Split page texts into overlapping word windows, keeping the page number."""
    out: list[tuple[int | None, str]] = []
    for page, text in pages:
        words = re.sub(r"\s+", " ", text or "").strip().split(" ")
        if not words or words == [""]:
            continue
        step = max(words_per_chunk - overlap, 1)
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + words_per_chunk])
            if len(piece.split()) >= 25 or start == 0:
                out.append((page, piece))
            if start + words_per_chunk >= len(words):
                break
    return out
