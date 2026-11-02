"""On-device redaction. Runs before anything is sent to the cloud or queued.

Rule-based on purpose: no model download, no network, predictable, and every
miss can be traced to a rule. Coverage is measured by bench/redaction_eval.py,
and misses are reported, not hidden.

Kinds: NAME, PHONE, EMAIL, ID, GPS, LOCATION, ADDRESS.
Optional gazetteers (one entry per line) widen coverage for a deployment:
    data/redaction/names.txt, data/redaction/places.txt
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class Span:
    kind: str
    start: int
    end: int
    text: str


# Words that look like names/places after a trigger but are agronomy, time or filler.
_SAFE_WORDS = {
    w.lower() for w in """
    Maize Corn Cassava Tomato Tomatoes Cashew Rice Paddy Padi Beans Bean Chilli Pepper Okra
    Fall Armyworm Armyworms Septoria Verticillium Mosaic Streak Blight Spot Rust Mite Mites
    Beetle Grasshopper Leaf Leaves Plant Plants Crop Crops Farm Farms Field Fields Garden
    January February March April May June July August September October November December
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday Spring Summer Autumn Winter
    The This That These Those It Its My Our Your Their His Her A An Some Any Each Every
    Extension Officer Officers Ministry Department Agriculture Agricultural Fieldwise
    Morning Evening Night Today Yesterday Tomorrow Rainy Dry Season Wet Harvest
    Worried Sure Not Very Really Happy Sad Tired Afraid Scared Here There Now
    God Allah English Malay French Swahili Twi Hausa
    """.split()
}

# Ordinary words that follow "I'm", "this is", "call me" etc. or open a message, so a lowercase
# word there is NOT taken for a name. (Added 31 Oct: farmers type names in lowercase, "my name is kofi",
# which the capital-letter rules missed: 0/20 names on a held-out set written by the developer.)
_COMMON_WORDS = _SAFE_WORDS | {
    w for w in """
    a an the my our your his her their its this that these those it he she we they you i me us them him
    not no yes very so too really just still also only even already now then here there back again soon
    later today tomorrow tonight yesterday morning evening afternoon night currently always never often
    on in at from to for with about of by off out up down over into near regarding concerning
    is am are was were be been being do does did have has had will would can could should may might must
    and or but if because when while since as than
    sure worried happy sad sorry glad afraid scared tired sick ill okay ok fine well good bad poor new old
    young big small confused desperate concerned interested unable able ready serious urgent alone busy
    need want think know see help ask write call grow plant spray use buy sell lose lost try hope find
    farmer farmers woman women man men boy girl lady student teacher officer worker member members mother
    father wife husband son daughter brother sister friend neighbour neighbor chief elder pastor group
    cooperative association community village town area everyone everybody someone somebody anyone people
    all many some few most others other one two three another what who how why where which whose
    please thanks thank sir madam hi hello hey dear greetings welcome
    yellow brown green black white red purple grey gray dark pale dry wet rotten dead wilted curled curly
    spotted sick infected damaged diseased everywhere spreading serious terrible worse worst
    """.split()
}

_NAME_WORD = r"[A-Z][a-zA-Z'’\-]+"
_NAME_SEQ = rf"{_NAME_WORD}(?:\s+(?:bin|binti|bte|bt|a/l|a/p|van|de|da|dos|al|el)?\s*{_NAME_WORD}){{0,3}}"

_NAME_TRIGGERS = (
    r"my name is|my name's|name is|i am|i'm|this is|it's|call me|name\s*:|nama saya|saya|"
    r"mr\.?|mrs\.?|ms\.?|miss|madam|madame|dr\.?|encik|en\.|puan|pn\.|cik|haji|hajah|tuan|"
    r"auntie|aunty|uncle|mama|baba|nana|"
    r"my (?:neighbou?r|wife|husband|son|daughter|brother|sister|friend|father|mother|uncle|aunt|"
    r"cousin|boss|worker|helper|landlord|tenant)|"
    r"farmer|contact|ask for|speak to|talk to|from farmer"
)

_PLACE_TRIGGERS = (
    r"in|at|from|near|around|outside|beside|"
    r"village of|town of|district of|region of|"
    r"kampung|kampong|kg\.?|village|district|region|province|county|parish|municipality|state of|"
    r"live in|living in|farm in|farm at|based in|located in"
)

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("GPS", re.compile(r"-?\b\d{1,2}\.\d{3,}\s*[,;]\s*-?\d{1,3}\.\d{3,}\b")),
    ("GPS", re.compile(r"\b\d{1,3}\s*°\s*\d{1,2}\s*['′]\s*(?:\d{1,2}(?:\.\d+)?\s*[\"″]\s*)?[NSEW]\b")),
    # Malaysian MyKad, Ghana Card, generic "ID: ..." forms
    ("ID", re.compile(r"\b\d{6}-\d{2}-\d{4}\b")),
    ("ID", re.compile(r"\bGHA-\d{9}-\d\b", re.IGNORECASE)),
    ("ID", re.compile(r"\b(?:ID|IC|NIN|NRIC|passport|national id)(?:\s*(?:no\.?|number|#))?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-]{5,})", re.IGNORECASE)),
    ("ADDRESS", re.compile(r"\bP\.?\s?O\.?\s+Box\s+\d+[A-Z]?\b", re.IGNORECASE)),
    ("NAME", re.compile(r"(?<![\w@])@[A-Za-z0-9_.]{3,30}\b")),   # social media handles
    ("PHONE", re.compile(r"\b(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)[\s,\-]+){6,}"
                         r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE)),
    ("ADDRESS", re.compile(r"\b(?:No\.?\s*\d+[A-Za-z]?,?\s*)?(?:Jalan|Jln\.?|Lorong|Lrg\.?|Taman|Tmn\.?)\s+[A-Z0-9][\w'\-]*(?:\s+[A-Z0-9][\w'\-]*){0,4}")),
    ("ADDRESS", re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Close|Crescent)\b\.?")),
]

# Phones: 9-15 digits with common separators, optionally +country code.
# Boundaries reject decimals ("0.0241234567") but allow sentence punctuation ("call 0241234567.").
_PHONE_CANDIDATE = re.compile(r"(?<!\w)(?<!\d\.)(?:\+|00)?\d[\d\s\-.()]{7,20}\d(?!\w)(?!\.\d)")
_UNIT_AFTER = re.compile(r"^\s*(?:%|ml|l\b|g\b|kg|cc|ppm|days?|weeks?|ha\b|acres?|m\b|cm|mm|plants?|seeds?)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _gazetteer(kind: str) -> tuple[str, ...]:
    path = Path(settings.data_dir) / "redaction" / f"{kind}.txt"
    if not path.exists():
        return ()
    items = [ln.strip() for ln in path.read_text("utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return tuple(sorted(set(items), key=len, reverse=True))


def _is_safe(words: str) -> bool:
    toks = re.findall(r"[A-Za-z'’\-]+", words)
    return bool(toks) and all(t.lower() in _SAFE_WORDS for t in toks)


def _plain_word(tok: str) -> bool:
    """True if a lowercase token is an ordinary word rather than a name."""
    t = tok.strip(".,!?;:").lower()
    return (len(t) < 2 or t in _COMMON_WORDS or t.endswith("ing") or t.endswith("ly")
            or not re.fullmatch(r"[a-z'’\-]+", t))


def _lowercase_name(seq: str) -> str:
    """'kofi mensah from' -> 'kofi mensah': leading tokens that aren't ordinary words (at most 3)."""
    out = []
    for tok in seq.split()[:3]:
        if _plain_word(tok):
            break
        out.append(tok.strip(".,!?;:"))
    return " ".join(out)


def _trim_safe_tail(seq: str) -> str:
    """'Kofi Maize' -> 'Kofi': stop a captured name at the first agronomy/time word."""
    out = []
    for tok in seq.split():
        if tok.strip(".,").lower() in _SAFE_WORDS:
            break
        out.append(tok)
    return " ".join(out)


def _find_spans(text: str) -> list[Span]:
    spans: list[Span] = []

    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            if m.groups():
                s, e = m.span(1)
            else:
                s, e = m.span()
            spans.append(Span(kind, s, e, text[s:e]))

    for m in _PHONE_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if not 9 <= len(digits) <= 15:
            continue
        if _UNIT_AFTER.match(text[m.end():m.end() + 8]):
            continue
        if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", m.group().strip()):
            continue  # a date
        spans.append(Span("PHONE", m.start(), m.end(), m.group()))

    name_re = re.compile(rf"\b(?i:{_NAME_TRIGGERS})\s+(?:(?:Dr|Mr|Mrs|Ms|Prof|Dato|Datin|Tuan|Puan|Encik)\.?\s+)?({_NAME_SEQ})")
    for m in name_re.finditer(text):
        raw = m.group(1)
        # the trigger is case-insensitive, the name itself must be capitalised
        if not raw[:1].isupper():
            continue
        name = _trim_safe_tail(raw)
        if name and not _is_safe(name):
            s = m.start(1)
            spans.append(Span("NAME", s, s + len(name), name))

    # Lowercase names after a self-introduction: "my name is kofi", "this is kwame", "i'm kojo".
    low_re = re.compile(r"\b(?i:my name is|my name's|my names|name is|name\s*:|call me|i am|i'm|i’m|im|this is)"
                        r"\s+([a-z][a-z'’\-]+(?:\s+[a-z][a-z'’\-]+){0,2})")
    for m in low_re.finditer(text):
        name = _lowercase_name(m.group(1))
        if name:
            s = m.start(1)
            spans.append(Span("NAME", s, s + len(name), name))

    # A name opening the message: "ama here", "akosua from Sunyani", "hello, yaw here".
    open_re = re.compile(r"^\s*(?:(?i:hi|hello|hey|good\s+(?:morning|afternoon|evening|day)|greetings)\b[\s,!.\-]*)?"
                         r"([A-Za-z][a-zA-Z'’\-]+)\s+(?i:here|from|speaking|writing)\b")
    m = open_re.match(text)
    if m and not _plain_word(m.group(1)):
        spans.append(Span("NAME", m.start(1), m.end(1), m.group(1)))

    place_re = re.compile(rf"\b(?i:{_PLACE_TRIGGERS})\s+({_NAME_WORD}(?:[\s\-]{_NAME_WORD}){{0,3}})")
    for m in place_re.finditer(text):
        raw = m.group(1)
        if not raw[:1].isupper():
            continue
        place = _trim_safe_tail(raw)
        if place and not _is_safe(place):
            s = m.start(1)
            spans.append(Span("LOCATION", s, s + len(place), place))

    for kind, gaz in (("NAME", _gazetteer("names")), ("LOCATION", _gazetteer("places"))):
        for entry in gaz:
            for m in re.finditer(rf"\b{re.escape(entry)}\b", text):
                spans.append(Span(kind, m.start(), m.end(), m.group()))

    return _resolve_overlaps(spans)


_PRIORITY = {"EMAIL": 0, "GPS": 1, "ID": 2, "ADDRESS": 3, "PHONE": 4, "NAME": 5, "LOCATION": 6}


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    spans = sorted(spans, key=lambda s: (s.start, _PRIORITY[s.kind], -(s.end - s.start)))
    kept: list[Span] = []
    for sp in spans:
        if kept and sp.start < kept[-1].end:
            prev = kept[-1]
            # keep the higher-priority (or longer) one
            if (_PRIORITY[sp.kind], -(sp.end - sp.start)) < (_PRIORITY[prev.kind], -(prev.end - prev.start)):
                kept[-1] = sp
            continue
        kept.append(sp)
    return kept


def redact_text(text: str) -> tuple[str, list[Span]]:
    spans = _find_spans(text or "")
    out = []
    last = 0
    for sp in spans:
        out.append(text[last:sp.start])
        out.append(f"[{sp.kind}]")
        last = sp.end
    out.append(text[last:])
    return "".join(out), spans


def summary(spans: list[Span]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sp in spans:
        counts[sp.kind] = counts.get(sp.kind, 0) + 1
    return counts
