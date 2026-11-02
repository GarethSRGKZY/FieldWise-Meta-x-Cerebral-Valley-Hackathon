"""Agronomic guardrails applied in code to every answer, local or cloud.

1. Quantity check: any dose, ratio, interval or rate in the answer must appear
   in a passage the answer cites. Sentences that fail are removed and flagged.
   This is what stops a model from inventing "mix 50 ml per 15 L".
2. Chemical reminder: if the answer mentions a pesticide or spraying, a fixed
   PPE and follow-the-label reminder is appended. The model can't drop it.
3. Citation check: an answer that claims support must cite a real passage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import Passage

REFERRAL = (
    "The extension guides on this device don't cover this well enough for a safe answer. "
    "Please contact your local agricultural extension officer."
)

PPE_REMINDER = (
    "Safety: if you use any pesticide, read and follow the product label exactly. Wear gloves, "
    "a mask, long sleeves and boots, keep children and animals away, never eat, drink or smoke "
    "while spraying, and wait the pre-harvest interval on the label before harvesting. "
    "Ask your extension officer if you are unsure."
)

_UNITS = (
    r"%|percent|ml|millilit(?:er|re)s?|l\b|lit(?:er|re)s?|g\b|grams?|kg|kilograms?|cc|ppm|"
    r"days?|weeks?|hours?|hrs?|ha\b|hectares?|acres?|cm|mm|metres?|meters?|"
    r"tablespoons?|tbsp|teaspoons?|tsp|cups?|sachets?|knapsacks?|tanks?|times|"
    r"g/l|ml/l|kg/ha|l/ha|g/ha|ml/ha"
)
_QTY = re.compile(
    rf"(\d+(?:[.,]\d+)?)\s*(?:-|–|to)?\s*(\d+(?:[.,]\d+)?)?\s*({_UNITS})"
    rf"|(\d+)\s*:\s*(\d+)"                      # ratios 1:10
    rf"|(\d+)\s+(?:in|per)\s+(\d+)",            # "1 in 10", "2 per 15"
    re.IGNORECASE,
)

_CHEM = re.compile(
    r"\b(pesticides?|insecticides?|fungicides?|herbicides?|acaricides?|miticides?|nematicides?|"
    r"bactericides?|chemicals?|spray(?:ing|s|ed)?|active ingredient|emamectin|chlorantraniliprole|"
    r"spinetoram|spinosad|lambda[- ]?cyhalothrin|cypermethrin|deltamethrin|mancozeb|chlorothalonil|"
    r"copper|metalaxyl|azoxystrobin|imidacloprid|abamectin|bacillus thuringiensis|\bbt\b|neem|"
    r"botanical pesticide|biopesticide)\b",
    re.IGNORECASE,
)

_CITE = re.compile(r"\[S(\d+)\]")
_NOT_DOSE = {"days", "day", "weeks", "week", "hours", "hour", "hrs", "hr", "times", "%", "percent",
             "cm", "mm", "metres", "metre", "meters", "meter", "ha", "hectare", "hectares", "acre", "acres"}


@dataclass
class SafetyReport:
    text: str
    removed_sentences: list[str] = field(default_factory=list)
    chemical: bool = False
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        f = []
        if self.removed_sentences:
            f.append(f"Removed {len(self.removed_sentences)} sentence(s) with a quantity not found in the cited guide.")
        if self.invalid_citations:
            f.append("Answer cited a source that doesn't exist.")
        if self.chemical:
            f.append("Pesticide safety reminder added.")
        return f


def _norm_num(s: str | None) -> str | None:
    return s.replace(",", ".") if s else s


def _quantities(text: str) -> list[tuple[str, ...]]:
    out = []
    for m in _QTY.finditer(text):
        if m.group(1):
            out.append(("unit", _norm_num(m.group(1)), _norm_num(m.group(2)), m.group(3).lower()))
        elif m.group(4):
            out.append(("ratio", m.group(4), m.group(5)))
        elif m.group(6):
            out.append(("per", m.group(6), m.group(7)))
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", text.strip())
    return [p for p in parts if p]


def check(answer: str, passages: list[Passage]) -> SafetyReport:
    cited = sorted({int(n) for n in _CITE.findall(answer)})
    valid = [n for n in cited if 1 <= n <= len(passages)]
    invalid = [n for n in cited if n not in valid]
    # quantities may come from any cited passage; if none cited, from none
    source_text = " ".join(passages[n - 1].text for n in valid)
    source_q = _quantities(source_text)
    source_numbers = {x for q in source_q for x in q[1:] if x and re.fullmatch(r"[\d.]+", x)}

    kept, removed = [], []
    for sent in _sentences(answer):
        qs = _quantities(sent)
        ok = True
        for q in qs:
            nums = [x for x in q[1:] if x and re.fullmatch(r"[\d.]+", x)]
            if any(n not in source_numbers for n in nums):
                ok = False
        (kept if ok else removed).append(sent)

    text = " ".join(kept).strip()
    # Trade names are endless, so any dose-like quantity (volume/mass/rate) also
    # triggers the reminder, not just the keyword list.
    # Sizes, distances, areas and times ("3 to 4 cm wide", "1.5 m above the ground", "every 0.5-2 ha")
    # are not doses, so they don't add a pesticide reminder to answers about moths or traps.
    dose_like = any(q[0] == "unit" and q[3] not in _NOT_DOSE for q in _quantities(text))
    chemical = bool(_CHEM.search(text)) or dose_like or bool(re.search(r"\b(apply|application|dose|dosage|mix)\b", text, re.IGNORECASE))
    if chemical:
        text = f"{text}\n\n{PPE_REMINDER}" if text else PPE_REMINDER
    return SafetyReport(text, removed, chemical, valid, invalid)
