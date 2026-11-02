"""Turn Glimmer's token logprobs into confidence scores.

The model ends its reply with a fixed line ("ANSWER: C" or "SUPPORTED: YES").
We find the token right after that marker and read the probability the model
gave to each allowed value at that position (from `top_logprobs`). For the
diagnosis, that gives a probability over the lettered classes rather than a
self-reported "I'm 90% sure".

If the runtime returns no logprobs, callers fall back to parsing the text and
mark the confidence source as "text" so the benchmark can report it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class TokenChoice:
    chosen: str | None            # value the model actually wrote
    probs: dict[str, float]       # normalised probability per allowed value
    source: str                   # "logprobs" | "text" | "missing"

    @property
    def p_chosen(self) -> float:
        if self.chosen is None:
            return 0.0
        return self.probs.get(self.chosen, 0.0)


def _tok_text(tok: dict) -> str:
    return tok.get("token") or ""


def _after_marker(tokens: list[dict], marker: str) -> int | None:
    """Index of the first non-blank token after the LAST occurrence of `marker`."""
    if not tokens:
        return None
    joined = ""
    starts = []
    for t in tokens:
        starts.append(len(joined))
        joined += _tok_text(t)
    pos = joined.upper().rfind(marker.upper())
    if pos < 0:
        return None
    end = pos + len(marker)
    for i, start in enumerate(starts):
        tok_end = start + len(_tok_text(tokens[i]))
        if tok_end <= end:
            continue
        # this token overlaps or follows the marker end
        piece = joined[max(start, end):tok_end]
        if piece.strip():
            return i
    return None


def choice_from_logprobs(
    logprob_tokens: list[dict] | None,
    marker: str,
    allowed: list[str],
    text: str,
) -> TokenChoice:
    allowed_up = [a.upper() for a in allowed]
    chosen_text = _choice_from_text(text, marker, allowed_up)

    if not logprob_tokens:
        return TokenChoice(chosen_text, {chosen_text: 1.0} if chosen_text else {}, "text" if chosen_text else "missing")

    idx = _after_marker(logprob_tokens, marker)
    if idx is None:
        return TokenChoice(chosen_text, {chosen_text: 1.0} if chosen_text else {}, "text" if chosen_text else "missing")

    tok = logprob_tokens[idx]
    cands: dict[str, float] = {}
    for alt in (tok.get("top_logprobs") or []) + [tok]:
        value = _tok_text(alt).strip().upper().rstrip(".:,)")
        lp = alt.get("logprob")
        if lp is None:
            continue
        # A token like "YES" or "C" must match exactly; prefixes like "Y" also map for YES/NO.
        match = next((a for a in allowed_up if value == a or (len(a) > 1 and value and a.startswith(value) and len(value) >= 2)), None)
        if match is None:
            continue
        cands[match] = max(cands.get(match, -math.inf), float(lp))

    if not cands:
        return TokenChoice(chosen_text, {chosen_text: 1.0} if chosen_text else {}, "text" if chosen_text else "missing")

    m = max(cands.values())
    exp = {k: math.exp(v - m) for k, v in cands.items()}
    z = sum(exp.values())
    probs = {k: v / z for k, v in exp.items()}
    chosen = _tok_text(tok).strip().upper().rstrip(".:,)")
    chosen = next((a for a in allowed_up if chosen == a or (len(a) > 1 and a.startswith(chosen) and len(chosen) >= 2)), chosen_text)
    return TokenChoice(chosen, probs, "logprobs")


def _choice_from_text(text: str, marker: str, allowed_up: list[str]) -> str | None:
    pattern = re.escape(marker) + r"\s*\**\s*([A-Za-z]+)"
    hits = re.findall(pattern, text or "", flags=re.IGNORECASE)
    for hit in reversed(hits):
        h = hit.upper()
        if h in allowed_up:
            return h
    return None


def extract_answer_text(text: str) -> str:
    """The free-text part after 'ANSWER:' and before 'SUPPORTED:'."""
    m = re.search(r"ANSWER:\s*(.*?)(?:\n\s*SUPPORTED:|\Z)", text or "", flags=re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()
