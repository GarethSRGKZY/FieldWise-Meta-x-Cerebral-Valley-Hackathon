"""Prompts shared by Glimmer (local) and Spark (cloud).

Both models get the same instructions and the same retrieved passages, so the
benchmark compares models rather than prompts. The fixed final-line format
("ANSWER: ...", "SUPPORTED: YES/NO") is what lets us read token
probabilities from Glimmer's logprobs.
"""
from __future__ import annotations

from .labels import options_for
from .retrieval import Passage

QA_SYSTEM = (
    "You are Fieldwise, a crop-protection assistant for smallholder farmers and "
    "agricultural extension workers. Answer ONLY from the numbered extension-guide "
    "passages you are given, and cite them like [S1]. Never invent pesticide names, "
    "doses, mixing ratios or pre-harvest intervals: only state ones that appear word "
    "for word in a passage. If the passages do not answer the question, say so plainly "
    "and advise the farmer to contact their local extension officer. Use short, simple "
    "sentences a farmer can act on."
)

QA_FORMAT = (
    "Reply in exactly this format, with nothing after the second line:\n"
    "ANSWER: <your answer, with citations like [S1]>\n"
    "SUPPORTED: <YES if the passages fully support your answer, otherwise NO>"
)

DX_SYSTEM = (
    "You are a plant-health assistant. Look carefully at the photo and choose the "
    "single best option. If the photo does not clearly match one option, choose the "
    "'Unsure' option instead of guessing."
)


def format_passages(passages: list[Passage]) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        where = f", p. {p.page}" if p.page else ""
        lines.append(f"[S{i}] ({p.title}{where})\n{p.text}")
    return "\n\n".join(lines) if lines else "(no passages found)"


def qa_user_prompt(question: str, passages: list[Passage], crop: str | None = None) -> str:
    crop_line = f"Crop: {crop}\n" if crop else ""
    return (
        f"Passages:\n{format_passages(passages)}\n\n"
        f"{crop_line}Question: {question}\n\n{QA_FORMAT}"
    )


def dx_user_prompt(crop: str, note: str | None = None) -> str:
    opts = options_for(crop)
    lines = [f"{letter}. {lab.name if lab else 'Unsure / none of these'}" for letter, lab in opts]
    note_line = f"Farmer's note: {note}\n" if note else ""
    return (
        f"Crop: {crop.capitalize()}\n{note_line}Options:\n" + "\n".join(lines) +
        "\n\nReply with exactly one final line: ANSWER: <letter>"
    )
