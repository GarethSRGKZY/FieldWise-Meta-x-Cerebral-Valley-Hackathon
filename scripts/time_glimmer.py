"""Time the local Glimmer on a few real Fieldwise requests (decides the hub hardware).

    python scripts/time_glimmer.py                    # 3 guide questions from the dev split
    python scripts/time_glimmer.py --image leaf.jpg   # plus one photo diagnosis (needs the mmproj)

Prints seconds per answer and tokens per second, and writes them to
bench/results/glimmer_timing.json together with the model file the server reports.
Run it with nothing else heavy open; close browser tabs on the hub.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from fieldwise.config import settings  # noqa: E402
from fieldwise.imageprep import prepare  # noqa: E402
from fieldwise.pipeline import Fieldwise  # noqa: E402
from fieldwise.providers.glimmer import GlimmerUnavailable  # noqa: E402
from run_bench import glimmer_is_stub  # noqa: E402


def server_models() -> list:
    """What the llama.cpp server says it is serving (records the quant file in the results)."""
    import httpx
    headers = {"Authorization": f"Bearer {settings.glimmer_api_key}"} if settings.glimmer_api_key else {}
    try:
        data = httpx.get(f"{settings.glimmer_base_url}/models", headers=headers, timeout=5, trust_env=False).json()
        return data.get("data", data.get("models", []))
    except Exception:  # noqa: BLE001
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--crop", default="maize")
    args = ap.parse_args()

    if glimmer_is_stub():
        sys.exit("The Glimmer endpoint is the fake stub; start the real model first.")
    fw = Fieldwise()
    if not fw.glimmer.health():
        sys.exit(f"No Glimmer server at {settings.glimmer_base_url}. Start scripts/start_glimmer.* first.")
    if not len(fw.index):
        sys.exit("No guide index. Run python scripts/ingest_guides.py first.")

    qs = [json.loads(l) for l in (ROOT / "bench" / "data" / "questions.jsonl").read_text("utf-8").splitlines() if l.strip()]
    qs = [q for q in qs if q["split"] == "dev" and q["answerable"]][: args.n]

    runs = []
    print("Warm-up (loads the model into memory; not counted)...", flush=True)
    t0 = time.perf_counter()
    try:
        fw.glimmer.chat("Answer briefly.", "Say OK.", want_logprobs=False, max_tokens=512)
    except GlimmerUnavailable as e:
        sys.exit(f"Glimmer failed on a trivial request: {e}")
    print(f"  warm-up took {time.perf_counter() - t0:.0f} s\n")

    for q in qs:
        r = fw.local_qa(q["question"], q.get("crop"))
        u = r["usage"] or {}
        out_tok = u.get("completion_tokens") or 0
        runs.append({"kind": "qa", "id": q["id"], "latency_s": round(r["latency_s"], 1),
                     "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": out_tok,
                     "tokens_per_s_overall": round(out_tok / r["latency_s"], 2) if r["latency_s"] else None,
                     "finish_reason": r["finish_reason"], "supported": r["supported"],
                     "confidence": round(r["confidence"], 3)})
        print(f"  {q['id']}: {r['latency_s']:.0f} s, {u.get('prompt_tokens')} prompt + {out_tok} output tokens "
              f"({runs[-1]['tokens_per_s_overall']} tok/s overall), finish={r['finish_reason']}", flush=True)
        if r["finish_reason"] == "length":
            print("    ! hit max_tokens: reasoning ran out of room; raise GLIMMER_MAX_TOKENS or the context.")

    if args.image:
        jpeg = prepare(args.image.read_bytes()).jpeg
        try:
            r = fw.local_dx(jpeg, args.crop)
            u = r["usage"] or {}
            runs.append({"kind": "photo", "latency_s": round(r["latency_s"], 1), "prompt_tokens": u.get("prompt_tokens"),
                         "completion_tokens": u.get("completion_tokens"), "label": r["label"],
                         "confidence": round(r["confidence"], 3), "finish_reason": r["finish_reason"]})
            print(f"  photo: {r['latency_s']:.0f} s, label {r['label']} @ {r['confidence']:.2f}")
        except GlimmerUnavailable as e:
            print(f"  photo failed: {e}  (is the server started with GLIMMER_MMPROJ?)")

    qa = [x["latency_s"] for x in runs if x["kind"] == "qa"]
    if qa:
        print(f"\nQ&A: median {sorted(qa)[len(qa) // 2]:.0f} s per answer over {len(qa)} questions.")
        n_total = 132
        print(f"At that speed one local pass over the {n_total} questions takes about "
              f"{sorted(qa)[len(qa) // 2] * n_total / 3600:.1f} h.")
    info = {"server_models": server_models(),
            "settings": {"ctx": "see GLIMMER_CTX", "max_tokens": settings.glimmer_max_tokens,
                         "reasoning_strength": settings.glimmer_reasoning_strength},
            "machine": {"os": platform.platform(), "cpu": platform.processor()},
            "note": "Fill in by hand: GPU model + VRAM, system RAM, quant file, -ngl value.", "runs": runs}
    out = ROOT / "bench" / "results" / "glimmer_timing.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
