"""List every Spark content-filter refusal with its question and time, for a support request.

    python bench/refusal_log.py

Records written by the current runner carry the exact send time ("ts", UTC). Older records don't, so
for those it prints the run's start time (from environment_*.json) and the results file's last-write
time, which bound when the request was sent.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MYT = timezone(timedelta(hours=8))


def fmt(dt: datetime) -> str:
    return f"{dt.astimezone(timezone.utc):%Y-%m-%d %H:%M UTC} ({dt.astimezone(MYT):%d %b %H:%M MYT})"


def main() -> None:
    qs = {json.loads(l)["id"]: json.loads(l)["question"] for l in (HERE / "data" / "questions.jsonl").read_text("utf-8").splitlines() if l.strip()}
    res = HERE / "results"
    for f in sorted(res.glob("environment*.json")):
        try:
            utc = json.loads(f.read_text("utf-8")).get("utc")
            print(f"{f.name}: latest run started {fmt(datetime.fromisoformat(utc))}")
        except (ValueError, TypeError, OSError):
            pass
    total = 0
    for f in sorted(res.glob("*_cloud*.jsonl")):
        # regrade_qa.py rewrote the results file but kept a .bak copy with the original write time
        bak = f.with_suffix(".jsonl.bak")
        mtime = datetime.fromtimestamp((bak if bak.exists() else f).stat().st_mtime, timezone.utc)
        rows = [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()]
        hits = [r for r in rows if r.get("blocked") or "content_polic" in (r.get("error") or "")]
        if not hits:
            continue
        print(f"\n{f.name} (last written {fmt(mtime)}): {len(hits)} refused request(s), "
              f"{len({r['id'] for r in hits})} distinct question(s)")
        for r in hits:
            when = fmt(datetime.fromisoformat(r["ts"])) if r.get("ts") else "time not recorded (older run)"
            print(f"  {r['id']:6s} {when}  {qs.get(r['id'], '')}")
        total += len(hits)
    print(f"\nTotal refused requests found: {total}")


if __name__ == "__main__":
    main()
