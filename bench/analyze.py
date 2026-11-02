"""Turn raw benchmark records into the numbers and chart for the write-up.

    python bench/analyze.py --task images
    python bench/analyze.py --task qa

What it does, in the order the integrity rules require:
1. Joins local (Glimmer) and cloud (Spark) records per item.
2. Picks the routing threshold on the DEV split only: highest routed
   accuracy, ties broken toward more answers on-device.
3. Reports TEST numbers for local-only, cloud-only, routed at the chosen
   threshold, and the full threshold sweep (so the trade-off is visible,
   not just the best point).
4. Writes failure cases, most dangerous first (confident local errors).

Accuracy comes with a 95% Wilson interval. With ~200 test items that interval
is several points wide; say so rather than over-reading small differences.
Cost per query counts Meta API spend only; local hardware and power are not
priced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = [round(x * 0.05, 2) for x in range(0, 21)]

INK, LEAF, SKY, GRIDC, MUTED = "#123524", "#2f7d3a", "#2563a8", "#e3e8e5", "#5b6f62"


# ------------------------------------------------------------------ loading
def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    recs = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r          # last record per id wins
    return recs


def apply_manual_grades(local: dict, cloud: dict, path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open() as fh:
        for row in csv.DictReader(fh):
            target = local if row["setup"] == "local" else cloud
            if row["id"] in target:
                target[row["id"]]["correct"] = row["correct"].strip().lower() in ("1", "true", "yes", "y")
                target[row["id"]]["graded_by"] = "manual"
                n += 1
    return n


# ------------------------------------------------------------------ stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))   # clamp float noise at 0% / 100%


def pctl(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Probability a correct answer gets higher confidence than a wrong one."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(scores: list[float], labels: list[bool], bins: int = 10) -> float:
    total, err = len(scores), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if idx:
            conf = sum(scores[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return err


def describe(items: list[dict], key_correct="correct", key_lat="latency_s", key_cost="cost_usd") -> dict:
    n = len(items)
    k = sum(bool(i[key_correct]) for i in items)
    lats = [i[key_lat] for i in items if i.get(key_lat) is not None]
    lo, hi = wilson(k, n)
    return {
        "n": n, "correct": k, "accuracy": k / n if n else float("nan"), "ci95": [lo, hi],
        "latency_mean_s": sum(lats) / len(lats) if lats else float("nan"),
        "latency_median_s": pctl(lats, 0.5), "latency_p90_s": pctl(lats, 0.9),
        "cost_per_query_usd": sum(i.get(key_cost) or 0 for i in items) / n if n else float("nan"),
    }


# ------------------------------------------------------------------ routing replay
def joined(local: dict, cloud: dict, split: str) -> tuple[list[tuple[dict, dict]], dict]:
    pairs, dropped = [], {"local_error": 0, "missing_cloud": 0}
    for iid, l in local.items():
        if l["split"] != split:
            continue
        if l.get("error"):
            dropped["local_error"] += 1
            continue
        c = cloud.get(iid)
        if c is None:
            dropped["missing_cloud"] += 1
            continue
        pairs.append((l, c))
    return pairs, dropped


def route(pairs: list[tuple[dict, dict]], tau: float) -> list[dict]:
    out = []
    for l, c in pairs:
        esc = (l.get("confidence") or 0.0) < tau
        cloud_ok = not c.get("error")
        use_cloud = esc and cloud_ok
        out.append({
            "id": l["id"], "escalated": esc, "cloud_failed": esc and not cloud_ok,
            "correct": c["correct"] if use_cloud else l["correct"],
            "latency_s": (l.get("latency_s") or 0) + ((c.get("latency_s") or 0) if esc else 0),
            "cost_usd": (c.get("cost_usd") or 0) if esc else 0.0,
            # carried through for the Q&A breakdown (absent for images)
            "answerable": l.get("answerable"), "retrieval_fact_hit": l.get("retrieval_fact_hit"),
            "removed_sentences": (c if use_cloud else l).get("removed_sentences"),
        })
    return out


def sweep(pairs, grid=GRID) -> list[dict]:
    rows = []
    for tau in grid:
        r = route(pairs, tau)
        d = describe(r)
        rows.append({"threshold": tau, "on_device_share": 1 - sum(x["escalated"] for x in r) / len(r),
                     "accuracy": d["accuracy"], "ci_low": d["ci95"][0], "ci_high": d["ci95"][1],
                     "latency_median_s": d["latency_median_s"], "cost_per_query_usd": d["cost_per_query_usd"]})
    return rows


def pick_threshold(dev_rows: list[dict], tolerance_pts: float) -> float:
    """Most on-device threshold whose dev accuracy is within `tolerance_pts`
    points of the best dev accuracy. Pure max-accuracy would usually pick
    "send nearly everything to the cloud", which defeats offline-first.
    Fix the tolerance BEFORE looking at test numbers."""
    best = max(r["accuracy"] for r in dev_rows)
    ok = [r for r in dev_rows if r["accuracy"] >= best - tolerance_pts / 100 - 1e-9]
    return max(ok, key=lambda r: (r["on_device_share"], r["accuracy"]))["threshold"]


# ------------------------------------------------------------------ chart
def chart(task: str, test_rows: list[dict], local_d: dict, cloud_d: dict, tau: float, out: Path, n_test: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    xs = [r["on_device_share"] * 100 for r in test_rows]
    ys = [r["accuracy"] * 100 for r in test_rows]
    ax.plot(xs, ys, color=INK, lw=2, solid_capstyle="round", solid_joinstyle="round", label="Routed (Glimmer, then Spark if unsure)", zorder=2)
    ax.scatter(xs, ys, s=14, color=INK, zorder=3)

    for name, d, x, color in (("Glimmer only", local_d, 100, LEAF), ("Spark only", cloud_d, 0, SKY)):
        y = d["accuracy"] * 100
        ax.errorbar([x], [y], yerr=[[y - d["ci95"][0] * 100], [d["ci95"][1] * 100 - y]], fmt="none",
                    ecolor=color, elinewidth=1.5, capsize=4, zorder=4)
        ax.scatter([x], [y], s=90, color=color, edgecolor="white", linewidth=2, zorder=5, label=f"{name} ({y:.0f}%)")
        # label sits above the CI cap so it never lands on the routed line
        ax.annotate(name, (x, d["ci95"][1] * 100), xytext=(-6 if x == 100 else 6, 5), textcoords="offset points",
                    ha="right" if x == 100 else "left", va="bottom", fontsize=9, color=INK)

    chosen = min(test_rows, key=lambda r: abs(r["threshold"] - tau))
    cx, cy = chosen["on_device_share"] * 100, chosen["accuracy"] * 100
    ax.scatter([cx], [cy], s=110, facecolor="white", edgecolor=INK, linewidth=2, zorder=6)
    ax.annotate(f"threshold {tau:.2f} (chosen on dev)\n{cy:.0f}% accurate, {cx:.0f}% on-device", (cx, cy),
                xytext=(0, -34), textcoords="offset points", ha="center", fontsize=8.5, color=INK)

    ax.set_xlim(-5, 105)
    lo = min(ys + [local_d["ci95"][0] * 100, cloud_d["ci95"][0] * 100])
    ax.set_ylim(max(0, lo - 12), 100)
    ax.set_xlabel("Answered on-device (%)", color=MUTED)
    ax.set_ylabel("Accuracy on held-out test set (%)", color=MUTED)
    what = "leaf-photo diagnosis" if task == "images" else "guide-grounded Q&A"
    ax.set_title(f"Fieldwise {what}: accuracy vs on-device share (n={n_test})", loc="left", fontsize=11, color=INK)
    ax.grid(color=GRIDC, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
def _rate(rows: list[dict], key="correct") -> dict:
    k = sum(1 for r in rows if r.get(key))
    lo, hi = wilson(k, len(rows)) if rows else (0.0, 0.0)
    return {"n": len(rows), "k": k, "rate": (k / len(rows)) if rows else None, "ci95": [lo, hi]}


def qa_breakdown(test_pairs: list[tuple[dict, dict]], routed: list[dict]) -> dict:
    """Where Q&A errors come from: retrieval missing the answer vs the model, and scope handling."""
    out = {}
    for name, recs in (("glimmer", [l for l, _ in test_pairs]), ("spark", [c for _, c in test_pairs]), ("routed", routed)):
        ans = [r for r in recs if r.get("answerable")]
        oos = [r for r in recs if r.get("answerable") is False]
        hit = [r for r in ans if r.get("retrieval_fact_hit") is True]
        miss = [r for r in ans if r.get("retrieval_fact_hit") is False]
        out[name] = {
            "answerable": _rate(ans), "out_of_scope_declined": _rate(oos),
            "answerable_when_retrieval_found_facts": _rate(hit),
            "answerable_when_retrieval_missed_facts": _rate(miss),
            "answers_with_quantity_removed_by_safety": sum(1 for r in recs if r.get("removed_sentences")),
        }
    ans_l = [l for l, _ in test_pairs if l.get("answerable")]
    out["retrieval_fact_hit"] = _rate(ans_l, key="retrieval_fact_hit") if any(
        "retrieval_fact_hit" in l for l in ans_l) else None
    # Spark's content-filter refusals, split by question type, and Spark's accuracy on the questions it did
    # answer: a secondary number only (the headline counts a refusal as no answer, as the farmer sees it).
    cloud = [c for _, c in test_pairs]
    blocked = [c for c in cloud if c.get("blocked")]
    out["spark_blocked"] = {
        "answerable": sum(1 for c in blocked if c.get("answerable")),
        "out_of_scope": sum(1 for c in blocked if c.get("answerable") is False),
        "ids": sorted(c["id"] for c in blocked),
        "accuracy_on_unblocked": _rate([c for c in cloud if not c.get("blocked")]),
        "accuracy_on_unblocked_answerable": _rate([c for c in cloud if not c.get("blocked") and c.get("answerable")]),
    }
    return out


def qa_breakdown_md(b: dict) -> list[str]:
    def pct(d):
        return "–" if not d or d["rate"] is None else f"{d['rate']*100:.0f}% ({d['k']}/{d['n']})"
    md = ["### Q&A breakdown (test)", ""]
    if b.get("retrieval_fact_hit"):
        md += [f"Retrieval put every key fact in front of the model for {pct(b['retrieval_fact_hit'])} "
               "of answerable questions; that is the ceiling for grounded answers.", ""]
    md += ["| | Glimmer | Spark | Routed |", "|---|---|---|---|"]
    rows = [("Answerable: correct", "answerable"), ("…when retrieval found the facts", "answerable_when_retrieval_found_facts"),
            ("…when retrieval missed", "answerable_when_retrieval_missed_facts"),
            ("Out of scope: declined and referred", "out_of_scope_declined")]
    for label, key in rows:
        md.append(f"| {label} | {pct(b['glimmer'][key])} | {pct(b['spark'][key])} | {pct(b['routed'][key])} |")
    md.append(f"| Answers with an unsupported quantity removed | {b['glimmer']['answers_with_quantity_removed_by_safety']} | "
              f"{b['spark']['answers_with_quantity_removed_by_safety']} | {b['routed']['answers_with_quantity_removed_by_safety']} |")
    sb = b.get("spark_blocked")
    if sb and sb["ids"]:
        md += ["", f"Spark's content filter refused {len(sb['ids'])} test questions ({sb['answerable']} answerable, "
               f"{sb['out_of_scope']} out of scope): {', '.join(sb['ids'])}. On the questions it did answer, Spark scored "
               f"{pct(sb['accuracy_on_unblocked'])} overall and {pct(sb['accuracy_on_unblocked_answerable'])} on answerable ones "
               "(secondary figure; the headline counts a refusal as no answer)."]
    return md + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["images", "qa"], required=True)
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="accuracy points you'll give up (vs best dev routed accuracy) to keep answers on-device")
    args = ap.parse_args()
    R, sfx = args.results_dir, (f"_{args.tag}" if args.tag else "")

    local = load(R / f"{args.task}_local{sfx}.jsonl")
    cloud = load(R / f"{args.task}_cloud{sfx}.jsonl")
    if not local or not cloud:
        raise SystemExit("Need both local and cloud result files. Run bench/run_bench.py first.")
    n_manual = apply_manual_grades(local, cloud, HERE / "data" / "qa_grades_manual.csv") if args.task == "qa" else 0

    dev_pairs, dev_drop = joined(local, cloud, "dev")
    test_pairs, test_drop = joined(local, cloud, "test")
    if not dev_pairs or not test_pairs:
        raise SystemExit("Need results for both dev and test splits.")
    dev_rows = sweep(dev_pairs)
    tau = pick_threshold(dev_rows, args.tolerance)
    test_rows = sweep(test_pairs)

    test_local = [l for l, _ in test_pairs]
    test_cloud = [c for _, c in test_pairs]
    routed = route(test_pairs, tau)
    local_d, cloud_d, routed_d = describe(test_local), describe(test_cloud), describe(routed)
    routed_d["on_device_share"] = 1 - sum(r["escalated"] for r in routed) / len(routed)
    routed_d["cloud_failures"] = sum(r["cloud_failed"] for r in routed)

    confs = [l.get("confidence") or 0.0 for l in test_local]
    corr = [bool(l["correct"]) for l in test_local]
    summary = {
        "task": args.task, "threshold_chosen_on_dev": tau,
        "threshold_rule": f"most on-device threshold within {args.tolerance} accuracy points of best dev accuracy",
        "dev": {"n": len(dev_pairs), "dropped": dev_drop},
        "test": {"n": len(test_pairs), "dropped": test_drop},
        "glimmer_only": local_d, "spark_only": cloud_d, "routed": routed_d,
        "confidence_quality_test": {
            "auroc": auroc(confs, corr), "ece_10bin": ece(confs, corr),
            "source_counts": {s: sum(1 for l in test_local if l.get("confidence_source") == s)
                              for s in {l.get("confidence_source") for l in test_local}},
        },
        "abstentions_test": {"glimmer": sum(1 for l in test_local if l.get("abstained") or l.get("supported") is False),
                             "spark": sum(1 for c in test_cloud if c.get("abstained") or c.get("supported") is False)},
        "manual_grades_applied": n_manual,
        "spark_content_filter_blocks": {
            "dev": sorted(c["id"] for _, c in dev_pairs if c.get("blocked")),
            "test": sorted(c["id"] for c in test_cloud if c.get("blocked")),
            "scored_as": "no answer: the app shows the extension-officer referral (images: 'Unsure')",
        },
        "notes": ["Routed results are replayed per item from independent Glimmer and Spark runs.",
                  "Cost per query counts Meta Model API spend only (Spark tokens at Standard-mode prices).",
                  "Latency for routed = Glimmer latency + Spark latency when escalated."],
    }
    if args.task == "qa":
        summary["qa_breakdown_test"] = qa_breakdown(test_pairs, routed)
    (R / f"summary_{args.task}{sfx}.json").write_text(json.dumps(summary, indent=2))

    with (R / f"tradeoff_{args.task}{sfx}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split"] + list(test_rows[0]))
        w.writeheader()
        for split, rows in (("dev", dev_rows), ("test", test_rows)):
            for r in rows:
                w.writerow({"split": split, **r})

    # failure cases: confident local mistakes first, then routed mistakes
    fails = []
    for (l, c), r in zip(test_pairs, routed):
        if not r["correct"] or not l["correct"]:
            fails.append({
                "id": l["id"], "routed_correct": r["correct"], "escalated": r["escalated"],
                "glimmer_correct": l["correct"], "glimmer_confidence": round(l.get("confidence") or 0, 3),
                "spark_correct": c["correct"], "gold": l.get("gold", l.get("answerable")),
                "glimmer_pred": l.get("pred", l.get("answer_shown", "")[:160] if l.get("answer_shown") else ""),
                "spark_pred": c.get("pred", c.get("answer_shown", "")[:160] if c.get("answer_shown") else ""),
                "removed_by_safety": len(l.get("removed_sentences") or []) + len(c.get("removed_sentences") or []),
                "spark_blocked": bool(c.get("blocked")),
            })
    fails.sort(key=lambda f: (f["routed_correct"], -f["glimmer_confidence"]))
    if fails:
        with (R / f"failures_{args.task}{sfx}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fails[0]))
            w.writeheader()
            w.writerows(fails)

    chart(args.task, test_rows, local_d, cloud_d, tau, R / f"tradeoff_{args.task}{sfx}.png", len(test_pairs))

    def line(name, d, share):
        return (f"| {name} | {d['accuracy']*100:.1f}% ({d['ci95'][0]*100:.0f}–{d['ci95'][1]*100:.0f}) | {share} | "
                f"{d['latency_median_s']:.1f} s / {d['latency_p90_s']:.1f} s | ${d['cost_per_query_usd']:.4f} |")
    md = [f"## {args.task} (test n={len(test_pairs)}, threshold {tau:.2f} chosen on dev n={len(dev_pairs)})", "",
          f"Threshold rule: {summary['threshold_rule']}.", "",
          "| Setup | Accuracy (95% CI) | On-device | Latency median / p90 | API cost per query |",
          "|---|---|---|---|---|",
          line("Glimmer only (offline)", local_d, "100%"),
          line("Spark only (cloud)", cloud_d, "0%"),
          line(f"Routed @ {tau:.2f}", routed_d, f"{routed_d['on_device_share']*100:.0f}%"), "",
          f"Confidence quality on test: AUROC {summary['confidence_quality_test']['auroc']:.2f}, "
          f"ECE {summary['confidence_quality_test']['ece_10bin']:.3f}.", "",
          f"Muse Spark's content filter declined {len(summary['spark_content_filter_blocks']['test'])} of "
          f"{len(test_pairs)} test items (scored as no answer).", ""]
    if args.task == "qa":
        md += qa_breakdown_md(summary["qa_breakdown_test"])
    (R / f"summary_{args.task}{sfx}.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Wrote summary, trade-off CSV/PNG and {len(fails)} failure cases to {R}")


if __name__ == "__main__":
    main()
