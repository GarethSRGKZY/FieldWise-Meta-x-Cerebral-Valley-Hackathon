import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from analyze import auroc, pick_threshold, route, sweep, wilson  # noqa: E402


def pair(i, conf, l_ok, c_ok, c_err=False):
    l = {"id": f"x{i}", "split": "test", "confidence": conf, "correct": l_ok, "latency_s": 10.0, "cost_usd": 0}
    c = {"id": f"x{i}", "split": "test", "correct": c_ok, "latency_s": 4.0, "cost_usd": 0.01}
    if c_err:
        c["error"] = "timeout"
    return l, c


def test_route_replays_escalation_latency_and_cost():
    pairs = [pair(0, 0.9, True, False), pair(1, 0.2, False, True)]
    r = route(pairs, 0.5)
    assert [x["escalated"] for x in r] == [False, True]
    assert [x["correct"] for x in r] == [True, True]
    assert r[1]["latency_s"] == 14.0 and r[1]["cost_usd"] == 0.01 and r[0]["cost_usd"] == 0


def test_failed_cloud_call_falls_back_to_local_answer():
    r = route([pair(0, 0.1, True, False, c_err=True)], 0.5)
    assert r[0]["correct"] is True and r[0]["cloud_failed"]


def test_threshold_rule_prefers_on_device_within_tolerance():
    rows = [
        {"threshold": 0.0, "accuracy": 0.80, "on_device_share": 1.0},
        {"threshold": 0.5, "accuracy": 0.86, "on_device_share": 0.6},
        {"threshold": 0.9, "accuracy": 0.87, "on_device_share": 0.1},
    ]
    assert pick_threshold(rows, tolerance_pts=2.0) == 0.5
    assert pick_threshold(rows, tolerance_pts=0.0) == 0.9


def test_sweep_endpoints_are_local_only_and_cloud_only():
    pairs = [pair(i, i / 10, i % 2 == 0, True) for i in range(10)]
    rows = sweep(pairs, grid=[0.0, 1.01])
    assert rows[0]["on_device_share"] == 1.0 and rows[1]["on_device_share"] == 0.0
    assert rows[1]["accuracy"] == 1.0


def test_stats_helpers():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
    assert auroc([0.9, 0.8, 0.2], [True, True, False]) == 1.0
    assert auroc([0.1, 0.9], [True, False]) == 0.0
