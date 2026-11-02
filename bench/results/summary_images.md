## images (test n=144, threshold 0.05 chosen on dev n=60)

Threshold rule: most on-device threshold within 2.0 accuracy points of best dev accuracy.

| Setup | Accuracy (95% CI) | On-device | Latency median / p90 | API cost per query |
|---|---|---|---|---|
| Glimmer only (offline) | 55.6% (47–63) | 100% | 12.6 s / 16.3 s | $0.0000 |
| Spark only (cloud) | 66.7% (59–74) | 0% | 4.3 s / 9.6 s | $0.0041 |
| Routed @ 0.05 | 60.4% (52–68) | 85% | 12.9 s / 19.1 s | $0.0007 |

Confidence quality on test: AUROC 0.76, ECE 0.299.

Muse Spark's content filter declined 0 of 144 test items (scored as no answer).
