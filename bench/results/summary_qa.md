## qa (test n=87, threshold 0.90 chosen on dev n=45)

Threshold rule: most on-device threshold within 2.0 accuracy points of best dev accuracy.

| Setup | Accuracy (95% CI) | On-device | Latency median / p90 | API cost per query |
|---|---|---|---|---|
| Glimmer only (offline) | 83.9% (75–90) | 100% | 21.6 s / 31.6 s | $0.0000 |
| Spark only (cloud) | 79.3% (70–86) | 0% | 5.4 s / 8.5 s | $0.0070 |
| Routed @ 0.90 | 82.8% (73–89) | 46% | 23.5 s / 35.0 s | $0.0036 |

Confidence quality on test: AUROC 0.63, ECE 0.160.

Muse Spark's content filter declined 9 of 87 test items (scored as no answer).

### Q&A breakdown (test)

Retrieval put every key fact in front of the model for 85% (64/75) of answerable questions; that is the ceiling for grounded answers.

| | Glimmer | Spark | Routed |
|---|---|---|---|
| Answerable: correct | 81% (61/75) | 76% (57/75) | 80% (60/75) |
| …when retrieval found the facts | 95% (61/64) | 89% (57/64) | 94% (60/64) |
| …when retrieval missed | 0% (0/11) | 0% (0/11) | 0% (0/11) |
| Out of scope: declined and referred | 100% (12/12) | 100% (12/12) | 100% (12/12) |
| Answers with an unsupported quantity removed | 1 | 1 | 0 |

Spark's content filter refused 9 test questions (6 answerable, 3 out of scope): q-c15, q-c18, q-m09, q-m21, q-m54, q-m63, q-o09, q-o15, q-o18. On the questions it did answer, Spark scored 85% (66/78) overall and 83% (57/69) on answerable ones (secondary figure; the headline counts a refusal as no answer).
