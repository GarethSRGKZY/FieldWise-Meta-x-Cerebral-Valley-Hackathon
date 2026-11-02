# Redaction evaluation (redaction_cases.jsonl, 35 messages)

| Kind | Caught | Total | Recall |
|---|---|---|---|
| ADDRESS | 2 | 2 | 100% |
| EMAIL | 1 | 1 | 100% |
| GPS | 2 | 2 | 100% |
| ID | 2 | 2 | 100% |
| LOCATION | 7 | 10 | 70% |
| NAME | 13 | 15 | 87% |
| PHONE | 8 | 8 | 100% |
| **All** | 35 | 40 | 88% |

## Misses (5)

- `r19` NAME: **kwame** survived in: _hi im kwame from wenchi, maize leaves have holes_
- `r19` LOCATION: **wenchi** survived in: _hi im kwame from wenchi, maize leaves have holes_
- `r20` NAME: **Abena** survived in: _Abena told me to spray, but I want to check the guide first._
- `r21` LOCATION: **Kumasi** survived in: _Kumasi farmers are all seeing this leaf blight._
- `r23` LOCATION: **Odumase** survived in: _Plot 14 behind the Methodist church, Odumase. Tomato leaf curl._

## Over-redaction (0)

None.
