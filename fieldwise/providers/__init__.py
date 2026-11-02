"""The three Meta models, one job each.

- glimmer.py  Muse Glimmer (open weights, local): first answer for every
              question and photo, offline.
- spark.py    Muse Spark (Model API): second opinion, only for low-confidence
              cases, only after on-device redaction.
- sam.py      SAM 3.1 (Model API): leaf and lesion masks for a severity %.
"""
