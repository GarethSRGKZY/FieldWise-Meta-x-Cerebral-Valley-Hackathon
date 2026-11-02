"""Benchmark plumbing: question validation, grading, retrieval-hit recording, Q&A breakdown."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from analyze import qa_breakdown, route  # noqa: E402
from run_bench import grade_qa, retrieval_fact_hit, validate_questions  # noqa: E402

from fieldwise.retrieval import Passage, tokenize  # noqa: E402

Q = ROOT / "bench" / "data" / "questions.jsonl"


def q(**kw):
    base = {"id": "q1", "split": "dev", "crop": "maize", "type": "pest_id", "question": "What eats my maize?",
            "answerable": True, "key_facts": [["fall armyworm", "spodoptera"]],
            "source": {"doc_id": "fao-faw-ffs-2018", "page": 3}, "review_status": "reviewed"}
    base.update(kw)
    return base


def test_uk_spelling_and_farmer_words_match_guide_terms():
    assert tokenize("recognise the colour") == tokenize("recognize the color")
    expanded = tokenize("worms on my maize", expand=True)
    assert "larvae" in expanded and "caterpillar" in expanded
    assert "larvae" not in tokenize("worms on my maize")   # synonyms only when asked (queries)


def test_draft_questions_block_strict_runs():
    rows = [q(id=f"q{i}") for i in range(100)]
    assert not validate_questions(rows)
    rows[5]["review_status"] = "draft"
    problems = validate_questions(rows)
    assert any("draft" in p for p in problems)


def test_grading_answerable_out_of_scope_and_referral():
    item = q()
    assert grade_qa(item, True, "This is Fall Armyworm [S1].")["auto_correct"]
    assert not grade_qa(item, True, "Probably stem borer [S1].")["auto_correct"]
    assert not grade_qa(item, False, "The guides do not cover this.")["auto_correct"]
    assert grade_qa(q(accept_referral=True), False, "Ask your extension officer.")["auto_correct"]
    oos = q(answerable=False, key_facts=[])
    assert grade_qa(oos, False, "Not covered.")["auto_correct"]
    assert not grade_qa(oos, True, "Spray X.")["auto_correct"]


def test_retrieval_fact_hit():
    p = [Passage("c1", "d", "t", 3, "Larvae of the Fall Armyworm feed in the whorl.")]
    assert retrieval_fact_hit(q(), p) is True
    assert retrieval_fact_hit(q(key_facts=[["stem borer"]]), p) is False
    assert retrieval_fact_hit(q(answerable=False), p) is None


def test_qa_breakdown_splits_by_retrieval_hit():
    def rec(i, ok, hit, answerable=True):
        return {"id": f"x{i}", "confidence": 0.9, "correct": ok, "answerable": answerable,
                "retrieval_fact_hit": hit if answerable else None, "latency_s": 1, "cost_usd": 0}
    pairs = [(rec(0, True, True), rec(0, True, True)), (rec(1, False, False), rec(1, True, False)),
             (rec(2, True, None, answerable=False), rec(2, True, None, answerable=False))]
    b = qa_breakdown(pairs, route(pairs, 0.5))
    assert b["retrieval_fact_hit"]["k"] == 1 and b["retrieval_fact_hit"]["n"] == 2
    assert b["glimmer"]["answerable_when_retrieval_found_facts"]["rate"] == 1.0
    assert b["glimmer"]["answerable_when_retrieval_missed_facts"]["rate"] == 0.0
    assert b["spark"]["answerable_when_retrieval_missed_facts"]["rate"] == 1.0
    assert b["routed"]["out_of_scope_declined"]["n"] == 1


def test_shipped_question_set_is_well_formed():
    rows = [json.loads(l) for l in Q.read_text("utf-8").splitlines() if l.strip()]
    problems = [p for p in validate_questions(rows) if "draft" not in p]
    assert not problems, problems
    assert len(rows) >= 100
    assert {r["split"] for r in rows} == {"dev", "test"}


def test_review_csv_key_fact_format_round_trips():
    from review_questions import cell_to_facts, facts_to_cell
    facts = [["fall armyworm", "spodoptera"], ["whorl"]]
    assert facts_to_cell(facts) == "fall armyworm | spodoptera ; whorl"
    assert cell_to_facts(facts_to_cell(facts)) == facts


def test_image_manifest_is_portable_and_hash_checked(tmp_path):
    import csv
    import hashlib

    import pytest
    from run_bench import load_images
    img = tmp_path / "CCMT" / "Raw Data" / "Maize" / "healthy" / "a.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"fake-jpeg-bytes")
    man = tmp_path / "m.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "split", "crop", "label", "path", "sha1"])
        w.writeheader()
        w.writerow({"id": "i1", "split": "test", "crop": "maize", "label": "maize/healthy",
                    "path": "Raw Data/Maize/healthy/a.jpg", "sha1": hashlib.sha1(b"fake-jpeg-bytes").hexdigest()})
    rows = load_images(man, "all", tmp_path / "CCMT")
    assert Path(rows[0]["path"]) == img
    img.write_bytes(b"different")
    with pytest.raises(SystemExit):
        load_images(man, "all", tmp_path / "CCMT")


def test_errored_items_are_retried_on_resume(tmp_path):
    from run_bench import done_ids
    f = tmp_path / "r.jsonl"
    f.write_text('{"id": "a", "correct": true}\n{"id": "b", "error": "timeout", "correct": false}\n'
                 '{"id": "c", "error": "timeout", "correct": false}\n{"id": "c", "correct": false}\n')
    assert done_ids(f) == {"a", "c"}


def test_blocked_cloud_answer_is_scored_not_retried(tmp_path, monkeypatch):
    import run_bench
    from fieldwise.providers.spark import SparkBlocked

    class FakeIndex:
        def search(self, q, k=6):
            from fieldwise.retrieval import SearchResult
            return SearchResult([Passage("c1", "d", "t", 3, "Fall armyworm larvae feed on maize.")], 0.8, 1.0)

    class FakeFw:
        index = FakeIndex()

        def cloud_qa(self, *a, **k):
            raise SparkBlocked("Spark content filter declined the request: content_policy")

    rec = run_bench.run_item(FakeFw(), "qa", "cloud", q())
    assert rec["blocked"] and "error" not in rec and rec["correct"] is False
    oos = run_bench.run_item(FakeFw(), "qa", "cloud", q(answerable=False, key_facts=[]))
    assert oos["blocked"] and oos["correct"] is True      # the farmer gets the referral, which is right here


def test_safety_reminder_words_do_not_count_as_key_facts():
    from fieldwise.safety import PPE_REMINDER
    item = q(key_facts=[["children"]])       # a real key fact that also appears in the PPE reminder
    assert not grade_qa(item, True, "Use a hoe to remove weeds. " + PPE_REMINDER)["auto_correct"]
    assert grade_qa(item, True, "Keep children away from treated plots. " + PPE_REMINDER)["auto_correct"]
