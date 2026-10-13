import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vocabulary_review import VocabularyReview


def service(tmp_path):
    catalog=tmp_path/"catalog.json"
    records=[{"tag":tag,"definition":definition,"aliases":[{"alias":alias,"owners":[tag],"ambiguous":False}],
              "focus":True,"issues":[]} for tag,definition,alias in [
        ("white_dress","白色连衣裙","白色连身裙"),
        ("white_skirt","白色半身裙","白裙"),
        ("headpat","摸他人的头","摸摸头")]]
    records[1]["aliases"][0].update(owners=["white_dress","white_skirt"],ambiguous=True)
    catalog.write_text(json.dumps({"records":records,"source":{"version":"0.9.8"}},ensure_ascii=False),encoding="utf-8")
    return VocabularyReview(catalog,tmp_path/"decisions.json")


def test_search_evidence_keeps_conflicts_and_ambiguity_without_applying(tmp_path):
    s=service(tmp_path)
    result=s.snapshot(existing={"白裙":"white_dress"},query="白裙",page_size=1)
    assert result["total"]==1
    row=result["rows"][0]
    assert row["tag"]=="white_skirt"
    assert row["aliases"][0]["mapping_state"]=="conflict"
    assert row["aliases"][0]["ambiguous"] is True
    assert result["applied"] is False
    assert not s.decisions_path.exists()
    fuzzy=s.snapshot(existing={},query="白色连身群")
    assert fuzzy["rows"][0]["tag"]=="white_dress"
    assert "字面相近" in fuzzy["rows"][0]["search_reason"]


def test_review_persists_only_notes_and_rejects_stale_revision(tmp_path):
    s=service(tmp_path)
    existing={"白裙":"white_skirt"}
    before=dict(existing)
    revision=s.snapshot(existing=existing)["revision"]
    payload={"base_revision":revision,"tag":"white_dress","state":"modified",
             "aliases":["白色连身裙"],"note":"限定连身结构，待正反例验收"}
    result=s.decide(payload)
    assert result["applied"] is False
    assert existing==before
    assert s.snapshot(existing=existing,state="modified")["rows"][0]["decision"]["aliases"]==["白色连身裙"]
    with pytest.raises(ValueError,match="刷新"):
        s.decide({**payload,"note":"旧页面的修改"})
    assert json.loads(s.decisions_path.read_text(encoding="utf-8"))["white_dress"]["note"]==payload["note"]
    reloaded=VocabularyReview(s.catalog_path,s.decisions_path)
    assert reloaded.snapshot(existing=existing,state="modified")["total"]==1


def test_catalog_and_pagination_changes_do_not_erase_review_notes(tmp_path):
    s=service(tmp_path)
    result=s.snapshot(existing={},page=99,page_size=2)
    assert result["page"]==2
    rev=result["revision"]
    raw=json.loads(s.catalog_path.read_text(encoding="utf-8"))
    raw["source"]["version"]="new"
    s.catalog_path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(ValueError,match="刷新"):
        s.decide({"base_revision":rev,"tag":"headpat","state":"deferred"})
    rev=s.snapshot(existing={})["revision"]
    with pytest.raises(ValueError,match="格式"):
        s.decide({"base_revision":rev,"tag":"headpat","state":"proposed","aliases":["a=b"]})
    assert not s.decisions_path.exists()
