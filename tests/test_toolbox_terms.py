from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toolbox_terms as importer


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sample(tmp_path):
    source, config, report = (tmp_path / name for name in ("tags.json", "config.json", "import.json"))
    write_json(source, {"general": [
        {"tag": "dress", "zhFull": ["连衣裙"], "zhShort": ["裙子"], "zhNick": ["洋装"]},
        {"tag": "skirt", "zhFull": ["裙子"]},
        {"tag": "white_dress", "zhFull": ["白裙"]},
    ], "character": [{"tag": "someone", "zhFull": ["某角色"]}]})
    write_json(config, {"lookup": {"danbooru_term_mappings": ["连衣裙=manual_dress"]},
                        "other": {"keep": "unchanged"}})
    return source, config, report


def test_general_aliases_only_and_ambiguity_not_resolved_by_iteration_order():
    records = [
        {"tag": "dress", "zhFull": [" 连衣裙 ", "连衣裙", "裙子", "x", "换行\n词", "甲=乙"],
         "zhShort": ["洋装"], "zhNick": ["洋服"], "zh": ["服装"]},
        {"tag": "skirt", "zhFull": ["裙子"]},
        {"tag": "bad/tag", "zhFull": ["无效标签"]},
    ]
    result = importer.extract_terms({"general": records, "artist": [
        {"tag": "artist", "zhFull": ["画师"]}]})
    assert result["candidates"] == {"洋服": "dress", "洋装": "dress", "服装": "dress", "连衣裙": "dress"}
    assert result["ambiguous"] == {"裙子": ["dress", "skirt"]}
    reversed_result = importer.extract_terms({"general": records[::-1]})
    for field in ("candidates", "ambiguous", "invalid_occurrences"):
        assert reversed_result[field] == result[field]
    assert {row["reason"] for row in result["invalid_records"]} == {
        "alias_length_out_of_range", "alias_contains_control", "alias_contains_separator", "invalid_canonical_tag"
    }


def test_preview_apply_idempotence_and_incremental_restore_keep_manual_edits(tmp_path):
    source, config, report = sample(tmp_path)
    before = config.read_bytes()
    preview = importer.run_import(source, config, report)
    assert config.read_bytes() == before
    assert preview["conflicts"]["连衣裙"]["existing"] == "manual_dress"
    assert preview["additions"] == {"洋装": "dress", "白裙": "white_dress"}
    applied = importer.run_import(source, config, report, apply=True)
    current = json.loads(config.read_text(encoding="utf-8"))
    assert current["other"] == {"keep": "unchanged"}
    assert current["lookup"]["danbooru_term_mappings"] == ["连衣裙=manual_dress", "洋装=dress", "白裙=white_dress"]
    repeated = importer.run_import(source, config, tmp_path / "second.json", apply=True)
    assert repeated["additions"] == {}
    assert repeated["total"] == 3
    with pytest.raises(ValueError, match="新的报告"):
        importer.run_import(source, config, report, apply=True)
    current["lookup"]["danbooru_term_mappings"] = ["连衣裙=manual_dress", "洋装=edited_dress", "白裙=white_dress", "新词=new_tag"]
    current["other"]["keep"] = "later edit"
    write_json(config, current)
    assert importer.restore_import(config, applied) == 1
    restored = json.loads(config.read_text(encoding="utf-8"))
    assert restored["lookup"]["danbooru_term_mappings"] == ["连衣裙=manual_dress", "洋装=edited_dress", "新词=new_tag"]
    assert restored["other"]["keep"] == "later edit"
    assert importer.restore_import(config, applied) == 0


def test_overflow_and_existing_conflict_stop_before_writes(tmp_path, monkeypatch):
    source, config, report = sample(tmp_path)
    before = config.read_bytes()
    monkeypatch.setattr(importer, "MAX_TERMS", 2)
    with pytest.raises(ValueError, match="超过"):
        importer.run_import(source, config, report, apply=True)
    assert config.read_bytes() == before
    assert not report.exists()
    with pytest.raises(ValueError, match="冲突别名"):
        importer.merge_terms(["裙子=dress", "裙子=skirt"], {"candidates": {}})


def test_concurrent_config_edit_is_not_overwritten(tmp_path, monkeypatch):
    source, config, report = sample(tmp_path)
    original = importer.atomic_json

    def mutate_after_preview(path, value, **kwargs):
        original(path, value, **kwargs)
        if path == report:
            write_json(config, {"danbooru_term_mappings": ["用户新词=user_tag"]})

    monkeypatch.setattr(importer, "atomic_json", mutate_after_preview)
    with pytest.raises(ValueError, match="配置已变更"):
        importer.run_import(source, config, report, apply=True)
    assert json.loads(config.read_text(encoding="utf-8"))["danbooru_term_mappings"] == ["用户新词=user_tag"]


def test_pending_journal_recovers_when_final_manifest_write_fails(tmp_path, monkeypatch):
    source, config, report = sample(tmp_path)
    original = importer.atomic_json

    def fail_final_report(path, value, **kwargs):
        if path == report and value.get("applied"):
            raise OSError("report unavailable")
        original(path, value, **kwargs)

    monkeypatch.setattr(importer, "atomic_json", fail_final_report)
    with pytest.raises(OSError, match="unavailable"):
        importer.run_import(source, config, report, apply=True)
    pending = json.loads(report.read_text(encoding="utf-8"))
    assert pending["apply_pending"] and not pending["applied"]
    assert importer.restore_import(config, pending) == 2
    assert json.loads(config.read_text(encoding="utf-8"))["lookup"]["danbooru_term_mappings"] == ["连衣裙=manual_dress"]


def test_concurrent_edit_after_pending_journal_is_not_written_or_rolled_back(tmp_path, monkeypatch):
    source, config, report = sample(tmp_path)
    original = importer.atomic_json

    def mutate_after_pending(path, value, **kwargs):
        original(path, value, **kwargs)
        if path == report and value.get("apply_pending"):
            write_json(config, {"danbooru_term_mappings": ["洋装=dress", "用户新词=user_tag"]})

    monkeypatch.setattr(importer, "atomic_json", mutate_after_pending)
    with pytest.raises(ValueError, match="配置已变更"):
        importer.run_import(source, config, report, apply=True)
    cancelled = json.loads(report.read_text(encoding="utf-8"))
    assert not cancelled.get("apply_pending") and not cancelled["applied"]
    with pytest.raises(ValueError, match="没有已应用"):
        importer.restore_import(config, cancelled)
    assert json.loads(config.read_text(encoding="utf-8"))["danbooru_term_mappings"] == ["洋装=dress", "用户新词=user_tag"]


def test_large_vocabulary_keeps_limit_and_matcher_observes_in_place_edits():
    from danbooru_resolver import DanbooruResolver

    candidates = {f"中文词条{i}": f"tag_{i}" for i in range(43049)}
    merged = importer.merge_terms([], {"candidates": candidates})
    config = {"danbooru_term_mappings": merged["rows"]}
    resolver = DanbooruResolver(
        logger=None, cache={}, config=config, get_bool=lambda _k, d: d,
        get_int=lambda _k, d: d, get_float=lambda _k, d: d, get_str=lambda _k, d: d,
    )
    assert merged["total"] == 43049
    assert resolver.matched_term_mappings_for_prompt("中文词条43048") == (("中文词条43048", "tag_43048"),)
    config["danbooru_term_mappings"][-1] = "中文词条43048=edited_tag"
    result = resolver.cached_term_mappings_for_prompt("中文词条43048")
    assert result.confirmed_tags == ()
    assert result.candidate_tags == ("edited_tag",)
    resolver._config["danbooru_term_mappings"] = ["中文词条43048=new_tag"]
    assert resolver.matched_term_mappings_for_prompt("中文词条43048") == (("中文词条43048", "new_tag"),)
