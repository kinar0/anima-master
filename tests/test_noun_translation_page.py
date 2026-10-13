import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import main
from main import ComfyUIAgentPlugin
from noun_translation_batch import parse_existing_mappings


def _handler(tmp_path):
    catalog = tmp_path / "anima.csv"
    catalog.write_text("large_breasts,0,3000\ndress,0,2000\n", encoding="utf-8")
    (tmp_path / "noun_translation_review.json").write_text(json.dumps([
        {"tag": "large_breasts", "aliases": ["巨乳", "大胸"]},
        {"tag": "dress", "aliases": ["连衣裙"]},
    ], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "noun_translation_verified.json").write_text(json.dumps([
        {"tag": "large_breasts", "approved_aliases": ["巨乳", "大胸"]},
    ], ensure_ascii=False), encoding="utf-8")
    config = {"danbooru_term_mappings": ["巨乳=huge_breasts"]}
    store = SimpleNamespace(saved=[])
    store.save_config = lambda *, replace_config: store.saved.append(replace_config)
    handler = SimpleNamespace(
        _plugin_data_path=tmp_path, _noun_catalog_path=lambda: catalog,
        _existing_term_mappings=lambda: parse_existing_mappings(config["danbooru_term_mappings"]),
        _noun_web_job={"state": "idle", "message": ""},
        _noun_translation_lock=asyncio.Lock(), _wardrobe_lock=asyncio.Lock(),
        _config_store=store, config=config,
        _schema_path=Path(__file__).resolve().parents[1] / "_conf_schema.json",
        _services=SimpleNamespace(danbooru_resolver=SimpleNamespace(_config={})),
        _ensure_dashboard_user=lambda: None,
    )
    return handler, store


def test_page_snapshot_shows_verified_alias_and_manual_conflict(tmp_path):
    handler, _ = _handler(tmp_path)
    result = ComfyUIAgentPlugin._noun_translation_snapshot(handler)
    assert (result["total"], result["translated"], result["verified"]) == (2, 2, 1)
    assert result["additions"] == {"大胸": "large_breasts"}
    assert result["skipped_aliases"]["巨乳"] == "与手工映射冲突"
    assert result["rows"][1]["approved_aliases"] is None
    assert result["enabled"] is True
    assert result["manual_review"] == 1
    assert result["existing"] == 0
    assert result["rows"][0]["ready"] is True
    assert result["rows"][0]["needs_manual_review"] is True


def test_page_import_uses_same_safe_plan_and_hot_applies(tmp_path, monkeypatch):
    handler, store = _handler(tmp_path)

    async def payload(*, default):
        return {"mode": "apply"}

    monkeypatch.setattr(main, "request", SimpleNamespace(json=payload))
    monkeypatch.setattr(main, "json_response", lambda value: value)
    result = asyncio.run(ComfyUIAgentPlugin.import_noun_translation(handler))
    assert result == {"imported": 1, "skipped": 1, "existing": 0}
    assert handler.config["danbooru_term_mappings"] == [
        "巨乳=huge_breasts", "大胸=large_breasts",
    ]
    assert handler._services.danbooru_resolver._config[
        "danbooru_term_mappings"
    ] == handler.config["danbooru_term_mappings"]
    assert len(store.saved) == 1
    snapshot = ComfyUIAgentPlugin._noun_translation_snapshot(handler)
    assert snapshot["existing"] == 1
    assert snapshot["manual_review"] == 1
    assert snapshot["importable"] == 0
    assert snapshot["rows"][0]["needs_manual_review"] is True


def test_page_switch_persists_and_hot_applies_without_erasing_mappings(
    tmp_path, monkeypatch,
):
    handler, store = _handler(tmp_path)

    async def payload(*, default):
        return {"enabled": False}

    monkeypatch.setattr(main, "request", SimpleNamespace(json=payload))
    monkeypatch.setattr(main, "json_response", lambda value: value)
    result = asyncio.run(ComfyUIAgentPlugin.set_noun_translation_enabled(handler))
    assert result == {"enabled": False}
    assert handler.config["danbooru_term_mappings_enabled"] is False
    assert handler._services.danbooru_resolver._config[
        "danbooru_term_mappings_enabled"
    ] is False
    assert handler.config["danbooru_term_mappings"] == ["巨乳=huge_breasts"]
    assert store.saved[0]["anima_master_tag_lookup"]["danbooru_term_mappings_enabled"] is False
    assert ComfyUIAgentPlugin._noun_translation_snapshot(handler)["enabled"] is False


def test_snapshot_cache_invalidates_for_mapping_checkpoint_and_catalog(tmp_path):
    handler,_=_handler(tmp_path)
    first=ComfyUIAgentPlugin._noun_translation_snapshot(handler)
    first["rows"].clear()
    cached=ComfyUIAgentPlugin._noun_translation_snapshot(handler)
    assert len(cached["rows"])==2
    handler.config["danbooru_term_mappings"].append("大胸=large_breasts")
    assert ComfyUIAgentPlugin._noun_translation_snapshot(handler)["importable"]==0
    checkpoint=tmp_path/"noun_translation_verified.json"
    checkpoint.write_text(json.dumps([
        {"tag":"large_breasts","approved_aliases":["巨乳","大胸"]},
        {"tag":"dress","approved_aliases":["连衣裙"]},
    ]),encoding="utf-8")
    assert ComfyUIAgentPlugin._noun_translation_snapshot(handler)["importable"]==1
    with handler._noun_catalog_path().open("a",encoding="utf-8") as f:
        f.write("skirt,0,1800\n")
    assert ComfyUIAgentPlugin._noun_translation_snapshot(handler)["total"]==3


def test_light_status_never_reads_translation_catalog(tmp_path,monkeypatch):
    handler,_=_handler(tmp_path)
    handler._noun_catalog_path=lambda: (_ for _ in ()).throw(AssertionError("heavy read"))
    handler._noun_web_job={"state":"running","message":"模型任务进行中"}
    monkeypatch.setattr(main,"json_response",lambda value:value)
    result=asyncio.run(ComfyUIAgentPlugin.get_noun_translation_status(handler))
    assert result["job"]["state"]=="running"
    assert result["enabled"] is True


def test_review_endpoints_require_login_and_never_use_config_save(tmp_path,monkeypatch):
    from vocabulary_review import VocabularyReview
    handler,store=_handler(tmp_path)
    path=tmp_path/"maintenance.json"
    path.write_text(json.dumps({"records":[{"tag":"dress","aliases":[],"focus":True}],"source":{}}),encoding="utf-8")
    handler._vocabulary_review=VocabularyReview(path,tmp_path/"human_notes.json")
    monkeypatch.setattr(main,"json_response",lambda value:value)
    monkeypatch.setattr(main,"error_response",lambda message,status_code:(message,status_code))
    monkeypatch.setattr(main,"request",SimpleNamespace(args={}))
    snapshot=asyncio.run(ComfyUIAgentPlugin.get_vocabulary_review(handler))
    async def body(*,default):
        return {"base_revision":snapshot["revision"],"tag":"dress","state":"deferred","note":"待核对"}
    monkeypatch.setattr(main,"request",SimpleNamespace(json=body))
    result=asyncio.run(ComfyUIAgentPlugin.save_vocabulary_review(handler))
    assert result["applied"] is False
    assert store.saved==[]
    assert handler.config["danbooru_term_mappings"]==["巨乳=huge_breasts"]
    def denied(): raise PermissionError("需要登录")
    handler._ensure_dashboard_user=denied
    assert asyncio.run(ComfyUIAgentPlugin.get_vocabulary_review(handler))[1]==403
    assert asyncio.run(ComfyUIAgentPlugin.save_vocabulary_review(handler))[1]==403
