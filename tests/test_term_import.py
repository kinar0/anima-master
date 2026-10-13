import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from main import ComfyUIAgentPlugin
from noun_translation_batch import parse_existing_mappings


def test_import_preview_apply_and_repeat_are_safe(tmp_path):
    (tmp_path / "noun_translation_review.json").write_text(json.dumps([
        {"tag": "large_breasts", "aliases": ["巨乳", "大胸"]},
        {"tag": "dress", "aliases": ["连衣裙"]},
    ], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "noun_translation_verified.json").write_text(json.dumps([
        {"tag": "large_breasts", "approved_aliases": ["巨乳", "大胸"]},
        {"tag": "dress", "approved_aliases": ["连衣裙"]},
    ], ensure_ascii=False), encoding="utf-8")

    class ConfigStore:
        def __init__(self):
            self.saved = []

        def save_config(self, *, replace_config):
            self.saved.append(replace_config)

    class Handler:
        def __init__(self):
            self._wardrobe_lock = asyncio.Lock()
            self._plugin_data_path = tmp_path
            self._config_store = ConfigStore()
            self._schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
            self.config = {"danbooru_term_mappings": ["巨乳=huge_breasts"]}
            self._services = SimpleNamespace(
                danbooru_resolver=SimpleNamespace(_config={})
            )

        def _existing_term_mappings(self):
            return parse_existing_mappings(self.config["danbooru_term_mappings"])

    event = SimpleNamespace(is_admin=lambda: True)
    handler = Handler()
    preview = asyncio.run(ComfyUIAgentPlugin._term_import_text(
        handler, event, "preview"
    ))
    assert "可导入 2" in preview
    assert not handler._config_store.saved

    applied = asyncio.run(ComfyUIAgentPlugin._term_import_text(
        handler, event, "apply"
    ))
    assert "已导入 2" in applied
    assert handler.config["danbooru_term_mappings"] == [
        "巨乳=huge_breasts", "大胸=large_breasts", "连衣裙=dress",
    ]
    assert handler._services.danbooru_resolver._config[
        "danbooru_term_mappings"
    ] == handler.config["danbooru_term_mappings"]
    assert len(handler._config_store.saved) == 1

    again = asyncio.run(ComfyUIAgentPlugin._term_import_text(
        handler, event, "apply"
    ))
    assert "已导入 0" in again
    assert "已有 2 个相同映射" in again
    assert "跳过 1 个冲突或不安全别名" in again
    assert len(handler._config_store.saved) == 1
