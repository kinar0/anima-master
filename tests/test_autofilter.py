from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autofilter_settings import AutofilterSettings  # noqa: E402
from autofilter_workflow import api_workflow  # noqa: E402
import comfyui_runtime as runtime_module  # noqa: E402
from comfyui_runtime import ComfyUIRuntime  # noqa: E402


def test_autofilter_workflow_preserves_widgets_and_replaces_only_image() -> None:
    graph = json.loads((ROOT / "autofilter2.json").read_text(encoding="utf-8"))
    api = api_workflow(graph, "AstrBot/new.png")
    assert api["1"]["inputs"]["image"] == "AstrBot/new.png"
    assert api["14"]["class_type"] == "GLSLShader"
    assert api["14"]["inputs"]["images.image0"] == ["1", 0]
    assert api["14"]["inputs"]["floats.u_float0"] == 20.0
    assert api["14"]["inputs"]["ints.u_int0"] == 0
    assert api["9"]["inputs"]["source"] == ["14", 0]
    assert api["2"]["inputs"]["images"] == ["9", 0]
    assert api["2"]["inputs"]["filename_prefix"].startswith("AstrBot/autofilter_")
    assert ":" not in api["2"]["inputs"]["filename_prefix"]
    assert graph["nodes"][0]["widgets_values_named"]["filename_prefix"].startswith("%date:")


def test_wardrobe_remains_the_plugin_page_entry() -> None:
    pages = sorted(path.parent.name for path in (ROOT / "pages").glob("*/index.html"))
    assert pages == ["wardrobe"]
    html = (ROOT / "pages" / "wardrobe" / "index.html").read_text(encoding="utf-8")
    assert "角色与通用服装词库" in html
    assert 'id="autofilter-settings"' in html
    assert '<script type="module" src="./autofilter.js"></script>' in html


def test_session_switches_are_independent_and_default_off(tmp_path: Path) -> None:
    store = AutofilterSettings(tmp_path / "settings.json")
    assert not store.enabled("qq:group:1")
    store.observe("qq:group:1")
    store.observe("qq:private:1")
    store.set_enabled("qq:group:1", True)
    reloaded = AutofilterSettings(store.path)
    assert reloaded.enabled("qq:group:1")
    assert not reloaded.enabled("qq:private:1")


class _Logger:
    def info(self, *_args): pass
    def warning(self, *_args): pass
    def exception(self, *_args): pass


class _Event:
    unified_msg_origin = "qq:group:1"
    def __init__(self): self.sent = []
    def chain_result(self, chain): return chain
    def plain_result(self, message): return message
    async def send(self, message): self.sent.append(message)
    def get_group_id(self): return "1"
    def get_sender_id(self): return "2"
    def get_platform_name(self): return "qq"


def _runtime(store: AutofilterSettings) -> ComfyUIRuntime:
    return ComfyUIRuntime(
        root=ROOT, tool=ROOT / "unused", prompt_tool=ROOT / "unused", python=ROOT / "unused",
        config={}, logger=_Logger(), get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default, get_str=lambda _key, default: default,
        autofilter_settings=store,
    )


def test_enabled_session_sends_filtered_file_and_failure_never_sends_original(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path / "original.png"; original.write_bytes(b"original")
    filtered = tmp_path / "filtered.png"; filtered.write_bytes(b"filtered")
    store = AutofilterSettings(tmp_path / "settings.json")
    store.set_enabled(_Event.unified_msg_origin, True)
    runtime = _runtime(store)
    assert runtime.autofilter_workflow.name == "autofilter2.json"
    monkeypatch.setattr(runtime_module, "filter_image", lambda *_args, **_kwargs: filtered)
    event = _Event()
    payload = {"ok": True, "outputs": [str(original)]}
    asyncio.run(runtime.send_payload(event, payload))
    assert payload["delivery"]["status"] == "sent"
    assert Path(event.sent[0][0].path) == filtered

    def fail(*_args, **_kwargs): raise RuntimeError("workflow failed")
    monkeypatch.setattr(runtime_module, "filter_image", fail)
    event = _Event()
    payload = {"ok": True, "outputs": [str(original)]}
    asyncio.run(runtime.send_payload(event, payload))
    assert payload["delivery"]["status"] == "send_failed"
    assert len(event.sent) == 1 and isinstance(event.sent[0], str)


def test_disabled_session_sends_original_without_running_filter(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path / "original.png"; original.write_bytes(b"original")
    store = AutofilterSettings(tmp_path / "settings.json")
    runtime = _runtime(store)
    monkeypatch.setattr(runtime_module, "filter_image", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
    event = _Event()
    asyncio.run(runtime.send_payload(event, {"ok": True, "outputs": [str(original)]}))
    assert Path(event.sent[0][0].path) == original
