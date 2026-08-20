from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = PLUGIN_DIR / "agent_tools"
for directory in (PLUGIN_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from comfyui_startup import ComfyUIStartupManager
from comfyui_status import build_status_payload


class _Logger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def info(self, *_args):
        pass


class _Event:
    def get_sender_id(self):
        return "sender"

    def is_admin(self):
        return False


def _ready_status() -> dict:
    return {
        "ok": True,
        "unet_available": True,
        "clip_available": True,
        "vae_available": True,
    }


def _timeout_status() -> dict:
    return {
        "ok": False,
        "connection_issue": "api_read_timeout",
        "connection_hint": "ComfyUI API 响应过慢。可能正在启动、卡住或负载过高。",
    }


def test_recent_validated_status_survives_transient_api_timeout():
    statuses = iter([_ready_status(), _timeout_status(), _timeout_status()])
    logger = _Logger()

    async def run_status():
        return next(statuses)

    manager = ComfyUIStartupManager(
        root=Path("."),
        config={"readiness_retry_delay_seconds": 0, "readiness_cache_seconds": 300},
        logger=logger,
        get_bool=lambda _key, default: default,
        get_int=lambda key, default: {
            "readiness_retry_delay_seconds": 0,
            "readiness_cache_seconds": 300,
        }.get(key, default),
        get_str=lambda _key, default="": default,
        run_status=run_status,
    )

    assert asyncio.run(manager.ensure_ready(_Event()))["ok"] is True
    result = asyncio.run(manager.ensure_ready(_Event()))

    assert result["ok"] is True
    assert result["status"]["readiness_source"] == "recent_validated_cache"
    assert logger.warnings


def test_status_retains_reachable_flag_when_object_info_times_out(monkeypatch):
    import requests
    import comfyui_status as status_module

    class _Client:
        def __init__(self, _config):
            pass

        def get_json(self, path, timeout):
            if path == "/system_stats":
                return {"system": {}, "devices": []}
            raise requests.exceptions.ReadTimeout("object info busy")

    monkeypatch.setattr(status_module, "ComfyUIHttpClient", _Client)

    payload = build_status_payload({}, [])

    assert payload["ok"] is False
    assert payload["comfyui_api_reachable"] is True
    assert payload["connection_issue"] == "api_read_timeout"
