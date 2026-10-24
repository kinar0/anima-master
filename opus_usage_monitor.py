"""Slow, independent polling of sanitized Opus usage through ComfyUI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests


def sanitize_subscription(value):
    if not isinstance(value, dict) or type(value.get("active")) is not bool:
        raise ValueError("Invalid subscription state")
    if type(value.get("tier")) is not int:
        raise ValueError("Invalid subscription tier")
    result = {"active": value["active"], "tier": value["tier"]}
    if value["active"] and value["tier"] == 3:
        if (
            type(value.get("percent")) is not int
            or value["percent"] < 0
            or type(value.get("is_negative")) is not bool
        ):
            raise ValueError("Invalid Opus usage")
        result.update(percent=value["percent"], is_negative=value["is_negative"])
        for key in ("purchased_percent", "time_until_next_percent"):
            if type(value.get(key)) is int and value[key] >= 0:
                result[key] = value[key]
    return result


def fetch_usage(base_url):
    with requests.get(base_url + "/rs_nai/opus_usage", timeout=(10, 45)) as response:
        if response.status_code == 404:
            raise LookupError("ComfyUI NAI extension needs an update")
        response.raise_for_status()
        return response.json()


class OpusUsageMonitor:
    def __init__(self, config, path: Path, *, fetch=fetch_usage, clock=time.time):
        self.config = config
        self.path = Path(path)
        self._fetch = fetch
        self._clock = clock
        self._lock = asyncio.Lock()
        self._task = None
        self._state = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._state = data
        except (OSError, ValueError):
            pass

    @property
    def interval_minutes(self):
        try:
            return max(
                5, min(10080, int(self.config.get("opus_usage_refresh_minutes", 120)))
            )
        except (TypeError, ValueError):
            return 120

    def _source(self):
        base = (
            str(self.config.get("comfyui_base_url") or "http://127.0.0.1:8188")
            .strip()
            .rstrip("/")
        )
        return base, hashlib.sha256(base.encode()).hexdigest()

    def snapshot(self):
        _, source = self._source()
        state = self._state if self._state.get("source") == source else {}
        last_attempt = state.get("last_attempt")
        if type(last_attempt) not in (int, float) or not math.isfinite(last_attempt):
            last_attempt = None
        enabled = bool(self.config.get("opus_usage_monitor_enabled", True))
        result = {
            "enabled": enabled,
            "interval_minutes": self.interval_minutes,
            "status": state.get("status", "waiting") if enabled else "disabled",
            "checked_at": state.get("checked_at"),
            "next_update_at": (
                datetime.fromtimestamp(
                    last_attempt + self.interval_minutes * 60, timezone.utc
                ).isoformat()
                if enabled and type(last_attempt) in (int, float)
                else None
            ),
            "stale": (
                not enabled
                or state.get("status") != "ok"
                or last_attempt is None
                or self._clock() - last_attempt >= self.interval_minutes * 60
            ),
            "message": state.get("message", ""),
            "subscription": None,
        }
        try:
            result["subscription"] = sanitize_subscription(state.get("subscription"))
        except ValueError:
            pass
        return result

    async def refresh_if_due(self):
        async with self._lock:
            if not self.config.get("opus_usage_monitor_enabled", True):
                return self.snapshot()
            base, source = self._source()
            if self._state.get("source") != source:
                self._state = {"source": source}
            now = self._clock()
            last = self._state.get("last_attempt")
            if (
                type(last) in (int, float)
                and 0 <= now - last < self.interval_minutes * 60
            ):
                return self.snapshot()
            self._state["last_attempt"] = now
            try:
                payload = await asyncio.to_thread(self._fetch, base)
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise ValueError("Usage unavailable")
                subscription = sanitize_subscription(payload.get("subscription"))
                checked_at = payload.get("checked_at")
                if not isinstance(checked_at, str):
                    raise ValueError("Missing timestamp")
                checked_at = (
                    datetime.fromisoformat(checked_at)
                    .astimezone(timezone.utc)
                    .isoformat()
                )
                self._state.update(
                    subscription=subscription,
                    checked_at=checked_at,
                    status="ok",
                    message="",
                )
            except Exception as exc:
                self._state.update(
                    status="unavailable",
                    message=(
                        "请更新 ComfyUI NAI 扩展并重启 ComfyUI。"
                        if isinstance(exc, LookupError)
                        else "额度查询失败；保留上次成功值，等待下次更新。"
                    ),
                )
            self._save()
            return self.snapshot()

    def _save(self):
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp.write_text(
                json.dumps(self._state, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(temp, self.path)
        except OSError:
            pass
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            await self.refresh_if_due()
            await asyncio.sleep(60)

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
