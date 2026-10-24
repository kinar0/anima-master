from __future__ import annotations

import asyncio
import json
import pytest
from astrbot.api.web import PluginRequest, bind_request_context
from starlette.requests import Request

from main import ComfyUIAgentPlugin
from opus_usage_monitor import OpusUsageMonitor


def response(percent=85):
    return {
        "ok": True,
        "checked_at": "2026-10-22T05:00:00+00:00",
        "subscription": {
            "active": True,
            "tier": 3,
            "percent": percent,
            "is_negative": False,
            "purchased_percent": 0,
            "time_until_next_percent": 600,
            "token": "must-not-persist",
            "email": "must-not-persist",
        },
    }


def test_interval_cache_survives_reload_and_concurrent_reads(tmp_path):
    async def run():
        now = [1800000000]
        calls = []
        config = {
            "comfyui_base_url": "http://comfy:8188",
            "opus_usage_refresh_minutes": 240,
        }

        def fetch(base):
            calls.append(base)
            return response()

        path = tmp_path / "usage.json"
        monitor = OpusUsageMonitor(config, path, fetch=fetch, clock=lambda: now[0])
        results = await asyncio.gather(*(monitor.refresh_if_due() for _ in range(8)))
        assert all(item["subscription"]["percent"] == 85 for item in results)
        assert calls == ["http://comfy:8188"]
        assert "must-not-persist" not in path.read_text(encoding="utf-8")
        restarted = OpusUsageMonitor(config, path, fetch=fetch, clock=lambda: now[0])
        for _ in range(5):
            assert restarted.snapshot()["subscription"]["percent"] == 85
        now[0] += 239 * 60
        await restarted.refresh_if_due()
        assert len(calls) == 1
        now[0] += 60
        assert restarted.snapshot()["stale"] is True
        await restarted.refresh_if_due()
        assert len(calls) == 2
        assert restarted.snapshot()["stale"] is False

    asyncio.run(run())


def test_failed_queries_keep_old_values_and_throttle_retry(tmp_path):
    async def run():
        now = [1800000000]
        monitor = OpusUsageMonitor(
            {},
            tmp_path / "usage.json",
            fetch=lambda base: response(),
            clock=lambda: now[0],
        )
        await monitor.refresh_if_due()
        calls = []

        def fail(_base):
            calls.append(1)
            raise RuntimeError("credential-that-must-not-leak")

        monitor._fetch = fail
        now[0] += 7200
        state = await monitor.refresh_if_due()
        assert state["status"] == "unavailable"
        assert state["stale"] is True
        assert state["subscription"]["percent"] == 85
        assert "credential" not in json.dumps(state)
        assert state["checked_at"].startswith("2026-10-22")
        await monitor.refresh_if_due()
        assert len(calls) == 1

    asyncio.run(run())


def test_initial_failure_is_unknown_and_switching_comfy_drops_old_account(tmp_path):
    async def run():
        config = {"comfyui_base_url": "http://one"}
        monitor = OpusUsageMonitor(
            config, tmp_path / "usage.json", fetch=lambda base: response()
        )
        await monitor.refresh_if_due()
        config["comfyui_base_url"] = "http://two"
        assert monitor.snapshot()["subscription"] is None
        monitor._fetch = lambda base: {"ok": False}
        state = await monitor.refresh_if_due()
        assert state["subscription"] is None
        assert state["status"] == "unavailable"

    asyncio.run(run())


@pytest.mark.parametrize(
    "subscription",
    [
        {"active": True, "tier": 3},
        {"active": True, "tier": 3, "percent": True, "is_negative": False},
        {"active": True, "tier": 3, "percent": -1, "is_negative": False},
        {"active": True, "tier": 3, "percent": 85, "is_negative": "false"},
    ],
)
def test_invalid_percent_never_turns_into_zero(tmp_path, subscription):
    async def run():
        data = response()
        data["subscription"] = subscription
        monitor = OpusUsageMonitor({}, tmp_path / "usage.json", fetch=lambda base: data)
        state = await monitor.refresh_if_due()
        assert state["subscription"] is None
        assert state["status"] == "unavailable"

    asyncio.run(run())


def test_inactive_and_non_opus_are_distinct_from_empty_allowance(tmp_path):
    async def run():
        for i, subscription in enumerate(
            [{"active": False, "tier": 3}, {"active": True, "tier": 2}]
        ):
            data = response()
            data["subscription"] = subscription
            monitor = OpusUsageMonitor(
                {}, tmp_path / f"usage{i}.json", fetch=lambda base: data
            )
            state = await monitor.refresh_if_due()
            assert state["status"] == "ok"
            assert "percent" not in state["subscription"]

    asyncio.run(run())


def test_switch_bounds_and_background_cleanup(tmp_path):
    async def run():
        config = {"opus_usage_monitor_enabled": False}

        def no_network(_base):
            pytest.fail("Disabled monitor must not query")

        monitor = OpusUsageMonitor(config, tmp_path / "usage.json", fetch=no_network)
        assert monitor.interval_minutes == 120
        assert (await monitor.refresh_if_due())["status"] == "disabled"
        config["opus_usage_refresh_minutes"] = 0
        assert monitor.interval_minutes == 5
        config["opus_usage_refresh_minutes"] = 999999
        assert monitor.interval_minutes == 10080
        monitor.start()
        task = monitor._task
        monitor.start()
        assert task is monitor._task
        await asyncio.sleep(0)
        await monitor.close()
        assert task.done()
        assert monitor._task is None

    asyncio.run(run())


def test_dashboard_read_uses_cache_only_and_requires_login(tmp_path):
    async def run():
        monitor = OpusUsageMonitor(
            {},
            tmp_path / "usage.json",
            fetch=lambda base: pytest.fail("GET must read cache only"),
        )
        plugin = object.__new__(ComfyUIAgentPlugin)
        plugin._opus_monitor = monitor
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/opus-usage",
                "headers": [],
                "query_string": b"",
            }
        )
        with bind_request_context(PluginRequest(request, username="admin")):
            result = await plugin.get_opus_usage()
        assert result.status_code == 200
        assert json.loads(result.body)["subscription"] is None
        with bind_request_context(PluginRequest(request)):
            assert (await plugin.get_opus_usage()).status_code == 403

    asyncio.run(run())
