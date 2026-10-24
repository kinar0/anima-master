"""Read-only configuration/task audit and offline installed-adapter wire capture.

All HTTP requests use httpx.MockTransport and a dummy key. No real API call.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT.parents[2]
sys.path.insert(0, str(CORE))

import httpx
from openai import AsyncOpenAI
from astrbot.core.star.context import Context
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial


async def capture(custom_body, requested):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-audit", "object": "chat.completion", "created": 0,
            "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    client = AsyncOpenAI(api_key="offline-dummy-key", base_url="https://api.deepseek.com/v1",
                         max_retries=0,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    provider = object.__new__(ProviderOpenAIOfficial)
    provider.client = client
    provider.api_keys = ["offline-dummy-key"]
    provider.provider_config = {"id": "offline", "type": "openai_chat_completion",
                                "provider": "deepseek", "custom_extra_body": custom_body}
    provider.provider_settings = {}
    provider.model_name = "deepseek-v4-flash"
    provider.reasoning_key = "reasoning_content"
    provider.default_params = inspect.signature(client.chat.completions.create).parameters.keys()

    async def get_provider(_):
        return provider

    context = object.__new__(Context)
    context.provider_manager = SimpleNamespace(get_provider_by_id=get_provider)
    try:
        await context.llm_generate(chat_provider_id="offline", prompt="offline request",
                                   system_prompt="offline system", **requested)
    finally:
        await client.close()
    assert len(captured) == 1
    return {"requested": requested, "provider_custom_extra_body": custom_body,
            "wire_body": captured[0]}


def task_stats():
    days = defaultdict(Counter)
    lengths = defaultdict(list)
    for path in (CORE / "data/plugin_data/astrbot_plugin_anima_master/tasks").glob("*.json"):
        try:
            task = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            continue
        day = str(task.get("time", ""))[:10]
        if day < "2026-10-18":
            continue
        summary = task.get("prompt", {}).get("summary", {})
        if not isinstance(summary, dict):
            continue
        row = days[day]
        row["tasks"] += 1
        row["successful"] += bool(task.get("ok"))
        row["nai_r_tasks"] += bool(summary.get("nai_r_mode"))
        row["deep_thinking_requested"] += bool(summary.get("deep_thinking"))
        for field in ("semantic_plan_attempt_count", "prompt_llm_attempt_count", "nai_character_plan_attempt_count"):
            value = summary.get(field)
            if isinstance(value, int):
                row[field + "_total"] += value
                row[field + "_recorded_tasks"] += 1
                row[field + "_retried_tasks"] += value > 1
        for field in ("semantic_plan_initial_raw", "semantic_plan_raw", "llm_prompt",
                      "prompt_llm_initial_content", "nai_character_plan_prompt", "nai_character_plan_initial_raw"):
            value = summary.get(field)
            if isinstance(value, str) and value:
                lengths[(day, field)].append(len(value))
    return {"daily": dict(sorted(days.items())), "text_chars_not_tokens": {
        f"{day}/{field}": {"samples": len(values), "mean": round(sum(values) / len(values)),
                            "min": min(values), "max": max(values)}
        for (day, field), values in sorted(lengths.items())}}


async def main():
    config = json.loads((CORE / "data/config/astrbot_plugin_anima_master_config.json").read_text(encoding="utf-8-sig"))
    flat = {key: value for section in config.values() if isinstance(section, dict)
            for key, value in section.items()}
    selected = {key: flat.get(key) for key in (
        "prompt_builder_provider_id", "prompt_builder_max_tokens", "prompt_builder_deep_thinking_enabled",
        "prompt_builder_reasoning_effort", "debug_llm_usage_enabled", "debug_llm_reasoning_enabled")}
    cases = {
        "per_call_disabled": await capture({}, {"max_tokens": 900, "thinking": {"type": "disabled"}}),
        "per_call_enabled": await capture({}, {"max_tokens": 5000, "thinking": {"type": "enabled"},
                                               "reasoning_effort": "high"}),
        "per_call_extra_body": await capture({}, {"extra_body": {"max_tokens": 900,
                                                                                "thinking": {"type": "disabled"}}}),
        "provider_custom_body_control": await capture({"max_tokens": 900, "thinking": {"type": "disabled"}}, {}),
    }
    for name in ("per_call_disabled", "per_call_enabled", "per_call_extra_body"):
        assert not any(key in cases[name]["wire_body"] for key in ("max_tokens", "thinking", "reasoning_effort"))
    assert cases["provider_custom_body_control"]["wire_body"]["thinking"] == {"type": "disabled"}
    assert cases["provider_custom_body_control"]["wire_body"]["max_tokens"] == 900
    report = {"network": "MockTransport only; no API calls or real credentials", "config": selected,
              "wire_capture": cases, "task_statistics": task_stats()}
    (Path(__file__).parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
