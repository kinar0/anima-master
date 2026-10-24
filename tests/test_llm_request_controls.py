"""Exercise the installed host and SDK through an offline HTTP transport."""

from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from types import SimpleNamespace as NS

import httpx
import pytest
from openai import AsyncOpenAI

from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.star.context import Context
from llm_request import generate_with_request_controls


def host(handler, custom_body=None):
    client = AsyncOpenAI(
        api_key="offline-dummy-key", base_url="https://api.deepseek.com/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = object.__new__(ProviderOpenAIOfficial)
    provider.client = client
    provider.api_keys = ["offline-dummy-key"]
    provider.provider_config = {
        "id": "offline", "provider": "deepseek", "type": "openai_chat_completion",
        "custom_extra_body": deepcopy(custom_body or {}),
    }
    provider.provider_settings = {}
    provider.model_name = "deepseek-v4-flash"
    provider.reasoning_key = "reasoning_content"
    provider.default_params = inspect.signature(client.chat.completions.create).parameters.keys()

    async def lookup(_):
        return provider

    context = object.__new__(Context)
    context.provider_manager = NS(get_provider_by_id=lookup)
    return context, provider, client


def completion(content="{}"):
    return httpx.Response(200, json={
        "id": "offline", "object": "chat.completion", "created": 0,
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
    })


@pytest.mark.parametrize("usage_logging", [False, True])
def test_real_host_serializes_stage_limits_with_diagnostics_on_or_off(usage_logging):
    from test_llm_diagnostics import Logger, pipeline

    async def run():
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            return completion()

        context, provider, client = host(handler)
        logger = Logger()
        builder = pipeline({"prompt_builder_max_tokens": 5000,
                            "debug_llm_usage_enabled": usage_logging}, context, logger)
        try:
            await builder._generate_semantic_plan_with_llm(provider_id="offline", user_prompt="角色甲站立")
            await builder._repair_semantic_plan_with_llm(
                provider_id="offline", user_prompt="角色甲站立", previous_raw="{}", issues=("missing",))
            await builder._generate_prompt_tags_with_llm(
                provider_id="offline", llm_prompt="ordinary", use_deep_thinking=False, fixed_character=False)
            await builder._generate_prompt_tags_with_llm(
                provider_id="offline", llm_prompt="deep", use_deep_thinking=True, fixed_character=False)
            await builder._generate_semantic_plan_with_llm(
                provider_id="offline", user_prompt="多人", max_tokens=7200)
        finally:
            await client.close()
        assert [body["max_tokens"] for body in captured] == [900, 900, 5000, 5000, 7200]
        assert [body["thinking"]["type"] for body in captured] == ["disabled"] * 3 + ["enabled", "disabled"]
        assert captured[3]["reasoning_effort"] == "high"
        assert all("reasoning_effort" not in body for body in captured[:3] + captured[4:])
        assert provider.provider_config["custom_extra_body"] == {}
        if usage_logging:
            events = logger.events("usage")
            assert len(events) == 5
            assert events[1]["attempt"] == "repair"
            assert events[0]["output_tokens"] == 20
            assert events[0]["reasoning_tokens"] == 0

    asyncio.run(run())


def test_concurrent_requests_override_only_their_own_config_and_preserve_content():
    async def run():
        captured = []

        async def handler(request):
            await asyncio.sleep(0)
            captured.append(json.loads(request.content))
            return completion()

        original = {"thinking": {"type": "enabled"}, "max_tokens": 99,
                    "reasoning_effort": "max", "stop": ["END"]}
        context, provider, client = host(handler, original)
        before = deepcopy(provider.provider_config)
        try:
            await asyncio.gather(*(
                generate_with_request_controls(
                    context, chat_provider_id="offline", prompt=label, system_prompt="system",
                    contexts=[{"role": "user", "content": "prior"}],
                    thinking={"type": state}, max_tokens=limit, **extra)
                for label, state, limit, extra in (
                    ("short", "disabled", 900, {}),
                    ("long", "enabled", 7200, {"reasoning_effort": "low"}),
                )
            ))
        finally:
            await client.close()
        short, long = sorted(captured, key=lambda body: body["max_tokens"])
        assert short["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in short
        assert (long["thinking"], long["reasoning_effort"]) == ({"type": "enabled"}, "low")
        assert long["max_tokens"] == 7200
        assert all(body["stop"] == ["END"] for body in captured)
        assert short["messages"] == [
            {"role": "system", "content": "system"}, {"role": "user", "content": "prior"},
            {"role": "user", "content": "short"},
        ]
        assert provider.provider_config == before
        assert provider.provider_config["custom_extra_body"] == original

    asyncio.run(run())


def test_nai_pipeline_keeps_two_calls_and_serializes_both_stage_budgets():
    from test_llm_diagnostics import Logger, pipeline
    from test_prompt_direct_writer import Resolver, plan, semantic_character

    async def run():
        outputs = [json.dumps({"characters": [semantic_character("角色甲"), semantic_character("角色乙")]}),
                   plan([("character_a", "girl, pink hair, standing", .3),
                         ("character_b", "girl, blue hair, standing", .7)])]
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            return completion(outputs.pop(0))

        context, provider, client = host(handler)
        async def current(_):
            return "offline"
        context.get_current_chat_provider_id = current
        builder = pipeline({"prompt_builder_max_tokens": 5000}, context, Logger())
        builder._danbooru_resolver = Resolver()
        builder._researcher = NS(plan=lambda _: NS(use_web_search=False, use_deep_thinking=False,
                                                 search_reason="", thinking_reason=""))
        try:
            result = await builder.build(NS(unified_msg_origin="session"), "-r 角色甲和角色乙站在花园")
        finally:
            await client.close()
        assert result.final_prompt, result.summary
        assert len(captured) == 2
        assert not outputs
        assert all(body["thinking"] == {"type": "disabled"} for body in captured)
        assert captured[0]["max_tokens"] == 7200
        assert captured[1]["max_tokens"] == 6920
        assert result.summary["prompt_output_mode"] == "nai_instances"
        assert provider.provider_config["custom_extra_body"] == {}

    asyncio.run(run())


def test_other_adapters_and_calls_without_controls_keep_context_path():
    async def run():
        expected = object()
        calls = []

        async def lookup(_):
            return NS()  # Not an OpenAI-compatible provider.

        async def generate(**kwargs):
            calls.append(kwargs)
            return expected

        context = NS(provider_manager=NS(get_provider_by_id=lookup), llm_generate=generate)
        request = dict(chat_provider_id="other", prompt="unchanged", max_tokens=123,
                       thinking={"type": "disabled"})
        assert await generate_with_request_controls(context, **request) is expected
        assert calls == [request]
        assert await generate_with_request_controls(context, chat_provider_id="other", prompt="plain") is expected
        assert calls[-1] == {"chat_provider_id": "other", "prompt": "plain"}

    asyncio.run(run())


def test_cancellation_does_not_change_shared_config_or_fallback_to_uncontrolled_request():
    async def run():
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            raise asyncio.CancelledError()

        context, provider, client = host(handler, {"thinking": {"type": "enabled"}})
        before = deepcopy(provider.provider_config)
        try:
            with pytest.raises(asyncio.CancelledError):
                await generate_with_request_controls(
                    context, chat_provider_id="offline", prompt="cancel", max_tokens=900,
                    thinking={"type": "disabled"})
        finally:
            await client.close()
        assert len(captured) == 1
        assert captured[0]["thinking"] == {"type": "disabled"}
        assert provider.provider_config == before

    asyncio.run(run())
