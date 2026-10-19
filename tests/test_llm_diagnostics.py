from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_diagnostics import LLMDiagnostics
from prompt_pipeline import PromptPipeline


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)

    def warning(self, *_args):
        pass

    def events(self, kind):
        prefix = f"[comfyui_agent] LLM {kind} "
        return [json.loads(line[len(prefix):].split("\n", 1)[0])
                for line in self.lines if line.startswith(prefix)]


class Context:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def observer(config=None, logger=None):
    config = config or {}
    logger = logger or Logger()
    return LLMDiagnostics(logger, lambda k, d: config.get(k, d)), logger


def response():
    return NS(
        completion_text="final answer",
        reasoning_content="独立推理内容 %s\n第二行",
        usage=NS(input=120, input_cached=80, input_other=40, output=35),
        raw_completion=NS(usage=NS(completion_tokens_details=NS(reasoning_tokens=20)),
                          choices=[NS(finish_reason="length")], secret="RAW_SECRET"),
    )


def test_disabled_logging_does_not_inspect_or_change_response():
    class UnreadableResponse:
        @property
        def usage(self):
            raise AssertionError("Disabled observer must not inspect response")

        @property
        def reasoning_content(self):
            raise AssertionError("Disabled observer must not inspect response")

    expected = UnreadableResponse()
    context = Context(expected)
    diagnostics, logger = observer()
    request = dict(chat_provider_id="provider", prompt="PROMPT_SECRET", max_tokens=50,
                   thinking={"type": "disabled"})
    assert asyncio.run(diagnostics.generate(context, stage="test", **request)) is expected
    assert context.calls == [request]
    assert logger.lines == []


def test_usage_only_is_exact_and_does_not_double_count_reasoning():
    diagnostics, logger = observer({"debug_llm_usage_enabled": True})
    expected = response()
    context = Context(expected)
    run_id = diagnostics.start_run()
    request = dict(chat_provider_id="provider", prompt="PROMPT_SECRET", max_tokens=90,
                   thinking={"type": "disabled"})
    assert asyncio.run(diagnostics.generate(context, stage="writer", attempt="repair", **request)) is expected
    assert context.calls == [request]
    event, = logger.events("usage")
    assert event["run_id"] == run_id
    assert (event["stage"], event["attempt"]) == ("writer", "repair")
    assert [event[k] for k in ("input_tokens", "input_cached_tokens", "input_uncached_tokens",
                               "output_tokens", "reasoning_tokens", "total_tokens")] == [120, 80, 40, 35, 20, 155]
    assert event["finish_reason"] == "length"
    assert event["requested_thinking"] == "disabled"
    assert event["elapsed_ms"] >= 0
    assert not logger.events("reasoning")
    assert not any(value in "\n".join(logger.lines)
                   for value in ("独立推理", "PROMPT_SECRET", "RAW_SECRET", "final answer"))


@pytest.mark.parametrize("value", [None, "", "  "])
def test_reasoning_only_reports_unavailable_without_inventing_text(value):
    diagnostics, logger = observer({"debug_llm_reasoning_enabled": True})
    asyncio.run(diagnostics.generate(Context(NS(reasoning_content=value, completion_text="final answer")), stage="test"))
    assert logger.events("reasoning")[0]["reasoning_status"] == "not_returned"
    assert "未返回推理内容" in logger.lines[0]
    assert "final answer" not in logger.lines[0]
    assert not logger.events("usage")


def test_both_switches_preserve_full_reasoning_and_share_call_id():
    diagnostics, logger = observer({"debug_llm_reasoning_enabled": True, "debug_llm_usage_enabled": True})
    expected = response()
    asyncio.run(diagnostics.generate(Context(expected), stage="test"))
    assert logger.events("reasoning")[0]["call_id"] == logger.events("usage")[0]["call_id"]
    assert expected.reasoning_content in logger.lines[-1]
    assert logger.events("reasoning")[0]["reasoning_status"] == "returned"


def test_unavailable_usage_stays_null_and_real_zero_is_preserved():
    diagnostics, logger = observer({"debug_llm_usage_enabled": True})
    asyncio.run(diagnostics.generate(Context(NS()), stage="missing"))
    event, = logger.events("usage")
    assert event["usage_available"] is False
    assert all(event[k] is None for k in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"))
    raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10,
                     "prompt_tokens_details": {"cached_tokens": 0},
                     "completion_tokens_details": {"reasoning_tokens": 0}},
           "choices": [{"finish_reason": "stop"}]}
    asyncio.run(diagnostics.generate(Context({"raw_completion": raw}), stage="zero"))
    event = logger.events("usage")[-1]
    assert event["output_tokens"] == event["reasoning_tokens"] == event["input_cached_tokens"] == 0
    assert event["input_uncached_tokens"] is None
    assert event["total_tokens"] == 10


def test_raw_provider_cache_counts_override_lossy_adapter_defaults():
    diagnostics, logger = observer({"debug_llm_usage_enabled": True})
    reply = NS(usage=NS(input=120, input_cached=0, input_other=120, output=35),
               raw_completion={"usage": {"prompt_tokens": 120, "completion_tokens": 35,
                                         "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 40}})
    asyncio.run(diagnostics.generate(Context(reply), stage="test"))
    event, = logger.events("usage")
    assert event["input_cached_tokens"] == 80
    assert event["input_uncached_tokens"] == 40
    assert event["total_tokens"] == 155


def test_provider_omitted_usage_is_not_reported_as_adapter_zero():
    diagnostics, logger = observer({"debug_llm_usage_enabled": True})
    reply = NS(usage=NS(input=0, input_cached=0, input_other=0, output=0),
               raw_completion={"usage": None, "choices": []})
    asyncio.run(diagnostics.generate(Context(reply), stage="test"))
    event, = logger.events("usage")
    assert event["usage_available"] is False
    assert event["total_tokens"] is None


@pytest.mark.parametrize("error", [RuntimeError("API_KEY_SECRET"), asyncio.CancelledError("API_KEY_SECRET")])
def test_exceptions_and_cancellation_propagate_without_sensitive_error_text(error):
    diagnostics, logger = observer({"debug_llm_usage_enabled": True})
    context = Context(error)
    with pytest.raises(type(error)) as caught:
        asyncio.run(diagnostics.generate(context, stage="writer"))
    assert caught.value is error
    assert len(context.calls) == 1
    event, = logger.events("usage")
    assert event["status"] == "error"
    assert event["error_type"] == type(error).__name__
    assert "API_KEY_SECRET" not in logger.lines[0]


def test_broken_logging_cannot_fail_generation():
    class BrokenLogger:
        def info(self, *_args):
            raise OSError("full log disk")

    diagnostics, _ = observer({"debug_llm_reasoning_enabled": True, "debug_llm_usage_enabled": True}, BrokenLogger())
    expected = response()
    assert asyncio.run(diagnostics.generate(Context(expected), stage="test")) is expected


def test_concurrent_runs_keep_separate_ids_and_share_id_within_each_run():
    diagnostics, logger = observer({"debug_llm_usage_enabled": True})

    async def run(stage):
        run_id = diagnostics.start_run()
        for _ in range(2):
            await asyncio.sleep(0)
            await diagnostics.generate(Context(NS()), stage=stage)
        return run_id

    async def both():
        return await asyncio.gather(run("a"), run("b"))

    ids = asyncio.run(both())
    assert ids[0] != ids[1]
    events = logger.events("usage")
    assert len({event["call_id"] for event in events}) == 4
    for stage, run_id in zip(("a", "b"), ids):
        assert {event["run_id"] for event in events if event["stage"] == stage} == {run_id}


def pipeline(config, context, logger):
    return PromptPipeline(
        context=context, config=config, logger=logger, danbooru_resolver=object(), researcher=object(),
        get_bool=lambda k, d: config.get(k, d), get_int=lambda k, d: config.get(k, d),
        get_float=lambda k, d: config.get(k, d), get_str=lambda k, d: config.get(k, d),
        shorten=lambda text, limit=600: text[:limit],
    )


def test_all_shared_prompt_llm_stages_observe_same_requests_and_outputs():
    async def run(config):
        context, logger = Context(response()), Logger()
        builder = pipeline(config, context, logger)
        outputs = [
            await builder._generate_semantic_plan_with_llm(provider_id="provider", user_prompt="test"),
            await builder._repair_semantic_plan_with_llm(provider_id="provider", user_prompt="test", previous_raw="{}", issues=("test",)),
            await builder._generate_prompt_tags_with_llm(provider_id="provider", llm_prompt="test", use_deep_thinking=True, fixed_character=False),
            await builder._generate_outfit_summary_with_llm(provider_id="provider", summary_prompt="test", use_deep_thinking=False),
            await builder._generate_character_candidates_with_llm(provider_id="provider", user_prompt="test", rejected_content="test"),
            await builder._generate_constraint_plan_with_llm(provider_id="provider", plan_prompt="test"),
            await builder._generate_multi_person_plan_with_llm(provider_id="provider", plan_prompt="test", use_deep_thinking=False),
        ]
        return context.calls, outputs, logger

    before, before_outputs, _ = asyncio.run(run({}))
    after, after_outputs, logger = asyncio.run(run({"debug_llm_usage_enabled": True}))
    assert after == before
    assert after_outputs == before_outputs
    assert [event["stage"] for event in logger.events("usage")] == [
        "semantic_planner", "semantic_planner", "prompt_writer", "outfit_summary",
        "character_candidates", "constraint_planner", "multi_person_planner",
    ]
    assert logger.events("usage")[1]["attempt"] == "repair"


def test_nai_full_pipeline_is_unchanged_with_diagnostics(monkeypatch):
    import test_prompt_direct_writer as direct

    logger = Logger()
    monkeypatch.setattr(direct, "Logger", lambda: logger)
    prompt = "-r 角色甲和角色乙站在花园"
    semantic = {"characters": [direct.semantic_character("角色甲"), direct.semantic_character("角色乙")]}
    writer = direct.plan([("character_a", "girl, pink hair, standing", .3),
                          ("character_b", "girl, blue hair, standing", .7)])
    before, before_calls = direct.run(prompt, semantic, writer)
    after, after_calls = direct.run(prompt, semantic, writer,
                                    {"debug_llm_usage_enabled": True, "debug_llm_reasoning_enabled": True})
    assert after.final_prompt == before.final_prompt
    assert after.final_prompt
    assert after_calls == before_calls
    events = logger.events("usage")
    assert [event["stage"] for event in events] == ["semantic_planner", "nai_direct_writer"]
    assert {event["run_id"] for event in events} == {after.summary["llm_diagnostics_run_id"]}
    assert "llm_diagnostics_run_id" not in before.summary
    assert "reasoning_content" not in after.summary


def test_verifier_and_reverse_image_log_without_changing_requests():
    from generation_verifier import GenerationVerifier
    from reference_context import ReferenceContextBuilder

    async def run(config):
        context, logger = Context(response()), Logger()
        context.get_config = lambda **_: {"provider_settings": {"default_provider_id": "provider"}}
        get_bool = lambda k, d: config.get(k, d)
        verifier = GenerationVerifier(
            context=context, task_recorder=None, generate_payload=None, logger=logger,
            get_bool=get_bool, get_int=lambda k, d: d, get_str=lambda k, d: d,
        )
        verify = verifier._make_verify_llm_call("provider")
        assert await verify("verify") == "final answer"
        assert await verify("verify") == "final answer"
        reverse = ReferenceContextBuilder(
            context=context, run_prompt_tool=None, event_image_input=None,
            logger=logger, get_int=lambda k, d: d, shorten=lambda t, n: t[:n], get_bool=get_bool,
        )
        assert await reverse.reverse_image_tags(NS(unified_msg_origin="session"), "image.png") == "final answer"
        return context.calls, logger

    before, _ = asyncio.run(run({}))
    after, logger = asyncio.run(run({"debug_llm_usage_enabled": True}))
    assert after == before
    assert [event["stage"] for event in logger.events("usage")] == ["image_verifier", "image_verifier", "image_reverse"]
    assert logger.events("usage")[1]["attempt"] == "retry_1"
