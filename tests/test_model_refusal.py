from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from comfyui_runtime import ComfyUIRuntime  # noqa: E402
from danbooru_resolver import DanbooruResolveOutcome  # noqa: E402
from generation_task import GenerationTaskRunner  # noqa: E402
from prompt_pipeline import (  # noqa: E402
    PromptPipeline,
    _extract_completion_text,
    is_chinese_model_refusal,
)


class _Logger:
    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass


class _Response:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class _Plan:
    use_web_search = False
    use_deep_thinking = False
    search_reason = ""
    thinking_reason = ""


class _Researcher:
    def plan(self, _prompt):
        return _Plan()


class _Resolver:
    def required_core_tags_for_prompt(self, _prompt):
        return ()

    async def resolve_detailed(self, **_kwargs):
        raise AssertionError("refusal must stop before Danbooru resolution")


class _Event:
    unified_msg_origin = "session"


def _refusing_pipeline(*, multi_person: bool = False):
    class _Context:
        def __init__(self) -> None:
            self.calls = 0

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **_kwargs):
            self.calls += 1
            return _Response("抱歉，我不能满足你的要求，也无法帮助生成该内容。")

    context = _Context()
    config = {"prompt_optimize_enabled": True}
    pipeline = PromptPipeline(
        context=context,
        config=config,
        logger=_Logger(),
        danbooru_resolver=_Resolver(),
        researcher=_Researcher(),
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda key, default: int(config.get(key, default)),
        get_float=lambda key, default: float(config.get(key, default)),
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=lambda text, limit: text[:limit],
    )
    result = asyncio.run(
        pipeline.build(_Event(), "画一个女孩", multi_person=multi_person)
    )
    return context, result


def test_chinese_refusal_detection_requires_refusal_language() -> None:
    assert is_chinese_model_refusal("抱歉，我不能满足你的要求。") is True
    assert is_chinese_model_refusal("我无法协助生成这一内容。") is True
    assert is_chinese_model_refusal("画一个穿蓝色连衣裙的女孩") is False
    assert is_chinese_model_refusal("1girl, solo, blue dress") is False


def test_normal_prompt_refusal_stops_without_structured_retry() -> None:
    context, result = _refusing_pipeline()

    assert context.calls == 1
    assert result.final_prompt == ""
    assert result.summary["model_refused_generation"] is True
    assert result.summary["skipped_reason"] == "model_refused_generation"


def test_multi_person_planner_refusal_is_not_reported_as_plan_failure() -> None:
    context, result = _refusing_pipeline(multi_person=True)

    assert context.calls == 1
    assert result.final_prompt == ""
    assert result.summary["model_refused_generation"] is True
    assert result.summary["multi_person_plan_failed"] is False


def test_generation_task_does_not_call_comfyui_after_model_refusal() -> None:
    class _Recorder:
        def build_generation_start(self, **_kwargs):
            return {"task_id": "task"}

        def mark_prompt_built(self, task, summary) -> None:
            task["prompt_summary"] = summary

        def mark_failure(self, task, error) -> None:
            task["error"] = error

        def write(self, task) -> None:
            self.task = dict(task)

    class _Inputs:
        last_summary = {}

    class _Reference:
        last_summary = {}

    async def run_tool(_args):
        raise AssertionError("ComfyUI must not be called after model refusal")

    runner = GenerationTaskRunner(
        task_recorder=_Recorder(),
        image_inputs=_Inputs(),
        reference_context=_Reference(),
        is_allowed=lambda _event: True,
        ensure_ready=lambda _event: asyncio.sleep(0, result={"ok": True}),
        wants_reference_image=lambda _prompt: False,
        augment_reference_image=lambda _event, prompt: asyncio.sleep(
            0, result=prompt
        ),
        augment_quoted_spell=lambda _event, prompt: prompt,
        build_prompt=lambda *_args, **_kwargs: asyncio.sleep(0, result=""),
        prompt_summary=lambda: {"model_refused_generation": True},
        run_tool=run_tool,
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=lambda text, limit: text[:limit],
    )

    payload = asyncio.run(runner.generate_payload(_Event(), "画一个女孩"))

    assert payload == {
        "ok": False,
        "error": "model_refused_generation",
        "task_id": "task",
    }


def test_delivery_sends_exact_model_refusal_message() -> None:
    class _SendEvent:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.sent.append(result)

    runtime = ComfyUIRuntime.__new__(ComfyUIRuntime)
    runtime._bool = lambda _key, default: default
    runtime.logger = _Logger()
    event = _SendEvent()
    payload = {"ok": False, "error": "model_refused_generation"}

    result = asyncio.run(runtime.send_payload(event, payload))

    assert result == "模型拒绝了生成"
    assert event.sent == ["模型拒绝了生成"]
    assert payload["delivery"]["generated"] is False


class _EmptyResponse:
    """LLM response whose visible completion is empty (deep-thinking case)."""

    completion_text = ""
    reasoning_content = ""
    text = ""


class _EmptyResolver:
    def required_core_tags_for_prompt(self, _prompt):
        return ()

    async def resolve_detailed(self, *, llm_content, **_kwargs):
        # Only reachable on the recovery path; the abort path must stop earlier.
        return DanbooruResolveOutcome(text=llm_content)


def _empty_llm_pipeline(*, deep_thinking: bool = False, second_response: str = ""):
    """Build a pipeline whose first LLM completion is empty.

    When ``deep_thinking`` is enabled the builder is expected to retry once
    without deep thinking; ``second_response`` is what that retry returns.
    """

    class _Context:
        def __init__(self) -> None:
            self.calls = 0
            self.second = second_response

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _EmptyResponse()
            return _Response(self.second)

    class _Plan:
        use_web_search = False
        use_deep_thinking = deep_thinking
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, _prompt):
            return _Plan()

    context = _Context()
    config = {"prompt_optimize_enabled": True}
    pipeline = PromptPipeline(
        context=context,
        config=config,
        logger=_Logger(),
        danbooru_resolver=_EmptyResolver(),
        researcher=_Researcher(),
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda key, default: int(config.get(key, default)),
        get_float=lambda key, default: float(config.get(key, default)),
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=lambda text, limit: text[:limit],
    )
    result = asyncio.run(
        pipeline.build(_Event(), "画一个穿蓝裙的女孩", multi_person=False)
    )
    return context, result


def test_empty_llm_response_aborts_instead_of_sending_chinese_prompt() -> None:
    context, result = _empty_llm_pipeline()

    assert context.calls == 1
    assert result.final_prompt == ""
    assert result.summary["skipped_reason"] == "empty_llm_response"
    assert result.summary["llm_failed"] is True
    assert result.summary["llm_error"] == "empty_llm_response"
    assert result.summary["final_prompt_head"] == ""
    assert result.summary["final_prompt_chars"] == 0
    # The raw Chinese user text must never reach the prompt builder output.
    assert "蓝裙" not in result.final_prompt


def test_empty_llm_response_retries_without_deep_thinking_then_aborts() -> None:
    context, result = _empty_llm_pipeline(
        deep_thinking=True, second_response=""
    )

    assert context.calls == 2
    assert result.final_prompt == ""
    assert result.summary["skipped_reason"] == "empty_llm_response"


def test_empty_llm_response_recovers_on_non_deep_thinking_retry() -> None:
    structured = (
        "{Count: 1girl}\n"
        "{Characters: blue_dress_girl}\n"
        "{Copyright:}\n"
        "{Identity: blue_dress_girl has blue dress}\n"
        "{Details: blue_dress_girl stands}\n"
        "{Tags: full body}\n"
        "{Nltags: A girl in a blue dress.}"
    )
    context, result = _empty_llm_pipeline(
        deep_thinking=True, second_response=structured
    )

    assert context.calls == 2
    assert result.summary.get("skipped_reason") != "empty_llm_response"
    assert "blue dress" in result.final_prompt.lower()


def test_extract_completion_text_prefers_visible_fields_over_empty_completion() -> None:
    class _ReasoningResponse:
        completion_text = ""
        reasoning_content = "long chain of thought"
        content = "1girl, solo"

    class _ChainResponse:
        completion_text = ""
        messages = [{"role": "assistant", "content": "red hair, school uniform"}]

    class _ListResponse:
        completion_text = ""
        output = ["masterpiece", "detailed"]

    assert _extract_completion_text(_ReasoningResponse()) == "1girl, solo"
    assert _extract_completion_text(_ChainResponse()) == "red hair, school uniform"
    assert _extract_completion_text(_ListResponse()) == "masterpiece\ndetailed"
    assert _extract_completion_text(_EmptyResponse()) == ""
    assert _extract_completion_text(object()) == ""


def test_generation_task_does_not_call_comfyui_after_empty_llm_response() -> None:
    class _Recorder:
        def build_generation_start(self, **_kwargs):
            return {"task_id": "task"}

        def mark_prompt_built(self, task, summary) -> None:
            task["prompt_summary"] = summary

        def mark_failure(self, task, error) -> None:
            task["error"] = error

        def write(self, task) -> None:
            self.task = dict(task)

    class _Inputs:
        last_summary = {}

    class _Reference:
        last_summary = {}

    async def run_tool(_args):
        raise AssertionError("ComfyUI must not be called after empty LLM response")

    runner = GenerationTaskRunner(
        task_recorder=_Recorder(),
        image_inputs=_Inputs(),
        reference_context=_Reference(),
        is_allowed=lambda _event: True,
        ensure_ready=lambda _event: asyncio.sleep(0, result={"ok": True}),
        wants_reference_image=lambda _prompt: False,
        augment_reference_image=lambda _event, prompt: asyncio.sleep(
            0, result=prompt
        ),
        augment_quoted_spell=lambda _event, prompt: prompt,
        build_prompt=lambda *_args, **_kwargs: asyncio.sleep(0, result=""),
        prompt_summary=lambda: {
            "skipped_reason": "empty_llm_response",
            "llm_error": "empty_llm_response",
        },
        run_tool=run_tool,
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=lambda text, limit: text[:limit],
    )

    payload = asyncio.run(runner.generate_payload(_Event(), "画一个女孩"))

    assert payload == {
        "ok": False,
        "error": "empty_llm_response",
        "task_id": "task",
        "llm_error": "empty_llm_response",
    }
