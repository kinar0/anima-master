from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from prompt_pipeline import (  # noqa: E402
    PromptPipeline,
    extract_structured_prompt,
    filter_unbound_directional_tags,
    normalize_structured_nltags,
)
from prompt_presets import looks_like_danbooru_tags  # noqa: E402
from prompt_templates import build_llm_prompt  # noqa: E402
from danbooru_resolver import DanbooruResolveOutcome  # noqa: E402
from task_summary import (  # noqa: E402
    apply_verification_summary,
    build_last_task_debug_lines,
    build_strategy_summary,
)


def _shorten(text: str, limit: int = 600) -> str:
    return text[:limit]


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _pipeline(config: dict | None = None) -> PromptPipeline:
    config = dict(config or {})
    return PromptPipeline(
        context=None,
        config=config,
        logger=_Logger(),
        danbooru_resolver=None,
        researcher=None,
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda key, default: int(config.get(key, default)),
        get_float=lambda key, default: float(config.get(key, default)),
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=_shorten,
    )


def test_prompt_pipeline_keeps_first_llm_result_after_tag_cleaning():
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [
                ", ".join(
                    [f"scene detail {index}" for index in range(40)]
                    + [
                        "light rays",
                        "sunbeams",
                        "glowing",
                        "illuminated",
                        "bright",
                        "luminous",
                        "radiant",
                        "floating particles",
                        "light particles",
                    ]
                )
            ]

        async def get_current_chat_provider_id(self, umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            return _Response(self.outputs.pop(0))

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, prompt):
            return _Plan()

    class _Resolver:
        def required_core_tags_for_prompt(self, prompt):
            return ()

        async def resolve_detailed(
            self,
            *,
            llm_content,
            user_prompt,
            fixed_character,
            candidate_hints=(),
        ):
            return DanbooruResolveOutcome(text=llm_content)

    class _Event:
        unified_msg_origin = "session"

    context = _Context()
    config = {"chiyo_preset_enabled": False}
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
        shorten=_shorten,
    )

    result = asyncio.run(pipeline.build(_Event(), "女孩在晨光中伸手"))

    assert "detail_refill_attempted" not in result.summary
    assert "detail_refill_retry" not in result.summary
    assert "short_content_retry" not in result.summary
    assert result.summary["removed_content_tag_count"] > 0
    assert "scene detail 39" in result.final_prompt
    assert context.outputs == []


def test_prompt_pipeline_uses_default_creative_generation():
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.calls = []

        async def get_current_chat_provider_id(self, umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(
                ", ".join(f"creative visual detail {index}" for index in range(52))
            )

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, prompt):
            return _Plan()

    class _Resolver:
        def required_core_tags_for_prompt(self, prompt):
            return ()

        async def resolve_detailed(
            self,
            *,
            llm_content,
            user_prompt,
            fixed_character,
            candidate_hints=(),
        ):
            return DanbooruResolveOutcome(text=llm_content)

    class _Event:
        unified_msg_origin = "session"

    context = _Context()
    config = {"chiyo_preset_enabled": False}
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
        shorten=_shorten,
    )

    result = asyncio.run(
        pipeline.build(
            _Event(),
            "1girl, solo, traveling magical girl, white dress, simple background",
        )
    )

    assert result.summary.get("danbooru_fast_path") is not True
    assert result.summary["llm_content_tag_count"] == 56
    assert result.summary["background_mode"] == "default_portrait"
    assert "white background" in result.final_prompt
    assert len(context.calls) == 2
    assert "以最终图像协调、精致、有表现力和好看为优先" in context.calls[0]["prompt"]
    assert "默认采用自由创作策略" in context.calls[0]["system_prompt"]
    assert "场景类Tag门控" not in context.calls[0]["system_prompt"]


def test_prompt_pipeline_retries_flat_output_for_structured_format() -> None:
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [
                "blue_hair, black_pantyhose",
                (
                    "{Count: 1girl, solo}\n"
                    "{Characters: togawa_sakiko}\n"
                    "{Identity: togawa_sakiko has blue_hair}\n"
                    "{Details: togawa_sakiko wears black_pantyhose}\n"
                    "{Tags: full_body, white_background}\n"
                    "{Nltags: togawa_sakiko wears black_pantyhose.}"
                ),
            ]

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **_kwargs):
            return _Response(self.outputs.pop(0))

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

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(
                text=llm_content,
                status="resolved",
                canonical_tag=llm_content,
                identity_tags=(llm_content,),
            )

    event = type("_Event", (), {"unified_msg_origin": "session"})()
    pipeline = PromptPipeline(
        context=_Context(),
        config={},
        logger=_Logger(),
        danbooru_resolver=_Resolver(),
        researcher=_Researcher(),
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )

    result = asyncio.run(pipeline.build(event, "draw Sakiko"))

    assert result.summary["structured_format_retry"] is True
    assert "_" not in result.final_prompt
    assert ", Nltags:" in result.final_prompt


def test_named_character_uses_evidence_candidate_and_stable_anchors():
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [
                "wrong_name_(example_work), 1girl, black hair, red eyes, white dress",
                (
                    '{"source_name":"伊诺","copyright":"example work",'
                    '"tag_candidates":["correct_name_(example_work)"]}'
                ),
            ]

        async def get_current_chat_provider_id(self, umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            return _Response(self.outputs.pop(0))

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, prompt):
            return _Plan()

    class _Resolver:
        def __init__(self):
            self.calls = []

        def required_core_tags_for_prompt(self, prompt):
            return ()

        async def resolve_detailed(
            self,
            *,
            llm_content,
            user_prompt,
            fixed_character,
            candidate_hints=(),
        ):
            self.calls.append(tuple(candidate_hints))
            if not candidate_hints:
                return DanbooruResolveOutcome(
                    text=llm_content,
                    status="resolved",
                    canonical_tag="wrong_name_(example_work)",
                    identity_tags=("wrong_name_(example_work)",),
                    explicit_request=True,
                )
            return DanbooruResolveOutcome(
                text=llm_content.replace(
                    "wrong_name_(example_work)",
                    "correct_name_(example_work)",
                ),
                status="resolved",
                canonical_tag="correct_name_(example_work)",
                identity_tags=(
                    "correct_name_(example_work)",
                    "blue hair",
                    "blue eyes",
                    "long hair",
                ),
                candidate_hints=tuple(candidate_hints),
            )

    resolver = _Resolver()
    config = {"chiyo_preset_enabled": False}
    pipeline = PromptPipeline(
        context=_Context(),
        config=config,
        logger=_Logger(),
        danbooru_resolver=resolver,
        researcher=_Researcher(),
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda key, default: int(config.get(key, default)),
        get_float=lambda key, default: float(config.get(key, default)),
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=_shorten,
    )

    event = type("_Event", (), {"unified_msg_origin": "session"})()
    result = asyncio.run(pipeline.build(event, "示例游戏角色伊诺"))

    assert resolver.calls == [(), ("correct_name_(example_work)",)]
    assert result.summary["character_resolution_status"] == "resolved"
    assert result.summary["character_canonical_tag"] == "correct_name_(example_work)"
    assert result.summary["character_identity_tags"] == [
        "correct_name_(example_work)",
        "blue hair",
        "blue eyes",
        "long hair",
    ]
    assert "correct name (example work)" in result.final_prompt
    assert "blue hair" in result.final_prompt
    assert "blue eyes" in result.final_prompt
    assert "black hair" not in result.final_prompt
    assert "red eyes" not in result.final_prompt


def test_danbooru_tag_fast_path_detection_accepts_tag_lists():
    assert looks_like_danbooru_tags(
        "masterpiece, best quality, 1girl, solo, white dress, simple background"
    )


def test_extract_structured_prompt_keeps_character_scopes() -> None:
    roster_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls, yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Identity: togawa_sakiko has blue hair and long hair; "
        "chihaya_anon has pink hair and long hair}\n"
        "{Details: togawa_sakiko cosplays Hatsune Miku; "
        "chihaya_anon cosplays Dark Magician Girl}\n"
        "{Tags: standing together, concert stage}\n"
        "{Nltags: Togawa Sakiko and Chihaya Anon pose together.}"
    )

    assert [character.name for character in characters] == [
        "togawa_sakiko",
        "chihaya_anon",
    ]
    assert roster_tags == ("2girls", "yuri")
    assert characters[0].identity_tags == "togawa_sakiko has blue hair and long hair"
    assert characters[0].detail_tags == "togawa_sakiko cosplays Hatsune Miku"
    assert scene == "standing together, concert stage"
    assert nltags == "Togawa Sakiko and Chihaya Anon pose together."


def test_llm_prompt_requires_bidirectional_directed_interaction_binding() -> None:
    prompt = build_llm_prompt(
        "一张白色的大床，穿红黑礼服的丰川祥子抱着穿黑色风衣的千早爱音",
        original_theme="一张白色的大床，穿红黑礼服的丰川祥子抱着穿黑色风衣的千早爱音",
    )

    assert "Return exactly six single-line brace blocks" in prompt
    assert "List the actor/holder/supporter before the recipient" in prompt
    assert "wraps her arms around" in prompt
    assert "is being held by" in prompt
    assert "Do not use `in someone's arms` as the only relationship cue" in prompt
    assert "Tags must not contain unbound" in prompt
    assert "recipient rests passively or is being held" in prompt


def test_structured_prompt_preserves_actor_first_bidirectional_details() -> None:
    roster_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls, yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Identity: togawa_sakiko has blue hair and yellow eyes; "
        "chihaya_anon has pink hair and grey eyes}\n"
        "{Details: togawa_sakiko sits upright and holds chihaya_anon in her arms, "
        "wrapping her arms around chihaya_anon; chihaya_anon lies against "
        "togawa_sakiko's chest and is being held by togawa_sakiko}\n"
        "{Tags: full body, large white bed, warm lighting}\n"
        "{Nltags: togawa_sakiko holds chihaya_anon. Sakiko's arms are wrapped "
        "around Anon. chihaya_anon rests passively against togawa_sakiko's chest.}"
    )

    assert roster_tags == ("2girls", "yuri")
    assert [character.name for character in characters] == [
        "togawa_sakiko",
        "chihaya_anon",
    ]
    assert "holds chihaya_anon" in characters[0].detail_tags
    assert "is being held by togawa_sakiko" in characters[1].detail_tags
    assert scene == "full body, large white bed, warm lighting"
    assert "rests passively" in nltags


def test_shared_tags_drop_unbound_directional_action_and_pose_words() -> None:
    filtered, removed = filter_unbound_directional_tags(
        "full body, embrace, cuddling, lying, sitting, large white bed, warm lighting"
    )

    assert filtered == "full body, large white bed, warm lighting"
    assert removed == ("embrace", "cuddling", "lying", "sitting")


def test_structured_prompt_rejects_a_relationship_without_count_tag() -> None:
    roster_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: togawa_sakiko sits; chihaya_anon lies down}\n"
        "{Tags: bedroom}\n"
        "{Nltags: togawa_sakiko and chihaya_anon are together.}"
    )

    assert roster_tags == ()
    assert characters == ()
    assert scene.startswith("{Count:")
    assert nltags == ""


def test_structured_prompt_rejects_character_only_mentioned_in_anothers_details() -> None:
    roster_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls, yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: chihaya_anon wears a black trench coat and lies in "
        "togawa_sakiko's arms}\n"
        "{Tags: red and black dress, white bed}\n"
        "{Nltags: Togawa Sakiko holds Chihaya Anon.}"
    )

    assert roster_tags == ()
    assert characters == ()
    assert scene.startswith("{Count:")
    assert nltags == ""


def test_structured_nltags_uses_roster_display_names() -> None:
    nltags = normalize_structured_nltags(
        "togawa_sakiko poses with a friend.",
        ("togawa_sakiko", "chihaya_anon"),
    )

    assert nltags == "chihaya anon. togawa sakiko poses with a friend."
    assert "_" not in nltags


def test_prompt_pipeline_marks_missing_provider_as_llm_failure() -> None:
    class _Context:
        async def get_current_chat_provider_id(self, _umo):
            return ""

        def get_config(self, *, umo):
            return {"provider_settings": {}}

    class _Event:
        unified_msg_origin = "session"

    config = {"chiyo_preset_enabled": False}
    pipeline = PromptPipeline(
        context=_Context(),
        config=config,
        logger=_Logger(),
        danbooru_resolver=None,
        researcher=None,
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda key, default: int(config.get(key, default)),
        get_float=lambda key, default: float(config.get(key, default)),
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=_shorten,
    )

    result = asyncio.run(pipeline.build(_Event(), "画一个蓝色连衣裙女孩"))

    assert result.final_prompt == "画一个蓝色连衣裙女孩"
    assert result.summary["llm_failed"] is True
    assert result.summary["llm_error"] == "no_chat_provider"


def test_danbooru_tag_fast_path_detection_rejects_short_chinese_requests():
    assert not looks_like_danbooru_tags("画一个白裙子的女孩，简单背景")


def test_danbooru_tag_fast_path_accepts_one_chinese_character_name():
    assert looks_like_danbooru_tags(
        "1girl, solo, 狐莉, knee up, standing on one leg, foreshortening, "
        "pov, from below, holding sword, fighting stance, serious"
    )


def test_strategy_summary_keeps_debug_flags_compact():
    task = {
        "reference_image_requested": True,
        "reference_context_applied": False,
    }
    prompt_summary = {
        "raw_mode": True,
        "llm_ok": False,
        "outfit_summary_ok": True,
        "web_search": False,
        "deep_thinking": False,
        "fixed_character": True,
        "fixed_character_name": "狐莉",
        "outfit_transfer": True,
        "llm_content_tag_count": 72,
        "final_prompt_chars": 360,
    }

    summary = build_strategy_summary(task, prompt_summary)

    assert summary["reference_requested"] is True
    assert summary["reference_applied"] is False
    assert summary["llm_ok"] is False
    assert summary["fixed_character_name"] == "狐莉"
    assert summary["content_tag_count"] == 72
    assert summary["final_prompt_chars"] == 360


def test_apply_verification_summary_updates_task_and_strategy():
    task = {
        "ok": True,
        "outputs": ["old.png"],
        "strategy_summary": {"raw_mode": False},
    }
    verification_summary = {
        "enabled": True,
        "skipped": False,
        "final_passed": False,
        "final_score": 5,
        "retry_count": 1,
    }
    payload = {"ok": True, "outputs": ["new.png"], "error": ""}

    updated = apply_verification_summary(task, verification_summary, payload)

    assert updated["outputs"] == ["new.png"]
    assert updated["verification_summary"] == verification_summary
    assert updated["strategy_summary"]["raw_mode"] is False
    assert updated["strategy_summary"]["verification"] == {
        "enabled": True,
        "forced_multi_person": False,
        "forced_named_character": False,
        "skipped": False,
        "passed": False,
        "score": 5,
        "retry_count": 1,
    }


def test_last_task_debug_lines_use_strategy_summary():
    lines = build_last_task_debug_lines(
        {
            "time": "2026-07-05T10:00:00",
            "action": "generate",
            "ok": True,
            "outputs": ["x.png"],
            "strategy_summary": {
                "reference_requested": False,
                "reference_applied": False,
                "raw_mode": True,
                "outfit_transfer": False,
                "llm_ok": True,
                "outfit_summary_ok": True,
                "fixed_character_name": "狐莉",
                "web_search": False,
                "deep_thinking": False,
                "final_prompt_chars": 120,
                "verification": {"passed": True, "retry_count": 0},
            },
            "verification_summary": {"enabled": True},
            "prompt_summary": {
                "stage_events": [
                    {"stage": "provider", "status": "ok"},
                    {"stage": "prompt_llm", "status": "ok"},
                ]
            },
        }
    )

    text = "\n".join(lines)
    assert "上次任务摘要" in text
    assert "角色：狐莉" in text
    assert "raw=True" in text
    assert (
        "自检：enabled=True 多人强制=False 角色强制=False passed=True retry=0" in text
    )
    assert "阶段事件：provider=ok，prompt_llm=ok" in text
