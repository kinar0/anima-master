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
    minimal_verified_outfit_nltags,
    normalize_structured_nltags,
)
from prompt_presets import looks_like_danbooru_tags  # noqa: E402
from prompt_templates import build_llm_prompt  # noqa: E402
from danbooru_resolver import DanbooruResolveOutcome  # noqa: E402
from danbooru_semantic import SemanticLookupResult  # noqa: E402
from task_summary import (  # noqa: E402
    apply_verification_summary,
    build_last_task_debug_lines,
    build_strategy_summary,
)


def _shorten(text: str, limit: int = 600) -> str:
    return text[:limit]


def test_semantic_planner_uses_configured_system_prompt() -> None:
    class _Response:
        completion_text = '{"anchors":[]}'

    class _Context:
        def __init__(self):
            self.calls = []

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    config = {"danbooru_semantic_system_prompt": "custom LLM1 system prompt"}
    context = _Context()
    pipeline = PromptPipeline(
        context=context,
        config=config,
        logger=_Logger(),
        danbooru_resolver=object(),
        researcher=object(),
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=_shorten,
    )

    asyncio.run(
        pipeline._generate_semantic_plan_with_llm(
            provider_id="provider",
            user_prompt="test",
        )
    )

    assert context.calls[0]["system_prompt"] == "custom LLM1 system prompt"
    assert context.calls[0]["max_tokens"] == 550
    assert context.calls[0]["thinking"] == {"type": "disabled"}


def test_prompt_builder_controls_thinking_and_output_budget() -> None:
    class _Response:
        completion_text = "{Count: 1girl}"

    class _Context:
        def __init__(self):
            self.calls = []

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    config = {
        "prompt_builder_max_tokens": 640,
        "prompt_builder_reasoning_effort": "high",
    }
    context = _Context()
    pipeline = PromptPipeline(
        context=context,
        config=config,
        logger=_Logger(),
        danbooru_resolver=object(),
        researcher=object(),
        get_bool=lambda _key, default: default,
        get_int=lambda key, default: int(config.get(key, default)),
        get_float=lambda _key, default: default,
        get_str=lambda key, default: str(config.get(key, default)),
        shorten=_shorten,
    )

    asyncio.run(
        pipeline._generate_prompt_tags_with_llm(
            provider_id="provider",
            llm_prompt="test",
            use_deep_thinking=False,
            fixed_character=False,
        )
    )
    asyncio.run(
        pipeline._generate_prompt_tags_with_llm(
            provider_id="provider",
            llm_prompt="test",
            use_deep_thinking=True,
            fixed_character=False,
        )
    )

    assert context.calls[0]["max_tokens"] == 640
    assert context.calls[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in context.calls[0]
    assert context.calls[1]["max_tokens"] == 640
    assert context.calls[1]["thinking"] == {"type": "enabled"}
    assert context.calls[1]["reasoning_effort"] == "high"


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
                    "{Copyright: bang_dream!}\n"
                    "{Identity: togawa_sakiko has blue_hair}\n"
                    "{Details: togawa_sakiko wears black_pantyhose}\n"
                    "{Tags: full_body, sitting, hugging, white_background}\n"
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
    assert result.summary["structured_copyright_tags"] == ["bang_dream!"]
    assert "1girl, solo, togawa sakiko" in result.final_prompt
    assert "bang dream!" in result.final_prompt
    assert "_" not in result.final_prompt
    assert ", Nltags:" in result.final_prompt
    ordered = (
        "1girl",
        "togawa sakiko",
        "bang dream!",
        "togawa sakiko has blue hair",
        "togawa sakiko wears black pantyhose",
        "full body",
        "Nltags: togawa sakiko wears black pantyhose.",
    )
    positions = [result.final_prompt.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)
    assert "has blue hair" not in result.final_prompt.split("Nltags:", 1)[1]
    assert "sitting" in result.final_prompt
    assert "hugging" in result.final_prompt
    assert result.summary["removed_unbound_directional_tags"] == []


def test_semantic_outfit_source_is_kept_separate_from_target_character() -> None:
    class _Response:
        completion_text = (
            "{Count: 1girl, solo}\n"
            "{Characters: chihaya_anon}\n"
            "{Copyright: bang_dream!}\n"
            "{Identity: chihaya_anon has pink hair and grey eyes}\n"
            "{Details: chihaya_anon wears a stage costume}\n"
            "{Tags: full body, white background, background_mode_default_portrait}\n"
            "{Nltags: chihaya_anon wears an Oblivionis stage costume.}"
        )

    class _Context:
        def __init__(self):
            self.calls = []
            self.outputs = [
                '{"anchors":[{"id":"target","role":"target_character",'
                '"group":"character","source_text":"千早爱音",'
                '"description":"Chihaya Anon",'
                '"candidates":["chihaya_anon"]},{"id":"source",'
                '"role":"outfit_source","group":"character",'
                '"source_text":"oblivionis","description":"Oblivionis outfit",'
                '"candidates":["oblivionis_(bang_dream!)"]}]}',
                _Response.completion_text,
            ]

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            response = _Response()
            response.completion_text = self.outputs.pop(0)
            return response

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

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            return SemanticLookupResult(
                confirmed_tags=(
                    "chihaya_anon",
                    "oblivionis_(bang_dream!)",
                    "bang_dream!",
                ),
                outfit_source_tags=("oblivionis_(bang_dream!)",),
                outfit_profile_tags=("red_dress", "puffy_sleeves", "black_mask"),
                anchors=anchors,
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(
                text=llm_content,
                status="resolved",
                canonical_tag=llm_content,
                identity_tags=(llm_content,),
            )

    context = _Context()
    event = type("_Event", (), {"unified_msg_origin": "session"})()
    pipeline = PromptPipeline(
        context=context,
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

    result = asyncio.run(
        pipeline.build(event, "穿着oblivionis服装的千早爱音")
    )

    assert "costume/cosplay source anchors" in context.calls[1]["prompt"]
    for tag in (
        "chihaya anon",
        "oblivionis (bang dream!)",
        "bang dream!",
        "red dress",
        "puffy sleeves",
        "black mask",
    ):
        assert tag in result.final_prompt
    assert result.final_prompt.index("chihaya anon") < result.final_prompt.index(
        "oblivionis (bang dream!)"
    )
    assert "togawa sakiko" not in result.final_prompt


def test_verified_outfit_nltags_drop_guessed_clothing_prose() -> None:
    result = minimal_verified_outfit_nltags(
        "wakaba mutsumi wears a black gothic dress with white lace. "
        "wakaba mutsumi and togawa sakiko stand side by side.",
        ("oblivionis_(bang_dream!)",),
    )

    assert result == (
        "Costume based on the character Oblivionis. "
        "wakaba mutsumi and togawa sakiko stand side by side."
    )


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


def test_unified_roster_uses_local_hints_and_resolves_only_unknown_characters():
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.calls = []

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(
                "{Count: 3girls}\n"
                "{Characters: nagasaki_soyo, chihaya_anon, hatsune_miku}\n"
                "{Copyright: bang_dream!, vocaloid}\n"
                "{Identity: nagasaki_soyo has long brown hair, blue eyes, and "
                "large breasts; chihaya_anon has long pink hair and grey eyes; "
                "hatsune_miku has aqua hair and aqua eyes}\n"
                "{Details: nagasaki_soyo wears a white shirt; chihaya_anon wears "
                "a school uniform; hatsune_miku holds a microphone}\n"
                "{Tags: full body, standing together, simple background, white "
                "background, background_mode_default_portrait}\n"
                "{Nltags: nagasaki_soyo, chihaya_anon, and hatsune_miku stand "
                "together.}"
            )

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, _prompt):
            return _Plan()

    class _Resolver:
        def __init__(self):
            self.calls = []

        def required_core_tags_for_prompt(self, _prompt):
            return ()

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            self.calls.append(llm_content)
            return DanbooruResolveOutcome(
                text=llm_content,
                status="resolved",
                canonical_tag=llm_content,
                identity_tags=(llm_content,),
            )

    config = {
        "fixed_characters": [
            (
                "长崎素世=nagasaki soyo, 1girl with long brown hair, "
                "blue eyes and large breasts"
            ),
            "千早爱音=chihaya anon, 1girl with long pink hair, grey eyes",
        ]
    }
    context = _Context()
    resolver = _Resolver()
    event = type("_Event", (), {"unified_msg_origin": "session"})()
    pipeline = PromptPipeline(
        context=context,
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

    result = asyncio.run(
        pipeline.build(event, "长崎素世和千早爱音与初音未来一起合影")
    )

    llm_prompt = context.calls[0]["prompt"]
    assert "- 长崎素世: nagasaki soyo, 1girl with long brown hair" in llm_prompt
    assert "- 千早爱音: chihaya anon, 1girl with long pink hair" in llm_prompt
    assert resolver.calls == ["hatsune_miku"]
    assert [
        item["status"] for item in result.summary["character_resolution_statuses"]
    ] == ["fixed", "fixed", "resolved"]
    assert result.summary["local_character_hints"] == ["长崎素世", "千早爱音"]
    assert "nagasaki soyo" in result.final_prompt
    assert "long brown hair" in result.final_prompt
    assert "hatsune miku" in result.final_prompt


def test_danbooru_tag_fast_path_detection_accepts_tag_lists():
    assert looks_like_danbooru_tags(
        "masterpiece, best quality, 1girl, solo, white dress, simple background"
    )


def test_extract_structured_prompt_keeps_character_scopes() -> None:
    roster_tags, copyright_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls, yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
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
    assert copyright_tags == ("bang_dream!",)
    assert characters[0].identity_tags == "togawa_sakiko has blue hair and long hair"
    assert characters[0].detail_tags == "togawa_sakiko cosplays Hatsune Miku"
    assert scene == "standing together, concert stage"
    assert nltags == "Togawa Sakiko and Chihaya Anon pose together."


def test_extract_structured_prompt_preserves_every_count_block_tag() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: 2girls, yuri, female_focus, group_hug}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: togawa_sakiko stands; chihaya_anon stands}\n"
        "{Tags: concert stage}\n"
        "{Nltags: Togawa Sakiko and Chihaya Anon stand together.}"
    )

    assert characters
    assert roster_tags == ("2girls", "yuri", "female_focus", "group_hug")


def test_extract_structured_prompt_normalizes_futa_and_female_count() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: 2girls, futanari}\n"
        "{Characters: futa_character, female_character}\n"
        "{Copyright:}\n"
        "{Identity: futa_character has silver hair; female_character has black hair}\n"
        "{Details: futa_character stands; female_character kneels}\n"
        "{Tags: full body, simple background}\n"
        "{Nltags: futa_character stands beside female_character.}"
    )

    assert characters
    assert roster_tags == ("futa with female",)


def test_extract_structured_prompt_accepts_canonical_futa_pair_count() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: futa with female}\n"
        "{Characters: futa_character, female_character}\n"
        "{Copyright:}\n"
        "{Identity: futa_character has silver hair; female_character has black hair}\n"
        "{Details: futa_character stands; female_character kneels}\n"
        "{Tags: full body, simple background}\n"
        "{Nltags: futa_character stands beside female_character.}"
    )

    assert characters
    assert roster_tags == ("futa with female",)


def test_llm_prompt_uses_configured_seven_field_template_without_english_suffix() -> None:
    prompt = build_llm_prompt(
        "一张白色的大床，穿红黑礼服的丰川祥子抱着穿黑色风衣的千早爱音",
        original_theme="一张白色的大床，穿红黑礼服的丰川祥子抱着穿黑色风衣的千早爱音",
    )

    assert "只输出七个单行花括号字段" in prompt
    assert "{Copyright:" in prompt
    assert "Return exactly seven single-line brace blocks" not in prompt
    assert "List the actor/holder/supporter before the recipient" not in prompt


def test_structured_prompt_preserves_actor_first_bidirectional_details() -> None:
    roster_tags, copyright_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls, yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
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
    assert copyright_tags == ("bang_dream!",)
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
    roster_tags, copyright_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: togawa_sakiko sits; chihaya_anon lies down}\n"
        "{Tags: bedroom}\n"
        "{Nltags: togawa_sakiko and chihaya_anon are together.}"
    )

    assert roster_tags == ()
    assert copyright_tags == ()
    assert characters == ()
    assert scene.startswith("{Count:")
    assert nltags == ""


def test_characterless_structured_prompt_accepts_no_humans_count() -> None:
    roster_tags, copyright_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: no_humans}\n"
        "{Characters:}\n"
        "{Copyright:}\n"
        "{Identity:}\n"
        "{Details:}\n"
        "{Tags: black_pantyhose, soles, feet, background_mode_default_portrait}\n"
        "{Nltags: exactly two feet are visible with their soles facing upward.}"
    )

    assert roster_tags == ("no_humans",)
    assert copyright_tags == ()
    assert characters == ()
    assert "black_pantyhose" in scene
    assert nltags == "exactly two feet are visible with their soles facing upward."


def test_characterless_pipeline_keeps_tags_without_chinese_fallback() -> None:
    class _Response:
        completion_text = (
            "{Count: no_humans}\n"
            "{Characters:}\n"
            "{Copyright:}\n"
            "{Identity:}\n"
            "{Details:}\n"
            "{Tags: black_pantyhose, soles, feet, background_mode_default_portrait}\n"
            "{Nltags: exactly two feet are visible with their soles facing upward.}"
        )

    class _Context:
        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **_kwargs):
            return _Response()

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
            raise AssertionError(
                "valid no-humans structure must bypass character resolution"
            )

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

    event = type("_Event", (), {"unified_msg_origin": "session"})()
    result = asyncio.run(pipeline.build(event, "无角色, 只有两只踮起的黑丝足底"))

    assert "no humans" in result.final_prompt
    assert "black pantyhose" in result.final_prompt
    assert "soles" in result.final_prompt
    assert "Nltags: exactly two feet" in result.final_prompt
    assert "无角色" not in result.final_prompt
    assert "full body" not in result.final_prompt
    assert result.summary["structured_prompt_mode"] is True
    assert result.summary["structured_character_mode"] is False


def test_structured_prompt_rejects_character_only_mentioned_in_anothers_details() -> None:
    roster_tags, copyright_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls, yuri}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: chihaya_anon wears a black trench coat and lies in "
        "togawa_sakiko's arms}\n"
        "{Tags: red and black dress, white bed}\n"
        "{Nltags: Togawa Sakiko holds Chihaya Anon.}"
    )

    assert roster_tags == ()
    assert copyright_tags == ()
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
