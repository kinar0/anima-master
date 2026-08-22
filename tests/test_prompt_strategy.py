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
    build_character_effective_outfits,
    controlled_character_outfit_detail,
    character_wardrobe_authority_context,
)
from prompt_presets import looks_like_danbooru_tags  # noqa: E402
from prompt_templates import build_llm_prompt  # noqa: E402
from danbooru_resolver import DanbooruResolveOutcome, DanbooruResolver  # noqa: E402
from danbooru_semantic import (  # noqa: E402
    SemanticAnchor,
    SemanticCharacterPlan,
    SemanticLookupResult,
    SemanticOutfitDirective,
    SemanticWardrobe,
)
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


def test_configured_character_anchors_enter_first_round_semantic_lookup() -> None:
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.calls = []
            self.outputs = [
                '{"anchors":[]}',
                (
                    "{Count: 1girl, solo}\n"
                    "{Characters: revenant_(elden_ring_nightreign)}\n"
                    "{Copyright: elden_ring}\n"
                    "{Identity: revenant_(elden_ring_nightreign) has long hair}\n"
                    "{Details: revenant_(elden_ring_nightreign) crosses her arms}\n"
                    "{Tags: crossed arms, white background, "
                    "background_mode_default_portrait}\n"
                    "{Nltags: revenant_(elden_ring_nightreign) crosses her arms.}"
                ),
            ]

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(self.outputs.pop(0))

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, _prompt):
            return _Plan()

    base_resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )

    class _Resolver:
        def __init__(self):
            self.semantic_anchors = ()
            self.second_round_resolution_calls = 0

        def configured_character_anchors_for_prompt(self, prompt):
            return base_resolver.configured_character_anchors_for_prompt(prompt)

        def required_core_tags_for_prompt(self, _prompt):
            return ()

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            self.semantic_anchors = anchors
            return SemanticLookupResult(
                confirmed_tags=("revenant_(elden_ring)", "elden_ring"),
                anchors=anchors,
                anchor_tags=(
                    ("configured_copyright_1", "elden_ring"),
                    ("configured_character_1", "revenant_(elden_ring)"),
                ),
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            self.second_round_resolution_calls += 1
            return DanbooruResolveOutcome(
                text=llm_content,
                status="resolved",
                canonical_tag=llm_content,
                identity_tags=(llm_content,),
            )

    resolver = _Resolver()
    context = _Context()
    pipeline = PromptPipeline(
        context=context,
        config={},
        logger=_Logger(),
        danbooru_resolver=resolver,
        researcher=_Researcher(),
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )
    event = type("_Event", (), {"unified_msg_origin": "session"})()

    result = asyncio.run(
        pipeline.build(event, "艾尔登法环黑夜君临的复仇者，双手抱胸")
    )

    assert [(anchor.role, anchor.candidates) for anchor in resolver.semantic_anchors] == [
        ("copyright", ("elden_ring",)),
        ("target_character", ("revenant_(elden_ring)",)),
    ]
    assert result.summary["danbooru_semantic_confirmed_tags"] == [
        "revenant_(elden_ring)",
        "elden_ring",
    ]
    assert "confirmed visible character roster (authoritative)" in context.calls[1][
        "prompt"
    ]
    assert "revenant_(elden_ring_nightreign)" not in result.final_prompt
    assert "revenant (elden ring)" in result.final_prompt
    assert resolver.second_round_resolution_calls == 0
    assert result.summary["character_resolution_statuses"][0]["status"] == (
        "semantic_confirmed"
    )


def test_unresolved_localized_character_is_refined_and_exactly_rechecked() -> None:
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.calls = []
            self.outputs = [
                (
                    '{"anchors":['
                    '{"id":"target","role":"target_character",'
                    '"group":"character","source_text":"追踪者",'
                    '"description":"localized hunter character",'
                    '"candidates":["tracker"]},'
                    '{"id":"work","role":"copyright","group":"series",'
                    '"source_text":"艾尔登法环黑夜君临",'
                    '"description":"Elden Ring Nightreign",'
                    '"candidates":["elden_ring"]}]}'
                ),
                (
                    '{"source_name":"追踪者","copyright":"elden_ring",'
                    '"tag_candidates":["wylder_(elden_ring)"]}'
                ),
                (
                    "{Count: 1boy, solo}\n"
                    "{Characters: tracker}\n"
                    "{Copyright: elden_ring}\n"
                    "{Identity: tracker has blonde hair and blue eyes}\n"
                    "{Details: tracker stands for a full body portrait}\n"
                    "{Tags: full body, white background, "
                    "background_mode_default_portrait}\n"
                    "{Nltags: tracker stands in a full body portrait.}"
                ),
            ]

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(self.outputs.pop(0))

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, _prompt):
            return _Plan()

    base_resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )

    class _Resolver:
        def __init__(self):
            self.semantic_calls = []
            self.second_round_resolution_calls = 0

        def configured_character_anchors_for_prompt(self, prompt):
            return base_resolver.configured_character_anchors_for_prompt(prompt)

        def required_core_tags_for_prompt(self, _prompt):
            return ()

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            self.semantic_calls.append(anchors)
            copyright_anchor = next(
                anchor for anchor in anchors if anchor.role == "copyright"
            )
            target_anchor = next(
                anchor for anchor in anchors if anchor.role == "target_character"
            )
            if len(self.semantic_calls) == 1:
                return SemanticLookupResult(
                    confirmed_tags=("elden_ring",),
                    missing_descriptions=(target_anchor.description,),
                    anchors=anchors,
                    anchor_tags=((copyright_anchor.anchor_id, "elden_ring"),),
                    status="resolved",
                )
            assert target_anchor.candidates[0] == "wylder_(elden_ring)"
            return SemanticLookupResult(
                confirmed_tags=("wylder_(elden_ring)", "elden_ring"),
                anchors=anchors,
                anchor_tags=(
                    (target_anchor.anchor_id, "wylder_(elden_ring)"),
                    (copyright_anchor.anchor_id, "elden_ring"),
                ),
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            self.second_round_resolution_calls += 1
            return DanbooruResolveOutcome(text=llm_content)

    resolver = _Resolver()
    context = _Context()
    pipeline = PromptPipeline(
        context=context,
        config={},
        logger=_Logger(),
        danbooru_resolver=resolver,
        researcher=_Researcher(),
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )
    event = type("_Event", (), {"unified_msg_origin": "session"})()

    result = asyncio.run(
        pipeline.build(event, "艾尔登法环黑夜君临的追踪者，全身照")
    )

    assert len(resolver.semantic_calls) == 2
    assert "Known work/copyright scope hints: elden_ring" in context.calls[1][
        "prompt"
    ]
    assert "tracker" not in result.final_prompt
    assert "wylder (elden ring)" in result.final_prompt
    assert result.summary["danbooru_semantic_confirmed_tags"] == [
        "wylder_(elden_ring)",
        "elden_ring",
    ]
    assert resolver.second_round_resolution_calls == 0


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


def test_character_outfits_stay_scoped_and_control_bottomless_per_target() -> None:
    anchors = (
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "Sakiko", ("togawa_sakiko",)),
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        SemanticAnchor("haneoka", "outfit", "outfit", "羽丘冬季校服", "Haneoka winter uniform", ("haneoka_school_uniform",)),
    )
    plans = (
        SemanticCharacterPlan(
            "sakiko",
            SemanticWardrobe("default_profile"),
            (SemanticOutfitDirective("remove", ("lower_body.all",), "", "下半身什么都没穿", "sakiko"),),
        ),
        SemanticCharacterPlan("anon", SemanticWardrobe("named_outfit", "haneoka")),
    )
    lookup = SemanticLookupResult(
        anchor_tags=(("sakiko", "togawa_sakiko"), ("anon", "chihaya_anon"), ("haneoka", "haneoka_school_uniform")),
        character_profiles=(("sakiko", "togawa_sakiko", ("red_shirt", "black_skirt", "black_pantyhose"), ("blue_hair",)),),
        anchor_outfit_profiles=(("haneoka", "haneoka_school_uniform", (), "winter"),),
        status="resolved",
    )

    effective = build_character_effective_outfits(
        anchors,
        plans,
        lookup,
        user_prompt="丰川祥子默认服装且下半身什么都没穿，千早爱音穿羽丘冬季校服",
        known_character_names=("丰川祥子", "千早爱音"),
    )

    sakiko, anon = effective
    assert sakiko.effective.effective_tags == ("red_shirt", "bottomless")
    assert anon.effective.effective_tags == ("haneoka_school_uniform",)
    sakiko_detail = controlled_character_outfit_detail(
        "togawa_sakiko wears haneoka winter school uniform and is bottomless",
        "togawa_sakiko",
        sakiko,
    )
    anon_detail = controlled_character_outfit_detail(
        "chihaya_anon wears hanasaki summer school uniform and is bottomless",
        "chihaya_anon",
        anon,
    )
    assert "red_shirt" in sakiko_detail and "bottomless" in sakiko_detail
    assert "haneoka" not in sakiko_detail
    assert "haneoka_school_uniform" in anon_detail
    assert "hanasaki" not in anon_detail and "bottomless" not in anon_detail


def test_cached_editor_profiles_bind_to_each_character_without_crossing() -> None:
    anchors = (
        SemanticAnchor("a", "target_character", "character", "角色甲", "A", ("character_a",)),
        SemanticAnchor("b", "target_character", "character", "角色乙", "B", ("character_b",)),
    )
    plans = (
        SemanticCharacterPlan("a", SemanticWardrobe("default_profile")),
        SemanticCharacterPlan("b", SemanticWardrobe("default_profile")),
    )
    lookup = SemanticLookupResult(
        outfit_profile_tags=("red_jacket", "blue_dress"),
        source_outfit_profiles=(
            ("角色甲", "character_a", ("red_jacket",), "default"),
            ("角色乙", "character_b", ("blue_dress",), "default"),
        ),
        status="profile_cache",
    )

    effective = build_character_effective_outfits(
        anchors,
        plans,
        lookup,
        user_prompt="角色甲和角色乙站在一起",
    )

    assert effective[0].effective.effective_tags == ("red_jacket",)
    assert effective[1].effective.effective_tags == ("blue_dress",)


def test_two_named_outfits_never_cross_character_details() -> None:
    anchors = (
        SemanticAnchor("a", "target_character", "character", "角色甲", "A", ("character_a",)),
        SemanticAnchor("b", "target_character", "character", "角色乙", "B", ("character_b",)),
        SemanticAnchor("look_a", "outfit", "outfit", "甲套装", "look A", ("named_look_a",)),
        SemanticAnchor("look_b", "outfit", "outfit", "乙套装", "look B", ("named_look_b",)),
    )
    plans = (
        SemanticCharacterPlan("a", SemanticWardrobe("named_outfit", "look_a")),
        SemanticCharacterPlan("b", SemanticWardrobe("named_outfit", "look_b")),
    )
    lookup = SemanticLookupResult(
        anchor_tags=(("look_a", "named_look_a"), ("look_b", "named_look_b")),
        anchor_outfit_profiles=(("look_a", "named_look_a", (), "default"), ("look_b", "named_look_b", (), "default")),
        status="resolved",
    )
    effective = build_character_effective_outfits(
        anchors, plans, lookup, user_prompt="角色甲穿甲套装，角色乙穿乙套装"
    )

    a_detail = controlled_character_outfit_detail("character_a wears named look b", "character_a", effective[0])
    b_detail = controlled_character_outfit_detail("character_b wears named look a", "character_b", effective[1])

    assert "named_look_a" in a_detail and "named_look_b" not in a_detail
    assert "named_look_b" in b_detail and "named_look_a" not in b_detail


def test_four_character_named_outfits_are_all_resolved_independently() -> None:
    character_anchors = tuple(
        SemanticAnchor(
            f"character_{index}",
            "target_character",
            "character",
            f"角色{index}",
            f"Character {index}",
            (f"character_{index}",),
        )
        for index in range(1, 5)
    )
    outfit_anchors = tuple(
        SemanticAnchor(
            f"outfit_{index}",
            "outfit",
            "outfit",
            f"套装{index}",
            f"Named outfit {index}",
            (f"named_outfit_{index}",),
        )
        for index in range(1, 5)
    )
    plans = tuple(
        SemanticCharacterPlan(
            f"character_{index}",
            SemanticWardrobe("named_outfit", f"outfit_{index}"),
        )
        for index in range(1, 5)
    )
    lookup = SemanticLookupResult(
        anchor_tags=tuple(
            (f"outfit_{index}", f"named_outfit_{index}")
            for index in range(1, 5)
        ),
        anchor_outfit_profiles=tuple(
            (f"outfit_{index}", f"named_outfit_{index}", (), "default")
            for index in range(1, 5)
        ),
        status="resolved",
    )

    effective = build_character_effective_outfits(
        (*character_anchors, *outfit_anchors),
        plans,
        lookup,
        user_prompt="角色1穿套装1，角色2穿套装2，角色3穿套装3，角色4穿套装4",
    )

    assert len(effective) == 4
    for index, item in enumerate(effective, start=1):
        detail = controlled_character_outfit_detail(
            f"character_{index} wears named outfit {5 - index}",
            f"character_{index}",
            item,
        )
        assert f"named_outfit_{index}" in detail
        assert all(
            f"named_outfit_{other}" not in detail
            for other in range(1, 5)
            if other != index
        )


def test_creative_wardrobe_keeps_pretty_garments_but_rejects_named_uniform_and_nudity() -> None:
    plan = build_character_effective_outfits(
        (
            SemanticAnchor("target", "target_character", "character", "角色甲", "A", ("character_a",)),
            SemanticAnchor("funeral", "outfit", "outfit", "日式葬礼服装", "Japanese funeral attire", ("japanese_funeral_outfit",)),
        ),
        (SemanticCharacterPlan("target", SemanticWardrobe("named_outfit", "funeral")),),
        SemanticLookupResult(status="resolved"),
        user_prompt="角色甲穿日式葬礼服装、黑色头纱并带魅惑要素",
    )[0]

    assert plan.wardrobe_kind == "creative_fallback"
    assert "角色甲: CREATIVE WARDROBE" in character_wardrobe_authority_context((plan,))

    detail = controlled_character_outfit_detail(
        "character_a wears black veil, lace-trimmed black thighhighs, "
        "Hanasaki summer school uniform and is bottomless",
        "character_a",
        plan,
    )

    assert "black veil" in detail
    assert "lace-trimmed black thighhighs" in detail
    assert "Hanasaki" not in detail
    assert "bottomless" not in detail


def test_structured_pipeline_injects_each_character_wardrobe_without_cross_leak() -> None:
    semantic_json = (
        '{"anchors":['
        '{"id":"sakiko","role":"target_character","group":"character","source_text":"丰川祥子","description":"Sakiko","candidates":["togawa_sakiko"]},'
        '{"id":"anon","role":"target_character","group":"character","source_text":"千早爱音","description":"Anon","candidates":["chihaya_anon"]},'
        '{"id":"mutsumi","role":"target_character","group":"character","source_text":"若叶睦","description":"Mutsumi","candidates":["wakaba_mutsumi"]},'
        '{"id":"haneoka","role":"outfit","group":"outfit","source_text":"羽丘冬季校服","description":"Haneoka winter uniform","candidates":["haneoka_school_uniform"]},'
        '{"id":"funeral","role":"outfit","group":"outfit","source_text":"日式葬礼服装","description":"Japanese funeral attire","candidates":["japanese_funeral_outfit"]}],'
        '"character_plans":['
        '{"target_anchor_id":"sakiko","wardrobe":{"kind":"default_profile"},"directives":[{"operation":"remove","slots":["lower_body.all"],"source_text":"下半身什么都没穿"}]},'
        '{"target_anchor_id":"anon","wardrobe":{"kind":"named_outfit","anchor_id":"haneoka"},"directives":[]},'
        '{"target_anchor_id":"mutsumi","wardrobe":{"kind":"named_outfit","anchor_id":"funeral"},"directives":[]}]}'
    )
    malicious_writer = (
        "{Count: 3girls}\n"
        "{Characters: togawa_sakiko, chihaya_anon, wakaba_mutsumi}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair; wakaba_mutsumi has green hair}\n"
        "{Details: togawa_sakiko wears haneoka winter school uniform and is bottomless; "
        "chihaya_anon wears Hanasaki summer school uniform and is bottomless; "
        "wakaba_mutsumi wears a black veil and lace-trimmed black thighhighs with an Oblivionis stage costume and is nude}\n"
        "{Tags: haneoka_school_uniform, hanasaki_summer_school_uniform, bottomless, black_veil, standing, background_mode_default_portrait}\n"
        "{Nltags: All three characters wear Hanasaki uniforms and are bottomless.}"
    )

    class _Response:
        def __init__(self, text):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [semantic_json, malicious_writer]
            self.calls = []

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
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
            return ("togawa_sakiko", "chihaya_anon", "wakaba_mutsumi")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            return SemanticLookupResult(
                confirmed_tags=("togawa_sakiko", "chihaya_anon", "wakaba_mutsumi", "haneoka_school_uniform", "bang_dream!"),
                named_outfit_tags=("haneoka_school_uniform",),
                anchors=anchors,
                anchor_tags=(("sakiko", "togawa_sakiko"), ("anon", "chihaya_anon"), ("mutsumi", "wakaba_mutsumi"), ("haneoka", "haneoka_school_uniform")),
                character_profiles=(("sakiko", "togawa_sakiko", ("red_shirt", "black_skirt", "black_pantyhose"), ("blue_hair",)),),
                anchor_outfit_profiles=(("haneoka", "haneoka_school_uniform", (), "winter"),),
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(text=llm_content, status="resolved", canonical_tag=llm_content, identity_tags=(llm_content,))

    context = _Context()
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
    event = type("_Event", (), {"unified_msg_origin": "session"})()

    result = asyncio.run(pipeline.build(
        event,
        "丰川祥子穿默认服装且下半身什么都没穿，千早爱音穿羽丘冬季校服，若叶睦穿日式葬礼服装、黑色头纱并带魅惑要素",
    ))

    assert "togawa sakiko wears red shirt" in result.final_prompt
    assert "togawa sakiko is bottomless" in result.final_prompt
    assert "chihaya anon wears haneoka school uniform" in result.final_prompt
    assert "Hanasaki" not in result.final_prompt and "hanasaki" not in result.final_prompt
    anon_tail = result.final_prompt.split("chihaya anon wears haneoka school uniform", 1)[1]
    assert "chihaya anon is bottomless" not in anon_tail
    assert "wakaba mutsumi wears a black veil" in result.final_prompt
    assert "lace-trimmed black thighhighs" in result.final_prompt
    assert "Oblivionis" not in result.final_prompt
    assert "wakaba mutsumi is nude" not in result.final_prompt
    assert result.summary["structured_character_count"] == 3
    assert "- 丰川祥子: CREATIVE WARDROBE" not in context.calls[1]["prompt"]
    assert "- 千早爱音: CREATIVE WARDROBE" not in context.calls[1]["prompt"]
    assert "- 若叶睦: CREATIVE WARDROBE" in context.calls[1]["prompt"]


def test_user_outfit_override_replaces_cached_color_across_pipeline() -> None:
    class _Response:
        completion_text = (
            "{Count: 1girl, solo}\n"
            "{Characters: togawa_sakiko}\n"
            "{Copyright: bang_dream!}\n"
            "{Identity: togawa_sakiko has blue hair and yellow eyes}\n"
            "{Details: togawa_sakiko wears a red shirt and black skirt}\n"
            "{Tags: red_shirt, blue_jacket, standing, "
            "background_mode_default_portrait}\n"
            "{Nltags: togawa_sakiko wears a red shirt.}"
        )

    class _Context:
        def __init__(self):
            self.calls = []

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
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

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_outfit_source(self, source):
            assert source == "oblivionis"
            return SemanticLookupResult(
                confirmed_tags=("oblivionis_(bang_dream!)", "bang_dream!"),
                outfit_source_tags=("oblivionis_(bang_dream!)",),
                outfit_profile_tags=(
                    "red_shirt",
                    "black_corset",
                    "black_skirt",
                    "black_mask",
                    "black_pantyhose",
                ),
                status="profile_cache",
            )

        def semantic_lookup_available(self):
            return True

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
        pipeline.build(
            event,
            "丰川祥子穿着oblivionis的衣服，但上衣是粉色的",
        )
    )

    assert "pink shirt" in result.final_prompt
    assert "red shirt" not in result.final_prompt
    assert "blue jacket" not in result.final_prompt
    assert "black skirt" in result.final_prompt
    assert "oblivionis (bang dream!)" not in result.final_prompt
    assert result.summary["outfit_transfer_target"] == "丰川祥子"
    assert result.summary["outfit_removed_tags"] == ["red_shirt"]
    assert result.summary["outfit_added_tags"] == ["pink_shirt"]
    assert result.summary["outfit_source_anchor_emitted"] is False
    assert "pink_shirt" in context.calls[0]["prompt"]
    assert "never restore" in context.calls[0]["prompt"]

    no_skirt_context = _Context()
    no_skirt_pipeline = PromptPipeline(
        context=no_skirt_context,
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
    no_skirt = asyncio.run(
        no_skirt_pipeline.build(
            event,
            "丰川祥子穿着oblivionis的衣服，但丰川祥子下半身没穿裙子",
        )
    )

    assert "black skirt" not in no_skirt.final_prompt
    assert "black pantyhose" in no_skirt.final_prompt
    assert "bottomless" not in no_skirt.final_prompt
    assert "togawa sakiko wears no skirt" in no_skirt.final_prompt
    assert "oblivionis (bang dream!)" not in no_skirt.final_prompt
    assert no_skirt.summary["outfit_removed_tags"] == ["black_skirt"]

    standalone_context = _Context()
    standalone_pipeline = PromptPipeline(
        context=standalone_context,
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
    standalone = asyncio.run(
        standalone_pipeline.build(event, "丰川祥子下半身没穿裙子")
    )

    assert "black skirt" not in standalone.final_prompt
    assert "red shirt" in standalone.final_prompt
    assert "blue jacket" in standalone.final_prompt
    assert "bottomless" not in standalone.final_prompt
    assert "togawa sakiko wears no skirt" in standalone.final_prompt
    assert standalone.summary["outfit_transfer"] is False


def test_cached_seasonal_named_outfit_is_deterministically_queried() -> None:
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [
                '{"anchors":[{"id":"duplicate_source","role":"outfit",'
                '"group":"outfit","source_text":"oblivionis",'
                '"description":"outfit belonging to oblivionis",'
                '"candidates":["oblivionis_outfit"]}]}',
                (
                    "{Count: 2girls}\n"
                    "{Characters: togawa_sakiko, chihaya_anon}\n"
                    "{Copyright: bang_dream!}\n"
                    "{Identity: togawa_sakiko has blue hair; "
                    "chihaya_anon has pink hair}\n"
                    "{Details: togawa_sakiko wears a red shirt; "
                    "chihaya_anon wears haneoka school uniform}\n"
                    "{Tags: red_shirt, haneoka_school_uniform, standing, "
                    "background_mode_default_portrait}\n"
                    "{Nltags: Both characters stand together.}"
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
        def __init__(self):
            self.semantic_calls = 0

        def required_core_tags_for_prompt(self, _prompt):
            return ("togawa_sakiko", "chihaya_anon")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_outfit_source(self, source):
            if source == "羽丘夏季校服":
                return None
            return SemanticLookupResult(
                confirmed_tags=("oblivionis_(bang_dream!)", "bang_dream!"),
                outfit_source_tags=("oblivionis_(bang_dream!)",),
                outfit_profile_tags=("red_shirt",),
                status="profile_cache",
            )

        def cached_named_outfits_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("haneoka_school_uniform",),
                named_outfit_tags=("haneoka_school_uniform",),
                anchors=(
                    SemanticAnchor(
                        "cached_haneoka_summer",
                        "outfit",
                        "outfit",
                        "羽丘夏季校服",
                        "Haneoka summer school uniform",
                        ("haneoka_school_uniform",),
                    ),
                ),
                status="profile_cache",
            )

        def outfit_source_refresh_needed(self, source):
            return source == "羽丘夏季校服"

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            self.semantic_calls += 1
            assert any(
                anchor.role == "outfit"
                and anchor.source_text == "羽丘夏季校服"
                for anchor in anchors
            )
            assert sum(
                "oblivionis" in anchor.source_text.lower()
                for anchor in anchors
            ) == 0
            return SemanticLookupResult(
                confirmed_tags=("haneoka_school_uniform",),
                named_outfit_tags=("haneoka_school_uniform",),
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(text=llm_content)

    resolver = _Resolver()
    pipeline = PromptPipeline(
        context=_Context(),
        config={},
        logger=_Logger(),
        danbooru_resolver=resolver,
        researcher=_Researcher(),
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )
    event = type("_Event", (), {"unified_msg_origin": "session"})()

    result = asyncio.run(
        pipeline.build(
            event,
            "丰川祥子穿着oblivionis的衣服，千早爱音穿着羽丘夏季校服",
        )
    )

    assert resolver.semantic_calls == 1
    assert "red shirt" in result.final_prompt
    assert "haneoka school uniform" in result.final_prompt
    assert result.summary["danbooru_semantic_named_outfit_tags"] == [
        "haneoka_school_uniform"
    ]


def test_cached_named_outfit_is_hard_tag_and_filters_invented_garments() -> None:
    class _Response:
        completion_text = (
            "{Count: 2girls}\n"
            "{Characters: wakaba_mutsumi, nagasaki_soyo}\n"
            "{Copyright: bang_dream!}\n"
            "{Identity: wakaba_mutsumi has green hair; "
            "nagasaki_soyo has brown hair}\n"
            "{Details: wakaba_mutsumi wears tsukinomori school uniform; "
            "nagasaki_soyo wears tsukinomori school uniform}\n"
            "{Tags: tsukinomori_school_uniform, pleated_skirt, white_shirt, "
            "hypnosis, dark_bedroom, background_mode_explicit_scene}\n"
            "{Nltags: Both characters wear Tsukinomori school uniforms.}"
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
            return ("wakaba_mutsumi", "nagasaki_soyo")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_named_outfits_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("tsukinomori_school_uniform",),
                named_outfit_tags=("tsukinomori_school_uniform",),
                status="profile_cache",
            )

        def semantic_lookup_available(self):
            return False

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(text=llm_content)

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

    result = asyncio.run(
        pipeline.build(event, "若叶睦和长崎素世穿着月之森校服，背景为昏暗卧室")
    )

    assert "tsukinomori school uniform" in result.final_prompt
    assert "pleated skirt" not in result.final_prompt
    assert "white shirt" not in result.final_prompt
    assert "hypnosis" in result.final_prompt


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
    # Anima counts the futa inside the girls count, so one female plus one
    # futa keeps the `2girls` anchor instead of collapsing to a bare pair tag.
    assert roster_tags == ("2girls", "futa with female")


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
    assert roster_tags == ("2girls", "futa with female")


def test_extract_structured_prompt_counts_futa_inside_girls_roster() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: 3girls, futa with female}\n"
        "{Characters: futa_character, female_a, female_b}\n"
        "{Copyright:}\n"
        "{Identity: futa_character has silver hair; female_a has black hair; female_b has brown hair}\n"
        "{Details: futa_character stands; female_a kneels; female_b sits}\n"
        "{Tags: full body, simple background}\n"
        "{Nltags: futa_character stands beside female_a and female_b.}"
    )

    assert characters
    assert roster_tags == ("3girls", "futa with female")


def test_extract_structured_prompt_keeps_lone_futa_as_one_girl() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: 1girl, futanari}\n"
        "{Characters: futa_character}\n"
        "{Copyright:}\n"
        "{Identity: futa_character has silver hair}\n"
        "{Details: futa_character stands}\n"
        "{Tags: full body, simple background}\n"
        "{Nltags: futa_character stands alone.}"
    )

    assert characters
    assert roster_tags == ("1girl", "futanari")


def test_extract_structured_prompt_keeps_futa_with_male_pair() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: futa with male}\n"
        "{Characters: futa_character, male_character}\n"
        "{Copyright:}\n"
        "{Identity: futa_character has silver hair; male_character has black hair}\n"
        "{Details: futa_character stands; male_character sits}\n"
        "{Tags: full body, simple background}\n"
        "{Nltags: futa_character stands beside male_character.}"
    )

    assert characters
    assert roster_tags == ("futa with male",)


def test_extract_structured_prompt_mixed_genders_keep_singular_boy() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: 2girls, 1boy, futanari}\n"
        "{Characters: futa_character, female_character, male_character}\n"
        "{Copyright:}\n"
        "{Identity: futa_character has silver hair; female_character has black hair; male_character has brown hair}\n"
        "{Details: futa_character stands; female_character sits; male_character kneels}\n"
        "{Tags: full body, simple background}\n"
        "{Nltags: futa_character stands beside female_character and male_character.}"
    )

    assert characters
    # The roster has three people: 1 female + 1 futa + 1 boy.  The `2girls`
    # count already includes the futa, and the lone boy keeps the singular form.
    assert roster_tags == ("2girls", "1boy", "futa with female")


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
