from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from prompt_pipeline import (  # noqa: E402
    _parse_structured_prompt,
    _replacement_writer_dimensions,
    CharacterEffectiveOutfit,
    PromptPipeline,
    StructuredPromptCharacter,
    appearance_override_dimensions,
    filter_character_appearance_prose,
    filter_shared_appearance_tags,
    explicit_identity_override_requested,
    enforce_structured_sexual_trait_authority,
    extract_structured_prompt,
    filter_unbound_directional_tags,
    minimal_verified_outfit_nltags,
    merge_authoritative_identity_block,
    normalize_anima_count_tags,
    normalize_structured_nltags,
    strip_outfit_narrative,
    structured_count_tags_match_roster,
    build_character_effective_outfits,
    build_wardrobe_authority,
    complete_character_wardrobe_states,
    controlled_character_outfit_detail,
    character_wardrobe_authority_context,
    annotate_requested_profile_variants,
    apply_character_wardrobe_baselines,
    apply_framing_to_character_outfits,
    apply_requested_wardrobe_mode,
    fallback_missing_unspecified_profiles,
    fallback_target_anchors_from_semantic_evidence,
    filter_profile_tags_for_composition,
    has_explicit_wardrobe_evidence,
    non_wardrobe_confirmed_tags,
    requested_wardrobe_mode,
    requested_wardrobe_modes_by_target,
    reconcile_character_hints_with_appearance,
    reconcile_confirmed_semantic_characters,
    add_explicit_cosplay_source_anchors,
    add_host_outfit_changes_to_plans,
    explicit_cosplay_assignments,
    repair_explicit_cosplay_plans,
    repair_single_target_cached_named_outfit_plan,
    resolve_unspecified_wardrobe_mode,
    requests_casual_life_outfit,
    safe_global_outfit_tags,
    scene_adaptive_wardrobe_marker,
    scoped_outfit_narrative,
)
from outfit_transfer import EffectiveOutfitPlan, UserOutfitPatch  # noqa: E402
from prompt_presets import looks_like_danbooru_tags  # noqa: E402
from prompt_templates import build_llm_prompt, has_positive_futa_request  # noqa: E402
from danbooru_resolver import DanbooruResolveOutcome, DanbooruResolver  # noqa: E402
from danbooru_semantic import (  # noqa: E402
    SemanticAnchor,
    SemanticAppearanceChange,
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


def test_reconcile_confirmed_semantic_characters_repairs_multiple_writer_names() -> None:
    characters, nltags = reconcile_confirmed_semantic_characters(
        (
            StructuredPromptCharacter(
                name="long_nagasaki_soyo",
                identity_tags="long_nagasaki_soyo has brown hair",
                detail_tags="long nagasaki soyo holds a bass",
            ),
            StructuredPromptCharacter(
                name="young_takamatsu_tomori",
                identity_tags="young takamatsu tomori has grey hair",
                detail_tags="young_takamatsu_tomori sings",
            ),
        ),
        "long nagasaki soyo stands beside young_takamatsu_tomori.",
        ("nagasaki_soyo", "takamatsu_tomori"),
    )

    assert tuple(character.name for character in characters) == (
        "nagasaki_soyo",
        "takamatsu_tomori",
    )
    assert characters[0].identity_tags == "nagasaki_soyo has brown hair"
    assert characters[0].detail_tags == "nagasaki_soyo holds a bass"
    assert characters[1].identity_tags == "takamatsu_tomori has grey hair"
    assert characters[1].detail_tags == "takamatsu_tomori sings"
    assert nltags == "nagasaki_soyo stands beside takamatsu_tomori."


def test_reconcile_confirmed_semantic_characters_keeps_ambiguous_and_unknown_names() -> None:
    source = (
        StructuredPromptCharacter(
            name="long_nagasaki_soyo",
            identity_tags="long_nagasaki_soyo has brown hair",
            detail_tags="long_nagasaki_soyo holds a bass",
        ),
        StructuredPromptCharacter(
            name="original_girl",
            identity_tags="original_girl has blue hair",
            detail_tags="original_girl smiles",
        ),
    )

    characters, nltags = reconcile_confirmed_semantic_characters(
        source,
        "long_nagasaki_soyo stands beside original_girl.",
        ("nagasaki_soyo", "soyo"),
    )

    assert characters == source
    assert nltags == "long_nagasaki_soyo stands beside original_girl."


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
    assert context.calls[0]["max_tokens"] == 900
    assert context.calls[0]["thinking"] == {"type": "disabled"}


def test_semantic_planner_migrates_legacy_builtin_system_prompt() -> None:
    class _Response:
        completion_text = '{"anchors":[],"character_plans":[]}'

    class _Context:
        def __init__(self):
            self.calls = []

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    old_builtins = (
        (
            "You extract semantic lookup anchors for a local Danbooru index. "
            "Return valid JSON only. Never claim that a candidate is verified."
        ),
        (
            "You build a character-to-wardrobe relation graph and bounded lookup "
            "anchors for a local Danbooru index. Relationship binding is the "
            "primary task: emit exactly one character_plans item for every visible "
            "target_character, preserve the wearer and its wardrobe/outfit_source "
            "even when the request also changes hair, color, pose, expression, or "
            "scene, and keep separate clauses scoped to their own wearers. Never "
            "merge overlapping source names or request-wide wardrobes. Return valid "
            "JSON only. Candidates are unverified lookup hints."
        ),
        (
            "You identify visible characters, what each one wears, and a few "
            "Danbooru lookup hints. Return the compact characters/lookups JSON "
            "requested by the user prompt. Copy every name and evidence phrase "
            "exactly from the request. Do not invent IDs, groups, descriptions, "
            "or cross-reference keys. Return valid JSON only. Candidate tags are "
            "unverified lookup hints."
        ),
    )
    config = {"danbooru_semantic_system_prompt": ""}
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

    for old_builtin in old_builtins:
        config["danbooru_semantic_system_prompt"] = old_builtin
        asyncio.run(
            pipeline._generate_semantic_plan_with_llm(
                provider_id="provider", user_prompt="test"
            )
        )

    assert len(context.calls) == 3
    assert all(
        call["system_prompt"] not in old_builtins for call in context.calls
    )
    assert all(
        "Extract only what the user said" in call["system_prompt"]
        and "never guess tags" in call["system_prompt"]
        for call in context.calls
    )


def test_full_prompt_debug_logs_semantic_planner_input_and_raw_output() -> None:
    raw_output = (
        '{"anchors":[{"id":"target","role":"target_character",'
        '"group":"character","source_text":"千早爱音",'
        '"description":"Anon","candidates":["chihaya_anon"]}],'
        '"character_plans":[]}'
    )

    class _Response:
        completion_text = raw_output

    class _Context:
        async def llm_generate(self, **_kwargs):
            return _Response()

    class _RecordingLogger:
        def __init__(self):
            self.entries = []

        def info(self, message, *args, **_kwargs):
            self.entries.append(message % args if args else message)

        def warning(self, *_args, **_kwargs):
            pass

    config = {"debug_prompt_enabled": True}
    logger = _RecordingLogger()
    pipeline = PromptPipeline(
        context=_Context(),
        config=config,
        logger=logger,
        danbooru_resolver=object(),
        researcher=object(),
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )

    result = asyncio.run(
        pipeline._generate_semantic_plan_with_llm(
            provider_id="provider",
            user_prompt="千早爱音正在cosplay重音teto",
        )
    )

    combined = "\n".join(logger.entries)
    assert result == raw_output
    assert "semantic planner LLM prompt" in combined
    assert "千早爱音正在cosplay重音teto" in combined
    assert "semantic planner LLM system prompt" in combined
    assert "semantic planner LLM output" in combined
    assert raw_output in combined


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
            allow_futa=True,
        )
    )

    assert context.calls[0]["max_tokens"] == 640
    assert context.calls[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in context.calls[0]
    assert context.calls[1]["max_tokens"] == 640
    assert context.calls[1]["thinking"] == {"type": "enabled"}
    assert context.calls[1]["reasoning_effort"] == "high"
    assert "futanari" not in context.calls[0]["system_prompt"]
    assert "1girl, futanari" in context.calls[1]["system_prompt"]


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
    assert "只输出七个单行花括号字段" in context.calls[0]["prompt"]
    assert "wardrobe.kind" not in context.calls[0]["prompt"]
    assert "Count must contain an exact Danbooru people-count tag" in context.calls[0]["system_prompt"]
    assert "场景类Tag门控" not in context.calls[0]["system_prompt"]


def test_prompt_pipeline_retries_incomplete_structured_parenthesized_character() -> None:
    class _Response:
        def __init__(self, text: str):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.calls = []
            self.outputs = [
                (
                    "{Count: 1girl, solo}\n"
                    "{Characters: revenant_(elden_ring)}\n"
                    "{Copyright: elden_ring}\n"
                    "{Identity: revenant_(elden_ring) has white_hair}\n"
                    # The missing Details field is a real envelope error. The
                    # canonical ``_(...)`` tag must not suppress a retry.
                    "{Tags: arms_crossed, angry}\n"
                    "{Nltags: revenant (elden ring) stands angrily.}"
                ),
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
            self.calls.append(_kwargs)
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
    config = {"debug_prompt_enabled": True}
    context = _Context()
    pipeline = PromptPipeline(
        context=context,
        config=config,
        logger=_Logger(),
        danbooru_resolver=_Resolver(),
        researcher=_Researcher(),
        get_bool=lambda key, default: bool(config.get(key, default)),
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )

    result = asyncio.run(pipeline.build(event, "draw Sakiko"))

    assert result.summary["structured_format_retry"] is True
    assert result.summary["prompt_llm_attempt_count"] == 2
    assert result.summary["prompt_llm_accepted_attempt"] == "retry"
    assert result.summary["structured_initial_validation_errors"] == [
        "missing fields: details"
    ]
    assert result.summary["structured_retry_validation_errors"] == []
    assert "{Characters: revenant_(elden_ring)}" in result.summary[
        "prompt_llm_initial_content"
    ]
    assert "{Characters: togawa_sakiko}" in result.summary[
        "prompt_llm_retry_content"
    ]
    assert "missing fields: details" in context.calls[1]["prompt"]
    assert "Previous response:" in context.calls[1]["prompt"]
    assert result.summary["structured_copyright_tags"] == ["bang_dream!"]
    assert "1girl, solo,\ntogawa sakiko" in result.final_prompt
    assert "bang dream!" in result.final_prompt
    assert "_" not in result.final_prompt
    assert "Nltags:" not in result.final_prompt
    ordered = (
        "1girl",
        "togawa sakiko",
        "bang dream!",
        "togawa sakiko has blue hair",
        "togawa sakiko wears black pantyhose",
        "full body",
    )
    positions = [result.final_prompt.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)
    assert result.final_prompt.count("has blue hair") == 1
    assert result.final_prompt.count("wears black pantyhose") == 1
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
                anchor_tags=(
                    ("target", "chihaya_anon"),
                    ("source", "oblivionis_(bang_dream!)"),
                ),
                anchor_outfit_profiles=((
                    "source",
                    "oblivionis_(bang_dream!)",
                    ("red_dress", "puffy_sleeves", "black_mask"),
                    "default",
                ),),
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

    assert "explicit clothing unresolved" in context.calls[1]["prompt"]
    assert "stage costume" in result.final_prompt
    assert result.summary["wardrobe_source"] == "explicit_but_unresolved"
    assert result.summary["wardrobe_resolution_states"] == {
        "千早爱音": "explicit_but_unresolved"
    }
    for tag in (
        "chihaya anon",
        "bang dream!",
    ):
        assert tag in result.final_prompt
    assert "red dress" not in result.final_prompt
    assert "puffy sleeves" not in result.final_prompt
    assert "black mask" not in result.final_prompt
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
    assert r"revenant \(elden ring\)" in result.final_prompt
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
    assert r"wylder \(elden ring\)" in result.final_prompt
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


def test_outfit_narrative_filter_keeps_non_clothing_fact_located_by_garment() -> None:
    result = strip_outfit_narrative(
        "Anon wears a green school uniform. "
        "Anon is a futanari and has an erection beneath her skirt."
    )

    assert result == "Anon is a futanari and has an erection beneath her skirt."


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
    authority = build_wardrobe_authority(effective, lookup, anchors)
    sakiko_detail = controlled_character_outfit_detail(
        "togawa_sakiko wears haneoka winter school uniform and is bottomless",
        "togawa_sakiko",
        sakiko,
        wardrobe_authority=authority,
    )
    anon_detail = controlled_character_outfit_detail(
        "chihaya_anon wears hanasaki summer school uniform and is bottomless",
        "chihaya_anon",
        anon,
        wardrobe_authority=authority,
    )
    assert "red_shirt" not in sakiko_detail and "bottomless" in sakiko_detail
    assert "haneoka" not in sakiko_detail
    assert "haneoka_school_uniform" not in anon_detail
    # Unknown writer additions are no longer rejected merely for being absent
    # from the database floor; the other wearer's grounded bottomless state is.
    assert "hanasaki" in anon_detail and "bottomless" not in anon_detail

    scoped = scoped_outfit_narrative(sakiko_detail)
    assert "togawa_sakiko is bottomless." in scoped
    anon_scoped = scoped_outfit_narrative(anon_detail)
    assert "chihaya_anon wears hanasaki summer school uniform." in anon_scoped
    assert "bottomless" not in anon_scoped
    assert safe_global_outfit_tags(effective) == ()


def test_cached_outfits_are_never_promoted_to_global_hard_tags() -> None:
    def plan(name: str, *tags: str) -> CharacterEffectiveOutfit:
        return CharacterEffectiveOutfit(
            target_anchor_id=name,
            target_source_text=name,
            target_candidates=(name,),
            wardrobe_kind="default_profile",
            wardrobe_anchor_id="",
            wardrobe_tag="",
            appearance_tags=(),
            effective=EffectiveOutfitPlan(effective_tags=tags),
        )

    red = plan("character_a", "school_uniform", "red_jacket")
    blue = plan("character_b", "school_uniform", "blue_cardigan")

    assert safe_global_outfit_tags((red,)) == ()
    assert safe_global_outfit_tags((red, blue)) == ()


def test_oblivionis_profile_is_context_only_and_not_appended_to_details() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="sakiko",
        target_source_text="丰川祥子",
        target_candidates=("togawa_sakiko",),
        wardrobe_kind="outfit_source",
        wardrobe_anchor_id="oblivionis",
        wardrobe_tag="oblivionis_(bang_dream!)",
        appearance_tags=(),
        effective=EffectiveOutfitPlan(effective_tags=(
            "black_corset", "red_shirt", "black_skirt", "puffy_short_sleeves",
            "black_gloves", "hair_ribbon", "black_ribbon", "black_mask",
            "masquerade_mask", "holding_mask", "brooch", "black_pantyhose",
        )),
    )

    detail = controlled_character_outfit_detail(
        "togawa_sakiko sits in a black corset, red shirt, black skirt, "
        "black gloves, and black pantyhose",
        "togawa_sakiko",
        plan,
    )

    assert "black corset" in detail
    assert "black mask" not in detail
    assert "masquerade mask" not in detail
    assert "holding mask" not in detail
    assert "brooch" not in detail
    assert "togawa_sakiko wears" not in detail
    assert safe_global_outfit_tags((plan,)) == ()


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


def test_creative_wardrobe_keeps_cached_stable_appearance() -> None:
    anchors = (
        SemanticAnchor(
            "raye",
            "target_character",
            "character",
            "闪刀姬零衣（sky striker ace raye）",
            "Raye",
            ("sky_striker_ace_-_raye",),
        ),
    )
    plans = (SemanticCharacterPlan("raye", SemanticWardrobe("creative_fallback")),)
    lookup = SemanticLookupResult(
        appearance_profile_tags=("blonde_hair", "long_hair", "green_eyes"),
        source_outfit_profiles=(
            (
                "sky_striker_ace_-_raye",
                "sky_striker_ace_-_raye",
                ("two-tone_dress",),
                "default",
            ),
        ),
        status="profile_cache",
    )

    effective = build_character_effective_outfits(
        anchors,
        plans,
        lookup,
        user_prompt="闪刀姬零衣被催眠了",
    )

    assert effective[0].wardrobe_kind == "creative_fallback"
    assert effective[0].effective.effective_tags == ()
    assert effective[0].appearance_tags == (
        "blonde_hair",
        "long_hair",
        "green_eyes",
    )


def test_multiple_saved_appearances_bind_per_character() -> None:
    anchors = (
        SemanticAnchor("miku", "target_character", "character", "初音未来", "", ("hatsune_miku",)),
        SemanticAnchor("teto", "target_character", "character", "重音テト", "", ("kasane_teto",)),
    )
    plans = (
        SemanticCharacterPlan("miku", SemanticWardrobe("creative_fallback")),
        SemanticCharacterPlan("teto", SemanticWardrobe("creative_fallback")),
    )
    lookup = SemanticLookupResult(
        character_appearance_profiles=(
            (("初音未来", "hatsune_miku"), "hatsune_miku", ("aqua_hair", "aqua_eyes")),
            (("重音テト", "kasane_teto"), "kasane_teto", ("red_hair", "red_eyes")),
        ),
        status="profile_cache",
    )

    effective = build_character_effective_outfits(
        anchors, plans, lookup, user_prompt="初音未来和重音テト"
    )

    assert effective[0].appearance_tags == ("aqua_hair", "aqua_eyes")
    assert effective[1].appearance_tags == ("red_hair", "red_eyes")


def test_saved_appearance_overrides_fresh_online_profile() -> None:
    anchors = (
        SemanticAnchor(
            "raye",
            "target_character",
            "character",
            "闪刀姬零衣",
            "",
            ("sky_striker_ace_-_raye",),
        ),
    )
    plans = (SemanticCharacterPlan("raye", SemanticWardrobe("creative_fallback")),)
    lookup = SemanticLookupResult(
        character_appearance_profiles=(
            (
                ("闪刀姬零衣", "sky_striker_ace_-_raye"),
                "sky_striker_ace_-_raye",
                ("blonde_hair", "long_hair", "green_eyes"),
            ),
        ),
        character_profiles=(
            (
                "raye",
                "sky_striker_ace_-_raye",
                (),
                ("short_blue_hair", "blue_eyes"),
            ),
        ),
        status="resolved",
    )

    effective = build_character_effective_outfits(
        anchors, plans, lookup, user_prompt="闪刀姬零衣"
    )

    assert effective[0].appearance_tags == (
        "blonde_hair",
        "long_hair",
        "green_eyes",
    )


def test_identity_override_requires_explicit_appearance_language() -> None:
    target = ("闪刀姬零衣", "sky_striker_ace_-_raye")

    assert not explicit_identity_override_requested(
        "闪刀姬零衣被催眠，胸部和大腿变丰满",
        target,
        character_count=1,
    )
    assert not explicit_identity_override_requested(
        "闪刀姬零衣的头发被风吹起，她看着对方的眼睛",
        target,
        character_count=1,
    )
    assert explicit_identity_override_requested(
        "把闪刀姬零衣的头发改成红色，眼睛改成金色",
        target,
        character_count=1,
    )


def test_raw_prompt_regex_does_not_unlock_identity_dimensions() -> None:
    dimensions = appearance_override_dimensions(
        "千早爱音绑着粉色的双钻头发型，就像重音teto那样",
        ("千早爱音", "chihaya_anon"),
        character_count=1,
    )
    merged = merge_authoritative_identity_block(
        "chihaya_anon",
        "chihaya_anon has long pink hair tied in twintails with twin drills, "
        "yellow eyes, and a small build",
        ("long_hair", "pink_hair", "grey_eyes", "flat_chest"),
        dimensions,
    )

    assert dimensions == frozenset()
    assert "twin drills" not in merged
    assert "grey_eyes" in merged
    assert "yellow eyes" not in merged
    assert "flat_chest" in merged


def test_identity_merge_preserves_mixed_is_predicate_grammar() -> None:
    merged = merge_authoritative_identity_block(
        "chihaya_anon",
        "chihaya_anon has pink hair and grey eyes and is a loli",
        ("long_hair", "pink_hair", "grey_eyes", "flat_chest"),
        frozenset(),
    )

    assert merged == (
        "chihaya_anon is a loli and has pink hair, grey eyes, long_hair, "
        "flat_chest"
    )
    assert "has is a loli" not in merged


def test_raw_prompt_regex_does_not_open_explicit_eye_change() -> None:
    dimensions = appearance_override_dimensions(
        "把角色甲的眼睛改成蓝色；角色乙站在旁边",
        ("角色甲", "character_a"),
        character_count=2,
    )
    other_dimensions = appearance_override_dimensions(
        "把角色甲的眼睛改成蓝色；角色乙站在旁边",
        ("角色乙", "character_b"),
        character_count=2,
    )
    merged = merge_authoritative_identity_block(
        "character_a",
        "character_a has black hair and blue eyes",
        ("black_hair", "red_eyes"),
        dimensions,
    )

    assert dimensions == frozenset()
    assert other_dimensions == frozenset()
    assert "blue eyes" not in merged
    assert "red_eyes" in merged
    assert "black hair" in merged
    assert "black_hair" not in merged


def test_heterochromia_does_not_prescribe_or_remove_one_eye_colour() -> None:
    dimensions = appearance_override_dimensions(
        "千早爱音变成异色瞳",
        ("千早爱音", "chihaya_anon"),
        character_count=1,
    )
    merged = merge_authoritative_identity_block(
        "chihaya_anon",
        "chihaya_anon has heterochromia",
        ("pink_hair", "grey_eyes"),
        dimensions,
    )

    assert dimensions == frozenset()
    assert "heterochromia" in merged
    assert "grey_eyes" in merged
    assert "pink_hair" in merged


def test_llm1_appearance_hint_allows_llm2_eye_detail_without_deleting_profile() -> None:
    merged = merge_authoritative_identity_block(
        "chihaya_anon",
        "chihaya_anon has golden eye and grey eye",
        ("pink_hair", "grey_eyes"),
        frozenset(),
        advisory_writer_dimensions=frozenset({"eye_color"}),
    )

    assert "golden eye" in merged
    assert "grey eye" in merged
    assert "grey_eyes" not in merged
    assert "pink_hair" in merged


def test_identity_merge_keeps_stable_dimensions_and_additive_dyed_eye_colour() -> None:
    merged = merge_authoritative_identity_block(
        "wakaba_mutsumi",
        "wakaba_mutsumi has long green hair in twintails, yellow eyes and "
        "pink eyes, and huge breasts",
        ("long_hair", "green_hair", "yellow_eyes", "hair_ornament", "small_breasts"),
        frozenset(),
        advisory_writer_dimensions=frozenset(
            {"hair_style", "eye_color", "chest_size"}
        ),
        replacement_writer_dimensions=frozenset({"hair_style", "chest_size"}),
    )

    assert "long green hair in twintails" in merged
    assert "yellow eyes" in merged
    assert "pink eyes" in merged
    assert "huge breasts" in merged
    assert "hair_ornament" in merged
    assert "small_breasts" not in merged
    assert "green_hair" not in merged
    assert "yellow_eyes" not in merged


def test_llm1_operation_controls_replacement_without_keyword_scanning() -> None:
    changes = (
        SemanticAppearanceChange(
            "若叶睦",
            "她的黄色眼睛染上了粉色，她被催眠了",
            ("eye_color",),
            "additive",
        ),
        SemanticAppearanceChange(
            "若叶睦",
            "胸部也变成了超级巨乳",
            ("chest_size",),
            "replace",
        ),
    )

    dimensions = _replacement_writer_dimensions(changes)

    assert dimensions == frozenset({"chest_size"})


def test_unspecified_legacy_operation_conservatively_keeps_profile_value() -> None:
    dimensions = _replacement_writer_dimensions(
        (
            SemanticAppearanceChange(
                "若叶睦",
                "眼睛发生变化",
                ("eye_color",),
            ),
        )
    )

    assert dimensions == frozenset()


def test_identity_merge_preserves_writer_profile_facts_instead_of_blindly_appending() -> None:
    merged = merge_authoritative_identity_block(
        "wakaba_mutsumi",
        "wakaba_mutsumi has long green hair and yellow eyes",
        ("long_hair", "green_hair", "yellow_eyes", "hair_ornament"),
        frozenset(),
    )

    assert "long green hair" in merged
    assert "yellow eyes" in merged
    assert "hair_ornament" in merged
    assert "green_hair" not in merged
    assert "yellow_eyes" not in merged


def test_raw_prompt_regex_is_disabled_for_all_appearance_wording() -> None:
    dimensions = appearance_override_dimensions(
        "爱音的头发被风吹起，她看着祥子的蓝色眼睛",
        ("爱音", "chihaya_anon"),
        character_count=2,
    )
    chest_dimensions = appearance_override_dimensions(
        "爱音的胸部变得更丰满",
        ("爱音", "chihaya_anon"),
        character_count=1,
    )
    vocative_dimensions = appearance_override_dimensions(
        "爱音，把眼睛改成蓝色；祥子保持原样",
        ("爱音", "chihaya_anon"),
        character_count=2,
    )

    assert dimensions == frozenset()
    assert chest_dimensions == frozenset()
    assert vocative_dimensions == frozenset()
    assert appearance_override_dimensions(
        "丰川祥子的视角", ("丰川祥子", "togawa_sakiko"), character_count=1
    ) == frozenset()


def test_appearance_authority_does_not_keep_orphaned_compound_colours() -> None:
    merged = merge_authoritative_identity_block(
        "chihaya_anon",
        "chihaya_anon has blonde and blue hair, yellow and green eyes",
        ("pink_hair", "grey_eyes"),
        frozenset(),
    )

    assert "blonde" not in merged
    assert "blue hair" not in merged
    assert "yellow" not in merged
    assert "green eyes" not in merged
    assert "pink_hair" in merged
    assert "grey_eyes" in merged


def test_appearance_authority_filters_locked_details_and_shared_tags() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="creative_fallback",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("pink_hair", "grey_eyes"),
        effective=EffectiveOutfitPlan(subject="千早爱音"),
    )

    detail = filter_character_appearance_prose(
        "chihaya_anon smiles, blue hair, yellow eyes, holding a book",
        plan.appearance_tags,
        frozenset(),
    )
    shared = filter_shared_appearance_tags(
        "full body, blue hair, yellow eyes, warm lighting",
        (plan,),
        user_prompt="千早爱音站着微笑",
    )

    assert "smiles" in detail and "holding a book" in detail
    assert "blue hair" not in detail and "yellow eyes" not in detail
    assert shared == "full body, warm lighting"


def test_raw_prompt_regex_does_not_open_dimensions_across_fields() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="creative_fallback",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("pink_hair", "grey_eyes"),
        effective=EffectiveOutfitPlan(subject="千早爱音"),
    )
    prompt = "把千早爱音的眼睛改成蓝色"
    dimensions = appearance_override_dimensions(
        prompt, ("千早爱音", "chihaya_anon"), character_count=1
    )

    detail = filter_character_appearance_prose(
        "chihaya_anon has blue eyes and smiles",
        plan.appearance_tags,
        dimensions,
    )
    shared = filter_shared_appearance_tags(
        "full body, blue eyes", (plan,), user_prompt=prompt
    )

    assert dimensions == frozenset()
    assert "blue eyes" not in detail
    assert "blue eyes" not in shared


def test_shared_appearance_tags_require_all_wearers_to_open_dimension() -> None:
    plans = tuple(
        CharacterEffectiveOutfit(
            target_anchor_id=anchor_id,
            target_source_text=name,
            target_candidates=(candidate,),
            wardrobe_kind="creative_fallback",
            wardrobe_anchor_id="",
            wardrobe_tag="",
            appearance_tags=appearance,
            effective=EffectiveOutfitPlan(subject=name),
        )
        for anchor_id, name, candidate, appearance in (
            ("anon", "爱音", "chihaya_anon", ("grey_eyes",)),
            ("sakiko", "祥子", "togawa_sakiko", ("yellow_eyes",)),
        )
    )

    scoped = filter_shared_appearance_tags(
        "full body, blue eyes",
        plans,
        user_prompt="把爱音的眼睛改成蓝色，祥子保持原样",
    )
    shared = filter_shared_appearance_tags(
        "full body, blue eyes",
        plans,
        user_prompt="爱音和祥子都改成蓝色眼睛",
    )

    assert scoped == "full body"
    assert shared == "full body"


def test_learned_appearance_replaces_stale_configured_identity_hint() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="outfit_source",
        wardrobe_anchor_id="teto",
        wardrobe_tag="kasane_teto",
        appearance_tags=("long_hair", "pink_hair", "grey_eyes", "flat_chest"),
        effective=EffectiveOutfitPlan(),
    )

    reconciled = reconcile_character_hints_with_appearance(
        {"千早爱音": "chihaya anon, long pink hair, blue eyes"}, (plan,)
    )

    assert "grey_eyes" in reconciled["千早爱音"]
    assert "blue eyes" not in reconciled["千早爱音"]


def test_single_target_explicit_cached_named_outfit_repairs_wrong_default_base() -> None:
    target = SemanticAnchor(
        "target",
        "target_character",
        "character",
        "千早爱音",
        "Anon",
        ("chihaya_anon",),
    )
    outfit = SemanticAnchor(
        "summer",
        "outfit",
        "outfit",
        "羽丘夏季校服",
        "summer named outfit",
        ("haneoka_school_uniform",),
    )
    directive = SemanticOutfitDirective(
        "remove", ("lower_body.skirt",), "", "没穿短裙", "target"
    )
    plan = SemanticCharacterPlan(
        "target", SemanticWardrobe("default_profile"), (directive,)
    )
    cached = SemanticLookupResult(
        named_outfit_tags=("haneoka_school_uniform",),
        anchors=(outfit,),
    )

    repaired = repair_single_target_cached_named_outfit_plan(
        (plan,),
        (target, outfit),
        cached,
        "千早爱音穿着羽丘夏季校服但没穿短裙",
    )

    assert repaired[0].wardrobe == SemanticWardrobe("named_outfit", "summer")
    assert repaired[0].directives == (directive,)
    assert repair_single_target_cached_named_outfit_plan(
        (plan,),
        (target, outfit),
        cached,
        "千早爱音站在陈列的羽丘夏季校服旁边",
    ) == (plan,)


def test_multi_target_cached_named_outfit_repairs_only_explicit_wearer_clauses() -> None:
    targets = (
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "", ("togawa_sakiko",)),
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "", ("chihaya_anon",)),
        SemanticAnchor("mutsumi", "target_character", "character", "若叶睦", "", ("wakaba_mutsumi",)),
    )
    outfit = SemanticAnchor(
        "summer", "outfit", "outfit", "羽丘夏季校服", "", ("haneoka_school_uniform",)
    )
    plans = tuple(
        SemanticCharacterPlan(anchor.anchor_id, SemanticWardrobe("creative_fallback"))
        for anchor in targets
    )
    cached = SemanticLookupResult(
        named_outfit_tags=("haneoka_school_uniform",), anchors=(outfit,)
    )

    repaired = repair_single_target_cached_named_outfit_plan(
        plans,
        (*targets, outfit),
        cached,
        "丰川祥子穿着羽丘夏季校服坐着，千早爱音穿着羽丘夏季校服站着，若叶睦站在旁边",
    )

    assert repaired[0].wardrobe == SemanticWardrobe("named_outfit", "summer")
    assert repaired[1].wardrobe == SemanticWardrobe("named_outfit", "summer")
    assert repaired[2].wardrobe == SemanticWardrobe("creative_fallback")


def test_generic_named_outfit_components_reach_writer_for_arbitrary_wearer() -> None:
    target = SemanticAnchor(
        "tomori", "target_character", "character", "高松灯", "", ("takamatsu_tomori",)
    )
    outfit = SemanticAnchor(
        "haneoka_winter",
        "outfit",
        "outfit",
        "羽丘冬季校服",
        "winter reusable named outfit",
        ("haneoka_school_uniform",),
    )
    cached = SemanticLookupResult(
        confirmed_tags=("haneoka_school_uniform",),
        named_outfit_tags=("haneoka_school_uniform",),
        anchors=(outfit,),
        anchor_outfit_profiles=((
            "haneoka_winter",
            "haneoka_school_uniform",
            ("grey_jacket", "white_shirt", "green_skirt", "green_necktie"),
            "winter",
        ),),
    )
    repaired = repair_single_target_cached_named_outfit_plan(
        (SemanticCharacterPlan("tomori", SemanticWardrobe("creative_fallback")),),
        (target, outfit),
        cached,
        "高松灯穿着羽丘冬季校服",
    )
    effective = build_character_effective_outfits(
        (target, outfit),
        repaired,
        cached,
        user_prompt="高松灯穿着羽丘冬季校服",
    )
    context = character_wardrobe_authority_context(effective)

    assert repaired[0].wardrobe == SemanticWardrobe(
        "named_outfit", "haneoka_winter"
    )
    assert effective[0].target_source_text == "高松灯"
    assert effective[0].wardrobe_kind == "named_outfit"
    assert effective[0].effective.effective_tags == (
        "haneoka_school_uniform",
        "grey_jacket",
        "white_shirt",
        "green_skirt",
        "green_necktie",
    )
    assert "named outfit = haneoka_school_uniform, grey_jacket" in context


def test_explicit_cosplay_wording_is_repaired_per_wearer() -> None:
    targets = (
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "", ("togawa_sakiko",)),
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "", ("chihaya_anon",)),
    )
    prompts = (
        "丰川祥子在cosplay初音未来，千早爱音站在旁边",
        "丰川祥子正在cos初音未来，千早爱音站在旁边",
        "丰川祥子扮成初音未来，千早爱音站在旁边",
        "丰川祥子穿着初音未来的cosplay服装，千早爱音站在旁边",
        "丰川祥子身穿初音未来的衣服，千早爱音站在旁边",
    )

    for prompt in prompts:
        anchors = add_explicit_cosplay_source_anchors(targets, prompt)
        assignments = explicit_cosplay_assignments(anchors, prompt)
        repaired = repair_explicit_cosplay_plans(
            (
                SemanticCharacterPlan("sakiko", SemanticWardrobe("creative_fallback")),
                SemanticCharacterPlan("anon", SemanticWardrobe("none")),
            ),
            anchors,
            prompt,
        )

        assert assignments == (("sakiko", "初音未来"),)
        source = next(anchor for anchor in anchors if anchor.role == "outfit_source")
        assert repaired[0].wardrobe == SemanticWardrobe("outfit_source", source.anchor_id)
        assert repaired[1].wardrobe == SemanticWardrobe("none")


def test_host_clothing_change_repairs_a_terse_llm1_plan() -> None:
    anchors = (
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "", ("togawa_sakiko",)),
        SemanticAnchor("source", "outfit_source", "character", "初音未来", "", ("hatsune_miku",)),
    )
    plans = (
        SemanticCharacterPlan("sakiko", SemanticWardrobe("outfit_source", "source")),
    )

    repaired = add_host_outfit_changes_to_plans(
        plans,
        anchors,
        "丰川祥子穿着初音未来的衣服，但上衣改成粉色",
    )

    assert repaired[0].directives == (
        SemanticOutfitDirective(
            "replace_color",
            ("upper_body.primary",),
            "pink",
            "但上衣改成粉色",
            "sakiko",
        ),
    )


def test_two_named_outfits_are_not_host_injected_into_writer_details() -> None:
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

    a_detail = controlled_character_outfit_detail(
        "character_a poses", "character_a", effective[0]
    )
    b_detail = controlled_character_outfit_detail(
        "character_b smiles", "character_b", effective[1]
    )

    assert a_detail == "character_a poses"
    assert b_detail == "character_b smiles"


def test_prompt_matched_saved_outfit_profile_binds_to_each_wearer_plan() -> None:
    anchors = (
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "Sakiko", ("togawa_sakiko",)),
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        # This reproduces LLM1 treating a saved uniform phrase as an outfit
        # source while the prompt-wide profile cache already resolved its alias.
        SemanticAnchor("summer", "outfit_source", "character", "羽丘夏季校服", "Haneoka summer uniform", ("summer_uniform",)),
    )
    plans = (
        SemanticCharacterPlan("sakiko", SemanticWardrobe("outfit_source", "summer")),
        SemanticCharacterPlan("anon", SemanticWardrobe("outfit_source", "summer")),
    )
    saved_tags = (
        "school_uniform",
        "summer_uniform",
        "haneoka_school_uniform",
        "brown_sweater_vest",
        "green_skirt",
        "diagonal-striped_necktie",
    )
    lookup = SemanticLookupResult(
        named_outfit_tags=("haneoka_school_uniform",),
        source_outfit_profiles=((
            "羽丘夏季校服", "summer_uniform", saved_tags, "summer",
        ),),
        status="resolved",
    )

    effective = build_character_effective_outfits(
        anchors,
        plans,
        lookup,
        user_prompt="丰川祥子和千早爱音都穿羽丘夏季校服",
    )

    assert len(effective) == 2
    assert all(item.wardrobe_kind == "named_outfit" for item in effective)
    assert all(item.wardrobe_tag == "summer_uniform" for item in effective)
    assert all(set(item.effective.effective_tags) == set(saved_tags) for item in effective)
    assert all("green_skirt" in item.effective.effective_tags for item in effective)
    authority = build_wardrobe_authority(effective, lookup, anchors)
    assert set(authority.selected_tags) == set(saved_tags)
    assert safe_global_outfit_tags(effective) == ()
    context = character_wardrobe_authority_context(effective)
    assert context.count("named outfit =") == 2

    detail = controlled_character_outfit_detail(
        "togawa_sakiko sits on a chair, wearing the Haneoka summer school "
        "uniform with a white short-sleeved shirt, green pleated skirt, and "
        "black pantyhose, looking up with a composed expression",
        "togawa_sakiko",
        effective[0],
    )
    assert "sits on a chair" in detail
    assert "looking up with a composed expression" in detail
    assert "green_skirt" not in detail
    assert "brown_sweater_vest" not in detail
    assert "green pleated skirt" in detail
    assert "black pantyhose" in detail


def test_four_character_named_outfits_remain_context_only() -> None:
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
            f"character_{index} poses", f"character_{index}", item
        )
        assert detail == f"character_{index} poses"


def test_unverified_named_outfit_remains_resolved_and_lets_writer_complete_it() -> None:
    plan = build_character_effective_outfits(
        (
            SemanticAnchor("target", "target_character", "character", "角色甲", "A", ("character_a",)),
            SemanticAnchor("funeral", "outfit", "outfit", "日式葬礼服装", "Japanese funeral attire", ("japanese_funeral_outfit",)),
        ),
        (SemanticCharacterPlan("target", SemanticWardrobe("named_outfit", "funeral")),),
        SemanticLookupResult(status="resolved"),
        user_prompt="角色甲穿日式葬礼服装、黑色头纱并带魅惑要素",
    )[0]

    assert plan.wardrobe_kind == "named_outfit"
    assert plan.wardrobe_anchor_id == "funeral"
    assert "verified clothing anchors" in character_wardrobe_authority_context((plan,))

    detail = controlled_character_outfit_detail(
        "character_a wears black veil, lace-trimmed black thighhighs, "
        "Hanasaki summer school uniform and is bottomless",
        "character_a",
        plan,
    )

    assert "black veil" in detail
    assert "lace-trimmed black thighhighs" in detail
    assert "Hanasaki" in detail
    assert "bottomless" in detail


def test_casual_outfit_intent_includes_official_casual_but_not_default() -> None:
    assert requests_casual_life_outfit("千早爱音和丰川祥子穿casual服装")
    assert requests_casual_life_outfit("千早爱音穿官方常服")
    assert not requests_casual_life_outfit("丰川祥子穿官方默认服装")
    assert requested_wardrobe_mode("千早爱音穿默认服装") == "default_profile"
    assert requested_wardrobe_mode("千早爱音穿私服") == "creative_fallback"
    assert requested_wardrobe_mode("千早爱音穿居家私服") == "creative_fallback"
    assert requested_wardrobe_mode("千早爱音穿休闲穿搭") == "creative_fallback"
    assert requested_wardrobe_mode("千早爱音穿着演出服") == "stage_profile"
    assert requested_wardrobe_mode("千早爱音穿夏季服装") == "summer_profile"
    assert requested_wardrobe_mode("千早爱音穿冬装") == "winter_profile"
    assert requested_wardrobe_mode("千早爱音站在舞台上") == ""
    assert requested_wardrobe_mode("千早爱音演出结束后休息") == ""


def test_stage_profile_is_scoped_to_its_wearer_and_annotates_resolver_query() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "千早爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "sakiko", "target_character", "character", "丰川祥子", "Sakiko",
            ("togawa_sakiko",),
        ),
    )
    prompt = "千早爱音穿着演出服，丰川祥子穿默认服装"

    assert requested_wardrobe_modes_by_target(prompt, anchors) == {
        "anon": "stage_profile",
        "sakiko": "default_profile",
    }
    annotated = annotate_requested_profile_variants(anchors, prompt)
    assert annotated[0].description.endswith("stage outfit variant")
    assert annotated[1].description == "Sakiko"


def test_stage_profile_prefers_matching_cached_variant_over_default_character_row() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "千早爱音", "Anon",
            ("chihaya_anon",),
        ),
    )
    semantic = SemanticLookupResult(
        # This legacy row has no qualifier and represents the common default
        # profile returned by an older resolver path.
        character_profiles=(
            (
                "anon", "chihaya_anon",
                ("haneoka_school_uniform", "green_skirt"),
                ("pink_hair", "grey_eyes"),
            ),
        ),
        outfit_profile_tags=(
            "blue_jacket", "cropped_jacket", "white_skirt"
        ),
        source_outfit_profiles=(
            (
                "千早爱音", "chihaya_anon",
                ("blue_jacket", "cropped_jacket", "white_skirt"), "stage",
            ),
        ),
    )
    plans = (
        SemanticCharacterPlan("anon", SemanticWardrobe("stage_profile")),
    )

    effective = build_character_effective_outfits(
        anchors, plans, semantic, user_prompt="千早爱音穿着演出服"
    )
    completed = complete_character_wardrobe_states(
        effective,
        anchors,
        semantic,
        explicit_wardrobe_evidence=True,
        requested_mode="stage_profile",
        user_prompt="千早爱音穿着演出服",
    )

    assert completed[0].wardrobe_kind == "stage_profile"
    assert completed[0].resolution_state == "resolved"
    assert completed[0].effective.effective_tags == (
        "blue_jacket", "cropped_jacket", "white_skirt"
    )
    assert "haneoka_school_uniform" not in completed[0].effective.effective_tags
    assert "explicit stage wardrobe" in character_wardrobe_authority_context(
        completed
    )


@pytest.mark.parametrize(
    "source_description",
    ("千早爱音的演出服", "千早爱音演出服"),
)
def test_cross_character_stage_source_selects_stage_not_default(
    source_description: str,
) -> None:
    anchors = (
        SemanticAnchor(
            "sakiko", "target_character", "character", "丰川祥子", "Sakiko",
            ("togawa_sakiko",),
        ),
        SemanticAnchor(
            "anon_source", "outfit_source", "character", "千早爱音",
            source_description, ("chihaya_anon",),
        ),
    )
    semantic = SemanticLookupResult(
        source_outfit_profiles=(
            (
                "千早爱音", "chihaya_anon",
                ("haneoka_school_uniform", "green_skirt"), "default",
            ),
            (
                source_description, "chihaya_anon",
                ("blue_jacket", "cropped_jacket", "white_skirt"), "stage",
            ),
        ),
    )
    plans = (
        SemanticCharacterPlan(
            "sakiko", SemanticWardrobe("outfit_source", "anon_source")
        ),
    )

    effective = build_character_effective_outfits(
        anchors,
        plans,
        semantic,
        user_prompt=f"丰川祥子穿着{source_description}",
    )

    assert effective[0].effective.effective_tags == (
        "blue_jacket", "cropped_jacket", "white_skirt"
    )
    assert "haneoka_school_uniform" not in effective[0].effective.effective_tags


def test_unspecified_wardrobe_policy_uses_only_strong_scene_cues() -> None:
    assert scene_adaptive_wardrobe_marker("若叶睦刚刚出浴") == "刚刚出浴"
    assert scene_adaptive_wardrobe_marker("爱音和祥子进行比赛") == "比赛"
    assert scene_adaptive_wardrobe_marker("爱音和祥子一起看比赛") == ""
    assert scene_adaptive_wardrobe_marker("爱音和祥子在公园里玩") == ""
    assert scene_adaptive_wardrobe_marker("若叶睦在家里吃饭") == ""
    assert scene_adaptive_wardrobe_marker("爱音和祥子在泳池边玩") == "泳池边"
    assert resolve_unspecified_wardrobe_mode("scene_adaptive") == (
        "default_profile",
        "configured_default",
    )
    assert resolve_unspecified_wardrobe_mode(
        "scene_adaptive", scene_marker="刚刚出浴"
    ) == ("default_profile", "scene_adaptive_reference")
    assert resolve_unspecified_wardrobe_mode(
        "scene_adaptive", sensual_mode=True
    ) == ("default_profile", "scene_adaptive_reference")
    assert resolve_unspecified_wardrobe_mode("creative_fallback") == (
        "creative_fallback",
        "configured_creative",
    )


def test_per_character_default_reference_does_not_inherit_another_casual_mode() -> None:
    anchors = (
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "Sakiko", ("togawa_sakiko",)),
    )
    plans = (
        SemanticCharacterPlan("anon", SemanticWardrobe("casual_profile")),
        SemanticCharacterPlan("sakiko", SemanticWardrobe("none")),
    )

    updated = apply_character_wardrobe_baselines(
        plans,
        anchors,
        implicit_mode="default_profile",
        user_prompt="千早爱音穿着常服和丰川祥子在泳池边玩耍",
        create_missing=False,
    )

    assert [plan.wardrobe.kind for plan in updated] == [
        "casual_profile",
        "default_reference",
    ]
    assert requested_wardrobe_modes_by_target(
        "千早爱音穿着常服和丰川祥子在泳池边玩耍", anchors
    ) == {"anon": "casual_profile"}
    assert requested_wardrobe_modes_by_target(
        "千早爱音和丰川祥子都穿常服", anchors
    ) == {"anon": "casual_profile", "sakiko": "casual_profile"}
    assert requested_wardrobe_modes_by_target(
        "千早爱音穿常服和丰川祥子一起站着", anchors
    ) == {"anon": "casual_profile"}


def test_wrong_default_plan_does_not_override_scoped_explicit_garment() -> None:
    anchors = (
        SemanticAnchor(
            "anon",
            "target_character",
            "character",
            "千早爱音",
            "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "sakiko",
            "target_character",
            "character",
            "丰川祥子",
            "Sakiko",
            ("togawa_sakiko",),
        ),
        SemanticAnchor(
            "dress",
            "clothing",
            "clothing",
            "白色婚纱",
            "white wedding dress",
            ("wedding_dress",),
        ),
    )
    plans = apply_character_wardrobe_baselines(
        (
            SemanticCharacterPlan("anon", SemanticWardrobe("default_profile")),
            SemanticCharacterPlan("sakiko", SemanticWardrobe("none")),
        ),
        anchors,
        implicit_mode="default_profile",
        user_prompt="千早爱音穿白色婚纱和丰川祥子一起站着",
        create_missing=False,
    )

    assert [plan.wardrobe.kind for plan in plans] == [
        "default_profile",
        "default_reference",
    ]


def test_modification_without_named_base_uses_default_reference() -> None:
    directive = SemanticOutfitDirective(
        "replace_color",
        ("lower_body.skirt",),
        color="blue",
        source_text="千早爱音的裙子变蓝色",
        target_anchor_id="anon",
    )
    plans = apply_character_wardrobe_baselines(
        (
            SemanticCharacterPlan(
                "anon",
                SemanticWardrobe("default_profile"),
                (directive,),
            ),
        ),
        (
            SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        ),
        implicit_mode="default_profile",
        user_prompt="千早爱音的裙子变蓝色",
        create_missing=False,
    )

    assert plans[0].wardrobe.kind == "default_reference"
    assert plans[0].directives == (directive,)


def test_explicit_ordinary_garment_with_modification_does_not_load_default() -> None:
    directive = SemanticOutfitDirective(
        "remove",
        ("footwear",),
        source_text="千早爱音穿连体式泳装但不穿鞋",
        target_anchor_id="anon",
    )
    plans = apply_character_wardrobe_baselines(
        (
            SemanticCharacterPlan(
                "anon",
                SemanticWardrobe("creative_fallback"),
                (directive,),
            ),
        ),
        (
            SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        ),
        implicit_mode="default_profile",
        user_prompt="千早爱音穿连体式泳装但不穿鞋",
        create_missing=False,
    )

    assert plans[0].wardrobe.kind == "creative_fallback"
    assert plans[0].directives == (directive,)


def test_two_explicit_swimsuits_do_not_load_either_default_wardrobe() -> None:
    anchors = (
        SemanticAnchor(
            "anon",
            "target_character",
            "character",
            "千早爱音",
            "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "sakiko",
            "target_character",
            "character",
            "丰川祥子",
            "Sakiko",
            ("togawa_sakiko",),
        ),
    )
    plans = (
        SemanticCharacterPlan("anon", SemanticWardrobe("creative_fallback")),
        SemanticCharacterPlan("sakiko", SemanticWardrobe("creative_fallback")),
    )
    semantic_result = SemanticLookupResult(
        character_profiles=(
            (
                "anon",
                "chihaya_anon",
                ("haneoka_school_uniform", "green_skirt"),
                ("pink_hair",),
            ),
            (
                "sakiko",
                "togawa_sakiko",
                ("tsukinomori_school_uniform", "blue_skirt"),
                ("blue_hair",),
            ),
        ),
        anchors=anchors,
        status="resolved",
    )

    effective = build_character_effective_outfits(
        anchors,
        plans,
        semantic_result,
        user_prompt="千早爱音穿连体式泳装，丰川祥子穿分体式泳装",
    )

    assert [item.wardrobe_kind for item in effective] == [
        "creative_fallback",
        "creative_fallback",
    ]
    assert all(not item.effective.base_tags for item in effective)
    assert [item.appearance_tags for item in effective] == [
        ("pink_hair",),
        ("blue_hair",),
    ]


def test_composition_omits_only_invisible_profile_components() -> None:
    tags = (
        "haneoka_school_uniform",
        "white_shirt",
        "green_skirt",
        "black_pantyhose",
        "brown_shoes",
    )

    kept, omitted = filter_profile_tags_for_composition(tags, "cowboy shot")

    assert kept == ("haneoka_school_uniform", "white_shirt", "green_skirt")
    assert omitted == ("black_pantyhose", "brown_shoes")

    kept, omitted = filter_profile_tags_for_composition(tags, "upper body")
    assert kept == ("haneoka_school_uniform", "white_shirt")
    assert omitted == ("green_skirt", "black_pantyhose", "brown_shoes")

    kept, omitted = filter_profile_tags_for_composition(tags, "只露出下半身")
    assert kept == (
        "haneoka_school_uniform",
        "green_skirt",
        "black_pantyhose",
        "brown_shoes",
    )
    assert omitted == ("white_shirt",)

    assert filter_profile_tags_for_composition(tags, "但下半身没穿") == (tags, ())


def test_llm_selected_framing_can_crop_already_resolved_profile_components() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="default_reference",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("pink_hair",),
        effective=EffectiveOutfitPlan(
            subject="千早爱音",
            base_tags=(
                "white_shirt",
                "green_skirt",
                "black_pantyhose",
                "brown_shoes",
            ),
            effective_tags=(
                "white_shirt",
                "green_skirt",
                "black_pantyhose",
                "brown_shoes",
            ),
        ),
        resolution_state="unspecified",
    )

    cropped = apply_framing_to_character_outfits((plan,), "upper body")[0]

    assert cropped.effective.base_tags == ("white_shirt",)
    assert cropped.effective.effective_tags == ("white_shirt",)
    assert cropped.composition_omitted_tags == (
        "green_skirt",
        "black_pantyhose",
        "brown_shoes",
    )


def test_pre_llm_crop_hides_omitted_tags_and_filters_compound_writer_prose() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="named_outfit",
        wardrobe_anchor_id="uniform",
        wardrobe_tag="haneoka_school_uniform",
        appearance_tags=("pink_hair",),
        effective=EffectiveOutfitPlan(
            subject="千早爱音",
            base_tags=("white_shirt", "green_skirt", "pleated_skirt"),
            effective_tags=("white_shirt", "green_skirt", "pleated_skirt"),
        ),
        complete_named_profile=True,
    )

    cropped = apply_framing_to_character_outfits(
        (plan,), "上半身肖像，只拍到腰部以上"
    )[0]
    context = character_wardrobe_authority_context((cropped,))
    authority = build_wardrobe_authority(
        (cropped,), SemanticLookupResult(), ()
    )
    detail = controlled_character_outfit_detail(
        "chihaya_anon wears a white shirt, green pleated skirt, and smiles",
        "chihaya_anon",
        cropped,
        wardrobe_authority=authority,
    )

    assert cropped.effective.effective_tags == ("white_shirt",)
    assert "green_skirt" not in context
    assert "pleated_skirt" not in context
    assert "outside the requested framing" in context
    assert "green pleated skirt" not in detail
    assert "white shirt" in detail
    assert "white_shirt" not in detail


def test_explicit_wardrobe_evidence_prevents_config_override() -> None:
    anchors = (
        SemanticAnchor("target", "target_character", "character", "角色甲", "A", ("character_a",)),
        SemanticAnchor("dress", "clothing", "clothing", "白色长裙", "white dress", ("white_dress",)),
    )
    plans = (SemanticCharacterPlan("target", SemanticWardrobe("none")),)

    assert has_explicit_wardrobe_evidence(
        requested_mode="",
        outfit_transfer_enabled=False,
        anchors=anchors,
        plans=plans,
    )

    unknown_named_look = SemanticAnchor(
        "unknown_look",
        "outfit",
        "outfit",
        "星辉祭典造型",
        "a fictional ceremonial look unknown to the host vocabulary",
        ("starlight_ceremonial_look",),
    )
    unknown_look_plan = SemanticCharacterPlan(
        "target", SemanticWardrobe("named_outfit", "unknown_look")
    )
    assert has_explicit_wardrobe_evidence(
        requested_mode="",
        outfit_transfer_enabled=False,
        anchors=(anchors[0], unknown_named_look),
        plans=(unknown_look_plan,),
    )


def test_wedding_dress_plan_does_not_revive_cached_haneoka_uniform_tags() -> None:
    semantic = SemanticLookupResult(
        confirmed_tags=(
            "chihaya_anon",
            "togawa_sakiko",
            "bang_dream!",
            "haneoka_school_uniform",
            "brown_sweater_vest",
            "white_shirt",
            "green_skirt",
            "wedding_dress",
        ),
        outfit_profile_tags=(
            "haneoka_school_uniform",
            "brown_sweater_vest",
            "white_shirt",
            "green_skirt",
        ),
        source_outfit_profiles=(
            (
                "千早爱音",
                "chihaya_anon",
                (
                    "haneoka_school_uniform",
                    "brown_sweater_vest",
                    "white_shirt",
                    "green_skirt",
                ),
                "default",
            ),
        ),
    )
    wedding_plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="creative_fallback",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("pink_hair",),
        effective=EffectiveOutfitPlan(),
    )

    assert non_wardrobe_confirmed_tags(
        semantic,
        (wedding_plan,),
        include_source_anchor=True,
    ) == (
        "chihaya_anon",
        "togawa_sakiko",
        "bang_dream!",
        "wedding_dress",
    )


def test_semantic_planner_runs_without_lookup_and_beats_cached_uniforms() -> None:
    semantic_json = (
        '{"anchors":['
        '{"id":"anon","role":"target_character","group":"character",'
        '"source_text":"千早爱音","description":"Anon",'
        '"candidates":["chihaya_anon"]},'
        '{"id":"sakiko","role":"target_character","group":"character",'
        '"source_text":"丰川祥子","description":"Sakiko",'
        '"candidates":["togawa_sakiko"]},'
        '{"id":"wedding","role":"clothing","group":"clothing",'
        '"source_text":"白色婚纱","description":"white wedding dress",'
        '"candidates":["wedding_dress"]}],'
        '"character_plans":['
        '{"target_anchor_id":"anon",'
        '"wardrobe":{"kind":"creative_fallback"},"directives":[]},'
        '{"target_anchor_id":"sakiko",'
        '"wardrobe":{"kind":"creative_fallback"},"directives":[]}]}'
    )
    writer = (
        "{Count: 2girls, yuri}\n"
        "{Characters: chihaya_anon, togawa_sakiko}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: chihaya_anon has pink hair and grey eyes; "
        "togawa_sakiko has blue hair and yellow eyes}\n"
        "{Details: chihaya_anon wears a white wedding dress, grey jacket, "
        "and long gloves; togawa_sakiko wears a white wedding dress, green "
        "skirt, and long gloves}\n"
        "{Tags: wedding_dress, long_gloves, haneoka_school_uniform, "
        "grey_jacket, green_skirt, bridal_carry, church}\n"
        "{Nltags: chihaya anon and togawa sakiko both wear white wedding "
        "dresses and long gloves in a church.}"
    )

    class _Response:
        def __init__(self, text):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [semantic_json, writer]
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

    cached_uniform = (
        "haneoka_school_uniform",
        "grey_jacket",
        "white_shirt",
        "green_skirt",
        "green_necktie",
        "black_socks",
    )

    class _Resolver:
        def required_core_tags_for_prompt(self, _prompt):
            return ("chihaya_anon", "togawa_sakiko")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("chihaya_anon", "togawa_sakiko"),
                outfit_profile_tags=cached_uniform,
                source_outfit_profiles=(
                    ("千早爱音", "chihaya_anon", cached_uniform, "default"),
                    ("丰川祥子", "togawa_sakiko", cached_uniform, "default"),
                ),
                character_appearance_profiles=(
                    (("千早爱音",), "chihaya_anon", ("pink_hair", "grey_eyes")),
                    (("丰川祥子",), "togawa_sakiko", ("blue_hair", "yellow_eyes")),
                ),
                status="profile_cache",
            )

        def semantic_lookup_available(self):
            return False

        async def resolve_semantic_anchors(self, _anchors):
            raise AssertionError("lookup must be skipped while planning still runs")

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(
                text=llm_content,
                status="resolved",
                canonical_tag=llm_content,
                identity_tags=(llm_content,),
            )

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
    prompt = (
        "百合婚礼、双女主、公主抱、千早爱音、丰川祥子、白色婚纱、"
        "头纱、长手套、脸红微笑、教堂、蓝天、阳光、闪光、浪漫氛围、"
        "低角度近景"
    )

    result = asyncio.run(pipeline.build(event, prompt))
    lowered = result.final_prompt.lower()
    planner_context = context.calls[0]["prompt"]
    writer_context = context.calls[1]["prompt"].lower()

    assert "双女主" in planner_context
    assert "Shared words such as" in planner_context
    assert "wedding dress" in lowered
    assert "chihaya anon wears a white wedding dress" in lowered
    assert "togawa sakiko wears a white wedding dress" in lowered
    assert all(
        tag.replace("_", " ") not in lowered for tag in cached_uniform
    ), lowered
    assert all(tag.replace("_", " ") not in writer_context for tag in cached_uniform)
    assert result.summary["explicit_wardrobe_evidence"] is True
    assert result.summary["wardrobe_source"] == "explicit_clothing"
    assert result.summary["outfit_summary_source"] == (
        "suppressed_by_character_authority"
    )
    assert all(tag not in result.summary["required_core_tags"] for tag in cached_uniform)
    assert set(result.summary["wardrobe_authority"]["stale_cached_tags"]) == set(
        cached_uniform
    )
    assert "wedding_dress" in result.summary["wardrobe_authority"]["explicit_tags"]
    assert len(result.summary["semantic_character_outfits"]) == 2
    assert all(
        item["wardrobe_kind"] == "creative_fallback"
        for item in result.summary["semantic_character_outfits"]
    )
def test_missing_implicit_default_profile_falls_back_to_creative() -> None:
    empty_effective = EffectiveOutfitPlan()
    plan = CharacterEffectiveOutfit(
        target_anchor_id="target",
        target_source_text="角色甲",
        target_candidates=("character_a",),
        wardrobe_kind="default_reference",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("blue_hair",),
        effective=empty_effective,
        resolution_state="unspecified",
    )
    selected_default = replace(
        plan,
        wardrobe_kind="default_profile",
        resolution_state="resolved",
    )

    fallback, changed = fallback_missing_unspecified_profiles(
        (plan,), explicit_wardrobe_evidence=False
    )
    explicit, explicit_changed = fallback_missing_unspecified_profiles(
        (selected_default,), explicit_wardrobe_evidence=True
    )

    assert changed and fallback[0].wardrobe_kind == "creative_fallback"
    assert not explicit_changed and explicit[0].wardrobe_kind == "default_profile"


def test_creative_private_request_synthesizes_per_character_wardrobe_plans() -> None:
    anchors = (
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "Sakiko", ("togawa_sakiko",)),
    )

    plans, unchanged_anchors = apply_requested_wardrobe_mode(
        (), anchors, "creative_fallback"
    )

    assert unchanged_anchors == anchors
    assert [plan.target_anchor_id for plan in plans] == ["anon", "sakiko"]
    assert all(plan.wardrobe.kind == "creative_fallback" for plan in plans)

    default_plans, _ = apply_requested_wardrobe_mode(
        (), anchors, "default_profile"
    )
    assert all(plan.wardrobe.kind == "default_profile" for plan in default_plans)
    corrected_plans, _ = apply_requested_wardrobe_mode(
        (
            SemanticCharacterPlan(
                "anon", SemanticWardrobe("creative_fallback")
            ),
        ),
        anchors,
        "default_profile",
    )
    assert corrected_plans[0].wardrobe.kind == "default_profile"


def test_missing_cosplay_binding_abstains_instead_of_host_recovery() -> None:
    anchors = (
        SemanticAnchor(
            "target", "target_character", "character", "千早爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "source", "outfit_source", "character", "重音teto", "Teto cosplay",
            ("kasane_teto",),
        ),
        SemanticAnchor(
            "pink", "appearance", "general", "粉色", "pink hair", ("pink_hair",),
        ),
    )

    completed = complete_character_wardrobe_states(
        (), anchors, SemanticLookupResult(), explicit_wardrobe_evidence=True
    )

    assert len(completed) == 1
    assert completed[0].resolution_state == "explicit_but_unresolved"
    assert completed[0].wardrobe_kind == "explicit_but_unresolved"
    assert completed[0].wardrobe_anchor_id == ""
    assert "重音teto" in completed[0].unresolved_evidence[0]


def test_planner_failure_uses_cached_target_only_to_suppress_stale_wardrobe() -> None:
    semantic = SemanticLookupResult(
        source_outfit_profiles=((
            "千早爱音", "chihaya_anon",
            ("haneoka_school_uniform", "green_skirt"), "default",
        ),),
        character_appearance_profiles=((
            ("千早爱音",), "chihaya_anon", ("pink_hair", "grey_eyes"),
        ),),
    )
    anchors = fallback_target_anchors_from_semantic_evidence(
        "千早爱音穿白色婚纱", (), semantic
    )
    completed = complete_character_wardrobe_states(
        (),
        anchors,
        semantic,
        explicit_wardrobe_evidence=True,
        user_prompt="千早爱音穿白色婚纱",
    )
    authority = build_wardrobe_authority(completed, semantic, anchors)

    assert len(anchors) == 1
    assert completed[0].resolution_state == "explicit_but_unresolved"
    assert set(authority.stale_cached_tags) == {
        "haneoka_school_uniform", "green_skirt"
    }


def test_planner_failure_fallback_does_not_match_latin_alias_inside_word() -> None:
    semantic = SemanticLookupResult(
        character_appearance_profiles=((
            ("ana",), "ana_character", ("red_hair",),
        ),),
    )

    assert fallback_target_anchors_from_semantic_evidence(
        "banana wearing a dress", (), semantic
    ) == ()
    matched = fallback_target_anchors_from_semantic_evidence(
        "Ana wearing a dress", (), semantic
    )
    assert len(matched) == 1
    assert matched[0].candidates == ("ana_character",)


def test_unresolved_binding_suppresses_target_default_but_keeps_source_as_soft_evidence() -> None:
    anchors = (
        SemanticAnchor(
            "target", "target_character", "character", "千早爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "source", "outfit_source", "character", "重音teto", "Teto cosplay",
            ("kasane_teto",),
        ),
    )
    semantic = SemanticLookupResult(source_outfit_profiles=(
        (
            "千早爱音", "chihaya_anon",
            ("haneoka_school_uniform", "green_skirt"), "default",
        ),
        (
            "重音teto", "kasane_teto",
            ("black_shirt", "black_skirt", "detached_sleeves"), "default",
        ),
    ))
    completed = complete_character_wardrobe_states(
        (), anchors, semantic, explicit_wardrobe_evidence=True
    )
    authority = build_wardrobe_authority(completed, semantic, anchors)
    detail = controlled_character_outfit_detail(
        "chihaya_anon wears black shirt and haneoka school uniform",
        "chihaya_anon",
        completed[0],
        wardrobe_authority=authority,
    )

    assert authority.selected_tags == ()
    assert set(authority.unbound_grounding_tags) == {
        "kasane_teto", "black_shirt", "black_skirt", "detached_sleeves"
    }
    assert set(authority.stale_cached_tags) == {
        "haneoka_school_uniform", "green_skirt"
    }
    assert "black shirt" in detail
    assert "haneoka" not in detail
    assert safe_global_outfit_tags(completed) == ()


def test_multiple_missing_cosplay_bindings_remain_unresolved_per_target() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "sakiko", "target_character", "character", "祥子", "Sakiko",
            ("togawa_sakiko",),
        ),
        SemanticAnchor(
            "magician", "outfit_source", "character", "黑魔导", "Dark Magician",
            ("dark_magician",),
        ),
        SemanticAnchor(
            "magician_girl", "outfit_source", "character", "黑魔导女孩",
            "Dark Magician Girl", ("dark_magician_girl",),
        ),
    )

    completed = complete_character_wardrobe_states(
        (), anchors, SemanticLookupResult(), explicit_wardrobe_evidence=True
    )

    assert [item.target_anchor_id for item in completed] == ["anon", "sakiko"]
    assert all(
        item.resolution_state == "explicit_but_unresolved" for item in completed
    )
    assert all(item.wardrobe_anchor_id == "" for item in completed)
    assert all(len(item.unresolved_evidence) == 2 for item in completed)


def test_incompatible_llm1_default_plan_abstains_when_user_named_clothing_exists() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "千早爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "wedding", "clothing", "clothing", "白色婚纱",
            "white wedding dress", ("wedding_dress",),
        ),
    )
    wrong = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="default_profile",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("pink_hair",),
        effective=EffectiveOutfitPlan(
            subject="千早爱音",
            base_tags=("haneoka_school_uniform", "green_skirt"),
            effective_tags=("haneoka_school_uniform", "green_skirt"),
        ),
    )

    completed = complete_character_wardrobe_states(
        (wrong,), anchors, SemanticLookupResult(), explicit_wardrobe_evidence=True
    )

    assert completed[0].resolution_state == "explicit_but_unresolved"
    assert completed[0].effective.effective_tags == ()
    assert completed[0].wardrobe_tag == ""
    assert "白色婚纱" in completed[0].unresolved_evidence[0]


def test_no_wardrobe_evidence_marks_implicit_plans_unspecified() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "sakiko", "target_character", "character", "祥子", "Sakiko",
            ("togawa_sakiko",),
        ),
    )
    implicit = (
        SemanticCharacterPlan("anon", SemanticWardrobe("default_profile")),
        SemanticCharacterPlan("sakiko", SemanticWardrobe("default_profile")),
    )
    effective = build_character_effective_outfits(
        anchors, implicit, SemanticLookupResult(), user_prompt="爱音和祥子"
    )
    completed = complete_character_wardrobe_states(
        effective, anchors, SemanticLookupResult(), explicit_wardrobe_evidence=False
    )

    assert all(item.resolution_state == "unspecified" for item in completed)


def test_default_reference_survives_other_characters_explicit_wardrobe() -> None:
    anchors = (
        SemanticAnchor("anon", "target_character", "character", "千早爱音", "Anon", ("chihaya_anon",)),
        SemanticAnchor("sakiko", "target_character", "character", "丰川祥子", "Sakiko", ("togawa_sakiko",)),
    )
    semantic = SemanticLookupResult(
        character_profiles=(
            ("anon", "chihaya_anon", ("pink_cardigan", "jeans"), ()),
            ("sakiko", "togawa_sakiko", ("school_uniform", "blue_skirt"), ()),
        ),
        source_outfit_profiles=(
            ("千早爱音", "chihaya_anon", ("pink_cardigan", "jeans"), "casual"),
            ("丰川祥子", "togawa_sakiko", ("school_uniform", "blue_skirt"), "default"),
        ),
    )
    plans = (
        SemanticCharacterPlan("anon", SemanticWardrobe("casual_profile")),
        SemanticCharacterPlan("sakiko", SemanticWardrobe("default_reference")),
    )
    effective = build_character_effective_outfits(
        anchors,
        plans,
        semantic,
        user_prompt="千早爱音穿着常服和丰川祥子在泳池边玩耍",
    )
    completed = complete_character_wardrobe_states(
        effective,
        anchors,
        semantic,
        explicit_wardrobe_evidence=True,
        user_prompt="千早爱音穿着常服和丰川祥子在泳池边玩耍",
    )
    authority = build_wardrobe_authority(completed, semantic, anchors)

    assert completed[0].resolution_state == "resolved"
    assert completed[1].resolution_state == "unspecified"
    assert completed[1].effective.effective_tags == (
        "school_uniform",
        "blue_skirt",
    )
    assert "school_uniform" not in authority.stale_cached_tags
    context = character_wardrobe_authority_context(completed)
    assert "default reference =" in context
    assert "Use when context fits" in context


def test_mixed_default_and_casual_choices_remain_resolved_per_wearer() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "千早爱音", "Anon",
            ("chihaya_anon",),
        ),
        SemanticAnchor(
            "sakiko", "target_character", "character", "丰川祥子", "Sakiko",
            ("togawa_sakiko",),
        ),
    )
    plans = (
        SemanticCharacterPlan("anon", SemanticWardrobe("casual_profile")),
        SemanticCharacterPlan("sakiko", SemanticWardrobe("default_profile")),
    )
    semantic = SemanticLookupResult(
        character_profiles=(
            (
                "anon", "chihaya_anon", ("pink_cardigan", "jeans"),
                ("pink_hair", "grey_eyes"),
            ),
            (
                "sakiko", "togawa_sakiko", ("school_uniform", "blue_skirt"),
                ("blue_hair", "yellow_eyes"),
            ),
        )
    )
    effective = build_character_effective_outfits(
        anchors,
        plans,
        semantic,
        user_prompt="千早爱音穿官方常服，丰川祥子穿默认服装",
    )

    completed = complete_character_wardrobe_states(
        effective,
        anchors,
        semantic,
        explicit_wardrobe_evidence=True,
        # The legacy request-wide detector sees only one value; wearer-scoped
        # clauses must still preserve both valid LLM1 assignments.
        requested_mode="default_profile",
        user_prompt="千早爱音穿官方常服，丰川祥子穿默认服装",
    )

    assert [item.resolution_state for item in completed] == ["resolved", "resolved"]
    assert completed[0].effective.effective_tags == ("pink_cardigan", "jeans")
    assert completed[1].effective.effective_tags == ("school_uniform", "blue_skirt")


def test_mixed_wardrobe_modes_support_connector_without_punctuation() -> None:
    anchors = (
        SemanticAnchor(
            "anon", "target_character", "character", "爱音", "Anon", (),
        ),
        SemanticAnchor(
            "sakiko", "target_character", "character", "祥子", "Sakiko", (),
        ),
    )

    assert requested_wardrobe_modes_by_target(
        "爱音穿官方常服而祥子穿默认服装", anchors
    ) == {
        "anon": "casual_profile",
        "sakiko": "default_profile",
    }


def test_futa_with_male_keeps_total_count_for_three_person_roster() -> None:
    assert normalize_anima_count_tags(("futa with male",), 2) == (
        "futa with male",
    )
    assert normalize_anima_count_tags(("futa with male",), 3) == (
        "3people",
        "futa with male",
    )
    assert normalize_anima_count_tags(("1boy", "futanari"), 3) == (
        "3people",
        "futa with male",
    )


def test_unspecified_default_is_fallback_not_override_of_llm2_raw_interpretation() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="anon",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="default_reference",
        wardrobe_anchor_id="",
        wardrobe_tag="",
        appearance_tags=("pink_hair",),
        effective=EffectiveOutfitPlan(
            subject="千早爱音",
            base_tags=("haneoka_school_uniform", "green_skirt"),
            effective_tags=("haneoka_school_uniform", "green_skirt"),
        ),
        resolution_state="unspecified",
    )

    corrected = controlled_character_outfit_detail(
        "chihaya_anon wears a white wedding dress and veil",
        "chihaya_anon",
        plan,
    )
    fallback = controlled_character_outfit_detail(
        "chihaya_anon smiles and waves",
        "chihaya_anon",
        plan,
    )

    assert "white wedding dress" in corrected
    assert "haneoka" not in corrected and "green_skirt" not in corrected
    assert "haneoka_school_uniform" not in fallback and "green_skirt" not in fallback
    context = character_wardrobe_authority_context((plan,))
    assert "default reference =" in context
    assert "Use when context fits" in context


def test_outfit_source_keeps_writer_completion_and_rejects_stale_uniform() -> None:
    plan = CharacterEffectiveOutfit(
        target_anchor_id="target",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="outfit_source",
        wardrobe_anchor_id="source",
        wardrobe_tag="kasane_teto",
        appearance_tags=("pink_hair", "grey_eyes"),
        effective=EffectiveOutfitPlan(
            base_tags=("black_skirt", "detached_sleeves"),
            effective_tags=("black_skirt", "detached_sleeves"),
        ),
    )
    authority = type("_Authority", (), {
        "filter_prose": lambda _self, text: text.replace(
            "haneoka school uniform", ""
        )
    })()

    detail = controlled_character_outfit_detail(
        "chihaya_anon wears a black sleeveless shirt, black thighhighs, boots, "
        "and haneoka school uniform",
        "chihaya_anon",
        plan,
        wardrobe_authority=authority,
    )

    assert "black sleeveless shirt" in detail
    assert "black thighhighs" in detail
    assert "boots" in detail
    assert "haneoka school uniform" not in detail
    assert "black_skirt" not in detail
    assert "detached_sleeves" not in detail
    context = character_wardrobe_authority_context((plan,))
    assert "cosplay source = kasane_teto" in context
    assert "verified outfit anchors = black_skirt, detached_sleeves" in context
    assert "Complete recognizable source clothing" in context


def test_outfit_source_variants_are_grounding_not_stale_target_clothes() -> None:
    target = SemanticAnchor(
        "target", "target_character", "character", "千早爱音", "Anon",
        ("chihaya_anon",),
    )
    source = SemanticAnchor(
        "source", "outfit_source", "character", "重音teto", "Teto cosplay",
        ("kasane_teto",),
    )
    plan = CharacterEffectiveOutfit(
        target_anchor_id="target",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="outfit_source",
        wardrobe_anchor_id="source",
        wardrobe_tag="kasane_teto",
        appearance_tags=("pink_hair", "grey_eyes"),
        effective=EffectiveOutfitPlan(
            base_tags=("black_skirt",),
            effective_tags=("black_skirt",),
        ),
    )
    semantic = SemanticLookupResult(
        source_outfit_profiles=(
            (
                "千早爱音", "chihaya_anon",
                ("haneoka_school_uniform", "green_skirt"), "default",
            ),
            (
                "重音teto", "kasane_teto",
                ("black_skirt", "black_bow", "black_thighhighs"), "default",
            ),
        ),
    )

    authority = build_wardrobe_authority((plan,), semantic, (target, source))

    assert set(authority.source_grounding_tags) == {
        "black_skirt", "black_bow", "black_thighhighs"
    }
    assert set(authority.stale_cached_tags) == {
        "haneoka_school_uniform", "green_skirt"
    }
    assert "black_bow" in authority.filter_prose("wears black_bow")
    assert "haneoka" not in authority.filter_prose("wears haneoka school uniform")


def test_outfit_source_mutations_override_pristine_source_grounding() -> None:
    target = SemanticAnchor(
        "target", "target_character", "character", "千早爱音", "Anon",
        ("chihaya_anon",),
    )
    source = SemanticAnchor(
        "source", "outfit_source", "character", "重音teto", "Teto cosplay",
        ("kasane_teto",),
    )
    patches = (
        UserOutfitPatch("千早爱音", "remove", "lower_body.skirt", evidence="裙子不翼而飞"),
        UserOutfitPatch("千早爱音", "damage", "upper_body.primary", evidence="上衣被撕破了"),
    )
    plan = CharacterEffectiveOutfit(
        target_anchor_id="target",
        target_source_text="千早爱音",
        target_candidates=("chihaya_anon",),
        wardrobe_kind="outfit_source",
        wardrobe_anchor_id="source",
        wardrobe_tag="kasane_teto",
        appearance_tags=("pink_hair", "grey_eyes"),
        effective=EffectiveOutfitPlan(
            base_tags=("black_shirt", "black_skirt", "detached_sleeves"),
            effective_tags=("black_shirt", "detached_sleeves"),
            removed_tags=("black_skirt",),
            patches=patches,
        ),
    )
    semantic = SemanticLookupResult(source_outfit_profiles=((
        "重音teto", "kasane_teto",
        ("black_shirt", "black_skirt", "detached_sleeves"), "default",
    ),))

    authority = build_wardrobe_authority((plan,), semantic, (target, source))
    detail = controlled_character_outfit_detail(
        "chihaya_anon cosplays Teto in a black shirt",
        "chihaya_anon",
        plan,
        wardrobe_authority=authority,
    )

    assert "black_skirt" not in authority.selected_tags
    assert "black_skirt" not in authority.source_grounding_tags
    assert "black_skirt" not in authority.filter_tags(("black_skirt",))
    assert "wears no skirt" not in detail
    assert "upper garment is visibly torn and ripped" not in detail
    assert "black shirt" in detail
    context = character_wardrobe_authority_context((plan,))
    assert "remove lower_body.skirt" in context
    assert "damage upper_body.primary" in context
    assert "Changes: remove lower_body.skirt" in context


def test_casual_request_uses_casual_evidence_instead_of_default_profiles() -> None:
    semantic_json = (
        '{"anchors":['
        '{"id":"sakiko","role":"target_character","group":"character","source_text":"丰川祥子","description":"Sakiko","candidates":["togawa_sakiko"]},'
        '{"id":"anon","role":"target_character","group":"character","source_text":"千早爱音","description":"Anon","candidates":["chihaya_anon"]}]}'
    )
    malicious_writer = (
        "{Count: 2girls}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: togawa_sakiko wears Tsukinomori school uniform and a blue cardigan; "
        "chihaya_anon wears Haneoka summer school uniform and a pink hoodie}\n"
        "{Tags: tsukinomori_school_uniform, haneoka_school_uniform, "
        "blue_cardigan, pink_hoodie, standing, background_mode_default_portrait}\n"
        "{Nltags: Both girls wear their school uniforms.}"
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
            return ("togawa_sakiko", "chihaya_anon")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_outfit_profiles_for_prompt(self, _prompt):
            return None

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            assert all(
                "casual outfit variant" in anchor.description
                for anchor in anchors
                if anchor.role == "target_character"
            )
            return SemanticLookupResult(
                confirmed_tags=("togawa_sakiko", "chihaya_anon", "bang_dream!"),
                outfit_profile_tags=(
                    "blue_cardigan",
                    "pink_hoodie",
                ),
                character_profiles=(
                    (
                        "sakiko",
                        "togawa_sakiko",
                        ("blue_cardigan",),
                        ("blue_hair",),
                    ),
                    (
                        "anon",
                        "chihaya_anon",
                        ("pink_hoodie",),
                        ("pink_hair",),
                    ),
                ),
                anchor_tags=(
                    ("sakiko", "togawa_sakiko"),
                    ("anon", "chihaya_anon"),
                ),
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

    result = asyncio.run(
        pipeline.build(event, "千早爱音和丰川祥子穿官方常服一起合影")
    )

    lowered = result.final_prompt.lower()
    assert "casual" in lowered
    assert "blue cardigan" in lowered
    assert "pink hoodie" in lowered
    assert all(
        item["resolution_state"] == "resolved"
        for item in result.summary["semantic_character_outfits"]
    )
    assert result.summary["wardrobe_source"] == "explicit_casual_profile"


def test_casual_profile_keeps_a_school_uniform_when_casual_evidence_contains_it() -> None:
    plan = build_character_effective_outfits(
        (
            SemanticAnchor(
                "target",
                "target_character",
                "character",
                "角色甲",
                "A casual outfit variant",
                ("character_a",),
            ),
        ),
        (
            SemanticCharacterPlan(
                "target",
                SemanticWardrobe("casual_profile"),
            ),
        ),
        SemanticLookupResult(
            character_profiles=(
                (
                    "target",
                    "character_a",
                    ("school_uniform", "blue_cardigan"),
                    (),
                ),
            ),
            status="resolved",
        ),
        user_prompt="角色甲穿casual服装",
    )[0]

    assert plan.wardrobe_kind == "casual_profile"
    assert plan.effective.effective_tags == ("school_uniform", "blue_cardigan")


def test_pipeline_binds_natural_stage_request_to_character_stage_profile() -> None:
    semantic_json = (
        '{"characters":[{"name":"千早爱音","aliases":[],'
        '"clothing":"演出服","clothing_source":null,'
        '"clothing_changes":[],"appearance_changes":[]}]}'
    )
    writer = (
        "{Count: 1girl}\n"
        "{Characters: chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: chihaya_anon has long pink hair and grey eyes}\n"
        "{Details: chihaya_anon wears her blue stage costume and poses}\n"
        "{Tags: blue jacket, white skirt, standing, "
        "background_mode_default_portrait}\n"
        "{Nltags: Chihaya Anon poses in her stage costume.}"
    )

    class _Response:
        def __init__(self, text):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [semantic_json, writer]
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
            return ()

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("chihaya_anon",),
                outfit_profile_tags=(
                    "blue_jacket", "cropped_jacket", "white_skirt",
                ),
                appearance_profile_tags=("pink_hair", "grey_eyes"),
                source_outfit_profiles=(
                    (
                        "千早爱音", "chihaya_anon",
                        ("blue_jacket", "cropped_jacket", "white_skirt"),
                        "stage",
                    ),
                ),
                character_appearance_profiles=(
                    (("千早爱音",), "chihaya_anon", ("pink_hair", "grey_eyes")),
                ),
                status="profile_cache",
            )

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            target = next(anchor for anchor in anchors if anchor.role == "target_character")
            assert target.description.endswith("stage outfit variant")
            return SemanticLookupResult(
                confirmed_tags=("chihaya_anon", "bang_dream!"),
                anchor_tags=((target.anchor_id, "chihaya_anon"),),
                # An unqualified/default compatibility row must not beat the
                # qualifier-bearing stage cache selected above.
                character_profiles=(
                    (
                        target.anchor_id, "chihaya_anon",
                        ("haneoka_school_uniform", "green_skirt"),
                        ("pink_hair", "grey_eyes"),
                    ),
                ),
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
    pipeline = PromptPipeline(
        context=context,
        config={"debug_prompt_enabled": True},
        logger=_Logger(),
        danbooru_resolver=_Resolver(),
        researcher=_Researcher(),
        get_bool=lambda key, default: key == "debug_prompt_enabled" or default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )
    event = type("_Event", (), {"unified_msg_origin": "session"})()

    result = asyncio.run(pipeline.build(event, "千早爱音穿着演出服"))

    outfit = result.summary["semantic_character_outfits"][0]
    assert result.summary["requested_outfit_mode"] == "stage_profile"
    assert result.summary["wardrobe_source"] == "explicit_stage_profile"
    assert result.summary["explicit_wardrobe_evidence"] is True
    assert outfit["wardrobe_kind"] == "stage_profile"
    assert outfit["resolution_state"] == "resolved"
    assert outfit["effective_tags"] == [
        "blue_jacket", "cropped_jacket", "white_skirt"
    ]
    assert "explicit stage wardrobe = blue_jacket" in context.calls[1]["prompt"]
    assert "haneoka_school_uniform" not in context.calls[1]["prompt"]


def test_creative_outfit_keeps_profile_identity_without_default_clothes() -> None:
    semantic_json = (
        '{"anchors":[{"id":"anon","role":"target_character",'
        '"group":"character","source_text":"千早爱音","description":"Anon",'
        '"candidates":["chihaya_anon"]}],"character_plans":['
        '{"target_anchor_id":"anon","wardrobe":{"kind":"default_profile"},'
        '"directives":[]}]}'
    )
    writer = (
        "{Count: 1girl, solo}\n"
        "{Characters: rayne}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: rayne has long light blue hair and blue eyes}\n"
        "{Details: rayne wears Haneoka school uniform, "
        "an oversized pink hoodie and denim shorts}\n"
        "{Tags: blue_hair, yellow_eyes, haneoka_school_uniform, "
        "oversized_clothes, pink_hoodie, denim_shorts, standing, "
        "background_mode_default_portrait}\n"
        "{Nltags: rayne smiles while the wind moves her pink hair. "
        "She wears a relaxed private outfit.}"
    )

    class _Response:
        def __init__(self, text):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [semantic_json, writer]

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

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("chihaya_anon",),
                outfit_profile_tags=("haneoka_school_uniform",),
                appearance_profile_tags=("pink_hair", "grey_eyes"),
                source_outfit_profiles=(
                    (
                        "千早爱音",
                        "chihaya_anon",
                        ("haneoka_school_uniform",),
                        "default",
                    ),
                ),
                status="profile_cache",
            )

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            return SemanticLookupResult(
                confirmed_tags=("bang_dream!",),
                outfit_profile_tags=("haneoka_school_uniform",),
                anchor_tags=(("anon", "chihaya_anon"),),
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

    event = type("_Event", (), {"unified_msg_origin": "session"})()
    prompts = (
        "千早爱音穿居家私服站立",
        "千早爱音被魅惑催眠了，她的胸部变大但身材依旧苗条",
    )
    for index, prompt in enumerate(prompts):
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
        result = asyncio.run(pipeline.build(event, prompt))

        lowered = result.final_prompt.lower()
        if index == 0:
            # Explicit private-wear intent makes the cached uniform stale.
            assert "haneoka" not in lowered, prompt
            assert result.summary["wardrobe_resolution_states"] == {
                "千早爱音": "resolved"
            }
        else:
            # An implicit scene-adaptive fallback does not overrule LLM2's
            # interpretation of the full raw request.
            assert "haneoka" in lowered, prompt
            assert result.summary["wardrobe_resolution_states"] == {
                "千早爱音": "unspecified"
            }
        assert "oversized pink hoodie" in lowered
        assert "denim shorts" in lowered
        assert "chihaya anon has pink hair, grey eyes" in lowered
        assert "long light blue hair" not in lowered
        assert "blue eyes" not in lowered
        assert "yellow eyes" not in lowered
        assert "wind moves her pink hair" in lowered
        assert result.summary["semantic_character_outfits"][0][
            "wardrobe_kind"
        ] == ("creative_fallback" if index == 0 else "default_reference")


def test_structured_pipeline_filters_conflicts_without_injecting_wardrobes() -> None:
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

    assert "togawa sakiko wears red shirt" not in result.final_prompt
    assert "togawa sakiko is bottomless" in result.final_prompt
    assert "chihaya anon wears haneoka school uniform" not in result.final_prompt
    assert "chihaya anon is bottomless" not in result.final_prompt
    assert "wakaba mutsumi wears a black veil" in result.final_prompt
    assert "lace-trimmed black thighhighs" in result.final_prompt
    assert "Oblivionis" in result.final_prompt
    assert "is nude" in result.final_prompt
    nltags = result.final_prompt.split("Nltags:", 1)[1]
    assert "togawa sakiko wears red shirt" not in nltags
    assert "togawa sakiko is bottomless" not in nltags
    assert "chihaya anon wears haneoka school uniform" not in nltags
    assert "wakaba mutsumi wears a black veil" not in nltags
    assert "is nude" not in nltags
    assert result.summary["global_outfit_reinforcement_tags"] == []
    assert result.summary["structured_character_count"] == 3
    assert "- 丰川祥子: CREATIVE WARDROBE" not in context.calls[1]["prompt"]
    assert "- 千早爱音: CREATIVE WARDROBE" not in context.calls[1]["prompt"]
    assert "- 若叶睦: verified clothing anchors" in context.calls[1]["prompt"]


def test_user_visible_prompt_keeps_writer_outfit_without_profile_injection() -> None:
    user_prompt = (
        "丰川祥子穿着羽丘夏季校服坐在椅子上，千早爱音是扶她，"
        "千早爱音穿着羽丘夏季校服站在丰川祥子身边。"
        "千早爱音的肉棒在裙子底下勃起着。背景是有壁炉的学生会室，"
        "低角度动态构图"
    )
    semantic_json = (
        '{"characters":['
        '{"name":"丰川祥子","tag":"togawa_sakiko",'
        '"wardrobe":{"kind":"named_outfit","source":"羽丘夏季校服"}},'
        '{"name":"千早爱音","tag":"chihaya_anon",'
        '"wardrobe":{"kind":"named_outfit","source":"羽丘夏季校服"}}],'
        '"lookups":[{"text":"羽丘夏季校服","role":"outfit_source",'
        '"tag":"haneoka_school_uniform"}]}'
    )
    writer = (
        "{Count: 2girls, futa with female}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: togawa_sakiko has blue hair; chihaya_anon has pink hair}\n"
        "{Details: togawa_sakiko wears a Haneoka summer uniform and sits on a chair; "
        "chihaya_anon wears a Haneoka summer uniform and stands beside her}\n"
        "{Tags: sitting, standing, chair, skirt_lift, shirt_tug, elbow_grab, "
        "fireplace, low_angle, dramatic_perspective, red_hair_ornament, "
        "blue_eyes_symbol, background_mode_explicit_scene}\n"
        "{Nltags: Wind moves Sakiko's blue hair while Anon is a futanari and "
        "has an erection beneath her skirt beside the fireplace.}"
    )
    uniform_tags = (
        "school_uniform",
        "summer_uniform",
        "shirt",
        "haneoka_school_uniform",
        "brown_sweater_vest",
        "diagonal-striped_clothes",
        "diagonal-striped_necktie",
        "green_skirt",
        "necktie",
        "pleated_skirt",
        "striped_clothes",
        "striped_necktie_sweater_vest",
    )

    class _Response:
        def __init__(self, text):
            self.completion_text = text

    class _Context:
        def __init__(self):
            self.outputs = [semantic_json, writer]
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
            return ("togawa_sakiko", "chihaya_anon")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            assert next(anchor for anchor in anchors if anchor.anchor_id == "lookup_1").role == "outfit"
            return SemanticLookupResult(
                confirmed_tags=(
                    "togawa_sakiko",
                    "chihaya_anon",
                    "haneoka_school_uniform",
                    "bang_dream!",
                ),
                named_outfit_tags=("haneoka_school_uniform",),
                missing_descriptions=(
                    "千早爱音的肉棒在裙子底下勃起着",
                    "千早爱音是扶她",
                ),
                anchors=anchors,
                anchor_tags=(
                    ("target_1", "togawa_sakiko"),
                    ("target_2", "chihaya_anon"),
                    ("lookup_1", "haneoka_school_uniform"),
                ),
                anchor_outfit_profiles=(
                    ("lookup_1", "haneoka_school_uniform", uniform_tags, "summer"),
                ),
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(text=llm_content)

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

    result = asyncio.run(pipeline.build(event, user_prompt))

    assert len(context.calls) == 2
    llm2_prompt = context.calls[1]["prompt"]
    assert llm2_prompt.count("brown_sweater_vest") >= 2
    assert llm2_prompt.count("green_skirt") >= 2
    assert all(
        set(outfit["effective_tags"]) == set(uniform_tags)
        for outfit in result.summary["semantic_character_outfits"]
    ), result.summary["semantic_character_outfits"]
    final_lower = result.final_prompt.lower()
    assert "brown sweater vest" not in final_lower
    assert "green skirt" not in final_lower
    assert final_lower.count("haneoka summer uniform") >= 2
    assert result.summary["global_outfit_reinforcement_tags"] == []
    nltags = result.final_prompt.split("Nltags:", 1)[1]
    assert "千早爱音" not in nltags
    assert "裙子底下" not in nltags
    assert "futanari" in nltags.lower()
    assert "erection beneath her skirt" in nltags.lower()
    assert "wind moves sakiko's blue hair" in nltags.lower()
    for tag in (
        "skirt lift",
        "shirt tug",
        "elbow grab",
        "fireplace",
        "low angle",
        "dramatic perspective",
        "red hair ornament",
        "blue eyes symbol",
    ):
        assert tag in final_lower


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
            prompt = kwargs.get("prompt", "")
            response = _Response()
            if "List each visible character" not in prompt:
                if "面具没有戴" in prompt:
                    response.completion_text = (
                        "{Count: 1girl, solo}\n"
                        "{Characters: togawa_sakiko}\n"
                        "{Copyright: bang_dream!}\n"
                        "{Identity: togawa_sakiko has blue hair and yellow eyes}\n"
                        "{Details: togawa_sakiko wears a black mask and black boots; "
                        "she is holding a masquerade mask}\n"
                        "{Tags: black_mask, masquerade_mask, holding_mask, black_boots, "
                        "standing, background_mode_default_portrait}\n"
                        "{Nltags: Sakiko holds a black masquerade mask while wearing "
                        "black boots. Her removed boots lie beside her.}"
                    )
                elif "上衣是粉色的" in prompt:
                    response.completion_text = (
                        "{Count: 1girl, solo}\n"
                        "{Characters: togawa_sakiko}\n"
                        "{Copyright: bang_dream!}\n"
                        "{Identity: togawa_sakiko has blue hair and yellow eyes}\n"
                        "{Details: togawa_sakiko wears a pink shirt and black skirt}\n"
                        "{Tags: blue_jacket, standing, "
                        "background_mode_default_portrait}\n"
                        "{Nltags: togawa_sakiko wears a pink shirt.}"
                    )
                elif "没穿裙子" in prompt:
                    response.completion_text = (
                        "{Count: 1girl, solo}\n"
                        "{Characters: togawa_sakiko}\n"
                        "{Copyright: bang_dream!}\n"
                        "{Identity: togawa_sakiko has blue hair and yellow eyes}\n"
                        "{Details: togawa_sakiko wears a red shirt and black pantyhose}\n"
                        "{Tags: blue_jacket, standing, "
                        "background_mode_default_portrait}\n"
                        "{Nltags: togawa_sakiko wears a red shirt without a skirt.}"
                    )
                return response
            if "面具没有戴" in prompt:
                response.completion_text = (
                    '{"characters":[{"name":"丰川祥子","aliases":[],'
                    '"clothing":"oblivionis的衣服","clothing_source":"oblivionis",'
                    '"clothing_changes":[{"operation":"removed",'
                    '"slots":["mask"],"source_text":"她的面具没有戴"},'
                    '{"operation":"removed","slots":["boots"],'
                    '"source_text":"丰川祥子的靴子被脱了下来丢在一边"}],'
                    '"appearance_changes":[]}]}'
                )
            elif "上衣是粉色的" in prompt:
                response.completion_text = (
                    '{"characters":[{"name":"丰川祥子",'
                    '"clothing":"oblivionis的衣服","clothing_source":"oblivionis",'
                    '"clothing_changes":[{"operation":"replace_color",'
                    '"slots":["upper_body.primary"],"color":"pink",'
                    '"source_text":"但上衣是粉色的"}],"appearance_changes":[]}]}'
                )
            elif "oblivionis" in prompt:
                response.completion_text = (
                    '{"characters":[{"name":"丰川祥子",'
                    '"clothing":"oblivionis的衣服","clothing_source":"oblivionis",'
                    '"clothing_changes":[{"operation":"remove",'
                    '"slots":["lower_body.skirt"],"color":"",'
                    '"source_text":"丰川祥子下半身没穿裙子"}],'
                    '"appearance_changes":[]}]}'
                )
            else:
                response.completion_text = (
                    '{"characters":[{"name":"丰川祥子","clothing":null,'
                    '"clothing_source":null,"clothing_changes":[{"operation":"remove",'
                    '"slots":["lower_body.skirt"],"color":"",'
                    '"source_text":"丰川祥子下半身没穿裙子"}],'
                    '"appearance_changes":[]}]}'
                )
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

        def cached_outfit_source(self, source):
            assert source == "oblivionis"
            return SemanticLookupResult(
                confirmed_tags=("oblivionis_(bang_dream!)", "bang_dream!"),
                outfit_source_tags=("oblivionis_(bang_dream!)",),
                outfit_profile_tags=(
                    "red_shirt",
                    "black_corset",
                    "black_skirt",
                    "mask",
                    "black_mask",
                    "masquerade_mask",
                    "black_pantyhose",
                    "black_boots",
                ),
                status="profile_cache",
            )

        def configured_character_anchors_for_prompt(self, prompt):
            if "丰川祥子" not in prompt:
                return ()
            return (
                SemanticAnchor(
                    "configured_sakiko", "target_character", "character",
                    "丰川祥子", "丰川祥子", ("togawa_sakiko",),
                ),
            )

        def prefer_cached_character_anchors(self, anchors):
            return tuple(
                replace(
                    anchor,
                    candidates=(
                        ("togawa_sakiko",)
                        if anchor.role == "target_character"
                        else ("oblivionis_(bang_dream!)",)
                        if anchor.role == "outfit_source"
                        else anchor.candidates
                    ),
                )
                for anchor in anchors
            )

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            target = next(anchor for anchor in anchors if anchor.role == "target_character")
            source = next(
                (anchor for anchor in anchors if anchor.role == "outfit_source"),
                None,
            )
            source_tags = (
                "red_shirt", "black_corset", "black_skirt", "mask",
                "black_mask", "masquerade_mask", "black_pantyhose",
                "black_boots",
            )
            anchor_tags = [(target.anchor_id, "togawa_sakiko")]
            source_profiles = ()
            outfit_sources = ()
            if source is not None:
                anchor_tags.append((source.anchor_id, "oblivionis_(bang_dream!)"))
                source_profiles = ((
                    source.source_text,
                    "oblivionis_(bang_dream!)",
                    source_tags,
                    "default",
                ),)
                outfit_sources = ("oblivionis_(bang_dream!)",)
            return SemanticLookupResult(
                confirmed_tags=("togawa_sakiko", "bang_dream!", *outfit_sources),
                outfit_source_tags=outfit_sources,
                outfit_profile_tags=source_tags,
                character_profiles=((
                    target.anchor_id,
                    "togawa_sakiko",
                    source_tags,
                    ("blue_hair", "yellow_eyes"),
                ),),
                source_outfit_profiles=source_profiles,
                anchor_tags=tuple(anchor_tags),
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
        pipeline.build(
            event,
            "丰川祥子穿着oblivionis的衣服，但上衣是粉色的",
        )
    )

    assert "pink shirt" in result.final_prompt
    assert "red shirt" not in result.final_prompt
    assert "blue jacket" in result.final_prompt
    assert "black skirt" in result.final_prompt
    assert "oblivionis (bang dream!)" not in result.final_prompt
    assert result.summary["outfit_transfer"] is False
    assert result.summary["outfit_removed_tags"] == ["red_shirt"]
    assert result.summary["outfit_added_tags"] == ["pink_shirt"]
    assert result.summary["outfit_source_anchor_emitted"] is False
    assert "pink_shirt" in context.calls[1]["prompt"]
    assert "不得恢复明确删除或替换的衣物" in context.calls[1]["prompt"]

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
    assert "without a skirt" in no_skirt.final_prompt
    assert "oblivionis (bang dream!)" not in no_skirt.final_prompt
    assert no_skirt.summary["outfit_removed_tags"] == ["black_skirt"]

    removed_accessories_context = _Context()
    removed_accessories_pipeline = PromptPipeline(
        context=removed_accessories_context,
        config={"debug_prompt_enabled": True},
        logger=_Logger(),
        danbooru_resolver=_Resolver(),
        researcher=_Researcher(),
        get_bool=lambda key, default: key == "debug_prompt_enabled" or default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=_shorten,
    )
    removed_accessories = asyncio.run(
        removed_accessories_pipeline.build(
            event,
            "丰川祥子穿着oblivionis的衣服，她的面具没有戴，"
            "丰川祥子的靴子被脱了下来丢在一边",
        )
    )

    llm2_prompt = removed_accessories_context.calls[1]["prompt"]
    assert "remove face_accessory.mask" in llm2_prompt
    assert "user evidence: 她的面具没有戴" in llm2_prompt
    assert "remove footwear" in llm2_prompt
    assert "user evidence: 丰川祥子的靴子被脱了下来丢在一边" in llm2_prompt
    assert set(removed_accessories.summary["outfit_removed_tags"]) == {
        "mask", "black_mask", "masquerade_mask", "black_boots"
    }
    removed_final = removed_accessories.final_prompt.lower()
    assert "black mask" not in removed_final
    assert "masquerade mask" not in removed_final
    assert "holding mask" not in removed_final
    assert "wearing black boots" not in removed_final
    assert "removed boots lie beside her" in removed_final

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
    assert "without a skirt" in standalone.final_prompt
    assert standalone.summary["outfit_transfer"] is False


def test_explicit_garment_terms_survive_outfit_transfer_allowlist() -> None:
    class _Response:
        completion_text = (
            "{Count: 1girl, solo}\n"
            "{Characters: wakaba_mutsumi}\n"
            "{Copyright: bang_dream!}\n"
            "{Identity: wakaba_mutsumi has green hair and green eyes}\n"
            "{Details: wakaba_mutsumi wears a ballet dress and white pantyhose}\n"
            "{Tags: black_dress, black_skirt, black_jacket, standing, "
            "background_mode_default_portrait}\n"
            "{Nltags: wakaba_mutsumi wears a ballet dress and white pantyhose.}"
        )

    class _Context:
        def __init__(self):
            self.calls = []

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            if "List each visible character" not in kwargs.get("prompt", ""):
                return _Response()
            response = _Response()
            response.completion_text = (
                '{"characters":[{"name":"若叶睦",'
                '"clothing":"mortis的衣服、芭蕾舞裙和白丝袜",'
                '"clothing_source":"mortis","clothing_changes":[],'
                '"appearance_changes":[]}]}'
            )
            return response

    class _Plan:
        use_web_search = False
        use_deep_thinking = False
        search_reason = ""
        thinking_reason = ""

    class _Researcher:
        def plan(self, _prompt):
            return _Plan()

    term_resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        config={
            "danbooru_term_mappings": [
                "芭蕾舞裙=tutu",
                "白丝袜=white_pantyhose",
            ]
        },
    )

    class _Resolver:
        def required_core_tags_for_prompt(self, _prompt):
            return ("wakaba_mutsumi", "bang_dream!")

        def required_profile_tags_for_prompt(self, _prompt):
            return ()

        def profile_hints_for_prompt(self, _prompt):
            return {}

        def cached_term_mappings_for_prompt(self, prompt):
            return term_resolver.cached_term_mappings_for_prompt(prompt)

        def cached_outfit_source(self, source):
            assert source == "mortis"
            return SemanticLookupResult(
                confirmed_tags=("mortis_(bang_dream!)", "bang_dream!"),
                outfit_source_tags=("mortis_(bang_dream!)",),
                outfit_profile_tags=(
                    "black_dress",
                    "black_skirt",
                    "black_jacket",
                    "black_pantyhose",
                ),
                status="profile_cache",
            )

        def configured_character_anchors_for_prompt(self, prompt):
            return (
                SemanticAnchor(
                    "configured_mutsumi", "target_character", "character",
                    "若叶睦", "若叶睦", ("wakaba_mutsumi",),
                ),
            ) if "若叶睦" in prompt else ()

        def prefer_cached_character_anchors(self, anchors):
            return tuple(
                replace(
                    anchor,
                    candidates=(
                        ("wakaba_mutsumi",)
                        if anchor.role == "target_character"
                        else ("mortis_(bang_dream!)",)
                        if anchor.role == "outfit_source"
                        else anchor.candidates
                    ),
                )
                for anchor in anchors
            )

        def semantic_lookup_available(self):
            return True

        async def resolve_semantic_anchors(self, anchors):
            target = next(anchor for anchor in anchors if anchor.role == "target_character")
            source = next(anchor for anchor in anchors if anchor.role == "outfit_source")
            outfit = (
                "black_dress", "black_skirt", "black_jacket", "black_pantyhose",
            )
            return SemanticLookupResult(
                confirmed_tags=(
                    "wakaba_mutsumi", "mortis_(bang_dream!)", "bang_dream!",
                ),
                outfit_source_tags=("mortis_(bang_dream!)",),
                outfit_profile_tags=outfit,
                character_profiles=((
                    target.anchor_id, "wakaba_mutsumi", (),
                    ("green_hair", "green_eyes"),
                ),),
                source_outfit_profiles=((
                    source.source_text, "mortis_(bang_dream!)", outfit, "default",
                ),),
                anchor_tags=(
                    (target.anchor_id, "wakaba_mutsumi"),
                    (source.anchor_id, "mortis_(bang_dream!)"),
                ),
                anchors=anchors,
                status="resolved",
            )

        async def resolve_detailed(self, *, llm_content, **_kwargs):
            return DanbooruResolveOutcome(text=llm_content, status="not_requested")

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
        pipeline.build(event, "若叶睦穿着mortis的衣服、芭蕾舞裙和白丝袜")
    )

    assert result.summary["outfit_transfer"] is False
    assert result.summary["semantic_character_outfits"][0]["wardrobe_kind"] == "outfit_source"
    assert result.summary["danbooru_explicit_term_tags"] == [
        "tutu",
        "white_pantyhose",
    ]
    assert "tutu" in result.summary["required_core_tags"]
    assert "white_pantyhose" in result.summary["required_core_tags"]
    assert "tutu" in result.final_prompt
    assert "white pantyhose" in result.final_prompt


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
    assert "pleated skirt" in result.final_prompt
    assert "white shirt" in result.final_prompt
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
    assert r"correct name \(example work\)" in result.final_prompt
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


def test_structured_photo_prefix_is_attributed_without_loss() -> None:
    _, _, characters, _, nltags = extract_structured_prompt(
        "{Count: 1girl}\n"
        "{Characters: hatsune_miku}\n"
        "{Copyright: vocaloid}\n"
        "{Identity: the wet photograph shows hatsune_miku with aqua hair}\n"
        "{Details: photo one shows hatsune_miku walking happily; "
        "in photo two she is hypnotized; in photo three she crouches}\n"
        "{Tags: multiple views, wet photos}\n"
        "{Nltags: Three photographs overlap.}"
    )

    assert len(characters) == 1
    assert characters[0].identity_tags == (
        "the wet photograph shows hatsune_miku with aqua hair"
    )
    assert characters[0].detail_tags == (
        "photo one shows hatsune_miku walking happily; "
        "in photo two she is hypnotized; in photo three she crouches"
    )
    assert nltags == "Three photographs overlap."


def test_extract_structured_prompt_inherits_continuations_per_character() -> None:
    _, _, characters, _, _ = extract_structured_prompt(
        "{Count: 2girls}\n"
        "{Characters: hatsune_miku, megurine_luka}\n"
        "{Copyright: vocaloid}\n"
        "{Identity: hatsune_miku has aqua hair; her hair is very long; "
        "megurine_luka has pink hair; hers falls over one shoulder}\n"
        "{Details: photo one shows hatsune_miku walking; then she is hypnotized; "
        "megurine_luka watches her; afterward she reaches out}\n"
        "{Tags: street}\n"
        "{Nltags: They share one scene.}"
    )

    assert characters[0].identity_tags.endswith("her hair is very long")
    assert characters[0].detail_tags.endswith("then she is hypnotized")
    assert characters[1].identity_tags.endswith("hers falls over one shoulder")
    assert characters[1].detail_tags.endswith("afterward she reaches out")


def test_extract_structured_prompt_preserves_unscoped_text_in_nltags() -> None:
    _, _, characters, _, nltags = extract_structured_prompt(
        "{Count: 1girl}\n"
        "{Characters: hatsune_miku}\n"
        "{Copyright: vocaloid}\n"
        "{Identity: long aqua hair with rainwater on every strand}\n"
        "{Details: three photos tell a continuous story}\n"
        "{Tags: multiple views}\n"
        "{Nltags: The photographs are wet.}"
    )

    assert len(characters) == 1
    assert characters[0].identity_tags == ""
    assert characters[0].detail_tags == ""
    assert nltags == (
        "The photographs are wet.; long aqua hair with rainwater on every strand; "
        "three photos tell a continuous story"
    )


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


def test_extract_structured_prompt_accepts_numeric_count_and_pipe_scopes() -> None:
    roster_tags, _, characters, _, _ = extract_structured_prompt(
        "{Count: 2}\n"
        "{Characters: togawa_sakiko, chihaya_anon}\n"
        "{Copyright:}\n"
        "{Identity: togawa_sakiko: blue_hair | chihaya_anon: pink_hair}\n"
        "{Details: togawa_sakiko: detached_sleeves | chihaya_anon: grey_dress}\n"
        "{Tags: standing, white background}\n"
        "{Nltags: They stand together.}\n"
        "background_mode_default_portrait"
    )

    assert roster_tags == ("2people",)
    assert [item.name for item in characters] == ["togawa_sakiko", "chihaya_anon"]
    assert characters[0].detail_tags == "togawa_sakiko: detached_sleeves"
    assert characters[1].detail_tags == "chihaya_anon: grey_dress"


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


def test_structured_count_validation_adds_mixed_gender_counts() -> None:
    assert structured_count_tags_match_roster(("1girl", "1boy"), 2)
    assert structured_count_tags_match_roster(
        ("2girls", "1boy", "futa with female"), 3
    )
    assert not structured_count_tags_match_roster(("1girl", "1boy"), 3)
    assert not structured_count_tags_match_roster(("2girls", "2people"), 2)


def test_unrequested_futa_and_male_genitals_are_removed_from_structured_output() -> None:
    parsed = _parse_structured_prompt(
        "{Count: 1girl, futanari}\n"
        "{Characters: chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: chihaya_anon is a futanari girl with pink hair}\n"
        "{Details: chihaya_anon lies on a bed, masturbating, strokes her erect penis}\n"
        "{Tags: solo, masturbation, pussy, penis, erection, hand_on_penis}\n"
        "{Nltags: chihaya_anon fingers her pussy, then strokes her erect penis.}"
    )

    filtered, removed = enforce_structured_sexual_trait_authority(
        parsed,
        "千早爱音躺在床上抠着小穴自慰。千早爱音不是扶她",
    )

    assert filtered.roster_tags == ("1girl",)
    combined = " ".join(
        (
            filtered.characters[0].identity_tags,
            filtered.characters[0].detail_tags,
            filtered.scene,
            filtered.nltags,
        )
    ).lower()
    assert "futanari" not in combined
    assert "penis" not in combined
    assert "erection" not in combined
    assert "masturbat" in combined
    assert "pussy" in combined
    assert removed


def test_explicit_futa_request_preserves_structured_futa_traits() -> None:
    parsed = _parse_structured_prompt(
        "{Count: 1girl, futanari}\n"
        "{Characters: chihaya_anon}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: chihaya_anon is a futanari girl with pink hair}\n"
        "{Details: chihaya_anon has an erect penis and masturbates}\n"
        "{Tags: solo, futanari, penis, erection}\n"
        "{Nltags: chihaya_anon strokes her erect penis.}"
    )

    filtered, removed = enforce_structured_sexual_trait_authority(
        parsed, "千早爱音长出了扶她肉棒并自慰"
    )

    assert filtered == parsed
    assert removed == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("千早爱音是扶她", True),
        ("千早爱音不是扶她", False),
        ("tag不要出现扶她", False),
        ("futanari chihaya anon", True),
        ("not futanari", False),
    ),
)
def test_positive_futa_request_requires_non_negated_user_evidence(
    text: str, expected: bool
) -> None:
    assert has_positive_futa_request(text) is expected


def test_llm_prompt_uses_configured_seven_field_template_without_english_suffix() -> None:
    prompt = build_llm_prompt(
        "一张白色的大床，穿红黑礼服的丰川祥子抱着穿黑色风衣的千早爱音",
        original_theme="一张白色的大床，穿红黑礼服的丰川祥子抱着穿黑色风衣的千早爱音",
    )

    assert "只输出七个单行花括号字段" in prompt
    assert "Count` 统计画面中所有可见人物" in prompt
    assert "Characters` 只列能确认 canonical 名称的角色" in prompt
    assert "禁止反复核算或自我怀疑" in prompt
    assert "{Copyright:" in prompt
    assert "Return exactly seven single-line brace blocks" not in prompt
    assert "List the actor/holder/supporter before the recipient" not in prompt
    assert "futanari" not in prompt
    assert "futa with female" not in prompt


def test_llm_prompt_adds_futa_count_policy_only_for_positive_request() -> None:
    positive = build_llm_prompt("千早爱音是扶她", original_theme="千早爱音是扶她")
    negated = build_llm_prompt(
        "千早爱音不是扶她", original_theme="千早爱音不是扶她"
    )

    assert "1girl, futanari" in positive
    assert "futa with female" in positive
    assert "1girl, futanari" not in negated
    assert "futa with female" not in negated


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


def test_anonymous_people_structured_prompt_accepts_count_without_characters() -> None:
    roster_tags, copyright_tags, characters, scene, nltags = extract_structured_prompt(
        "{Count: 2girls}\n"
        "{Characters:}\n"
        "{Copyright:}\n"
        "{Identity:}\n"
        "{Details:}\n"
        "{Tags: close-up, two pairs of feet, black pantyhose feet, "
        "white pantyhose feet, blurred background}\n"
        "{Nltags: The image focuses solely on two pairs of feet while their "
        "anonymous owners are blurred in the background.}"
    )

    assert roster_tags == ("2girls",)
    assert copyright_tags == ()
    assert characters == ()
    assert "two pairs of feet" in scene
    assert "anonymous owners" in nltags


def test_anonymous_people_pipeline_keeps_tags_without_format_retry() -> None:
    class _Response:
        completion_text = (
            "{Count: 2girls}\n"
            "{Characters:}\n"
            "{Copyright:}\n"
            "{Identity:}\n"
            "{Details:}\n"
            "{Tags: close-up, two pairs of feet, black pantyhose feet, "
            "white pantyhose feet, blurred background}\n"
            "{Nltags: The image focuses solely on two pairs of feet while their "
            "owners are blurred in the background.}\n"
            "background_mode_explicit_scene"
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
                "valid anonymous-people structure must bypass character resolution"
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
    result = asyncio.run(
        pipeline.build(
            event,
            "2girls。一对收拢的黑丝脚和一对收拢的白丝脚，主人在背景中模糊可见",
        )
    )

    assert "2girls" in result.final_prompt
    assert "black pantyhose feet" in result.final_prompt
    assert "white pantyhose feet" in result.final_prompt
    assert "Nltags: The image focuses solely on two pairs of feet" in result.final_prompt
    assert "收拢" not in result.final_prompt
    assert "full body" not in result.final_prompt
    assert "centered" not in result.final_prompt
    assert "simple background" not in result.final_prompt
    assert "white background" not in result.final_prompt
    assert result.summary["structured_prompt_mode"] is True
    assert result.summary["structured_character_mode"] is False
    assert result.summary["prompt_llm_attempt_count"] == 1
    assert result.summary["structured_initial_validation_errors"] == []
    assert result.summary["background_mode"] == "explicit_scene"
    assert result.summary["background_mode_source"] == "llm_marker"


def test_structured_relationship_survives_missing_role_detail() -> None:
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

    assert roster_tags == ("2girls", "yuri")
    assert copyright_tags == ("bang_dream!",)
    assert [character.name for character in characters] == [
        "togawa_sakiko",
        "chihaya_anon",
    ]
    assert characters[0].detail_tags == ""
    assert characters[1].detail_tags == (
        "chihaya_anon wears a black trench coat and lies in togawa_sakiko's arms"
    )
    assert scene == "red and black dress, white bed"
    assert nltags == "Togawa Sakiko holds Chihaya Anon."


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

    assert result.final_prompt == ""
    assert result.summary["skipped_reason"] == "no_chat_provider"
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
        "prompt_llm_attempt_count": 2,
        "prompt_llm_accepted_attempt": "retry",
        "llm_content_tag_count": 72,
        "final_prompt_chars": 360,
    }

    summary = build_strategy_summary(task, prompt_summary)

    assert summary["reference_requested"] is True
    assert summary["reference_applied"] is False
    assert summary["llm_ok"] is False
    assert summary["fixed_character_name"] == "狐莉"
    assert summary["prompt_llm_attempt_count"] == 2
    assert summary["prompt_llm_accepted_attempt"] == "retry"
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
                "prompt_llm_attempt_count": 2,
                "prompt_llm_accepted_attempt": "retry",
                "fixed_character_name": "狐莉",
                "web_search": False,
                "deep_thinking": False,
                "final_prompt_chars": 120,
                "verification": {"passed": True, "retry_count": 0},
            },
            "verification_summary": {"enabled": True},
            "prompt_summary": {
                "structured_initial_validation_errors": [
                    "missing fields: details"
                ],
                "structured_retry_validation_errors": [],
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
        "LLM2：attempts=2 accepted=retry "
        "initial_errors=missing fields: details retry_errors=无" in text
    )
    assert (
        "自检：enabled=True 多人强制=False 角色强制=False passed=True retry=0" in text
    )
    assert "阶段事件：provider=ok，prompt_llm=ok" in text
