from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = PLUGIN_DIR.parents[2]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from astrbot.core.provider.sources.openai_source import (  # noqa: E402
    ProviderOpenAIOfficial,
)
from config_defaults import flatten_config  # noqa: E402
from danbooru_resolver import DanbooruResolver  # noqa: E402
from prompt_pipeline import PromptPipeline  # noqa: E402
from prompt_research import PromptResearcher  # noqa: E402


pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(
        os.getenv("ANIMA_RUN_LIVE_LLM_E2E") != "1",
        reason="set ANIMA_RUN_LIVE_LLM_E2E=1 to call the configured live provider",
    ),
]

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PLUGIN_CONFIG = (
    CORE_DIR / "data" / "config" / "astrbot_plugin_anima_master_config.json"
)
_ASTRBOT_CONFIG = CORE_DIR / "data" / "cmd_config.json"
_PROFILE_CACHE = (
    CORE_DIR
    / "data"
    / "plugin_data"
    / "astrbot_plugin_anima_master"
    / "danbooru_outfit_profiles.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolved_keys(config: dict) -> dict:
    result = dict(config)
    keys = result.get("key", [])
    if not isinstance(keys, list):
        return result
    result["key"] = [
        os.getenv(value[2:-1] if value.startswith("${") else value[1:], "")
        if isinstance(value, str) and value.startswith("$")
        else value
        for value in keys
    ]
    return result


class _LiveContext:
    def __init__(self, provider, provider_id: str, astrbot_config: dict):
        self.provider = provider
        self.provider_id = provider_id
        self.astrbot_config = astrbot_config
        self.calls: list[dict] = []

    async def get_current_chat_provider_id(self, _umo):
        return self.provider_id

    def get_config(self, umo=None):
        return self.astrbot_config

    async def llm_generate(self, **kwargs):
        assert kwargs.get("chat_provider_id") == self.provider_id
        call = {
            "system_prompt": kwargs.get("system_prompt", ""),
            "prompt": kwargs.get("prompt", ""),
        }
        self.calls.append(call)
        forwarded = {
            key: value
            for key, value in kwargs.items()
            if key not in {"chat_provider_id", "contexts"}
        }
        response = await self.provider.text_chat(**forwarded)
        call["response"] = str(getattr(response, "completion_text", "") or "")
        return response


class _Event:
    unified_msg_origin = "live-e2e:anima-prompt"


@pytest.fixture
def live_pipeline(tmp_path_factory):
    if not _PLUGIN_CONFIG.exists() or not _ASTRBOT_CONFIG.exists():
        pytest.skip("current AstrBot runtime configuration is unavailable")

    astrbot_config = _read_json(_ASTRBOT_CONFIG)
    plugin_config = flatten_config(_read_json(_PLUGIN_CONFIG))
    provider_id = (
        str(plugin_config.get("prompt_builder_provider_id") or "").strip()
        or str(
            astrbot_config.get("provider_settings", {}).get(
                "default_provider_id", ""
            )
        ).strip()
    )
    provider_row = next(
        (
            item
            for item in astrbot_config.get("provider", [])
            if item.get("id") == provider_id
        ),
        None,
    )
    if not provider_row:
        pytest.skip(f"configured provider {provider_id!r} is unavailable")
    source_row = next(
        (
            item
            for item in astrbot_config.get("provider_sources", [])
            if item.get("id") == provider_row.get("provider_source_id")
        ),
        {},
    )
    provider_config = _resolved_keys({**source_row, **provider_row, "id": provider_id})
    if provider_config.get("type") != "openai_chat_completion":
        pytest.skip("live harness currently supports AstrBot OpenAI-compatible providers")
    if not any(str(key or "").strip() for key in provider_config.get("key", [])):
        pytest.skip("configured provider has no usable API key")

    provider = ProviderOpenAIOfficial(
        provider_config,
        astrbot_config.get("provider_settings", {}),
    )
    context = _LiveContext(provider, provider_id, astrbot_config)
    logger = logging.getLogger("anima.live_e2e")

    def get_bool(key, default):
        return bool(plugin_config.get(key, default))

    def get_int(key, default):
        try:
            return int(plugin_config.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_float(key, default):
        try:
            return float(plugin_config.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_str(key, default):
        value = plugin_config.get(key, default)
        return default if value is None else str(value)

    temp_profile = tmp_path_factory.mktemp("live_anima_profiles") / "profiles.json"
    if _PROFILE_CACHE.exists():
        shutil.copyfile(_PROFILE_CACHE, temp_profile)
    resolver = DanbooruResolver(
        logger=logger,
        cache={},
        get_bool=get_bool,
        get_int=get_int,
        get_float=get_float,
        get_str=get_str,
        config=plugin_config,
        profile_cache_path=temp_profile,
    )
    researcher = PromptResearcher(
        context=context,
        logger=logger,
        get_bool=get_bool,
        get_int=get_int,
        get_str=get_str,
    )
    pipeline = PromptPipeline(
        context=context,
        config=plugin_config,
        logger=logger,
        danbooru_resolver=resolver,
        researcher=researcher,
        get_bool=get_bool,
        get_int=get_int,
        get_float=get_float,
        get_str=get_str,
        shorten=lambda text, limit=600: str(text)[:limit],
    )
    return pipeline, context


def _build(live_pipeline, prompt: str, *, multi_person: bool = False):
    pipeline, context = live_pipeline

    async def run():
        try:
            result = await pipeline.build(_Event(), prompt, multi_person=multi_person)
            return result, list(context.calls)
        finally:
            await context.provider.terminate()

    return asyncio.run(run())


def _assert_successful_english_prompt(result) -> str:
    assert result.final_prompt, result.summary
    assert not result.summary.get("llm_failed"), result.summary
    assert not _CJK_RE.search(result.final_prompt), result.final_prompt
    return result.final_prompt.lower()


@pytest.mark.parametrize("order", ("alias_first", "alias_last"))
def test_live_cached_alias_keeps_qualified_character_identity(live_pipeline, tmp_path, order):
    names = (
        ("粥祥", "丰川祥子", "千早爱音") if order == "alias_first"
        else ("千早爱音", "丰川祥子", "粥祥")
    )
    prompt = (
        f"3girls，{names[0]}站在左边，{names[1]}坐在中间，{names[2]}站在右边。"
        "三人在室内客厅合影。丰川祥子穿着oblivionis的衣服，粥祥穿默认服装，"
        "千早爱音穿白色连衣裙。粥祥头上有一对角，丰川祥子头上没有角。"
    )
    result, calls = _build(live_pipeline, prompt)
    report = {
        "prompt": prompt, "provider_id": live_pipeline[1].provider_id,
        "calls": calls, "summary": result.summary, "final_prompt": result.final_prompt,
    }
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"cached_alias_{order}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Live identity evidence: {report_path}")

    final = _assert_successful_english_prompt(result)
    canonical = "togawa_sakiko_(master_of_melodia)"
    expected = {"chihaya_anon", "togawa_sakiko", canonical}
    bindings = {row["source"]: row["canonical_tag"]
                for row in result.summary["confirmed_character_bindings"]}
    assert bindings["粥祥"] == canonical
    statuses = result.summary["character_resolution_statuses"]
    assert len(statuses) == 3, statuses
    assert {row["canonical_tag"] for row in statuses} == expected
    assert all(row["status"] == "semantic_confirmed" for row in statuses)
    assert not re.search(r"zhou[ _-]*xiang|togawa[ _]saiko", final)
    # The identity must occupy the primary roster, not just a shared hard tag.
    assert r"togawa sakiko \(master of melodia\)" in final.splitlines()[1]
    assert any(f"粥祥 => {canonical}" in call["prompt"] for call in calls)
    assert any("characters" in call["response"] for call in calls)
    assert any("{Characters:" in call["response"] for call in calls)
    outfits = {row["target"]: row for row in result.summary["semantic_character_outfits"]}
    assert outfits["粥祥"]["effective_tags"]
    assert outfits["丰川祥子"]["wardrobe_kind"] == "outfit_source"


def test_live_shared_named_uniform_is_complete_and_nltags_stays_english(
    live_pipeline,
) -> None:
    result, calls = _build(
        live_pipeline,
        "丰川祥子穿着羽丘夏季校服坐在椅子上，千早爱音是扶她，"
        "千早爱音穿着羽丘夏季校服站在丰川祥子身边。"
        "千早爱音的肉棒在裙子底下勃起着",
    )
    final = _assert_successful_english_prompt(result)
    outfits = result.summary["semantic_character_outfits"]
    writer_call = next(
        call for call in calls if "只输出七个单行花括号字段" in call["prompt"]
    )

    assert len(calls) in {2, 3}, result.summary
    assert len(outfits) == 2, outfits
    assert all(item["resolution_state"] == "resolved" for item in outfits)
    assert all(item["complete_named_profile"] for item in outfits)
    for item in outfits:
        tags = set(item["effective_tags"])
        assert {
            "haneoka_school_uniform",
            "brown_sweater_vest",
            "green_skirt",
            "pleated_skirt",
            "diagonal-striped_necktie",
        } <= tags
    assert writer_call["prompt"].count(
        "named outfit = haneoka_school_uniform"
    ) == 2
    assert writer_call["prompt"].count("brown_sweater_vest") >= 2
    assert writer_call["prompt"].count("green_skirt") >= 2
    assert "haneoka school uniform" in final
    assert "futa" in final
    assert any(word in final for word in ("erect", "erection", "penis", "cock"))


def test_live_haneoka_winter_uniform_reaches_writer_as_generic_named_set(
    live_pipeline,
) -> None:
    result, calls = _build(
        live_pipeline,
        "竖屏，上半身特写。高松灯穿着羽丘冬季校服站在学校走廊里，"
        "微笑着看着自己指间的粉色光晕。不要写looking at viewer。不要改动衣物。",
    )
    final = _assert_successful_english_prompt(result)
    outfit = result.summary["semantic_character_outfits"][0]
    writer_call = next(
        call for call in calls if "只输出七个单行花括号字段" in call["prompt"]
    )
    tags = set(outfit["effective_tags"])

    assert len(calls) == 2, calls
    assert outfit["target"] == "高松灯"
    assert outfit["wardrobe_kind"] == "named_outfit"
    assert outfit["resolution_state"] == "resolved"
    assert outfit["complete_named_profile"] is True
    assert {
        "haneoka_school_uniform",
        "winter_uniform",
        "green_necktie",
    } <= tags
    assert "named outfit = haneoka_school_uniform" in writer_call["prompt"]
    assert "grey_jacket" in writer_call["prompt"]
    assert "white_shirt" in writer_call["prompt"]
    assert "tsukinomori_school_uniform" not in writer_call["prompt"]
    assert "stale cached wardrobes" not in writer_call["prompt"].lower()
    assert "looking at viewer" not in final
    assert "school hallway" in final or "school corridor" in final
    assert any(word in final for word in ("grey jacket", "gray jacket", "winter uniform"))


def test_live_character_stage_profile_and_generic_uniform_remain_separate(
    live_pipeline,
) -> None:
    result, calls = _build(
        live_pipeline,
        "千早爱音穿着演出服站在左边，高松灯穿着羽丘冬季校服站在右边，"
        "两人在学校走廊合影",
    )
    final = _assert_successful_english_prompt(result)
    outfits = {
        item["target"]: item for item in result.summary["semantic_character_outfits"]
    }
    writer_call = next(
        call for call in calls if "只输出七个单行花括号字段" in call["prompt"]
    )

    assert outfits["千早爱音"]["wardrobe_kind"] == "stage_profile"
    assert outfits["千早爱音"]["resolution_state"] == "resolved"
    assert outfits["高松灯"]["wardrobe_kind"] == "named_outfit"
    assert outfits["高松灯"]["resolution_state"] == "resolved"
    assert outfits["高松灯"]["complete_named_profile"] is True
    assert "haneoka_school_uniform" not in set(
        outfits["千早爱音"]["effective_tags"]
    )
    assert "haneoka_school_uniform" in set(outfits["高松灯"]["effective_tags"])
    assert "explicit stage wardrobe" in writer_call["prompt"].lower()
    assert "named outfit = haneoka_school_uniform" in writer_call["prompt"]
    assert "chihaya anon" in final
    assert "takamatsu tomori" in final
    assert "school hallway" in final or "school corridor" in final


def test_live_explicit_swimwear_does_not_restore_default_uniforms(live_pipeline) -> None:
    result, _calls = _build(
        live_pipeline,
        "千早爱音穿着连体式泳装，丰川祥子穿着分体式泳装，"
        "两人在泳池边站在一起",
    )
    final = _assert_successful_english_prompt(result)

    assert "haneoka school uniform" not in final
    assert "tsukinomori school uniform" not in final
    assert any(word in final for word in ("one-piece swimsuit", "one piece swimsuit"))
    assert any(word in final for word in ("two-piece swimsuit", "two piece swimsuit", "bikini"))
    assert "poolside" in final or "swimming pool" in final
    assert "white background" not in final
    assert "simple background" not in final


def test_live_named_uniform_removal_keeps_top_and_removes_skirt(live_pipeline) -> None:
    result, _calls = _build(
        live_pipeline,
        "千早爱音穿着羽丘夏季校服但没穿短裙，站在教室里",
    )
    final = _assert_successful_english_prompt(result)
    outfit = result.summary["semantic_character_outfits"][0]
    tags = set(outfit["effective_tags"])

    assert outfit["resolution_state"] == "resolved"
    assert "brown_sweater_vest" in tags
    assert "shirt" in tags
    assert "green_skirt" not in tags
    assert "pleated_skirt" not in tags
    assert "brown sweater vest" in final
    assert "green skirt" not in final
    assert "pleated skirt" not in final


def test_live_upper_body_framing_crops_hidden_named_uniform_slots(live_pipeline) -> None:
    result, _calls = _build(
        live_pipeline,
        "千早爱音穿着羽丘夏季校服，上半身肖像，只拍到腰部以上",
    )
    final = _assert_successful_english_prompt(result)
    outfit = result.summary["semantic_character_outfits"][0]
    tags = set(outfit["effective_tags"])

    assert "brown_sweater_vest" in tags
    assert "shirt" in tags
    assert "green_skirt" not in tags
    assert "pleated_skirt" not in tags
    assert "green skirt" not in final
    assert "pleated skirt" not in final
    assert any(word in final for word in ("upper body", "waist up", "portrait"))


@pytest.mark.parametrize(
    "wording",
    (
        "丰川祥子在cosplay初音未来",
        "丰川祥子正在cos初音未来",
        "丰川祥子扮成初音未来",
        "丰川祥子穿着初音未来的cosplay服装",
        "丰川祥子身穿初音未来的衣服",
    ),
)
def test_live_cosplay_wordings_bind_source_profile_to_the_wearer(
    live_pipeline, wording: str
) -> None:
    result, calls = _build(
        live_pipeline,
        f"{wording}，千早爱音穿着常服站在她身边",
    )
    final = _assert_successful_english_prompt(result)
    planner_call = next(call for call in calls if "List each visible character" in call["prompt"])
    writer_call = next(
        call for call in calls if "只输出七个单行花括号字段" in call["prompt"]
    )
    outfits = {
        item["target"]: item for item in result.summary["semantic_character_outfits"]
    }

    assert len(planner_call["prompt"]) < 1800
    assert "wardrobe.kind" not in planner_call["prompt"]
    assert "初音未来" in planner_call["response"]
    assert "最终主体必须是固定角色" not in writer_call["prompt"]
    assert not re.search(r"\{Count:\s*\d+\s*\}", writer_call["response"])
    assert "hatsune_miku" in writer_call["prompt"]
    assert outfits["丰川祥子"]["wardrobe_kind"] == "outfit_source"
    assert {
        "sleeveless_shirt",
        "black_skirt",
        "detached_sleeves",
        "aqua_necktie",
    } <= set(outfits["丰川祥子"]["effective_tags"])
    assert outfits["千早爱音"]["wardrobe_kind"] == "casual_profile"
    assert "aqua necktie" in final
    assert "detached sleeves" in final


def test_live_multi_person_route_uses_one_model_call_without_llm1(live_pipeline) -> None:
    result, calls = _build(
        live_pipeline,
        "一个白发红眼狐耳女孩和一个黑发蓝眼男孩并肩牵手站立",
        multi_person=True,
    )
    final = _assert_successful_english_prompt(result)

    assert len(calls) == 1, calls
    assert result.summary["multi_person_mode"] is True
    assert result.summary["multi_person_plan_failed"] is False
    assert "semantic_plan_attempt_count" not in result.summary
    assert "white hair" in final
    assert "red eyes" in final
    assert "fox ears" in final
    assert "black hair" in final
    assert "blue eyes" in final
    assert "holding hands" in final
