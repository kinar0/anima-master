from __future__ import annotations

import asyncio
import hashlib
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


def _build(live_pipeline, prompt: str, *, multi_person: bool = False, canvas_size=None):
    pipeline, context = live_pipeline

    async def run():
        try:
            result = await pipeline.build(
                _Event(), prompt, multi_person=multi_person,
                canvas_size=canvas_size, canvas_size_explicit=canvas_size is not None,
            )
            if report_root := os.getenv("ANIMA_LIVE_REPORT_DIR"):
                report_dir = Path(report_root)
                report_dir.mkdir(parents=True, exist_ok=True)
                identifier = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
                (report_dir / f"build_{identifier}.json").write_text(json.dumps({
                    "prompt": prompt, "calls": context.calls, "summary": result.summary,
                    "final_prompt": result.final_prompt,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            return result, list(context.calls)
        finally:
            await context.provider.terminate()

    return asyncio.run(run())


def _assert_successful_english_prompt(result) -> str:
    assert result.final_prompt, result.summary
    assert not result.summary.get("llm_failed"), result.summary
    assert not _CJK_RE.search(result.final_prompt), result.final_prompt
    return result.final_prompt.lower()


@pytest.mark.parametrize("case", ["mortis_focus", "cut_skirt", "cut_skirt_nai", "other_garment", "two_wearers"])
def test_live_borrowed_outfit_partial_changes(live_pipeline, case):
    """Real planner + writer; no ComfyUI submission or production cache writes."""
    if case == "mortis_focus":
        prompt = "若叶睦穿着mortis的衣服（没有面具），一脸厌恶地抬起一只脚，用黑丝脚底对着镜头。脚底散发着蒸汽。热气腾腾的高筒皮靴放在地上.foot focus"
    elif case.startswith("cut_skirt"):
        prompt = "丰川祥子穿着oblivionis的衣服（没有面具），裙子经过剪裁，只留下身体侧面的部分，露出被黑丝包裹的双腿"
        if case.endswith("nai"):
            prompt = "-r " + prompt
    elif case == "other_garment":
        prompt = "千早爱音穿着oblivionis的衣服（没有面具），上衣经过剪裁，只留下肩膀和袖口部分；其余衣物保持原样，全身站立。"
    else:
        prompt = "千早爱音穿着oblivionis的衣服（没有面具），丰川祥子穿着mortis的衣服（没有帽子）。两人全身并排站立。"
    result, calls = _build(live_pipeline, prompt)
    assert result.final_prompt and not result.summary.get("llm_failed"), result.summary
    from tag_cleaner import split_tags

    def has_source_tag(text, tag):
        key = lambda value: re.sub(r"[\s_\\]+", "", value.lower())
        return key(tag) in {key(part) for part in split_tags(text)}

    if case.endswith("nai"):
        rows = result.summary["nai_characters"]
        assert len(rows) == 1 and rows[0]["name"] == "togawa_sakiko", rows
        actual = rows[0]["prompt"].lower().replace("_", " ")
        assert "black pantyhose" in actual, actual
        assert any(item in actual for item in ("red shirt", "corset")), actual
        assert has_source_tag(rows[0]["prompt"], "oblivionis_(bang_dream!)"), rows
        assert not has_source_tag(result.final_prompt, "oblivionis_(bang_dream!)")
    else:
        actual = _assert_successful_english_prompt(result)
        outfits = result.summary["semantic_character_outfits"]
        assert len(outfits) == (2 if case == "two_wearers" else 1), outfits
        for row in outfits:
            assert not set(row["removed_tags"]).intersection({"black_pantyhose", "black_corset", "red_shirt", "white_shirt"}), row
        if case != "other_garment":
            assert "pantyhose" in actual, actual
        if case == "mortis_focus":
            assert "white shirt" in actual and "red hat" in actual, actual
            assert not any("mask" in tag for tag in result.summary["wardrobe_authority"]["selected_tags"])
        elif case == "cut_skirt":
            assert "black pantyhose" in actual and any(t in actual for t in ("red shirt", "corset")), actual
        elif case == "two_wearers":
            # Check the wearer's actual prose, not an unscoped tag left elsewhere.
            sakiko_clause = next(clause for clause in actual.replace("\n", ";").split(";")
                                 if "togawa sakiko wears" in clause)
            assert "green brooch" in sakiko_clause, sakiko_clause
        for tag in (("oblivionis_(bang_dream!)", "mortis_(bang_dream!)") if case == "two_wearers"
                    else ("mortis_(bang_dream!)",) if case == "mortis_focus"
                    else ("oblivionis_(bang_dream!)",)):
            assert has_source_tag(result.final_prompt, tag), result.final_prompt
    # Real replies and final text are saved by _build, including any bounded repair.
    assert all(not call.get("synthetic_first_stage") for call in calls)


def test_live_borrowed_outfit_repairs_recorded_missing_donor_review(live_pipeline):
    """Replay only the recorded broken writer draft; planner and repair are real."""
    evidence = PLUGIN_DIR / "diagnostics/borrowed-outfit-2026-10-20/reported-evidence.json"
    if not evidence.exists():
        pytest.skip("reported draft evidence is unavailable")
    original = next(row for row in _read_json(evidence)
                    if row["task_id"] == "a30844314446429da512a518dcd26634")
    raw = original["summary"]["nai_character_plan_raw"]
    pipeline, context = live_pipeline
    generate = context.llm_generate
    supplied_draft = False

    async def replay_initial_draft(**kwargs):
        nonlocal supplied_draft
        prompt = kwargs.get("prompt", "")
        if "Design a complete, coherent NAI image" in prompt and not supplied_draft:
            supplied_draft = True
            context.calls.append(dict(prompt=prompt, system_prompt=kwargs.get("system_prompt", ""),
                                      response=raw, replayed_writer_draft=True))
            return type("Response", (), {"completion_text": raw})()
        return await generate(**kwargs)

    context.llm_generate = replay_initial_draft
    result, calls = _build(live_pipeline, original["original_prompt"])
    assert result.final_prompt, result.summary
    assert result.summary["nai_identity_initial_validation_errors"]
    assert not result.summary["nai_identity_retry_validation_errors"]
    assert len(result.summary["nai_characters"]) == 1
    assert any(row["visibility"] == "donor" for row in result.summary["nai_identity_review"])
    assert any(call.get("replayed_writer_draft") for call in calls)
    assert "identity_review" in calls[-1]["response"]


@pytest.mark.parametrize("case", ["shared_clothes_1", "shared_clothes_2", "donor", "quoted", "offscreen"])
def test_live_nai_known_identity_completeness(live_pipeline, tmp_path, case):
    """Neutral scenes only; inspect both actual model replies and final boxes."""
    if case.startswith("shared_clothes"):
        prompt = "-r 丰川祥子和粥祥衣服略显凌乱地坐在沙发上，千早爱音在旁边拿着自拍杆。三人在客厅合影。"
    elif case == "donor":
        prompt = "-r 丰川祥子穿粥祥的衣服坐在沙发上，千早爱音在旁边拿着自拍杆。粥祥只作为服装来源，不出现在画面中。"
    elif case == "quoted":
        prompt = "-r 丰川祥子和千早爱音在客厅合影，墙上的牌子写着粥祥的名字，粥祥本人不出现。"
    else:
        prompt = "-r 丰川祥子和千早爱音坐在沙发上，粥祥在画外说话，不画粥祥。"
    result, calls = _build(live_pipeline, prompt)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"nai_identity_{case}.json").write_text(json.dumps({
        "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result.final_prompt, result.summary
    rows = result.summary["nai_characters"]
    names = [row["name"] for row in rows]
    expected = {"togawa_sakiko", "chihaya_anon"}
    if case.startswith("shared_clothes"):
        expected.add("togawa_sakiko_(master_of_melodia)")
    assert set(names) == expected and len(names) == len(expected), names
    assert len(calls) in {2, 3, 4}, result.summary
    assert all("粥祥" != row["name"] for row in rows)
    for row in rows:
        if row["name"] == "togawa_sakiko_(master_of_melodia)":
            assert "brown_hair" not in row["prompt"]


@pytest.mark.parametrize("source_misclassified", [False, True])
def test_live_nai_writer_recovers_incomplete_first_stage(live_pipeline, tmp_path, source_misclassified):
    pipeline, context = live_pipeline
    first = {"characters": [
        dict(name="丰川祥子", aliases=[], clothing="粥祥衣服" if source_misclassified else None,
             clothing_source="粥祥" if source_misclassified else None,
             clothing_changes=[], appearance_changes=[]),
        dict(name="千早爱音", aliases=[], clothing=None, clothing_source=None,
             clothing_changes=[], appearance_changes=[]),
    ]}

    async def injected_first_stage(**_kwargs):
        raw = json.dumps(first, ensure_ascii=False)
        context.calls.append(dict(synthetic_first_stage=True, response=raw))
        return raw

    pipeline._generate_semantic_plan_with_llm = injected_first_stage
    prompt = "-r 丰川祥子和粥祥衣服略显凌乱地坐在沙发上，千早爱音在旁边拿着自拍杆。"
    result, calls = _build(live_pipeline, prompt)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"nai_identity_injected_source_{source_misclassified}.json").write_text(json.dumps({
        "prompt": prompt, "synthetic_first_stage": True, "real_writer": True,
        "calls": calls, "summary": result.summary, "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result.final_prompt, result.summary
    names = [row["name"] for row in result.summary["nai_characters"]]
    assert set(names) == {"togawa_sakiko", "togawa_sakiko_(master_of_melodia)", "chihaya_anon"}
    assert len(names) == 3
    assert result.summary["nai_bound_local_profile_references"]


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
    # Verify each wearer's actual evidence, not an obsolete context label.
    for wearer in ("丰川祥子", "千早爱音"):
        assert any(line.startswith(f"- {wearer}:") and "haneoka_school_uniform" in line
                   for line in writer_call["prompt"].splitlines())
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


@pytest.mark.parametrize("case,prompt", [
    ("pinning", "-r 千早爱音羽丘校服，被丰川祥子羽丘校服摁在床上，两人对视，丰川祥子手塞进嘴里摸虎牙"),
    ("different_clothes", "-r 千早爱音穿白色连衣裙坐在椅子上，丰川祥子穿羽丘校服站在旁边，祥子用手扶住爱音的肩膀，两人对视"),
    ("explicit_mouth", "-r 千早爱音穿羽丘校服被丰川祥子摁在床上。丰川祥子也穿羽丘校服，把自己的手伸进千早爱音的嘴巴里摸爱音的虎牙。两人对视。"),
    ("reciprocal_hug_dark", "-r 千早爱音穿白色连衣裙，丰川祥子穿羽丘校服，两人互相拥抱。房间昏暗无光，没有任何光源，不要补光。"),
])
def test_live_nai_chinese_and_per_instance_ownership(live_pipeline, tmp_path, case, prompt):
    live_pipeline[0].config["debug_prompt_enabled"] = True
    result, calls = _build(live_pipeline, prompt)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"nai_chinese_ownership_{case}.json"
    report_path.write_text(json.dumps({
        "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Live NAI ownership evidence: {report_path}")
    assert result.final_prompt, result.summary.get("llm_error")
    assert len(calls) in (2, 3)
    assert not result.summary.get("nai_text_retry_validation_errors")
    characters = {row["name"]: row["prompt"] for row in result.summary["nai_characters"]}
    assert set(characters) == {"chihaya_anon", "togawa_sakiko"}
    assert all(_CJK_RE.search(text) for text in characters.values()), characters
    assert _CJK_RE.search(result.final_prompt), result.final_prompt
    # A paired source tag does not license an unqualified self-body assertion.
    from tag_cleaner import split_tags
    assert all(not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", atom)
               for text in characters.values() for atom in split_tags(text)), characters
    if case in {"pinning", "explicit_mouth"}:
        source_atoms = {part.replace("_", " ").casefold() for part in split_tags(characters["togawa_sakiko"])}
        assert not source_atoms & {"hand in mouth", "finger in mouth", "fingers in mouth"}, characters
        assert any(term in characters["chihaya_anon"] for term in ("嘴", "口中", "mouth"))
    else:
        assert all(part.split("#", 1)[-1].replace("_", " ").casefold()
                   not in {"hand in mouth", "finger in mouth", "fingers in mouth"}
                   for text in characters.values() for part in split_tags(text)), characters
    if case == "reciprocal_hug_dark":
        assert all(any(term in text for term in ("拥抱", "hug")) for text in characters.values())
    if case != "reciprocal_hug_dark":
        # "No extra lighting effects" denies embellishment; it is not a light
        # source. Strip only complete negated clauses, preserving any positive
        # lighting in the same output for the assertion below.
        lighting_text = ", ".join(part for part in split_tags(result.final_prompt)
                                  if not re.fullmatch(
                                      r"(?:无|没有|不加|不添加|不要)(?:任何|额外|特殊|默认)"
                                      r"(?:灯光|照明|光照|光线)(?:效果)?[。.!]?|"
                                      r"no (?:extra|special|additional) lighting(?: effects?)?[.!]?",
                                      part.strip(), re.I))
        assert not re.search(r"lighting|sunlight|moonlight|backlight|光照|光线|光影|灯光|柔光|阳光|月光",
                             lighting_text, re.I), result.final_prompt
    else:
        assert not any(term in result.final_prompt.casefold()
                       for term in ("soft lighting", "warm lighting", "sunlight", "lamp", "candle", "柔光", "阳光", "台灯", "烛光"))
    accepted = json.loads(result.summary["nai_character_plan_raw"])
    for raw_row in accepted["characters"]:
        if raw_row["name"] == "chihaya_anon":
            for phrase in ("粉色长发", "灰眼", "灰色眼睛"):
                if phrase in raw_row["prompt"]:
                    assert phrase in characters["chihaya_anon"], (raw_row["prompt"], characters)
    assert "school uniform" not in result.final_prompt.replace("_", " ")
    assert "haneoka school uniform" in characters["togawa_sakiko"].replace("_", " ")
    if case in {"pinning", "explicit_mouth"}:
        assert "haneoka school uniform" in characters["chihaya_anon"].replace("_", " ")
    else:
        assert "haneoka school uniform" not in characters["chihaya_anon"].replace("_", " ")
        assert any(term in characters["chihaya_anon"].replace("_", " ")
                   for term in ("white dress", "白色连衣裙"))


@pytest.mark.parametrize("case,prompt,recipient", [
    ("reported", "-r 千早爱音羽丘校服，被丰川祥子羽丘校服摁在床上，两人对视，千早爱音抗拒表情，丰川祥子一脸戏谑，丰川祥子手塞进千早爱音嘴里摸虎牙", "chihaya_anon"),
    ("reversed", "-r 千早爱音和丰川祥子穿羽丘校服。千早爱音用手指轻触丰川祥子嘴里的虎牙，祥子张嘴让爱音检查。", "togawa_sakiko"),
    ("different_people", "-r 高松灯穿羽丘校服，若叶睦穿月之森校服。灯用手指轻触睦嘴里的虎牙，睦张嘴让灯检查。", "wakaba_mutsumi"),
    ("individual", "-r 千早爱音穿羽丘校服，自己把手指放在自己嘴里碰虎牙，正面上半身构图。", "chihaya_anon"),
    ("explicit_camera", "-r 单幅侧面全身构图。千早爱音和丰川祥子穿羽丘校服，面对面站立，祥子用手指轻触爱音嘴里的虎牙，爱音张嘴。保持侧面全身构图。", "chihaya_anon"),
])
def test_live_nai_focal_contact_and_canonical_tags(live_pipeline, tmp_path, case, prompt, recipient):
    """Preserve raw/final evidence for manual camera and contact review too."""
    from tag_cleaner import normalize_tag_key, split_tags
    from agent_tools.comfyui_workflows import custom_t2i_workflow

    live_pipeline[0].config["debug_prompt_enabled"] = True
    result, calls = _build(live_pipeline, prompt, canvas_size=(1216, 832))
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    graph = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, result.final_prompt, "blurry",
        1216, 832, 28, 6, 123, nai_characters=result.summary.get("nai_characters", []),
    ) if result.final_prompt else {}
    (report_dir / f"contact_{case}.json").write_text(json.dumps({
        "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt, "workflow": graph,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result.final_prompt, result.summary.get("llm_error")
    assert len(calls) == 2
    raw = json.loads(result.summary["nai_character_plan_raw"])
    for rows in (raw["characters"], result.summary["nai_characters"]):
        for row in rows:
            atoms = {normalize_tag_key(p) for p in split_tags(row["prompt"])}
            assert not atoms & {"fangs", "clenched teeth", "closed mouth"}, row
            if row["name"] == recipient:
                assert "fang" in atoms, row
            elif case != "individual":
                assert not atoms & {"hand in mouth", "finger in mouth", "fingers in mouth"}, row
            if case in {"reported", "reversed", "explicit_camera", "individual"} and row["name"] == "chihaya_anon":
                assert not atoms & {"hair ribbon", "hair bow", "yellow bow", "black ribbon"}, row
            assert _CJK_RE.search(row["prompt"]), row
    # Exact matching duplicate suppression is deterministic, regardless of the
    # writer's compliance. The raw reply remains available for separate review.
    for row in result.summary["nai_characters"]:
        atoms = {normalize_tag_key(p) for p in split_tags(row["prompt"])}
        for action in row["interaction_tags"]:
            assert normalize_tag_key(action.partition("#")[2]) not in atoms, row
    if case == "individual":
        assert not result.summary["nai_characters"][0]["interaction_tags"]
    if case == "explicit_camera":
        assert "full body" in {normalize_tag_key(p) for p in split_tags(result.final_prompt)}
    if case == "reported":
        assert "嘴" in raw["global_prompt"]
    selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    for i, row in enumerate(result.summary["nai_characters"], 1):
        assert graph[selector[f"character{i}"][0]]["inputs"]["comfyui_prompt"] == row["prompt"]


@pytest.mark.parametrize("case,prompt", [
    ("joint_reach", "-r 单幅横向构图，三分之四侧视图。千早爱音穿白衬衫站在左边，丰川祥子穿黑色外套站在右边，两人靠近面对面。爱音同时用左手扶住祥子的右肩、右手握住祥子的左手，两项接触都要入镜。"),
    ("seated_support", "-r 单幅侧视全身构图。千早爱音穿白色连衣裙坐在左侧椅子上，丰川祥子穿黑色外套在右侧地上单膝跪地，面朝爱音。祥子一只手握着爱音的手，另一只手托住爱音的肘部，同时完成两项动作。"),
    ("separate_panels", "-r 左右两格，左格千早爱音穿白衬衫站在桌旁，右格千早爱音穿黑色外套独自坐在椅子上，两格之间留白，两个人物没有互动，不要跨格接触。"),
])
def test_live_nai_spatial_action_constraints(live_pipeline, tmp_path, case, prompt):
    """Retain evidence for semantic spatial review; coordinates alone prove no anatomy."""
    from nai_character_mode import parse_nai_character_plan
    from agent_tools.comfyui_workflows import custom_t2i_workflow

    live_pipeline[0].config["debug_prompt_enabled"] = True
    canvas_size = (1216, 832) if case == "joint_reach" else (832, 1216)
    result, calls = _build(live_pipeline, prompt, canvas_size=canvas_size)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"nai_spatial_{case}.json"
    report_path.write_text(json.dumps({
        "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Live NAI spatial evidence: {report_path}")
    assert result.final_prompt, result.summary.get("llm_error")
    assert len(calls) in (2, 3), calls
    assert (result.summary["nai_canvas"]["width"], result.summary["nai_canvas"]["height"]) == canvas_size
    raw_plan = parse_nai_character_plan(result.summary["nai_character_plan_raw"])
    if case != "separate_panels":
        for global_text in (raw_plan["global_prompt"], result.final_prompt):
            assert "手" in global_text and ("肩" if case == "joint_reach" else "肘") in global_text, global_text
    for rows in (raw_plan["characters"], result.summary["nai_characters"]):
        assert len(rows) == 2
        assert all(_CJK_RE.search(row["prompt"]) for row in rows)
        if case == "separate_panels":
            assert all(row["name"] == "chihaya_anon" for row in rows)
            assert all(not row.get("interaction_tags") for row in rows)
            left, right = sorted(rows, key=lambda row: row["x"])
            assert left["x"] < .5 < right["x"], rows
            assert "站" in left["prompt"] and "坐" in right["prompt"], rows
        else:
            by_name = {row["name"]: row for row in rows}
            assert set(by_name) == {"chihaya_anon", "togawa_sakiko"}
            assert by_name["chihaya_anon"]["x"] < by_name["togawa_sakiko"]["x"], rows
            text = " ".join(row["prompt"] for row in rows)
            assert "手" in text and ("肩" if case == "joint_reach" else "肘") in text, rows
    graph = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, result.final_prompt, "blurry",
        *canvas_size, 28, 6, 123, override_size=True,
        nai_characters=result.summary["nai_characters"],
    )
    selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    for index, row in enumerate(result.summary["nai_characters"], 1):
        assert selector[f"character{index}_x"] == row["x"]
        assert selector[f"character{index}_y"] == row["y"]


@pytest.mark.parametrize("case", ["named", "ordinary_implicit_donor", "recolor"])
def test_live_nai_bilingual_details_with_repeated_identity(live_pipeline, tmp_path, case):
    """English details must come from the writer, alongside Chinese prose."""
    from nai_character_mode import parse_nai_character_plan
    from tag_cleaner import normalize_tag_key, split_tags

    prompt = (
        "-r 三人上半身构图，同一画面里出现两个丰川祥子。千早爱音穿羽丘校服坐在中间；"
        "左侧丰川祥子穿月之森校服，右侧丰川祥子穿着oblivionis的衣服，"
        "两人分别用手扶住爱音的左肩和右肩，微笑看着爱音。"
    )
    if case != "named":
        prompt = prompt.replace("千早爱音穿羽丘校服", "千早爱音穿白衬衫")
    if case == "ordinary_implicit_donor":
        prompt = prompt.replace("穿着oblivionis的衣服", "穿oblivionis衣服")
    elif case == "recolor":
        prompt = prompt.replace("穿月之森校服", "穿黑色外套").replace("穿着oblivionis的衣服", "穿白色外套")
    live_pipeline[0].config["debug_prompt_enabled"] = True
    result, calls = _build(live_pipeline, prompt)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"nai_bilingual_repeated_identity_{case}.json"
    report_path.write_text(json.dumps({
        "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Live NAI bilingual evidence: {report_path}")
    assert result.final_prompt, result.summary
    assert len(calls) in (2, 3)
    assert not result.summary.get("nai_text_retry_validation_errors")
    raw_plan = parse_nai_character_plan(result.summary["nai_character_plan_raw"])
    if case != "recolor":
        assert any("clothing=月之森校服;" in line and "wardrobe=named_outfit;" in line
                   for line in calls[1]["prompt"].splitlines()), calls[1]["prompt"]

    def english_tags(text):
        return {normalize_tag_key(part) for part in split_tags(text)
                if not _CJK_RE.search(part) and "#" not in part}

    # Check raw writer output as well as final captions: profile restoration
    # must not conceal a writer that supplied only Chinese keywords.
    for global_text, rows in (
        (raw_plan["global_prompt"], raw_plan["characters"]),
        (result.final_prompt, result.summary["nai_characters"]),
    ):
        assert len(rows) == 3
        assert [row["name"] for row in rows].count("chihaya_anon") == 1
        assert [row["name"] for row in rows].count("togawa_sakiko") == 2
        assert _CJK_RE.search(global_text), global_text
        global_tags = english_tags(global_text)
        assert "3girls" in global_tags, global_text
        assert not {"1girls", "2girls"} & global_tags, global_text
        assert "upper body" in global_tags or "upper body shot" in global_tags, global_text
        assert "tsukinomori school uniform" not in global_tags
        assert all(_CJK_RE.search(row["prompt"]) for row in rows), rows
        anon = next(row for row in rows if row["name"] == "chihaya_anon")
        anon_clothing = "haneoka school uniform" if case == "named" else "white shirt"
        assert {anon_clothing, "sitting"} <= english_tags(anon["prompt"]), anon
        if case != "named":
            assert "haneoka school uniform" not in english_tags(anon["prompt"]), anon
        sakikos = sorted((row for row in rows if row["name"] == "togawa_sakiko"),
                         key=lambda row: row["x"])
        left, right = sakikos
        assert left["anchor_id"] != right["anchor_id"]
        if case == "recolor":
            assert {"black jacket", "black coat"} & english_tags(left["prompt"]), left
            assert not {"white jacket", "white coat"} & english_tags(left["prompt"]), left
            assert {"white jacket", "white coat"} & english_tags(right["prompt"]), right
            assert not {"black jacket", "black coat"} & english_tags(right["prompt"]), right
        else:
            assert "tsukinomori school uniform" in english_tags(left["prompt"]), left
            assert {"black corset", "red shirt"} & english_tags(right["prompt"]), right
        assert "tsukinomori school uniform" not in english_tags(right["prompt"]), right
        assert all({"smile", "smiling", "grin", "grinning"} & english_tags(row["prompt"])
                   for row in sakikos), sakikos


@pytest.mark.parametrize("case", ["reported_six_panel", "single_scene_counterexample"])
def test_live_nai_comic_identity_instances(live_pipeline, tmp_path, case):
    """Repeated panel appearances and repeated names are different cardinalities."""
    expected = {"togawa_sakiko": 4, "chihaya_anon": 3}
    if case == "reported_six_panel":
        failure = _read_json(PLUGIN_DIR / "diagnostics/nai-comic-identity-review/original-failure.json")
        prompt = failure["original_prompt"]
    else:
        prompt = (
            "-r 单一连续画面，千早爱音和丰川祥子在桌边吃面。"
            "爱音穿白衬衫、扎马尾，开心地拿着筷子；祥子穿灰色毛衣，低头害羞地吃面。"
            "两人都只出现一次，没有分格。"
        )
        expected = {"togawa_sakiko": 1, "chihaya_anon": 1}
    live_pipeline[0].config["debug_prompt_enabled"] = True
    result, calls = _build(live_pipeline, prompt)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{case}.json"
    report_path.write_text(json.dumps({
        "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Live comic identity evidence: {report_path}")
    assert result.final_prompt, result.summary
    rows = result.summary["nai_characters"]
    assert len(rows) == sum(expected.values()), rows
    for name, count in expected.items():
        assert sum(row["name"] == name for row in rows) == count
    assert len({(row["x"], row["y"]) for row in rows}) == len(rows)
    assert len(calls) <= 3
    assert all(_CJK_RE.search(row["prompt"]) for row in rows)
    assert not result.summary.get("nai_identity_retry_validation_errors")


@pytest.mark.parametrize("case", [
    "reported", "reported_repeat", "different_source_and_recipient", "eye_color",
    "two_wearers", "nai_repeated_instances", "real_clothing_source", "unknown_reference",
])
def test_live_dimension_scoped_appearance_references(live_pipeline, tmp_path, case):
    """Real LLM1 + real LLM2; never submit ComfyUI/NAI generation."""
    requests = {
        "reported": "千早爱音大头照，把头发梳成丰川祥子的发型，脖子戴黑色项圈，背景是蓝天白云。",
        "reported_repeat": "千早爱音超级大头照，千早爱音采用丰川祥子的发型，粉色头发和灰色眼睛不变，戴黑色项圈，头顶蓝天和云。",
        "different_source_and_recipient": "若叶睦大头照，采用重音tetoSV的发型，保留原有绿色头发和黄色眼睛，白底。",
        "eye_color": "千早爱音脸部特写，瞳色采用丰川祥子的瞳色，头发保持粉色，白底。",
        "two_wearers": "千早爱音穿白色毛衣，采用丰川祥子的发型；丰川祥子穿黑色连衣裙，采用重音tetoSV的发型。两人上半身同框合影，白底。",
        "nai_repeated_instances": "-r 两格漫画。左格：千早爱音采用丰川祥子的发型，戴黑色项圈，大头照。右格：千早爱音采用重音tetoSV的发型，戴白色项圈，大头照。两格都保持爱音自己的粉发灰眼，白底。丰川祥子和重音tetoSV只提供发型参考，本人不出场。",
        "real_clothing_source": "千早爱音穿丰川祥子的衣服，保持自己的粉色头发和灰色眼睛，上半身肖像，白底。",
        "unknown_reference": "千早爱音大头照，采用陌生角色Zeta的发型，粉发灰眼不变，白底。Zeta只作发型参考，不出场。",
    }
    pipeline, _context = live_pipeline
    pipeline.config["debug_prompt_enabled"] = True
    prompt = requests[case]
    result, calls = _build(live_pipeline, prompt)
    report_dir = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"appearance_reference_{case}.json").write_text(json.dumps({
        "case": case, "prompt": prompt, "calls": calls, "summary": result.summary,
        "final_prompt": result.final_prompt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result.final_prompt and not result.summary.get("llm_failed"), result.summary
    assert 2 <= len(calls) <= 4
    first = json.loads(result.summary["semantic_plan_raw"])
    if case == "real_clothing_source":
        assert first["characters"][0]["clothing_source"] == "丰川祥子"
        assert any(row["wardrobe_kind"] == "outfit_source"
                   for row in result.summary["semantic_character_outfits"])
        return
    assert all(not row.get("clothing_source") for row in first["characters"]), first
    refs = result.summary["appearance_reference_evidence"]
    if case == "unknown_reference":
        assert not refs
        assert len(first["characters"]) == 1
        return
    assert refs, result.summary
    if case == "eye_color":
        assert any(row["dimensions"] == ["eye_color"] and row["tags"] == ["yellow_eyes"] for row in refs)
        assert "yellow eyes" in result.final_prompt.lower().replace("_", " ")
        assert "blue hair" not in result.final_prompt.lower().replace("_", " ")
    elif case in {"reported", "reported_repeat"}:
        assert any("two_side_up" in row["tags"] for row in refs)
        text = result.final_prompt.lower().replace("_", " ")
        assert "two side up" in text
        assert "green necktie" not in text and "long sleeves" not in text
        assert not any(tag in text for tag in ("twin braids", "side ponytail", "twin buns", "lens flare", "bokeh", "depth of field", "rim light"))
    elif case == "different_source_and_recipient":
        assert any("drill_hair" in row["tags"] for row in refs)
        text = result.final_prompt.lower().replace("_", " ")
        assert "drill" in text
        assert "red hair" not in text and "red eyes" not in text
    elif case == "two_wearers":
        assert {row["character"] for row in refs} == {"千早爱音", "丰川祥子"}
        assert len(first["characters"]) == 2
        assert {row["reference"] for row in refs} == {"丰川祥子", "重音tetoSV"}
    else:
        instances = result.summary["nai_characters"]
        assert len(instances) == 2 and all(row["name"] == "chihaya_anon" for row in instances)
        assert all(_CJK_RE.search(row["prompt"]) for row in instances)
        assert any("two_side_up" in row["prompt"] for row in instances)
        assert any("drill" in row["prompt"] for row in instances)
