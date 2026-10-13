import logging
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from danbooru_resolver import DanbooruResolver


def engine(rows):
    return DanbooruResolver(logger=logging.getLogger("test"),cache={},get_bool=lambda k,d:d,
        get_int=lambda k,d:d,get_float=lambda k,d:d,get_str=lambda k,d:d,
        config={"danbooru_term_mappings":rows})


@pytest.mark.parametrize("text",[
    "她不想戴半指手套", "她没想穿半指手套", "她不打算使用半指手套",
    "她没有戴半指手套", "她不戴黑色的半指手套", "移除半指手套",
    "她不要帽子和半指手套", "她不需要任何半指手套",
])
def test_negative_scope_does_not_leak_internal_short_alias(text):
    r=engine(["半指手套=fingerless_gloves","手套=gloves","帽子=hat"])
    assert r.matched_term_mappings_for_prompt(text)==()


@pytest.mark.parametrize("text",[
    "她不戴半指手套，同伴戴手套", "她不想戴半指手套但是同伴戴手套",
    "不要帽子和半指手套，手套放在桌上", "她不戴帽子穿手套",
    "不要帽子\n手套放在桌上",
    "她不但戴手套而且穿帽衫", "她并非不戴手套，但手套仍在她手上",
])
def test_independent_positive_reference_survives(text):
    r=engine(["半指手套=fingerless_gloves","手套=gloves","帽子=hat"])
    assert ("手套","gloves") in r.matched_term_mappings_for_prompt(text)


@pytest.mark.parametrize("text",[
    "她说公园两个字，画面白底", "她说：\"教室\"，实际站在街上",
    "她说道：“公园”，背景白底", "心里的教室只是比喻，画面纯白",
    "她念教室这个词，画面纯白",
    "公园这个字写在纸上，背景白底",
])
def test_literal_or_explicit_metaphor_is_not_scene_reference(text):
    assert engine(["公园=park","教室=classroom"]).matched_term_mappings_for_prompt(text)==()


@pytest.mark.parametrize("text",[
    "她在公园说：“教室”", "她在“公园”里散步", "她说：“你好”，站在公园",
    "背景公园里立着写有“教室”的标牌", "她像鸟一样走在公园",
    "她没有朋友，在公园散步", "她不是老师在公园散步",
])
def test_real_scene_and_quoted_design_name_survive(text):
    assert ("公园","park") in engine(["公园=park","教室=classroom"]).matched_term_mappings_for_prompt(text)


def test_normalized_coordinates_cache_update_and_overlap():
    r=engine(["手套=gloves","半指手套=fingerless_gloves","mask=mask"])
    assert r.matched_term_mappings_for_prompt("  她   不想戴半指手套，手套在桌上 ")==(("手套","gloves"),)
    assert r.matched_term_mappings_for_prompt("not a mask")==()
    assert r.matched_term_mappings_for_prompt("kimono mask")==(("mask","mask"),)
    r._config["danbooru_term_mappings"]=["球鞋=sneakers"]
    assert r.matched_term_mappings_for_prompt("她穿球鞋但不想戴手套")==(("球鞋","sneakers"),)
    r._config["danbooru_term_mappings_enabled"]=False
    assert r.matched_term_mappings_for_prompt("她穿球鞋")==()


def test_configured_absence_and_asymmetric_attributes_remain_positive_references():
    r=engine(["没穿内裤=no_panties","内裤=panties","不同发型=alternate_hairstyle",
              "不对称发型=asymmetrical_hair","no pants=no_pants","pants=pants"])
    assert r.matched_term_mappings_for_prompt("她没穿内裤，使用不同发型和不对称发型，no pants")==(
        ("没穿内裤","no_panties"),("不同发型","alternate_hairstyle"),
        ("不对称发型","asymmetrical_hair"),("no pants","no_pants"))
    assert r.matched_term_mappings_for_prompt("不要没穿内裤")==()


def test_pin_accessory_verb_is_not_negative_imperative():
    r=engine(["蝴蝶结=bow","发饰=hair_ornament"])
    assert r.matched_term_mappings_for_prompt("她头发上别着蝴蝶结和发饰")==(("蝴蝶结","bow"),("发饰","hair_ornament"))
    assert r.matched_term_mappings_for_prompt("别戴蝴蝶结和发饰")==()
