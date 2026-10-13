from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping_syntax import parse_mapping_entry
from noun_translation_batch import parse_existing_mappings
from toolbox_terms import existing_terms
from danbooru_resolver import DanbooruResolver


@pytest.mark.parametrize("row,alias,tag", [
    ("作品：副标题制服=series_uniform", "作品：副标题制服", "series_uniform"),
    ("性别转换（男→女）=genderswap_(mtf)", "性别转换（男→女）", "genderswap_(mtf)"),
    ("角色：名字=>name:tag", "角色：名字", "name:tag"),
    ("中文别名＝canonical_tag", "中文别名", "canonical_tag"),
    ("裙子→skirt", "裙子", "skirt"),
    ("裙子：skirt", "裙子", "skirt"),
    ("裙子:skirt", "裙子", "skirt"),
])
def test_import_review_editor_and_matcher_share_separator_precedence(row, alias, tag):
    assert parse_mapping_entry(row) == (alias, tag)
    assert parse_existing_mappings([row]) == existing_terms([row]) == {alias: tag}
    resolver = DanbooruResolver(logger=None, cache={}, config={"danbooru_term_mappings": [row]},
                                get_bool=lambda k, d: d, get_int=lambda k, d: d,
                                get_float=lambda k, d: d, get_str=lambda k, d: d)
    assert resolver.wardrobe_snapshot()["terms"] == [{"alias": alias, "tag": tag}]
    assert resolver.matched_term_mappings_for_prompt(alias) == ((alias, tag),)
    assert resolver.matched_term_mappings_for_prompt("无关输入") == ()


def test_missing_separator_does_not_become_a_mapping():
    assert parse_mapping_entry("只是一个词语") is None
