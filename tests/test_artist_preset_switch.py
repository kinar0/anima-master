from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from prompt_builder import build_final_prompt  # noqa: E402
from prompt_presets import (  # noqa: E402
    active_artist_tags,
    artist_preset_list,
    extract_artist_preset_switch,
)

build_final_prompt_low_cfg = pytest.importorskip(
    "variants.turbo.low_cfg_harness.prompt_builder",
    reason="low_cfg_harness variant not importable in this environment",
).build_final_prompt


def _config(
    presets: dict[str, str] | None = None,
    default_tags: str = "",
    active_preset: str = "",
) -> dict:
    return {
        "default_artist_tags": default_tags,
        "artist_presets": [
            f"{name}={tags}" for name, tags in (presets or {}).items()
        ],
        "active_artist_preset": active_preset,
    }


def test_artist_preset_list_orders_default_first_then_sorted():
    config = _config(
        presets={"zeta": "artist zeta", "alpha": "artist alpha"},
        default_tags="default tag",
        active_preset="zeta",
    )
    entries = artist_preset_list(config)
    assert entries == [
        ("默认", "default tag"),
        ("alpha", "artist alpha"),
        ("zeta", "artist zeta"),
    ]


def test_artist_preset_list_skips_empty_default():
    config = _config(
        presets={"zeta": "artist zeta", "alpha": "artist alpha"},
        default_tags="",
    )
    assert artist_preset_list(config) == [
        ("alpha", "artist alpha"),
        ("zeta", "artist zeta"),
    ]


def test_active_artist_tags_with_index_selects_nth_entry():
    config = _config(
        presets={"zeta": "artist zeta", "alpha": "artist alpha"},
        default_tags="default tag",
        active_preset="zeta",
    )
    # 默认画师 tags 为第 1 个，排序后画师组 alpha 为第 2 个、zeta 为第 3 个。
    assert active_artist_tags(config, 1) == "default tag"
    assert active_artist_tags(config, 2) == "artist alpha"
    assert active_artist_tags(config, 3) == "artist zeta"


def test_active_artist_tags_out_of_range_returns_empty():
    config = _config(presets={"zeta": "artist zeta"})
    assert active_artist_tags(config, 0) == ""
    assert active_artist_tags(config, 2) == ""


def test_extract_switch_detects_lower_and_upper_case():
    config = _config(presets={"zeta": "artist zeta"})
    for raw in ("-s1 少女", "-S1 少女", "-s1,少女"):
        index, cleaned, error = extract_artist_preset_switch(raw, config)
        assert index == 1, raw
        assert cleaned == "少女", raw
        assert error is None, raw


def test_extract_switch_cleans_like_other_switches():
    config = _config(presets={"zeta": "artist zeta"})
    index, cleaned, error = extract_artist_preset_switch(
        "-s1 横图：少女站在河岸", config
    )
    assert index == 1
    assert cleaned == "横图：少女站在河岸"
    assert error is None


def test_extract_switch_out_of_range_returns_error():
    config = _config(presets={"zeta": "artist zeta"})
    index, cleaned, error = extract_artist_preset_switch("-s2 少女", config)
    assert index is None
    assert cleaned == "少女"
    assert "越界" in error


def test_extract_switch_zero_returns_error():
    config = _config(presets={"zeta": "artist zeta"})
    index, cleaned, error = extract_artist_preset_switch("-s0 少女", config)
    assert index is None
    assert cleaned == "少女"
    assert "正整数" in error


def test_extract_switch_no_entries_returns_error():
    config = _config()
    index, cleaned, error = extract_artist_preset_switch("-s1 少女", config)
    assert index is None
    assert cleaned == "少女"
    assert "没有可用的画师串" in error


def test_build_final_prompt_uses_nth_artist_string():
    config = _config(
        presets={"zeta": "artist zeta", "alpha": "artist alpha"},
        default_tags="default tag",
        active_preset="zeta",
    )
    built = build_final_prompt(
        user_prompt="-s2 少女",
        llm_content="1girl, standing",
        config=dict(config),
    )
    # -s2 -> 排序后第 2 个画师串 = alpha，而非当前启用的 zeta。
    assert "artist alpha" in built.final_prompt
    assert "artist zeta" not in built.final_prompt
    assert "-s2" not in built.final_prompt


def test_build_final_prompt_keeps_enabled_artist_without_switch():
    config = _config(
        presets={"zeta": "artist zeta", "alpha": "artist alpha"},
        active_preset="zeta",
    )
    built = build_final_prompt(
        user_prompt="少女",
        llm_content="1girl, standing",
        config=dict(config),
    )
    assert "artist zeta" in built.final_prompt
    assert "artist alpha" not in built.final_prompt


def test_build_final_prompt_falls_back_on_invalid_switch():
    config = _config(
        presets={"zeta": "artist zeta"},
        active_preset="zeta",
    )
    built = build_final_prompt(
        user_prompt="-s9 少女",
        llm_content="1girl, standing",
        config=dict(config),
    )
    # 越界开关回退到当前启用的画师串。
    assert "artist zeta" in built.final_prompt
    assert "-s9" not in built.final_prompt


def test_build_final_prompt_raw_mode_after_switch():
    config = _config(presets={"zeta": "artist zeta"})
    built = build_final_prompt(
        user_prompt="-s1 原样 少女",
        llm_content="",
        config=dict(config),
    )
    assert built.raw_mode is True
    assert "-s1" not in built.final_prompt
    assert "少女" in built.final_prompt


def test_low_cfg_variant_uses_nth_artist_string():
    config = _config(
        presets={"zeta": "artist zeta", "alpha": "artist alpha"},
        default_tags="default tag",
        active_preset="zeta",
    )
    built = build_final_prompt_low_cfg(
        user_prompt="-s2 少女",
        llm_content="1girl, standing",
        config=dict(config),
    )
    assert "artist alpha" in built.final_prompt
    assert "artist zeta" not in built.final_prompt
    assert "-s2" not in built.final_prompt
