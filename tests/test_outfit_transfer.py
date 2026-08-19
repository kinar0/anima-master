from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from outfit_transfer import (
    build_effective_outfit_plan,
    build_outfit_constraint_narrative,
    build_outfit_transfer_context,
    bind_explicit_outfit_patch_target,
    detect_outfit_transfer,
    filter_outfit_tags,
    keep_only_verified_outfit_tags,
    rewrite_target_outfit_detail,
)


def test_detects_named_outfit_transfer_to_fixed_character():
    plan = detect_outfit_transfer("联网检索碧蓝档案角色妃咲，深度思考后将其服饰应用于狐莉身上", "狐莉")

    assert plan.enabled is True
    assert plan.source_subject == "妃咲"
    assert plan.target_character == "狐莉"
    assert plan.source_from_search is True


def test_no_fixed_character_does_not_force_outfit_transfer():
    plan = detect_outfit_transfer("联网检索碧蓝档案角色妃咲的服装", "")

    assert plan.enabled is False


def test_filter_outfit_tags_drops_identity_features():
    filtered = filter_outfit_tags(
        "blue hair, red eyes, fox ears, tail, white dress, ribbon, gold trim, boots"
    )

    assert "white dress" in filtered
    assert "ribbon" in filtered
    assert "gold trim" in filtered
    assert "boots" in filtered
    assert "blue hair" not in filtered
    assert "red eyes" not in filtered
    assert "fox ears" not in filtered
    assert "tail" not in filtered


def test_outfit_transfer_context_filters_reference_tags():
    plan = detect_outfit_transfer("狐莉穿上图中角色的衣服\n参考图视觉反推 tags：blue hair, red eyes, white dress, ribbon, boots", "狐莉")
    context = build_outfit_transfer_context(plan, prompt=plan.directive_prompt + "\n参考图视觉反推 tags：blue hair, red eyes, white dress, ribbon, boots")

    assert context.enabled is True
    assert context.outfit_summary_source == "reference_filter"
    assert "white dress" in context.outfit_summary
    assert "ribbon" in context.outfit_summary
    assert "blue hair" not in context.outfit_summary
    assert "red eyes" not in context.outfit_summary
    assert "hair" in context.forbidden_identity_features


def test_named_target_wearing_variant_extracts_only_variant_source() -> None:
    plan = detect_outfit_transfer("若叶睦穿着mortis的衣服", "若叶睦")

    assert plan.enabled is True
    assert plan.target_character == "若叶睦"
    assert plan.source_subject == "mortis"


def test_variant_outfit_before_multiple_targets_is_detected_without_fixed_target() -> None:
    plan = detect_outfit_transfer(
        "穿着oblivionis服装的若叶睦和丰川祥子",
        "",
    )

    assert plan.enabled is True
    assert plan.source_subject == "oblivionis"


def test_unverified_llm_outfit_tags_are_removed() -> None:
    filtered = keep_only_verified_outfit_tags(
        "full body, black dress, gothic lolita, white lace, feather hair ornament, "
        "floor length dress, black mask, standing",
        (),
    )

    assert filtered == "full body, standing"


def test_only_post_verified_outfit_tags_survive() -> None:
    filtered = keep_only_verified_outfit_tags(
        "black dress, puffy sleeves, white lace, black mask, standing",
        ("black_dress", "puffy_sleeves", "black_mask"),
    )

    assert filtered == "black dress, puffy sleeves, black mask, standing"


def test_named_wearer_wins_over_unrelated_selected_fixed_character() -> None:
    plan = detect_outfit_transfer(
        "丰川祥子穿着oblivionis的衣服，千早爱音穿校服",
        "千早爱音",
        ("丰川祥子", "千早爱音"),
    )

    assert plan.enabled is True
    assert plan.source_subject == "oblivionis"
    assert plan.target_character == "丰川祥子"


def test_explicit_pink_shirt_replaces_cached_red_shirt() -> None:
    prompt = "丰川祥子穿着oblivionis的衣服，但上衣是粉色的"
    transfer = detect_outfit_transfer(
        prompt,
        "丰川祥子",
        ("丰川祥子",),
    )
    effective = build_effective_outfit_plan(
        transfer,
        user_prompt=prompt,
        base_tags=(
            "red_shirt",
            "black_corset",
            "black_skirt",
            "black_mask",
        ),
        known_character_names=("丰川祥子",),
    )

    assert effective.has_destructive_override is True
    assert "red_shirt" in effective.removed_tags
    assert "red_shirt" not in effective.effective_tags
    assert "pink_shirt" in effective.effective_tags
    assert "black_corset" in effective.effective_tags
    assert "black_skirt" in effective.effective_tags

    filtered = keep_only_verified_outfit_tags(
        "red shirt, pink shirt, blue jacket, black skirt, standing",
        effective.effective_tags,
        effective.forbidden_slots,
    )
    assert filtered == "pink shirt, black skirt, standing"

    detail = rewrite_target_outfit_detail(
        "togawa_sakiko wears a red shirt and black corset",
        effective,
    )
    assert "pink shirt" in detail
    assert "red shirt" not in detail


def test_no_skirt_removes_only_skirt_layer_without_implying_bottomless() -> None:
    prompt = "丰川祥子穿着oblivionis的衣服，但丰川祥子下半身没穿裙子"
    transfer = detect_outfit_transfer(
        prompt,
        "丰川祥子",
        ("丰川祥子",),
    )
    effective = build_effective_outfit_plan(
        transfer,
        user_prompt=prompt,
        base_tags=(
            "red_shirt",
            "black_corset",
            "black_skirt",
            "black_pantyhose",
            "black_boots",
        ),
        known_character_names=("丰川祥子",),
    )

    assert effective.removed_tags == ("black_skirt",)
    assert "black_skirt" not in effective.effective_tags
    assert "black_pantyhose" in effective.effective_tags
    assert "black_boots" in effective.effective_tags
    assert "bottomless" not in effective.effective_tags

    narrative = build_outfit_constraint_narrative(
        effective,
        subject="togawa sakiko",
        source_tags=("oblivionis_(bang_dream!)",),
    )
    assert narrative == "togawa sakiko wears no skirt."
    detail = rewrite_target_outfit_detail(
        "togawa_sakiko wears a red shirt and black skirt",
        effective,
    )
    assert "black skirt" not in detail
    assert "no skirt" in detail


def test_outfit_patch_does_not_leak_to_another_named_character() -> None:
    prompt = (
        "丰川祥子穿着oblivionis的衣服，"
        "千早爱音下半身没穿裙子"
    )
    transfer = detect_outfit_transfer(
        prompt,
        "千早爱音",
        ("丰川祥子", "千早爱音"),
    )
    effective = build_effective_outfit_plan(
        transfer,
        user_prompt=prompt,
        base_tags=("red_shirt", "black_skirt"),
        known_character_names=("丰川祥子", "千早爱音"),
    )

    assert transfer.target_character == "丰川祥子"
    assert effective.patches == ()
    assert effective.effective_tags == ("red_shirt", "black_skirt")


def test_explicit_outerwear_addition_does_not_replace_inner_shirt() -> None:
    prompt = "丰川祥子穿着oblivionis的衣服，再穿一件白色外套"
    transfer = detect_outfit_transfer(prompt, "丰川祥子", ("丰川祥子",))
    effective = build_effective_outfit_plan(
        transfer,
        user_prompt=prompt,
        base_tags=("red_shirt", "black_skirt"),
        known_character_names=("丰川祥子",),
    )

    assert effective.has_destructive_override is False
    assert effective.removed_tags == ()
    assert effective.added_tags == ("white_jacket",)
    assert effective.effective_tags == (
        "red_shirt",
        "black_skirt",
        "white_jacket",
    )


def test_standalone_no_skirt_binds_target_without_enabling_strict_allowlist() -> None:
    prompt = "丰川祥子下半身没穿裙子"
    transfer = bind_explicit_outfit_patch_target(
        detect_outfit_transfer(prompt),
        user_prompt=prompt,
    )
    effective = build_effective_outfit_plan(
        transfer,
        user_prompt=prompt,
        base_tags=(),
    )

    assert transfer.enabled is False
    assert transfer.target_character == "丰川祥子"
    assert effective.forbidden_slots == ("lower_body.skirt",)
    filtered = keep_only_verified_outfit_tags(
        "red shirt, black skirt, blue jacket, standing",
        effective.effective_tags,
        effective.forbidden_slots,
        strict_allowlist=False,
    )
    assert filtered == "red shirt, blue jacket, standing"
