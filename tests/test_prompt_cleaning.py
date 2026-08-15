from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from prompt_builder import build_final_prompt  # noqa: E402
from tag_cleaner import clean_content_tags, join_prompt_parts  # noqa: E402


def _config() -> dict:
    return {
        "chiyo_preset_enabled": False,
        "fixed_characters": [
            "测试角色=1girl, solo, white hair, blue eyes, fox girl, fox ears,"
        ],
        "quality_prefix": "masterpiece, best quality,",
        "default_artist_tags": "@configured artist,",
        "style_tags": "",
    }


def test_fixed_character_content_keeps_only_non_identity_details() -> None:
    result = build_final_prompt(
        user_prompt="测试角色挥手",
        llm_content=(
            "masterpiece, best quality, @llm artist, white hair, blue eyes, "
            "loli, fox girl, fox ears, cute, kawaii, white dress, lace trim, "
            "waving, 2girls, crowd"
        ),
        config=_config(),
    )

    assert result.content_tags == "cute, kawaii, white dress, lace trim, waving"
    assert "@configured artist" in result.final_prompt
    assert "@llm artist" not in result.final_prompt


def test_explicit_multi_character_request_keeps_multi_character_tags() -> None:
    result = build_final_prompt(
        user_prompt="测试角色和另一人，双人构图",
        llm_content="2girls, multiple girls, holding hands, white dress",
        config=_config(),
    )

    assert "2girls" in result.content_tags
    assert "multiple girls" in result.content_tags
    assert "holding hands" in result.content_tags


def test_content_cleaning_uses_default_content_tag_limit() -> None:
    tags = ", ".join(f"visual detail {index}" for index in range(150))

    result = build_final_prompt(
        user_prompt="上限测试",
        llm_content=tags,
        config=_config(),
    )

    assert len(result.content_tags.split(", ")) == 65
    assert len(result.final_prompt.split(", ")) == 68
    assert "@configured artist" in result.final_prompt


def test_content_tag_limit_can_be_configured() -> None:
    tags = ", ".join(f"visual detail {index}" for index in range(20))
    config = _config()
    config["prompt_builder_max_content_tags"] = 12

    result = build_final_prompt(
        user_prompt="自定义上限测试",
        llm_content=tags,
        config=config,
    )

    assert len(result.content_tags.split(", ")) == 12


def test_nltags_is_appended_after_cleaned_tags_without_tag_processing() -> None:
    result = build_final_prompt(
        user_prompt="single girl",
        llm_content="1girl, standing",
        config=_config(),
        nltags=(
            "The girl gently gathers her hair with one hand while looking back, "
            "her skirt catching the light."
        ),
    )

    assert result.content_tags == "1girl, standing"
    assert result.final_prompt.endswith(
        ", Nltags: The girl gently gathers her hair with one hand while looking back, "
        "her skirt catching the light."
    )
    assert result.final_prompt.count("Nltags:") == 1


def test_structured_sections_keep_seven_block_order_and_hard_tags() -> None:
    config = _config()
    config["prompt_builder_max_content_tags"] = 1
    result = build_final_prompt(
        user_prompt="structured request",
        llm_content="full_body, white_background",
        config=config,
        required_count_tags=("1girl", "solo"),
        required_core_tags=("togawa_sakiko", "bang_dream!", "blue_hair"),
        structured_character_tags=("togawa_sakiko",),
        structured_copyright_tags=("bang_dream!",),
        structured_identity_blocks=("togawa_sakiko has blue_hair",),
        structured_detail_blocks=("togawa_sakiko wears black_pantyhose",),
        structured_tag_tags=("blue_hair", "black_pantyhose"),
        preserve_structured_order=True,
        nltags="togawa_sakiko stands against a white wall.",
    )

    assert (
        "1girl, solo, togawa sakiko, bang dream!, "
        "@configured artist, "
        "togawa sakiko has blue hair, "
        "togawa sakiko wears black pantyhose, "
        "blue hair, black pantyhose, full body, "
        "Nltags: togawa sakiko stands against a white wall."
    ) in result.final_prompt
    assert result.final_prompt.count("blue hair") == 2
    assert result.final_prompt.count("black pantyhose") == 2
    assert result.final_prompt.count("Nltags:") == 1
    assert "has blue hair" not in result.final_prompt.split("Nltags:", 1)[1]


def test_structured_tags_do_not_drop_appearance_or_pose_categories() -> None:
    result = build_final_prompt(
        user_prompt="structured request",
        llm_content=(
            "blue_hair, yellow_eyes, long_hair, animal_ears, tail, "
            "sitting, lying, hugging"
        ),
        config=_config(),
        required_count_tags=("1girl", "solo"),
        required_core_tags=("togawa_sakiko",),
        structured_character_tags=("togawa_sakiko",),
        structured_identity_blocks=("togawa_sakiko has blue hair",),
        structured_detail_blocks=("togawa_sakiko is sitting and hugging a friend",),
        preserve_structured_order=True,
    )

    content_tags = result.content_tags.replace("_", " ")
    for tag in (
        "blue hair",
        "yellow eyes",
        "long hair",
        "animal ears",
        "tail",
        "sitting",
        "lying",
        "hugging",
    ):
        assert tag in content_tags


def test_existing_hard_tags_and_nltags_prevent_chinese_fallback() -> None:
    result = build_final_prompt(
        user_prompt="无角色, 只有两只踮起的黑丝足底",
        llm_content="",
        config=_config(),
        required_core_tags=("black_pantyhose", "thighhighs"),
        nltags="exactly two feet visible. bottom of the feet / soles.",
    )

    assert "black pantyhose" in result.final_prompt
    assert "Nltags: exactly two feet visible" in result.final_prompt
    assert "无角色" not in result.final_prompt


def test_final_prompt_uses_spaces_for_danbooru_word_separators() -> None:
    result = build_final_prompt(
        user_prompt="character",
        llm_content="hair_ornament, black_dress",
        config=_config(),
        required_core_tags=("togawa_sakiko_(bang_dream!)",),
        nltags="togawa_sakiko wears a black_dress.",
    )

    assert "_" not in result.final_prompt
    assert "togawa sakiko (bang dream!)" in result.final_prompt
    assert "hair ornament" in result.final_prompt
    assert "black dress" in result.final_prompt


def test_raw_mode_does_not_apply_content_tag_limit() -> None:
    tags = ", ".join(f"raw detail {index}" for index in range(120))

    result = build_final_prompt(
        user_prompt=f"raw {tags}",
        llm_content="ignored",
        config=_config(),
    )

    assert result.raw_mode is True
    assert len(result.content_tags.split(", ")) == 120


def test_content_cleaner_removes_high_confidence_semantic_conflicts() -> None:
    cleaned = clean_content_tags(
        "{{holding sword}}, Point a sword at the audience, looking away, "
        "looking at viewer, nude, naked, topless, bottomless, holding nothing, "
        "mist, morning mist, pear blossoms, punis, sheer fabric, "
        "translucent fabric",
        strip_character_tags=False,
    )

    assert "sword pointed at viewer" in cleaned
    assert "holding sword" in cleaned
    assert "looking away" in cleaned
    assert "looking at viewer" not in cleaned
    assert "nude" in cleaned
    assert "naked" not in cleaned
    assert "topless" not in cleaned
    assert "bottomless" not in cleaned
    assert "holding nothing" not in cleaned
    assert "morning mist" in cleaned
    assert ", mist," not in f", {cleaned},"
    assert "pear blossoms" in cleaned
    assert "penis" in cleaned
    assert "punis" not in cleaned
    assert "sheer fabric" in cleaned
    assert "translucent fabric" not in cleaned


def test_dangling_english_conjunction_is_removed_from_tag_fragment() -> None:
    result = build_final_prompt(
        user_prompt="测试角色",
        llm_content="and large breasts, full body",
        config=_config(),
    )

    assert "and large breasts" not in result.final_prompt
    assert "large breasts, full body" in result.final_prompt


def test_specific_clothing_tags_replace_low_information_roots() -> None:
    cleaned = clean_content_tags(
        "shirt, grey shirt, skirt, pleated skirt, thighhighs, black thighhighs",
        strip_character_tags=False,
    )

    assert cleaned == "grey shirt, pleated skirt, black thighhighs"


def test_underscore_and_space_spellings_are_deduplicated() -> None:
    assert join_prompt_parts(["black_thighhighs, black thighhighs"]) == (
        "black thighhighs"
    )


def test_action_or_accessory_suffix_does_not_replace_garment_root() -> None:
    cleaned = clean_content_tags(
        "dress, dress bow, skirt, skirt lift, shirt, shirt tug",
        strip_character_tags=False,
    )

    assert cleaned == "dress, dress bow, skirt, skirt lift, shirt, shirt tug"


def test_scoped_core_filter_handles_copyright_punctuation() -> None:
    cleaned = clean_content_tags(
        "oblivionis_(bang_dream!), wrong_variant_(bang_dream!), "
        "another wrong (bang dream!), black dress",
        strip_character_tags=False,
        protected_core_tags=("oblivionis_(bang_dream!)",),
    )

    assert cleaned == "oblivionis_(bang_dream!), black dress"


def test_markdown_fence_language_does_not_become_a_tag() -> None:
    cleaned = clean_content_tags(
        "```danbooru\nblack_dress, red_bow\n```",
        strip_character_tags=False,
    )

    assert cleaned == "black_dress, red_bow"


def test_all_structured_count_tags_bypass_content_cleaning() -> None:
    examples = (
        ("1girl", "solo"),
        ("1girl", "1boy"),
        ("2girls", "yuri"),
        ("3girls", "multiple girls"),
    )
    for count_tags in examples:
        result = build_final_prompt(
            user_prompt="character",
            llm_content=", ".join((*count_tags, "standing")),
            config=_config(),
            required_count_tags=count_tags,
            required_core_tags=("togawa_sakiko_(bang_dream!)",),
        )

        assert result.required_count_tags == count_tags
        final_head = result.final_prompt.split("togawa sakiko", 1)[0]
        assert all(tag.replace("_", " ") in final_head for tag in count_tags)


def test_content_cleaner_removes_viewer_gaze_when_eyes_are_closed() -> None:
    cleaned = clean_content_tags(
        "looking up at viewer, singing, closed eyes, gentle smile",
        strip_character_tags=False,
    )

    assert "looking up at viewer" not in cleaned
    assert cleaned == "singing, closed eyes, gentle smile"


def test_fixed_character_cleaning_preserves_explicit_hair_details() -> None:
    cleaned = clean_content_tags(
        "white hair, blue eyes, fox girl, lotus hair ornament, floating hair, "
        "braid, pubic hair",
        strip_character_tags=True,
    )

    assert cleaned == "lotus hair ornament, floating hair, braid, pubic hair"


def test_content_cleaner_limits_synonyms_without_removing_distinct_light_roles() -> (
    None
):
    cleaned = clean_content_tags(
        "light rays, sunbeams, glowing, illuminated, bright, luminous, radiant, "
        "backlighting, rim lighting, cast shadows, floating particles, "
        "light particles, flowing dress, dress flowing, embroidered hem",
        strip_character_tags=False,
    )

    assert cleaned == (
        "light rays, glowing, illuminated, backlighting, rim lighting, "
        "cast shadows, floating particles, flowing dress, embroidered hem"
    )
