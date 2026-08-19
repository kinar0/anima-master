from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import danbooru_semantic as semantic_module  # noqa: E402
import danbooru_resolver as resolver_module  # noqa: E402
from danbooru_semantic import (  # noqa: E402
    SemanticAnchor,
    SemanticLookupResult,
    lookup_semantic_anchors,
    merge_semantic_results,
    parse_semantic_plan,
)
from danbooru_resolver import DanbooruResolver  # noqa: E402
from danbooru_tags import VariantOutfitProfile  # noqa: E402


def test_semantic_source_phrase_matches_ascii_case_insensitively() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"source","role":"outfit_source",'
        '"group":"character","source_text":"oblivionis",'
        '"description":"Oblivionis outfit",'
        '"candidates":["oblivionis_(bang_dream!)"]}]}',
        "穿着Oblivionis服装",
    )

    assert plan and plan[0].role == "outfit_source"


def test_semantic_plan_accepts_independently_named_outfit_role() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"outfit","role":"outfit",'
        '"group":"outfit","source_text":"羽丘校服",'
        '"description":"Haneoka school uniform",'
        '"candidates":["haneoka_school_uniform"]}]}',
        "千早爱音穿着羽丘校服",
    )

    assert plan and plan[0].role == "outfit"


def test_semantic_plan_recovers_full_named_uniform_source_from_generic_source() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"outfit","role":"outfit",'
        '"group":"outfit","source_text":"校服",'
        '"description":"Tsukinomori school uniform",'
        '"candidates":["school_uniform"]}]}',
        "若叶睦和长崎素世穿着月之森校服",
    )

    assert plan and plan[0].source_text == "月之森校服"


def test_specific_named_uniform_is_not_satisfied_by_generic_uniform_tag(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        anchor_id="outfit",
        role="outfit",
        group="outfit",
        source_text="月之森校服",
        description="Tsukinomori school uniform",
        candidates=("school_uniform",),
    )
    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "confirmed_tags": {
                        "general": [{"tag": "school_uniform"}]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ()
    assert result.named_outfit_tags == ()
    assert result.missing_descriptions == ("Tsukinomori school uniform",)


def test_exact_general_fallback_named_outfit_is_screened_and_promoted(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        anchor_id="outfit",
        role="outfit",
        group="outfit",
        source_text="羽丘校服",
        description="Haneoka school uniform",
        candidates=("haneoka_school_uniform",),
    )

    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "candidate_tags": {
                        "general": [
                            {
                                "tag": "haneoka_school_uniform",
                                "category": "general",
                                "source_category": "general",
                                "count": 3432,
                                "match_layer": "group_general_fallback",
                            }
                        ]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ("haneoka_school_uniform",)
    assert result.named_outfit_tags == ("haneoka_school_uniform",)


def test_general_fallback_individual_garment_is_not_promoted(monkeypatch) -> None:
    anchor = SemanticAnchor(
        anchor_id="shirt",
        role="clothing",
        group="clothing",
        source_text="白衬衫",
        description="white shirt",
        candidates=("white_shirt",),
    )
    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "candidate_tags": {
                        "general": [
                            {
                                "tag": "white_shirt",
                                "source_category": "general",
                                "count": 100000,
                                "match_layer": "group_general_fallback",
                            }
                        ]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ()
    assert result.named_outfit_tags == ()


def test_confirmed_named_outfit_is_marked_for_persistence(monkeypatch) -> None:
    anchor = SemanticAnchor(
        anchor_id="wedding",
        role="outfit",
        group="outfit",
        source_text="婚礼衣服",
        description="wedding dress outfit",
        candidates=("wedding_dress",),
    )
    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "confirmed_tags": {
                        "general": [{"tag": "wedding_dress"}]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ("wedding_dress",)
    assert result.named_outfit_tags == ("wedding_dress",)


def test_semantic_cache_and_fresh_lookup_merge_without_shadowing() -> None:
    cached = SemanticLookupResult(
        confirmed_tags=("oblivionis_(bang_dream!)",),
        outfit_source_tags=("oblivionis_(bang_dream!)",),
        outfit_profile_tags=("red_shirt",),
        status="profile_cache",
    )
    fresh = SemanticLookupResult(
        confirmed_tags=("haneoka_school_uniform",),
        named_outfit_tags=("haneoka_school_uniform",),
        status="resolved",
    )

    merged = merge_semantic_results(cached, fresh)

    assert merged.status == "resolved"
    assert merged.outfit_profile_tags == ("red_shirt",)
    assert merged.named_outfit_tags == ("haneoka_school_uniform",)


def test_outfit_profile_cache_survives_resolver_recreation() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        build(path).remember_outfit_summary(
            "初音未来",
            ("hatsune_miku",),
            ("grey_shirt", "detached_sleeves", "pleated_skirt"),
        )
        cached = build(path).cached_outfit_source("初音未来")

    assert cached is not None
    assert cached.status == "profile_cache"
    assert cached.outfit_source_tags == ("hatsune_miku",)
    assert cached.outfit_profile_tags == (
        "grey_shirt",
        "detached_sleeves",
        "pleated_skirt",
    )


def test_named_outfit_profile_survives_recreation_and_matches_english_alias() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        build(path).remember_named_outfit(
            "羽丘校服", "haneoka_school_uniform"
        )
        cached = build(path).cached_named_outfits_for_prompt(
            "chihaya anon wears haneoka school uniform"
        )

    assert cached is not None
    assert cached.confirmed_tags == ("haneoka_school_uniform",)
    assert cached.named_outfit_tags == ("haneoka_school_uniform",)


def test_generic_school_uniform_is_neither_persisted_nor_loaded_as_named_set() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "profiles": {
                        "校服": {
                            "kind": "named_outfit",
                            "aliases": ["校服", "school_uniform"],
                            "outfit_tags": ["school_uniform"],
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resolver = build(path)
        assert resolver.cached_named_outfits_for_prompt("羽丘校服") is None
        before = path.read_text(encoding="utf-8")
        resolver.remember_named_outfit("月之森校服", "school_uniform")
        after = path.read_text(encoding="utf-8")

    assert after == before


def test_configured_named_uniform_survives_unavailable_semantic_lookup() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={
                "danbooru_named_outfit_mappings": [
                    "月之森校服=tsukinomori_school_uniform"
                ]
            },
        )
        cached = resolver.cached_named_outfits_for_prompt(
            "若叶睦和长崎素世穿着月之森校服"
        )

    assert cached is not None
    assert cached.confirmed_tags == ("tsukinomori_school_uniform",)
    assert cached.named_outfit_tags == ("tsukinomori_school_uniform",)


def test_configured_named_outfit_mapping_validates_and_accepts_separators() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={
                "danbooru_named_outfit_mappings": [
                    "羽丘校服 → haneoka_school_uniform",
                    "坏映射=not a canonical tag?",
                ]
            },
        )

        matched = resolver.cached_named_outfits_for_prompt("千早爱音穿着羽丘校服")
        invalid = resolver.cached_named_outfits_for_prompt("角色穿着坏映射")

    assert matched is not None
    assert matched.named_outfit_tags == ("haneoka_school_uniform",)
    assert invalid is None


def test_configured_term_mapping_is_returned_as_request_hard_tag() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={"danbooru_term_mappings": ["舞会面具=masquerade_mask"]},
        )

        result = resolver.cached_term_mappings_for_prompt("少女戴着舞会面具")

    assert result is not None
    assert result.confirmed_tags == ("masquerade_mask",)


def test_wardrobe_editor_round_trip_updates_profiles_and_mappings() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={
                "danbooru_named_outfit_mappings": [
                    "月之森校服=tsukinomori_school_uniform"
                ],
                "danbooru_term_mappings": [],
            },
        )
        initial = resolver.wardrobe_snapshot()
        saved = resolver.save_wardrobe(
            {
                "baseRevision": initial["revision"],
                "outfits": [
                    {
                        "key": "amoris",
                        "aliases": ["Amoris"],
                        "sourceTags": ["amoris_(bang_dream!)"],
                        "tags": ["black_corset", "red_shorts"],
                        "qualifier": "default",
                    }
                ],
                "outfitSets": [
                    {
                        "alias": "月之森校服",
                        "tag": "tsukinomori_school_uniform",
                        "origin": "configured",
                        "profileKey": "",
                    }
                ],
                "terms": [{"alias": "舞会面具", "tag": "masquerade_mask"}],
            }
        )
        persisted = json.loads(path.read_text(encoding="utf-8"))

    assert saved["revision"] != initial["revision"]
    assert saved["outfits"][0]["tags"] == ["black_corset", "red_shorts"]
    assert persisted["profiles"]["amoris"]["source_tags"] == [
        "amoris_(bang_dream!)"
    ]
    assert resolver._config["danbooru_term_mappings"] == [
        "舞会面具=masquerade_mask"
    ]


def test_character_outfit_profile_refreshes_once_per_algorithm_window() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        assert resolver.outfit_source_refresh_needed("amoris") is True
        resolver.remember_outfit_summary(
            "amoris",
            ("amoris_(bang_dream!)",),
            ("black_corset", "red_shorts"),
            {
                "sample_mode": "single_character_anchor",
                "total_posts": 100,
                "selected_posts": 21,
                "focused_posts": 21,
                "anchor_tag": "black_corset",
            },
        )
        assert resolver.outfit_source_refresh_needed("amoris") is False
        saved = json.loads(path.read_text(encoding="utf-8"))

    evidence = saved["profiles"]["amoris"]["evidence"]
    assert evidence["algorithm_version"] == 4
    assert evidence["sample_mode"] == "single_character_anchor"


def test_multiple_outfit_sources_are_extracted_and_persisted_separately(
    monkeypatch,
) -> None:
    anchors = (
        SemanticAnchor(
            "amoris",
            "outfit_source",
            "character",
            "Amoris",
            "Amoris costume source",
            ("amoris_(bang_dream!)",),
        ),
        SemanticAnchor(
            "anon_stage",
            "outfit_source",
            "character",
            "千早爱音",
            "stage outfit of Anon Chihaya",
            ("chihaya_anon",),
        ),
    )
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=("amoris_(bang_dream!)", "chihaya_anon"),
            outfit_source_tags=("amoris_(bang_dream!)", "chihaya_anon"),
            anchors=anchors,
            status="resolved",
        ),
    )

    def fake_profile(source_tag, *, outfit_kind, **_kwargs):
        if source_tag == "amoris_(bang_dream!)":
            assert outfit_kind == "default"
            return VariantOutfitProfile(
                tags=("black_corset", "red_shorts"),
                sample_mode="single_character_anchor",
                total_posts=100,
                selected_posts=21,
                focused_posts=21,
                anchor_tag="black_corset",
            )
        assert source_tag == "chihaya_anon"
        assert outfit_kind == "stage"
        return VariantOutfitProfile(
            tags=("blue_jacket", "white_skirt", "black_choker"),
            sample_mode="stage_single_character_anchor",
            total_posts=100,
            selected_posts=26,
            focused_posts=10,
            anchor_tag="blue_jacket",
        )

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", fake_profile
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        monkeypatch.setattr(resolver, "_local_cli_path", lambda: Path("cli.exe"))
        result = asyncio.run(resolver.resolve_semantic_anchors(anchors))
        saved = json.loads(path.read_text(encoding="utf-8"))

    assert result.source_outfit_profiles == (
        (
            "Amoris",
            "amoris_(bang_dream!)",
            ("black_corset", "red_shorts"),
            "default",
        ),
        (
            "千早爱音",
            "chihaya_anon",
            ("blue_jacket", "white_skirt", "black_choker"),
            "stage",
        ),
    )
    assert saved["profiles"]["amoris"]["source_tags"] == [
        "amoris_(bang_dream!)"
    ]
    assert saved["profiles"]["amoris"]["outfit_tags"] == [
        "black_corset",
        "red_shorts",
    ]
    assert saved["profiles"]["千早爱音::stage"]["source_tags"] == [
        "chihaya_anon"
    ]
    assert "haneoka_school_uniform" not in saved["profiles"][
        "千早爱音::stage"
    ]["outfit_tags"]


def test_modified_cached_outfit_uses_effective_tags_without_hard_source_anchor() -> None:
    cached = SemanticLookupResult(
        confirmed_tags=("oblivionis_(bang_dream!)", "bang_dream!"),
        outfit_source_tags=("oblivionis_(bang_dream!)",),
        outfit_profile_tags=("red_shirt", "black_skirt"),
        status="profile_cache",
    )

    context = cached.prompt_context(
        include_outfit_source_anchor=False,
        effective_outfit_tags=("pink_shirt", "black_skirt"),
        removed_outfit_tags=("red_shirt",),
    )

    assert "confirmed hard tags: bang_dream!" in context
    assert "context only; do not emit as a hard tag" in context
    assert "pink_shirt, black_skirt" in context
    assert "never restore: red_shirt" in context
