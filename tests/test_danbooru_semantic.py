from __future__ import annotations

import sys
import tempfile
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from danbooru_semantic import parse_semantic_plan  # noqa: E402
from danbooru_resolver import DanbooruResolver  # noqa: E402


def test_semantic_source_phrase_matches_ascii_case_insensitively() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"source","role":"outfit_source",'
        '"group":"character","source_text":"oblivionis",'
        '"description":"Oblivionis outfit",'
        '"candidates":["oblivionis_(bang_dream!)"]}]}',
        "穿着Oblivionis服装",
    )

    assert plan and plan[0].role == "outfit_source"


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
