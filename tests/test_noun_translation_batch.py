import asyncio
import csv
import json
from types import SimpleNamespace

import pytest

from noun_translation_batch import (
    build_import_plan,
    parse_existing_mappings,
    parse_verification,
    parse_translation,
    review_rows,
    select_general_tags,
    translate_with_astrbot,
    translation_review_rows,
    verify_with_astrbot,
)


def test_select_only_popular_general_tags(tmp_path):
    source = tmp_path / "tags.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([
            ["dress", "0", "1500", ""],
            ["some_character", "4", "99999", ""],
            ["skirt", "0", "999", ""],
            ["blouse", "0", "2000", ""],
            ["blouse", "0", "1800", ""],
        ])
    assert select_general_tags(source) == [
        {"tag": "blouse", "post_count": 2000},
        {"tag": "dress", "post_count": 1500},
    ]


def test_full_catalog_expansion_resumes_after_original_thousand(tmp_path):
    source = tmp_path / "tags.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([
            [f"tag_{index:04d}", "0", str(3000 - index)]
            for index in reversed(range(1060))
        ])
        writer.writerows([
            ["cold_tag", "0", "999"],
            ["some_character", "4", "99999"],
            ["some_artist", "1", "99999"],
            ["some_series", "3", "99999"],
            ["tag_0000", "0", "3000"],
        ])
    original = select_general_tags(source, limit=1000)
    expanded = select_general_tags(source)
    assert len(expanded) == 1060
    assert expanded[:1000] == original
    checkpoint = tmp_path / "checkpoint.json"
    saved = [{**row, "aliases": []} for row in original]
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")

    class FakeContext:
        def __init__(self):
            self.sent = []

        async def llm_generate(self, **kwargs):
            tags = json.loads(kwargs["prompt"].split("\n")[-1])
            self.sent.extend(tags)
            return json.dumps([{"tag": tag, "aliases": []} for tag in tags])

    context = FakeContext()
    result = asyncio.run(translate_with_astrbot(
        context, "deepseek", expanded, checkpoint,
    ))
    assert context.sent == [row["tag"] for row in expanded[1000:1050]]
    assert (result["calls"], result["newly_translated"], result["remaining"]) == (2, 50, 10)
    assert json.loads(checkpoint.read_text(encoding="utf-8"))[:1000] == saved


def test_reply_must_cover_exact_tags_and_review_collisions():
    batch = [{"tag": "large_breasts", "post_count": 2000},
             {"tag": "huge_breasts", "post_count": 1000}]
    with pytest.raises(ValueError):
        parse_translation('[{"tag":"large_breasts","aliases":["巨乳"]}]', batch)
    with pytest.raises(ValueError):
        parse_translation(json.dumps([
            {"tag": "huge_breasts", "aliases": ["巨乳"]},
            {"tag": "large_breasts", "aliases": ["巨乳"]},
        ], ensure_ascii=False), batch)
    parsed = parse_translation(json.dumps([
        {"tag": "large_breasts", "aliases": ["巨乳"]},
        {"tag": "huge_breasts", "aliases": ["巨乳"]},
    ], ensure_ascii=False), batch)
    assert all(row["review_flags"] == ["别名冲突:巨乳"] for row in review_rows(parsed))


def test_review_flags_both_sides_of_substring_collision():
    rows = [
        {"tag": "long_hair", "aliases": ["长发"], "post_count": 2000},
        {"tag": "very_long_hair", "aliases": ["超长发"], "post_count": 1000},
    ]
    flagged = review_rows(rows)
    assert all(any("包含关系" in flag for flag in row["review_flags"])
               for row in flagged)


def test_review_detects_existing_manual_mapping_conflict():
    rows = [{"tag": "large_breasts", "aliases": ["巨乳"],
             "post_count": 1000}]
    existing = parse_existing_mappings(["巨乳=huge_breasts"])
    assert review_rows(rows, existing_mappings=existing)[0][
        "review_flags"
    ] == ["别名冲突:巨乳"]


def test_batch_checkpoint_resumes_without_rebilling_completed_tags(tmp_path):
    candidates = [{"tag": "dress", "post_count": 5000},
                  {"tag": "skirt", "post_count": 3000},
                  {"tag": "blouse", "post_count": 2000}]

    class FakeContext:
        def __init__(self):
            self.calls = []

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            tags = json.loads(kwargs["prompt"].split("\n")[-1])
            return SimpleNamespace(completion_text=json.dumps([
                {"tag": tag, "aliases": [f"中文{index}"]}
                for index, tag in enumerate(tags)
            ], ensure_ascii=False))

    context = FakeContext()
    checkpoint = tmp_path / "checkpoint.json"
    first = asyncio.run(translate_with_astrbot(
        context, "deepseek", candidates, checkpoint, batch_size=2, max_batches=1,
    ))
    second = asyncio.run(translate_with_astrbot(
        context, "deepseek", candidates, checkpoint, batch_size=2, max_batches=1,
    ))
    assert (first["calls"], first["completed"], first["remaining"]) == (1, 2, 1)
    assert (second["calls"], second["completed"], second["remaining"]) == (1, 3, 0)
    assert (first["newly_translated"], second["newly_translated"]) == (2, 1)
    assert len(context.calls) == 2
    assert [row["tag"] for row in json.loads(checkpoint.read_text(encoding="utf-8"))] == [
        "dress", "skirt", "blouse",
    ]


def test_model_verification_only_accepts_source_aliases_and_import_skips_conflicts(tmp_path):
    translated = [
        {"tag": "large_breasts", "aliases": ["巨乳", "大胸"]},
        {"tag": "dress", "aliases": ["连衣裙", "裙子"]},
        {"tag": "skirt", "aliases": ["裙子", "短裙"]},
    ]
    with pytest.raises(ValueError):
        parse_verification(json.dumps([
            {"tag": "large_breasts", "approved_aliases": ["丰满"]},
            {"tag": "dress", "approved_aliases": []},
            {"tag": "skirt", "approved_aliases": []},
        ], ensure_ascii=False), translated)

    class FakeContext:
        async def llm_generate(self, **kwargs):
            tags = [row["tag"] for row in translated]
            assert all(tag in kwargs["prompt"] for tag in tags)
            return SimpleNamespace(completion_text=json.dumps([
                {"tag": "large_breasts", "approved_aliases": ["巨乳", "大胸"]},
                {"tag": "dress", "approved_aliases": ["连衣裙"]},
                {"tag": "skirt", "approved_aliases": ["裙子"]},
            ], ensure_ascii=False))

    raw_path = tmp_path / "raw.json"
    verified_path = tmp_path / "verified.json"
    raw_path.write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")
    existing = {"巨乳": "huge_breasts"}
    result = asyncio.run(verify_with_astrbot(
        FakeContext(), "deepseek", raw_path, verified_path, existing,
    ))
    assert (result["calls"], result["verified"], result["remaining"], result["newly_verified"]) == (1, 3, 0, 3)
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    plan = build_import_plan(translated, verified, existing)
    assert plan["additions"] == {
        "大胸": "large_breasts", "连衣裙": "dress", "裙子": "skirt",
    }
    assert plan["skipped"]["巨乳"] == "与手工映射冲突"
    resumed = asyncio.run(verify_with_astrbot(
        FakeContext(), "deepseek", raw_path, verified_path, existing,
    ))
    assert resumed["calls"] == 0
    assert resumed["newly_verified"] == 0


def test_review_distinguishes_existing_partial_rejected_empty_and_pending():
    translated = [
        {"tag": "dress", "aliases": ["连衣裙"]},
        {"tag": "large_breasts", "aliases": ["大胸", "巨乳"]},
        {"tag": "skirt", "aliases": ["半身裙", "短裙"]},
        {"tag": "ambiguous", "aliases": []},
        {"tag": "blouse", "aliases": ["女式衬衫"]},
        {"tag": "long_hair", "aliases": ["长发"]},
        {"tag": "very_long_hair", "aliases": ["超长发"]},
        {"tag": "short_alias", "aliases": ["衣"]},
    ]
    verified = [
        {"tag": row["tag"], "approved_aliases": row["aliases"]}
        for row in translated if row["tag"] != "blouse"
    ]
    verified[2]["approved_aliases"] = ["半身裙"]
    existing = {"连衣裙": "dress", "巨乳": "huge_breasts",
                "长发": "long_hair", "超长发": "very_long_hair"}
    plan = build_import_plan(translated, verified, existing)
    assert plan["existing"] == {"连衣裙": "dress", "长发": "long_hair", "超长发": "very_long_hair"}
    assert "连衣裙" not in plan["skipped"]
    rows = translation_review_rows(review_rows(translated, existing_mappings=existing),
                                   verified, existing, plan)
    by_tag = {row["tag"]: row for row in rows}
    assert {row["tag"] for row in rows if row["needs_manual_review"]} == {
        "large_breasts", "skirt", "ambiguous", "short_alias",
    }
    assert by_tag["dress"]["in_lexicon"]
    assert by_tag["long_hair"]["review_flags"]  # Advisory overlap is not an import failure.
    assert by_tag["long_hair"]["in_lexicon"]
    assert by_tag["large_breasts"]["ready"]
    assert by_tag["skirt"]["ready"]
    assert "huge_breasts" in by_tag["large_breasts"]["manual_review_reasons"][0]
    assert by_tag["blouse"]["approved_aliases"] is None
    assert not by_tag["blouse"]["in_lexicon"]

    # Import updates classification without rewriting translation checkpoints.
    existing.update(plan["additions"])
    after = translation_review_rows(translated, verified, existing,
                                    build_import_plan(translated, verified, existing))
    assert not any(row["ready"] for row in after)
    assert {row["tag"] for row in after if row["needs_manual_review"]} == {
        "large_breasts", "skirt", "ambiguous", "short_alias",
    }
    # A manual exact mapping resolves even a model-rejected alias.
    existing["短裙"] = "skirt"
    resolved = translation_review_rows(translated, verified, existing,
                                       build_import_plan(translated, verified, existing))
    assert resolved[2]["in_lexicon"]
    assert not resolved[2]["needs_manual_review"]


def test_review_shared_alias_is_existing_only_for_its_actual_owner():
    translated = [{"tag": tag, "aliases": ["裙子"]} for tag in ("dress", "skirt")]
    verified = [{"tag": row["tag"], "approved_aliases": row["aliases"]}
                for row in translated]
    existing = {"裙子": "skirt"}
    rows = translation_review_rows(translated, verified, existing,
                                   build_import_plan(translated, verified, existing))
    assert rows[0]["needs_manual_review"]
    assert not rows[0]["in_lexicon"]
    assert rows[1]["in_lexicon"]
    assert not rows[1]["needs_manual_review"]


def test_verification_only_sends_new_rows_after_partial_checkpoint(tmp_path):
    translated = [
        {"tag": "dress", "aliases": ["连衣裙"]},
        {"tag": "skirt", "aliases": ["半身裙"]},
        {"tag": "blouse", "aliases": ["女式衬衫"]},
    ]
    raw_path = tmp_path / "raw.json"
    verified_path = tmp_path / "verified.json"
    raw_path.write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")
    verified_path.write_text(json.dumps([
        {"tag": "dress", "approved_aliases": ["连衣裙"]},
        {"tag": "skirt", "approved_aliases": ["半身裙"]},
    ], ensure_ascii=False), encoding="utf-8")

    class FakeContext:
        calls = 0

        async def llm_generate(self, **kwargs):
            self.calls += 1
            assert '"tag": "blouse"' in kwargs["prompt"]
            assert '"tag": "dress"' not in kwargs["prompt"]
            return SimpleNamespace(completion_text=json.dumps([
                {"tag": "blouse", "approved_aliases": ["女式衬衫"]},
            ], ensure_ascii=False))

    context = FakeContext()
    result = asyncio.run(verify_with_astrbot(
        context, "deepseek", raw_path, verified_path, {},
    ))
    assert context.calls == 1
    assert (result["newly_verified"], result["verified"], result["calls"]) == (1, 3, 1)
