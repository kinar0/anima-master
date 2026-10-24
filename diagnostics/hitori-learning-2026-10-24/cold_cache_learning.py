"""Replay accepted LLM1 plans through real candidate lookup and isolated learning."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(CORE), str(ROOT / "tests/live"), str(Path(__file__).parent)]

from test_deepseek_prompt_e2e import live_pipeline
from danbooru_semantic import parse_semantic_plan
from live_candidates import Audit, read


async def main():
    output = Path(__file__).parent / "cold-cache-learning"
    output.mkdir(exist_ok=False)
    production_cache = CORE / "data/plugin_data/astrbot_plugin_anima_master/danbooru_outfit_profiles.json"
    digest_before = hashlib.sha256(production_cache.read_bytes()).hexdigest()
    reports = []
    for label, task_id in (("double", "89a2f68d9dd142eb829c671f754d3157"),
                           ("single", "d7c81ee4cbc04d4d9ee6f18ca7df2f61")):
        case_dir = output / label
        case_dir.mkdir()
        builder, context = live_pipeline.__wrapped__(NS(mktemp=lambda _: case_dir))
        resolver = builder._danbooru_resolver
        profile_path = case_dir / "profiles.json"
        profiles = read(profile_path)
        removed = [key for key, row in profiles["profiles"].items()
                   if "gotoh_hitori" in row.get("source_tags", [])]
        for key in removed:
            del profiles["profiles"][key]
        profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        assert resolver._profile_cache_data is None
        async def lookup(_):
            return context.provider
        context.provider_manager = NS(get_provider_by_id=lookup)
        calls = []
        builder._llm_diagnostics = Audit(calls, case_dir / "calls.json")
        builder._llm_diagnostics.label = label
        task = read(CORE / f"data/plugin_data/astrbot_plugin_anima_master/tasks/{task_id}.json")
        summary = task["prompt_summary"]
        prompt = summary["original_prompt_head"]
        evidence = []
        try:
            anchors = resolver.prefer_cached_character_anchors(parse_semantic_plan(summary["semantic_plan_raw"], prompt))
            before = await resolver.resolve_semantic_anchors(anchors)
            assert not any(tag == "gotoh_hitori" for _, tag in before.anchor_tags)
            refined = await builder._refine_unresolved_semantic_characters(provider_id=context.provider_id,
                user_prompt=prompt, anchors=anchors, result=before, diagnostics=evidence)
            after = await resolver.resolve_semantic_anchors(refined)
            stored = read(profile_path)
            learned = [(key, row) for key, row in stored["profiles"].items()
                       if "gotoh_hitori" in row.get("source_tags", [])]
            report = {"case": label, "task_id": task_id, "removed_profiles": removed,
                      "before_bindings": before.anchor_tags, "after_bindings": after.anchor_tags,
                      "candidate_attempts": evidence, "learned_profiles": learned,
                      "character_profiles": after.character_profiles}
            reports.append(report)
            (output / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
            assert any(tag == "gotoh_hitori" for _, tag in after.anchor_tags), report
            assert learned, "real sampling did not create the isolated profile"
            print(json.dumps({"case": label, "confirmed": after.anchor_tags,
                              "learned": [(key, row.get("evidence", {}).get("focused_posts")) for key, row in learned]},
                             ensure_ascii=False), flush=True)
        finally:
            await context.provider.terminate()
    assert hashlib.sha256(production_cache.read_bytes()).hexdigest() == digest_before
    (output / "production-cache-unchanged.txt").write_text(digest_before, encoding="ascii")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
