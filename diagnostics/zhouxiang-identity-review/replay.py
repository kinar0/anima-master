"""Offline identity-path diagnosis; neutral fixtures, no provider or image API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT.parents[2])]

from danbooru_resolver import DanbooruResolver
from danbooru_semantic import SemanticLookupResult
from test_prompt_direct_writer import Logger, Resolver, plan, run, semantic_character, with_identity_review
from prompt_pipeline import fallback_target_anchors_from_semantic_evidence, saved_appearance_for_names

DATA = ROOT.parents[2] / "data/plugin_data/astrbot_plugin_anima_master"
snapshot = json.loads((DATA / "danbooru_outfit_profiles.json").read_text(encoding="utf-8-sig"))
local = DanbooruResolver(
    logger=Logger(), cache={}, config={}, get_bool=lambda k, d: d,
    get_int=lambda k, d: d, get_float=lambda k, d: d, get_str=lambda k, d: d,
)
local._profile_cache_data = snapshot
VARIANT = "togawa_sakiko_(master_of_melodia)"


class LocalOnly(Resolver):
    def cached_outfit_profiles_for_prompt(self, prompt):
        return local.cached_outfit_profiles_for_prompt(prompt)

    def prefer_cached_character_anchors(self, anchors):
        return local.prefer_cached_character_anchors(anchors)


request = "-r 丰川祥子和粥祥衣服略显凌乱地坐在沙发上，千早爱音在旁边拿着自拍杆。"
complete = [semantic_character(name) for name in ("丰川祥子", "粥祥", "千早爱音")]
omitted = [semantic_character(name) for name in ("丰川祥子", "千早爱音")]
misclassified = [
    dict(semantic_character("丰川祥子", "粥祥衣服"), clothing_source="粥祥"),
    semantic_character("千早爱音"),
]
two = [("togawa_sakiko", "girl, sitting, 丰川祥子坐在沙发上。", .3),
       ("chihaya_anon", "girl, holding_selfie_stick, 千早爱音拿着自拍杆。", .7)]
three = [two[0], (VARIANT, "girl, blue_hair, horns, sitting, 粥祥坐在沙发上。", .5), two[1]]
unmapped = [two[0], ("粥祥", "girl, brown_hair, sitting, 粥祥坐在沙发上。", .5), two[1]]
cases = [
    ("complete_roster", request, complete, three),
    ("omitted_rescued_by_reference", request, omitted, three),
    ("reference_not_mandatory_writer_omits", request, omitted, two),
    ("hypothetical_source_misclassification", request, misclassified, two),
    ("hypothetical_source_and_unmapped_name", request, misclassified, unmapped),
    ("legitimate_offscreen_donor", "-r 丰川祥子穿粥祥衣服坐在沙发上，千早爱音在旁边拿着自拍杆。",
     misclassified, two),
]
reports = []
for name, text, first, second in cases:
    output = plan(second, f"{len(second)}girls, living_room, 几人在沙发旁合影。")
    repaired = plan(three, "3girls, living_room, 三人在客厅合影。")
    if name.startswith("hypothetical"):
        repaired = with_identity_review(repaired, VARIANT, "visible", request.removeprefix("-r "))
    if name == "legitimate_offscreen_donor":
        output = with_identity_review(output, VARIANT, "donor", text.removeprefix("-r "))
    result, calls = run(text, {"characters": first}, output,
                        {"debug_prompt_enabled": True}, resolver=LocalOnly(), repair=repaired)
    s = result.summary
    report = dict(case=name, hypothetical=name.startswith("hypothetical"),
                  neutral_request=text, synthetic_llm1=first, synthetic_llm2=json.loads(output),
                  successful=bool(result.final_prompt), error=s.get("llm_error"),
                  bindings=s.get("confirmed_character_bindings"),
                  references=s.get("nai_local_profile_references"),
                  bound_references=s.get("nai_bound_local_profile_references"),
                  identity_review=s.get("nai_identity_review"),
                  initial_identity_errors=s.get("nai_identity_initial_validation_errors"),
                  final_characters=s.get("nai_characters"), call_count=len(calls))
    reports.append(report)
    print(name, json.dumps({k: report[k] for k in ("successful", "bindings", "references", "error")}, ensure_ascii=False))

assert all(row["successful"] for row in reports)
assert [row["call_count"] for row in reports] == [2, 2, 3, 3, 3, 2]
assert reports[1]["references"] and reports[1]["bound_references"]
assert all(len(row["final_characters"]) == 3 for row in reports[:5])
assert reports[3]["references"] and reports[4]["references"]
assert reports[4]["final_characters"][1]["name"] == VARIANT
assert "brown_hair" not in reports[4]["final_characters"][1]["prompt"]
assert len(reports[5]["final_characters"]) == 2 and not reports[5]["bound_references"]
(Path(__file__).parent / "after-fix-offline-replay.json").write_text(
    json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8",
)

# Preserve only identity diagnostics from historical requests, without copying
# their unrelated scene descriptions into this report.
import hashlib
historical = []
for task_id in ("db7305114a274d2484675727044a5f09", "a14b0db09eba48059a1116c9c1831c62",
                "1ebf558cc89b4410a4aed6fe58e55b58", "bc481e9f06614c1e904cd901af96146f",
                "3a61b2dc81ca41ec99506a3f59a9f7d4"):
    task = json.loads((DATA / "tasks" / f"{task_id}.json").read_text(encoding="utf-8-sig"))
    s = task["prompt_summary"]
    lookup = SemanticLookupResult(
        confirmed_tags=tuple(s["danbooru_semantic_confirmed_tags"]),
        named_outfit_tags=tuple(s["danbooru_semantic_named_outfit_tags"]),
        outfit_source_tags=tuple(s["danbooru_semantic_outfit_sources"]),
        character_appearance_profiles=tuple(
            (tuple(row["aliases"]), row["source_tag"], tuple(row["appearance_tags"]))
            for row in s["danbooru_semantic_character_appearance_profiles"]),
        source_outfit_profiles=tuple(
            (row["alias"], row["source_tag"], tuple(row["outfit_tags"]), row["qualifier"])
            for row in s["danbooru_semantic_source_outfit_profiles"]),
    )
    recovered = fallback_target_anchors_from_semantic_evidence(task["original_prompt"], (), lookup)
    assert any(a.source_text == "粥祥" and VARIANT in a.candidates for a in recovered)
    alias_owners = {source for aliases, source, _ in lookup.character_appearance_profiles if "粥祥" in aliases}
    assert alias_owners == {VARIANT} and VARIANT not in lookup.named_outfit_tags
    historical.append(dict(
        task_id=task_id, time=task["time"], ok=task["ok"],
        request_sha256=hashlib.sha256(task["original_prompt"].encode()).hexdigest(),
        cached_profile_found=True, alias_owners=sorted(alias_owners),
        confirmed_character_bindings=s["confirmed_character_bindings"],
        local_references=s["nai_local_profile_references"],
        bound_references=s["nai_bound_local_profile_references"],
        final_names=[row["name"] for row in s["nai_characters"]],
        raw_planner_saved="semantic_plan_raw" in s,
        planner_validation_errors=s["semantic_plan_validation_errors"],
        identity_only_appearance_lookup=saved_appearance_for_names(lookup, ("粥祥",)),
        alias_plus_canonical_appearance_lookup=saved_appearance_for_names(lookup, ("粥祥", VARIANT)),
    ))
assert historical[2]["request_sha256"] == historical[3]["request_sha256"]
(Path(__file__).parent / "after-fix-historical-lookup.json").write_text(
    json.dumps(historical, ensure_ascii=False, indent=2), encoding="utf-8",
)
