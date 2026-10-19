"""Offline audit of stored failures; no provider, ComfyUI or config mutations.

Only structural metadata is exported. Original requests, dialogue and generated
prompts remain in their original task records.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from danbooru_semantic import SemanticAnchor, SemanticLookupResult
from danbooru_resolver import DanbooruResolver
from nai_character_mode import parse_nai_character_plan
from nai_text import extract_nai_text_request
from tag_cleaner import split_tags

# Load the unchanged pure review functions without importing AstrBot's service
# initialization (which imports its database and other unrelated dependencies).
_functions = {"_normalized_character_key", "confirmed_semantic_character_tags",
              "fallback_target_anchors_from_semantic_evidence", "nai_local_identity_mentions",
              "nai_confirmed_identity_anchors", "nai_cross_panel_instance_issues", "review_nai_identities"}
_tree = ast.parse((ROOT / "prompt_pipeline.py").read_text(encoding="utf-8"))
_nodes = [node for node in _tree.body
          if isinstance(node, ast.FunctionDef) and node.name in _functions
          or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
              and target.id == "_NAI_PANEL_HEADING" for target in node.targets)]
_module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                         *_nodes], type_ignores=[])
exec(compile(ast.fix_missing_locations(_module), str(ROOT / "prompt_pipeline.py"), "exec"), globals())

OUT = Path(__file__).parent
TASKS = ROOT.parents[1] / "plugin_data/astrbot_plugin_anima_master/tasks"
DAY = "2026-10-19"
FIXED = "--expect-fixed" in sys.argv


def read_json(raw):
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I))


def inspect_raw(raw):
    result = {"chars": len(raw), "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    try:
        data = read_json(raw)
    except json.JSONDecodeError as exc:
        result.update(json_error=exc.msg, error_offset=exc.pos,
                      ends_with_object_close=raw.rstrip().endswith("}"))
        # Diagnosis only: prove the syntactic error location, never use this
        # modified output to generate an image or persist a repaired prompt.
        candidate, count = re.subn(r'"\s*}\s*,\s*"interaction_tags"',
                                   '","interaction_tags"', raw)
        try:
            parsed = read_json(candidate)
        except json.JSONDecodeError:
            pass
        else:
            result.update(extra_closures_before_interaction_tags=count,
                          json_parses_after_removing_those_closures=True,
                          diagnostic_character_count=len(parsed.get("characters", [])))
        return result
    result.update(
        json_valid=True,
        analysis_keys=list(data.get("composition_analysis", {})),
        character_count=len(data.get("characters", [])),
        character_fields=[list(row) for row in data.get("characters", [])],
        coordinates=[[row.get("x"), row.get("y")] for row in data.get("characters", [])],
        review_names=[row.get("name") for row in data.get("identity_review", [])],
        panel_labels=[row.get("panel") for row in data.get("text_regions", [])],
    )
    return result


def replay_review(task):
    """Reconstruct recorded identity inputs, not the whole historical pipeline."""
    s = task["prompt_summary"]
    source = s["nai_character_plan_prompt"].split("User request:\n", 1)[1].split(
        "\nLocal character evidence:", 1)[0].strip()
    anchors = tuple(SemanticAnchor(row["id"], row["role"], "character", row["source"], "",
                                   tuple(row["candidates"])) for row in s["nai_anchor_roles"])
    bindings = {row["source"]: row["canonical_tag"] for row in s["confirmed_character_bindings"]}
    result = SemanticLookupResult(
        confirmed_tags=tuple(s["danbooru_semantic_confirmed_tags"]),
        outfit_source_tags=tuple(s["danbooru_semantic_outfit_sources"]),
        named_outfit_tags=tuple(s["danbooru_semantic_named_outfit_tags"]),
        character_appearance_profiles=tuple(
            (tuple(row["aliases"]), row["source_tag"], tuple(row["appearance_tags"]))
            for row in s["danbooru_semantic_character_appearance_profiles"]),
        source_outfit_profiles=tuple(
            (row["alias"], row["source_tag"], tuple(row["outfit_tags"]), row["qualifier"])
            for row in s["danbooru_semantic_source_outfit_profiles"]),
        anchors=anchors,
        anchor_tags=tuple((row.anchor_id, bindings[row.source_text])
                          for row in anchors if row.source_text in bindings),
    )
    mentions = nai_local_identity_mentions(source, result)
    reconstructed = [{"source": row.source_text, "canonical_tags": list(row.candidates)}
                     for row in mentions]
    if not FIXED:
        assert reconstructed == s["nai_identity_mentions"], task["task_id"]
    _, text_request = extract_nai_text_request(source)
    text_request = text_request.require_known_speakers(tuple(
        row.source_text for row in anchors if row.role == "target_character"))
    parsed = parse_nai_character_plan(s["nai_character_plan_retry_raw"], text_request=text_request)
    identity_errors = review_nai_identities(parsed, mentions, result, anchors, source,
                                           scrutinize_donors=False)
    assert list(identity_errors) == ([] if FIXED else s["nai_identity_retry_validation_errors"]), task["task_id"]
    assert list(parsed.get("text_validation_errors", ())) == s["nai_text_retry_validation_errors"]
    known_tags = {tag for row in mentions for tag in row.candidates}
    # Remove only optional rows about identities outside this checker's known
    # roster to identify the veto. This is an in-memory counterfactual, not a fix.
    diagnostic = copy.deepcopy(parsed)
    diagnostic["identity_review"] = [row for row in diagnostic["identity_review"]
                                     if row["name"] in known_tags
                                     or row["name"] in {item.source_text for item in mentions}]
    after = review_nai_identities(diagnostic, mentions, result, anchors, source,
                                 scrutinize_donors=False)
    panel_grounding = []
    for row in parsed.get("text_regions", []):
        panel_grounding.append({"id": row["id"], "panel": row["panel"],
                                "in_registry_context": row["panel"] in text_request.context(row["id"])})
    diagnostic_regions = copy.deepcopy(parsed.get("text_regions", []))
    for row in diagnostic_regions:
        if row["panel"] and row["panel"] not in text_request.context(row["id"]):
            row["panel"] = ""
    _, panel_counterfactual_errors = text_request.validate_regions(
        diagnostic_regions, parsed["characters"], parsed["global_prompt"])
    return {
        "identity_errors_match_expected_mode": True,
        "historical_identity_errors": s["nai_identity_retry_validation_errors"],
        "known_profile_mentions": reconstructed,
        "confirmed_bindings": s["confirmed_character_bindings"],
        "identity_errors": list(identity_errors),
        "text_errors": list(parsed.get("text_validation_errors", ())),
        "identity_errors_without_out_of_roster_review_rows": list(after),
        "panel_grounding": panel_grounding,
        "text_errors_without_invented_panel_labels": list(panel_counterfactual_errors),
    }


def main():
    rows = [json.loads(path.read_text(encoding="utf-8-sig")) for path in TASKS.glob("*.json")]
    today = sorted((row for row in rows if row.get("time", "").startswith(DAY)), key=lambda row: row["time"])
    explicit = [row for row in today if re.search(r"(?<!\S)-r(?!\S)", row.get("original_prompt", ""))]
    failures = [row for row in explicit if row.get("ok") is False]
    evidence = []
    for task in failures:
        s = task["prompt_summary"]
        assert s.get("final_prompt_chars") == 0 and task["outputs"] == []
        if "nai_character_plan_retry_raw" not in s:
            try:
                parse_nai_character_plan(s["nai_character_plan_initial_raw"])
            except ValueError as exc:
                assert str(exc) == s["llm_error"], task["task_id"]
            else:
                raise AssertionError(task["task_id"])
        item = {"task_id": task["task_id"], "time": task["time"], "error": task["error"],
                "llm_error": s.get("llm_error"), "writer_attempts": s.get("nai_character_plan_attempt_count"),
                "final_prompt_chars": s.get("final_prompt_chars"), "outputs": task["outputs"],
                "elapsed_seconds": task.get("elapsed_seconds"),
                "recorded_llm_ok": s.get("llm_ok"),
                "raw": {key: inspect_raw(s[key]) for key in (
                    "nai_character_plan_initial_raw", "nai_character_plan_retry_raw") if key in s},
                "later_exact_success": [{"task_id": other["task_id"], "time": other["time"]}
                    for other in today if other.get("ok") is True and other["time"] > task["time"]
                    and other.get("original_prompt") == task.get("original_prompt")]}
        if "nai_character_plan_retry_raw" in s:
            item["replay"] = replay_review(task)
        evidence.append(item)
    report = {
        "day": DAY, "timezone": "Asia/Shanghai", "latest_task_start": today[-1]["time"],
        "all_tasks": len(today), "explicit_r_tasks": len(explicit),
        "explicit_r_failures": len(failures),
        "failure_rate_percent": round(len(failures) / len(explicit) * 100, 2),
        "excluded_failures": [{"task_id": task["task_id"], "time": task["time"], "error": task["error"]}
                              for task in today if task.get("ok") is False and task not in failures],
        "cases": evidence,
        "limits": ["Historical tasks have no immutable source-code revision or per-call usage attached.",
                   "Review replay uses current code and recorded identity inputs, not the full runtime.",
                   "No model, image request, config change, or running-service reload was performed.",
                   "No original requests, dialogue, or generated prompt strings are exported here."],
    }
    (OUT / ("fixed-evidence.json" if FIXED else "evidence.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("all_tasks", "explicit_r_tasks", "explicit_r_failures",
                                                   "failure_rate_percent", "latest_task_start")}, ensure_ascii=False))
    print("Historical reviews checked:", sum("replay" in case for case in evidence),
          "mode:", "fixed" if FIXED else "baseline")


if __name__ == "__main__":
    main()
