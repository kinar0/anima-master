"""Read-only vocabulary evaluation against the existing production matcher.

Only explicit offline artifacts under this directory are written. No provider,
HTTP, generation, configuration persistence or Toolbox importer is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLUGIN = ROOT.parent
sys.path.insert(0, str(PLUGIN))
from danbooru_resolver import DanbooruResolver
from mapping_syntax import parse_mapping_entry


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolver(rows):
    return DanbooruResolver(logger=logging.getLogger("offline"), cache={},
        get_bool=lambda k,d:d, get_int=lambda k,d:d, get_float=lambda k,d:d,
        get_str=lambda k,d:d, config={"danbooru_term_mappings":rows})


def merge(original, additions):
    result = list(original)
    known = resolver(original)._configured_mappings("danbooru_term_mappings")
    skipped = []
    for entry in additions:
        alias, tag = parse_mapping_entry(entry)
        if alias.lower() in known:
            skipped.append({"entry":entry, "existing":known[alias.lower()],
                            "reason":"same" if known[alias.lower()]==tag else "conflict"})
        else:
            result.append(entry)
            known[alias.lower()] = tag
    return result, skipped


def evaluate(rows, samples):
    engine = resolver(rows)
    adjunct = read(ROOT/"semantic_adjudication.json").get("samples",{}) if (ROOT/"semantic_adjudication.json").exists() else {}
    results = []
    counts = {split:Counter() for split in ("development","acceptance")}
    for sample in samples:
        matches = engine.matched_term_mappings_for_prompt(sample["text"])
        tags = {tag for alias, tag in matches}
        judgement=adjunct.get(sample["id"],{})
        focus = tags & set(sample["suitable_concepts"])
        permissible = tags & set(judgement.get("permissible",[]))
        good = focus | permissible
        bad = tags & (set(sample["forbidden_concepts"]) | set(judgement.get("wrong",[])))
        # Excludes acceptable abstention: suitable concepts on counterexamples
        # are permissible incidental references, not required coverage targets.
        missing = set(sample["suitable_concepts"]) - tags if not sample["acceptable_no_match"] else set()
        unjudged = tags - good - bad
        abstained = sample["acceptable_no_match"] and not tags
        record = {"id":sample["id"], "split":sample["split"], "matches":matches,
                  "correct_references":sorted(good), "focus_correct_references":sorted(focus),
                  "permissible_parent_references":sorted(permissible), "wrong_references":sorted(bad),
                  "missing_references":sorted(missing), "unjudged_references":sorted(unjudged),
                  "acceptable_abstention":abstained}
        results.append(record)
        c = counts[sample["split"]]
        c.update(samples=1, correct_references=len(good), focus_correct_references=len(focus),
                 permissible_parent_references=len(permissible), wrong_references=len(bad),
                 missing_references=len(missing), unjudged_references=len(unjudged),
                 acceptable_abstentions=int(abstained), samples_with_wrong_reference=int(bool(bad)))
    return {"counts":{k:dict(v) for k,v in counts.items()}, "samples":results}


def timings(rows, samples):
    cold=[]; warm=[]
    for _ in range(7):
        engine=resolver(rows)
        start=time.perf_counter()
        engine.matched_term_mappings_for_prompt(samples[0]["text"])
        cold.append((time.perf_counter()-start)*1000)
        start=time.perf_counter()
        for sample in samples:
            engine.matched_term_mappings_for_prompt(sample["text"])
        warm.append((time.perf_counter()-start)*1000)
    return {"cold_first_request_ms":cold, "warm_dataset_ms":warm,
            "cold_median_ms":statistics.median(cold), "warm_median_ms":statistics.median(warm),
            "method":"7 fresh resolvers, one cold request then identical full set; local wall time only"}


def snapshot(config):
    raw=read(config)
    section=raw["anima_master_tag_lookup"]
    rows=section["danbooru_term_mappings"]
    previous=read(PLUGIN/"toolbox_import/applied.json")["previous_rows"]
    # Store only vocabulary and hashes, never the full configuration.
    if (ROOT/"baseline_manifest.json").exists():
        raise SystemExit("Snapshot already exists; refusing to overwrite it")
    if (ROOT/"original_mappings.json").exists():
        if read(ROOT/"original_mappings.json") != rows:
            raise SystemExit("Partial snapshot differs; refusing overwrite")
    else:
        write("original_mappings.json", rows)
    names=subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard"],cwd=PLUGIN).decode("utf-8").split("\0")
    existing_files={name:sha(PLUGIN/name) for name in names if name
                    and not name.startswith("vocabulary_quality/") and (PLUGIN/name).is_file()}
    write("baseline_manifest.json", {"row_count":len(rows), "exact_restored_previous_rows":rows==previous,
        "mappings_sha256":sha(ROOT/"original_mappings.json"), "config_sha256":sha(config),
        "term_enabled_on_disk":section.get("danbooru_term_mappings_enabled",True),
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PLUGIN,text=True).strip(),
        "existing_file_hashes":existing_files, "runtime":"unknown; no restart or page save"})
    print("Snapshot:",len(rows),"exact restored:",rows==previous)


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    args=parser.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
        return
    locked=read(ROOT/"evaluation_lock.json")
    if sha(ROOT/"evaluation.jsonl") != locked["sha256"]:
        raise SystemExit("Frozen evaluation changed")
    samples=[json.loads(line) for line in (ROOT/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    original=read(ROOT/"original_mappings.json")
    base=evaluate(original,samples)
    write("baseline.json",base)
    if not (ROOT/"candidates.json").exists():
        print(json.dumps(base["counts"],ensure_ascii=False))
        return
    candidates=read(ROOT/"candidates.json")["concepts"]
    pool=[f"{alias}={c['canonical_tag']}" for c in candidates
          if c["review_decision"]=="proposed" for alias in c["aliases"]]
    # Development gate evaluates aliases independently AND the combined set.
    # Acceptance outcomes are never used to revise this pre-acceptance set.
    accepted=[]; reviews=[]
    dev=[s for s in samples if s["split"]=="development"]
    dev_base={r["id"]:r for r in base["samples"] if r["split"]=="development"}
    for entry in pool:
        merged,skipped=merge(original,[entry])
        result=evaluate(merged,dev)
        introduced=[{"id":r["id"],"tags":sorted(set(r["wrong_references"])-set(dev_base[r["id"]]["wrong_references"]))}
                    for r in result["samples"] if set(r["wrong_references"])-set(dev_base[r["id"]]["wrong_references"])]
        gain=sum(len(set(r["focus_correct_references"])-set(dev_base[r["id"]]["focus_correct_references"])) for r in result["samples"])
        decision="conflict_or_existing" if skipped else "defer_new_error" if introduced else "defer_no_measured_gain" if not gain else "retain_for_comparison"
        if decision=="retain_for_comparison": accepted.append(entry)
        reviews.append({"entry":entry,"decision":decision,"development_correct_gain":gain,
                        "development_new_errors":introduced,"merge_skipped":skipped})
    merged,skipped=merge(original,accepted)
    assert merge(merged,accepted)[0]==merged
    assert merged[:len(original)]==original
    comparison=evaluate(merged,samples)
    delta=[]
    for before,after in zip(base["samples"],comparison["samples"]):
        delta.append({"id":before["id"],"split":before["split"],
            "new_correct":sorted(set(after["correct_references"])-set(before["correct_references"])),
            "new_focus_correct":sorted(set(after["focus_correct_references"])-set(before["focus_correct_references"])),
            "lost_correct":sorted(set(before["correct_references"])-set(after["correct_references"])),
            "new_wrong":sorted(set(after["wrong_references"])-set(before["wrong_references"])),
            "removed_wrong":sorted(set(before["wrong_references"])-set(after["wrong_references"])),
            "remaining_missing":after["missing_references"],
            "unjudged":after["unjudged_references"]})
    write("candidate_review.json",reviews)
    write("comparison.json",{"evaluation_sha256":locked["sha256"],"adjudication_sha256":sha(ROOT/"semantic_adjudication.json"),"pre_acceptance_mapping_rows":accepted,
        "merge_skipped":skipped,"counts":comparison["counts"],"samples":comparison["samples"],"delta":delta,
        "acceptance_new_wrong_references":sum(len(d["new_wrong"]) for d in delta if d["split"]=="acceptance"),
        "release_gate":"failed_do_not_apply" if any(d["new_wrong"] for d in delta if d["split"]=="acceptance") else "limited_offline_pass_not_applied"})
    write("performance.json",{"baseline":timings(original,samples),"with_candidates":timings(merged,samples)})
    print("Retained for comparison",len(accepted),"rows",json.dumps(comparison["counts"],ensure_ascii=False))
    print("New acceptance errors",[(d["id"],d["new_wrong"]) for d in delta if d["split"]=="acceptance" and d["new_wrong"]])


if __name__=="__main__":
    run()
