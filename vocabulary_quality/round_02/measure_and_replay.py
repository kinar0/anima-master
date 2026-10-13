"""Post-change local timings, regression replay and preservation checks."""
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
PLUGIN=ROOT.parents[1]
sys.path.insert(0,str(PLUGIN))


def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    baseline=read(ROOT/"baseline.json")
    config_path=PLUGIN/"../../../data/config/astrbot_plugin_anima_master_config.json"
    data_path=PLUGIN/"../../plugin_data/astrbot_plugin_anima_master"
    rows=read(config_path)["anima_master_tag_lookup"]["danbooru_term_mappings"]
    from main import ComfyUIAgentPlugin
    from noun_translation_batch import parse_existing_mappings
    from vocabulary_review import VocabularyReview
    from vocabulary_quality.run_evaluation import evaluate, merge
    handler=SimpleNamespace(_plugin_data_path=data_path,
        _noun_catalog_path=lambda:Path.home()/".codex/skills/danbooru-tags/anima-1.0.csv",
        _existing_term_mappings=lambda:parse_existing_mappings(rows),
        _noun_web_job={"state":"idle","message":""},config={"danbooru_term_mappings":rows},
        _ensure_dashboard_user=lambda:None)
    times=[]
    for _ in range(7):
        start=time.perf_counter(); value=ComfyUIAgentPlugin._noun_translation_snapshot(handler); times.append((time.perf_counter()-start)*1000)
    perf={"method":"same 1325 translated/verified checkpoint rows and current 1427 mappings; seven calls",
        "before":baseline["timings"],"batch_after_ms":times,"batch_after_cold_ms":times[0],
        "batch_after_warm_median_ms":statistics.median(times[1:]),
        "full_batch_json_bytes":len(json.dumps(value,ensure_ascii=False).encode("utf-8")),
        "status_json_bytes":len(json.dumps({"job":handler._noun_web_job,"enabled":True}).encode("utf-8"))}
    service=VocabularyReview(PLUGIN/"third_party/llm_nai_toolbox/vocabulary_review_catalog.json",ROOT/"not_created_decisions.json")
    start=time.perf_counter(); catalog=service.snapshot(existing=parse_existing_mappings(rows),state="focus");
    perf["review_first_page_ms"]=(time.perf_counter()-start)*1000
    perf["review_first_page_json_bytes"]=len(json.dumps(catalog,ensure_ascii=False).encode("utf-8"))
    samples=[json.loads(line) for line in (PLUGIN/"vocabulary_quality/evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    initial=read(PLUGIN/"vocabulary_quality/comparison.json")
    joined,_=merge(rows,initial["pre_acceptance_mapping_rows"])
    regression={"purpose":"首轮 132 条已揭晓样本仅作回归，不是新批次独立验收或发布许可",
        "current_original":evaluate(rows,samples),"with_previous_33_draft_rows":evaluate(joined,samples)}
    observed={r["id"]:r for r in regression["with_previous_33_draft_rows"]["samples"]}
    regression["first_round_four_failures"]={key:observed[key] for key in ("A25","A26","A27","A28")}
    integrity={"entire_config_unchanged":sha(config_path)==baseline["config_sha256"],
        "current_1427_rows_preserved":rows==baseline["original_rows"],
        "checkpoint_hashes_unchanged":all(sha(data_path/name)==digest for name,digest in baseline["checkpoint_hashes"].items()),
        "runtime_decisions_file_created":(data_path/"vocabulary_review_decisions.json").exists(),
        "changed_existing_files":[name for name,digest in baseline["existing_files"].items() if not (PLUGIN/name).exists() or sha(PLUGIN/name)!=digest],
        "candidate_applied":False,"model_calls":0,"image_calls":0,"restart":False}
    for name,value in [("performance.json",perf),("match_regression.json",regression),("integrity.json",integrity)]:
        (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Performance",perf)
    print("Regression",regression["current_original"]["counts"],regression["with_previous_33_draft_rows"]["counts"])
    print("Original four new errors",{key:value["wrong_references"] for key,value in regression["first_round_four_failures"].items()})
    print("Integrity",integrity)


if __name__=="__main__": main()
