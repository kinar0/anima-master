"""Refresh local exact evidence for the maintenance selection, never import."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PLUGIN=ROOT.parents[1]


def main():
    catalog_path=PLUGIN/"third_party/llm_nai_toolbox/vocabulary_review_catalog.json"
    catalog=json.loads(catalog_path.read_text(encoding="utf-8"))
    query_path=ROOT/"tag_queries.json"
    query_path.write_text(json.dumps({"queries":[{"id":r["tag"],"keyword":r["tag"],"limit":5,"match_mode":"exact"} for r in catalog["records"]]},ensure_ascii=False),encoding="utf-8")
    validation_path=ROOT/"tag_validation.json"
    if validation_path.exists(): raise SystemExit("Validation evidence exists; refusing overwrite")
    skills_root=os.environ.get("COMFYUI_GOOD_ANIMA_SKILLS_DIR")
    if not skills_root: raise SystemExit("COMFYUI_GOOD_ANIMA_SKILLS_DIR is required")
    cli=Path(skills_root)/"danbooru-tags/bin/danbooru-tags.exe"
    result=subprocess.run([str(cli),"--batch-file",str(query_path),"--batch-workers","8","--match-mode","exact","--for-prompt","--json","--compact"],capture_output=True,check=True)
    validation=json.loads(result.stdout.decode("utf-8"))
    if "results" not in validation: raise SystemExit("Unexpected local CLI schema; no fallback")
    validation_path.write_text(json.dumps(validation,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for row in catalog["records"]:
        evidence=validation["results"].get(row["tag"],{})
        exact=[r for rs in evidence.get("confirmed_tags",{}).values() for r in rs if r["tag"]==row["tag"] and r["match_layer"]=="exact_tag"]
        row["local_coverage"]="exact_present" if exact else "missing_in_local_catalog"
        row["local_categories"]=[r["category"] for r in exact]
        row["confirmed_context"]=[i for i in row["issues"] if i["type"]=="user_confirmed_context"]
        row["issues"]=[i for i in row["issues"] if i["type"]=="issue_flagged"]
    catalog["source"]["local_validation_sha256"]=hashlib.sha256(validation_path.read_bytes()).hexdigest()
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    summary_path=ROOT/"reuse_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        local_exact_concepts=sum(r["local_coverage"]=="exact_present" for r in catalog["records"]),
        local_missing_concepts=sum(r["local_coverage"]!="exact_present" for r in catalog["records"]))
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Catalog local exact",summary["local_exact_concepts"],"missing",summary["local_missing_concepts"],"not applied")


if __name__=="__main__": main()
