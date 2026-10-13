"""Capture only hashes, vocabulary and local read timings; no plugin startup."""
import hashlib
import json
import logging
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
PLUGIN=ROOT.parents[1]
sys.path.insert(0,str(PLUGIN))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    config_path=PLUGIN/"../../../data/config/astrbot_plugin_anima_master_config.json"
    data_path=PLUGIN/"../../plugin_data/astrbot_plugin_anima_master"
    config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    rows=config["anima_master_tag_lookup"]["danbooru_term_mappings"]
    from main import ComfyUIAgentPlugin
    from noun_translation_batch import parse_existing_mappings
    from danbooru_resolver import DanbooruResolver
    handler=SimpleNamespace(_plugin_data_path=data_path,
        _noun_catalog_path=lambda:Path.home()/".codex/skills/danbooru-tags/anima-1.0.csv",
        _existing_term_mappings=lambda:parse_existing_mappings(rows),
        _noun_web_job={"state":"idle","message":""},config={"danbooru_term_mappings":rows})
    engine=DanbooruResolver(logger=logging.getLogger("offline"),cache={},get_bool=lambda k,d:d,
        get_int=lambda k,d:d,get_float=lambda k,d:d,get_str=lambda k,d:d,
        config={**config["anima_master_tag_lookup"]},profile_cache_path=data_path/"danbooru_outfit_profiles.json")
    names=subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard"],cwd=PLUGIN).decode("utf-8").split("\0")
    checkpoint_files=list(data_path.glob("noun_translation*.json"))
    baseline={"config_sha256":digest(config_path),"original_rows":rows,
        "exact_restored_previous_rows":rows==json.loads((PLUGIN/"toolbox_import/applied.json").read_text(encoding="utf-8"))["previous_rows"],
        "existing_files":{n:digest(PLUGIN/n) for n in names if n and not n.startswith("vocabulary_quality/round_02/") and (PLUGIN/n).is_file()},
        "checkpoint_hashes":{p.name:digest(p) for p in checkpoint_files},"timings":{}}
    path=ROOT/"baseline.json"
    if path.exists(): raise SystemExit("Baseline exists; refusing overwrite")
    path.write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for label,fn in [("wardrobe",engine.wardrobe_snapshot),("batch",lambda:ComfyUIAgentPlugin._noun_translation_snapshot(handler))]:
        times=[]; size=0; counts={}
        for _ in range(3):
            start=time.perf_counter(); value=fn(); times.append((time.perf_counter()-start)*1000)
            size=len(json.dumps(value,ensure_ascii=False).encode("utf-8"))
            counts={k:len(value[k]) if isinstance(value[k],list) else value[k] for k in ("total","translated","verified","rows","outfits","outfitSets","terms") if k in value}
        baseline["timings"][label]={"milliseconds":times,"median_ms":statistics.median(times),"json_bytes":size,"counts":counts}
        print(label,baseline["timings"][label],flush=True)
    path.write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("1427 baseline exact",baseline["exact_restored_previous_rows"],"checkpoint files",len(checkpoint_files))


if __name__=="__main__": main()
