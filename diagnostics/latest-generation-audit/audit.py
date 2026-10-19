"""Read-only historical prompt audit; no model or image-generation requests."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from generation_task import insert_manual_prompt_suffix, split_manual_prompt_suffix
from danbooru_semantic import build_semantic_plan_prompt
from tag_cleaner import split_tags
from config_defaults import flatten_config
from prompt_presets import artist_preset_list

TASK_ID = "ca259e35234145c6957f3b988dae2c72"
task_path = ROOT.parents[1] / f"plugin_data/astrbot_plugin_anima_master/tasks/{TASK_ID}.json"
task = json.loads(task_path.read_text(encoding="utf-8-sig"))
s = task["prompt_summary"]
raw = json.loads(s["nai_character_plan_raw"])
with Image.open(task["outputs"][0]) as image:
    dimensions = image.size
    workflow = json.loads(image.info["prompt"])

extension = Path(r"D:\Users\69493\Desktop\ComfyUI-aki-v1.4\custom_nodes\ComfyUI_RS_NAI_API_Request")
spec = importlib.util.spec_from_file_location("audit_nai_converters", extension / "converters.py")
converters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converters)


def input_text(value):
    """Resolve only audited text-node types, never authentication inputs."""
    if isinstance(value, str):
        return value
    node = workflow[str(value[0])]
    if node["class_type"] == "Textbox":
        return input_text(node["inputs"]["text"])
    if node["class_type"] == "ComfyUIToNovelAIV4":
        return input_text(node["inputs"]["comfyui_prompt"])
    raise AssertionError(f"Unsupported text node: {node['class_type']}")


generator_id, generator = next((key, node) for key, node in workflow.items()
                               if node["class_type"] == "NovelAIGenerator")
g = generator["inputs"]
selector_id = str(g["characterPrompts"][0])
selector = workflow[selector_id]["inputs"]
main, suffix = split_manual_prompt_suffix(task["original_prompt"])
base = input_text(g["prompt"])
assert base == insert_manual_prompt_suffix(s["nai_global_prompt"], suffix)
assert dimensions == (s["nai_canvas"]["width"], s["nai_canvas"]["height"])
assert s["nai_character_count"] == 3
assert s["semantic_plan_attempt_count"] == s["nai_character_plan_attempt_count"] == 1

instances = []
for index, row in enumerate(s["nai_characters"], 1):
    text = input_text(selector[f"character{index}"])
    assert text == row["prompt"]
    assert selector[f"character{index}_x"] == row["x"]
    assert selector[f"character{index}_y"] == row["y"]
    if index > 1:
        assert selector[f"character{index}_enable"] is True
    converted = converters.ComfyUIToNovelAIV4Converter().convert_prompt(text)[0]
    assert row["name"] in converted
    assert all(tag in converted for tag in row["interaction_tags"])
    before = split_tags(raw["characters"][index - 1]["prompt"])
    after = split_tags(text)
    instances.append(dict(name=row["name"], x=row["x"], y=row["y"],
                          raw_prompt=raw["characters"][index - 1]["prompt"],
                          final_prompt=text, converted_reconstructed=converted,
                          added_tokens=[tag for tag in after if tag not in before],
                          removed_tokens=[tag for tag in before if tag not in after],
                          interaction_tags=row["interaction_tags"],
                          exactly_matches_embedded_workflow=True))

# Character/string counts are deliberately not labelled token counts. The
# historical task did not preserve each provider response's usage or request id.
writer_prompt = s["nai_character_plan_prompt"]
request_start = writer_prompt.index("User request:\n") + len("User request:\n")
request_end = writer_prompt.index("\nLocal character evidence:", request_start)
clean_request = writer_prompt[request_start:request_end].strip()
planner_reconstructed = build_semantic_plan_prompt(clean_request)
system_reconstructed = next(
    ast.literal_eval(node.value)
    for node in ast.walk(ast.parse((ROOT / "prompt_pipeline.py").read_text(encoding="utf-8")))
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "writer_system_prompt" for target in node.targets)
)
profile = next(item for item in s["danbooru_semantic_character_appearance_profiles"]
               if item["source_tag"] == "togawa_sakiko")
assert "yellow_eyes_blue_hair" in profile["appearance_tags"]
assert "yellow_eyes_blue_hair" not in raw["characters"][0]["prompt"]
assert "yellow_eyes_blue_hair" in instances[0]["final_prompt"]

usage = dict(source="user-provided DeepSeek dashboard figures; no per-call usage saved",
             total=6593, input_cached=3712, input_uncached=1634, output=1247)
usage["input_total"] = usage["input_cached"] + usage["input_uncached"]
usage["input_cache_hit_percent"] = round(100 * usage["input_cached"] / usage["input_total"], 2)
assert usage["total"] == usage["input_total"] + usage["output"]
config = flatten_config(json.loads((ROOT.parents[1] / "config/astrbot_plugin_anima_master_config.json").read_text(encoding="utf-8-sig")))
selected_style_name, selected_style_tags = artist_preset_list(config)[7]
report = dict(
    task_id=TASK_ID, time=task["time"], ok=task["ok"],
    clean_request=clean_request, hand_suffix_excluded_from_models=suffix not in writer_prompt,
    semantic_plan=json.loads(s["semantic_plan_raw"]),
    confirmed_bindings=s["confirmed_character_bindings"],
    anchor_roles=s["nai_anchor_roles"], identity_review=s["nai_identity_review"],
    attempts=dict(semantic=s["semantic_plan_attempt_count"], writer=s["nai_character_plan_attempt_count"]),
    global_raw=raw["global_prompt"], global_final=s["nai_global_prompt"],
    global_submitted_before_conversion=base,
    global_converted_reconstructed=converters.ComfyUIToNovelAIV4Converter().convert_prompt(base)[0],
    instances=instances, composition=raw["composition_analysis"],
    appearance_profiles=s["danbooru_semantic_character_appearance_profiles"],
    wardrobe_profiles=s["danbooru_semantic_source_outfit_profiles"],
    debug_field_duplicates=dict(
        writer_prompt_same=s["llm_prompt"] == writer_prompt,
        writer_raw_same=s["prompt_llm_initial_content"] == s["nai_character_plan_raw"]),
    character_counts=dict(
        writer_user=len(writer_prompt), writer_system_reconstructed=len(system_reconstructed),
        planner_user_reconstructed=len(planner_reconstructed),
        semantic_output=len(s["semantic_plan_raw"]), writer_output=len(s["nai_character_plan_raw"]),
        static_writer_user_prefix=writer_prompt.index("Actual canvas:")),
    usage=usage,
    current_config_style_8=dict(name=selected_style_name, tags=selected_style_tags),
    workflow=dict(generator_id=generator_id, selector_id=selector_id, model=g["model"],
                  width=g["width"], height=g["height"], steps=g["steps"],
                  sampler=g["sampler"], seed=g["seed"]),
    delivery=dict(status=task["delivery"]["status"], generated=task["delivery"]["generated"],
                  sent=task["delivery"]["sent"]),
    limitations=["No live model or image request was made by this audit.",
                 "Embedded PNG proves submitted ComfyUI inputs, not a captured NAI HTTP request.",
                 "Conversions were reconstructed offline using the current installed converter.",
                 "System/planner prompts are reconstructed from current code, not stored historical messages.",
                 "Per-call usage and API request ids were not stored; dashboard grouping cannot be confirmed.",
                 "This is a prompt/data-flow review, not visual image or anatomy validation."],
)
(OUT / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"task_id": TASK_ID, "attempts": report["attempts"],
                  "workflow_exact_match": all(item["exactly_matches_embedded_workflow"] for item in instances),
                  "postprocess_additions": {item["name"]: item["added_tokens"] for item in instances},
                  "character_counts": report["character_counts"], "usage": usage}, ensure_ascii=False))
