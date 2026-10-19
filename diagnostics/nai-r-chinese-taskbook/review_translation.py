"""Offline traceability checks for a translation draft, never an LLM runner.

These checks cover structure and literal preservation, not semantic equivalence
or model quality. They read saved messages and current source without importing
the generation pipeline, reading credentials, or making network calls.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def partition(text: str, headings: list[str]) -> list[str]:
    starts = [text.index(heading) for heading in headings]
    assert starts[0] == 0 and starts == sorted(set(starts))
    parts = [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]
    assert "".join(parts) == text
    return parts


def json_shapes(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_shapes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_shapes(item) for item in value]
    return type(value).__name__


def main() -> None:
    source, target = read("user.source.txt"), read("user.zh.txt")
    system_source, system_target = read("system.source.txt"), read("system.zh.txt")
    baseline = json.loads(read("baseline.json"))
    assert digest(source) == baseline["user_sha256"]
    assert digest(system_source) == baseline["system_sha256"]

    headings = [
        ("opening", "Design a complete, coherent NAI image", "请根据用户要求与本地证据"),
        ("schema", "Output JSON only, using this schema:", "只输出 JSON，使用以下结构："),
        ("languages", "Two complementary language layers", "现有提示词字符串中的两层语言："),
        ("instances", "Instances and evidence:", "可见实例与证据："),
        ("scope", "Scope and population:", "全局与逐实例作用范围、人数："),
        ("composition", "Composition:", "构图与位置："),
        ("interactions", "Interactions:", "人物互动："),
        ("writing_review", "Writing and review:", "写作与复核："),
        ("visibility", "Visible appearance:", "可见外貌："),
        ("canvas", "Actual canvas:", "实际画布："),
        ("vocabulary", "Vocabulary references", "词汇参考（"),
        ("request", "User request:", "用户要求："),
        ("local_evidence", "Local character evidence:", "本地角色证据："),
        ("original_request", "Original user request (background and trait authority):", "用户原始要求（背景与特征的权威依据）："),
        ("anchors", "Local target anchors", "本地目标锚点（"),
        ("identity_review", "Identity completeness review:", "身份完整性复核："),
        ("wardrobe", "Wardrobe authority:", "服装权限："),
    ]
    source_parts = partition(source, [item[1] for item in headings])
    target_parts = partition(target, [item[2] for item in headings])
    sections = [
        {"id": item[0], "source": old, "translation": new}
        for item, old, new in zip(headings, source_parts, target_parts, strict=True)
    ]

    original_schema = next(line for line in source_parts[1].splitlines() if line.startswith("{"))
    translated_schema = next(line for line in target_parts[1].splitlines() if line.startswith("{"))
    assert json_shapes(json.loads(original_schema)) == json_shapes(json.loads(translated_schema))
    for identifier in ("global_prompt", "characters", "name", "anchor_id", "source_text", "prompt", "interaction_tags", "position_reason", "x", "y"):
        assert identifier in translated_schema
    for marker in ("source#", "target#", "mutual#", "girl", "boy", "other", "Npeople", "Ngirls", "Nboys", "3girls", "1girls", "2girls", "composition_analysis", "position_reason", "identity_review", "visible|donor|quoted|offscreen"):
        assert marker in source and marker in target, marker

    request = source_parts[11].splitlines()[1]
    original_request = source_parts[13].splitlines()[1]
    assert request == target_parts[11].splitlines()[1]
    assert original_request == target_parts[13].splitlines()[1]
    source_anchors = [line for line in source_parts[14].splitlines() if line.startswith("- ")]
    target_anchors = [line for line in target_parts[14].splitlines() if line.startswith("- ")]
    assert source_anchors == target_anchors
    source_vocab = [line for line in source_parts[10].splitlines() if line.startswith("- ")]
    target_vocab = [line for line in target_parts[10].splitlines() if line.startswith("- ")]
    assert source_vocab == target_vocab
    for example in (
        "'3girls, upper_body, simple_background, 三人上半身入镜。'",
        "'white_shirt, sitting, 穿着白衬衫坐着。'",
        "'girl, character_a, 白衬衫，坐着'",
        "'A and B have rumpled clothes and sit together'",
        "'A wears B's clothes'",
    ):
        assert example in source and example in target, example
    for example in ("source#hand_in_mouth", "target#hand_in_mouth", "A和B衣服凌乱地坐在一起"):
        assert example in system_source and example in system_target
    # Preserved reference data includes contradictory/noisy inputs: translation
    # is not permission to clean or reinterpret the user's current evidence.
    for literal in ("(flat_chest:1.4)", "diagonal-striped_necktie", "blue_necktie", "green_necktie", "- 自然 => nature", "832x1216", "13:19", "0.6842", "22"):
        assert literal in source and literal in target, literal
    for literal in set(re.findall(r"[A-Za-z][A-Za-z0-9-]*_[A-Za-z0-9_()-]+", source_parts[12])):
        assert literal in target_parts[12], literal

    # Rebuild just the message builder offline to detect a stale saved contract.
    # No PromptPipeline, provider or image-generation entry point is executed.
    module = ast.parse((ROOT / "nai_character_mode.py").read_text(encoding="utf-8-sig"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "build_nai_character_plan_prompt")
    visibility_module = ast.parse((ROOT / "prompt_visibility.py").read_text(encoding="utf-8-sig"))
    visibility = next(ast.literal_eval(node.value) for node in visibility_module.body if isinstance(node, ast.Assign) and any(isinstance(name, ast.Name) and name.id == "APPEARANCE_VISIBILITY_POLICY" for name in node.targets))
    namespace = {"APPEARANCE_VISIBILITY_POLICY": visibility, "MAX_NAI_CHARACTERS": 22}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<saved-builder-check>", "exec"), namespace)
    rebuilt = namespace["build_nai_character_plan_prompt"]("", "", {"width": 832, "height": 1216, "aspect_ratio": "13:19", "orientation": "portrait", "character_limit": 22})
    assert rebuilt.split("\nUser request:")[0].rstrip() == source.split("\nVocabulary references")[0].rstrip()

    current_system = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse((ROOT / "prompt_pipeline.py").read_text(encoding="utf-8-sig"))) if isinstance(node, ast.Assign) and any(isinstance(name, ast.Name) and name.id == "writer_system_prompt" for name in node.targets))
    assert current_system == system_source
    changed = []
    for name, old_digest in baseline["source_hashes"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != old_digest:
            changed.append(name)
    assert not changed, changed

    report = {
        "task_id": baseline["task_id"],
        "user_sections": len(sections),
        "source_partition_complete": True,
        "schema_shape_preserved": True,
        "request_copies_preserved": 2,
        "vocabulary_rows_preserved": len(source_vocab),
        "anchor_rows_preserved": len(source_anchors),
        "saved_rules_match_current_builder": True,
        "source_files_changed_since_snapshot": changed,
        "user_characters": {"source": len(source), "translation": len(target)},
        "system_characters": {"source": len(system_source), "translation": len(system_target)},
        "llm_calls": 0,
        "image_calls": 0,
        "semantic_equivalence_proven": False,
        "model_quality_or_cost_savings_verified": False,
    }
    (HERE / "sections.json").write_text(json.dumps({"system": {"source": system_source, "translation": system_target}, "user": sections}, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = ["# NAI -r 中文任务书逐段对照", "", "独立翻译草稿，未接入运行。英文协议字段、tag、原始证据与反例按原文保留。逐段覆盖及字面检查不等于模型行为等价。", "", "## system", "", "### 原文", "", system_source, "", "### 中文", "", system_target]
    for item in sections:
        comparison.extend(["", f"## {item['id']}", "", "### 原文", "", item["source"].strip(), "", "### 中文", "", item["translation"].strip()])
    (HERE / "comparison.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
