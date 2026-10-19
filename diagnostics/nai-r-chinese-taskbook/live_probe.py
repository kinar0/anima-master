"""One explicitly authorized live writer call, then offline pipeline replay.

Never generates an image, edits plugin configuration, or installs the draft.
Provider credentials are read through the existing live fixture and never saved.
The first-stage response is historical evidence, not a new model call.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CORE = ROOT.parents[2]
for directory in (ROOT, CORE):
    sys.path.insert(0, str(directory))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def find(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            result = find(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find(child, key)
            if result is not None:
                return result


def load_fixture():
    spec = importlib.util.spec_from_file_location("taskbook_existing_live_fixture", ROOT / "tests/live/test_deepseek_prompt_e2e.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    replay_only = len(sys.argv) == 3 and sys.argv[1] == "--replay-report"
    output = Path(sys.argv[2]).resolve() if replay_only else HERE / "live" / stamp
    assert output.is_relative_to(HERE / "live")
    if not replay_only:
        output.mkdir(parents=True, exist_ok=False)
    module = load_fixture()
    # Keep SDK debug request dumps out of diagnostic console output.
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    class WorkspaceFactory:
        def mktemp(self, name):
            directory = output / (name + "-offline-" + stamp if replay_only else name)
            directory.mkdir(exist_ok=False)
            return directory

    pipeline, context = module.live_pipeline.__wrapped__(WorkspaceFactory())
    from llm_diagnostics import _response_usage
    from nai_character_mode import parse_nai_character_plan

    submitted = {
        "system_prompt": (HERE / "system.zh.txt").read_text(encoding="utf-8"),
        "prompt": (HERE / "user.zh.txt").read_text(encoding="utf-8"),
        "max_tokens": 6920,
        "thinking": {"type": "disabled"},
    }
    if not replay_only:
        save(output / "submitted.json", submitted)
    report = {
        "output_directory": str(output),
        "source_task_id": "76ffd0c270f349d6810153167ec195cc",
        "provider_id": context.provider_id,
        "live_application_calls": 0,
        "image_generation_calls": 0,
        "production_configuration_changed": False,
        "baseline_origin": "User-supplied usage logs for the recorded English writer call; not a new English control call.",
        "baseline": {
            "input_tokens": 5077, "input_cached_tokens": 4864,
            "input_uncached_tokens": 213, "output_tokens": 1883,
            "total_tokens": 6960, "elapsed_ms": 7951.81,
            "finish_reason": "stop",
        },
    }

    async def live():
        started = perf_counter()
        try:
            report["live_application_calls"] += 1
            response = await context.llm_generate(chat_provider_id=context.provider_id, **submitted)
            report["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
            report["usage"] = _response_usage(response)
            raw = str(response.completion_text or "")
            (output / "response.txt").write_text(raw, encoding="utf-8")
            raw_completion = getattr(response, "raw_completion", None)
            report["returned_model"] = getattr(raw_completion, "model", None)
            report["reasoning_returned"] = bool(getattr(response, "reasoning_content", None))
            return raw
        finally:
            await context.provider.terminate()

    try:
        if replay_only:
            report = read_json(output / "report.json")
            raw = (output / "response.txt").read_text(encoding="utf-8")
            asyncio.run(context.provider.terminate())
        else:
            raw = asyncio.run(live())
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        save(output / "report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
        raise SystemExit(1) from None

    save(output / "report.json", report)  # Preserve real usage before offline validation.
    try:
        parsed = parse_nai_character_plan(raw, character_limit=22)
        report["parser"] = {
            "ok": True,
            "instance_count": len(parsed["characters"]),
            "identity_counts": dict(Counter(row["name"] for row in parsed["characters"])),
            "global_has_chinese": bool(re.search(r"[\u3400-\u9fff]", parsed["global_prompt"])),
            "all_instances_have_chinese": all(re.search(r"[\u3400-\u9fff]", row["prompt"]) for row in parsed["characters"]),
        }
        save(output / "parsed.json", parsed)
    except Exception as exc:
        report["parser"] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

    # Replay the actual new writer reply through current filtering/assembly.
    # Local profile evidence is a copied snapshot from the existing live fixture.
    # Disable network tag lookup and researcher calls. No repair is requested live.
    task = read_json(CORE / "data/plugin_data/astrbot_plugin_anima_master/tasks/76ffd0c270f349d6810153167ec195cc.json")
    original_semantic = find(task, "semantic_plan_raw")
    replay_outputs = [original_semantic, raw]
    replay_calls = []

    async def replay_generate(**kwargs):
        replay_calls.append({"prompt": kwargs.get("prompt"), "system_prompt": kwargs.get("system_prompt")})
        if not replay_outputs:
            raise RuntimeError("Offline replay requested repair; extra live calls are disabled")
        return SimpleNamespace(completion_text=replay_outputs.pop(0))

    class NoResearch:
        def plan(self, _prompt):
            return SimpleNamespace(use_web_search=False, use_deep_thinking=False, search_reason="", thinking_reason="")

    context.llm_generate = replay_generate
    pipeline._researcher = NoResearch()
    pipeline._danbooru_resolver.semantic_lookup_available = lambda: False
    pipeline.config["debug_prompt_enabled"] = True
    pipeline.config["debug_llm_reasoning_enabled"] = False
    pipeline.config["debug_llm_usage_enabled"] = False
    try:
        result = asyncio.run(pipeline.build(module._Event(), task["original_prompt"]))
        save(output / "offline-replay.json", {"final_prompt": result.final_prompt, "summary": result.summary, "replayed_calls": replay_calls})
        rows = result.summary.get("nai_characters", [])
        report["offline_replay"] = {
            "ok": bool(result.final_prompt),
            "replayed_stage_count": len(replay_calls),
            "extra_live_calls": 0,
            "instance_count": len(rows),
            "identity_counts": dict(Counter(row["name"] for row in rows)),
            "error": result.summary.get("llm_error", ""),
            "identity_validation_errors": result.summary.get("nai_identity_initial_validation_errors", []),
            "population_correction": result.summary.get("nai_population_correction"),
            "evidence_origin": "Historical LLM1 response, current configuration and copied current profiles; regenerated English taskbook is used only for offline post-processing context, never sent to a model.",
        }
    except Exception as exc:
        report["offline_replay"] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "extra_live_calls": 0}
    save(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
