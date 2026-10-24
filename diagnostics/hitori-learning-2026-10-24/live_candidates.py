"""Explicitly authorized identity-only live regression; never calls an image backend.

Uses the installed provider and production candidate method. Historical mode
replays the pre-request-controls Context route; current mode uses the fixed route.
Only selected request/response fields are saved, never headers or credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(CORE)]

from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from config_defaults import flatten_config
from danbooru_semantic import SemanticAnchor, lookup_semantic_anchors, resolve_local_cli_path
from llm_diagnostics import LLMDiagnostics, _response_usage
from prompt_pipeline import PromptPipeline, _extract_completion_text


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Audit(LLMDiagnostics):
    def __init__(self, rows, output):
        super().__init__(logging.getLogger("candidate_audit"), lambda k, d: False)
        self.rows, self.output = rows, output
        self.label = ""

    async def generate(self, context, **kwargs):
        row = {"label": self.label, "stage": kwargs["stage"], "requested": {
            k: kwargs[k] for k in ("prompt", "system_prompt", "max_tokens", "thinking") if k in kwargs}}
        self.rows.append(row)
        try:
            response = await super().generate(context, **kwargs)
            row.update(raw=_extract_completion_text(response), **_response_usage(response))
            return response
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            self.output.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8")


class Context:
    def __init__(self, provider, controlled):
        self.provider = provider
        if controlled:
            async def lookup(_):
                return provider
            self.provider_manager = NS(get_provider_by_id=lookup)

    async def llm_generate(self, **kwargs):
        kwargs.pop("chat_provider_id", None)
        return await self.provider.text_chat(**kwargs)


async def run(args):
    config = read(CORE / "data/cmd_config.json")
    plugin = flatten_config(read(CORE / "data/config/astrbot_plugin_anima_master_config.json"))
    provider_id = plugin.get("prompt_builder_provider_id") or config.get("agent_runner", {}).get("config", {}).get("model", {}).get("provider_id")
    model = next(p for p in config["provider"] if p["id"] == provider_id)
    source = next((p for p in config.get("provider_sources", []) if p["id"] == model.get("provider_source_id")), {})
    provider_config = {**source, **model, "id": provider_id}
    provider_config["key"] = [os.getenv(k[2:-1] if k.startswith("${") else k[1:], "")
                              if k.startswith("$") else k for k in provider_config.get("key", [])]
    assert provider_config.get("type") == "openai_chat_completion"
    assert any(provider_config.get("key", [])), "No configured credential"
    provider = ProviderOpenAIOfficial(provider_config, config.get("provider_settings", {}))
    task_dir = CORE / "data/plugin_data/astrbot_plugin_anima_master/tasks"
    cases = {label: read(task_dir / f"{task_id}.json")["prompt_summary"]["original_prompt_head"]
             for label, task_id in (("single", "d7c81ee4cbc04d4d9ee6f18ca7df2f61"),
                                    ("double", "89a2f68d9dd142eb829c671f754d3157"))}
    if args.neutral:
        cases = {"neutral_double": "千早爱音和后藤独并排站在一起", "other_double": "初音未来和绫波丽并排站在一起"}
    if args.extended:
        cases = {"reversed": "后藤独和千早爱音并排站在一起",
                 "other_double": "初音未来和绫波丽并排站在一起",
                 "variant": "重音Teto（kasane teto (sv)）和初音未来站在一起",
                 "punctuation": "小夜骑士（S:P Little Knight）和初音未来站在一起",
                 "unnamed": "一位没有名字的原创人物站在白色背景前"}
    if args.costumes:
        cases = {"oblivionis": "Oblivionis（BanG Dream!）和Mortis（BanG Dream!）站在一起",
                 "mortis": "Oblivionis（BanG Dream!）和Mortis（BanG Dream!）站在一起",
                 "borrowed_costume": "丰川祥子穿着Oblivionis（BanG Dream!）的衣服站立"}
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, wire = [], []

    async def capture(request):
        body = json.loads(request.content)
        wire.append({"at": datetime.now(timezone.utc).isoformat(),
                     "label": audit.label, "body": {k: body[k] for k in
                        ("model", "messages", "max_tokens", "thinking", "reasoning_effort") if k in body}})
        (output_dir / "wire.json").write_text(json.dumps(wire, ensure_ascii=False, indent=2), encoding="utf-8")

    provider.client._client.event_hooks["request"].append(capture)
    audit = Audit(rows, output_dir / "calls.json")
    try:
        for mode in args.modes.split(","):
            context = Context(provider, mode == "current")
            builder = PromptPipeline(context=context, config=plugin, logger=logging.getLogger("audit"),
                danbooru_resolver=object(), researcher=object(), get_bool=lambda k,d: plugin.get(k,d),
                get_int=lambda k,d: plugin.get(k,d), get_float=lambda k,d: plugin.get(k,d),
                get_str=lambda k,d: plugin.get(k,d), shorten=lambda text, limit=600: text[:limit])
            builder._llm_diagnostics = audit
            for round_no in range(args.rounds):
                for label, prompt in cases.items():
                    audit.label = f"{mode}/{label}/{round_no + 1}"
                    target = {"other_double": "绫波丽", "variant": "重音Teto（kasane teto (sv)）",
                              "punctuation": "小夜骑士（S:P Little Knight）", "unnamed": "",
                              "oblivionis": "Oblivionis", "mortis": "Mortis",
                              "borrowed_costume": "Oblivionis"}.get(label, "后藤独")
                    try:
                        candidates = await asyncio.wait_for(builder._generate_character_candidates_with_llm(
                            provider_id=provider_id, user_prompt=prompt, rejected_content="",
                            target_name=target, copyright_hints=()), timeout=180)
                    except Exception as exc:
                        candidates = ()
                        print(json.dumps({"label": audit.label, "error_type": type(exc).__name__}), flush=True)
                    rows[-1]["parsed_candidates"] = list(candidates)
                    if candidates:
                        lookup = await asyncio.to_thread(lookup_semantic_anchors,
                            (SemanticAnchor("target", "target_character", "character", target,
                                            target, candidates),), cli_path=resolve_local_cli_path(), timeout=6)
                        rows[-1]["local_confirmed"] = list(lookup.anchor_tags)
                        rows[-1]["local_missing"] = list(lookup.missing_descriptions)
                    (output_dir / "calls.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(json.dumps({"label": audit.label, "candidates": candidates,
                                      "raw": rows[-1].get("raw"), "finish": rows[-1].get("finish_reason"),
                                      "confirmed": rows[-1].get("local_confirmed"),
                                      "output_tokens": rows[-1].get("output_tokens"),
                                      "reasoning_tokens": rows[-1].get("reasoning_tokens")}, ensure_ascii=False), flush=True)
    finally:
        await provider.terminate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--modes", default="historical,current")
    parser.add_argument("--output", default="live-baseline")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--costumes", action="store_true")
    asyncio.run(run(parser.parse_args()))
