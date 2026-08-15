from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from generation_task import (  # noqa: E402
    GenerationTaskRunner,
    insert_manual_prompt_suffix,
    split_manual_prompt_suffix,
)


def test_manual_suffix_is_inserted_immediately_before_nltags() -> None:
    prompt, suffix = split_manual_prompt_suffix(
        "蓝色连衣裙女孩 # masterpiece, (dramatic lighting:1.2)"
    )

    assert prompt == "蓝色连衣裙女孩"
    assert suffix == "masterpiece, (dramatic lighting:1.2)"
    assert insert_manual_prompt_suffix(
        "1girl, solo, blue dress, Nltags: A girl poses.", suffix
    ) == (
        "1girl, solo, blue dress, masterpiece, (dramatic lighting:1.2), "
        "Nltags: A girl poses."
    )


def test_manual_suffix_is_appended_when_nltags_are_absent() -> None:
    assert insert_manual_prompt_suffix(
        "1girl, solo, blue dress", "masterpiece, backlighting"
    ) == "1girl, solo, blue dress, masterpiece, backlighting"


def test_generation_ignores_manual_suffix_until_comfyui_submission() -> None:
    class _Recorder:
        def build_generation_start(self, **_kwargs):
            return {"task_id": "task"}

        def mark_prompt_built(self, task, summary) -> None:
            task["prompt_summary"] = dict(summary)

        def mark_completed(self, task, **_kwargs) -> None:
            self.completed = dict(task)

        def write(self, task) -> None:
            self.written = dict(task)

    class _Inputs:
        last_summary = {}

    class _Reference:
        last_summary = {}

    build_calls: list[tuple[str, str]] = []
    tool_calls: list[list[str]] = []
    summary = {"llm_failed": False}

    async def build_prompt(_event, prompt, **kwargs):
        build_calls.append((prompt, kwargs["original_user_prompt"]))
        return "1girl, solo, blue dress, Nltags: A girl stands by a window."

    async def run_tool(args):
        tool_calls.append(list(args))
        return {"ok": True, "outputs": []}

    runner = GenerationTaskRunner(
        task_recorder=_Recorder(),
        image_inputs=_Inputs(),
        reference_context=_Reference(),
        is_allowed=lambda _event: True,
        ensure_ready=lambda _event: asyncio.sleep(0, result={"ok": True}),
        wants_reference_image=lambda _prompt: False,
        augment_reference_image=lambda _event, prompt: asyncio.sleep(
            0, result=prompt
        ),
        augment_quoted_spell=lambda _event, prompt: prompt,
        build_prompt=build_prompt,
        prompt_summary=lambda: summary,
        run_tool=run_tool,
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=lambda text, limit: text[:limit],
    )

    payload = asyncio.run(
        runner.generate_payload(
            object(),
            "蓝色连衣裙女孩#masterpiece, (dramatic lighting:1.2)",
        )
    )

    assert payload["ok"] is True
    assert build_calls == [("蓝色连衣裙女孩", "蓝色连衣裙女孩")]
    prompt_index = tool_calls[0].index("--prompt") + 1
    assert tool_calls[0][prompt_index] == (
        "1girl, solo, blue dress, masterpiece, (dramatic lighting:1.2), "
        "Nltags: A girl stands by a window."
    )
