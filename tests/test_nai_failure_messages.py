from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from comfyui_runtime import ComfyUIRuntime  # noqa: E402


@pytest.mark.parametrize(
    "cause, expected",
    [
        ("nai_character_plan_invalid_interaction_pairs: source#hug has no target#hug",
         "角色互动方向标签未能正确配对"),
        ("nai_character_plan_invalid_json", "不是完整、有效的 JSON"),
        ("nai_character_plan_invalid_structure", "输出结构无效"),
        ("nai_character_plan_character_count", "数量不符合当前模型限制"),
        ("nai_character_plan_invalid_character", "缺少有效名称或提示词"),
        ("nai_character_plan_invalid_position", "坐标缺失或不在 0–1 范围内"),
        ("nai_character_plan_missing_composition_analysis", "缺少完整构图分析"),
        ("nai_character_plan_missing_global_prompt", "缺少全局提示词"),
        ("nai_character_plan_missing_position_reason", "缺少角色位置说明"),
        ("provider request failed", "具体原因请查看任务记录"),
        ("", "具体原因请查看任务记录"),
    ],
)
def test_nai_failure_reports_cause_without_guessing_refusal_or_five_person_limit(
    cause, expected
):
    runtime = ComfyUIRuntime.__new__(ComfyUIRuntime)
    reason = runtime.failure_reason({
        "error": "nai_character_plan_failed", "llm_error": cause,
    })

    assert expected in reason
    assert "五个" not in reason
    assert "重试" not in reason
    assert "模型拒绝" not in reason


def test_nai_specific_error_can_be_reported_directly():
    runtime = ComfyUIRuntime.__new__(ComfyUIRuntime)
    assert "角色互动方向标签未能正确配对" in runtime.failure_reason({
        "error": "nai_character_plan_invalid_interaction_pairs: source#hug",
    })


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"ok": False, "error": "nai_character_plan_failed",
          "llm_error": "nai_character_plan_invalid_interaction_pairs: source#hug"},
         "提示词构建失败：-r 的角色互动方向标签未能正确配对，已停止生成。"),
        ({"ok": False, "error": "model_refused_generation"}, "模型拒绝了生成"),
        ({"ok": False, "error": "workflow_failed"},
         "ComfyUI 操作失败：ComfyUI 工作流执行失败。"),
    ],
)
def test_delivery_distinguishes_planning_refusal_and_workflow_failures(payload, expected):
    class Event:
        def __init__(self):
            self.sent = []

        def plain_result(self, text):
            return text

        async def send(self, result):
            self.sent.append(result)

    runtime = ComfyUIRuntime.__new__(ComfyUIRuntime)
    runtime._bool = lambda _key, default: default
    event = Event()

    assert asyncio.run(runtime.send_payload(event, payload)) == expected
    assert event.sent == [expected]
    assert payload["delivery"]["generated"] is False
    assert payload["delivery"]["sent"] is False
