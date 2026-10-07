from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from comfyui_runtime import ComfyUIRuntime  # noqa: E402
from comic_dialogue import (  # noqa: E402
    ComicDialogueError,
    ComicLine,
    extract_comic_dialogue,
    letter_comic_image,
)
from generation_task import GenerationTaskRunner  # noqa: E402


def test_extracts_exact_recent_comic_script_without_scene_text() -> None:
    prompt = (
        "剧情原文：爱音告知祥子。 "
        "漫画对话： 爱音：放心，我不会杀你，你早就已经死了。 "
        "祥子（空洞失神）：对哦……我已经死了…… "
        "正向提示词：双人面部特写，左右格对比"
    )
    assert extract_comic_dialogue(prompt) == (
        ComicLine("speech", "爱音", "放心，我不会杀你，你早就已经死了。"),
        ComicLine("speech", "祥子", "对哦……我已经死了……"),
    )


def test_thought_narration_and_inline_speaker_are_distinct() -> None:
    prompt = (
        "内心OS：今天弹的是《黄毛丫头》。 "
        "漫画台词（爱音拦住准备离开的灯）：灯灯，等一下。 "
        "高松灯（歪头）：怎么了，小爱？ "
        "正向提示词：校园走廊"
    )
    assert extract_comic_dialogue(prompt) == (
        ComicLine("thought", "", "今天弹的是《黄毛丫头》。"),
        ComicLine("speech", "", "灯灯，等一下。"),
        ComicLine("speech", "高松灯", "怎么了，小爱？"),
    )


def test_scene_colon_and_no_dialogue_do_not_trigger_lettering() -> None:
    assert extract_comic_dialogue(
        "剧情原文：她拉开门。无对话，仅惊吓音效气泡。正向提示词：动态分镜"
    ) == ()
    assert extract_comic_dialogue("爱音在摩天轮里向素世表白") == ()
    assert extract_comic_dialogue("漫画对话：爱音：你好。正向提示词：双人特写") == (
        ComicLine("speech", "爱音", "你好。"),
    )
    assert extract_comic_dialogue("剧情原文：两人对视。漫画对话：祥子：别走。正向提示词：特写") == (
        ComicLine("speech", "祥子", "别走。"),
    )


def test_attributed_quotes_work_without_comic_heading() -> None:
    assert extract_comic_dialogue(
        "爱音说：“我不会杀你。”祥子心想：“她真的看得见我？”"
    ) == (
        ComicLine("speech", "爱音", "我不会杀你。"),
        ComicLine("thought", "祥子", "她真的看得见我？"),
    )
    assert extract_comic_dialogue("走廊里响起《黄毛丫头》，爱音停住脚步。") == ()
    assert extract_comic_dialogue("“我不会杀你。”她举起手枪。") == ()


def test_unparseable_explicit_comic_script_fails_closed() -> None:
    with pytest.raises(ComicDialogueError):
        extract_comic_dialogue("漫画对话： 正向提示词：双人对峙")


def test_short_dialogue_is_lettered_in_frame_without_changing_original(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "frame.png"
    Image.new("RGB", (512, 640), "#778899").save(source)
    before = source.read_bytes()
    lines = (
        ComicLine("speech", "爱音", "放心，我不会杀你。"),
        ComicLine("thought", "祥子", "对哦……我已经死了……"),
    )
    output = letter_comic_image(source, lines, tmp_path / "lettered", task_id="t1", index=0)
    with Image.open(output) as page:
        assert page.width == 512
        assert page.height == 640
        assert page.getpixel((100, page.height - 100)) == (119, 136, 153)
    assert source.read_bytes() == before


def test_long_dialogue_uses_caption_fallback(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "frame.png"
    Image.new("RGB", (512, 640), "#778899").save(source)
    lines = tuple(
        ComicLine("speech", f"角色{index}", "这是必须完整保留的漫画台词。" * 3)
        for index in range(5)
    )
    output = letter_comic_image(source, lines, tmp_path / "lettered", task_id="t3", index=0)
    with Image.open(output) as page:
        assert page.width == 512
        assert page.height > 640
        assert page.getpixel((100, page.height - 100)) == (119, 136, 153)


def test_generation_attaches_dialogue_only_when_switch_is_on() -> None:
    class Recorder:
        def build_generation_start(self, **_kwargs):
            return {"task_id": "test"}

        def mark_prompt_built(self, task, summary):
            task["prompt_summary"] = summary

        def mark_completed(self, _task, **_kwargs):
            pass

        def write(self, _task):
            pass

    class Stub:
        last_summary = {}

    async def build_prompt(_event, _prompt, **_kwargs):
        return "2girls, school hallway"

    async def run_tool(_args):
        return {"ok": True, "outputs": ["frame.png"]}

    def runner(enabled: bool):
        return GenerationTaskRunner(
            task_recorder=Recorder(),
            image_inputs=Stub(),
            reference_context=Stub(),
            is_allowed=lambda _event: True,
            ensure_ready=lambda _event: asyncio.sleep(0, result={"ok": True}),
            wants_reference_image=lambda _text: False,
            augment_reference_image=lambda _event, text: asyncio.sleep(0, result=text),
            augment_quoted_spell=lambda _event, text: text,
            build_prompt=build_prompt,
            prompt_summary=lambda: {},
            run_tool=run_tool,
            get_bool=lambda key, default: enabled if key == "comic_dialogue_enabled" else default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            shorten=lambda text, limit: text[:limit],
        )

    prompt = "漫画对话： 爱音：我不会杀你。 正向提示词：双人对峙"
    off = asyncio.run(runner(False).generate_payload(object(), prompt))
    on = asyncio.run(runner(True).generate_payload(object(), prompt))
    assert "comic_dialogue_lines" not in off
    assert on["comic_dialogue_lines"] == [
        {"kind": "speech", "speaker": "爱音", "text": "我不会杀你。"}
    ]


def test_lettering_failure_never_sends_unlettered_image(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "frame.png"
    source.write_bytes(b"fake image")

    def fail(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr("comfyui_runtime.letter_comic_image", fail)

    class Logger:
        def exception(self, *_args, **_kwargs):
            pass

    class Event:
        unified_msg_origin = "test"

        def __init__(self):
            self.sent = []

        def plain_result(self, text):
            return text

        async def send(self, result):
            self.sent.append(result)

    runtime = ComfyUIRuntime.__new__(ComfyUIRuntime)
    runtime._bool = lambda _key, default: default
    runtime.logger = Logger()
    runtime.root = tmp_path
    runtime.autofilter_settings = None
    event = Event()
    payload = {
        "ok": True,
        "task_id": "t1",
        "outputs": [str(source)],
        "comic_dialogue_lines": [
            {"kind": "speech", "speaker": "爱音", "text": "我不会杀你。"}
        ],
    }
    asyncio.run(runtime.send_payload(event, payload))
    assert payload["error"] == "comic_dialogue_render_failed"
    assert payload["delivery"]["sent"] is False
    assert len(event.sent) == 1
    assert "未发送图片" in event.sent[0]


def test_delivery_uses_lettered_copy_when_switch_supplies_dialogue(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "frame.png"
    Image.new("RGB", (512, 640), "#778899").save(source)

    class Logger:
        def info(self, *_args, **_kwargs):
            pass

    class Event:
        unified_msg_origin = "test"

    runtime = ComfyUIRuntime.__new__(ComfyUIRuntime)
    runtime._bool = lambda key, default: False if key == "send_result_to_chat" else default
    runtime._int = lambda _key, default: default
    runtime.logger = Logger()
    runtime.root = tmp_path
    runtime.autofilter_settings = None
    payload = {
        "ok": True,
        "task_id": "t2",
        "outputs": [str(source)],
        "comic_dialogue_lines": [
            {"kind": "speech", "speaker": "爱音", "text": "我不会杀你。"}
        ],
    }

    asyncio.run(runtime.send_payload(Event(), payload))

    assert payload["comic_original_outputs"] == [str(source)]
    assert payload["outputs"] != [str(source)]
    assert Path(payload["outputs"][0]).exists()
    assert payload["delivery"]["outputs"] == payload["outputs"]
    assert source.exists()
