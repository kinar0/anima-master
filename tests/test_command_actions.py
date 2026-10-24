from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from command_actions import CommandActionHandler  # noqa: E402
from command_catalog import COMMAND_ENTRIES  # noqa: E402
from multi_person_prompt import MULTI_PERSON_NEGATIVE_TAGS  # noqa: E402
from prompt_presets import DEFAULT_NEGATIVE_PROMPT  # noqa: E402
from tag_cleaner import join_prompt_parts  # noqa: E402


class _Recorder:
    def debug_status_text(self, config):
        return "debug"


def _noop_async(*args, **kwargs):
    async def _inner():
        return "ok"

    return _inner()


def _handler(config=None, generate=None) -> CommandActionHandler:
    return CommandActionHandler(
        config=config or {},
        task_recorder=_Recorder(),
        reference_context=None,
        is_allowed=lambda event: True,
        run_tool=lambda args: _noop_async(),
        ensure_ready=lambda event: _noop_async(),
        send_payload=lambda event, payload: _noop_async(),
        generate=generate or (lambda event, prompt, **kwargs: _noop_async()),
        event_image_input=lambda event: _noop_async(),
        build_prompt=lambda event, prompt, mode="txt2img": _noop_async(),
        format_spell_payload=lambda payload: "spell",
        get_bool=lambda key, default: default,
        shorten=lambda text, limit: text[:limit],
        config_store=None,
    )


def test_action_dispatch_covers_catalog_actions():
    handler = _handler()
    catalog_actions = {
        entry.action for entry in COMMAND_ENTRIES if entry.action != "raw_generate"
    }

    assert catalog_actions <= handler.action_names()


def test_unknown_action_returns_error():
    handler = _handler()

    result = asyncio.run(handler.handle_action(object(), "nope", ""))
    assert result == "未知 Anima 指令。"


def test_create_artist_preset_preserves_nai_numeric_emphasis() -> None:
    config = {"quality_prefix": "test"}
    handler = _handler(config)
    response = handler.create_artist_preset(
        "naipaint=1.2::artist:banpai akira ::, "
        "0.6::artist:chen_bin ::, -0.5:: lips::"
    )

    assert "已保存并启用" in response
    assert config["artist_presets"] == [
        "naipaint=1.2::artist:banpai akira ::, "
        "0.6::artist:chen_bin ::, -0.5:: lips::,"
    ]


def test_create_and_append_artist_group_preserves_nai_scope():
    config = {"quality_prefix": "test"}
    handler = _handler(config)
    artists = "<artist> 1.2::artist:alpha, artist:beta:: </artist>"
    style = "<style> -2::yellow themed, orange themed:: </style>"
    handler.create_artist_preset(f"nai={artists}, {style}")
    handler.append_artist_tags("nai=orange themed, 2::flat color, orange themed::")
    assert config["artist_presets"] == [
        f"nai={artists}, {style}, orange themed, 2::flat color, orange themed::,"
    ]


def test_edit_blocks_combination_before_comfyui_readiness() -> None:
    handler = _handler(
        config={
            "blocked_prompt_rules": [
                {"group_a": ["词甲"], "group_b": ["词乙"]}
            ],
        }
    )
    handler._bool = lambda key, default: True if key == "img2img_enabled" else default

    async def unexpected_ready(_event):
        raise AssertionError("blocked edit must not check ComfyUI")

    handler._ensure_ready = unexpected_ready
    result = asyncio.run(handler.edit(object(), "词乙和词甲"))

    assert result == (
        "请求包含禁止的词语组合，已拒绝生成。\n"
        "本次命中词语：「词甲」、「词乙」。\n"
        "请修改后再试。"
    )


def test_generate_action_passes_one_time_size_override():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs))

    handler = _handler(
        config={"allowed_sizes": ["1024x1024", "1216x832"]},
        generate=generate,
    )

    result = asyncio.run(
        handler.handle_action(object(), "generate", "横图：少女站在河岸")
    )

    assert result is None
    assert calls == [("少女站在河岸", {"width": 1216, "height": 832})]


def test_generate_action_accepts_space_after_explicit_size():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs))

    handler = _handler(
        config={"allowed_sizes": ["1024x1024", "1216x832"]},
        generate=generate,
    )

    result = asyncio.run(
        handler.handle_action(object(), "generate", "1216x832 少女站在河岸")
    )

    assert result is None
    assert calls == [("少女站在河岸", {"width": 1216, "height": 832})]


def test_glued_chinese_size_is_checked_before_quota_and_generation():
    from main import ComfyUIAgentPlugin

    class Event:
        def __init__(self, sender_id):
            self.sender_id = sender_id
            self.messages = []

        def get_sender_id(self):
            return self.sender_id

        def plain_result(self, text):
            return text

        async def send(self, result):
            self.messages.append(result)

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {"generation_whitelist_sender_ids": ["white"]}
    reserved = []

    def reserve(event):
        reserved.append(event.sender_id)
        # Stop at the quota boundary: no LLM or ComfyUI request is needed.
        return "已通过尺寸检查"

    async def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("must stop before prompt building or submission")

    plugin._reserve_generation_call = reserve
    plugin._generate_payload = unexpected_generation
    handler = _handler(
        config={"artist_presets": ["test=artist:test"]},
        generate=plugin._generate,
    )
    for action in ("generate", "multi_person"):
        for prefix in ("", "-r ", "-s1 ", "-r -s1 "):
            for size in ("1024*1536", "1536×1024", "1024x1028"):
                event = Event("ordinary")
                assert asyncio.run(handler.handle_action(
                    event, action, f"{prefix}{size}两名少女合影"
                )) is None
                assert len(event.messages) == 1 and "1MP" in event.messages[0]
    assert reserved == []

    # The same syntax must preserve valid sizes and the whitelist exemption.
    for sender, size in (
        ("ordinary", "1024x1024"),
        ("ordinary", "1216x832"),
        ("ordinary", "512x1756"),
        ("white", "1024*1536"),
    ):
        event = Event(sender)
        asyncio.run(handler.handle_action(event, "generate", f"{size}少女合影"))
        assert event.messages == ["已通过尺寸检查"]
    assert reserved == ["ordinary", "ordinary", "ordinary", "white"]


def test_generate_action_accepts_size_after_generation_switches():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs))

    handler = _handler(
        config={
            "allowed_sizes": ["1216x832"],
            "artist_presets": ["test=artist:test"],
        },
        generate=generate,
    )

    for switches in ("-r", "-s1", "-r -s1"):
        result = asyncio.run(
            handler.handle_action(
                object(), "generate", f"{switches} 1216x832 少女站在河岸"
            )
        )
        assert result is None
        assert calls[-1] == (
            f"{switches} 少女站在河岸",
            {"width": 1216, "height": 832},
        )


def test_generate_action_accepts_valid_size_not_in_configured_candidates():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs))

    handler = _handler(
        config={"allowed_sizes": ["1024x1024"]},
        generate=generate,
    )

    result = asyncio.run(handler.handle_action(object(), "generate", "1000x1400：少女"))

    assert result is None
    assert calls == [("少女", {"width": 1000, "height": 1400})]


def test_img2img_refusal_does_not_call_comfyui() -> None:
    tool_calls: list[list[str]] = []

    async def run_tool(args: list[str]):
        tool_calls.append(args)
        return {"ok": True}

    handler = CommandActionHandler(
        config={"img2img_enabled": True, "prompt_optimize_img2img_enabled": True},
        task_recorder=_Recorder(),
        reference_context=None,
        is_allowed=lambda _event: True,
        run_tool=run_tool,
        ensure_ready=lambda _event: asyncio.sleep(0, result={"ok": True}),
        send_payload=lambda _event, _payload: asyncio.sleep(0, result="ok"),
        generate=lambda *_args, **_kwargs: asyncio.sleep(0),
        event_image_input=lambda _event: asyncio.sleep(0, result="input.png"),
        build_prompt=lambda *_args, **_kwargs: asyncio.sleep(0, result=""),
        format_spell_payload=lambda _payload: "spell",
        get_bool=lambda key, default: bool(
            {
                "img2img_enabled": True,
                "prompt_optimize_img2img_enabled": True,
            }.get(key, default)
        ),
        shorten=lambda text, limit: text[:limit],
        config_store=None,
    )

    result = asyncio.run(handler.edit(object(), "修改衣服"))

    assert result == "模型拒绝了生成"
    assert tool_calls == []


def test_multi_person_action_uses_square_default_and_request_flag():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs))

    handler = _handler(
        config={"allowed_sizes": ["1024x1024", "1216x832", "832x1216"]},
        generate=generate,
    )

    result = asyncio.run(
        handler.handle_action(
            object(),
            "multi_person",
            "左边若叶睦，右边千早爱音，两人牵手",
        )
    )

    assert result is None
    assert calls == [
        (
            "左边若叶睦，右边千早爱音，两人牵手",
            {
                "width": 1024,
                "height": 1024,
                "negative_prompt": join_prompt_parts(
                    [DEFAULT_NEGATIVE_PROMPT, ", ".join(MULTI_PERSON_NEGATIVE_TAGS)]
                ),
                "multi_person": True,
                "size_explicit": False,
            },
        )
    ]


def test_multi_person_action_preserves_explicit_vertical_size():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs))

    handler = _handler(
        config={"allowed_sizes": ["1216x832", "832x1216"]},
        generate=generate,
    )

    result = asyncio.run(
        handler.handle_action(
            object(),
            "multi_person",
            "竖图：前景一个女孩，背景一个男孩",
        )
    )

    assert result is None
    assert calls == [
        (
            "前景一个女孩，背景一个男孩",
            {
                "width": 832,
                "height": 1216,
                "negative_prompt": join_prompt_parts(
                    [DEFAULT_NEGATIVE_PROMPT, ", ".join(MULTI_PERSON_NEGATIVE_TAGS)]
                ),
                "multi_person": True,
                "size_explicit": True,
            },
        )
    ]


def test_multi_person_action_chooses_layout_aware_defaults():
    calls = []

    async def generate(event, prompt, **kwargs):
        calls.append((prompt, kwargs["width"], kwargs["height"]))

    handler = _handler(
        config={
            "allowed_sizes": [
                "1024x1024",
                "1152x896",
                "1216x832",
                "1024x1536",
            ]
        },
        generate=generate,
    )

    asyncio.run(handler.handle_action(object(), "multi_person", "两个女孩并肩站立"))
    asyncio.run(handler.handle_action(object(), "multi_person", "狐娘骑在少女肩膀上"))
    asyncio.run(handler.handle_action(object(), "multi_person", "三人组成乐队合影"))

    assert calls == [
        ("两个女孩并肩站立", 1152, 896),
        ("狐娘骑在少女肩膀上", 1024, 1536),
        ("三人组成乐队合影", 1216, 832),
    ]
