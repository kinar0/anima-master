import asyncio

from command_router import parse_hard_route
from main import ComfyUIAgentPlugin


def test_term_commands_are_reserved_before_generation_fallback():
    assert parse_hard_route("/anm term_preview") == ("term_preview", "")
    assert parse_hard_route("/anima 名词翻译预览") == ("term_preview", "")
    assert parse_hard_route("/anm term_translate deepseek") == (
        "term_translate", "deepseek",
    )
    assert parse_hard_route("/anm term_translate") == ("term_translate", "")
    assert parse_hard_route("/anm term_verify deepseek") == ("term_verify", "deepseek")
    assert parse_hard_route("/anm term_import preview") == ("term_import", "preview")
    assert parse_hard_route("/anm term_previewing girl") == (
        "generate", "term_previewing girl",
    )


def test_hard_route_dispatches_term_commands_without_entering_generation():
    class Event:
        def __init__(self, message):
            self.message = message
            self.sent = []
            self.stopped = False

        def get_message_str(self):
            return self.message

        def get_message_outline(self):
            return ""

        def stop_event(self):
            self.stopped = True

        def plain_result(self, message):
            return message

        async def send(self, message):
            self.sent.append(message)

    class Handler:
        def _term_preview_text(self, event):
            return "预览已执行"

        async def _term_translate_text(self, event, provider_id):
            return f"翻译入口:{provider_id or '缺参数'}"

        async def _term_verify_text(self, event, provider_id):
            return f"审核入口:{provider_id}"

        async def _term_import_text(self, event, mode):
            return f"导入入口:{mode}"

        def _is_allowed(self, event):
            raise AssertionError("管理命令不应进入生图权限路径")

        async def _handle_action(self, event, action, prompt):
            raise AssertionError("不应进入生图动作分发")

    for message, expected in (
        ("/anm term_preview", "预览已执行"),
        ("/anm term_translate", "翻译入口:缺参数"),
        ("/anm term_translate deepseek", "翻译入口:deepseek"),
        ("/anm term_verify deepseek", "审核入口:deepseek"),
        ("/anm term_import preview", "导入入口:preview"),
    ):
        event = Event(message)
        asyncio.run(ComfyUIAgentPlugin.hard_route_comfyui(Handler(), event))
        assert event.stopped
        assert event.sent == [expected]
