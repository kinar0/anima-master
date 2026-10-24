import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from astrbot.api.star import Context
from astrbot.api.web import PluginRequest, bind_request_context
from starlette.requests import Request

import main
import service_container


def test_full_plugin_initialization_registers_readable_apis_on_installed_host(tmp_path, monkeypatch):
    paths = service_container.resolve_service_paths()
    isolated = replace(
        paths, root=tmp_path, workspace=tmp_path / "workspace",
        inputs=tmp_path / "workspace" / "inputs",
        plugin_data=tmp_path / "data" / "plugin_data" / "astrbot_plugin_anima_master",
    )
    monkeypatch.setattr(service_container, "resolve_service_paths", lambda: isolated)
    monkeypatch.setattr(main, "get_astrbot_plugin_data_path", lambda: str(isolated.plugin_data.parent))
    context = object.__new__(Context)
    context.registered_web_apis = []
    tools = Mock()
    tools.deactivate_llm_tool.return_value = True
    context.provider_manager = SimpleNamespace(llm_tools=tools)
    plugin = main.ComfyUIAgentPlugin(context)
    plugin.config["opus_usage_monitor_enabled"] = False

    async def run():
        await plugin.initialize()
        routes = {route: (handler, methods) for route, handler, methods, _ in context.registered_web_apis}
        assert len(routes) == 16
        for suffix in ("wardrobe", "noun-translation/status", "autofilter", "opus-usage"):
            path = "/astrbot_plugin_anima_master/" + suffix
            handler, methods = routes[path]
            assert methods == ["GET"]
            raw = Request({
                "type": "http", "method": "GET", "path": path,
                "query_string": b"", "headers": [], "scheme": "http",
                "server": ("testserver", 80), "client": ("testclient", 12345),
            })
            with bind_request_context(PluginRequest(raw, username="test-user")):
                response = await handler()
            assert response.status_code == 200
            assert isinstance(json.loads(response.body), dict)
        await plugin.terminate()

    asyncio.run(run())
    assert plugin._runtime.root == tmp_path
    assert plugin._generation_task is not None
    assert plugin._services.generation_verifier is not None
    assert plugin._llm_tool_bridge is not None
    assert {call.args[0] for call in tools.deactivate_llm_tool.call_args_list} == {
        "comfyui_edit", "comfyui_remove_bg",
    }
