from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from artist_session_settings import ArtistSessionSettings  # noqa: E402
from command_actions import CommandActionHandler  # noqa: E402
from prompt_builder import build_final_prompt  # noqa: E402


def _handler(config: dict, settings: ArtistSessionSettings) -> CommandActionHandler:
    handler = CommandActionHandler(
        config=config,
        task_recorder=None,
        reference_context=None,
        is_allowed=lambda event: True,
        run_tool=lambda args: None,
        ensure_ready=lambda event: None,
        send_payload=lambda event, payload: None,
        generate=lambda event, prompt, **kwargs: None,
        event_image_input=lambda event: None,
        build_prompt=lambda event, prompt, **kwargs: None,
        format_spell_payload=lambda payload: "",
        get_bool=lambda key, default: default,
        shorten=lambda text, limit: text[:limit],
    )
    handler.artist_session_settings = settings
    return handler


def test_chat_artist_selection_is_persistent_and_isolated(tmp_path: Path) -> None:
    config = {
        "artist_presets": ["alpha=artist alpha", "beta=artist beta"],
        "active_artist_preset": "alpha",
        "default_artist_tags": "default artist",
    }
    settings = ArtistSessionSettings(tmp_path / "artist_sessions.json")
    handler = _handler(config, settings)
    group = SimpleNamespace(unified_msg_origin="qq:GroupMessage:123")
    private = SimpleNamespace(unified_msg_origin="qq:FriendMessage:123")

    asyncio.run(handler.handle_action(group, "use_artist_preset", "beta"))
    asyncio.run(handler.handle_action(private, "use_artist_preset", "默认"))
    assert config["active_artist_preset"] == "alpha"
    assert "beta（当前）" in handler.list_artist_presets(group)
    assert "默认画师 tags：已配置（当前）" in handler.list_artist_presets(private)
    assert ArtistSessionSettings(settings.path).get(group.unified_msg_origin) == "beta"

    group_config = dict(config, active_artist_preset=settings.get(group.unified_msg_origin))
    private_config = dict(config, active_artist_preset=settings.get(private.unified_msg_origin))
    def built(config_for_chat: dict, prompt: str = "少女") -> str:
        return build_final_prompt(
            user_prompt=prompt, llm_content="1girl", config=config_for_chat
        ).final_prompt

    assert "artist beta" in built(group_config)
    assert "default artist" in built(private_config)
    assert "artist alpha" in built(config)

    # A one-request selector overrides the prompt, without changing the saved chat choice.
    assert "artist alpha" in built(group_config, "-s2 少女")
    assert settings.get(group.unified_msg_origin) == "beta"

    asyncio.run(handler.handle_action(group, "delete_artist_preset", "beta"))
    assert settings.get(group.unified_msg_origin) == ""
    assert settings.get(private.unified_msg_origin) == ""
