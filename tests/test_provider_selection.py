import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot.api.star import Context
from astrbot.core.provider.provider import Provider
from generation_verifier import GenerationVerifier
from provider_selection import image_provider_id
from reference_context import ReferenceContextBuilder


EVENT = SimpleNamespace(unified_msg_origin="qq:GroupMessage:audit")


def sdk_context(settings=None):
    """Use the installed Context's real session/provider resolution methods."""
    context = object.__new__(Context)
    context.get_config = lambda **_: {"provider_settings": settings or {}}
    provider = Mock(spec=Provider)
    provider.meta.return_value = SimpleNamespace(id="session/vision")
    context.provider_manager = SimpleNamespace(
        get_using_provider_async=AsyncMock(return_value=provider),
    )
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(completion_text="1girl, white dress"),
    )
    return context


def consumers(context, config=None):
    config = config or {}
    logger = logging.getLogger("anima.provider-test")
    reference = ReferenceContextBuilder(
        context=context, run_prompt_tool=None, event_image_input=None,
        logger=logger, get_int=lambda _, default: default,
        shorten=lambda value, limit: value[:limit],
    )
    verifier = GenerationVerifier(
        context=context, task_recorder=None, generate_payload=None, logger=logger,
        get_bool=lambda _, default: default, get_int=lambda _, default: default,
        get_str=lambda key, default: config.get(key, default),
    )
    return reference, verifier


def test_reverse_and_verification_use_installed_sdk_without_removed_default_field():
    context = sdk_context()
    reference, verifier = consumers(context)

    async def run():
        assert await reference.reverse_image_tags(EVENT, "image.png") == "1girl, white dress"
        assert await verifier._verify_provider_id(EVENT, multi_person=False) == "session/vision"
        assert await verifier._verify_provider_id(EVENT, multi_person=True) == "session/vision"

    asyncio.run(run())
    assert context.llm_generate.await_args.kwargs["chat_provider_id"] == "session/vision"
    assert context.llm_generate.await_args.kwargs["image_urls"] == ["image.png"]
    calls = context.provider_manager.get_using_provider_async.await_args_list
    assert len(calls) == 3
    assert all(call.kwargs["umo"] == EVENT.unified_msg_origin for call in calls)


def test_explicit_image_provider_does_not_switch_to_session_or_legacy_default():
    context = sdk_context({
        "default_image_caption_provider_id": " dedicated/vision ",
        "default_provider_id": "old/default",
    })
    reference, verifier = consumers(context)

    async def run():
        assert await reference._image_caption_provider_id(EVENT) == "dedicated/vision"
        assert await verifier._verify_provider_id(EVENT, multi_person=False) == "dedicated/vision"

    asyncio.run(run())
    context.provider_manager.get_using_provider_async.assert_not_awaited()


@pytest.mark.parametrize("multi,config,expected", [
    (False, {"verify_provider_id": "verify/model", "multi_verify_provider_id": "multi/model"}, "verify/model"),
    (True, {"verify_provider_id": "verify/model", "multi_verify_provider_id": "multi/model"}, "multi/model"),
    (True, {"verify_provider_id": "verify/model"}, "verify/model"),
])
def test_plugin_verification_provider_overrides_caption_and_session(multi, config, expected):
    context = sdk_context({"default_image_caption_provider_id": "caption/model"})
    _, verifier = consumers(context, config)
    assert asyncio.run(verifier._verify_provider_id(EVENT, multi_person=multi)) == expected
    context.provider_manager.get_using_provider_async.assert_not_awaited()


def test_session_preference_wins_over_legacy_global_default():
    context = sdk_context({"default_provider_id": "old/default"})
    assert asyncio.run(image_provider_id(context, EVENT)) == "session/vision"


@pytest.mark.parametrize("mode", ["missing", "error", "empty"])
def test_old_host_fallback_and_missing_provider_remain_nonfatal(mode):
    context = SimpleNamespace(get_config=lambda **_: {
        "provider_settings": {"default_provider_id": "old/default"},
    })
    if mode != "missing":
        context.get_current_chat_provider_id = AsyncMock(
            side_effect=RuntimeError("no provider") if mode == "error" else None,
            return_value="",
        )
    assert asyncio.run(image_provider_id(context, EVENT)) == "old/default"
    context.get_config = lambda **_: {"provider_settings": {}}
    assert asyncio.run(image_provider_id(context, EVENT)) == ""


def test_sdk_resolution_survives_config_lookup_failure():
    context = sdk_context()

    def unavailable(**_):
        raise RuntimeError("config unavailable")

    context.get_config = unavailable
    assert asyncio.run(image_provider_id(context, EVENT)) == "session/vision"


def test_invalid_legacy_settings_do_not_block_sdk_resolution():
    context = sdk_context()
    context.get_config = lambda **_: {"provider_settings": None}
    assert asyncio.run(image_provider_id(context, EVENT)) == "session/vision"
