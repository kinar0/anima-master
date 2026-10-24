"""Deliver plugin request controls through AstrBot's OpenAI-compatible adapter."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import Any


async def generate_with_request_controls(context: Any, **kwargs: Any) -> Any:
    """Preserve per-call limits without mutating a shared provider or its config.

    AstrBot 4.29.1 accepts generation kwargs but drops them while preparing an
    OpenAI payload. Its provider-level custom_extra_body is still serialized.
    Use that supported channel on a request-local provider copy. Other adapters
    and contexts retain their existing SDK path. Do not retry a failed request
    through the original context: that would silently lose the controls again.
    """
    controls = {
        key: kwargs[key]
        for key in ("max_tokens", "thinking", "reasoning_effort")
        if key in kwargs and kwargs[key] is not None
    }
    if not controls:
        return await context.llm_generate(**kwargs)

    manager = getattr(context, "provider_manager", None)
    lookup = getattr(manager, "get_provider_by_id", None)
    if not callable(lookup):
        return await context.llm_generate(**kwargs)

    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial

    provider = await lookup(kwargs["chat_provider_id"])
    if not isinstance(provider, ProviderOpenAIOfficial):
        return await context.llm_generate(**kwargs)

    request_provider = copy(provider)
    request_config = dict(provider.provider_config)
    configured_body = request_config.get("custom_extra_body", {})
    body = deepcopy(configured_body) if isinstance(configured_body, dict) else {}
    # An inherited effort can enable thinking on DeepSeek. An explicit disabled
    # request must not accidentally retain a provider's enabled/high default.
    thinking = controls.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        body.pop("reasoning_effort", None)
    body.update(deepcopy(controls))
    request_config["custom_extra_body"] = body
    request_provider.provider_config = request_config

    request = dict(kwargs)
    request.pop("chat_provider_id")
    tools = request.pop("tools", None)
    # Same tools -> func_tool translation as Context.llm_generate. All content,
    # images, contexts and remaining kwargs continue through the real adapter.
    return await request_provider.text_chat(func_tool=tools, **request)
