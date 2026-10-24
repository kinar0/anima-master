"""Resolve image-provider fallbacks through AstrBot's session-aware SDK."""

from typing import Any


async def image_provider_id(context: Any, event: Any) -> str:
    """Prefer the image-caption provider, then the session's chat provider.

    Keep the old default-provider field as a fallback for older AstrBot hosts.
    Model capabilities are still checked by the provider when images are sent.
    """
    settings: dict[str, Any] = {}
    try:
        configured_settings = context.get_config(umo=event.unified_msg_origin).get(
            "provider_settings", {}
        )
        if isinstance(configured_settings, dict):
            settings = configured_settings
    except Exception:  # Config discovery must not prevent SDK session lookup.
        pass
    configured = str(settings.get("default_image_caption_provider_id") or "").strip()
    if configured:
        return configured
    try:
        current = str(
            await context.get_current_chat_provider_id(event.unified_msg_origin) or ""
        ).strip()
        if current:
            return current
    except Exception:  # Older hosts or sessions without a selected provider.
        pass
    return str(settings.get("default_provider_id") or "").strip()
