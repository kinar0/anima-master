"""Keep the original default entry when adding independently managed pages."""
import asyncio
from pathlib import Path

from astrbot.dashboard.services.plugin_page_service import PluginPageService


def test_actual_page_discovery_keeps_original_wardrobe_first():
    root = Path(__file__).resolve().parents[1]
    service = object.__new__(PluginPageService)

    async def pages_root(_plugin):
        return root / 'pages'

    service.resolve_plugin_pages_root = pages_root
    pages = asyncio.run(service.discover_plugin_pages(object()))
    names = [page.name for page in pages]
    # Both sidebar and plugin-list quick entry use pages[0]. Keep the existing
    # wardrobe dashboard accessible there while the detail page lists both.
    assert names[0] == 'wardrobe'
    assert 'wardrobe_toolbox' in names
