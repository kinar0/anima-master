"""Keep the original default entry when adding independently managed pages."""
import asyncio
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


def test_compat_entry_roundtrip_uses_real_page_ticket_rewriting():
    root = Path(__file__).resolve().parents[1]
    service = object.__new__(PluginPageService)

    def rendered(path):
        return Links(service.rewrite_plugin_page_html(
            (root/'pages/wardrobe'/path).read_text(encoding='utf-8'),
            'astrbot_plugin_anima_master', 'wardrobe', path,
            theme=None, extra_query_params={'asset_token':'test-page-ticket'},
        )).items

    entry = next(attrs['href'] for _, attrs in rendered('index.html') if attrs.get('id') == 'toolbox-entry')
    back = next(attrs['href'] for _, attrs in rendered('toolbox/index.html') if attrs.get('id') == 'return-wardrobe')
    assert urlsplit(entry).path.endswith('/wardrobe/toolbox/index.html')
    assert urlsplit(back).path.endswith('/wardrobe/index.html')
    for url in (entry, back):
        assert parse_qs(urlsplit(url).query)['asset_token'] == ['test-page-ticket']
    assets = [attrs.get('src', attrs.get('href', '')) for tag, attrs in rendered('toolbox/index.html') if tag in {'script','link'}]
    for name in ('app.js', 'style.css'):
        url = next(x for x in assets if urlsplit(x).path.endswith('/'+name))
        assert urlsplit(url).path.endswith('/wardrobe/toolbox/'+name)
        assert parse_qs(urlsplit(url).query)['asset_token'] == ['test-page-ticket']


def test_compatibility_bundle_stays_in_sync_with_standalone_editor():
    from scripts.toolbox_compat_entry import render_entry
    root = Path(__file__).resolve().parents[1]
    source = root/'pages/wardrobe_toolbox'
    target = root/'pages/wardrobe/toolbox'
    assert (target/'index.html').read_text(encoding='utf-8') == render_entry((source/'index.html').read_text(encoding='utf-8'))
    for name in ('app.js', 'style.css'):
        assert (target/name).read_bytes() == (source/name).read_bytes()
