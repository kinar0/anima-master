from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
AGENT_TOOLS_DIR = PLUGIN_DIR / "agent_tools"
if str(AGENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS_DIR))

from comfyui_sizes import (  # noqa: E402
    allowed_sizes,
    generation_size,
    is_valid_generation_size,
)


def test_generation_size_contract_accepts_boundaries_and_arbitrary_valid_pairs():
    assert is_valid_generation_size(512, 512)
    assert is_valid_generation_size(832, 1756)
    assert is_valid_generation_size(1756, 832)
    assert is_valid_generation_size(1000, 1400)


def test_generation_size_contract_rejects_invalid_sides():
    assert not is_valid_generation_size(508, 1024)
    assert not is_valid_generation_size(1024, 1760)
    assert not is_valid_generation_size(834, 1024)


def test_generation_size_does_not_snap_valid_pair_to_configured_candidates():
    assert generation_size(
        {"allowed_sizes": ["1024x1024"]},
        ["1024x1024"],
        1000,
        1400,
    ) == (1000, 1400)


def test_allowed_size_alias_candidates_follow_the_same_contract():
    assert allowed_sizes(
        {"allowed_sizes": ["508x1344", "512x1024", "832x1216", "1756x832"]},
        ["1024x1024"],
    ) == [(512, 1024), (832, 1216), (1756, 832)]
