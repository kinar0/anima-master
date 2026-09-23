from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

PLUGIN_DIR = Path(__file__).resolve().parents[1]
AGENT_TOOLS_DIR = PLUGIN_DIR / "agent_tools"
for import_path in (PLUGIN_DIR, AGENT_TOOLS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from agent_tools import comfyui_operations  # noqa: E402


def test_edit_uses_configured_scale_for_image_scale_node(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (832, 1216)).save(image_path)
    submitted: dict[str, object] = {}

    class _HttpClient:
        def __init__(self, _config) -> None:
            pass

        def upload_image(self, _path: Path) -> str:
            return "uploaded/input.png"

    def fake_run_prompt(_config, _outputs, prompt_body):
        submitted.update(prompt_body)
        return "prompt-id", {}

    monkeypatch.setattr(comfyui_operations, "ComfyUIHttpClient", _HttpClient)
    monkeypatch.setattr(comfyui_operations, "_run_prompt", fake_run_prompt)
    monkeypatch.setattr(
        comfyui_operations,
        "_save_history_images",
        lambda _config, _outputs, _history: ([], 0),
    )

    result = comfyui_operations.edit_payload(
        {
            "max_image_side": 1536,
            "upscale_factor": 1.5,
            "steps": 20,
            "cfg": 4.0,
            "edit_denoise": 0.55,
        },
        tmp_path,
        lambda _value: image_path,
        SimpleNamespace(
            input="input.png",
            steps=None,
            cfg=None,
            denoise=None,
            seed=42,
        ),
        "test prompt",
    )

    scale_inputs = submitted["13"]["inputs"]
    assert scale_inputs["width"] == 1248
    assert scale_inputs["height"] == 1824
    assert result["width"] == 1248
    assert result["height"] == 1824
    assert result["scale"] == 1.5


def test_edit_limits_input_before_applying_scale(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (1600, 2400)).save(image_path)

    assert comfyui_operations._image_size(image_path, 1024, 1.5) == (1024, 1536)
