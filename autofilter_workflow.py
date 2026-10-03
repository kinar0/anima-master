"""Run the bundled visual ComfyUI censor graph against one local image."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


def _expand_image_blur(node: dict[str, Any], definition: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Flatten the blur subgraph in autofilter2 to its executable shader node."""
    internal = {item["type"]: item for item in definition.get("nodes", [])}
    if not {"GLSLShader", "CustomCombo", "PrimitiveFloat"} <= internal.keys():
        raise ValueError("autofilter2 的 Image Blur 子图结构无法识别")
    shader_widgets = internal["GLSLShader"]["widgets_values_named"]
    combo = internal["CustomCombo"]["widgets_values_named"]
    options = [combo.get(f"option{index}") for index in range(1, 5)]
    choice = node.get("widgets_values_named", {}).get("choice")
    if choice not in options:
        raise ValueError(f"autofilter2 的模糊方式无法识别: {choice}")
    value = node.get("widgets_values_named", {}).get("value")
    return {
        "class_type": "GLSLShader",
        "inputs": {
            "images.image0": inputs["images.image0"],
            "floats.u_float0": float(value),
            "ints.u_int0": options.index(choice),
            "fragment_shader": shader_widgets["fragment_shader"],
            "size_mode": shader_widgets["size_mode"],
        },
    }


def api_workflow(graph: dict[str, Any], uploaded_name: str) -> dict[str, Any]:
    """Convert the bundled UI graph using its existing widgets and links."""
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    subgraphs = {
        item["id"]: item for item in graph.get("definitions", {}).get("subgraphs", [])
    }
    result: dict[str, Any] = {}
    load_count = save_count = 0
    for node_id, node in nodes.items():
        kind = node["type"]
        inputs: dict[str, Any] = {}
        values = node.get("widgets_values_named", {})
        for item in node.get("inputs", []):
            name = item["name"]
            if item.get("link") is not None:
                link = links[item["link"]]
                inputs[name] = [str(link[1]), link[2]]
            elif name in values and item.get("type") != "IMAGEUPLOAD":
                inputs[name] = values[name]
        if kind == "LoadImage":
            load_count += 1
            inputs["image"] = uploaded_name
        if kind == "SaveImage":
            save_count += 1
            # The bundled visual graph uses %date:...% in its output prefix.
            # Stock ComfyUI on Windows treats that as a literal directory name
            # and ':' makes the save fail with WinError 267. Change only the
            # submitted copy; never edit autofilter2.json itself.
            inputs["filename_prefix"] = f"AstrBot/autofilter_{uuid.uuid4().hex}"
        result[node_id] = (
            _expand_image_blur(node, subgraphs[kind], inputs)
            if kind in subgraphs
            else {"class_type": kind, "inputs": inputs}
        )
    if load_count != 1 or save_count != 1:
        raise ValueError("autofilter 工作流必须恰好包含一个 LoadImage 和 SaveImage")
    return result


def filter_image(image: Path, *, base_url: str, workflow_path: Path, timeout: int) -> Path:
    """Upload, execute and download the censored result. Raises on any failure."""
    graph = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
    base = base_url.rstrip("/")
    with image.open("rb") as handle:
        response = requests.post(
            base + "/upload/image",
            files={"image": (image.name, handle, "application/octet-stream")},
            data={"subfolder": "AstrBot", "type": "input", "overwrite": "false"},
            timeout=120,
        )
    response.raise_for_status()
    uploaded = response.json()
    name = str(uploaded["name"])
    subfolder = str(uploaded.get("subfolder") or "").strip("/")
    uploaded_name = f"{subfolder}/{name}" if subfolder else name
    prompt = api_workflow(graph, uploaded_name)
    response = requests.post(
        base + "/prompt", json={"prompt": prompt, "client_id": str(uuid.uuid4())}, timeout=20
    )
    response.raise_for_status()
    prompt_id = str(response.json()["prompt_id"])
    deadline = time.monotonic() + max(1, timeout)
    while time.monotonic() < deadline:
        response = requests.get(base + f"/history/{prompt_id}", timeout=20)
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") not in (None, "success"):
                details = []
                for message in status.get("messages", []):
                    if not isinstance(message, (list, tuple)) or len(message) < 2:
                        continue
                    payload = message[1]
                    if isinstance(payload, dict) and message[0] == "execution_error":
                        details.append(
                            f"{payload.get('node_type', 'unknown')} "
                            f"({payload.get('node_id', '?')}): "
                            f"{str(payload.get('exception_message') or 'unknown error')[:300]}"
                        )
                detail = "; ".join(details) or str(status.get("status_str"))
                raise RuntimeError(f"autofilter 工作流执行失败 prompt_id={prompt_id}: {detail}")
            images = history.get("outputs", {}).get(
                next(node_id for node_id, node in prompt.items() if node["class_type"] == "SaveImage"), {}
            ).get("images", [])
            if not images:
                raise RuntimeError("autofilter 工作流没有输出图片")
            descriptor = images[0]
            query = urlencode({
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            })
            response = requests.get(base + f"/view?{query}", timeout=120)
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("autofilter 输出图片为空")
            output = image.with_name(f"{image.stem}_autofilter_{uuid.uuid4().hex[:8]}{Path(descriptor['filename']).suffix or '.png'}")
            output.write_bytes(response.content)
            return output
        time.sleep(2)
    raise TimeoutError("autofilter 工作流等待超时")
