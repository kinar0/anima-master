from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, artist name"
)


def t2i_filename_prefix(now: datetime | None = None) -> str:
    """Build a Windows-safe dated ComfyUI output prefix."""
    timestamp = now or datetime.now()
    return f"astrbot/{timestamp:%Y-%m-%d}/{timestamp:%m%d%H%M%S}"


def anima_t2i_workflow(
    config: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
) -> dict[str, Any]:
    """Build the Anima text-to-image workflow graph."""
    return {
        "44": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config.get("unet_name", "anima_baseV10.safetensors"),
                "weight_dtype": "default",
            },
        },
        "45": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": config.get("clip_name", "qwen_3_06b_base.safetensors"),
                "type": "stable_diffusion",
                "device": "default",
            },
        },
        "15": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": config.get("vae_name", "qwen_image_vae.safetensors")
            },
        },
        "28": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "11": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["45", 0]},
        },
        "12": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["45", 0],
            },
        },
        "19": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["44", 0],
                "positive": ["11", 0],
                "negative": ["12", 0],
                "latent_image": ["28", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": config.get("sampler_name", "er_sde"),
                "scheduler": config.get("scheduler", "normal"),
                "denoise": 1,
            },
        },
        "47": {
            "class_type": "easy cleanGpuUsed",
            "inputs": {"anything": ["19", 0]},
            "_meta": {"title": "清理显存占用（解码前）"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["47", 0], "vae": ["15", 0]},
        },
        "46": {
            "class_type": "easy cleanGpuUsed",
            "inputs": {"anything": ["8", 0]},
            "_meta": {"title": "清理显存占用"},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["46", 0],
                "filename_prefix": t2i_filename_prefix(),
            },
        },
    }


def anima_img2img_workflow(
    config: dict[str, Any],
    prompt: str,
    image_name: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float,
) -> dict[str, Any]:
    """Build the Anima image-to-image workflow graph."""
    negative_prompt = str(config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    workflow = anima_t2i_workflow(
        config, prompt, negative_prompt, width, height, steps, cfg, seed
    )
    workflow["10"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    workflow["13"] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["10", 0],
            "upscale_method": "lanczos",
            "width": width,
            "height": height,
            "crop": "disabled",
        },
    }
    workflow["14"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["13", 0], "vae": ["15", 0]},
    }
    workflow["19"]["inputs"]["latent_image"] = ["14", 0]
    workflow["19"]["inputs"]["denoise"] = denoise
    workflow["9"]["inputs"]["filename_prefix"] = "astrbot/edit"
    return workflow


def upscale_workflow(
    config: dict[str, Any], image_name: str, scale: float
) -> dict[str, Any]:
    """Build a simple image upscale workflow graph."""
    return {
        "10": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "20": {
            "class_type": "ImageScaleBy",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "lanczos",
                "scale_by": scale,
            },
        },
        "30": {
            "class_type": "SaveImage",
            "inputs": {"images": ["20", 0], "filename_prefix": "astrbot/upscale"},
        },
    }


def remove_bg_workflow(config: dict[str, Any], image_name: str) -> dict[str, Any]:
    """Build a background-removal workflow graph."""
    return {
        "10": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "20": {
            "class_type": "BiRefNetRMBG",
            "inputs": {
                "image": ["10", 0],
                "model": config.get("remove_bg_model", "BiRefNet_lite"),
                "mask_blur": 1,
                "mask_offset": 0,
                "invert_output": False,
                "refine_foreground": True,
                "background": "Alpha",
                "background_color": "#222222",
            },
        },
        "30": {
            "class_type": "SaveImage",
            "inputs": {"images": ["20", 0], "filename_prefix": "astrbot/remove_bg"},
        },
    }


def workflow(
    config: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    *,
    override_size: bool = False,
    nai_characters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the configured generation workflow graph.

    Args:
        config: Active helper configuration.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        width: Effective output width.
        height: Effective output height.
        steps: Effective sampling steps.
        cfg: Effective CFG scale.
        seed: Generation seed.
        override_size: Whether this request explicitly selected its canvas size.

    Returns:
        ComfyUI API workflow graph.
    """
    if bool(config.get("custom_workflow_enabled", False)):
        return custom_t2i_workflow(
            config,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg,
            seed,
            override_size=override_size,
            nai_characters=nai_characters,
        )

    if nai_characters is not None:
        raise SystemExit("nai_character_mode_requires_nai_workflow")

    workflow_name = str(config.get("workflow") or "anima_t2i")
    if workflow_name != "anima_t2i":
        raise SystemExit(f"unsupported workflow: {workflow_name}")
    return anima_t2i_workflow(
        config, prompt, negative_prompt, width, height, steps, cfg, seed
    )


def custom_t2i_workflow(
    config: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    *,
    override_size: bool = False,
    nai_characters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a text-to-image workflow from a user-provided ComfyUI API JSON.

    Args:
        config: Active helper configuration.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        width: Effective output width.
        height: Effective output height.
        steps: Effective sampling steps.
        cfg: Effective CFG scale.
        seed: Generation seed.
        override_size: Whether to override only latent canvas dimensions.

    Returns:
        Customized ComfyUI API workflow graph.
    """
    path_text = str(config.get("custom_workflow_path") or "").strip()
    if not path_text:
        raise SystemExit("custom_workflow_path_not_configured")

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise SystemExit(f"custom_workflow_not_found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    body = (
        raw.get("prompt")
        if isinstance(raw, dict) and isinstance(raw.get("prompt"), dict)
        else raw
    )
    if not isinstance(body, dict):
        raise SystemExit("custom_workflow_invalid_json")
    if isinstance(body.get("nodes"), list):
        raise SystemExit("custom_workflow_requires_api_export")

    workflow_body = copy.deepcopy(body)
    _apply_custom_workflow_inputs(
        workflow_body,
        config,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        seed,
        override_size=override_size,
        nai_characters=nai_characters,
    )
    return workflow_body


def _apply_custom_workflow_inputs(
    workflow_body: dict[str, Any],
    config: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    *,
    override_size: bool = False,
    nai_characters: list[dict[str, Any]] | None = None,
) -> None:
    text_nodes = set(_text_encode_nodes(workflow_body))
    positive_ids = _conditioning_text_node_ids(
        workflow_body,
        "positive",
        text_nodes,
    )
    negative_ids = _conditioning_text_node_ids(
        workflow_body,
        "negative",
        text_nodes,
    )

    nai_ids = [
        str(node_id)
        for node_id, node in workflow_body.items()
        if isinstance(node, dict) and node.get("class_type") == "NovelAIGenerator"
    ]
    positive_slots = [(node_id, "text") for node_id in positive_ids]
    negative_slots = [(node_id, "text") for node_id in negative_ids]
    for node_id in nai_ids:
        positive_slots.append(_nai_prompt_slot(workflow_body, node_id, "prompt"))
        negative_slots.append(
            _nai_prompt_slot(workflow_body, node_id, "negative_prompt")
        )

    if not positive_slots:
        raise SystemExit("custom_workflow_positive_node_not_found")
    if not negative_slots:
        raise SystemExit("custom_workflow_negative_node_not_found")
    if set(positive_slots) & set(negative_slots):
        raise SystemExit("custom_workflow_prompt_nodes_ambiguous")

    for node_id, input_name in positive_slots:
        _set_node_input(workflow_body, node_id, input_name, prompt)
    for node_id, input_name in negative_slots:
        _set_node_input(workflow_body, node_id, input_name, negative_prompt)

    if nai_characters is not None:
        if len(nai_ids) != 1:
            raise SystemExit("nai_character_mode_requires_nai_workflow")
        _apply_nai_character_prompts(workflow_body, nai_ids[0], nai_characters)

    filename_prefix = t2i_filename_prefix()
    for node in workflow_body.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if class_type == "SaveImage" and "filename_prefix" in inputs:
            inputs["filename_prefix"] = filename_prefix
        override_parameters = bool(
            config.get("custom_workflow_override_parameters", False)
        )
        if class_type == "NovelAIGenerator":
            if override_parameters or override_size:
                if any(
                    side < 64 or side > 4096 or side % 64 for side in (width, height)
                ):
                    raise SystemExit("nai_size_requires_multiple_of_64")
                inputs.update(width=width, height=height)
            if override_parameters:
                if not 1 <= steps <= 50 or not 0 <= cfg <= 30:
                    raise SystemExit("nai_sampling_parameters_out_of_range")
                inputs.update(steps=steps, cfg_scale=cfg)
            # NAI sampler/scheduler names are a separate API vocabulary. Keep
            # the exported values instead of injecting Anima's er_sde/normal.
        if (override_parameters or override_size) and class_type == "EmptyLatentImage":
            if "width" in inputs:
                inputs["width"] = width
            if "height" in inputs:
                inputs["height"] = height
        if override_parameters and class_type in {"KSampler", "KSamplerAdvanced"}:
            for input_name, value in (
                ("steps", steps),
                ("cfg", cfg),
            ):
                if input_name in inputs:
                    inputs[input_name] = value
            for input_name in ("sampler_name", "scheduler"):
                configured = str(config.get(input_name) or "").strip()
                if configured and input_name in inputs:
                    inputs[input_name] = configured
        if "seed" in inputs:
            inputs["seed"] = seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = seed


def _apply_nai_character_prompts(
    workflow_body: dict[str, Any],
    generator_id: str,
    characters: list[dict[str, Any]],
) -> None:
    """Connect per-instance text and coordinates to NAI's character selector."""
    generator = workflow_body[generator_id]
    generator_inputs = generator.get("inputs", {})
    if (
        not isinstance(generator_inputs, dict)
        or not isinstance(characters, list)
        or not 1 <= len(characters) <= 5
    ):
        raise SystemExit("nai_character_plan_character_count")
    used = {str(key) for key in workflow_body}
    next_id = max((int(key) for key in used if key.isdigit()), default=0) + 1

    def allocate() -> str:
        nonlocal next_id
        while str(next_id) in used:
            next_id += 1
        value = str(next_id)
        used.add(value)
        next_id += 1
        return value

    selector_id = allocate()
    selector_inputs: dict[str, Any] = {}
    for index in range(1, 6):
        item = characters[index - 1] if index <= len(characters) else None
        if index > 1:
            selector_inputs[f"character{index}_enable"] = item is not None
        selector_inputs[f"character{index}_uc"] = ""
        selector_inputs[f"character{index}_x"] = float(item["x"]) if item else 0.5
        selector_inputs[f"character{index}_y"] = float(item["y"]) if item else 0.5
        if item:
            converter_id = allocate()
            workflow_body[converter_id] = {
                "class_type": "ComfyUIToNovelAIV4",
                "inputs": {"comfyui_prompt": str(item["prompt"]).strip()},
            }
            selector_inputs[f"character{index}"] = [converter_id, 0]
        else:
            selector_inputs[f"character{index}"] = ""
    workflow_body[selector_id] = {
        "class_type": "CharacterPromptSelect",
        "inputs": selector_inputs,
    }
    generator_inputs["characterPrompts"] = [selector_id, 0]


def _nai_prompt_slot(
    workflow_body: dict[str, Any], node_id: str, input_name: str
) -> tuple[str, str]:
    """Follow supported STRING links without bypassing NAI weight conversion."""
    visited: set[tuple[str, str]] = set()
    while (node_id, input_name) not in visited:
        visited.add((node_id, input_name))
        node = workflow_body.get(node_id)
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            break
        value = inputs.get(input_name)
        if isinstance(value, str):
            return node_id, input_name
        if not isinstance(value, list) or len(value) != 2 or value[1] != 0:
            break
        node_id = str(value[0])
        source = workflow_body.get(node_id)
        if not isinstance(source, dict):
            break
        source_type = source.get("class_type")
        if source_type == "ComfyUIToNovelAIV4":
            input_name = "comfyui_prompt"
        elif source_type == "Textbox":
            source_inputs = source.get("inputs", {})
            if not isinstance(source_inputs, dict):
                break
            input_name = "passthrough" if "passthrough" in source_inputs else "text"
        else:
            break
    raise SystemExit("nai_prompt_link_not_supported")


def _text_encode_nodes(workflow_body: dict[str, Any]) -> list[str]:
    node_ids: list[str] = []
    for node_id, node in workflow_body.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if (
            "TextEncode" in class_type
            and isinstance(inputs, dict)
            and isinstance(inputs.get("text"), str)
        ):
            node_ids.append(str(node_id))
    return node_ids


def _conditioning_text_node_ids(
    workflow_body: dict[str, Any],
    input_name: str,
    text_nodes: set[str],
) -> list[str]:
    """Find text encoders feeding a sampler conditioning input.

    Args:
        workflow_body: ComfyUI API workflow graph.
        input_name: Sampler input name, either positive or negative.
        text_nodes: Known text encoder node identifiers.

    Returns:
        Text encoder identifiers reachable from the conditioning input.
    """
    pending: list[str] = []
    for node in workflow_body.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") not in {
            "KSampler",
            "KSamplerAdvanced",
        }:
            continue
        inputs = node.get("inputs")
        link = inputs.get(input_name) if isinstance(inputs, dict) else None
        if isinstance(link, list) and link:
            pending.append(str(link[0]))

    found: list[str] = []
    visited: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in text_nodes:
            found.append(node_id)
            continue
        node = workflow_body.get(node_id)
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, list) and value:
                source_id = str(value[0])
                if source_id in workflow_body:
                    pending.append(source_id)
    return found


def _set_node_input(
    workflow_body: dict[str, Any], node_id: str, input_name: str, value: Any
) -> None:
    node = workflow_body.get(str(node_id))
    if not isinstance(node, dict):
        raise SystemExit(f"custom_workflow_node_not_found: {node_id}")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit(f"custom_workflow_node_has_no_inputs: {node_id}")
    inputs[input_name] = value
