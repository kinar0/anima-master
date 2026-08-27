from __future__ import annotations

import re

DEFAULT_PORTRAIT = "default_portrait"
EXPLICIT_SCENE = "explicit_scene"
DEFAULT_PORTRAIT_MARKER = "background_mode_default_portrait"
EXPLICIT_SCENE_MARKER = "background_mode_explicit_scene"

_FRAMING_TAG_GROUPS = {
    "head": frozenset(
        {"close up", "eye focus", "face focus", "hair focus", "headshot", "mouth focus"}
    ),
    "upper": frozenset({"breast focus", "bust", "cropped torso", "upper body"}),
    "cowboy": frozenset({"cowboy shot"}),
    "navel": frozenset({"navel focus"}),
    "lower": frozenset({"lower body"}),
    "thigh": frozenset({"ass focus", "thigh focus"}),
    "leg": frozenset({"leg focus"}),
    "foot": frozenset({"feet focus", "foot focus"}),
    "hand": frozenset({"hand focus"}),
    "head_out": frozenset({"head out of frame"}),
}
_FRAMING_TAGS = frozenset(
    {
        "face",
        "portrait",
        *(
            tag.replace("close up", "close-up")
            for tags in _FRAMING_TAG_GROUPS.values()
            for tag in tags
        ),
    }
)

_OUTFIT_SLOT_VISIBILITY_BY_FRAME = {
    "head": frozenset(
        {
            "upper_body.primary",
            "upper_body.corset",
            "lower_body.skirt",
            "lower_body.pants",
            "lower_body.underwear",
            "one_piece.dress",
            "outerwear",
            "handwear",
            "legwear",
            "footwear",
        }
    ),
    "upper": frozenset(
        {
            "lower_body.skirt",
            "lower_body.pants",
            "lower_body.underwear",
            "legwear",
            "footwear",
        }
    ),
    "cowboy": frozenset({"legwear", "footwear"}),
    "navel": frozenset(
        {
            "headwear",
            "hair_accessory",
            "face_accessory.mask",
            "handwear",
            "legwear",
            "footwear",
        }
    ),
    "lower": frozenset(
        {
            "upper_body.primary",
            "upper_body.corset",
            "headwear",
            "hair_accessory",
            "face_accessory.mask",
            "handwear",
        }
    ),
    "thigh": frozenset(
        {
            "upper_body.primary",
            "upper_body.corset",
            "headwear",
            "hair_accessory",
            "face_accessory.mask",
            "handwear",
            "footwear",
        }
    ),
    "leg": frozenset(
        {
            "upper_body.primary",
            "upper_body.corset",
            "headwear",
            "hair_accessory",
            "face_accessory.mask",
            "handwear",
        }
    ),
    "foot": frozenset(
        {
            "upper_body.primary",
            "upper_body.corset",
            "outerwear",
            "headwear",
            "hair_accessory",
            "face_accessory.mask",
            "handwear",
        }
    ),
    "hand": frozenset(
        {
            "lower_body.skirt",
            "lower_body.pants",
            "lower_body.underwear",
            "headwear",
            "hair_accessory",
            "face_accessory.mask",
            "legwear",
            "footwear",
        }
    ),
    "head_out": frozenset(
        {"headwear", "hair_accessory", "face_accessory.mask"}
    ),
}


def framing_hidden_outfit_slots(text: str) -> frozenset[str]:
    """Return garment slots that are deterministically outside the requested view.

    This extends the same framing vocabulary used to prevent accidental
    ``full body`` injection. Chinese body words require camera/visibility syntax,
    so a clothing mutation such as ``下半身没穿`` is not mistaken for a crop.
    Ambiguous framing such as ``portrait`` deliberately does not remove clothes.
    """
    raw = str(text or "")
    normalized = re.sub(r"\s+", " ", raw.lower().replace("_", " ").replace("-", " "))
    normalized_segments = {
        segment.strip()
        for segment in re.split(r"[,;\n]", normalized)
        if segment.strip()
    }

    def english_group(group: str) -> bool:
        for tag in _FRAMING_TAG_GROUPS[group]:
            if group in {"upper", "lower"}:
                # Body-region words also occur in nudity/mutation prose. Treat
                # them as framing only when they are standalone tags or carry a
                # camera/visibility cue.
                if tag in normalized_segments or normalized.strip() == tag:
                    return True
                if re.search(
                    rf"(?<![a-z])(?:only\s+)?(?:showing|visible|framing|view|shot)"
                    rf"(?:\s+of)?(?:\s+(?:only|her|his|their|the)){{0,2}}"
                    rf"\s+{re.escape(tag)}(?![a-z])|"
                    rf"(?<![a-z]){re.escape(tag)}\s+(?:view|shot|framing|portrait)(?![a-z])",
                    normalized,
                    re.I,
                ):
                    return True
                continue
            if re.search(
                rf"(?<![a-z]){re.escape(tag)}(?![a-z])", normalized, re.I
            ):
                return True
        return False

    chinese_view = re.compile(
        r"(?:只(?:露出|拍|画|显示|看见)|重点(?:拍|画|显示))\s*"
        r"(?P<part>上半身|下半身|头部|脸部|面部|眼部|嘴部|头发|手部|双手|"
        r"脚部|足部|双脚|双足|腿部|大腿|臀部|腹部|肚脐)|"
        r"(?P<part2>上半身|下半身|头部|脸部|面部|眼部|嘴部|头发|手部|双手|"
        r"脚部|足部|双脚|双足|腿部|大腿|臀部|腹部|肚脐)"
        r"(?:肖像|人像|构图|镜头|特写|入镜|可见|出镜|聚焦|视图|画面)",
        re.I,
    )
    chinese_parts = {
        match.group("part") or match.group("part2")
        for match in chinese_view.finditer(raw)
    }
    # Boundary phrasing describes the lowest visible body region rather than
    # naming a conventional camera tag. Keep this separate from clothing/body
    # mutations by requiring an explicit restrictive camera verb.
    chinese_boundary = re.compile(
        r"(?:只|仅|画面(?:中)?只)\s*"
        r"(?:拍|拍摄|画|显示|展示|保留|截取|裁切)\s*"
        r"(?:到|至)?\s*(?P<boundary>肩|胸|胸部|腰|腰部|腹部|臀|臀部|"
        r"大腿|膝|膝盖|小腿|脚踝|脚部|足部)\s*"
        r"(?:以上|为止|位置)?",
        re.I,
    )
    boundary_match = chinese_boundary.search(raw)
    boundary = boundary_match.group("boundary") if boundary_match else ""
    boundary_frame = ""
    if boundary == "肩":
        boundary_frame = "head"
    elif boundary in {"胸", "胸部", "腰", "腰部", "腹部"}:
        boundary_frame = "upper"
    elif boundary in {"臀", "臀部", "大腿", "膝", "膝盖"}:
        boundary_frame = "cowboy"

    if boundary_frame:
        frame = boundary_frame
    elif (
        english_group("head")
        or "头像" in raw
        or bool(chinese_parts & {"头部", "脸部", "面部", "眼部", "嘴部", "头发"})
    ):
        frame = "head"
    elif english_group("hand") or bool(chinese_parts & {"手部", "双手"}):
        frame = "hand"
    elif english_group("foot") or bool(
        chinese_parts & {"脚部", "足部", "双脚", "双足"}
    ):
        frame = "foot"
    elif english_group("thigh") or bool(
        chinese_parts & {"大腿", "臀部"}
    ):
        frame = "thigh"
    elif english_group("leg") or "腿部" in chinese_parts:
        frame = "leg"
    elif english_group("navel") or bool(chinese_parts & {"腹部", "肚脐"}):
        frame = "navel"
    elif english_group("lower") or "下半身" in chinese_parts:
        frame = "lower"
    elif (
        english_group("upper")
        or "上半身" in chinese_parts
        or any(marker in raw for marker in ("半身像", "胸像"))
    ):
        frame = "upper"
    elif english_group("cowboy") or re.search(
        r"(?:牛仔(?:镜头|构图)|膝上(?:构图|镜头)|(?:裁|截)到膝)", raw, re.I
    ):
        frame = "cowboy"
    elif english_group("head_out") or re.search(
        r"(?:头部|脑袋)(?:出框|不入镜|不在画面)", raw, re.I
    ):
        frame = "head_out"
    else:
        return frozenset()
    return _OUTFIT_SLOT_VISIBILITY_BY_FRAME[frame]

_CHINESE_SCENE_MARKERS = (
    "室内",
    "室外",
    "户外",
    "房间",
    "卧室",
    "客厅",
    "教室",
    "厨房",
    "浴室",
    "泳池",
    "游泳池",
    "池畔",
    "和室",
    "庭院",
    "街道",
    "城市",
    "海边",
    "沙滩",
    "森林",
    "公园",
    "花园",
    "车站",
    "舞台",
    "餐厅",
    "咖啡馆",
    "办公室",
    "图书馆",
    "神社",
    "寺庙",
    "城堡",
    "天空",
    "雪景",
    "雨景",
    "夜景",
)

_ENGLISH_SCENE_RE = re.compile(
    r"\b(?:indoors?|outdoors?|bedroom|living room|classroom|kitchen|bathroom|"
    r"poolside|swimming pool|pool|street|city|beach|forest|park|garden|station|stage|restaurant|cafe|"
    r"office|library|shrine|temple|castle|nightscape)\b",
    flags=re.I,
)


def _positive_background_text(text: str) -> str:
    """Remove common negative background requests before intent detection."""
    positive = re.sub(
        r"(?:不要|无需|不需要|没有|无)"
        r"(?:任何|特定|具体|复杂|简单|简约|简洁|纯白色?|白色)?"
        r"(?:的)?(?:背景|场景)",
        "",
        str(text or ""),
    )
    positive = re.sub(r"(?:不要|无需|不需要|没有|无)白底", "", positive)
    return re.sub(
        r"\b(?:no|without)\s+(?:a\s+)?"
        r"(?:(?:white|simple|complex)\s+)?background\b",
        "",
        positive,
        flags=re.I,
    )


def user_requests_explicit_background(text: str) -> bool:
    """Return whether the user's own text positively requests a scene/background."""
    raw = str(text or "").strip()
    if not raw:
        return False
    positive = _positive_background_text(raw)
    if "背景" in positive or "场景" in positive:
        return True
    if any(marker in positive for marker in _CHINESE_SCENE_MARKERS):
        return True
    return bool(
        _ENGLISH_SCENE_RE.search(positive)
        or re.search(r"\b(?:in|inside|at|on)\s+(?:an?\s+|the\s+)?\w+", positive, re.I)
    )


def strip_unrequested_default_background_tags(text: str, user_text: str) -> str:
    """Remove white/simple background tags invented despite an explicit scene."""
    request = _positive_background_text(user_text)
    keep_white = bool(
        re.search(r"(?:纯白|白色|白底)(?:的)?背景|\bwhite background\b", request, re.I)
    )
    keep_simple = bool(
        re.search(r"(?:简单|简约|简洁)(?:的)?背景|\bsimple background\b", request, re.I)
    )
    kept: list[str] = []
    for part in str(text or "").split(","):
        tag = part.strip()
        normalized = tag.lower().replace("_", " ")
        if normalized == "white background" and not keep_white:
            continue
        if normalized == "simple background" and not keep_simple:
            continue
        if tag:
            kept.append(tag)
    return ", ".join(kept)


def strip_unrequested_default_background_prose(text: str, user_text: str) -> str:
    """Remove contradictory default-background phrases without dropping a sentence."""
    request = _positive_background_text(user_text)
    if not user_requests_explicit_background(request):
        return str(text or "")
    keep_white = bool(
        re.search(r"(?:纯白|白色|白底)(?:的)?背景|\bwhite background\b", request, re.I)
    )
    keep_simple = bool(
        re.search(r"(?:简单|简约|简洁)(?:的)?背景|\bsimple background\b", request, re.I)
    )
    descriptors: list[str] = []
    if not keep_simple:
        descriptors.append("simple")
    if not keep_white:
        descriptors.append("white")
    if not descriptors:
        return str(text or "")
    descriptor = "|".join(descriptors)
    cleaned = re.sub(
        rf"\s*\b(?:with|against|on|before|over)\s+(?:an?\s+|the\s+)?"
        rf"(?:(?:{descriptor})\s+)+(?:plain\s+)?background\b",
        "",
        str(text or ""),
        flags=re.I,
    )
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def enforce_user_background_intent(
    text: str, background_mode: str, user_text: str
) -> tuple[str, str, bool]:
    """Let an explicit user scene override an incorrect LLM portrait decision."""
    if not user_requests_explicit_background(user_text):
        return str(text or ""), background_mode, False
    return (
        strip_unrequested_default_background_tags(text, user_text),
        EXPLICIT_SCENE,
        background_mode != EXPLICIT_SCENE,
    )


def extract_background_mode(text: str) -> tuple[str, str]:
    """Remove the LLM control marker and return its background decision.

    Args:
        text: Raw comma-separated LLM output.

    Returns:
        A pair containing cleaned tags and the normalized background mode. The
        mode is empty when the LLM omitted or contradicted the control marker.
    """
    raw = str(text or "").strip()
    found = {
        marker
        for marker in (DEFAULT_PORTRAIT_MARKER, EXPLICIT_SCENE_MARKER)
        if re.search(rf"(?<![\w]){re.escape(marker)}(?![\w])", raw, flags=re.I)
    }
    cleaned = raw
    for marker in (DEFAULT_PORTRAIT_MARKER, EXPLICIT_SCENE_MARKER):
        cleaned = re.sub(
            rf"\s*,?\s*(?<![\w]){re.escape(marker)}(?![\w])\s*,?\s*",
            ", ",
            cleaned,
            flags=re.I,
        )
    cleaned = re.sub(r"(?:\s*,\s*){2,}", ", ", cleaned).strip(" ,")
    if found == {DEFAULT_PORTRAIT_MARKER}:
        return cleaned, DEFAULT_PORTRAIT
    if found == {EXPLICIT_SCENE_MARKER}:
        return cleaned, EXPLICIT_SCENE
    return cleaned, ""


def apply_default_portrait_tags(
    text: str, *, include_full_body: bool = True
) -> str:
    """Ensure a clean white-background character illustration tag set.

    Args:
        text: Cleaned content tags produced by the normal prompt pipeline.

    Returns:
        Tags with compatible portrait framing and the required simple white
        background. Existing explicit framing is preserved.
    """
    tags = [part.strip() for part in str(text or "").split(",") if part.strip()]
    normalized = {tag.lower().replace("_", " ") for tag in tags}
    additions: list[str] = []
    if (
        include_full_body
        and not normalized.intersection(_FRAMING_TAGS)
        and "full body" not in normalized
    ):
        additions.append("full body")
    if "centered" not in normalized:
        additions.append("centered")
    for tag in ("simple background", "white background"):
        if tag not in normalized:
            additions.append(tag)
    return ", ".join((*tags, *additions))
