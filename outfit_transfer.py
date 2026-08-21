from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from .tag_cleaner import normalize_tag_key, split_tags
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from tag_cleaner import normalize_tag_key, split_tags


REFERENCE_MARKERS = (
    "参考引用图",
    "引用图",
    "参考这张图",
    "参考这图",
    "按这张图",
    "照这张图",
    "照着这张图",
    "根据这张图",
    "用这张图",
    "以这张图",
    "图中角色",
    "图里的角色",
    "图中衣服",
    "图里的衣服",
    "同款衣服",
    "参考图片",
    "参考图",
)

SEARCH_MARKERS = (
    "联网",
    "搜索",
    "搜一下",
    "查一下",
    "官方图",
    "设定图",
    "立绘",
    "资料",
)

OUTFIT_MARKERS = (
    "衣服",
    "服装",
    "换装",
    "同款",
    "服装tag",
    "服装 tag",
    "服饰",
)

_DIRECTIVE_SPLIT_RE = re.compile(
    r"(?:参考图视觉反推 tags[：:]|参考图原始正面提示词[：:]|引用法术正面提示词[：:])",
    re.S,
)
_SEARCH_SOURCE_RE = re.compile(
    r"(?:联网搜索|搜索|搜一下|查一下)(?P<subject>.*?)(?:的立绘|立绘|官方图|设定图|服装|衣服|外观|人设)",
    re.S,
)
_DIRECT_SOURCE_RE = re.compile(
    r"(?:穿上|换成|换上|套用|应用)(?P<subject>[\u4e00-\u9fffA-Za-z0-9·_\-]{1,32})的(?:衣服|服装)",
    re.S,
)
_WEARING_SOURCE_RE = re.compile(
    r"(?:穿着|穿上|换成|换上|套着|wearing|cosplaying(?:\s+as)?)\s*"
    r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9_.'!:+\-]{1,32}?)"
    r"(?:的)?(?:衣服|服装|演出服|outfit|costume)",
    re.I | re.S,
)
_CHARACTER_SOURCE_RE = re.compile(
    r"(?:角色|人物)(?P<subject>[\u4e00-\u9fffA-Za-z0-9·_\-]{1,32})",
    re.S,
)
_SUBJECT_TAIL_TRIM_RE = re.compile(
    r"(?:风格|衣服|服装|立绘|官方图|设定图|外观|人设|解析其?|提取其?|深度思考|应用到.*|为.*穿上.*)$"
)
_NEGATIVE_TARGET_RE = re.compile(r"(?:不要|不用|不使用)(?:固定角色|角色)")
_SPELL_NEGATIVE_SPLIT_RE = re.compile(r"\n\s*引用法术负面提示词[：:]", re.S)
_PRONOUN_SUBJECTS = {
    "她",
    "他",
    "它",
    "她们",
    "他们",
    "它们",
    "这个",
    "那个",
    "该角色",
    "这位角色",
}

_OUTFIT_HINTS = (
    "dress",
    "skirt",
    "gown",
    "apron",
    "apron dress",
    "uniform",
    "lolita",
    "gothic lolita",
    "robe",
    "cloak",
    "cape",
    "coat",
    "jacket",
    "blouse",
    "shirt",
    "top",
    "bodice",
    "corset",
    "sleeve",
    "sleeves",
    "collar",
    "neckline",
    "hem",
    "ruffle",
    "ruffled",
    "frill",
    "frills",
    "lace",
    "trim",
    "ribbon",
    "bow",
    "sash",
    "obi",
    "belt",
    "brooch",
    "pendant",
    "necklace",
    "choker",
    "tassel",
    "gem",
    "gemstone",
    "jewel",
    "ornament",
    "hair ornament",
    "hair ribbon",
    "hairpin",
    "headdress",
    "hat",
    "cap",
    "bonnet",
    "veil",
    "glove",
    "gloves",
    "stocking",
    "stockings",
    "thighhigh",
    "thighhighs",
    "sock",
    "socks",
    "pantyhose",
    "boots",
    "boot",
    "heels",
    "shoes",
    "shoe",
    "mary janes",
    "laurel",
    "mask",
    "cross motif",
    "church motif",
    "embroider",
    "embroidery",
    "brocade",
    "velvet",
    "silk",
    "satin",
    "chiffon",
    "organza",
    "metal trim",
    "gold trim",
    "silver trim",
    "detached sleeves",
    "puffy short sleeves",
    "long sleeves",
    "short sleeves",
)

_IDENTITY_HINTS = (
    "1girl",
    "1 girl",
    "solo",
    "girl",
    "boy",
    "hair",
    "eye",
    "eyes",
    "eyebrow",
    "eyelash",
    "mouth",
    "face",
    "skin",
    "ears",
    "tail",
    "wing",
    "wings",
    "halo",
    "horn",
    "horns",
    "fang",
    "animal ears",
    "cat ears",
    "fox ears",
    "long hair",
    "short hair",
    "twintails",
    "heterochromia",
    "vampire",
    "angel",
    "demon",
    "maid",
)

_SCENE_HINTS = (
    "background",
    "lighting",
    "light",
    "shadow",
    "angle",
    "full body",
    "upper body",
    "standing",
    "sitting",
    "pose",
    "expression",
    "smile",
    "blush",
    "looking at viewer",
    "white background",
    "simple background",
    "clean background",
)


@dataclass(frozen=True)
class OutfitTransferPlan:
    enabled: bool = False
    source_subject: str = ""
    target_character: str = ""
    source_from_reference: bool = False
    source_from_search: bool = False
    directive_prompt: str = ""


@dataclass(frozen=True)
class OutfitTransferContext:
    """Prompt context containing outfit tags without source identity features."""

    enabled: bool = False
    outfit_summary_source: str = ""
    outfit_summary: str = ""
    forbidden_identity_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserOutfitPatch:
    """One explicit, request-scoped change to a source outfit."""

    subject: str
    operation: str
    slot: str
    value: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class EffectiveOutfitPlan:
    """Source outfit after applying explicit user changes for this request."""

    subject: str = ""
    base_tags: tuple[str, ...] = ()
    effective_tags: tuple[str, ...] = ()
    removed_tags: tuple[str, ...] = ()
    added_tags: tuple[str, ...] = ()
    forbidden_slots: tuple[str, ...] = ()
    patches: tuple[UserOutfitPatch, ...] = ()

    @property
    def modified(self) -> bool:
        return bool(self.patches)

    @property
    def has_destructive_override(self) -> bool:
        return any(
            patch.operation in {"remove", "replace", "keep_only"}
            for patch in self.patches
        )

    @property
    def has_allowlist_override(self) -> bool:
        """Whether the request explicitly restricts clothing to listed layers."""
        return any(patch.operation == "keep_only" for patch in self.patches)


_COLOR_WORDS = {
    "粉红色": "pink",
    "粉色": "pink",
    "红色": "red",
    "白色": "white",
    "黑色": "black",
    "蓝色": "blue",
    "绿色": "green",
    "黄色": "yellow",
    "紫色": "purple",
    "灰色": "grey",
    "棕色": "brown",
    "褐色": "brown",
    "金色": "gold",
    "银色": "silver",
}

_GARMENT_SLOTS = {
    "上衣": "upper_body.primary",
    "衬衫": "upper_body.primary",
    "shirt": "upper_body.primary",
    "裙子": "lower_body.skirt",
    "短裙": "lower_body.skirt",
    "长裙": "lower_body.skirt",
    "半身裙": "lower_body.skirt",
    "skirt": "lower_body.skirt",
    "连衣裙": "one_piece.dress",
    "礼服": "one_piece.dress",
    "dress": "one_piece.dress",
    "外套": "outerwear",
    "夹克": "outerwear",
    "大衣": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "面具": "face_accessory.mask",
    "面罩": "face_accessory.mask",
    "mask": "face_accessory.mask",
    "手套": "handwear",
    "gloves": "handwear",
    "丝袜": "legwear",
    "连裤袜": "legwear",
    "裤袜": "legwear",
    "pantyhose": "legwear",
    "长筒袜": "legwear",
    "stockings": "legwear",
    "鞋": "footwear",
    "靴子": "footwear",
    "boots": "footwear",
}

_COLOR_TAG_WORDS = tuple(
    dict.fromkeys((*_COLOR_WORDS.values(), "orange", "beige", "navy"))
)


def outfit_tag_slot(tag: str) -> str:
    """Return the mutable garment slot occupied by one outfit tag."""
    key = normalize_tag_key(tag)
    if not key:
        return ""
    if "dress" in key or "gown" in key:
        return "one_piece.dress"
    if "skirt" in key:
        return "lower_body.skirt"
    if any(word in key for word in ("shorts", "pants", "trousers")):
        return "lower_body.pants"
    if any(word in key for word in ("panties", "underwear")):
        return "lower_body.underwear"
    if any(word in key for word in ("pantyhose", "stocking", "thighhigh", "sock")):
        return "legwear"
    if any(word in key for word in ("shirt", "blouse", "top", "bodice")):
        return "upper_body.primary"
    if "corset" in key:
        return "upper_body.corset"
    if any(word in key for word in ("jacket", "coat", "cloak", "cape")):
        return "outerwear"
    if "mask" in key:
        return "face_accessory.mask"
    if "glove" in key:
        return "handwear"
    if any(word in key for word in ("boot", "shoe", "heel", "mary jane")):
        return "footwear"
    if any(word in key for word in ("veil", "headdress", "hat", "headwear")):
        return "headwear"
    if any(word in key for word in ("hair ribbon", "hair ornament", "hairpin")):
        return "hair_accessory"
    if "ribbon" in key or "bow" in key:
        return "accessory.ribbon"
    return ""


def _tag_matches_slot(tag: str, slot: str) -> bool:
    actual = outfit_tag_slot(tag)
    if slot == "lower_body.all":
        return actual.startswith("lower_body.") or actual == "legwear"
    return actual == slot


def _character_aliases(name: str) -> tuple[str, ...]:
    text = str(name or "").strip()
    aliases = [text] if text else []
    chinese = re.sub(r"[^\u3400-\u9fff]", "", text)
    if len(chinese) >= 3:
        aliases.append(chinese[-2:])
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _clause_targets_character(
    clause: str, target_character: str, known_character_names: tuple[str, ...]
) -> bool:
    mentioned: set[str] = set()
    for name in known_character_names:
        if any(alias in clause for alias in _character_aliases(name)):
            mentioned.add(name)
    if not mentioned:
        return True
    return target_character in mentioned


def parse_user_outfit_patches(
    user_prompt: str,
    target_character: str,
    *,
    known_character_names: tuple[str, ...] = (),
) -> tuple[UserOutfitPatch, ...]:
    """Extract explicit outfit changes while retaining their source evidence."""
    if not target_character:
        return ()
    names = tuple(dict.fromkeys((*known_character_names, target_character)))
    garment_pattern = "|".join(
        sorted((re.escape(name) for name in _GARMENT_SLOTS), key=len, reverse=True)
    )
    color_pattern = "|".join(
        sorted((re.escape(name) for name in _COLOR_WORDS), key=len, reverse=True)
    )
    color_after_re = re.compile(
        rf"(?P<garment>{garment_pattern})(?:的)?(?:颜色)?\s*"
        rf"(?:改成|换成|变成|设为|是|为)\s*(?P<color>{color_pattern})"
    )
    color_before_re = re.compile(
        rf"(?P<color>{color_pattern})(?:的)?(?P<garment>{garment_pattern})"
    )
    remove_re = re.compile(
        rf"(?:没穿|没有穿|不穿|未穿|脱掉(?:了)?|去掉(?:了)?|不要)\s*(?:着)?(?:任何)?"
        rf"(?P<garment>{garment_pattern})"
    )
    add_re = re.compile(
        rf"(?:加上|加一件|添加|搭配|再穿|外面穿|戴上)\s*"
        rf"(?:一件|一条|一个|一双)?\s*(?P<color>{color_pattern})?"
        rf"(?:的)?(?P<garment>{garment_pattern})"
    )
    patches: list[UserOutfitPatch] = []
    for raw_clause in re.split(r"[，,。；;！？!?\n]+", _directive_text(user_prompt)):
        clause = raw_clause.strip()
        if not clause or not _clause_targets_character(clause, target_character, names):
            continue
        if re.search(r"下半身(?:什么|任何东西)?都没穿|下半身什么也没穿", clause):
            patches.append(
                UserOutfitPatch(
                    subject=target_character,
                    operation="remove",
                    slot="lower_body.all",
                    value="bottomless",
                    evidence=clause,
                )
            )
            continue
        remove_match = remove_re.search(clause)
        if remove_match:
            patches.append(
                UserOutfitPatch(
                    subject=target_character,
                    operation="remove",
                    slot=_GARMENT_SLOTS[remove_match.group("garment")],
                    evidence=clause,
                )
            )
            continue
        add_match = add_re.search(clause)
        if add_match:
            color_text = add_match.group("color") or ""
            patches.append(
                UserOutfitPatch(
                    subject=target_character,
                    operation="add",
                    slot=_GARMENT_SLOTS[add_match.group("garment")],
                    value=_COLOR_WORDS.get(color_text, ""),
                    evidence=clause,
                )
            )
            continue
        color_match = color_after_re.search(clause) or color_before_re.search(clause)
        if color_match:
            patches.append(
                UserOutfitPatch(
                    subject=target_character,
                    operation="replace",
                    slot=_GARMENT_SLOTS[color_match.group("garment")],
                    value=_COLOR_WORDS[color_match.group("color")],
                    evidence=clause,
                )
            )
    unique: list[UserOutfitPatch] = []
    seen: set[tuple[str, str, str]] = set()
    for patch in patches:
        key = (patch.operation, patch.slot, patch.value)
        if key not in seen:
            seen.add(key)
            unique.append(patch)
    return tuple(unique)


def _replace_tag_color(tag: str, color: str) -> str:
    words = normalize_tag_key(tag).split()
    remaining = [word for word in words if word not in _COLOR_TAG_WORDS]
    return "_".join((color, *remaining)) if remaining else color


def _default_tag_for_patch(patch: UserOutfitPatch) -> str:
    noun = {
        "upper_body.primary": "shirt",
        "lower_body.skirt": "skirt",
        "one_piece.dress": "dress",
        "outerwear": "jacket",
        "face_accessory.mask": "mask",
        "handwear": "gloves",
        "legwear": "pantyhose",
        "footwear": "boots",
    }.get(patch.slot, "")
    return "_".join(part for part in (patch.value, noun) if part)


def build_effective_outfit_plan(
    plan: OutfitTransferPlan,
    *,
    user_prompt: str,
    base_tags: tuple[str, ...],
    known_character_names: tuple[str, ...] = (),
    semantic_patches: tuple[UserOutfitPatch, ...] = (),
) -> EffectiveOutfitPlan:
    """Apply explicit user patches to a verified source-outfit profile."""
    normalized_base = tuple(dict.fromkeys(tag for tag in base_tags if str(tag).strip()))
    patches = tuple(
        dict.fromkeys(
            (
                *parse_user_outfit_patches(
                    user_prompt,
                    plan.target_character,
                    known_character_names=known_character_names,
                ),
                *semantic_patches,
            )
        )
    )
    effective = list(normalized_base)
    removed: list[str] = []
    added: list[str] = []
    forbidden_slots: list[str] = []
    for patch in patches:
        if patch.operation == "keep_only":
            allowed_slots = set(filter(None, patch.slot.split(",")))
            for tag in tuple(effective):
                if outfit_tag_slot(tag) not in allowed_slots:
                    effective.remove(tag)
                    if tag not in removed:
                        removed.append(tag)
            continue
        matching = [tag for tag in effective if _tag_matches_slot(tag, patch.slot)]
        if patch.operation in {"remove", "replace"}:
            for tag in matching:
                effective.remove(tag)
                if tag not in removed:
                    removed.append(tag)
            if patch.slot not in forbidden_slots:
                forbidden_slots.append(patch.slot)
        if patch.operation == "replace":
            color_targets = matching or (_default_tag_for_patch(patch),)
            for tag in color_targets:
                replacement = _replace_tag_color(tag, patch.value)
                if replacement and replacement not in effective:
                    effective.append(replacement)
                    added.append(replacement)
        elif patch.operation == "add":
            addition = _default_tag_for_patch(patch)
            if addition and addition not in effective:
                effective.append(addition)
                added.append(addition)
        elif patch.operation == "remove" and patch.value == "bottomless":
            if "bottomless" not in effective:
                effective.append("bottomless")
                added.append("bottomless")
    return EffectiveOutfitPlan(
        subject=plan.target_character,
        base_tags=normalized_base,
        effective_tags=tuple(effective),
        removed_tags=tuple(removed),
        added_tags=tuple(dict.fromkeys(added)),
        forbidden_slots=tuple(forbidden_slots),
        patches=patches,
    )


def bind_explicit_outfit_patch_target(
    plan: OutfitTransferPlan,
    *,
    user_prompt: str,
    known_character_names: tuple[str, ...] = (),
    fallback_character: str = "",
) -> OutfitTransferPlan:
    """Attach standalone clothing changes to their explicitly named character."""
    if plan.target_character:
        return plan
    directive = _directive_text(user_prompt)
    for name in known_character_names:
        aliases = _character_aliases(name)
        if not any(alias in directive for alias in aliases):
            continue
        if parse_user_outfit_patches(
            directive,
            name,
            known_character_names=known_character_names,
        ):
            return OutfitTransferPlan(
                enabled=plan.enabled,
                source_subject=plan.source_subject,
                target_character=name,
                source_from_reference=plan.source_from_reference,
                source_from_search=plan.source_from_search,
                directive_prompt=plan.directive_prompt,
            )
    explicit = re.search(
        r"(?P<target>[\u3400-\u9fffA-Za-z0-9·_\-]{1,24}?)(?:的)?"
        r"(?:下半身)?(?:没穿|没有穿|不穿|未穿|脱掉(?:了)?|去掉(?:了)?|不要)",
        directive,
    )
    target = explicit.group("target") if explicit else fallback_character
    target = re.sub(r"^(?:请画|画出|画|让)", "", str(target or "")).strip()
    if not target:
        return plan
    candidate = OutfitTransferPlan(
        enabled=plan.enabled,
        source_subject=plan.source_subject,
        target_character=target,
        source_from_reference=plan.source_from_reference,
        source_from_search=plan.source_from_search,
        directive_prompt=plan.directive_prompt,
    )
    if not parse_user_outfit_patches(
        directive,
        target,
        known_character_names=known_character_names,
    ):
        return plan
    return candidate


def rewrite_target_outfit_detail(
    detail: str, effective_plan: EffectiveOutfitPlan
) -> str:
    """Rewrite conflicting clothing prose only inside the target's Details block."""
    result = str(detail or "").strip()
    if not result or not effective_plan.modified:
        return result
    slot_nouns = {
        "upper_body.primary": ("shirt", "blouse", "top"),
        "lower_body.skirt": ("skirt",),
        "one_piece.dress": ("dress", "gown"),
        "outerwear": ("jacket", "coat", "cloak", "cape"),
        "face_accessory.mask": ("mask",),
        "handwear": ("gloves?",),
        "legwear": ("pantyhose", "stockings?", "thighhighs?"),
        "footwear": ("boots?", "shoes?"),
    }
    color_words = "|".join(_COLOR_TAG_WORDS)
    constraints: list[str] = []
    for patch in effective_plan.patches:
        nouns = slot_nouns.get(patch.slot, ())
        if patch.operation == "replace" and nouns:
            noun_pattern = "|".join(nouns)
            result = re.sub(
                rf"\b(?:{color_words})\s+(?P<noun>{noun_pattern})\b",
                lambda match: f"{patch.value} {match.group('noun')}",
                result,
                flags=re.I,
            )
            noun = re.sub(r"[?\\]", "", nouns[0])
            constraint = f"{patch.value} {noun}"
            if constraint.lower() not in result.lower():
                constraints.append(constraint)
        elif patch.operation == "remove":
            if patch.slot == "lower_body.all":
                constraints.append("nothing worn on the lower body")
                continue
            if nouns:
                noun_pattern = "|".join(nouns)
                result = re.sub(
                    rf"\bwears?\s+(?:an?\s+|the\s+)?(?:{color_words}\s+)?(?:{noun_pattern})\b",
                    lambda match: "wears no " + re.sub(
                        r"^(?:wears?\s+)(?:an?\s+|the\s+)?(?:"
                        + color_words
                        + r"\s+)?",
                        "",
                        match.group(0),
                        flags=re.I,
                    ),
                    result,
                    flags=re.I,
                )
                result = re.sub(
                    rf"(?:,\s*|\s+and\s+)(?:an?\s+|the\s+)?"
                    rf"(?:[a-z]+\s+){{0,2}}(?:{noun_pattern})\b",
                    "",
                    result,
                    flags=re.I,
                )
                noun = re.sub(r"[?\\]", "", nouns[0])
                constraint = f"no {noun}"
                if constraint.lower() not in result.lower():
                    constraints.append(constraint)
    if constraints:
        subject = result.split(None, 1)[0] if result else "the target character"
        clause = f"{subject} has " + " and ".join(dict.fromkeys(constraints))
        if clause.lower() not in result.lower():
            result = f"{result}; {clause}".strip("; ")
    return result


def build_outfit_constraint_narrative(
    effective_plan: EffectiveOutfitPlan,
    *,
    subject: str,
    source_tags: tuple[str, ...] = (),
) -> str:
    """Render explicit user outfit patches without implying extra nudity."""
    if not effective_plan.modified:
        return ""
    display_subject = str(subject or effective_plan.subject or "the target character")
    source = str(source_tags[0] if source_tags else "source costume").split("_(", 1)[0]
    source_label = source.replace("_", " ").strip()
    statements: list[str] = []
    slot_noun = {
        "upper_body.primary": "shirt",
        "lower_body.skirt": "skirt",
        "one_piece.dress": "dress",
        "outerwear": "outerwear",
        "face_accessory.mask": "mask",
        "handwear": "gloves",
        "legwear": "legwear",
        "footwear": "footwear",
    }
    for patch in effective_plan.patches:
        noun = slot_noun.get(patch.slot, "clothing")
        if patch.operation == "replace":
            statements.append(
                f"{display_subject} wears a {patch.value} {noun} as part of the "
                f"customized {source_label}-inspired outfit."
            )
        elif patch.operation == "remove" and patch.slot == "lower_body.all":
            statements.append(f"{display_subject} wears nothing on the lower body.")
        elif patch.operation == "remove":
            statements.append(f"{display_subject} wears no {noun}.")
    return " ".join(dict.fromkeys(statements))


def build_outfit_transfer_context(
    plan: OutfitTransferPlan, *, prompt: str
) -> OutfitTransferContext:
    """Build outfit-only context from an outfit transfer directive."""
    if not plan.enabled:
        return OutfitTransferContext()
    reference_tags = extract_reference_tag_text(prompt)
    summary = filter_outfit_tags(reference_tags)
    return OutfitTransferContext(
        enabled=True,
        outfit_summary_source="reference_filter" if summary else "directive",
        outfit_summary=summary,
        forbidden_identity_features=("hair", "eyes", "face", "ears", "tail"),
    )


def detect_outfit_transfer(
    prompt: str,
    fixed_character_name: str = "",
    fixed_character_names: tuple[str, ...] = (),
) -> OutfitTransferPlan:
    """Detect the "source outfit -> target character" task pattern."""
    text = str(prompt or "").strip()
    directive = _directive_text(text)
    if _NEGATIVE_TARGET_RE.search(directive):
        return OutfitTransferPlan(directive_prompt=directive)
    if not any(marker in directive for marker in OUTFIT_MARKERS):
        return OutfitTransferPlan(directive_prompt=directive)
    if (
        not fixed_character_name
        and not fixed_character_names
        and not _WEARING_SOURCE_RE.search(directive)
    ):
        return OutfitTransferPlan(directive_prompt=directive)
    known_targets = tuple(
        dict.fromkeys(
            name
            for name in (*fixed_character_names, fixed_character_name)
            if str(name).strip()
        )
    )
    target_character = _extract_target_character(
        directive, known_targets, fixed_character_name
    )
    source_subject = _extract_source_subject(directive, target_character)
    source_from_reference = any(marker in directive for marker in REFERENCE_MARKERS)
    source_from_search = any(marker in directive for marker in SEARCH_MARKERS)
    if not source_subject and not source_from_reference and not source_from_search:
        return OutfitTransferPlan(directive_prompt=directive)
    return OutfitTransferPlan(
        enabled=True,
        source_subject=source_subject,
        target_character=target_character,
        source_from_reference=source_from_reference,
        source_from_search=source_from_search,
        directive_prompt=directive,
    )


def preferred_search_prompt(plan: OutfitTransferPlan, prompt: str) -> str:
    """Return a search-focused prompt for outfit transfer tasks."""
    if not plan.enabled or not plan.source_from_search:
        return str(prompt or "").strip()
    subject = plan.source_subject or _directive_text(prompt)
    subject = _clean_subject(subject)
    if not subject:
        return str(prompt or "").strip()
    return f"{subject} 角色立绘 服装 外观 配色 饰品 官方图"


def extract_reference_tag_text(prompt: str) -> str:
    """Extract tag-like text from prompt-embedded reference blocks."""
    text = str(prompt or "").strip()
    spell_match = re.search(r"引用法术正面提示词[：:]\s*(.*)\Z", text, flags=re.S)
    if spell_match:
        return _SPELL_NEGATIVE_SPLIT_RE.split(spell_match.group(1), maxsplit=1)[
            0
        ].strip()
    for marker in ("参考图视觉反推 tags", "参考图原始正面提示词"):
        match = re.search(rf"{re.escape(marker)}[：:]\s*(.*)\Z", text, flags=re.S)
        if match:
            return match.group(1).strip()
    return ""


def filter_outfit_tags(text: str, max_tags: int = 48) -> str:
    """Keep only outfit-related tags from a tag-like stream."""
    tags = split_tags(text)
    seen: set[str] = set()
    kept: list[str] = []
    for tag in tags:
        key = normalize_tag_key(tag)
        if not key or key in seen:
            continue
        if _should_drop_tag(key):
            continue
        if not _looks_like_outfit_tag(key):
            continue
        seen.add(key)
        kept.append(tag)
        if len(kept) >= max_tags:
            break
    return ", ".join(kept)


def keep_only_verified_outfit_tags(
    text: str,
    verified_outfit_tags: tuple[str, ...],
    forbidden_slots: tuple[str, ...] = (),
    *,
    strict_allowlist: bool = True,
) -> str:
    """Drop LLM-invented clothing while retaining verified outfit evidence."""
    allowed = {normalize_tag_key(tag) for tag in verified_outfit_tags}
    kept: list[str] = []
    for tag in split_tags(text):
        key = normalize_tag_key(tag)
        if any(_tag_matches_slot(key, slot) for slot in forbidden_slots) and key not in allowed:
            continue
        if strict_allowlist and _looks_like_outfit_tag(key) and key not in allowed:
            continue
        kept.append(tag)
    return ", ".join(kept)


def build_outfit_summary_prompt(
    plan: OutfitTransferPlan,
    *,
    original_prompt: str,
    source_context: str,
) -> str:
    """Build a dedicated LLM prompt for outfit-only extraction."""
    source_subject = plan.source_subject or "参考对象"
    target_character = plan.target_character or "目标角色"
    directive = plan.directive_prompt or str(original_prompt or "").strip()
    return (
        "你是二次元角色服装解析助手。\n"
        "你的任务是从资料中提取一套可迁移给目标角色的服装 tags。\n"
        "只关注服装结构、层次、材质、配色、头饰、胸饰、腰饰、手套、袜鞋、装饰件。\n"
        "不要输出角色身份、名字、发色、瞳色、耳朵、尾巴、角、翅膀、年龄感、体型、表情、动作、背景、质量词或画师词。\n"
        "如果资料不足，可以做保守补全，但必须紧贴已有视觉线索，不要自行改主题。\n"
        "输出 24-60 个英文 Danbooru tags，用英文逗号分隔；不要解释，不要 Markdown。\n\n"
        f"来源对象：{source_subject}\n"
        f"目标角色：{target_character}\n"
        f"用户要求：{directive}\n\n"
        "资料：\n"
        f"{source_context.strip()}"
    )


def build_outfit_transfer_block(
    plan: OutfitTransferPlan,
    outfit_summary: str,
    effective_plan: EffectiveOutfitPlan | None = None,
) -> str:
    """Render the prompt-template block for outfit transfer tasks."""
    effective_plan = effective_plan or EffectiveOutfitPlan()
    if not plan.enabled and not effective_plan.modified:
        return ""
    lines = ["-----------"]
    if plan.enabled:
        lines.extend(
            [
                "本次是“来源角色/来源参考 -> 目标固定角色”的服装迁移任务。",
                f"最终主体必须是固定角色“{plan.target_character or '目标角色'}”。",
                "来源对象只用于提供服装结构、材质、配色和装饰；不要复制来源对象的角色身份、发色、瞳色、种族、耳朵、尾巴、角、翅膀、年龄感和体型。",
                "请优先让目标角色穿上来源服装，并保留目标角色自己的身份设定。",
                "如果资料不足，可以补齐服装细节，但不要擅自换成别的服装主题。",
            ]
        )
    if plan.source_subject:
        lines.append(f"来源对象：{plan.source_subject}")
    if outfit_summary:
        lines.extend(
            [
                "来源服装摘要 tags：",
                outfit_summary,
                "请优先围绕这份服装摘要生成最终内容 tags，而不是重新发散到来源对象的整套角色设定。",
            ]
        )
    if effective_plan.modified:
        if plan.enabled:
            lines.extend(
                [
                    "用户对来源服装作了本次请求级修改；用户修改的优先级高于来源服装摘要。",
                    "本次唯一有效的服装 tags：",
                    ", ".join(effective_plan.effective_tags),
                ]
            )
        else:
            lines.append(
                "用户明确规定了角色的衣着缺失或修改；只约束被点名的服装槽位，其他衣着不受影响。"
            )
        for patch in effective_plan.patches:
            value = f" -> {patch.value}" if patch.value else ""
            lines.append(
                f"用户明确修改：{patch.subject} {patch.operation} {patch.slot}{value}；原文证据：{patch.evidence}"
            )
        if effective_plan.removed_tags:
            lines.append(
                "已被用户覆盖或删除、禁止恢复的来源 tags："
                + ", ".join(effective_plan.removed_tags)
            )
        if plan.enabled:
            lines.append(
                "不要因为来源角色的默认造型而恢复已删除项，也不要添加本次有效服装列表以外的新服装。"
            )
        else:
            lines.append(
                "不得重新添加用户明确删除的衣物；除非用户同时明确说明，否则不要由此推断 bottomless、nude 或其他额外裸露状态。"
            )
    return "\n".join(lines)


def _directive_text(text: str) -> str:
    parts = _DIRECTIVE_SPLIT_RE.split(str(text or "").strip(), maxsplit=1)
    return parts[0].strip() if parts else str(text or "").strip()


def _extract_source_subject(directive: str, fixed_character_name: str) -> str:
    text = str(directive or "").strip()
    wearing_source = _WEARING_SOURCE_RE.search(text)
    if wearing_source:
        subject = _clean_subject(wearing_source.group("subject"))
        if subject and subject not in _PRONOUN_SUBJECTS:
            return subject
    if fixed_character_name:
        wearing_match = re.search(
            rf"{re.escape(fixed_character_name)}(?:正)?(?:穿着|穿上|换上|套着)"
            r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9·_\-]{1,32}?)(?:的)?(?:衣服|服装)",
            text,
            flags=re.S,
        )
        if wearing_match:
            subject = _clean_subject(wearing_match.group("subject"))
            if subject and subject not in _PRONOUN_SUBJECTS:
                return subject
    for pattern in (_SEARCH_SOURCE_RE, _DIRECT_SOURCE_RE, _CHARACTER_SOURCE_RE):
        match = pattern.search(text)
        if not match:
            continue
        subject = _clean_subject(match.group("subject"))
        if (
            subject
            and subject not in _PRONOUN_SUBJECTS
            and subject != fixed_character_name
        ):
            return subject
    outfit_match = re.search(
        r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9·_\-]{1,32})的(?:衣服|服装)",
        text,
        flags=re.S,
    )
    if outfit_match:
        subject = _clean_subject(outfit_match.group("subject"))
        if (
            subject
            and subject not in _PRONOUN_SUBJECTS
            and subject != fixed_character_name
        ):
            return subject
    return ""


def _extract_target_character(
    directive: str,
    known_targets: tuple[str, ...],
    fallback: str,
) -> str:
    """Bind an outfit transfer to the named wearer instead of config order."""
    text = str(directive or "")
    for name in known_targets:
        for alias in _character_aliases(name):
            if re.search(
                rf"{re.escape(alias)}\s*[（(]?\s*(?:正)?(?:穿着|穿上|换上|套着)",
                text,
            ):
                return name
    for name in known_targets:
        for alias in _character_aliases(name):
            if re.search(
                rf"(?:应用到|穿到|给|为)\s*{re.escape(alias)}|"
                rf"{re.escape(alias)}(?:身上|身穿)",
                text,
            ):
                return name
    explicit_wearer = re.search(
        r"(?P<target>[\u3400-\u9fffA-Za-z0-9·_\-]{1,32})\s*[（(]?\s*"
        r"(?:正)?(?:穿着|穿上|换上|套着)",
        text,
    )
    if explicit_wearer:
        target = explicit_wearer.group("target")
        target = re.sub(r"^(?:请画|画出|画|让)", "", target).strip()
        if target:
            return target
    return fallback or (known_targets[0] if len(known_targets) == 1 else "")


def _clean_subject(subject: str) -> str:
    text = str(subject or "").strip(" ，,。；;：:\n\t")
    text = _SUBJECT_TAIL_TRIM_RE.sub("", text).strip(" ，,。；;：:\n\t")
    text = re.sub(r"^(?:角色|人物)", "", text).strip()
    return text


def _should_drop_tag(key: str) -> bool:
    if any(hint in key for hint in _IDENTITY_HINTS):
        return True
    if any(hint in key for hint in _SCENE_HINTS):
        return True
    return False


def _looks_like_outfit_tag(key: str) -> bool:
    return any(hint in key for hint in _OUTFIT_HINTS)
