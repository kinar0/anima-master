from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - requests is expected in AstrBot env.
    requests = None


CHARACTER_CATEGORY = 4

DEFAULT_DONMAI_BASE_URLS = (
    "https://safebooru.donmai.us",
    "https://danbooru.donmai.us",
)
DEFAULT_SAFEBOORU_DAPI_URL = "https://safebooru.org/index.php"
DEFAULT_USER_AGENT = "AstrBotComfyUIAgent/0.13"

KNOWN_CORE_ALIASES: dict[str, tuple[str, ...]] = {
    "忍野忍": ("oshino_shinobu", "shinobu_oshino"),
    "洛茜": ("rossi_(arknights)", "rossi_(arknights:endfield)", "rossi"),
    "妃咲": ("kisaki_(blue_archive)", "kisaki", "hisaki_(blue_archive)", "hisaki"),
    "Kisaki": ("kisaki_(blue_archive)", "kisaki", "hisaki_(blue_archive)", "hisaki"),
    "kisaki": ("kisaki_(blue_archive)", "kisaki", "hisaki_(blue_archive)", "hisaki"),
    "キサキ": ("kisaki_(blue_archive)", "kisaki", "hisaki_(blue_archive)", "hisaki"),
    "铃兰": ("suzuran_(arknights)", "suzuran"),
    "鈴蘭": ("suzuran_(arknights)", "suzuran"),
    "Suzuran": ("suzuran_(arknights)", "suzuran"),
    "suzuran": ("suzuran_(arknights)", "suzuran"),
}
KNOWN_CANONICAL_CORE_TAGS: dict[str, str] = {
    "shinobu_oshino": "oshino_shinobu",
    "rossi": "rossi_(arknights)",
    "rossi_(arknights:endfield)": "rossi_(arknights)",
    "kisaki": "kisaki_(blue_archive)",
    "hisaki": "kisaki_(blue_archive)",
    "hisaki_(blue_archive)": "kisaki_(blue_archive)",
    "suzuran": "suzuran_(arknights)",
}

def _contains_alias(text: str, alias: str) -> bool:
    """Return whether text contains an alias, case-insensitively for ASCII."""
    return str(alias or "").lower() in str(text or "").lower()


def required_profile_tags_for_prompt(user_prompt: str) -> tuple[str, ...]:
    """Compatibility hook; variant profiles now come from semantic lookup."""
    return ()


def profile_hints_for_prompt(user_prompt: str) -> dict[str, str]:
    """Compatibility hook; local semantic evidence now supplies LLM context."""
    return {}

GENERAL_TAG_WORDS = {
    "arms",
    "background",
    "belt",
    "black",
    "blonde",
    "blue",
    "boots",
    "boy",
    "bow",
    "breasts",
    "brown",
    "cape",
    "choker",
    "dress",
    "ear",
    "ears",
    "eye",
    "eyes",
    "fang",
    "frill",
    "frilled",
    "gloves",
    "gold",
    "green",
    "grey",
    "hair",
    "hat",
    "hood",
    "jacket",
    "large",
    "long",
    "looking",
    "medium",
    "pale",
    "pink",
    "purple",
    "red",
    "ribbon",
    "shirt",
    "short",
    "skirt",
    "sleeves",
    "small",
    "smile",
    "solo",
    "standing",
    "stockings",
    "thighhighs",
    "trim",
    "white",
    "yellow",
    "girl",
    "cloak",
    "young",
}


@dataclass(frozen=True)
class TagRecord:
    name: str
    category: int
    post_count: int
    deprecated: bool = False
    source: str = ""


@dataclass(frozen=True)
class CoreTagResolution:
    text: str
    replacements: tuple[tuple[str, str, int, str], ...]
    inserted: tuple[tuple[str, int, str], ...]
    verified: tuple[tuple[str, int, str], ...] = ()
    status: str = "not_requested"
    canonical_tag: str = ""
    identity_tags: tuple[str, ...] = ()
    candidate_hints: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    explicit_request: bool = False


_STABLE_IDENTITY_EXACT_TAGS = {
    "ahoge",
    "animal_ears",
    "bat_wings",
    "cat_ears",
    "cat_tail",
    "demon_horns",
    "demon_wings",
    "double_bun",
    "fang",
    "fox_ears",
    "fox_tail",
    "gradient_hair",
    "halo",
    "heterochromia",
    "horns",
    "long_hair",
    "low_twintails",
    "multicolored_hair",
    "one_side_up",
    "pointy_ears",
    "ponytail",
    "short_hair",
    "side_ponytail",
    "streaked_hair",
    "tail",
    "twintails",
    "very_long_hair",
    "wings",
}
_STABLE_IDENTITY_PATTERNS = (
    re.compile(
        r"^(?:black|blonde|blue|brown|green|grey|gray|orange|pink|purple|red|silver|white)_hair$"
    ),
    re.compile(
        r"^(?:amber|black|blue|brown|gold|golden|green|grey|gray|orange|pink|purple|red|yellow)_eyes$"
    ),
)

_OUTFIT_PROFILE_TERMS = (
    "apron",
    "belt",
    "blouse",
    "bodice",
    "bonnet",
    "boots",
    "bow",
    "brooch",
    "cape",
    "choker",
    "cloak",
    "coat",
    "corset",
    "collar",
    "corset",
    "costume",
    "dress",
    "footwear",
    "frill",
    "gloves",
    "gown",
    "hat",
    "headdress",
    "jacket",
    "lace",
    "mask",
    "necktie",
    "necklace",
    "pantyhose",
    "pants",
    "pendant",
    "ribbon",
    "robe",
    "shirt",
    "shorts",
    "shoes",
    "skirt",
    "sleeves",
    "socks",
    "stockings",
    "thighhighs",
    "uniform",
    "veil",
)

_OUTFIT_GARMENT_ROOTS = {
    "blouse",
    "bodice",
    "coat",
    "corset",
    "dress",
    "gown",
    "jacket",
    "pants",
    "shirt",
    "shorts",
    "skirt",
    "uniform",
}
_OUTFIT_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}


def _outfit_slot_rank(tag: str) -> int:
    parts = set(tag.split("_"))
    if parts & {"shirt", "blouse", "bodice", "corset", "jacket", "coat", "dress", "gown", "uniform"}:
        return 0
    if parts & {"skirt", "shorts", "pants"}:
        return 1
    if "sleeves" in parts or "sleeve" in parts:
        return 2
    if "gloves" in parts or "glove" in parts:
        return 3
    if "ribbon" in parts or parts & {"hat", "headdress", "bonnet"}:
        return 4
    if "mask" in parts or "veil" in parts:
        return 5
    if parts & {"brooch", "necktie", "choker", "necklace", "pendant", "bow"}:
        return 6
    if parts & {"pantyhose", "stockings", "thighhighs", "socks"}:
        return 7
    if parts & {"boots", "shoes", "footwear"}:
        return 8
    return 9


@dataclass(frozen=True)
class VariantOutfitProfile:
    """One evidence-backed outfit cluster inferred from a character variant."""

    tags: tuple[str, ...] = ()
    appearance_tags: tuple[str, ...] = ()
    sample_mode: str = "none"
    total_posts: int = 0
    selected_posts: int = 0
    focused_posts: int = 0
    anchor_tag: str = ""
    tag_counts: tuple[tuple[str, int], ...] = ()


def _outfit_anchor_priority(tag: str) -> int:
    root = tag.split("_")[-1]
    if root in {"uniform", "corset"}:
        return 4
    if root in {"dress", "gown", "jacket", "coat", "bodice"}:
        return 3
    if root in {"shirt", "blouse"}:
        return 2
    if root in {"skirt", "shorts", "pants"}:
        return 1
    return 0


def _build_variant_outfit_profile(
    post_tag_strings: list[str],
    *,
    sample_mode: str = "unscoped",
    total_posts: int | None = None,
) -> VariantOutfitProfile:
    """Select a coherent recurring outfit and retain its extraction evidence."""
    post_sets = [set(tags.split()) for tags in post_tag_strings if tags.strip()]
    if len(post_sets) < 3:
        return VariantOutfitProfile(
            sample_mode=sample_mode,
            total_posts=total_posts if total_posts is not None else len(post_sets),
            selected_posts=len(post_sets),
        )

    def is_outfit_tag(tag: str) -> bool:
        parts = tag.split("_")
        return any(term in parts for term in _OUTFIT_PROFILE_TERMS) or any(
            term in tag for term in ("thighhigh", "pantyhose", "sleeve")
        )

    def is_appearance_tag(tag: str) -> bool:
        parts = set(tag.split("_"))
        return bool(
            parts
            & {"hair", "eyes", "skin", "horns", "ears", "tail", "wings", "scar"}
        )

    all_counts = Counter(tag for tags in post_sets for tag in tags if is_outfit_tag(tag))
    max_anchor_count = max(all_counts.values(), default=0)
    anchor_floor = max(3, (max_anchor_count * 35 + 99) // 100)
    anchors = [
        tag
        for tag, count in all_counts.items()
        if count >= anchor_floor
        and len(tag.split("_")) >= 2
        and tag.split("_")[-1] in _OUTFIT_GARMENT_ROOTS
    ]
    anchor = max(
        anchors,
        key=lambda tag: (
            _outfit_anchor_priority(tag),
            all_counts[tag],
            len(tag),
        ),
        default="",
    )
    focused = [tags for tags in post_sets if anchor in tags] if anchor else post_sets
    if len(focused) < 3:
        focused = post_sets

    counts = Counter(tag for tags in focused for tag in tags if is_outfit_tag(tag))
    # Once posts are character-pure and anchor-focused, a component appearing
    # in roughly 8% of the cluster is useful evidence.  Keep an absolute floor
    # of two posts so a single mistag cannot enter the persistent profile.
    floor = max(2, (len(focused) * 8 + 99) // 100)
    selected = {tag for tag, count in counts.items() if count >= floor}

    # A weakly occurring primary garment usually represents an alternate
    # costume, even inside character-pure posts.  Require 20% support for
    # competing tops/bottoms while retaining low-frequency accessories such as
    # masks and jewelry that are commonly under-tagged.
    selected = {
        tag
        for tag in selected
        if tag == anchor
        or tag.split("_")[-1] not in _OUTFIT_GARMENT_ROOTS
        or counts[tag] * 5 >= len(focused)
    }

    # Prefer one dominant color per garment/accessory root while allowing a
    # complementary form tag such as `masquerade_mask` beside `black_mask`.
    for root in (
        "shirt",
        "corset",
        "dress",
        "skirt",
        "gloves",
        "ribbon",
        "mask",
        "pantyhose",
        "stockings",
        "thighhighs",
        "boots",
        "shoes",
    ):
        colored = [
            tag
            for tag in selected
            if tag.endswith("_" + root) and tag.split("_", 1)[0] in _OUTFIT_COLOR_WORDS
        ]
        if len(colored) > 1:
            winner = max(colored, key=lambda tag: counts[tag])
            selected.difference_update(tag for tag in colored if tag != winner)

    # A dominant separate top+bottom outfit should not also be described as a
    # dress merely because some uploaders tagged the whole silhouette that way.
    anchor_root = anchor.split("_")[-1] if anchor else ""
    if anchor_root in {"shirt", "blouse", "bodice", "jacket", "coat"}:
        best_skirt = max(
            (counts[tag] for tag in selected if tag.split("_")[-1] == "skirt"),
            default=0,
        )
        best_dress = max(
            (counts[tag] for tag in selected if tag.split("_")[-1] in {"dress", "gown"}),
            default=0,
        )
        if best_skirt >= best_dress:
            selected = {
                tag for tag in selected if tag.split("_")[-1] not in {"dress", "gown"}
            }
    elif anchor_root in {"dress", "gown"}:
        selected = {
            tag
            for tag in selected
            if tag.split("_")[-1] not in {"shirt", "blouse", "skirt"}
            or tag == anchor
        }

    # Resolve contradictory sleeve lengths, then let a sufficiently common
    # composite (`puffy_short_sleeves`) replace its generic component tags.
    short_score = max(
        (counts[tag] for tag in selected if "short_sleeves" in tag), default=0
    )
    long_score = max(
        (counts[tag] for tag in selected if "long_sleeves" in tag), default=0
    )
    if short_score > long_score:
        selected = {tag for tag in selected if "long_sleeves" not in tag}
    elif long_score > short_score:
        selected = {tag for tag in selected if "short_sleeves" not in tag}

    for generic in tuple(selected):
        generic_parts = set(generic.split("_"))
        if any(
            generic != specific
            and generic_parts < set(specific.split("_"))
            and counts[specific] * 2 >= counts[generic]
            for specific in selected
        ):
            selected.discard(generic)

    tags = tuple(
        sorted(
            selected,
            key=lambda tag: (_outfit_slot_rank(tag), -counts[tag], tag),
        )[:14]
    )
    appearance_counts = Counter(
        tag for tags in focused for tag in tags if is_appearance_tag(tag)
    )
    appearance_floor = max(2, (len(focused) * 25 + 99) // 100)
    appearance_tags = tuple(
        tag
        for tag, _count in sorted(
            (
                (tag, count)
                for tag, count in appearance_counts.items()
                if count >= appearance_floor
            ),
            key=lambda item: (-item[1], item[0]),
        )[:10]
    )
    return VariantOutfitProfile(
        tags=tags,
        appearance_tags=appearance_tags,
        sample_mode=sample_mode,
        total_posts=total_posts if total_posts is not None else len(post_sets),
        selected_posts=len(post_sets),
        focused_posts=len(focused),
        anchor_tag=anchor,
        tag_counts=tuple((tag, counts[tag]) for tag in tags),
    )


def _select_variant_outfit_profile(post_tag_strings: list[str]) -> tuple[str, ...]:
    """Compatibility wrapper returning only the selected outfit tags."""
    return _build_variant_outfit_profile(post_tag_strings).tags


def required_core_tags_for_prompt(user_prompt: str) -> tuple[str, ...]:
    """Return locally known character anchors explicitly requested by the user."""
    text = str(user_prompt or "")
    anchors: list[str] = []
    for alias, queries in KNOWN_CORE_ALIASES.items():
        if not _contains_alias(text, alias):
            continue
        record = _known_canonical_record(list(queries))
        if record and record.name not in anchors:
            anchors.append(record.name)
    return tuple(anchors)


def _split_tags(text: str) -> list[str]:
    cleaned = str(text or "")
    cleaned = cleaned.replace("，", ",").replace("、", ",").replace(";", ",")
    cleaned = cleaned.replace("\n", ",")
    cleaned = re.sub(
        r"^(?:positive|prompt|tags|提示词|正向提示词)\s*[:：]",
        "",
        cleaned.strip(),
        flags=re.I,
    )
    parts = [part.strip(" \t\r\n,.;:：") for part in cleaned.split(",")]
    return [part for part in parts if part]


def _normalize_query(tag: str) -> str:
    value = str(tag or "").strip().lower()
    value = re.sub(r":\s*[\d.]+$", "", value)
    value = value.strip(" []{}")
    if (
        value.startswith("(")
        and value.endswith(")")
        and value.count("(") == 1
        and value.count(")") == 1
    ):
        value = value[1:-1].strip()
    value = re.sub(r"\s+", "_", value)
    return value


def _candidate_queries(tag: str) -> list[str]:
    query = _normalize_query(tag)
    if not query:
        return []
    queries = [query]
    parenthesized = re.match(r"^([a-z0-9_.'-]+)_\([^)]+\)$", query)
    if parenthesized:
        queries.append(parenthesized.group(1))
    if "(" not in query and ")" not in query:
        parts = [part for part in query.split("_") if part]
        if len(parts) == 2 and all(part not in GENERAL_TAG_WORDS for part in parts):
            queries.append(f"{parts[1]}_{parts[0]}")
    deduped: list[str] = []
    for item in queries:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _looks_like_core_tag(tag: str) -> bool:
    raw = str(tag or "").strip()
    if " " in raw and "(" not in raw and ")" not in raw:
        return False
    query = _normalize_query(tag)
    if not re.fullmatch(r"[a-z0-9_():.'-]{3,80}", query):
        return False
    if query.startswith("@"):
        return False
    if "_" not in query and "(" not in query:
        return False
    compact = query.replace("(", "_").replace(")", "_")
    parts = [part for part in compact.split("_") if part]
    if not parts or all(part in GENERAL_TAG_WORDS for part in parts):
        return False
    if parts[-1] in GENERAL_TAG_WORDS:
        return False
    if any(
        part in {"hair", "eyes", "dress", "skirt", "background", "smile"}
        for part in parts
    ):
        return False
    return True


def _http_get_json(
    url: str, *, params: dict[str, Any], timeout: float, user_agent: str
) -> Any:
    if requests is None:
        return None
    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json,text/plain,*/*",
            },
        )
        if response.status_code != 200:
            return None
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type and not response.text.lstrip().startswith(
            ("[", "{")
        ):
            return None
        return response.json()
    except (requests.exceptions.RequestException, ValueError):
        # One Donmai endpoint being slow or returning malformed JSON is an
        # expected source-level failure.  Let the caller try the next endpoint
        # and the read-only Safebooru DAPI fallback within the total budget.
        return None


def _fetch_donmai_tag(
    base_url: str, query: str, *, timeout: float, user_agent: str
) -> list[TagRecord]:
    data = _http_get_json(
        base_url.rstrip("/") + "/tags.json",
        params={"search[name_matches]": query, "limit": 10},
        timeout=timeout,
        user_agent=user_agent,
    )
    if not isinstance(data, list):
        return []
    records: list[TagRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        records.append(
            TagRecord(
                name=name,
                category=int(item.get("category") or 0),
                post_count=int(item.get("post_count") or 0),
                deprecated=bool(item.get("is_deprecated")),
                source=base_url,
            )
        )
    return records


def _fetch_donmai_autocomplete(
    base_url: str,
    query: str,
    *,
    timeout: float,
    user_agent: str,
) -> list[TagRecord]:
    data = _http_get_json(
        base_url.rstrip("/") + "/autocomplete.json",
        params={"search[type]": "tag_query", "search[query]": query},
        timeout=timeout,
        user_agent=user_agent,
    )
    if not isinstance(data, list):
        return []
    records: list[TagRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag") if isinstance(item.get("tag"), dict) else item
        name = str(item.get("value") or tag.get("name") or "").strip()
        if not name:
            continue
        records.append(
            TagRecord(
                name=name,
                category=int(item.get("category") or tag.get("category") or 0),
                post_count=int(item.get("post_count") or tag.get("post_count") or 0),
                deprecated=bool(tag.get("is_deprecated")),
                source=base_url + "/autocomplete",
            )
        )
    return records


def _fetch_safebooru_dapi_tag(
    query: str,
    *,
    timeout: float,
    user_agent: str,
) -> list[TagRecord]:
    """Fetch an exact tag from the Safebooru-compatible read-only DAPI.

    Args:
        query: Normalized tag name to query.
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header value.

    Returns:
        Parsed tag records, or an empty list when the request is unavailable.
    """
    if requests is None:
        return []
    try:
        response = requests.get(
            DEFAULT_SAFEBOORU_DAPI_URL,
            params={
                "page": "dapi",
                "s": "tag",
                "q": "index",
                "name": query,
            },
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/xml,text/xml,*/*",
            },
        )
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.text)
    except Exception:
        return []

    records: list[TagRecord] = []
    for item in root.findall("tag"):
        name = str(item.attrib.get("name") or "").strip()
        if not name:
            continue
        try:
            category = int(item.attrib.get("type") or 0)
            post_count = int(item.attrib.get("count") or 0)
        except (TypeError, ValueError):
            continue
        records.append(
            TagRecord(
                name=name,
                category=category,
                post_count=post_count,
                source="https://safebooru.org/dapi",
            )
        )
    return records


def _fetch_stable_identity_tags(
    canonical_tag: str,
    *,
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
) -> tuple[str, ...]:
    """Infer stable visible identity tags from solo posts.

    Args:
        canonical_tag: Verified Danbooru-style character tag.
        timeout: HTTP timeout for the Safebooru request.
        user_agent: User-Agent header value.
        cache: Shared resolver cache.

    Returns:
        High-frequency hair, eye, and anatomy identity tags.
    """
    cache_key = f"identity:{_normalize_query(canonical_tag)}"
    cached = cache.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        cached_at, cached_tags = cached
        ttl = 86400.0 if cached_tags else 600.0
        if time.monotonic() - float(cached_at) < ttl:
            return tuple(cached_tags)
    if requests is None:
        return ()
    try:
        response = requests.get(
            DEFAULT_SAFEBOORU_DAPI_URL,
            params={
                "page": "dapi",
                "s": "post",
                "q": "index",
                "tags": f"{canonical_tag} solo",
                "limit": 100,
            },
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/xml,text/xml,*/*",
            },
        )
        if response.status_code != 200:
            cache[cache_key] = (time.monotonic(), ())
            return ()
        root = ET.fromstring(response.text)
    except Exception:
        cache[cache_key] = (time.monotonic(), ())
        return ()

    posts = root.findall("post")
    if len(posts) < 4:
        cache[cache_key] = (time.monotonic(), ())
        return ()
    counts: Counter[str] = Counter()
    for post in posts:
        for tag in str(post.attrib.get("tags") or "").split():
            normalized = tag.strip().lower()
            if normalized in _STABLE_IDENTITY_EXACT_TAGS or any(
                pattern.fullmatch(normalized) for pattern in _STABLE_IDENTITY_PATTERNS
            ):
                counts[normalized] += 1
    threshold = max(1, int(len(posts) * 0.55 + 0.999))
    stable = tuple(
        tag.replace("_", " ")
        for tag, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= threshold
    )[:10]
    cache[cache_key] = (time.monotonic(), stable)
    return stable


def fetch_variant_outfit_profile(
    canonical_tag: str,
    *,
    outfit_kind: str = "default",
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
    donmai_base_urls: tuple[str, ...] = DEFAULT_DONMAI_BASE_URLS,
) -> VariantOutfitProfile:
    """Infer recurring visible outfit tags for a verified variant/persona.

    The local tag index validates the source tag but contains no post
    co-occurrence data.  This bounded read-only post sample supplies that
    missing relationship without any per-character profile table.
    """
    canonical_key = _normalize_query(canonical_tag)
    requested_kind = str(outfit_kind).strip().lower()
    kind = (
        requested_kind
        if requested_kind in {"default", "stage", "summer", "winter"}
        else "default"
    )
    cache_key = f"outfit-profile-v5:{canonical_key}:{kind}"
    cached = cache.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        cached_at, cached_profile = cached
        ttl = 86400.0 if getattr(cached_profile, "tags", ()) else 600.0
        if time.monotonic() - float(cached_at) < ttl:
            if isinstance(cached_profile, VariantOutfitProfile):
                return cached_profile
    if requests is None:
        return VariantOutfitProfile()
    post_samples: list[tuple[str, str]] = []
    for base_url in donmai_base_urls:
        try:
            response = requests.get(
                f"{base_url.rstrip('/')}/posts.json",
                    params={
                        "tags": " ".join(
                            filter(
                                None,
                                (
                                    canonical_tag,
                                    "instrument" if kind == "stage" else "",
                                    (
                                        f"{kind}_uniform"
                                        if kind in {"summer", "winter"}
                                        else ""
                                    ),
                                ),
                            )
                        ),
                        "limit": 100,
                    },
                timeout=timeout,
                headers={"User-Agent": user_agent, "Accept": "application/json"},
            )
            if response.status_code != 200:
                continue
            payload = response.json()
            if isinstance(payload, list):
                post_samples = [
                    (
                        str(post.get("tag_string_general") or ""),
                        str(post.get("tag_string_character") or ""),
                    )
                    for post in payload
                    if isinstance(post, dict)
                ]
            if len(post_samples) >= 3:
                break
        except Exception:
            continue
    if len(post_samples) < 3:
        try:
            response = requests.get(
                DEFAULT_SAFEBOORU_DAPI_URL,
                params={
                    "page": "dapi",
                    "s": "post",
                    "q": "index",
                    "tags": " ".join(
                        filter(
                            None,
                            (
                                canonical_tag,
                                (
                                    f"{kind}_uniform"
                                    if kind in {"summer", "winter"}
                                    else ""
                                ),
                            ),
                        )
                    ),
                    "limit": 100,
                },
                timeout=timeout,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/xml,text/xml,*/*",
                },
            )
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                post_samples = [
                    (str(post.attrib.get("tags") or ""), "")
                    for post in root.findall("post")
                ]
        except Exception:
            post_samples = []
    if len(post_samples) < 3:
        empty = VariantOutfitProfile(total_posts=len(post_samples))
        cache[cache_key] = (time.monotonic(), empty)
        return empty

    character_sets = [
        frozenset(characters.split())
        for _general, characters in post_samples
        if canonical_key in characters.split()
    ]
    signature_counts = Counter(character_sets)
    pure_signature = min(
        (
            signature
            for signature, count in signature_counts.items()
            if count >= 3
        ),
        key=lambda signature: (
            len(signature),
            -signature_counts[signature],
            sorted(signature),
        ),
        default=frozenset(),
    )
    single_character = [
        general
        for general, characters in post_samples
        if pure_signature and frozenset(characters.split()) == pure_signature
    ]
    if kind == "stage":
        single_character = [
            general
            for general in single_character
            if not {
                "school_uniform",
                "haneoka_school_uniform",
            }
            & set(general.split())
        ]
    elif kind in {"summer", "winter"}:
        seasonal_tag = f"{kind}_uniform"
        single_character = [
            general for general in single_character if seasonal_tag in general.split()
        ]
        if len(single_character) < 3:
            single_character = [
                general
                for general, _characters in post_samples
                if seasonal_tag in general.split()
            ]
    if len(single_character) >= 3:
        selected = single_character
        sample_mode = (
            (
                f"{kind}_single_character"
                if pure_signature
                else f"{kind}_filtered"
            )
            if kind in {"summer", "winter"}
            else "stage_single_character"
            if kind == "stage"
            else "single_character"
        )
    else:
        if kind in {"stage", "summer", "winter"}:
            empty = VariantOutfitProfile(
                sample_mode=f"{kind}_evidence_insufficient",
                total_posts=len(post_samples),
                selected_posts=len(single_character),
            )
            cache[cache_key] = (time.monotonic(), empty)
            return empty
        solo = [
            general
            for general, _characters in post_samples
            if {"solo", "1girl", "1boy"} & set(general.split())
        ]
        if len(solo) >= 3:
            selected = solo
            sample_mode = "solo_fallback"
        else:
            selected = [general for general, _characters in post_samples]
            sample_mode = "unscoped_fallback"
    profile = _build_variant_outfit_profile(
        selected,
        sample_mode=sample_mode,
        total_posts=len(post_samples),
    )
    if (
        sample_mode
        in {
            "single_character",
            "stage_single_character",
            "summer_single_character",
            "winter_single_character",
            "summer_filtered",
            "winter_filtered",
        }
        and pure_signature
        and profile.anchor_tag
        and requests is not None
    ):
        refined_samples: list[str] = []
        for base_url in donmai_base_urls:
            try:
                response = requests.get(
                    f"{base_url.rstrip('/')}/posts.json",
                    params={
                        "tags": f"{canonical_tag} {profile.anchor_tag}",
                        "limit": 100,
                    },
                    timeout=timeout,
                    headers={"User-Agent": user_agent, "Accept": "application/json"},
                )
                if response.status_code != 200:
                    continue
                payload = response.json()
                if isinstance(payload, list):
                    refined_samples = [
                        str(post.get("tag_string_general") or "")
                        for post in payload
                        if isinstance(post, dict)
                        and frozenset(
                            str(post.get("tag_string_character") or "").split()
                        )
                        == pure_signature
                        and not (
                            kind == "stage"
                            and {
                                "school_uniform",
                                "haneoka_school_uniform",
                            }
                            & set(str(post.get("tag_string_general") or "").split())
                        )
                        and (
                            kind not in {"summer", "winter"}
                            or f"{kind}_uniform"
                            in set(str(post.get("tag_string_general") or "").split())
                        )
                    ]
                if len(refined_samples) >= 3:
                    break
            except Exception:
                continue
        if len(refined_samples) > profile.focused_posts:
            profile = _build_variant_outfit_profile(
                refined_samples,
                sample_mode=(
                    f"{kind}_single_character_anchor"
                    if kind != "default"
                    else "single_character_anchor"
                ),
                total_posts=len(post_samples),
            )
    cache[cache_key] = (time.monotonic(), profile)
    return profile


def fetch_variant_outfit_tags(
    canonical_tag: str,
    *,
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
    outfit_kind: str = "default",
    donmai_base_urls: tuple[str, ...] = DEFAULT_DONMAI_BASE_URLS,
) -> tuple[str, ...]:
    """Compatibility wrapper returning tags from the evidence-backed profile."""
    return fetch_variant_outfit_profile(
        canonical_tag,
        outfit_kind=outfit_kind,
        timeout=timeout,
        user_agent=user_agent,
        cache=cache,
        donmai_base_urls=donmai_base_urls,
    ).tags


def _fetch_tag_records(
    query: str,
    *,
    donmai_base_urls: tuple[str, ...],
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
) -> list[TagRecord]:
    query = _normalize_query(query)
    if not query:
        return []
    cached = cache.get(query)
    if isinstance(cached, tuple) and len(cached) == 2:
        cached_at, cached_records = cached
        ttl = 3600.0 if cached_records else 60.0
        if time.monotonic() - float(cached_at) < ttl:
            return list(cached_records)
    elif isinstance(cached, list):
        return cached

    records: list[TagRecord] = []
    for base_url in donmai_base_urls:
        records = _fetch_donmai_tag(
            base_url, query, timeout=timeout, user_agent=user_agent
        )
        if records:
            break
    if not records:
        records = _fetch_safebooru_dapi_tag(
            query,
            timeout=timeout,
            user_agent=user_agent,
        )
    cache[query] = (time.monotonic(), records)
    return records


def _fetch_autocomplete_records(
    query: str,
    *,
    donmai_base_urls: tuple[str, ...],
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
) -> list[TagRecord]:
    query = _normalize_query(query)
    if not query:
        return []
    cache_key = f"autocomplete:{query}"
    cached = cache.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        cached_at, cached_records = cached
        ttl = 3600.0 if cached_records else 60.0
        if time.monotonic() - float(cached_at) < ttl:
            return list(cached_records)
    elif isinstance(cached, list):
        return cached

    records: list[TagRecord] = []
    for base_url in donmai_base_urls:
        records = _fetch_donmai_autocomplete(
            base_url,
            query,
            timeout=timeout,
            user_agent=user_agent,
        )
        if records:
            break
    cache[cache_key] = (time.monotonic(), records)
    return records


def _best_character(
    queries: list[str],
    *,
    donmai_base_urls: tuple[str, ...],
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
    use_autocomplete: bool = False,
) -> TagRecord | None:
    records: list[TagRecord] = []
    for query in queries:
        records.extend(
            _fetch_tag_records(
                query,
                donmai_base_urls=donmai_base_urls,
                timeout=timeout,
                user_agent=user_agent,
                cache=cache,
            )
        )
    if use_autocomplete:
        for query in queries:
            records.extend(
                _fetch_autocomplete_records(
                    query,
                    donmai_base_urls=donmai_base_urls,
                    timeout=timeout,
                    user_agent=user_agent,
                    cache=cache,
                )
            )
    characters = [
        record
        for record in records
        if record.category == CHARACTER_CATEGORY and not record.deprecated
    ]
    if characters:
        return max(characters, key=lambda record: record.post_count)
    return None


def _known_canonical_record(queries: list[str]) -> TagRecord | None:
    for query in queries:
        canonical = KNOWN_CANONICAL_CORE_TAGS.get(_normalize_query(query))
        if canonical:
            return TagRecord(
                name=canonical,
                category=CHARACTER_CATEGORY,
                post_count=0,
                deprecated=False,
                source="local_alias",
            )
    return None


def _resolve_evidence_candidates(
    candidates: tuple[str, ...],
    *,
    donmai_base_urls: tuple[str, ...],
    timeout: float,
    user_agent: str,
    cache: dict[str, Any],
) -> tuple[TagRecord | None, tuple[str, ...]]:
    """Select a canonical character tag from evidence-backed LLM candidates.

    Args:
        candidates: Bounded Danbooru-style candidates proposed by the LLM.
        donmai_base_urls: Ordered Donmai-compatible API roots.
        timeout: Per-request HTTP timeout.
        user_agent: User-Agent header value.
        cache: Shared resolver cache.

    Returns:
        Best verified record and compact evidence strings.
    """
    scored: list[tuple[int, TagRecord, str]] = []
    for candidate in candidates[:8]:
        normalized = _normalize_query(candidate)
        if not re.fullmatch(r"[a-z0-9_.'():-]{3,100}", normalized):
            continue
        scoped_identity = bool(
            re.fullmatch(
                r"[a-z0-9_.'-]{2,}_\([a-z0-9_.' :-]{2,}\)",
                normalized,
            )
        )
        for query in _candidate_queries(normalized):
            records = _fetch_tag_records(
                query,
                donmai_base_urls=donmai_base_urls,
                timeout=timeout,
                user_agent=user_agent,
                cache=cache,
            )
            for record in records:
                record_name = _normalize_query(record.name)
                exact = record_name == normalized
                if record.category == CHARACTER_CATEGORY:
                    score = 100 + (30 if exact else 0)
                elif (
                    exact
                    and scoped_identity
                    and record.post_count > 0
                    and record.source.endswith("/dapi")
                ):
                    # Safebooru's DAPI reports many imported character tags as
                    # type=0, so a scoped exact tag with real posts is accepted
                    # as degraded evidence instead of being discarded.
                    score = 70
                else:
                    continue
                score += min(20, len(str(max(1, record.post_count))))
                detail = (
                    f"{record.name}|score={score}|count={record.post_count}|"
                    f"source={record.source}|category={record.category}"
                )
                scored.append((score, record, detail))
    if not scored:
        return None, ()
    scored.sort(key=lambda item: (item[0], item[1].post_count), reverse=True)
    best_score, best_record, _ = scored[0]
    if len(scored) > 1 and best_score - scored[1][0] < 5:
        first = _normalize_query(best_record.name)
        second = _normalize_query(scored[1][1].name)
        if first != second:
            return None, tuple(item[2] for item in scored[:6])
    return best_record, tuple(item[2] for item in scored[:6])


def _user_character_queries(user_prompt: str) -> list[str]:
    """Extract likely explicitly named characters from the raw user request.

    Args:
        user_prompt: Original natural-language request.

    Returns:
        A small ordered list of names suitable for autocomplete lookup.
    """
    text = str(user_prompt or "").strip()
    queries: list[str] = []
    name_chars = r"\u4e00-\u9fffぁ-んァ-ヶー"
    stop = (
        r"(?=穿|戴|拿|手持|站|坐|躺|跑|走|看|望|笑|哭|和|与|在|，|。|、|"
        r",|;|；|\s|$)"
    )
    patterns = (
        rf"(?:游戏|手游|动画|动漫|漫画)?角色\s*([{name_chars}]{{2,12}}?){stop}",
        rf"(?:画|生成|绘制|来一张|生图)\s*([{name_chars}]{{2,12}}?){stop}",
    )
    generic_people = {
        "一个女孩",
        "一位女孩",
        "一个少女",
        "一位少女",
        "女孩",
        "少女",
        "人物",
        "角色",
        "女人",
        "男性",
        "男人",
        "少年",
    }
    if (
        re.fullmatch(rf"[{name_chars}]{{2,8}}", text)
        and text not in generic_people
        and "角色" not in text
        and not any(
            marker in text
            for marker in (
                "穿",
                "戴",
                "拿",
                "站",
                "坐",
                "躺",
                "跑",
                "走",
                "看",
                "望",
                "笑",
                "哭",
                "在",
                "和",
                "与",
            )
        )
    ):
        queries.append(text)
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            if name in generic_people:
                continue
            if name.startswith(("一个", "一位")):
                name = name[2:]
            if 2 <= len(name) <= 12 and name not in generic_people:
                if name not in queries:
                    queries.append(name)
            if len(queries) >= 2:
                return queries
    return queries


def character_resolution_requested(
    text: str,
    *,
    user_prompt: str = "",
    candidate_hints: tuple[str, ...] = (),
) -> bool:
    """Return whether a request carries credible named-character evidence.

    Args:
        text: LLM-generated Danbooru tags.
        user_prompt: Original natural-language request.
        candidate_hints: Bounded candidates from the character planner.

    Returns:
        True when character lookup failure should be surfaced to the caller.
    """
    scoped_candidate = any(
        "_(" in _normalize_query(tag)
        for tag in _split_tags(text)
        if _looks_like_core_tag(tag)
    )
    return bool(
        candidate_hints
        or scoped_candidate
        or _user_character_queries(user_prompt)
        or any(_contains_alias(user_prompt, alias) for alias in KNOWN_CORE_ALIASES)
    )


def resolve_core_tags(
    text: str,
    *,
    user_prompt: str = "",
    allow_insert: bool = False,
    candidate_hints: tuple[str, ...] = (),
    max_candidates: int = 6,
    timeout: float = 6.0,
    donmai_base_urls: tuple[str, ...] = DEFAULT_DONMAI_BASE_URLS,
    user_agent: str = DEFAULT_USER_AGENT,
    cache: dict[str, Any] | None = None,
) -> CoreTagResolution:
    tag_cache = cache if cache is not None else {}
    tags = _split_tags(text)
    replacements: list[tuple[str, str, int, str]] = []
    inserted: list[tuple[str, int, str]] = []
    verified: list[tuple[str, int, str]] = []
    evidence: tuple[str, ...] = ()
    canonical_record: TagRecord | None = None

    candidate_indexes: list[int] = []
    for index, tag in enumerate(tags[: max(max_candidates * 3, 12)]):
        if _looks_like_core_tag(tag):
            candidate_indexes.append(index)
        if len(candidate_indexes) >= max_candidates:
            break

    for index in candidate_indexes:
        original = tags[index]
        queries = _candidate_queries(original)
        best = _known_canonical_record(queries) or _best_character(
            queries,
            donmai_base_urls=donmai_base_urls,
            timeout=timeout,
            user_agent=user_agent,
            cache=tag_cache,
            use_autocomplete=False,
        )
        original_key = _normalize_query(original)
        if not best:
            continue
        canonical_record = best
        verified.append((best.name, best.post_count, best.source))
        if best.name != original_key or best.name != original.strip():
            tags[index] = best.name
            replacements.append((original, best.name, best.post_count, best.source))
        break

    if allow_insert:
        existing = {_normalize_query(tag) for tag in tags}
        for alias, queries in KNOWN_CORE_ALIASES.items():
            if not _contains_alias(user_prompt, alias):
                continue
            alias_queries = list(queries)
            best = _known_canonical_record(alias_queries) or _best_character(
                alias_queries,
                donmai_base_urls=donmai_base_urls,
                timeout=timeout,
                user_agent=user_agent,
                cache=tag_cache,
                use_autocomplete=True,
            )
            if not best:
                continue
            canonical_record = best
            alias_keys = {_normalize_query(query) for query in alias_queries}
            replaced = False
            for index, tag in enumerate(tags):
                if (
                    _normalize_query(tag) not in alias_keys
                    or _normalize_query(tag) == best.name
                ):
                    continue
                old = tags[index]
                tags[index] = best.name
                replacements.append((old, best.name, best.post_count, best.source))
                existing.discard(_normalize_query(old))
                existing.add(best.name)
                replaced = True
                break
            if not replaced and best.name not in existing:
                tags.insert(0, best.name)
                existing.add(best.name)
                inserted.append((best.name, best.post_count, best.source))

        if candidate_hints:
            best, evidence = _resolve_evidence_candidates(
                tuple(candidate_hints),
                donmai_base_urls=donmai_base_urls,
                timeout=timeout,
                user_agent=user_agent,
                cache=tag_cache,
            )
            if best:
                canonical_record = best
                verified.append((best.name, best.post_count, best.source))
                existing_index = next(
                    (
                        index
                        for index, tag in enumerate(tags)
                        if _normalize_query(tag) == _normalize_query(best.name)
                    ),
                    None,
                )
                if existing_index is not None:
                    existing.add(_normalize_query(best.name))
                elif candidate_indexes:
                    index = candidate_indexes[0]
                    old = tags[index]
                    tags[index] = best.name
                    existing.discard(_normalize_query(old))
                    existing.add(_normalize_query(best.name))
                    replacements.append((old, best.name, best.post_count, best.source))
                elif _normalize_query(best.name) not in existing:
                    tags.insert(0, best.name)
                    existing.add(_normalize_query(best.name))
                    inserted.append((best.name, best.post_count, best.source))

        if not inserted:
            for raw_name in _user_character_queries(user_prompt):
                best = _best_character(
                    [raw_name],
                    donmai_base_urls=donmai_base_urls,
                    timeout=timeout,
                    user_agent=user_agent,
                    cache=tag_cache,
                    use_autocomplete=True,
                )
                if not best or best.name in existing:
                    continue
                canonical_record = best
                if candidate_indexes:
                    index = candidate_indexes[0]
                    old = tags[index]
                    tags[index] = best.name
                    existing.discard(_normalize_query(old))
                    existing.add(best.name)
                    replacements.append((old, best.name, best.post_count, best.source))
                else:
                    tags.insert(0, best.name)
                    existing.add(best.name)
                    inserted.append((best.name, best.post_count, best.source))
                break

    identity_tags: tuple[str, ...] = ()
    if canonical_record:
        stable_tags = _fetch_stable_identity_tags(
            canonical_record.name,
            timeout=timeout,
            user_agent=user_agent,
            cache=tag_cache,
        )
        identity_tags = (canonical_record.name, *stable_tags)
    explicit_request = bool(
        allow_insert
        and (
            _user_character_queries(user_prompt)
            or any(
                _contains_alias(user_prompt, alias)
                for alias in KNOWN_CORE_ALIASES
            )
        )
    )
    requested = character_resolution_requested(
        text,
        user_prompt=user_prompt if allow_insert else "",
        candidate_hints=candidate_hints,
    )
    return CoreTagResolution(
        text=", ".join(tags),
        replacements=tuple(replacements),
        inserted=tuple(inserted),
        verified=tuple(verified),
        status=(
            "resolved"
            if canonical_record
            else ("unresolved" if requested else "not_requested")
        ),
        canonical_tag=canonical_record.name if canonical_record else "",
        identity_tags=identity_tags,
        candidate_hints=tuple(candidate_hints),
        evidence=evidence,
        explicit_request=explicit_request,
    )
