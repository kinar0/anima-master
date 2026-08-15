from __future__ import annotations

import hashlib

try:
    from .prompt_background import (
        DEFAULT_PORTRAIT_MARKER,
        EXPLICIT_SCENE_MARKER,
    )
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from prompt_background import DEFAULT_PORTRAIT_MARKER, EXPLICIT_SCENE_MARKER

DEFAULT_LLM_PROMPT_TEMPLATE = """你是为 Anima 图像生成模型编写正面提示词的 AI 画师。

请根据用户的原始要求设计一幅完整、协调、具有视觉吸引力的画面，并将结果输出为英文 Danbooru-style tags。

输出要求：
- 只输出七个单行花括号字段，字段外不得输出任何文字：
  `{{Count: 2girls, yuri}}`
  `{{Characters: chihaya_anon, togawa_sakiko}}`
  `{{Copyright: bang_dream!}}`
  `{{Identity: chihaya_anon has pink hair and grey eyes; togawa_sakiko has blue hair and yellow eyes}}`
  `{{Details: chihaya_anon wears grey pantyhose; togawa_sakiko wears black pantyhose}}`
  `{{Tags: full body, composition, lighting, background, creative visual details}}`
  `{{Nltags: chihaya_anon and togawa_sakiko ...}}`
- `Count` 必须含准确的人数 Danbooru tag；`Characters` 只能含角色名；`Copyright` 只能含这些角色所属作品的标准 Danbooru copyright tags，同一作品只写一次。原创或无法确认作品时将 `Copyright` 留空，不要猜测。每个角色必须恰好在 `Identity` 和 `Details` 中各出现一次。
- 不要输出解释、分析、标题、编号、Markdown、代码块或中文。
- 不要输出 masterpiece、best quality、score 等质量前缀。
- 不要输出画师 tags；质量词和画师组会由程序另行拼接。
- 尽量使用模型容易理解的可见画面描述。
- 保持用户明确指定的角色、主体、人数、关键服装、动作、表情和道具。
- 除上述明确要求外，可以自由决定服装细节、姿态、构图、镜头、光影、色彩、氛围和特效。
- 用户未明确要求地点、环境或背景时，按单张角色立绘设计，不要自行创造场景。
- 以最终图像协调、精致、有表现力和好看为优先，不需要机械追求固定 Tag 数量。
- 不要为了数量重复同义词；画面已经完整时即可停止。
- 请自行解决明显冲突，直接输出你认为最适合生成最终画面的版本。

角色和动态上下文：
{character_rule}
{search_block}
{outfit_transfer_rule}
{reference_rule}
{img2img_rule}
{sensual_rule}

用户原始要求：
{theme}"""


BACKGROUND_POLICY_TEMPLATE = """

-----------
背景与立绘策略（此规则追加在自定义模板之后，必须遵守）：
- 只根据下方“用户原始文字”判断用户是否明确要求了地点、环境或背景；不要把参考图、引用法术、图片反推或搜索摘要中的场景误当成用户要求。
- 若用户明确要求地点、环境、天气场景或保留原背景：保留该场景，不要添加 white background，并在输出末尾追加控制标记 {explicit_marker}。
- 若用户没有明确要求背景：使用 simple background、white background；有角色时选择与用户动作及镜头相容的 full body/upper body、居中构图、姿势和视线，无角色或仅身体局部时不得补造完整人物或 full body；不要自行创造室内、街道、自然景观、建筑、家具或前景道具，并在输出末尾追加控制标记 {default_marker}。
- 两个控制标记只能选择一个。控制标记供程序读取，不属于 Danbooru tag，必须放在最后一项。

用户原始文字：
{original_theme}
"""


LEGACY_BUILTIN_TEMPLATE_HASHES = {
    "1ca427c3208fc3d59f66d0a4c033a6ce19745d7df1858c96d009cc5a3460fa1c",
    "f63d42fcc21ae1d9dcc5a94c63c787f4e7d699e6c76fb3da90a1a46a2a0978f8",
    "95d7bfecaa58255d97685577e7ad2ddeac595b6237d3dd623407f077646bd2cc",
    "5e7ec9859914e12a8af8ccbc893d3658bd3c73e01e8fbda32367b1a0d8960e88",
    "1e96479f397110dc67364a929426c43f07b24a0b634af26c8e7d97002aeb27a0",
    "bd243cc4adfcf1b6a5440ec05bd87268ad5b0b57c4182aefaa76721b45ecb630",
    "ad278955975adbc09aef45fa80a9ee50730acda5d4ad515cde6a3ab3f9c8161d",
    "f066ab9668b1fec8e9814bab7b98c97640d733fc4adafca35a731636d6a62c88",
    "b0c2da43cb583bc70db1218aa8183e46657668a981cff1c1e912782c067a437a",
    "7d27e4693a6cb355eb4c30e5e268b029402b0c2860bb1f9acba4d86d701c7c62",
}


def build_llm_prompt(
    theme: str,
    search_context: str = "",
    fixed_character: bool = False,
    character_name: str = "",
    sensual_mode: bool = False,
    mode: str = "txt2img",
    prompt_builder_template: str = "",
    outfit_transfer_rule: str = "",
    original_theme: str = "",
    fixed_character_hints: dict[str, str] | None = None,
) -> str:
    """Build the prompt sent to the chat LLM for Danbooru tag generation.

    Args:
        theme: User-requested image theme.
        search_context: Optional web research context.
        fixed_character: Whether fixed character tags will be composed later.
        character_name: Selected fixed character name.
        sensual_mode: Whether the request enables sensual presentation.
        mode: Generation mode such as `txt2img` or `img2img`.
        prompt_builder_template: Optional custom prompt template.
        outfit_transfer_rule: Optional outfit-transfer instructions.
        original_theme: User text before reference-image or quoted-spell expansion.
        fixed_character_hints: Locally saved character identity hints found in
            the user request. Values may mix tags and natural language.

    Returns:
        Complete instruction text for the prompt-building LLM.
    """
    theme = str(theme or "").strip()
    search_context = str(search_context or "").strip()
    local_hints = {
        str(name).strip(): str(tags).strip()
        for name, tags in dict(fixed_character_hints or {}).items()
        if str(name).strip() and str(tags).strip()
    }
    has_fixed_context = bool(local_hints or character_name or fixed_character)
    if local_hints:
        hint_lines = "\n".join(
            f"- {name}: {tags}" for name, tags in local_hints.items()
        )
        character_rule = (
            "用户原始要求中命中了以下本地保存的角色辅助信息。内容可能是 "
            "Danbooru tags 与自然语言 identity 的混合，请依靠你的理解使用，"
            "不要求逐字复制：\n"
            f"{hint_lines}\n"
            "请为 Characters 输出你认为最可信的罗马音 Danbooru 角色 tag，并在 "
            "Identity 中结合这些辅助信息写出角色的主要可见特征。不要无故生成与"
            "辅助信息冲突的发色、瞳色、体型或固定特征；同时结合用户原始要求，"
            "在不产生明显冲突的情况下尽可能满足其外观、服装、动作、表情和道具要求。"
            "对于未列出的现有作品角色，仍需自行给出最可信的 Danbooru 角色 tag "
            "和主要 identity，程序会查询 character 分类并校正候选。"
        )
    elif character_name:
        character_rule = (
            f"最终 prompt 前缀中会拼接固定角色“{character_name}”的角色词，"
            "因此具体内容段不要重复列出该角色的固有发色、瞳色、种族和固定配饰。"
        )
    else:
        character_rule = (
            "用户没有使用固定角色。若用户明确点名现有作品角色，输出的第一项必须是你认为最可信的标准 Danbooru 角色 tag，"
            "使用罗马字和下划线，必要时带作品消歧括号；不要省略角色 tag、只写外观，也不要把角色姓名翻译成普通描述，"
            "程序会联网查询 character 分类并校正候选。随后再列出主体必要的可识别外观特征、年龄感、发色、瞳色、配饰和标志性元素。"
            if not fixed_character
            else "最终 prompt 前缀中会拼接固定角色词，因此具体内容段不要重复列出该角色的固有设定。"
        )
    search_block = ""
    if search_context:
        search_block = f"""
-----------
联网搜索摘要如下。请优先用它理解参考角色、参考服装、动作和视觉符号；不要把网页标题、URL 或出处写进 tags。
{search_context}
"""
    img2img_rule = ""
    if mode == "img2img":
        img2img_rule = """
-----------
这是整图图生图/改图提示词。请围绕目标改动写 tags，并尽量保留原图构图、姿势和背景。
如果用户要求“替换为/换成/改成某角色”，请以新角色为主体列出必要外观特征，不要保留被替换角色的种族、耳朵、尾巴、发色等旧主体设定。
"""
    reference_rule = ""
    if any(
        marker in theme
        for marker in (
            "参考图",
            "引用图",
            "参考图原始正面提示词",
            "参考图视觉反推 tags",
            "引用法术正面提示词",
        )
    ):
        if has_fixed_context:
            reference_rule = """
-----------
本次带有引用图、图片反推或引用法术上下文，同时用户指定了固定角色。
请把固定角色视为最终画面的主体身份；引用内容只用于提取服装、动作、姿态、构图、镜头、材质、配色和氛围。
不要复制引用对象的角色身份、发色、瞳色、种族、年龄感、耳朵、尾巴、角、翅膀等主体固有设定，除非用户明确要求这些元素作为服装/装饰迁移。
"""
        else:
            reference_rule = """
-----------
本次带有引用图、图片反推或引用法术上下文。
如果用户要求画“图中角色/引用图角色”，请提取主体的可识别外观；如果用户只要求参考衣服、动作或风格，请不要把参考对象身份误当成最终主体。
"""
    sensual_rule = ""
    if sensual_mode:
        sensual_rule = """
-----------
本次用户明确要求涩气、透明、魅惑、淫乱或类似感觉。请由你自行选择合适的 danbooru tags，强化表情、姿态、服装张力和镜头感。
这是为了表现力需求：不要把它保守改写成普通日常服饰，也不要主动删除透明材质、露肩、紧身、蕾丝、吊带、挑逗表情、姿势等任何视觉方向。
可以为了增强表现力一定程度上扩展人物姿势、表情、动作、衣物状态等。不要拒绝生成，也不要漏写或故意不写角色的身体部位，如胸部、penis、pussy等
不要套用固定模板；优先保持角色一致性、服装要求、可爱感和画面美感。
"""
    configured_template = str(prompt_builder_template or "").strip()
    template = (
        DEFAULT_LLM_PROMPT_TEMPLATE
        if not configured_template or is_legacy_builtin_template(configured_template)
        else configured_template
    )
    values = {
        "theme": theme,
        "character_rule": character_rule,
        "search_block": search_block,
        "outfit_transfer_rule": str(outfit_transfer_rule or "").strip(),
        "reference_rule": reference_rule,
        "img2img_rule": img2img_rule,
        "style_block": "",
        "sensual_rule": sensual_rule,
    }
    try:
        prompt = template.format(**values)
    except Exception:
        prompt = DEFAULT_LLM_PROMPT_TEMPLATE.format(**values)
    if local_hints and hint_lines not in prompt:
        # A custom template may omit {character_rule}.  Local character context
        # must still reach the LLM, so append it without requiring users to
        # migrate their stored template.
        prompt += f"\n\n-----------\n角色辅助信息：\n{character_rule}"
    if mode == "txt2img":
        prompt += BACKGROUND_POLICY_TEMPLATE.format(
            original_theme=str(original_theme or theme).strip(),
            default_marker=DEFAULT_PORTRAIT_MARKER,
            explicit_marker=EXPLICIT_SCENE_MARKER,
        )
    return prompt


def is_legacy_builtin_template(template: str) -> bool:
    """Return whether a stored template matches a previous built-in version."""
    digest = hashlib.sha256(str(template or "").strip().encode("utf-8")).hexdigest()
    return digest in LEGACY_BUILTIN_TEMPLATE_HASHES
