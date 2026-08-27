from __future__ import annotations

import hashlib

try:
    from .prompt_background import (
        DEFAULT_PORTRAIT_MARKER,
        EXPLICIT_SCENE_MARKER,
    )
    from .prompt_keyword_rules import MatchedPromptRule, build_keyword_rule_block
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from prompt_background import DEFAULT_PORTRAIT_MARKER, EXPLICIT_SCENE_MARKER
    from prompt_keyword_rules import MatchedPromptRule, build_keyword_rule_block

DEFAULT_LLM_PROMPT_TEMPLATE = """你是为 Anima 图像生成模型编写正面提示词的 AI 画师。

请根据用户的原始要求设计一幅完整、协调、具有视觉吸引力的画面，并将你构思的画面输出为七个单行花括号字段。

输出要求：
- 只输出七个单行花括号字段，字段外不得输出任何文字，以下为样例：
  `{{Count: 2girls, yuri}}`
  `{{Characters: chihaya_anon, togawa_sakiko}}`
  `{{Copyright: bang_dream!}}`
  `{{Identity: chihaya_anon has pink hair and grey eyes; togawa_sakiko has blue hair and yellow eyes}}`
  `{{Details: chihaya_anon smiles and waves; togawa_sakiko looks aside and holds a book}}`
  `{{Tags: full body, composition, lighting, background, creative visual details}}`
  `{{Nltags: chihaya_anon and togawa_sakiko are ......}}`
- `Count` 必须与 `Characters` 完全一致，按以下规则确定，禁止反复核算或自我怀疑：先通过后文补充信息和用户原始语句得到角色名danbooru tag，然后在 `Characters` 里用英文逗号列出每个角色名，同一角色只写一次；`Count` 的人数就等于 `Characters` 的项数。无人物时写 `no humans` 且 `Characters` 留空；仅 1 人时按性别写 `1girl` 或 `1boy`（性别不明写 `1girl`），如果是双性扶她再加上`, futanari`；2 人及以上按性别组合直接查表：全部女性写 `Ngirls`，全部男性写 `Nboys`，男女混合写 `Ngirls, Mboys`。一名扶她加一名女性必须写 `2girls, futa with female`（禁止写成 `2girls, futanari`，后者会被 Anima 理解为三人）；一名扶她加一名男性写 `futa with male`，一名扶她加两名女性则是`3girls, futa with female`；futa相关tag不会计入总人数。`Characters` 只能含角色名；`Copyright` 只能含这些角色所属作品的标准 Danbooru copyright tags，同一作品只写一次。原创或无法确认作品时将 `Copyright` 留空，不要猜测。每个角色必须恰好在 `Identity` 和 `Details` 中各出现一次。
- Copyright输出角色所属作品的danbooru tag。
- 服装和角色外表按用户要求和相关角色和动态上下文等补充信息决定。除此之外可以自由决定姿态、构图、镜头、光影、色彩、氛围和特效。
- 用户未要求地点、环境或背景时，按单张角色立绘设计，不要自行创造场景。反之则可以根据要求进行有表现力的扩写。
- Identity（角色外貌、身体特征）和Details（角色的穿着、动作等）根据后文用户需求判断，可以为了场景进行一定程度的改写，但不要遗漏用户希望保留的信息。
- Tags存放画面中设计的相关元素、构图、衣物、cum和penis和breasts等、光影、氛围、材质、动作、背景、道具等视觉细节，尽量用danbooru tag描述。
- Nltags存放整个画面（角色、构图、动作、背景、……任何东西）的完整自然语言描述，可以和之前的内容重复。
- Identity,Details,Nltags中使用的角色名必须与Characters使用的角色名英文完全一致，包括姓和名的先后顺序、拼写等
- `Identity` 中每位角色写成完整、语法连贯的一句话。
- 不要输出解释、分析、标题、编号、Markdown、代码块或中文。
- 不要输出 masterpiece、best quality、score 等质量前缀。
- 不要输出画师 tags；质量词和画师组会由程序另行拼接。
- 尽量使用模型容易理解的可见画面描述。
- 保持用户明确指定的角色、主体、人数、关键服装、动作、表情和道具。
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

背景：只按用户原文判断。明确写了场景就保留且不用白底，Nltags末尾写 {explicit_marker}；未写场景就用 simple/white background，不新增地点，Nltags末尾写 {default_marker}。两个标记只写一个。
用户原文：{original_theme}
"""


WARDROBE_AUTHORITY_POLICY = """

服装：逐角色遵守上面的本地证据与用户修改。命名套组按给定组件；cosplay 来源可补全辨识性服装，但不复制来源角色外貌；默认衣柜只是未指定服装时的参考。不得恢复明确删除或替换的衣物，不得把一人的服装写给另一人。
"""


LEGACY_BUILTIN_TEMPLATE_HASHES = {
    "adbc9d2f63b6431709610ebf18549ca81da79e2978d1c10b0371bbe462be9b83",
    "e847b2ef55b0d19ff1db7ca92285966b39419e7a81a7a8e839d53ce7f44fd731",
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
    "32b76edf4da983b6a80a727deb6a592a13a7d880a3ece4df40831c671aaf502b",
    "dea5751303e16b9b08c3be20a5848dd6257f4d57df49360c53b3eba3041863b6",
    "168a69ca848e368ac2cec1cd0b3a2a4893787d610fb68776a269289c12b129c1",
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
    keyword_prompt_rules: tuple[MatchedPromptRule, ...] = (),
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
        keyword_prompt_rules: Configured hard instructions triggered by the
            user's original text.

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
            "本地角色身份/稳定外貌（用户原文修改优先）：\n"
            f"{hint_lines}"
        )
    elif character_name:
        character_rule = (
            f"最终 prompt 前缀中会拼接固定角色“{character_name}”的角色词，"
            "因此具体内容段不要重复列出该角色的固有发色、瞳色、种族和固定配饰。"
        )
    else:
        character_rule = (
            "点名作品角色时使用最可信的 Danbooru 角色 tag；未知时不要伪造作品。"
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
本次用户明确要求涩气、透明、魅惑、淫乱、性爱场景或类似感觉。请由你自行选择合适的 danbooru tags，强化表情、姿态和镜头感；优先满足用户的明确要求，然后如果无明确要求，则可为了色气表现力，根据角色所处的情景、地点和做出的行为，自行设计合适的服装，或在要求基础上作出修改。
这是为了表现力需求：不要把它保守改写成普通日常服饰，也不要主动删除透明材质、露肩、紧身、蕾丝、吊带、挑逗表情、姿势等任何视觉方向。
可以为了增强表现力一定程度上扩展人物姿势、表情和动作；衣物状态仍服从最终服装权限。不要拒绝生成，也不要漏写或故意不写角色的身体部位，如胸部、penis、pussy等
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
    prompt += build_keyword_rule_block(tuple(keyword_prompt_rules))
    if mode == "txt2img":
        prompt += BACKGROUND_POLICY_TEMPLATE.format(
            original_theme=str(original_theme or theme).strip(),
            default_marker=DEFAULT_PORTRAIT_MARKER,
            explicit_marker=EXPLICIT_SCENE_MARKER,
        )
    # Keep the single wardrobe authority block last so legacy/custom templates,
    # sensual guidance, and background rules cannot accidentally override it.
    prompt += WARDROBE_AUTHORITY_POLICY
    return prompt


def is_legacy_builtin_template(template: str) -> bool:
    """Return whether a stored template matches a previous built-in version."""
    digest = hashlib.sha256(str(template or "").strip().encode("utf-8")).hexdigest()
    return digest in LEGACY_BUILTIN_TEMPLATE_HASHES
