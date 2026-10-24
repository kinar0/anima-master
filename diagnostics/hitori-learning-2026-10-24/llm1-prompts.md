# LLM1 提示词核对

按当前源码和当前保存配置重建；不是当时控制台请求原文。历史任务保存的用户输入与 LLM1 输出一并列出。未修改业务提示词或配置。

## 当前实际选择的 system

```text
Extract only what the user said about each visible character. Copy phrases exactly, use null for unknown clothing, never guess tags or lore, and return JSON only.
```

当前保存值覆盖内置默认值；没有触发旧模板迁移。

## 内置默认 system（供对照）

```text
Extract only what the user said about each visible character. Copy phrases exactly, preserve the user's complete visible-character roster without merging or omitting anyone, use null for unknown clothing, never guess tags or lore, and return JSON only.
```

## 2026-10-24T20:22:02.355160 / 89a2f68d9dd142eb829c671f754d3157

重建的完整 user prompt：

```text
List each visible character once. Preserve the complete visible-character roster. Do not merge, summarize, sample, or omit characters. A character used only as an appearance, cosplay or clothing reference is not visible. Copy all text values exactly from the request; never translate, guess tags, or fill missing clothes.

Return only:
{"characters":[{"name":"exact visible entity mention","aliases":[],"clothing":null,"clothing_source":null,"clothing_changes":[],"appearance_changes":[]}]}

name = only the exact entity mention for the visible character, never a whole descriptive clause. aliases = exact alternate names explicitly paired with that same entity in A (B), otherwise []. For example, 复仇者（revenant） uses name=复仇者 and aliases=[revenant], while 巨乳（huge breasts）的粉发美少女 uses name=粉发美少女 and aliases=[]; appearance, clothing, pose, action, and translation glosses are not character aliases.
clothing = the exact stated garment/set phrase, or null when unstated.
clothing_source = the exact character/persona whose outfit is copied, or null. For A cosplay C or A wearing C's clothes (including A穿C衣服 without 的), A is the wearer and C is clothing_source; clothing still copies the full original garment/source phrase. A clothing donor is not a visible instance unless independently present in the scene. A reference to someone's hair, eyes, body feature or other appearance is NOT clothing: keep clothing and clothing_source null unless clothes are independently requested. Never put a hair/appearance phrase into clothing.
clothing_changes = explicit changes only. Every item MUST be an object with operation, slots, color, source_text; source_text must be exact. operation MUST be one of remove, damage, modify, replace_color, recolor_all, add, keep_only. slots MUST use only upper_body.primary, lower_body.skirt, one_piece.dress, outerwear, headwear, face_accessory.mask, handwear, legwear, footwear, lower_body.all, misc. If an item does not clearly fit a named slot, ALWAYS use misc instead of inventing another slot. Use remove (never removed/removal) for an item the character no longer wears; use damage for torn/ripped clothing, not remove. Torn or damaged clothing is a change, not removal. Use [] only when there is no explicit clothing change. Use modify for local tailoring, cutouts or reshaping of a garment; preserve the other garments and copy the exact alteration in source_text. keep_only means the character wears ONLY the listed whole garment slots and removes ALL other clothing. It MUST NOT represent retaining only part of one garment: '裙子只留下侧面的部分' is modify lower_body.skirt, not keep_only, and must not remove the shirt, gloves or legwear. Explicit accessory absence such as 没有面具/不戴帽子 belongs in remove for that slot, including when it is parenthesized inside the clothing phrase.
appearance_changes = explicit request phrases only. Prefer objects with dimension (eye_color, hair_color, hair_length, hair_style, skin_color, chest_size, animal_ears, tail, horns, age_presentation, or gender_presentation), operation (replace, additive, or omit), and exact source_text, plus reference_character (the exact name of the person whose feature is copied, otherwise null); do not output tags. Copying B's hairstyle/eye color/horns onto A changes only the stated dimension of A. B is an appearance reference, not an outfit donor or an additional visible person unless independently present. Clothing and appearance references may differ in the same request. Do not infer unstated source features or accessories. Use omit when the user says a visible feature must not be drawn or must be completely hidden (for example, no visible eyes); this suppresses that dimension's saved profile. Do not use omit for missing highlights, a dull gaze, closed eyes, or a changed eye color. operation=replace means the new value supersedes the old value; operation=additive means both remain visibly present. A trait immediately before a name belongs only to that person (扶她千早爱音 means only 千早爱音 is futanari); futanari is not chest_size. Shared words such as both/all/双方/两人/都 must be copied onto every affected character.

User request: 千早爱音和后藤独并排站在一起比胸部大小，只穿了内衣
```

历史任务保存的实际 LLM1 输出：

```json
{
  "characters": [
    {
      "name": "千早爱音",
      "aliases": [],
      "clothing": "内衣",
      "clothing_source": null,
      "clothing_changes": [],
      "appearance_changes": []
    },
    {
      "name": "后藤独",
      "aliases": [],
      "clothing": "内衣",
      "clothing_source": null,
      "clothing_changes": [],
      "appearance_changes": []
    }
  ]
}
```

## 2026-10-24T20:24:49.988981 / d7c81ee4cbc04d4d9ee6f18ca7df2f61

重建的完整 user prompt：

```text
List each visible character once. Preserve the complete visible-character roster. Do not merge, summarize, sample, or omit characters. A character used only as an appearance, cosplay or clothing reference is not visible. Copy all text values exactly from the request; never translate, guess tags, or fill missing clothes.

Return only:
{"characters":[{"name":"exact visible entity mention","aliases":[],"clothing":null,"clothing_source":null,"clothing_changes":[],"appearance_changes":[]}]}

name = only the exact entity mention for the visible character, never a whole descriptive clause. aliases = exact alternate names explicitly paired with that same entity in A (B), otherwise []. For example, 复仇者（revenant） uses name=复仇者 and aliases=[revenant], while 巨乳（huge breasts）的粉发美少女 uses name=粉发美少女 and aliases=[]; appearance, clothing, pose, action, and translation glosses are not character aliases.
clothing = the exact stated garment/set phrase, or null when unstated.
clothing_source = the exact character/persona whose outfit is copied, or null. For A cosplay C or A wearing C's clothes (including A穿C衣服 without 的), A is the wearer and C is clothing_source; clothing still copies the full original garment/source phrase. A clothing donor is not a visible instance unless independently present in the scene. A reference to someone's hair, eyes, body feature or other appearance is NOT clothing: keep clothing and clothing_source null unless clothes are independently requested. Never put a hair/appearance phrase into clothing.
clothing_changes = explicit changes only. Every item MUST be an object with operation, slots, color, source_text; source_text must be exact. operation MUST be one of remove, damage, modify, replace_color, recolor_all, add, keep_only. slots MUST use only upper_body.primary, lower_body.skirt, one_piece.dress, outerwear, headwear, face_accessory.mask, handwear, legwear, footwear, lower_body.all, misc. If an item does not clearly fit a named slot, ALWAYS use misc instead of inventing another slot. Use remove (never removed/removal) for an item the character no longer wears; use damage for torn/ripped clothing, not remove. Torn or damaged clothing is a change, not removal. Use [] only when there is no explicit clothing change. Use modify for local tailoring, cutouts or reshaping of a garment; preserve the other garments and copy the exact alteration in source_text. keep_only means the character wears ONLY the listed whole garment slots and removes ALL other clothing. It MUST NOT represent retaining only part of one garment: '裙子只留下侧面的部分' is modify lower_body.skirt, not keep_only, and must not remove the shirt, gloves or legwear. Explicit accessory absence such as 没有面具/不戴帽子 belongs in remove for that slot, including when it is parenthesized inside the clothing phrase.
appearance_changes = explicit request phrases only. Prefer objects with dimension (eye_color, hair_color, hair_length, hair_style, skin_color, chest_size, animal_ears, tail, horns, age_presentation, or gender_presentation), operation (replace, additive, or omit), and exact source_text, plus reference_character (the exact name of the person whose feature is copied, otherwise null); do not output tags. Copying B's hairstyle/eye color/horns onto A changes only the stated dimension of A. B is an appearance reference, not an outfit donor or an additional visible person unless independently present. Clothing and appearance references may differ in the same request. Do not infer unstated source features or accessories. Use omit when the user says a visible feature must not be drawn or must be completely hidden (for example, no visible eyes); this suppresses that dimension's saved profile. Do not use omit for missing highlights, a dull gaze, closed eyes, or a changed eye color. operation=replace means the new value supersedes the old value; operation=additive means both remain visibly present. A trait immediately before a name belongs only to that person (扶她千早爱音 means only 千早爱音 is futanari); futanari is not chest_size. Shared words such as both/all/双方/两人/都 must be copied onto every affected character.

User request: 画一个后藤独
```

历史任务保存的实际 LLM1 输出：

```json
{
  "characters": [
    {
      "name": "后藤独",
      "aliases": [],
      "clothing": null,
      "clothing_source": null,
      "clothing_changes": [],
      "appearance_changes": []
    }
  ]
}
```

## 与本次学习失败的关系

LLM1 明确禁止翻译和猜 tags，只复制原文实体与服装/外貌意图。两条历史输出均完成角色识别；双人请求也逐人记录了内衣。中文角色转 canonical 候选由独立 character_candidates 调用承担，不能把该调用失败称为 LLM1 漏人。候选调用的 source_name 校验仅要求名字出现在整段请求，没有严格核对当前 Target name；这是需单独审查的边界，但现有历史证据不足以认定它导致了本次空候选。
