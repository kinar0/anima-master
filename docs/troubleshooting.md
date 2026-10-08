# 故障排查

## 漫画台词没有出现在交付图中

先检查 `comic_dialogue_enabled` 是否开启。该开关识别 `漫画对话：`、`漫画台词：`、`内心OS：`、`旁白：` 等显式标记及其后续 `角色：台词`，也识别 `角色说：“台词”` 或 `角色：“台词”` 等带说话人的引号。仅在剧情中写“表白”“无对话”或没有说话人的引号不会触发。任务记录的 `comic_dialogue.line_count` 表示已抽取的台词条数；发送成功后的 `delivery.outputs` 指向排字版 PNG。原图保留在 ComfyUI 输出目录。短台词优先排在生成画面内的气泡，空间不足时放在图片上方的对白区。

若显式标记存在却无可恢复台词，任务在提交 ComfyUI 前返回 `comic_dialogue_parse_failed`；若生成后缺中文字体、图片不合要求或排字失败，返回 `comic_dialogue_render_failed`，不发送无字原图。开关关闭时依旧按普通七字段提示词流程，模型可能把对白概括为表情或气泡。

## NAI 自定义工作流

`-r` 模式先生成完整七段提示词，再由布局模型返回全局提示词与各角色实例的
独立提示词、坐标和位置理由。V5 最多 22 个实例，V4/V4.5 最多 6 个；布局输出
上限按模型角色上限扩展，最高 7200 tokens。若摘要显示
`nai_character_plan_failed`，检查 `llm_error`：`nai_character_plan_invalid_json`
通常表示返回的 JSON 不完整或格式不合法；复杂多人请求可缩短逐人描述，
再检查模型的可见输出。失败时不会提交未经拆分的原提示词。
`nai_dropped_interaction_tags` 非空表示布局模型给某个角色的
`interaction_tags` 写了空动作、多个 `#`、逗号串、非英文 tag，或把方向标签写进了普通角色描述而未放进 `interaction_tags` 数组。
应让模型只为明确互动的角色各写一项 `source#tag`、`target#tag` 或
`mutual#tag`；无明确方向时使用空列表。布局模型根据完整原文判断互动，
不再由固定动作词组门控。主机只校验格式，若标签格式正确但互动对象或方向不对，
需检查 `nai_character_plan_raw` 与原文；`nai_dropped_interaction_tags` 不会记录这种语义错误。

若一名角色的性别、衣服、动作、表情、外貌或视线影响了其他角色，检查任务摘要中的
`nai_global_prompt` 和 `nai_dropped_global_character_tags`。视线 tag 应只出现在
对应的 `nai_characters[].prompt`；其他明确角色实例条件也应进入该角色框。解析器会从全局提示词删除可明确识别的误放项，并把原始项
列入 `nai_dropped_global_character_tags`。若删除列表非空但目标角色框里也没有该
条件，说明布局模型没有完成重新分配，应对照调试日志中的位置规划任务书和原始回复。
不要用该列表判断所有重复词都必须删除：官方 base prompt 本来就负责场景，
`bed`、`table`、`classroom` 等共享环境物件可以保留；角色框再用
`lying on bed`、`sitting on table` 表达局部关系。

若 `-r` 的全局提示词没有画师串，先确认插件已重载到包含画师恢复逻辑的版本，
再比较开启 `debug_prompt_enabled` 后的 `nai_full_prompt_before_split` 与
`nai_global_prompt`。当前代码会按启用的画师组或本次 `-sN` 选择补回画师项，
并将同名畸形副本替换为配置值。配置中的 `(yd_(orange_maru):1.1)` 应在
最终提示词中保持原样；若只有布局模型原始返回缺画师，而最终提示词保留它，
则属正常恢复。

- `custom_workflow_requires_api_export`：读取了可视化编辑格式。当前 NAI 接入
  应填写 `nai_api.json`；`nai.json` 留作 ComfyUI 编辑参考。
- `nai_prompt_link_not_supported`：NAI 正负文本经过了尚未支持的节点、缺少
  文本输入或存在循环。当前支持直接字符串、Textbox 和 ComfyUIToNovelAIV4。
- `custom_workflow_prompt_nodes_ambiguous`：正负提示词共用同一个可写输入，
  需在 ComfyUI 中分开后重新导出。
- `nai_size_requires_multiple_of_64`：明确尺寸不满足 NAI 的 64 倍数要求。
- `nai_sampling_parameters_out_of_range`：开启覆盖后，步数不在 1–50 或 CFG
  不在 0–30。可关闭参数覆盖，使用 NAI 工作流本身的设置。

账户认证、额度不足和远端生成失败需结合 ComfyUI 的 NAI 扩展日志检查。
AstrBot 的 ComfyUI 地址仍填写 ComfyUI 服务，不填写 NovelAI 网站地址。

## 会话打码

当前 `autofilter2.json` 将 `GrowMask` 的遮罩扩张值设为 `15`；若打码边缘
范围与旧版本不同，先检查工作流文件是否为当前版本。打码失败时插件不会
把未打码原图发回聊天，具体节点错误应从 ComfyUI 执行记录排查。

若提示 `UltralyticsDetectorProvider` 找不到 `bbox/censor_detect_v1.0_s.pt`，
到 [模型发布目录](https://huggingface.co/deepghs/anime_censor_detection/tree/main/censor_detect_v1.0_s)
下载 `model.pt`，改名为 `censor_detect_v1.0_s.pt`，并确认它位于
`ComfyUI/models/ultralytics/bbox/`，而不是插件目录下。

## `/anm` 没反应

先确认 AstrBot 是否收到消息。

如果 AstrBot 日志没有新消息，通常是聊天平台适配器、NapCat 或 OneBot 连接问题。

如果 AstrBot 收到消息但插件没有响应，请检查：

- 插件是否启用。
- 用户是否被权限配置拦截。
- 指令是否写成 `/anm`、`/anima` 或 `/comfyui`。

## ComfyUI 离线

确认：

- ComfyUI 正在运行。
- `comfyui_base_url` 是 AstrBot 能访问到的地址。
- 跨机器部署时防火墙没有阻断端口。

可以在聊天中发送：

```text
/anm 状态
```

## 图片没有发回聊天

确认：

- `send_result_to_chat = true`
- ComfyUI 生成了图片。
- 平台适配器允许发送图片。

如果日志显示生成成功但发送失败，优先检查 OneBot / 平台适配器。

## 参考图拿不到

请直接带图发送，或引用一条包含图片的消息后再使用 `/anm`。

如果提示“检测到了图片消息，但没有成功下载/保存参考图”，说明插件看到了图片消息，但平台没有返回可用图片数据。可以稍后重试，或改用引用图片消息。

## 法术解析读不到提示词

只有部分图片会保留生成信息。

如果图片经过 QQ、微信、网页或截图工具转码，生成信息可能已经丢失。PNG 原图更容易被解析。

## 联网搜索没有生效

联网搜索需要 AstrBot 全局 Tavily key。

搜索失败会自动降级，不会中断生图。

## 生图提示“LLM 返回为空”/“empty_llm_response”

提示词构建 LLM 返回空内容时，插件**不会**再把你的中文原文直接丢给 ComfyUI，
而是先做一次“关闭深度思考”的降级重试；若仍为空，则中止本次生成并返回
`empty_llm_response`。这是为了防止中文原文被当作 Danbooru 标签导致生图失败。

常见成因：

- **深度思考占满 token 预算**：开启深度思考时，思考模型可能把全部
  `prompt_builder_max_tokens`（默认 1000）消耗在思考链上，没有留下可见输出。
  插件现在会自动关闭深度思考重试一次。
- **思考链卡在 `Count` 决策上**：实测日志里思考模型会反复核算
  `Count` 与 `Characters` 是否一致（例如“如果我们写 Characters: ... 就重复了”），
  在输出阶段前耗尽 token 而被截断。为此内置模板把 `Count` 改成“一步查表”
  决策（`Count` 人数直接等于 `Characters` 项数，扶她计入女生人数），并禁止
  反复核算；系统提示词同步强化了这一规则。
- **回复内容不在 `completion_text` 字段**：部分提供商会把可见回答放在
  `reasoning_content`、`content`、`messages` 等字段。插件已兼容这些常见字段。
- **上游服务波动**：LLM 服务超时、限流或异常返回空串。可稍后重试。

处理建议：

- 适当调大 `prompt_builder_max_tokens`（如 1500-2000），给深度思考留出输出余量。
- 检查所选模型是否支持深度思考；不支持时建议关闭
  `prompt_builder_deep_thinking_enabled`。
- 即使 LLM 输出的 `Count` 与 `Characters` 不一致，单人结构化路径也会用
  角色清单的长度做确定性兜底，把错误人数 tag 替换为 `Npeople` 后继续，
  不会再把错误的 `2girls, futanari` 折叠成缺少人数锚点的 `futa with female`。
- 混合性别 Count 按各性别数量相加核对；`1girl, 1boy` 对两人 roster 是合法值，
  不会仅因任一单项不是 `2` 而被误修复。
- 若问题持续，把日志中 `prompt builder LLM returned empty content` 附近的内容
  提供给作者排查。

## 生图提示 `invalid_structured_prompt`

这表示 LLM2 的七字段外壳或人数锚点不完整，不是 ComfyUI 工作流的
`node_errors`。当前硬校验检查七个命名字段是否齐全、同名字段有无冲突值，以及 `Count`
是否包含有效人数 tag；完全相同的重复字段会折叠成一份。七个字段是 `Count`、`Characters`、`Copyright`、
`Identity`、`Details`、`Tags` 和 `Nltags`。
Identity/Details 不会再因为角色名没有位于分句开头而令整份回复失败：照片分镜
中的唯一角色名可用于归属，无角色名的后续分号句继承上一明确角色，仍无法归属
的文字会保留到 Nltags。
`Characters` 留空也不等于 `no humans`：当画面只有匿名、裁切或被景深遮蔽的
人物时，只要 `Count` 有有效人数锚点，七字段回复仍然合法；其可见特征可以写在
`Tags` 和 `Nltags`。只有真正没有人物时才使用 `no humans`。

结构确实不完整时会带着具体错误和第一次回复重试一次。开启
`debug_prompt_enabled` 后检查：

- `prompt_llm_initial_content`：第一次 LLM2 回复；
- `structured_initial_validation_errors`：初次结构错误；
- `prompt_llm_retry_content`：格式重试回复；
- `structured_retry_validation_errors`：重试结构错误；
- `prompt_llm_accepted_attempt`：最终采用 `initial` 还是 `retry`。

这样查看 final prompt 时，不会再把被拒绝的初稿误认为它的直接输入。

## 明确指定新服装，却混入角色默认衣柜

例如“爱音和祥子都穿婚纱”最终仍出现 `haneoka school uniform`，不要只检查第二次 LLM 的输出。当前流程应先由第一次 LLM 为每名角色建立服装计划，再由 `WardrobeAuthority` 把未选中的角色缓存衣柜标为陈旧，并在所有后续通道中移除。

先区分 default 是否本来就该被读取：

- “爱音穿连体式泳装，祥子穿分体式泳装”：两人都应为 `creative_fallback/resolved`，默认衣柜都应进入 stale，不能出现在角色有效服装中。
- “爱音穿常服，祥子站在旁边”：爱音使用明确 casual；祥子没有基础服装，应为 `default_reference/unspecified`，默认衣柜可作为祥子的软参考。
- “爱音的裙子变蓝色”：应为 `default_reference/unspecified` 并带换色 patch；default 是修改基线，不是必须原样恢复的套装。
- “爱音穿羽丘夏季校服但没穿短裙”：应使用命名套组及删除 patch，不应再读取爱音 default。

若构图属于头部/脸部、上半身、cowboy shot、肚脐、下半身、腿/大腿/臀部、足部、手部或头部出框，`composition_omitted_tags` 应列出该视域下被省略的档案 tag；“上半身肖像”“只拍到腰部以上”等表达也应在 LLM2 前产生省略项。具体省略 tag 不应再出现在 LLM2 的 `Per-character clothing evidence` 中；最终 `Details` 若仍出现 `green pleated skirt` 一类带修饰词的画外服装，说明后置 prose 权限过滤失效。若原文只是“下半身没穿”等服装修改，则不应产生构图省略项；出现时说明 framing 误判。

开启 `debug_prompt_enabled` 后检查最近任务摘要：

1. `semantic_character_outfits` 应同时包含两名角色的婚纱计划；“双方、两人、都、both、all”等共享范围必须展开到每个目标角色。
2. `explicit_wardrobe_evidence` 应为 `true`，`wardrobe_source` 不应是 `configured_default`。
3. `wardrobe_authority.selected_tags` 应包含本次婚纱标签；`stale_cached_tags` 应包含未选中的校服缓存。
4. 已有逐角色计划时，`outfit_summary_source` 应为 `suppressed_by_character_authority`。
5. 给第二次 LLM 的任务书、七段输出、`required_profile_tags`、`required_core_tags` 和最终 prompt 都不应再出现陈旧校服。

本地 Danbooru 查询不可用时，第一次 LLM 仍应运行；只会跳过本地证据匹配/验证。LLM1 本身不应输出候选 tag。如果更新代码后摘要仍只有 `danbooru_semantic_status=profile_cache`、且 `semantic_character_outfits=[]`，优先确认插件是否已经重载到新版本。

注意两个显式绕过入口：raw / 无优化模式不经过服装权限流程；`#` 后的手工尾缀会原样插回，也不会被 `WardrobeAuthority` 清理。若冲突 tag 来自这两处，需由用户自行删除。

对于“角色 A cosplay 角色 B”，检查 `wardrobe_resolution_states`、`semantic_character_outfits` 和 `source_grounding_tags`。LLM1 正常时会在 A 的 `clothing_source` 中原样返回 B；即使 LLM1 给了错误 wardrobe 意图，主机也会对明确的“A cosplay B / A 穿 B 的衣服”逐角色修复为 `outfit_source`。普通文字请求不得出现旧换装块的“最终主体必须是固定角色”；只有参考图/搜索换装仍可进入旧路线。若句式本身无法确定穿着者，才保留 `explicit_but_unresolved`，且不得恢复目标默认衣柜。

新 LLM1 原始结果应是 `characters[].name / aliases / clothing / clothing_source / clothing_changes / appearance_changes`，不应再看到 `wardrobe.kind` 指令。`name` 应是原文中的角色实体短语；`aliases` 只列与它明确构成同一 `A（B）` 对的别名。宿主最多保留 22 名可见角色；原文明示“6 个角色 / 六人 / 4girls and 2boys”等人数而回复只有 4 项时，`semantic_plan_validation_errors` 应出现 `characters must contain exactly 6 visible characters; got 4`，随后 `semantic_plan_attempt_count=2`。若原始回复已有 6 项而解析后的 anchors/plans 只有 4 项，说明仍在运行带历史 `characters[:4]` 截断的旧代码，应重载插件。修复失败时仍保留初稿中可解析的部分。旧日志里的 `wardrobe.kind/source/anchor_id` 是兼容格式，不应据此修改当前提示词。

`clothing_changes` 的标准槽位是 `upper_body.primary / lower_body.skirt / one_piece.dress / outerwear / headwear / face_accessory.mask / handwear / legwear / footwear / lower_body.all / misc`。例如真实返回中的 `operation=removed, slots=[mask]` 与 `slots=[boots]` 会分别归一为 `remove face_accessory.mask` 和 `remove footwear`；未知但非空的槽位归入 `misc`。若操作、颜色或 `source_text` 仍无法校验，`semantic_plan_validation_errors` 必须出现具体错误并触发一次修复，不能只在最终计划中悄悄消失。检查给 LLM2 的 `Per-character clothing evidence` 是否保留 `Changes:` 和原始用户证据。

若最终出现一整段 `character wears <档案全部组件>`，或 `global_outfit_reinforcement_tags` 非空，说明仍运行着旧的自动注入代码。当前结构化路径必须让 `global_outfit_reinforcement_tags=[]`，并且 `controlled_character_outfit_detail` 只过滤 LLM2 的原始 Details，不再拼接 `effective_tags`。`effective_tags` 留在摘要中仅用于证明 LLM2 看到了什么证据、以及本地应过滤哪些冲突，不等于最终必须输出。

新协议不再要求 LLM1 输出 lookup role 或候选 tag；它只保留 source 原文，由主机匹配命名衣柜。若调试日志仍出现带 `lookups` 的旧兼容响应，role 写反只在“唯一 lookup 且所有引用 wardrobe kind 一致”时安全归一化。命中衣柜后应看到 `complete_named_profile=true` 和完整 `effective_tags`。最终 `Nltags` 出现中文也不是 LLM2 的正常结果：resolver 的 `missing_descriptions` 只能进入 LLM2 上下文与摘要，不能由主机直接拼接到最终文本。

真实模型回归位于 `tests/live/test_deepseek_prompt_e2e.py`。普通 `pytest` 会明确跳过，避免无意产生模型费用；在 AstrBot 当前 provider 和密钥可用时，用 `ANIMA_RUN_LIVE_LLM_E2E=1` 运行。它不伪造 LLM1/LLM2 输出，直接经插件 `PromptPipeline` 调用当前 DeepSeek provider，并验收最终 `final_prompt`。该组属于“真实模型提示词管线 e2e”，仍不提交 ComfyUI 生图，不能冒充包含工作流执行和图片下载的全系统 e2e。

若同一个触发别名反复生成角色视觉档案，对照由 `semantic_plan_raw` 的 `name` / `clothing_source` 转换出的内部 target/source anchors 与已有档案的 `sourceTags`。最长边界别名命中后，实际语义查询应使用已有 `sourceTags`，新鲜档案的当前请求组件也应直接来自保存的 `tags`；日志出现 `refusing duplicate outfit profile alias` 表示本地查询仍返回了冲突来源，但写入已被拒绝。旧版本已经形成的冲突条目不会被自动删除，以免误删人工维护档案，应在服装词库页面确认正确 canonical source 后手工合并或删除。

若“角色穿演出服/夏装/冬装”只得到泛化服装描述，检查 `requested_outfit_mode` 和 `semantic_character_outfits`。前者应分别是 `stage_profile / summer_profile / winter_profile`，后者应保持同一 wardrobe kind、`resolution_state=resolved`，并在 `effective_tags` 中出现该角色相应 qualifier 档案的组件。`danbooru_semantic_source_outfit_profiles` 中的 qualifier 也应一致。只有日志显示档案命中、但逐角色计划仍是 `creative_fallback`，说明运行的仍是旧版变体绑定流程；保存代码后重载插件再复现。“站在舞台上”和“演出结束后”不应产生 `stage_profile`。

## 衣柜命中但角色名变成拼音

2026-09-13 18:56 的“粥祥”记录命中本地档案，但 LLM2 的权威 roster 只有爱音与基础祥子，同时把粥祥列为 unresolved；最终 `zhouxiang` 保留在角色及叙述中，正确完整 tag 仅被额外追加。当前在本地证据合并后将精确确认的 owner 绑定回可见 anchor，摘要 `confirmed_character_bindings` 应包含“粥祥 → togawa_sakiko_(master_of_melodia)”。LLM2 收到同一映射，输出后校正唯一可确定的改名并核对缺失身份；无法唯一对应则重试，重试仍不满足时停止生成。检查 `structured_initial_validation_errors`、`structured_retry_validation_errors` 和 `character_resolution_statuses`；已修正角色应为 `semantic_confirmed`。这是按本地 owner 配置执行，不把任意服装 source 当作角色，也没有硬编码“粥祥”的拼音。

## Identity 出现错误发色、瞳色或来源角色外貌

### 明确“不画眼睛”却仍出现 `grey eyes`

先看 `semantic_plan_raw` 的对应角色 `appearance_changes` 和摘要 `appearance_omissions`。角色视觉档案中的 `grey_eyes` 是正常基线；用户明确要求眼睛不可见时，必须得到该角色的 `eye_color/omit`，而不是把“无眼睛”当成新的瞳色。旧 planner 对同一原文输出 `eye_color/replace` 或错误的 `face_accessory.mask` 时，主机会从可定位的明确缺席证据归一为 `omit`；错误维度还会触发一次修复请求。

`llm_prompt` 中被省略角色的稳定外貌列表不应再含其瞳色，最终结构化 `Identity`、`Details`、`Nltags` 也不得补回。NAI `-r` 还要检查 `nai_characters[].prompt`，因为位置规划模型会重新写每个人的提示词；同画面未省略眼睛的其他角色仍可保留自己的瞳色。`#` 手工尾缀和原样模式属于直通，不执行上述权限。

例如爱音视觉档案是 `grey_eyes`，用户只要求 Teto 风格的粉色双钻头发型，但最终 Identity 出现 `yellow eyes`。这不是服装 tag 冲突，应分别检查三层证据：

1. `danbooru_semantic_character_appearance_profiles` 或 `semantic_character_outfits[].appearance_tags` 是否仍是目标角色自己的稳定外貌，而不是 cosplay 来源角色的外貌。
2. 给第二次 LLM 的完整提示中，`Character-scoped stable appearance authority` 是否按穿着者列出档案；命中的固定角色辅助提示是否已经被最新档案重建，不再携带陈旧瞳色。
3. 最终 `structured_identity_blocks` 是否只保留用户明确修改的外貌维度，并恢复未提及维度的档案值；同时检查 `structured_detail_blocks`、共享 `Tags` 和最终 prompt，确认它们没有重新带入同一未开放维度的竞争特征。

当前外貌裁决单位是“角色 × 维度”，不是整份 Identity。只改发型不会解锁瞳色，只改 A 的瞳色不会影响 B。检查 LLM1 对每条修改给出的 `operation`：`replace` 应排除该维度旧值，`additive` 应保留兼容旧值并加入新值；主机不再靠固定中文关键词决定两者。用户没有提瞳色时，LLM 写出的黄眼、蓝眼等冲突描述必须被删除。writer 已正确表达的稳定外貌应保留自然语言，主机只补缺失维度，不应在最终提示词末尾再次盲目拼入整份档案。

若 LLM2 输出 `blonde and blue hair`、`yellow and green eyes` 这类并列描述，还要确认拆分后的孤立颜色词没有残留。程序会让孤立颜色继承右侧 `hair` / `eyes` 维度后再应用同一权限，不能把 `blonde` 或 `yellow` 当作不受约束的普通特征。

若括号英文翻译被误识别成额外角色，同时检查 `semantic_plan_raw.characters[].name` 与 `aliases`。角色实体和别名必须在原文中精确构成 `name（alias）` 或 `alias（name）`；仅仅出现在同一句、作为 `name` 的子串或位于另一项特征旁边均不得升级。匿名描述如“有着巨乳（huge breasts）的粉发美少女”应得到 `name=粉发美少女, aliases=[]`。新角色仍不需要本地预先确认，只要这个语义绑定与原文相邻关系都成立，就会继续进入 resolver 与学习流程。

## 女性角色被错误写成扶她

先检查 `semantic_plan_raw` 与角色视觉档案。如果两者只有正常发色、瞳色、胸部尺寸等信息，而 `prompt_llm_initial_content` 首次出现 `futanari / futa with female / penis / erection`，错误来自第二阶段模型，不是角色档案。当前默认任务书不会在普通请求中展示扶她示例；只有用户原文正向明确写出扶她时才动态加入计数规则。

结构化路径还会记录 `sexual_trait_authority`。其中 `futa_allowed=false` 时，LLM2 自行添加的扶她 Count、Identity、Details、Tags 和 Nltags 会被移除；`male_genitals_allowed=false` 时同样移除未经正向要求的男性生殖器内容。自慰、色情场景、体位、服装、贫乳或巨乳都不能单独开启这些权限，“不是扶她”“不要出现扶她”属于明确否定。用户在 `#` 后手工添加的原样 tags 不受此门控。

旧版本内置模板若已保存到配置，插件重载时会通过模板指纹迁移到当前默认模板。真正由用户改写过、且不匹配历史内置指纹的自定义模板不会自动覆盖；这类模板若仍常驻列举扶她和阴茎示例，应手工清空“主提示词模板”以恢复内置规则。


## 角色服装学习混入多人图，或只有 3 个聚类样本

检查档案 `evidence` 中的 `sample_mode / total_posts / selected_posts / focused_posts`。旧版 `casual_filtered` 允许缺少人物归属的整图 tags 参与统计，`single_character` 也可能只是最小角色标签组合，不能据旧模式名证明是单人图。页面中的“聚类样本”表示参与最终服装统计的帖子数，不表示不同服装套数，也不是视觉模型看过的图片数。

新采样会排除多人证据、分页补样并对重叠帖子去重。聚类少于 6 个样本时返回 `casual_evidence_insufficient` 等状态，不自动建档。日志 `outfit learning` 会列出 fetched、accepted、focused、rejected、queries 和 errors；网络失败与确实缺少帖子应结合 errors 区分。成功档案也保存样本标识和查询记录。完整边界见 [学习样本的归属与支持度](角色视觉档案与服装优先级.md#学习样本的归属与支持度)。

已有服装组件和带 `appearance_manual_override` 的外貌不会被自动刷新覆盖；新采样成功也不代表页面中的旧组件已经被替换。应先核对并修订旧档案，不能只通过重载插件期待错误外貌自动消失。


角色 Wiki 的“casual outfit”说明与帖子 `casual` 标签不是同一数据源。页面有常服说明和示例，不代表示例帖子带 `casual` 标签。当前自动学习仅统计帖子，尚未接入可稳定访问的 Wiki 资料渠道；HTTP 403 / Cloudflare 验证失败不能作为“角色没有常服资料”的结论。Wiki 获取能力未解决前，不应声称已完成基于该页面的自动学习。

通过词库页面成功保存（包括删除档案）会清理内存中的服装采样缓存，避免立即重试仍复用旧的成功结果或十分钟失败结果。角色标签查询缓存保留。只清空档案中的 tags 仍属于编辑现有档案，不等于删除该条档案并重建。


## 本地有 S:P 标签，但角色没有自动建档

2026-09-09 的三次记录中，前两次括号英文是 `S:P little night`，第三次已改为正确的 `S:P little knight`，但角色仍出现在 `danbooru_semantic_missing`。本地词典实际包含 `s:p_little_knight`；失败发生在采样之前：括号角色别名的旧字符规则不允许 `:`，因此即使 LLM1 正确返回 aliases，也会被主机丢弃。已有 I:P 档案可通过缓存角色匹配继续使用，不能据此判断新角色入口正常。

当前括号角色别名与普通候选使用一致的标点范围，保留 `:`、`!` 和作品消歧括号。别名候选直接绑定到原角色的 `target_1` 等 ID，服装计划继续引用同一个角色，不再另建一个英文角色项而留下未解析的中文项。明确括号配对与本地验证仍然必需；`little night` 这类拼写错误不会被硬编码成正确角色。

resolver 内部继续保存未转义 canonical tag `s:p_little_knight`，最终拼接到 Anima prompt 时才输出 `s\:p little knight`。作品消歧括号同样输出为 `\(` / `\)`；完整权重表达式（例如 `(watercolor:2)`）和 `#` 后用户手工尾缀不被破坏。若调试摘要的 `anchor_tags` 正确但 `final_prompt_head` 仍是未转义的 `s:p little knight`，说明运行实例尚未加载最终显示层修复。

验证时应同时看到 `anchor_tags` 中原角色 ID 对应 `s:p_little_knight`，以及后续服装采样调用。修复后的只读实测由 Safebooru 回退取得 46 条帖子，其中 27 条通过筛选；这证明本次可以进入采样，不等于这些统计组件已经经过逐图视觉或 Wiki 校验。接口可用性仍可能变化。


## Master of Melodia 没有学到，反而使用校服

2026-09-12 的三份任务记录分别出现：括号造型被拆成普通别名、完整来源被基础角色缓存覆盖、简称来源未解析。完整来源应保持 `togawa_sakiko_(master_of_melodia)`，不能变成 `togawa_sakiko`。修复后该 tag 在在线回退中确认为普通分类，作为独立服装来源采样；不会作为角色或作品 tag 注入。只读实测取得 56 个聚类样本，未改写运行中的衣橱。

重载插件后用“丰川祥子穿着togawa sakiko (master of melodia)的衣服，站立”复现。查看 `danbooru_semantic_outfit_sources` 是否是完整来源，以及 `semantic_character_outfits` 的组件是否来自新造型。旧的错误档案不会自动删除，手工服装组件仍受保护。页面现在支持自由服装名称、所属角色和学习来源；详细字段与限制见 [自由命名衣橱](角色与服装配置工作流.md#自由命名衣橱与完整造型标签2026-09-12)。
