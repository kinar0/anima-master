# 故障排查

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

新 LLM1 原始结果应是 `characters[].name / aliases / clothing / clothing_source / clothing_changes / appearance_changes`，不应再看到 `wardrobe.kind` 指令。`name` 应是原文中的角色实体短语；`aliases` 只列与它明确构成同一 `A（B）` 对的别名。`semantic_plan_attempt_count=2` 只用于 JSON、原文证据或基本字段外形损坏；修复失败时仍保留初稿中可解析的部分。旧日志里的 `wardrobe.kind/source/anchor_id` 是兼容格式，不应据此修改当前提示词。

新协议不再要求 LLM1 输出 lookup role 或候选 tag；它只保留 source 原文，由主机匹配命名衣柜。若调试日志仍出现带 `lookups` 的旧兼容响应，role 写反只在“唯一 lookup 且所有引用 wardrobe kind 一致”时安全归一化。命中衣柜后应看到 `complete_named_profile=true` 和完整 `effective_tags`。最终 `Nltags` 出现中文也不是 LLM2 的正常结果：resolver 的 `missing_descriptions` 只能进入 LLM2 上下文与摘要，不能由主机直接拼接到最终文本。

真实模型回归位于 `tests/live/test_deepseek_prompt_e2e.py`。普通 `pytest` 会明确跳过，避免无意产生模型费用；在 AstrBot 当前 provider 和密钥可用时，用 `ANIMA_RUN_LIVE_LLM_E2E=1` 运行。它不伪造 LLM1/LLM2 输出，直接经插件 `PromptPipeline` 调用当前 DeepSeek provider，并验收最终 `final_prompt`。该组属于“真实模型提示词管线 e2e”，仍不提交 ComfyUI 生图，不能冒充包含工作流执行和图片下载的全系统 e2e。

若同一个触发别名反复生成角色视觉档案，对照由 `semantic_plan_raw` 的 `name` / `clothing_source` 转换出的内部 target/source anchors 与已有档案的 `sourceTags`。最长边界别名命中后，实际语义查询应使用已有 `sourceTags`，新鲜档案的当前请求组件也应直接来自保存的 `tags`；日志出现 `refusing duplicate outfit profile alias` 表示本地查询仍返回了冲突来源，但写入已被拒绝。旧版本已经形成的冲突条目不会被自动删除，以免误删人工维护档案，应在服装词库页面确认正确 canonical source 后手工合并或删除。

## Identity 出现错误发色、瞳色或来源角色外貌

例如爱音视觉档案是 `grey_eyes`，用户只要求 Teto 风格的粉色双钻头发型，但最终 Identity 出现 `yellow eyes`。这不是服装 tag 冲突，应分别检查三层证据：

1. `danbooru_semantic_character_appearance_profiles` 或 `semantic_character_outfits[].appearance_tags` 是否仍是目标角色自己的稳定外貌，而不是 cosplay 来源角色的外貌。
2. 给第二次 LLM 的完整提示中，`Character-scoped stable appearance authority` 是否按穿着者列出档案；命中的固定角色辅助提示是否已经被最新档案重建，不再携带陈旧瞳色。
3. 最终 `structured_identity_blocks` 是否只保留用户明确修改的外貌维度，并恢复未提及维度的档案值；同时检查 `structured_detail_blocks`、共享 `Tags` 和最终 prompt，确认它们没有重新带入同一未开放维度的竞争特征。

当前外貌裁决单位是“角色 × 维度”，不是整份 Identity。只改发型不会解锁瞳色，只改 A 的瞳色不会影响 B。检查 LLM1 对每条修改给出的 `operation`：`replace` 应排除该维度旧值，`additive` 应保留兼容旧值并加入新值；主机不再靠固定中文关键词决定两者。用户没有提瞳色时，LLM 写出的黄眼、蓝眼等冲突描述必须被删除。writer 已正确表达的稳定外貌应保留自然语言，主机只补缺失维度，不应在最终提示词末尾再次盲目拼入整份档案。

若 LLM2 输出 `blonde and blue hair`、`yellow and green eyes` 这类并列描述，还要确认拆分后的孤立颜色词没有残留。程序会让孤立颜色继承右侧 `hair` / `eyes` 维度后再应用同一权限，不能把 `blonde` 或 `yellow` 当作不受约束的普通特征。

若括号英文翻译被误识别成额外角色，同时检查 `semantic_plan_raw.characters[].name` 与 `aliases`。角色实体和别名必须在原文中精确构成 `name（alias）` 或 `alias（name）`；仅仅出现在同一句、作为 `name` 的子串或位于另一项特征旁边均不得升级。匿名描述如“有着巨乳（huge breasts）的粉发美少女”应得到 `name=粉发美少女, aliases=[]`。新角色仍不需要本地预先确认，只要这个语义绑定与原文相邻关系都成立，就会继续进入 resolver 与学习流程。
