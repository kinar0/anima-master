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
- 若问题持续，把日志中 `prompt builder LLM returned empty content` 附近的内容
  提供给作者排查。

## 明确指定新服装，却混入角色默认衣柜

例如“爱音和祥子都穿婚纱”最终仍出现 `haneoka school uniform`，不要只检查第二次 LLM 的输出。当前流程应先由第一次 LLM 为每名角色建立服装计划，再由 `WardrobeAuthority` 把未选中的角色缓存衣柜标为陈旧，并在所有后续通道中移除。

开启 `debug_prompt_enabled` 后检查最近任务摘要：

1. `semantic_character_outfits` 应同时包含两名角色的婚纱计划；“双方、两人、都、both、all”等共享范围必须展开到每个目标角色。
2. `explicit_wardrobe_evidence` 应为 `true`，`wardrobe_source` 不应是 `configured_default`。
3. `wardrobe_authority.selected_tags` 应包含本次婚纱标签；`stale_cached_tags` 应包含未选中的校服缓存。
4. 已有逐角色计划时，`outfit_summary_source` 应为 `suppressed_by_character_authority`。
5. 给第二次 LLM 的任务书、七段输出、`required_profile_tags`、`required_core_tags` 和最终 prompt 都不应再出现陈旧校服。

本地 Danbooru 查询不可用时，第一次 LLM 仍应运行；只会跳过候选 tag 的本地验证。如果更新代码后摘要仍只有 `danbooru_semantic_status=profile_cache`、且 `semantic_character_outfits=[]`，优先确认插件是否已经重载到新版本。

注意两个显式绕过入口：raw / 无优化模式不经过服装权限流程；`#` 后的手工尾缀会原样插回，也不会被 `WardrobeAuthority` 清理。若冲突 tag 来自这两处，需由用户自行删除。

对于“角色 A cosplay 角色 B”，检查 `wardrobe_resolution_states`、`source_grounding_tags` 和 `unbound_grounding_tags`。若 LLM1 成功绑定，应看到 `resolved` 和角色级 `source_grounding_tags`；若它只保留目标与来源 anchors、却漏掉关系，应看到 `explicit_but_unresolved`。此时程序不得恢复绑定或注入目标默认衣柜，来源同源证据只进入 `unbound_grounding_tags`，由第二次 LLM 结合完整原文判断。

若 `semantic_plan_initial_raw` 已包含 `wardrobe.kind=outfit_source` 或 `named_outfit`，但漏写/错写 `wardrobe.anchor_id`，应看到 `semantic_plan_attempt_count=2`，并检查 `semantic_plan_validation_errors`、`semantic_plan_repair_prompt` 和最终 `semantic_plan_raw`。修复成功后错误列表应为空且角色状态应为 `resolved`；完全未输出 `character_plans` 属于语义弃权，不触发结构修复调用。

若同一个触发别名反复生成角色视觉档案，对照 `semantic_plan_raw` 中 `target_character` / `outfit_source` 的候选与已有档案的 `sourceTags`。最长边界别名命中后，实际语义查询应使用已有 `sourceTags`，新鲜档案的当前请求组件也应直接来自保存的 `tags`；日志出现 `refusing duplicate outfit profile alias` 表示 LLM1/本地查询仍返回了冲突来源，但写入已被拒绝。旧版本已经形成的冲突条目不会被自动删除，以免误删人工维护档案，应在服装词库页面确认正确 canonical source 后手工合并或删除。

## Identity 出现错误发色、瞳色或来源角色外貌

例如爱音视觉档案是 `grey_eyes`，用户只要求 Teto 风格的粉色双钻头发型，但最终 Identity 出现 `yellow eyes`。这不是服装 tag 冲突，应分别检查三层证据：

1. `danbooru_semantic_character_appearance_profiles` 或 `semantic_character_outfits[].appearance_tags` 是否仍是目标角色自己的稳定外貌，而不是 cosplay 来源角色的外貌。
2. 给第二次 LLM 的完整提示中，`Character-scoped stable appearance authority` 是否按穿着者列出档案；命中的固定角色辅助提示是否已经被最新档案重建，不再携带陈旧瞳色。
3. 最终 `structured_identity_blocks` 是否只保留用户明确修改的外貌维度，并恢复未提及维度的档案值。

当前外貌裁决单位是“角色 × 维度”，不是整份 Identity。只改发型不会解锁瞳色，只改 A 的瞳色不会影响 B。若用户明确改成蓝眼睛，最终应保留蓝眼并排除档案灰眼；若用户没有提瞳色，则 LLM 写出的黄眼、蓝眼等冲突描述必须被删除，而不是与 `grey_eyes` 同时追加到末尾竞争。
