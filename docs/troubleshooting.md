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
