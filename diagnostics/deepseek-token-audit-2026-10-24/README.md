# DeepSeek 输出 token 异常排查（2026-10-24）

## 后续修复

下文审计结论与 report.json 保留修复前基线。用户随后授权修复，新增 `llm_request.py`，
统一 LLM 入口在 OpenAI 兼容 adapter 上使用每次调用独立的 provider/config 副本，
通过 `custom_extra_body` 传递当前阶段的三项控制；不修改主机源码、运行配置或共享 provider。
调用控制优先于 provider 默认值，明确 disabled 时移除继承的 reasoning_effort。
原样传递正文、system、上下文、图片和 tools；非 OpenAI adapter 沿用 Context SDK。
诊断开关不影响控制传递，发生错误/取消不回退成不受控调用。

`tests/test_llm_request_controls.py` 用真实安装的 Context/adapter/SDK 与 MockTransport
验证序列化 JSON，包括普通 LLM1 的 900、普通 LLM2 的 5000、显式思考 high、
NAI LLM1 的 7200 与 LLM2 的 6920（测试画布上限22人的原预算）、并发配置隔离、
修复和取消；NAI 完整流水线仍只有两次调用。未改任何 LLM 任务书，未调用收费模型或生图。
生效需要重载插件；真实 token 占比与图片质量没有作为离线回归的验收结论。

验证结果：`test_llm_request_controls / test_llm_diagnostics / test_prompt_strategy /
test_prompt_direct_writer / test_generation_verifier / test_astrbot_host_compatibility`
共 285 项通过。首次合跑为 283 passed、2 项 tmp_path 初始化权限错误；
改用插件内本任务独立 TEMP/TMP 和全新 basetemp 后，补跑该两项及六项请求体测试，
8 项通过。三份 Python 改动 AST 解析及本次修改文件的 `git diff --check` 通过。
仅有现有主机 audioop/deactivate_llm_tool 弃用警告。

## 结论与证据强度

已经离线复现当前安装的 AstrBot OpenAI 适配器丢弃调用级生成参数。
插件要求 `thinking.type=disabled`、`max_tokens=900/5000`，实际序列化的
HTTP JSON 均没有 `thinking`、`reasoning_effort` 或 `max_tokens`。
因此插件的“未启用深度思考”摘要不等于远端非思考模式，配置的输出预算也没有生效。

这是可直接解释异常的已确认控制失效。DeepSeek 官方当前默认开启 high 思考，
输出 usage 包含推理 token；省略 max_tokens 时思考模式默认输出预算为 64K。
这使一个短 JSON/提示词回答可能伴随大量不会进入最终绘图提示词的推理输出。
64K 是上限，不是每次实际消耗。

没有访问用户的 DeepSeek 控制台、没有调用真实模型，因此不能把整站历史账单
精确归因到该插件，也不能给出历史推理 token 占比或节省金额。

## 当前配置（只读）

- 主提示词 provider：`deepseek/deepseek-v4-flash`，直连 `https://api.deepseek.com/v1`。
- 对应模型的 `custom_extra_body={}`，`reasoning=false`；后者在该适配器请求路径中
  没有转换为 DeepSeek 的 thinking 开关，不能起到关闭思考的作用。
- 插件 `prompt_builder_max_tokens=5000`，深度思考功能开启、effort=high。
  插件功能开关仍受关键词触发；近期任务的请求意图均为不思考。
- `debug_llm_usage_enabled=false`，`debug_llm_reasoning_enabled=false`；
  主机文件日志关闭。任务文件没有保存逐次真实 usage，无法从正文字符数反推推理量。
- 官方当前把旧名 `deepseek-v4-flash` 映射到新版 Flash；此处记录官方当前说明，
  没有用虚拟响应的 model 字段证明远端实际版本。

## 完整请求路径与失效位置

命令入口 → generation task/raw 与手工尾缀处理 → PromptPipeline → LLM1 语义规划
→ 本地角色/Danbooru 证据 → 服装与外貌权限 → LLM2 普通七字段或 NAI 实例 JSON
→ 确定性过滤与组装 → ComfyUI 工作流提交/history/下载 → 聊天交付。

失效位于两次 LLM 的共同调用下层，不在 ComfyUI、衣柜过滤或结果交付：

1. `prompt_pipeline.py:5003` 的语义规划明确传入关闭思考，普通预算至多 900；
   NAI 路径可传 7200。普通 writer 在 `:4735` 附近根据研究计划传思考开关；
   NAI direct writer 在 `:6106` 附近明确关闭思考，预算按角色上限扩展、至多 7200。
2. `llm_diagnostics.py` 原样转交 kwargs，没有删除这些参数；诊断中的
   `requested_*` 也只代表请求意图，不能当作实际 HTTP 参数。
3. 当前主机 `astrbot/core/star/context.py:175` 的 `Context.llm_generate`
   将 kwargs 传给 provider 的 `text_chat`。
4. `astrbot/core/provider/sources/openai_source.py:956` 的 `_prepare_chat_payload`
   接收 `**kwargs`，但只从 messages 和 model 构建 payload，未合并 kwargs。
5. 同文件 `:560` 附近的 `_query` 另外合并 provider 配置的 `custom_extra_body`。
   这是本次对照实验中可把开关送入实际 HTTP JSON 的配置通道。

普通优化通常两次模型调用；NAI `-r` 直接用自己的 LLM2，不额外先写一版七字段。
raw/关闭优化绕过该提示词调用；手工 `#` 尾缀不交给语义阶段。
研究、服装摘要、低 CFG、旧多人及生成后验图还可能另有调用，不应与这两阶段混算。

## 离线传输验证

`audit.py` 使用已安装的真实 Context、ProviderOpenAIOfficial、OpenAI SDK，
经 `httpx.MockTransport` 捕获序列化请求体。只用虚拟 key，所有 HTTP 由内存处理，
不产生 API 或生图费用。四个断言组通过：

| 输入位置 | 请求参数 | 实际 HTTP JSON |
| --- | --- | --- |
| 调用 kwargs | disabled + 900 | 两项均丢失 |
| 调用 kwargs | enabled + high + 5000 | 三项均丢失 |
| 调用 kwargs.extra_body | disabled + 900 | 两项均丢失 |
| provider.custom_extra_body | disabled + 900 | 两项均保留 |

最后一行仅为有效通道对照，900 不是部署建议，尤其不适合直接套给复杂 NAI writer。
不能只把插件调用改成 `extra_body=...`，当前主机会同样丢弃该 kwargs。

## 历史任务核对

`report.json` 汇总 10 月 18–24 日保留的任务，按文件读取时的快照统计，
不输出用户原文或身份。各天任务均未请求深度思考；这不能证明 API 真正未思考。

10 月 24 日快照共 107 条任务、86 条 NAI `-r`：8 条记录语义规划修复，
writer 记录没有整份修复。修复有额外消耗，但不能用频繁 writer 重试解释当天现象。
计数字段是已记录尝试，包含失败/零次等情况，不等于严格计费请求数；
也不覆盖 SDK/主机的网络重试和共享 key 的其他应用。

当日有保存 NAI writer 输入的 83 条样本，平均 user 任务书约 23661 字符；
84 条初稿输出样本平均约 2138 字符。两组样本不完全配对，且都不是 token 数。
这个量级不支持把可见 JSON 正文本身视为十倍膨胀的直接证据，
隐藏推理仍需用真实 `completion_tokens_details.reasoning_tokens` 验证。

此外，若控制台“输入”指未命中缓存部分，必须把缓存命中与未命中相加再与输出比较：
`prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens`。
缓存命中减少输入计费，不代表没有输入；输出含推理，而最终 prompt 只保留正文。

## 处理方向（本次未实施）

优先修复调用级参数的实际透传，保留 LLM1/LLM2 各自的开关与预算，
并用最终 HTTP JSON 做回归。避免临时改共享 provider 对象造成并发请求相互覆盖。
若选择配置应急方案，使用独立的生图 provider，在其 `custom_extra_body`
明确设置关闭思考；它会统一关闭该 provider 的思考，改变原来关键词可开启思考的行为，
需要把这个契约变化与永久的调用级修复区分。

恢复可观测性时记录 input/cached/uncached/output/reasoning/finish_reason、
stage 和 attempt。reasoning 是 output 的子集，不能重复相加。
不必打印推理正文即可分析用量。

本次仅添加审计脚本与报告；没有改提示词、运行配置、生产代码、provider 或重载服务。
保持两阶段、角色/服装/外貌权限、原文/双语职责、修复边界及 ComfyUI 交付流程。

## 官方参考（2026-10-24 查询）

- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)：默认 enabled/high，开关格式。
- [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)：max_tokens 默认预算与 usage 字段。
- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)：旧 Flash 名称的当前映射。
- [Context Caching](https://api-docs.deepseek.com/guides/kv_cache/)：命中/未命中输入含义。
