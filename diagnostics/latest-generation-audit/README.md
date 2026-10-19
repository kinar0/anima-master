# 17:19 三人互动生成的提示词审计

任务：`ca259e35234145c6957f3b988dae2c72`，开始于 2026-10-17 17:19:27，原图保存于 17:19:43，任务成功且交付状态为 sent。审计读取当前六份流程文档、任务保存的两阶段原始回复、实际 LLM2 任务书、缓存证据和原图 PNG 内嵌的 ComfyUI API 工作流；核对相关管线及已安装转换节点源码，并离线执行文本转换。没有重新调用 DeepSeek、NAI 或 ComfyUI，没有查看图片像素、修改图片、运行配置、档案或生产代码。

结论：**两阶段链路、三人身份、方向、工作流传递符合预期；标签与可见性质量仍有缺口，不能将“任务成功”理解成整份提示词全部正确。**

## 逐阶段核对

| 阶段 | 本次实际证据 | 判断 |
| --- | --- | --- |
| 命令与绕过 | `-r` 生效；没有 raw、旧独立多人或参考图；横幅 1216×832 | 符合专用逐实例路线 |
| 画风与手工尾缀 | `-s8` 对应当前显示顺序中的 `nai3d`；`#` 后 14 字符尾缀未进入 LLM2 任务书，只在模型完成后加入实际全局输入 | 符合边界；summary 的全局 prompt 是插入尾缀之前的快照，原图工作流才包含尾缀 |
| LLM1 原始结果 | 三项分别为丰川祥子、粥祥、千早爱音；衣服和来源均 null，修改均为空 | 正确识别三名独立可见人物，没有把粥祥当作供衣者；脸红没有被误当稳定外貌修改 |
| 本地绑定 | 三个 target 分别绑定 `togawa_sakiko`、`togawa_sakiko_(master_of_melodia)`、`chihaya_anon`；没有 outfit_source | 粥祥的后缀身份完整；此次不需要条件参考救回 |
| 衣柜与外貌 | 三人均未指定服装，各自收到 default_reference；appearance 按人物分列 | 使用软参考而非擅自将衣柜认定为用户指定；缓存中有脏标签，详见下文 |
| LLM2 原始结果 | 全局 3girls，三个独立角色框；全局与各框均有英文细节标签、中文关系句；三项 identity_review 均 visible 且原文可定位 | 身份完整性检查确实已在运行；不是旧代码或只保存未加载 |
| 动作与位置 | 丰川祥子 `(0.2,0.55)`、粥祥 `(0.8,0.55)`、爱音 `(0.5,0.6)`；两名 source#teasing、一个 target#teasing | 左右夹中间的关系一致；多主动方指向同一接受方的结构成立；中文也明确相应人物，不只依赖方向标签 |
| 后处理 | 从全局删除 sitting / sitting_on_bed，角色框仍保留；方向标签由数组写入框内，未重复普通 teasing；无方向降级、无实例证据丢弃 | 作用域和方向处理正确；唯一额外普通 token 是从祥子档案恢复的 yellow_eyes_blue_hair |
| 工作流 | 原图含 NovelAIGenerator V5 Full、CharacterPromptSelect 和每框独立转换节点；三个框的转换前文本及坐标逐字/逐值匹配任务最终实例 | 没有在宿主到 ComfyUI 的传递中丢粥祥、换序或改坐标；离线执行当前转换器也保留完整身份与方向标签 |
| 交付 | 一张原图，之后经既有发送前过滤生成交付文件，状态 sent | 本次没有任务层面的重复出图或交付失败证据 |

LLM1 与 LLM2 各一次、均接受初稿；没有结构修复、身份修复或第三次拆分调用。这里确认的是插件应用层调用，未捕获 HTTP 层请求日志。`sensual_mode=false` 与尾缀边界一致：手工词不参加模型语义分析或该模式触发。

## 已确认的问题与来源

### 1. 缓存外貌中的复合伪标签被当成权威事实恢复

丰川祥子的保存外貌含 `yellow_eyes_blue_hair`。LLM2 原始 prompt 没有这个 token，后处理后的框内却新增了它；这是本次证据可直接证明的宿主新增项。该字符串在本地标签库中未确认，而原始框早已分别有 yellow_eyes 和 blue_hair。

路径为：已保存 appearance profile → 检测本框已经表达的维度 → stable 恢复 → 追加该档案 token。现有匹配精确身份优先的修复正确找到了档案，但档案中的非规范 token 仍被信任；身份绑定成功不代表档案每一项已重新验过。这里没有修改档案，也没有将错误笼统归为模型生成。

### 2. 单复数拼写和模型错词仍可直接进入成品

使用本地 `danbooru-tags` 技能的 CLI，先按语义分组 exact 查询，再对漏匹配项按 general category exact 补查一次：

| 字符串 | 本地库精确结果 | 本次来源和影响 |
| --- | --- | --- |
| two_side_up | exact_tag | 丰川祥子的实际发型标签；此拼写有本地证据 |
| two_sides_up | 未确认 | 粥祥档案与 LLM2 原始输出已有，最终保留；祥子缓存中也另存了这个复数形式 |
| yellow_eyes_blue_hair | 未确认 | 祥子保存外貌中的复合 token，由后处理追加 |
| partable_lips | 未确认 | 爱音 LLM2 初稿的明显拼写问题，后处理没有纠正；parted_lips 可精确确认 |
| hands_on_own_lap | 未确认 | LLM2 初稿中已有，最终保留；中文手部位置表达仍在 |
| sitting_on_bed | 未确认 | LLM2 初稿中已有，最终在实例保留，中文坐姿表达仍在 |
| bed_sheet | exact_tag | 全局保留正常；第一次 scene 分组没命中，general 补查确认 |
| teasing | exact_tag | 本次动作名有本地证据；不能因 action 分组没命中就宣称不存在 |
| master_of_melodia | 未确认 | 被列在 LLM2 上下文的 Other confirmed non-wardrobe hard tags，但最终未输出；“confirmed”措辞证据过强，未污染本次提交 |

“本地未确认”不等于证明 NAI 完全不能理解该短语。它意味着不能把该词宣称为已验证 canonical。对已经有明确正确拼写的错词，以及拆成两个独立标签的复合外貌，问题更直接。当前 NAI 直接写作路线的后处理主要核对身份、原文依据、作用域、方向与衣柜/外貌权限，并没有对 LLM2 每个普通 tag 做全量词库校验。

### 3. 服装组件的可见性取舍仍偏弱

模型选择平视中景，同时从软参考复制了祥子的鞋袜，以及粥祥的鞋、靴、多种腿装。它们在模型原始回复中已经存在，不是后处理临时补出的服装。爱音没有拿到其他人的 hair_ribbon，发饰串人的旧现象在本次未复现。

本次没有 nai_composition_omitted_tags；当前 framing 分类没有把 medium_shot 单独当作明确的上半身裁切。因此可确认的是“取景与整套组件之间缺少更谨慎的选择”，不能仅凭中景这个词就断言每件鞋袜必然在画外，更不能声称像素层面已经验证。粥祥多项腿装也来自档案，应先检查它们究竟是可共存组合还是被统计混合的候选。

坐标的左右关系正确，但 `.2/.5/.8` 本身只能表达中心，不能证明肩臂真的相接。此次没有检查图片像素或肢体效果。

### 其它容易误判的记录

- 全局的 detailed light / ray tracing / cinematic lighting 等来自用户选定的 nai3d 画风组，LLM2 原始全局未自行添加它们，不能算作模型违反“未要求不补光”的证据。
- 爱音档案中的 `(flat_chest:1.4)` 在实际框内成为无权重 flat_chest：事实保留，档案权重未继承。这是当前写作与去重路径的可观察行为；不能将它误报为体型档案未命中，也不能据此宣称所有保存权重都得到硬保留。
- 顶层 workflow 字段仍写 anima_t2i，因为任务开始时保存的是该配置名；实际自定义工作流由 PNG 中 NovelAIGenerator 证明。strategy_summary 中旧多人统计的零值也不能覆盖 nai_character_count=3 与实际三个节点输入。
- debug 中 llm_prompt 与 nai_character_plan_prompt 内容相同；prompt_llm_initial_content 与 nai_character_plan_raw 内容相同，且 prompt_summary 还有嵌套副本。它们是重复存储，不代表重复发送多次模型请求。

## Token 核对

以下数值由用户提供的 DeepSeek 页面读取，不是本次插件任务自行保存的 usage：

| 项目 | token |
| --- | ---: |
| 输入：缓存命中 | 3712 |
| 输入：缓存未命中 | 1634 |
| 输入合计 | 5346 |
| 输出 | 1247 |
| 总计 | 6593 |

算术完全一致：3712 + 1634 = 5346，5346 + 1247 = 6593。输入缓存命中率为 **69.44%**，分母应使用输入 5346，而不是含输出的 6593。[DeepSeek API 文档](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/)定义 prompt_tokens 为缓存命中与未命中的和，total_tokens 为 prompt 与 completion 的和。

[DeepSeek 缓存文档](https://api-docs.deepseek.com/zh-cn/guides/kv_cache/)说明它复用已落盘的相同输入前缀；命中不表示省略了对应指令，也不表示直接复用了旧输出。此次固定规则放在大段稳定前缀、动态请求和逐角色证据在后，与出现较高输入命中率相容；但不能从聚合数字精确倒推出命中了哪一段。

保存的 LLM2 user 任务书为 19502 字符，其中首次动态 Actual canvas 之前的规则前缀为 15355 字符；当前代码重建的 writer system 为 3088 字符、LLM1 user 为 3041 字符。实际保存的两阶段输出分别 356、2691 字符。**这些是字符数，不是 DeepSeek tokenizer 的精确 token 数；system 与 LLM1 任务书是当前代码重建，非历史 HTTP 消息抓包。** 可确认短中文请求外还有两阶段协议、逐角色证据、JSON、双语视觉描述、位置理由和身份核对输出，不能只用用户原句长度判断消耗异常。

当前插件提取 completion_text 后没有把每次响应的 usage / API request id 写入任务。宿主 provider 本身虽有 input_cached/input_other/output 字段，这里仍没有可据以逐阶段回算的历史值。因此：

- 插件证据支持“两次应用层调用、没有提示词重试”，不支持根据 6593 推断存在额外循环。
- 若 API 页面是每请求一行，这组数值可能只是一条调用；若页面按时间段聚合，还须确认聚合范围。仅凭任务和这四个数字，不能独立证明 6593 已合并本次两阶段，也不能给第一阶段、第二阶段分别编一个精确数值。
- 不据本次数字删减提示词职责。需要精确记账时，应保存每阶段 provider/model/request id、输入命中/未命中、输出 token 和 attempt，再与 API 页面汇总；不需要保存密钥。

## 证据与复查

- `evidence.json`：安全摘取的原始/最终实例、逐 token 差异、身份、档案、工作流对应、用量算术及限制；不包含生成节点认证字段或聊天用户 ID。
- `tag-queries.json / tag-results.json`：第一轮按组 exact 查询。
- `tag-supplement-queries.json / tag-supplement-results.json`：general category exact 补查，避免把分组漏匹配当作标签不存在。
- `audit.py`：只读历史任务与 PNG metadata；断言三个框、身份、坐标、画布和手工尾缀与实际提交一致，并离线重建文本转换。不会调用模型、生图或网络 API。

本轮只完成审计。没有自动清理用户档案、改提示词/规则、重载插件或重新生成图片。
