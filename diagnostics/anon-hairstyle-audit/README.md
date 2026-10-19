# 2026-10-19 爱音借用祥子发型：根因审计

结论：**用户观察到的扩写确实存在，但问题不只是 tag 多。主链没有把“只借发型”稳定地限定为有证据的局部外貌迁移，写作者猜错关键发型、扩写衣柜和光影，后处理又误删部分发型及发饰，最终强化了次要条件。**

本次仅检查，没有修改生产代码、运行配置、角色档案或 LLM 提示词，也没有重载或重新生成。新增的文件仅为本目录审计材料。

## 范围与实证

- 找到 17:33–18:43 的 13 条优化尝试，以及 19:04–19:07 的 5 条原样尝试。优化记录中一条没有保存输出图。
- 读取全部相关任务的 LLM1 原始 JSON、实际 LLM2 任务书、LLM2 原始七字段、后处理结果及衣柜权限。
- 17 张原图的 PNG 内嵌工作流均匹配任务最终文本：优化任务包括恢复 `#rurudo` 后的文本；原样任务等于去掉“无优化”前缀后的输入。实际为 NAI Diffusion V5 Full，未启用 `-r`，优化任务走普通七字段写作。
- 已读取仓库要求的六份主流程文档，并检查相应实现。使用当前源码的原始纯函数定义进行离线重放，复现发型误删、发饰过滤和构图识别缺口；不是新模型回归或整条流水线重新执行。
- 本地 `danbooru-tags` 精确查询确认 `two_side_up` 与 `hair_ribbon`；`two_sides_up` 未获得精确命中。appearance 分组未命中的 `two_side_up` 已通过同批 general 精确查询确认，不能把分组漏匹配当成词不存在。查询仅验证拼写，不能证明图片效果。

## 1. 核心发型没有得到准确、稳定的表达

用户用中文引用祥子发型的优化尝试，都没有在模型七字段输出中出现准确的独立 `two_side_up`。模型先后写成：

| 时间 | 任务 ID 前缀 | LLM2 原始表达 |
| --- | --- | --- |
| 17:33 | c75441a3 | `twin_braids, side_ponytail` |
| 17:43 | 1368e69e | `parted bangs, side-swept bangs, hair between eyes`，另用 prose 猜测 half-up silhouette |
| 18:32 | 9d11fecd | `togaawa_sakiko` 的 twin-tail-like half-updo，名字还多拼了一个 a |
| 18:36 | affa3bdf | 普通 `twintails` |
| 18:40 | 61900b5c | `half-up_ponytail` |
| 18:43 | b7375fe2 | `side ponytail` |

直接写英文的尝试同样不稳定：17:34 的 `two side up` 与 `twin braids` 并列；17:36 又与 `twin buns` 并列；17:40 使用未精确确认的 `two_sides_up`。因此，不是删除几十个装饰 tag 就必然修好；关键条件本身已被换成竞争造型。

保存的祥子外貌资料里有 `two_side_up`，但也混有 `two_sides_up`、`yellow_eyes_blue_hair` 等旧 token。普通 LLM2 任务书只列出爱音自己的稳定外貌和“把头发梳成祥子发型”的原文修改，没有将祥子已知发型作为明确限定的来源证据传入。17:43 只额外得到一个 `Other confirmed non-wardrobe hard tags: togawa_sakiko`，模型仍需凭自身知识猜发型。`prompt_pipeline.py` 的逐角色稳定外貌上下文和修改上下文在约 7140–7208 行。

## 2. 借发型被误当借衣服，进一步扩大衣柜条件

17:43 的 LLM1 正确返回 `hair_style/replace`，`clothing=null`、`clothing_source=null`，但未指定服装仍触发爱音的 `default_reference`，完整校服组件进入 LLM2 软参考。模型将这些参考展开成白衬衫、灰外套、领带、百褶裙。

18:40 和 18:43 更进一步：原文仍只说换发型，LLM1 却返回 `clothing_source=丰川祥子`。主机接受为 `outfit_source`，给 LLM2 的实际上下文变为：

> 千早爱音: cosplay source = togawa_sakiko; verified outfit anchors = white_shirt, collared_shirt, haneoka_school_uniform, grey_jacket, plaid_skirt, ... white_socks, black_shoes.

写作者随后输出完整校服甚至鞋袜。这是“外貌来源被误用为服装来源”的语义与权限边界问题。字符串确实能在原文定位，只能证明引用存在，不能证明它承担借衣关系。

`global_outfit_reinforcement_tags` 全部为空，衣服已经存在于 LLM2 原始回复，**没有证据表明主机在最后自动注入整套校服**。主机的贡献是错误或过量的上游证据、以及后置保留/删除，而不是把全部衣柜直接拼到末尾。

## 3. 外貌过滤把发型里的 tail 误判为身体尾巴

当前 `prompt_pipeline.py:1362` 附近的 tail 维度规则能匹配独立 `tail/tails`。它也会命中：

- `two side up twin tails`
- `half-up twin-tail-like silhouette`
- `twin-tail-like half-updo`

这些片段被分类为未授权的身体 `tail` 维度。即使 LLM1 已明确允许 `hair_style/replace`，它也没有开放身体尾巴维度；`merge_authoritative_identity_block` 在检查到任一锁定维度不符合档案时删除整个片段。

离线执行当前原函数与历史结果一致：

- 17:34 的长粉发与 two-side-up 整段从 Identity 消失，最终只补回 `long_hair, pink_hair` 等档案事实。
- 17:43 的 half-up twin-tail-like 片段消失，只剩刘海描述。
- 18:32 的整段发型描述消失，Identity 只剩灰眼、表情及补回的长粉发。

这不是模型最初没写，也不是“显式发型被正确覆盖”。它是主机的维度分类误伤。真实 `fox tail` 作为附近反例仍应归身体尾巴；未来修复应区分两类意义，不能全局放开 tail 权限。

## 4. 发饰被旧衣柜过滤误伤，中文大头照未提前裁剪衣柜

17:33、18:32 等任务的 LLM2 原始 Tags 有 `hair_ribbon`，最终共享 Tags 却没有。祥子档案中的 `hair_ribbon` 被列入未选衣柜的 `stale_cached_tags`；对保存的权限对象执行 `filter_tags(('hair_ribbon',))` 得到空。来自新发型的发饰没有独立权限，因此同名就被当成旧衣柜污染清掉。

`framing_hidden_outfit_slots('千早爱音超级大头照')` 返回空，说明中文“大头照”没有在 LLM2 之前触发服装裁剪，模型收到完整校服参考。后续英文 `headshot` 能裁掉衬衫、外套、裙袜，却仍有两个缺口：

- `green_necktie`、`long_sleeves` 的服装槽位为空，不受相应画外槽位裁剪，最终仍保留。
- prose 中的删除依赖语法边界，出现主段已删衣物、Nltags 仍残留衣物，以及双逗号等不一致。

这解释了为何大头照最终仍有领带、长袖、校服叙述，而发型需要的发带反而可能消失。源码入口为 `prompt_background.py:131`、`prompt_pipeline.py:3171/3192/1920`、`outfit_transfer.py` 的服装槽位分类。

## 5. 普通写作契约确实鼓励扩写和重复

所有优化任务都保存 `prompt_builder_template_customized=true`，应以各任务实际任务书为准，不能假定运行的就是当前默认模板；但不能将问题全部归咎于自定义模板，因为当前内置模板也保留以下两条：

- 可自由决定姿态、构图、镜头、光影、色彩、氛围和特效。
- Nltags 保存完整画面自然语言，“可以和之前的内容重复”。

18:43 原始回复由此额外产生 `sunlight, soft lighting, wind, gentle smile, blush, depth of field, bokeh`，并把校服和发型再次写进 Nltags。未指定光照时不额外润色的约束在普通七字段入口没有形成和 `-r` 同等明确的一致要求。背景附加规则虽要求不增加无关光照，任务书内部仍存在相反的扩写倾向。

当前 Nltags 去重按句子的规范化全文/包含关系比较；一条长句重新组织同样事实，不能因此被有效去重。18:43 实际送入转换节点的文本为 1191 字符，19:07 原样最终一条为 174 字符。此处是字符数，不是 token，也不能以长度单独判断好坏；差异的实质是错误发型、额外条件和重复强化。

`masterpiece/best quality/questionable`、当时的 `nsfw/high quality` 等前缀是后续本地组装内容，不在 LLM2 原始七字段中。`#rurudo` 也由用户尾缀通道插回。不能把所有最终新增项都算成模型自主添加。

## 用户自测支持什么

用户自测保留了精确的 `two side up`、`hair ribbon`、爱音身份/发瞳色及少量场景与项圈条件，并显式加强发型。原样入口绕开两阶段模型、衣柜参考及其冲突过滤，因此同时避开上述扩写、猜发型和误删问题。这与用户报告的效果改善相容。

不过历史尝试同时改变发型拼写/权重、镜头词、项圈词、画师写法和 seed；本次没有检查图片像素或控制变量重生成，因此不能从现有资料量化每个因素对成图的贡献，也不能断言唯一原因就是长度。

附带发现：19:07:50 的 `无优化 1024x1024 ...` 将 `1024x1024` 保留为提示词字面量，PNG 实际仍为 832×1216；早先 `1024*1536...` 也残留在模型原文。它是尺寸解析/参数位置的另一项问题，不是本次额外 tag 的主要来源。

## 修复应守住的通用不变量

1. 区分外貌来源、服装来源和独立可见人物；只借某一外貌维度不能授予整套衣柜权限。按来源的相应维度传递已有证据，不能只给来源姓名让 writer 猜。
2. 发型中的 tail 与身体尾巴必须分开分类，保护已授权发型整段，同时继续阻止无依据身体尾巴。
3. 新发型的配饰不能仅因与其他人的旧衣柜同名就删除；也不能直接开放全部历史发饰。
4. 构图应在写作前限制参考，写作后统一检查 Tags/Details/Nltags；领带、袖子及服装身份不能成为画外参考的漏网项。
5. 普通写作保留完整语义职责，但不能用错误近义发型、未请求的装饰和多字段重复覆盖核心修改。不应靠固定 tag 上限或某个角色特例修补。

本次未实施上述修复，也未以删除大段 LLM 指令替代根因处理。后续如改提示词，需要遵守仓库提示词维护契约，逐项对照职责，并以真实模型多样例和反例验收。

## 复查文件

后续修复已在同一工作区实施，通用规则、职责对照及真实模型回归见 [修复记录](../appearance-reference-fix/README.md)。本文及 `evidence.json` 保留修复前基线，不以新结果覆盖旧失败。`audit.py` 现将当前源码复放写入 `evidence-current.json`，继续核对同一批 18 个任务和 17 份 PNG 提交文本；不会重发图片或调用生成端点。历史请求级衣柜快照没有新增的穿着者发型配饰权限，完整新链路由回归测试验证。

- `evidence.json`：18 条任务的安全摘取、17 份实际提交文本、原始两阶段结果与任务书、当前源码纯函数重放。
- `audit.py`：只读任务和 PNG 元数据；不会调用模型、NAI 或 ComfyUI，不解码图片像素，不保存生成节点认证字段或聊天用户标识。
- `tag-queries.json`、`tag-results.json`：本地精确标签检查及结果摘要。
