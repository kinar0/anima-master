# 外貌来源与接收者重叠时的误修复

用户日志：2026-10-22 21:27:58，原请求为“爱音祥子交换发色发型和瞳色”。日志止于第一阶段修复任务书开头，没有第二阶段回复或最终图片，因此不能推断该次请求最终是否成功出图。

LLM1 初稿正确返回爱音、祥子两行，各自将 `hair_color / hair_style / eye_color` 设为 `replace` 并引用对方。每条修改的 `source_text` 都是完整请求。宿主却要求删除双方，原因均为 `occurs only inside another person's appearance reference`。

`_appearance_reference_only_names` 在 `58c1afb` 中加入，既被规划校验调用，也在可读协议转换为内部 anchors/角色计划时调用。旧实现只检查名字是否全部位于另一人的引用范围；完整共享原文会同时覆盖参与者和来源，不能证明某人只是来源。修复失败后沿用初稿也会再次丢人。

本次只收窄该确定性判定：复用已有外貌修改解析器，具有原文支持、受支持维度及有效操作的修改接收者，不因同时成为他人的外貌来源而被排除。不通过姓名、交换关键词或推测具体外貌恢复人物。没有自身有效修改且所有出现均在来源引用中的人物仍过滤；独立出现仍按原规则保留。LLM1/LLM2 的 system、user、repair 任务书均未改写，衣柜、来源维度、raw/#、旧多人旁路和 ComfyUI 提交/下载/交付保持原流程。

验证记录：

- 新增用例在修复前为 6 失败、4 通过：原姓名/其他姓名与维度、三方传递、来源另有自身修改，以及普通和 NAI `-r` 提示词链路均能复现误判；无效证据反例保持过滤。
- 修复后运行 `tests/test_appearance_reference_scope.py`、`tests/test_danbooru_semantic.py`、`tests/test_prompt_direct_writer.py`、`tests/test_outfit_partial_changes.py`：**249 passed**。使用 AstrBot Python 3.12、core PYTHONPATH、关闭 cacheprovider 和本任务独立临时目录。
- [replay.json](replay.json) 保存用户日志中真实 LLM1 初稿的回放：校验错误为空，两名目标、两份角色计划、六条外貌修改均保留。
- 普通和 NAI 管线测试检查了写作者收到的双向维度参考、无额外规划修复，以及最终两人的对应外貌；第二阶段输出为本地替身，不能当作真实模型复测或图片质量保证。

没有调用外部 provider、没有提交 ComfyUI/NAI 图片，没有自动重载运行服务。更新后的判定需重载插件后才用于新请求。
