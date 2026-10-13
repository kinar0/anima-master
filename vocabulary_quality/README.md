# 中文口语词库离线审查

当前结果、撤回原因及下一步见 [comparison.md](comparison.md)。只新增审查材料；原库 1427 行未改，候选未应用。`事后=aftersex` 按用户确认语境保留。

132 条样本全部为人工合成，不是聊天记录，36 条预先留作独立验收。数据、候选和原库分开保存。原库 54 条问题标记只是初筛，包含语境风险，不意味着全部应删除。其余行也未得到全语境正确认证。

33 行候选对比产生 4 项验收新错误，批次未通过门槛。相关候选已暂缓，剩余 29 行仅是待新验收集验证的维护草案。`final_dispositions.json` 的 `provisional_rows` 不是可直接导入的发布清单。不要再用同一验收集筛词并宣称新批次已通过。

本地复核（在插件目录；不调用模型、生图、导入器或配置保存入口）：

```powershell
$taskCore = (Resolve-Path '..\..\..').Path
$taskPython = Resolve-Path '..\..\..\.venv\Scripts\python.exe'
$env:PYTHONPATH = if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $taskCore
} else {
    "$taskCore;$env:PYTHONPATH"
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $taskPython vocabulary_quality/run_evaluation.py
& $taskPython vocabulary_quality/review_original.py
& $taskPython vocabulary_quality/finalize_report.py
```

这些命令更新本目录的衍生报告，不改冻结的 `evaluation.jsonl`、原映射快照或运行配置。源码之后改变会导致复现结果变化；对照 `integrity.json` 与初始模块散列，不用历史结果代替当前验证。完整性报告中的并发修改不是本任务改写；全部保留。

`prepare_evaluation.py` 是评估集来源记录，现有文件存在时拒绝覆盖。`prepare_candidates.py` 是人工候选来源与本地 exact 查询入口，现有候选别名改变时拒绝覆盖；重跑只复用已保存的查询证据，不联网。若需要刷新目录证据，另开新审查批次，不覆盖本轮的历史证据。`--snapshot` 只读运行配置并导出词汇，当前快照存在时拒绝覆盖，不能作为应用工具。

本轮未复用 Toolbox 的中文别名，没有全量导入；仓库原有 `third_party/llm_nai_toolbox/` 的来源与 MIT 说明保持原样。
