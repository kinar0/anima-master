# Command-line environment

This repository is an AstrBot plugin workspace. Every new Codex task starts in
this directory unless the task context explicitly says otherwise:

`C:\Users\69493\.astrbot_launcher\instances\1617d9a4-b150-48c1-a04e-cc719eaf7a73\core\data\plugins\astrbot_plugin_anima_master`

- Shell: PowerShell on Windows.
- Use PowerShell syntax and Windows paths in shell commands.
- Python sources and tests are in this plugin directory. The current terminal
  has no globally available `python` or `py`; use AstrBot's bundled virtual
  environment for targeted tests and formatting checks instead.
- After the AstrBot 4.29.1 launcher update, the instance's working Python is
  Python 3.12 in the instance-level `venv`, alongside `core` (not `core/.venv`).
  From this plugin directory, invoke it explicitly:

  ```powershell
  $taskCore = (Resolve-Path '..\..\..').Path
  $taskPython = Resolve-Path '..\..\..\..\venv\Scripts\python.exe'
  $env:PYTHONPATH = if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
      $taskCore
  } else {
      "$taskCore;$env:PYTHONPATH"
  }
  & $taskPython --version
  & $taskPython -m pytest -p no:cacheprovider tests/<targeted_test>.py
  ```

  Every pytest invocation from the plugin directory must use this setup on the
  first run, not only after collection fails. Adding the `core` directory to
  `PYTHONPATH` makes the parent `astrbot` package importable, while
  `-p no:cacheprovider` prevents pytest from attempting to write `.pytest_cache`.
  A collection failure caused by omitting either option is an environment setup
  error and should be avoided rather than used as a diagnostic attempt.

  If the system pytest temporary directory is inaccessible, set `TEMP` and
  `TMP` to a task-specific directory inside this plugin and pass a fresh
  `--basetemp` path there. Do not reuse or delete another task's temp directory.

  Do not install, download, or create a replacement Python runtime unless the
  task specifically requires it.
- Before relying on runtime-specific details (current directory, Python
  executable/version, active virtual environment, or Git state), inspect them
  in the current task with read-only commands such as:

  ```powershell
  Get-Location
  $taskPython = Resolve-Path '..\..\..\..\venv\Scripts\python.exe'
  & $taskPython --version
  git status --short
  ```

Treat the task-provided working directory and permission boundaries as
authoritative if they differ from this file.

# LLM prompt maintenance: do not casually simplify

**严正警告，适用于后来的维护者、AI 代理和用户本人：不要随便精简或简化给 LLM 的提示词。**
提示词承载功能契约，不能仅因文字长、看似重复或为了节省 token，就删减职责、弱化强制要求、移除反例或把具体规则替换成泛泛自审。
用户笼统要求“整理／精简／去重／优化”时，默认保留全部功能、约束强度和当前工作流；这不等于授权删功能或重新设计调用链。用户明确要求改变具体行为时，应说明对应契约变化。
改动前逐项对照当前实际拼装文本、工作流文档与必要的 Git 历史；改动后检查模型原始输出和最终生成文本，按已有授权做必要回归并说明未验证范围。字符数下降、合法 JSON、纯替身测试或一次成功均不能单独证明等价。
详细要求见 [提示词维护严正警告](docs/prompting.md#严正警告不要随便精简或简化给-llm-的提示词)。涉及提示词改写时必须先读该节。

# Bug-fix prerequisite: understand the image-generation flow

Before attempting to diagnose or fix any bug that can affect image generation,
read the repository's existing Markdown documentation in the current Codex
task. Do not rely only on memory from an earlier task, a task summary, the bug
report, or a single source file.

At minimum, read these documents completely, in this order:

1. `README.md` — supported commands, modes, and user-visible behavior.
2. `docs/中文提示词到英文提示词流程.md` — the end-to-end path from command
   intake and prompt processing through ComfyUI submission and result delivery.
3. `docs/角色与服装配置工作流.md` — the two LLM stages, character/profile
   evidence, wardrobe authority, appearance merging, and final tag filtering.
4. `docs/角色视觉档案与服装优先级.md` — character appearance and wardrobe
   precedence, modification semantics, and known unstable cases.
5. `docs/prompting.md` — prompt modes, multi-person behavior, raw bypasses,
   presets, and prompt-related user contracts.
6. `docs/troubleshooting.md` — known failure signatures and the intended
   diagnostic evidence.

Also read the relevant supporting Markdown before changing code in these
areas:

- configuration or defaults: `docs/configuration.md`;
- ComfyUI connectivity, startup, or cross-machine behavior:
  `docs/deployment.md` and `docs/quickstart.md`;
- Turbo or low-CFG behavior: `variants/README.md`,
  `variants/turbo/README.md`, and
  `variants/turbo/low_cfg_harness/README.md`.

Before editing code, be able to place the reported failure in the complete
request path: command dispatch -> generation task and bypass handling -> prompt
pipeline -> semantic planner -> character/Danbooru evidence -> wardrobe and
appearance authority -> prompt-writer output -> deterministic post-processing
and final prompt assembly -> ComfyUI workflow submission/history/download ->
chat delivery. Identify the upstream inputs, downstream consumers, relevant
bypass or compatibility paths, and the invariants that the fix must preserve.

The documentation is the required starting model, not a substitute for code
inspection. Verify the documented flow against the relevant implementation and
tests before deciding on a root cause. If a fix intentionally changes the
documented flow or user-visible contract, update the affected Markdown in the
same task so that future bug fixes do not begin from stale documentation.

# Bug-fix design: prefer general rules over case-specific patches

When a user reports a failed generation, use the example to locate the broken
invariant. Fix that invariant for the whole class of inputs instead of adding a
character name, outfit name, exact sentence, or one-off tag to make the reported
example pass. Check how the same failure would behave with a different character,
an unseen garment or named outfit, different modifiers, and multiple wearers.

Keep deterministic recovery within evidence that can be tied to the request.
When ownership or meaning is ambiguous, preserve the user's original wording
for the next semantic stage and prevent an incompatible default from taking
over; do not guess a specific outfit or silently restore a complete saved
profile. Explicit user changes to a selected outfit take precedence over its
stored components. Add regression tests for the broader behavior and for a
nearby counterexample that must *not* match. Update the relevant workflow docs
when the behavior changes.

Reference: commit `d56cd20` fixed missed clothing in the prompt pipeline. The
initial example involved a modified Tsukinomori uniform, but the final rule in
`prompt_pipeline.py` handles directly stated wearer-to-clothing relationships:
it binds any locally confirmed named outfit, while an unrecognized garment stays
unresolved and cannot trigger the wearer's default uniform. The tests also check
that another person's clothing and a uniform mentioned only in the scene are
not assigned to the wearer. Apply the same reasoning to future bug reports;
the example is evidence for the general rule, not a template for another
special-case branch.
