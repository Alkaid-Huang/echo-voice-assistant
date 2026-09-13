# 编码 Agent 能力集成设计

> 版本：v0.1 ｜ 更新日期：2026-09-13
> 目标：把 coding agent 的六项能力**长进 Echo**，而不是另起项目。

---

## 1. 定位调整

Echo 从"Agent 驱动的 Live2D 伴侣"扩展为：

> **一套 Agent 内核 + 两组工具包（陪伴 / 编码）+ 可替换的身体（Live2D）**

| 部分 | 内容 | 状态 |
|------|------|------|
| 内核 | 工具调用循环、事件流、记忆、预算控制 | ✅ 已有（M2） |
| 陪伴工具包 | 时间、天气、记住、回忆 | ✅ 已有 |
| 编码工具包 | `view` / `search` / `edit` / `run`（在目标仓库内工作） | 🆕 本设计 |
| 身体 | Live2D 呈现契约与渲染 | ⏳ M3 |
| 感官 | VAD / ASR / TTS | ✅ 已有（M1） |
| 保障 | 沙箱、权限、回滚、变更日志 | 🆕 本设计 |
| 评测 | 任务集 + 指标报表（pass@1、回归率、越界率…） | 🆕 本设计 |

**两种用法**：陪伴（聊天 / 查天气 / 记事）与**编码**（"帮我看下 tests 为什么挂"→ 检索 → 编辑 → 跑测试 → 播报结果），
共用事件流与评测体系。语音 + Live2D 让编码过程"可见可说"，这是与纯 CLI 编码 Agent 的差异点。

---

## 2. 模块布局（新增部分）

```
src/echo/agent/
  agent.py             # 内核循环（现有；演进为：工具包注入 + 任务状态 + 预算）
  tools/
    base.py            # ToolSpec / ToolRegistry（由 tools.py 迁入）
    companion.py       # 陪伴工具包：时间 / 天气 / 记住 / 回忆
    coding.py          # 编码工具包：view / search / edit / run
  sandbox.py           # 沙箱与权限策略（worktree / none / docker 预留）
  journal.py           # 变更日志：每次改动 = diff + 理由（可审计）
  task_state.py        # 任务状态机 + 检查点（planning→editing→testing→done/failed）

src/echo/index/
  scanner.py           # 仓库遍历与忽略规则（.venv / models / outputs / .git）
  chunker.py           # 符号级切片（先用 ast，后补 tree-sitter）
  bm25.py              # 词面检索（纯 Python 实现，零依赖）
  retrieval.py         # 检索接口与混合检索（BM25 + 向量 → RRF 融合 → 可选 rerank）
  context.py           # 上下文预算与压缩（token 配额 + 摘要 + 行号锚点）

src/echo/eval/
  tasks.py             # 任务集：真实 bug 复现 + 注入 bug
  runner.py            # 逐任务执行、收集指标
  report.py            # 生成 benchmarks/codeagent_results.md
```

**不变量**（与项目既有规范一致）：每个模块只依赖接口；沙箱与检索都通过配置注入；
所有动作发事件（`thinking` / `tool_call` / `tool_result` / `diff` / `test_result` / `speech`）。

---

## 3. 六项能力 → 落地模块 → 度量

| 能力 | 落地模块 | 关键技术 | 度量 |
|------|---------|---------|------|
| 仓库理解 | `index/` | 符号级切片 + BM25→混合检索 + 上下文预算 | recall@k、上下文 token 数、定位准确率 |
| 工具调用（ACI） | `agent/tools/coding.py` | 四个正交工具、唯一匹配校验、编辑后语法检查 | 工具成功率、编辑应用率、无效编辑数 |
| 反思修复 | `agent/agent.py` + `task_state.py` | 测试驱动循环、失败输出回灌、策略切换 | pass@1、平均轮次、回归引入率 |
| 状态与记忆 | `task_state.py` + `journal.py` | 状态机、变更日志（diff+理由）、检查点 | 可恢复性、重复犯错率、长任务成功率 |
| 安全 | `sandbox.py` | worktree 隔离、路径/命令白名单、回滚 | 拦截次数、越界修改数、回滚成功率 |
| 评测 | `eval/` | 任务集 + 自动判定 + 报表 | 通过率、diff 质量、漏洞引入、token 成本 |

---

## 4. 安全策略（硬性红线）

1. **路径**：所有文件操作解析为绝对路径后必须校验在 `workspace_root` 内；`protected_paths`（`.git`/`.venv`/`models`/`outputs`）只读。
2. **命令**：白名单前缀（如 `python -m pytest`、`ruff`、`git diff`、`git status`）；
   `rm` / `git reset --hard` / `curl | bash` / 包安装 一律**默认拒绝**，需显式开关授权。
3. **网络**：默认关闭；需要时按域名白名单开放。
4. **回滚**：每次写操作前打检查点；`rollback` 一键回到检查点；不可逆操作先 dry-run 并输出将执行的内容。
5. **隔离**：默认在**临时 git worktree** 中工作（不污染主工作区）；Docker 作为后续强化选项。
6. **审计**：每次工具调用与每个 diff 写入 `journal`（时间、工具、参数摘要、结果、理由）。

---

## 5. 分阶段计划（学习与实现同步）

每个阶段产出一份**学习笔记**（概念 → 参照系统怎么做的 → 我们怎么实现 → 实测数字），
笔记放学习仓库，代码与设计放产品仓库。

| 阶段 | 实现内容 | 学习笔记主题 | 交付与度量 | 预计 |
|------|---------|------------|-----------|------|
| **P1 工具与沙箱** | `tools/coding.py` 四工具 + `sandbox.py` 权限策略 + worktree 隔离 | ACI 设计（SWE-agent）：工具粒度、编辑格式、编辑后校验 | 单测通过；越界/危险命令被拦截的测试 | 2–3 天 |
| **P2 检索与上下文** | `index/`：scanner→chunker→bm25→retrieval→context | 仓库级检索：BM25 与向量、RRF 融合、上下文预算 | recall@k 对比（无检索 vs BM25 vs 混合）；token 消耗对比 | 2–3 天 |
| **P3 循环与状态** | 测试驱动循环 + 反思 + 预算上限 + `task_state`/`journal` + 检查点回滚 | 反思与修复（Reflexion / Agentless 三段式）、长任务状态管理 | 3 个真实 bug 任务跑通；平均轮次统计 | 2–3 天 |
| **P4 评测** | `eval/`：10 个注入 bug + 3 个真实 bug；指标报表 | 编码 Agent 评测：pass@1、回归率、diff 质量、成本 | `benchmarks/codeagent_results.md` 报表 | 2 天 |
| **P5 进阶** | 混合检索接入向量 + RRF + rerank；Docker 沙箱；Semgrep 安全扫描 | 上下文压缩（LLMLingua 思路）、容器隔离、漏洞引入检测 | 检索/压缩消融实验；安全扫描对比 | +3–5 天 |

---

## 6. 评测设计

**任务集**（`eval/tasks.py`）

- 真实 bug 复现：CUDA 降级、silero Tensor/采样率、批处理编码（历史提交里有原样修复）
- 注入 bug：在副本仓库里用脚本注入 10 个（类型错、边界条件、路径处理、并发竞态等），每个附一条失败测试与一句现象描述

**指标**（`eval/runner.py`）

| 指标 | 定义 |
|------|------|
| pass@1 | 一次提交后目标测试通过的比例 |
| 回归引入率 | 原本通过的测试被弄坏的比例 |
| 平均轮次 / 工具调用数 | 效率 |
| token 消耗 / 墙钟时间 | 成本 |
| 越界修改次数 | 改动 `workspace_root` 之外的文件数（期望 0） |
| 危险操作拦截数 | 被策略拒绝的命令数 |
| diff 质量 | 改动文件数、行数、lint 是否通过 |

**报表**：写入 `benchmarks/codeagent_results.md`，格式与延迟基线一致（环境 / 任务 / 数值 / 结论）。

---

## 7. 与现有代码的关系（复用点）

| 复用 | 说明 |
|------|------|
| Agent 循环 | `agent.py` 的 `thinking → tool_call → tool_result` 结构直接沿用，只换工具包与提示词 |
| 工具注册表 | `ToolRegistry` 已支持"按名调用 + 异常转文本"，新增 `coding` 工具包即可 |
| 事件流 | 复用 `on_event`，新增 `diff` / `test_result` 事件供 Live2D 与前端展示 |
| 配置体系 | 新增 `coding_config`，继续用 Pydantic 校验 |
| 测试体系 | 沙箱、检索、编辑工具全部可用 mock（假文件系统、假命令执行器） |
| 基准体系 | 评测报表并入 `benchmarks/`，与延迟基线同一套写法 |

---

## 8. 面试话术

> "我的 Agent 内核是通用的：同一套循环、事件流、配置与评测，挂两组工具包——
> 陪伴工具（时间/天气/记忆）和编码工具（检索/编辑/跑测试）。
> 编码侧我按 ACI 的思路设计工具（少而正交、编辑后立刻校验、匹配不唯一就拒绝），
> 用 git worktree 做隔离、路径与命令白名单做权限、检查点做回滚、journal 做审计；
> 评测用 10 个注入 bug + 3 个真实 bug，报 pass@1、回归引入率、越界修改数等指标。
> 语音与 Live2D 让整个过程可见可说，这是它和纯 CLI 编码 Agent 的区别。"
