# MortyClaw Architecture

本文档描述 MortyClaw 当前真实架构。它是一个基于 LangGraph 的透明 Agent 运行平台，核心目标是让 Agent 的决策、工具调用、任务调度、记忆召回和运行状态都可追踪。

## 一句话架构

MortyClaw = CLI/UI 入口 + LangGraph Agent 工作流 + SQLite 运行态 + SQLite 记忆层 + 沙盒工具层 + JSONL 监控层。

它当前不是严格意义上的多 Agent 协作系统，而是单 Agent 图工作流。router、planner、approval gate、reviewer 是状态机节点，负责让一个 Agent 的执行过程更透明、更可控。

## 分层结构

| 层级 | 主要文件 | 职责 |
| --- | --- | --- |
| Entry/UI 层 | `entry/cli.py`, `entry/main.py`, `entry/monitor.py` | CLI 命令、交互终端、监控终端 |
| Agent 装配层 | `mortyclaw/core/agent.py` | 组装 LLM、工具、动态技能和 Graph 节点 |
| Graph 编排层 | `mortyclaw/core/runtime_graph.py` | 定义 LangGraph 节点和状态跳转 |
| 策略层 | `routing.py`, `planning.py`, `approval.py`, `prompt_builder.py` | 路由、计划、审批、Prompt 构建 |
| 记忆层 | `memory.py`, `memory_policy.py`, `context.py` | 结构化记忆、FTS、冲突治理、上下文裁剪 |
| 运行态层 | `runtime_store.py`, `heartbeat.py`, `runtime_context.py` | sessions、tasks、task_runs、session_inbox、心跳投递 |
| 工具层 | `tools/builtins.py`, `tools/sandbox_tools.py`, `tools/web_tools.py` | 内置工具、沙盒文件/shell、Tavily、arxiv_rag、summarize |
| 扩展层 | `skill_loader.py`, `workspace/office/skills` | 动态技能加载，help -> run 两段式调用 |
| 观测层 | `logger.py`, `logs/*.jsonl` | 异步 JSONL 审计日志 |

## 主流程

用户在 `mortyclaw run` 的交互终端输入内容后，主进程会将输入送入 LangGraph：

```text
Human input
  -> router
  -> fast path 或 slow path
  -> LLM
  -> tools
  -> LLM final answer
  -> JSONL audit log
```

如果是定时任务触发，输入来源不是键盘，而是 `session_inbox`：

```text
heartbeat
  -> tasks 到期扫描
  -> session_inbox 写入 heartbeat_task
  -> run 进程 inbox_poller 消费
  -> 转成 HumanMessage 输入 Agent
```

## LangGraph 工作流

Graph 定义在 `mortyclaw/core/runtime_graph.py`。

```text
START
  -> router

router
  -> fast_agent
  -> planner

fast_agent
  -> fast_tools
  -> END

planner
  -> approval_gate

approval_gate
  -> slow_agent
  -> END

slow_agent
  -> slow_tools
  -> reviewer

reviewer
  -> slow_agent
  -> approval_gate
  -> planner
  -> END
```

### router

router 的职责是判断任务路径：

- 简单低风险问题走 `fast`。
- 多步骤任务走 `slow`。
- 写入、删除、shell、任务修改等高风险请求走 `slow`。
- arXiv/论文类问题优先直接调用 `arxiv_rag_ask`。

router 同时会做两件和记忆有关的事情：

- 从当前 query 中抽取 session memory。
- 将可能的长期记忆写入异步 memory writer。

### planner

planner 对 slow path 请求做步骤拆分，并给每一步标记 risk level。它不会让模型一次性自由执行整个复杂任务，而是将目标拆成当前步骤，后续由 reviewer 推进。

### approval gate

approval gate 负责高风险确认。只要当前步骤是 high risk，系统会先输出确认提示，要求用户回复“确认执行”或“取消”。没有明确确认时，不进入实际执行。

### fast_agent / slow_agent

两个节点共享同一个 ReAct 执行函数，但 slow path 会限制当前步骤可用工具，避免模型越过当前计划提前执行后续操作。

### reviewer

reviewer 检查 slow path 当前步骤结果：

- 如果看起来失败，先 retry。
- retry 超限后 replan。
- 成功则推进下一步。
- 下一步高风险时重新进入 approval gate。

## 数据存储

| 文件 | 用途 |
| --- | --- |
| `workspace/state.sqlite3` | LangGraph checkpointer，保存会话状态和消息历史 |
| `workspace/runtime.sqlite3` | sessions、tasks、task_runs、session_inbox |
| `workspace/memory/memory.sqlite3` | memory_records 和 FTS5 索引 |
| `workspace/tasks.json` | 旧任务格式兼容镜像，不是主存储 |
| `workspace/memory/user_profile.md` | 人类可读的用户画像快照 |
| `logs/{thread_id}.jsonl` | 每个会话独立审计日志 |

## 工具体系

内置工具集中在 `mortyclaw/core/tools/builtins.py`。工具可以分为：

- 基础工具：时间、计算器、模型信息。
- 任务工具：schedule/list/modify/delete。
- 记忆工具：save_user_profile。
- 文件工具：list/read/write office file。
- shell 工具：execute_office_shell。
- 外部信息工具：Tavily search、arxiv_rag。
- 外部摘要工具：summarize_content，用于网页、PDF、YouTube、Podcast、音频、视频、图片和普通文档；PDF 优先用本地 pypdf 抽取文本，网页等内容由外部 summarize CLI 以 extract-only 模式抽取，最终摘要由 MortyClaw 当前 Agent 完成，代码文件和项目配置文件仍由原有代码分析工具处理。

动态技能由 `skill_loader.py` 扫描 `workspace/office/skills`，将每个技能包装成 LangChain StructuredTool。技能必须先 `mode='help'` 读取说明，再 `mode='run'` 执行命令。

## 安全设计

当前安全边界主要有三层：

- 路由层：高风险请求进入 slow path。
- 审批层：高风险步骤必须用户确认。
- 工具层：写入和 shell 执行限制在 `workspace/office`。

需要注意的是，这仍是个人本地 Agent 沙盒，不是强隔离生产沙盒。若要面向更高安全级别，建议增加容器隔离、命令白名单、文件 diff 预览和权限策略表。

## 可扩展点

- 新工具：在 `mortyclaw/core/tools` 中实现并加入 `BUILTIN_TOOLS`。
- 新技能：放入 `workspace/office/skills/<skill>/SKILL.md`。
- 新模型：扩展 `provider.py`。
- 新路由策略：扩展 `routing.py`。
- 新审批策略：扩展 `approval.py`。
- 新记忆抽取策略：扩展 `memory_policy.py`。
- 新运行态数据：扩展 `runtime_store.py`。

## 当前边界

- 不是严格多 Agent。当前是单 Agent 图编排。
- FTS5 是轻量检索，不是 embedding 语义检索。
- 监控是终端日志流，不是完整 Web dashboard。
- UI 已有 `/sessions` 和 `/tasks`，但记忆、inbox、task_runs 还没有完整交互式管理界面。
