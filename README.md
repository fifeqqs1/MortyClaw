<div align="center">

![MortyClaw Logo](docs/morty_logo.png)

# MortyClaw

### 当 AI 开始“黑箱操作”，你需要一双透视眼

[![MortyClaw](https://img.shields.io/badge/MortyClaw-1.0.0-purple.svg?logo=cyberpunk)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-blue.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-blue.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**透明、可控、带记忆的个人 Agent 终端**

[快速开始](#快速开始) · [核心能力](#核心能力) · [架构文档](docs/ARCHITECTURE.md) · [记忆系统](docs/MEMORY.md) · [运行时系统](docs/RUNTIME.md)

</div>

---

## 项目定位

MortyClaw 是一个基于 LangGraph 的透明 Agent 运行平台。它不是单纯聊天机器人，而是把“模型推理、工具调用、风险审批、任务调度、会话隔离、记忆检索、运行监控”放在同一套可观察架构里。

当前架构可以概括为：

- **单 Agent + 图工作流编排**：router / planner / approval gate / reviewer 组成可追踪状态机。
- **分层记忆**：working memory、session memory、long-term memory，并支持 FTS5 检索和冲突治理。
- **SQLite 运行态**：sessions、tasks、task_runs、session_inbox 统一存储到 `workspace/runtime.sqlite3`。
- **透明监控**：每个会话写独立 JSONL 日志，`monitor` 可按 thread_id 查看实时事件流。
- **安全工具层**：写入和 shell 执行限制在 `workspace/office`，高风险任务进入审批流程。

更详细的设计说明见：

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：整体架构、模块职责、Graph 流程。
- [docs/MEMORY.md](docs/MEMORY.md)：分层记忆、FTS 检索、冲突治理、缓存策略。
- [docs/RUNTIME.md](docs/RUNTIME.md)：SQLite 表、会话、任务、heartbeat、monitor 流程。

## 核心能力

| 能力 | 当前实现 | 价值 |
| --- | --- | --- |
| 透明 Agent 编排 | LangGraph 状态机，fast/slow path 分流 | 可定位每一步决策来源 |
| 高风险审批 | 写文件、执行命令、删除/修改任务等进入 approval gate | 避免 Agent 擅自执行危险操作 |
| 分层记忆 | working/session/long_term 三层记忆 | 既保留当前任务状态，也能记住长期偏好 |
| 记忆检索 | SQLite FTS5 + 中文二元词增强 + fallback LIKE | 不引入向量库也能低延迟召回相关记忆 |
| 记忆冲突治理 | 同 scope/type/subject 的长期记忆自动 supersede 旧记录 | 避免“以后用中文”和“以后用英文”同时生效 |
| 多会话 | `thread_id` 隔离状态、日志、任务和 inbox | 不同会话互不污染 |
| 定时任务 | SQLite tasks + 独立 heartbeat + session_inbox 投递 | 主程序和调度器解耦，重启可恢复 |
| 监控终端 | JSONL 审计日志 + Rich 渲染 | 看得到 LLM 输入、工具调用、AI 回复、系统动作 |
| 技能扩展 | 自动加载 `workspace/office/skills/*/SKILL.md` | 可挂载外部技能，使用 help -> run 两段式 |

## 快速开始

### 安装

```bash
cd MortyClaw
pip install -e .
```

### 配置模型

```bash
mortyclaw config
```

也可以手动编辑 `.env`：

```bash
DEFAULT_PROVIDER=aliyun
DEFAULT_MODEL=glm-5
# 可选：单独指定 planner / route-classifier 轻量模型
# ROUTE_CLASSIFIER_MODEL=qwen3.5-flash
ALIYUN_API_KEY=sk-your-qwen-key
ALIYUN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 如果你还要同时保留 OpenAI / GPT 配置，可以并存：
OPENAI_API_KEY=sk-your-openai-key
OPENAI_API_BASE=https://api.ikuncode.cc/v1
```

支持的 provider 包括 `openai`、`anthropic`、`aliyun`、`tencent`、`z.ai`、`ollama`、`other`。

### 启动交互终端

```bash
mortyclaw run
```

指定会话：

```bash
mortyclaw run --thread-id local_geek_master
mortyclaw run --new
```

交互界面内置快捷命令：

| 命令 | 作用 |
| --- | --- |
| `/sessions` | 在当前聊天界面中查看最近会话、状态、模型和最后活跃时间 |
| `/tasks` | 在当前聊天界面中查看当前会话的待执行任务 |
| `/exit` 或 `/quit` | 退出当前交互终端 |

### 启动监控终端

```bash
mortyclaw monitor --latest
mortyclaw monitor --thread-id local_geek_master
mortyclaw monitor --list-sessions
```

### 启动 heartbeat

```bash
mortyclaw heartbeat
mortyclaw heartbeat --interval 5
mortyclaw heartbeat --once
```

heartbeat 会扫描到期任务，将事件写入 `session_inbox`。对应会话的 `mortyclaw run` 进程会消费 inbox，并把到期任务作为系统内部输入交给 Agent 处理。

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `mortyclaw config` | 交互式配置模型 provider、model、API key 和 base url |
| `mortyclaw run` | 启动默认会话 `local_geek_master` |
| `mortyclaw run --thread-id <id>` | 启动或恢复指定会话 |
| `mortyclaw run --new` | 创建短编号新会话，例如 `session-1`、`session-2` |
| `mortyclaw run --new-session` | `--new` 的兼容别名 |
| `mortyclaw monitor --latest` | 监控最近活跃会话 |
| `mortyclaw monitor --thread-id <id>` | 监控指定会话 |
| `mortyclaw sessions` | 在 CLI 查看会话列表 |
| `mortyclaw heartbeat` | 启动独立心跳进程 |
| `mortyclaw migrate-tasks` | 将旧 `tasks.json` 导入 SQLite |

## 功能模块

### Agent 工作流

MortyClaw 的核心工作流在 `mortyclaw/core/runtime_graph.py` 中定义：

```text
START
  -> router
  -> fast_agent -> fast_tools -> fast_agent -> END
  -> planner -> approval_gate -> slow_agent -> slow_tools -> slow_agent -> reviewer
  -> reviewer -> approval / replan / execute / END
```

router 判断任务是否简单、高风险或多步骤；planner 将复杂任务拆成步骤；approval gate 对高风险步骤请求确认；reviewer 检查步骤结果，必要时重试或重规划。

### 记忆系统

记忆存储在 `workspace/memory/memory.sqlite3`，核心表是 `memory_records` 和 `memory_records_fts`。长期记忆类型包括：

- `user_preference`
- `project_fact`
- `workflow_preference`
- `safety_preference`

长期记忆通过 `type + subject` 做冲突治理。例如“以后用中文回答”和“以后用英文回答”会被归到 `user_preference / response_language`，新的 active 记录会 supersede 旧记录。

### 任务和会话系统

运行态数据库在 `workspace/runtime.sqlite3`，包含：

- `sessions`：会话元数据、状态、模型、日志路径。
- `tasks`：定时任务主表。
- `task_runs`：每次任务触发记录。
- `session_inbox`：跨进程投递事件，heartbeat 通过它把到期任务交给 run 进程。

旧版 `workspace/tasks.json` 仍保留为兼容镜像，但主存储已经是 SQLite。

### 工具和技能

内置工具包括：

- 时间和计算：`get_current_time`、`calculator`
- 任务管理：`schedule_task`、`list_scheduled_tasks`、`modify_scheduled_task`、`delete_scheduled_task`
- 记忆画像：`save_user_profile`
- 文件和 shell：`list_office_files`、`read_office_file`、`write_office_file`、`execute_office_shell`
- 联网和论文：`tavily_web_search`、`arxiv_rag_ask`
- 外部内容摘要：`summarize_content`，用于网页、PDF、YouTube、Podcast、音频、视频、图片和普通文档；PDF 优先用本地 `pypdf` 抽取文本，网页等内容用外部 CLI 抽取，最终摘要由 MortyClaw 自己完成，不处理代码或项目配置文件
- 系统信息：`get_system_model_info`

动态技能从 `workspace/office/skills` 加载。每个技能目录需要包含 `SKILL.md` 或 `README.md`，运行时会被包装成 `mode='help'` / `mode='run'` 两段式工具。

### 安全边界

MortyClaw 当前的安全策略是“个人本地 Agent 沙盒”：

- 写文件和 shell 执行只能在 `workspace/office` 内。
- 读取工具允许用户明确给出的绝对路径，但只读。
- shell 命令有路径越权正则拦截和 60 秒超时。
- 高风险步骤进入 slow path 并请求用户确认。

这套策略适合本地开发和个人使用；如果要生产部署，建议进一步加入容器隔离、命令白名单和更严格的权限模型。

## 项目结构

```text
MortyClaw/
├── entry/
│   ├── cli.py                 # Typer CLI: config/run/monitor/heartbeat/sessions/migrate-tasks
│   ├── main.py                # 交互终端、会话启动、inbox 消费
│   └── monitor.py             # Rich 监控终端
├── mortyclaw/core/
│   ├── agent.py               # Agent 装配器
│   ├── runtime_graph.py       # LangGraph 节点和边
│   ├── routing.py             # fast/slow/arxiv/Tavily 路由策略
│   ├── planning.py            # 计划拆分、风险判断、工具作用域
│   ├── approval.py            # 高风险确认
│   ├── prompt_builder.py      # System prompt、上下文摘要 prompt
│   ├── memory.py              # SQLite 记忆存储、FTS、冲突治理
│   ├── memory_policy.py       # 记忆抽取、召回、Prompt cache
│   ├── runtime_store.py       # sessions/tasks/task_runs/session_inbox
│   ├── heartbeat.py           # 到期任务扫描和 inbox 投递
│   ├── logger.py              # 异步 JSONL 审计日志
│   ├── provider.py            # 多模型 provider 适配
│   ├── skill_loader.py        # 动态技能加载
│   └── tools/                 # 内置工具
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MEMORY.md
│   └── RUNTIME.md
├── tests/                     # unittest 回归测试
├── workspace/                 # 本地运行态数据，默认不提交
└── logs/                      # 会话 JSONL 日志，默认不提交
```

## 测试

当前测试使用 `unittest`：

```bash
./rick/bin/python -m unittest discover -s tests -q
```

测试覆盖 Agent 路由、审批、内置工具、沙盒工具、上下文裁剪、heartbeat、SQLite runtime、记忆缓存、FTS 检索和冲突治理。

## 当前边界

MortyClaw 目前不是严格的多 Agent 系统，而是“单 Agent + 多节点图工作流”。router、planner、reviewer 是状态机节点，不是独立 Agent 实例。

记忆检索当前是 FTS5 轻量检索，不是向量数据库语义检索。优点是简单、低延迟、和 SQLite 架构一致；缺点是跨表达方式的语义泛化不如 embedding 检索。

UI 已支持 `/sessions` 和 `/tasks`，但更完整的 dashboard、记忆管理、任务编辑 UI 仍可继续扩展。

## License

MIT
