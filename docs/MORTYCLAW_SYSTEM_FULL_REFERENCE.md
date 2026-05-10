# MortyClaw 系统全景说明书

复盘时间：2026-05-10
复盘范围：`/mnt/A/hust_chp/hust_chp/Project/MortyClaw` 当前源码快照
目标读者：需要接手、审计、扩展或调试 MortyClaw Agent 的工程师

> 说明：仓库里已有 `ARCHITECTURE.md`、`RUNTIME.md`、`MEMORY.md` 等文档，其中部分内容描述的是较早的单 Agent 架构。本说明书以当前源码为准，覆盖当前实现中的运行图、上下文工程、prompt 工程、记忆系统、工具系统、tool result 持久化、fast/slow 双链路、程序化执行、子 Agent 编排、存储、观测与测试面。

## 1. 系统定位

MortyClaw 是一个本地优先的工程型 Agent 运行时。它把 LangGraph 的状态图、LangChain 的工具绑定、多模型 Provider、SQLite 持久化、项目文件工具、任务调度、长期记忆、上下文压缩、子 Agent 编排、程序化工具 DSL 和终端 UI 组合成一个可以在本地项目上持续工作的 Agent。

系统的核心设计目标是：

- 在普通对话、联网检索、论文问答、代码阅读、代码修改、测试执行和复杂任务拆解之间自动切换执行链路。
- 通过 `fast`、`slow autonomous`、`slow structured` 三种主执行模式控制成本、风险和可解释性。
- 使用明确的 permission mode、approval gate、execution guard、tool scope、write scope 和 path lock 降低危险操作风险。
- 使用动态上下文、会话摘要、长期记忆、上下文裁剪和 auto compact 控制长会话上下文膨胀。
- 使用 blocking `delegate_subagents` 把可并行的探索、验证、实现分派给 worker，并让主 Agent 直接消费 compact worker summaries。
- 使用 `execute_tool_program` 在需要连续工具调用、循环、批处理或读改测链路时降低 LLM 往返次数。

## 2. 技术栈

### 2.1 Python 包和入口

- 包名：`mortyclaw`
- 版本：`1.0.0`
- Console script：`mortyclaw = mortyclaw.entry.cli:main`
- 主要入口：
  - `mortyclaw/entry/cli.py`：Typer CLI 命令。
  - `mortyclaw/entry/main.py`：交互式终端主循环。
  - `mortyclaw/core/agent/app.py`：创建 LangGraph 应用。
  - `mortyclaw/core/runtime/graph.py`：声明运行时状态图。

### 2.2 主要依赖

项目使用的主要依赖包括：

- CLI / TUI：`typer`、`questionary`、`rich`、`prompt_toolkit`
- 环境配置：`python-dotenv`
- Agent / Graph：`langchain-core`、`langchain`、`langgraph`、`langgraph-checkpoint-sqlite`
- Provider：`langchain-openai`、`langchain-anthropic`、`langchain-community`
- 官方 SDK：`openai`、`anthropic`
- 数据库：`aiosqlite`、SQLite、FTS5
- 数据建模：`pydantic`
- 文档解析：`pypdf`
- 可选 token 估算：`tiktoken`

### 2.3 LLM Provider

`mortyclaw/core/provider.py` 支持多种模型后端：

- OpenAI：`OPENAI_API_KEY`、`OPENAI_API_BASE`
- 阿里云 DashScope/OpenAI-compatible：`ALIYUN_API_KEY`、`DASHSCOPE_API_KEY`、`ALIYUN_BASE_URL`、`DASHSCOPE_BASE_URL`
- Z.AI/智谱兼容接口：`ZAI_API_KEY`、`ZAI_BASE_URL`
- 腾讯兼容接口：`TENCENT_API_KEY`、`TENCENT_BASE_URL`
- Anthropic：`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`
- Ollama：`OLLAMA_BASE_URL`，默认 `http://localhost:11434`
- 其他 OpenAI-compatible Provider：通过 base URL 和 API key 接入。

主聊天模型和轻量路由/规划模型可以分离。路由分类模型默认来自配置，当前默认轻量模型是 `qwen3.5-flash` 风格的 fast classifier 配置。

## 3. 目录和数据平面

### 3.1 关键路径

配置集中在 `mortyclaw/core/config.py`。

- `WORKSPACE_DIR`：默认 `<repo>/workspace`，可用 `MORTYCLAW_WORKSPACE` 覆盖。
- `DB_PATH`：`workspace/state.sqlite3`，LangGraph checkpoint。
- `RUNTIME_DB_PATH`：`workspace/runtime.sqlite3`，会话、任务、worker、tool program、conversation store。
- `MEMORY_DB_PATH`：`workspace/memory/memory.sqlite3`，长期/会话记忆。
- `OFFICE_DIR`：`workspace/office`，Office 文件和可写沙盒。
- `SKILLS_DIR`：`workspace/office/skills`，动态 skill。
- `TASKS_FILE`：`workspace/tasks.json`，旧任务文件兼容层。
- `LOGS_DIR`：`<repo>/logs`，JSONL 事件日志。
- `RUNTIME_ARTIFACTS_DIR`：`workspace/runtime/artifacts`，超长 tool result 持久化文件。
- `workspace/code_index.sqlite3`：项目代码索引数据库。

### 3.2 SQLite 数据库

MortyClaw 的数据平面由多个 SQLite 文件组成：

- `state.sqlite3`：LangGraph checkpoints 和 writes。
- `runtime.sqlite3`：
  - `sessions`
  - `tasks`
  - `task_runs`
  - `session_inbox`
  - `conversation_messages`
  - `conversation_tool_calls`
  - `conversation_messages_fts`
  - `conversation_summaries`
  - `worker_runs`
  - `tool_program_runs`
- `memory.sqlite3`：
  - `memory_records`
  - `memory_records_fts`
- `code_index.sqlite3`：
  - `files`
  - `symbols`
  - `calls`
  - `imports`

## 4. CLI 和交互入口

### 4.1 `mortyclaw config`

`config` 命令通过交互式问题配置 provider、model、API key、base URL，并写入 `.env`。它是用户首次使用系统时的配置入口。

### 4.2 `mortyclaw run`

`run` 是主交互入口，重要参数包括：

- `--thread-id`：指定会话 thread。
- `--new` / `--new-session`：创建新会话。
- `--branch-from`：从历史消息分支出新会话。

`entry/main.py` 启动后会：

1. 初始化 prompt toolkit 终端 UI。
2. 打开 `AsyncSqliteSaver(DB_PATH)`。
3. 调用 `create_agent_app` 创建 LangGraph app。
4. 初始化或恢复 session。
5. 启动 inbox poller。
6. 读取用户输入或后台任务消息。
7. 为每个用户 turn 创建 `turn_id` 和 `HumanMessage`。
8. 写入 conversation store。
9. 调用 `app.astream(..., stream_mode="updates")` 流式运行图。
10. 将 AI message、tool call、tool result、runtime event 写入日志和数据库。

内置交互命令包括：

- `/sessions`
- `/tasks`
- `/new`
- `/clear`
- `/reset`
- `/exit`

### 4.3 Inbox 和后台任务

`session_inbox` 是后台任务和异步通知进入主会话的桥。

- inbox poller 大约每 1 秒检查当前 thread 的 pending event。
- heartbeat 任务到期后会向 inbox 写入 `heartbeat_task` 内容。
- worker 完成后也会写入 parent session inbox，但当前主链路优先通过 blocking `delegate_subagents` 的返回值消费 worker summary。

### 4.4 Monitor

`mortyclaw monitor` 读取 logs JSONL，渲染 runtime overview。它关注的事件包括：

- `llm_input`
- `tool_call`
- `tool_result`
- `ai_message`
- `tool_call_adjusted`
- `system_action`
- worker event
- tool program event
- path lock event

## 5. LangGraph 总执行链路

### 5.1 图节点

核心图在 `mortyclaw/core/runtime/graph.py`。

节点包括：

- `router`
- `planner`
- `approval_gate`
- `execution_guard`
- `fast_agent`
- `fast_tools`
- `slow_agent`
- `slow_tools`
- `reviewer`
- `finalizer`

### 5.2 图边

当前状态图的主要流向是：

```text
START -> router

router:
  fast    -> fast_agent
  planner -> planner
  slow    -> approval_gate

planner:
  fast -> fast_agent
  slow -> approval_gate

approval_gate:
  execute -> execution_guard
  end     -> END

execution_guard:
  execute -> slow_agent
  replan  -> planner
  end     -> END

fast_agent:
  planner -> planner
  tools   -> fast_tools
  __end__ -> END

slow_agent:
  tools    -> slow_tools
  approval -> approval_gate
  replan   -> planner
  retry    -> slow_agent
  finalize -> finalizer
  reviewer -> reviewer
  end      -> END

reviewer:
  execute  -> slow_agent
  approval -> approval_gate
  replan   -> planner
  finalize -> finalizer
  end      -> END

finalizer -> END
fast_tools -> fast_agent
slow_tools -> slow_agent
```

### 5.3 一个用户 turn 的完整路径

一次典型用户输入会经过下面的链路：

1. CLI 生成 `turn_id`，把用户文本写入 conversation store。
2. LangGraph 从 checkpoint 中恢复 `AgentState`。
3. `router` 同步会话记忆、加载当前项目路径、判断 fast/slow/planner。
4. 如果需要结构化规划，进入 `planner`，生成 1-5 个线性步骤。
5. `approval_gate` 根据 permission mode、风险和 pending tool calls 决定是否需要用户确认。
6. `execution_guard` 校验审批上下文是否仍然有效。
7. `fast_agent` 或 `slow_agent` 调用 ReAct node。
8. ReAct node 构造 prompt bundle、动态上下文、记忆上下文、deferred tool catalog、工具 schema。
9. LLM 返回文本或 tool calls。
10. 如 tool calls 需要审批，写入 pending execution snapshot，转到 approval。
11. ToolNode 执行工具。
12. tool result 进入 artifact 持久化和预算裁剪。
13. fast 模式回到 fast ReAct；slow structured 模式进入 reviewer；slow autonomous 可直接 final 或继续工具循环。
14. `reviewer` 判断当前 step 成功、失败、重试、replan 或 finalize。
15. `finalizer` 汇总计划、todo、验证结果、剩余问题并结束。
16. 所有关键输入输出写入 checkpoint、runtime DB 和 JSONL event log。

## 6. AgentState 状态模型

`mortyclaw/core/runtime/state.py` 中的 `AgentState` 是运行图的共享状态，重要字段包括：

- `messages`：LangChain message 历史。
- `summary`：压缩后的会话摘要。
- `working_memory`：当前 turn 的短期状态快照。
- `route`、`goal`、`complexity`、`risk`
- `planner_required`、`route_locked`、`route_source`、`route_reason`、`route_confidence`
- `plan`、`current_step_index`、`step_results`
- `plan_source`、`replan_reason`
- `approval_status`、`approval_request`、`permission_mode`
- `pending_tool_calls`、`pending_execution_snapshot`
- `retry`、`run_status`
- `todos`
- `slow_execution_mode`
- worker 相关字段
- tool program 相关字段
- `current_project_path`
- subdirectory context hints
- compact generation / compact reason
- `final_answer`

`build_working_memory_snapshot` 会把关键状态和最近 3 个 step result 组合成一个轻量快照，给后续 prompt 和运行时节点使用。

## 7. Fast / Slow 双链路

MortyClaw 的执行不是单一路径，而是由 router 和 planner 在多个链路之间切换。

### 7.1 Fast Path

Fast path 用于简单、低风险、偏查询或只读的请求。

特点：

- 不进入 planner/reviewer/finalizer。
- 直接进入 `fast_agent`。
- 绑定工具较少，排除高风险和复杂写入链路。
- 适合普通问答、简单计算、时间查询、简单网页检索、只读项目分析、论文直连问答。
- 如果模型在 fast path 中产生破坏性 tool call，会被升级到 slow/planner。

### 7.2 Slow Autonomous Path

Slow autonomous 用于需要工具、风险较高或多步但不需要严格 step review 的任务。

特点：

- 进入 `approval_gate` 和 `execution_guard`。
- ReAct node 自主决定工具调用序列。
- 可以使用 todo、项目读写、测试、命令、tool program、subagent。
- 对高风险工具按 permission mode 触发审批。
- 输出可以直接作为 final answer，也可以在需要时进入 finalizer。

### 7.3 Slow Structured Path

Slow structured 用于复杂、多阶段、需要明确计划和验证的任务。

特点：

- 先经过 planner 生成线性 plan。
- 每次只执行一个 current step。
- tool scope 按 current step 的 intent 和 execution mode 裁剪。
- step 完成后由 reviewer 判定成功、失败、重试、replan 或 finalize。
- finalizer 统一汇总所有 step result、todo 和验证结果。

### 7.4 Programmatic 和 Delegated 是慢链路内的执行方式

`execute_tool_program` 和 `delegate_subagents` 不是独立 route，而是 slow 链路中的执行工具：

- `execute_tool_program`：适合连续工具链、循环、批处理、读改测。
- `delegate_subagents`：适合结构上独立、可并行的探索、验证或实现子任务。

## 8. 路由系统

### 8.1 Router 的职责

`mortyclaw/core/runtime/nodes/router.py` 的职责包括：

- 处理 worker isolation mode。
- 检查 pending approval 和 permission selection resume。
- 启动长期记忆捕获。
- 同步会话记忆。
- 从 session memory 中恢复当前项目路径。
- 运行 rule-based route 和 LLM classifier。
- 决定进入 fast、slow autonomous 或 planner。

### 8.2 Rule-Based Route

`mortyclaw/core/routing/rules.py` 负责确定性规则：

- 纯论文/arxiv 查询可以走 fast arxiv direct。
- 论文 + 本地仓库混合分析走 slow structured。
- 明显高风险操作走 slow，通常 route locked。
- 多步任务走 slow。
- 简单能力问答、只读分析和低风险查询走 fast。
- 不确定或复杂任务倾向 planner。

### 8.3 LLM Classifier

`mortyclaw/core/routing/classifier.py` 使用轻量模型输出 JSON 分类。

实现中的重要阈值：

- classifier route 接受阈值大致为 `confidence >= 0.5`。
- planner 降级到 fast 通常要求更高置信度，约 `confidence >= 0.7`。
- 高风险判断会覆盖 fast 倾向。

### 8.4 Web Query 归一化

`mortyclaw/core/routing/web.py` 会在 Tavily 搜索前做查询归一化：

- 识别 `today`、`tomorrow`、`yesterday`、`this week` 等相对时间。
- 注入当前绝对日期。
- 推断 topic：`general` 或 `news`。
- 将调整写入 `tool_call_adjusted` 事件。

## 9. Planner 工程

Planner 位于 `mortyclaw/core/runtime/nodes/planner.py` 和 `mortyclaw/core/planning/*`。

### 9.1 Planner 输出

Planner 要求 LLM 输出 JSON，包括：

- `route`
- `goal`
- `reason`
- `confidence`
- `steps`

Plan 约束：

- step 数量为 1-5。
- step 必须线性可执行。
- 不生成“询问用户”作为 step。
- 每个 step 包含 intent、execution mode、risk、success criteria、verification hint、needs tools。

### 9.2 Step Intent

常见 intent 包括：

- `analyze`
- `read`
- `paper_research`
- `code_edit`
- `file_write`
- `shell_execute`
- `test_verify`
- `summarize`
- `report`

### 9.3 Execution Mode

Planner 可为 step 选择：

- `structured`：普通慢链路单步执行。
- `programmatic`：用 `execute_tool_program` 批量执行工具链。
- `delegated`：用 `delegate_subagents` 并行委派。

### 9.4 Planner Fallback

如果 LLM planner 失败，规则 fallback 会按中文和英文连接词拆分用户请求，形成保守的线性计划。

### 9.5 当前实现注意点

配置里存在 `ENABLE_DYNAMIC_CONTEXT_FOR_PLANNER`，动态上下文 builder 也支持 planner compact 模式；但当前 planner 调用中传入 LLM 的 `dynamic_context_text` 为空。也就是说，rich dynamic context 主要在 ReAct node 中生效，planner 侧当前没有完整使用动态上下文。

## 10. Approval、Permission 和 Execution Guard

### 10.1 Permission Mode

MortyClaw 支持三种 permission mode：

- `ask`：默认模式。破坏性或高风险工具需要用户确认。
- `plan`：只读/计划模式。阻止 destructive tools。如果任务本身必须写入、测试或运行命令，可能直接终止或要求切换权限。
- `auto`：自动批准允许的破坏性操作，但仍阻止 `execute_office_shell`。

常见确认别名：

- yes：`确认`、`继续`、`同意`、`批准`、`可以`、`yes`、`ok`、`y`
- no：`取消`、`停止`、`不用了`、`算了`、`no`、`n`

### 10.2 Approval Gate

`mortyclaw/core/runtime/nodes/approval.py` 根据当前 state 和 pending tool calls 生成审批请求。审批消息会包含：

- 当前计划或当前 step。
- active todo。
- pending tool names。
- 风险说明。
- permission mode 建议。

### 10.3 Execution Guard

`mortyclaw/core/runtime/execution_guard.py` 在审批后真正执行前做二次校验，防止审批上下文漂移。

校验对象包括：

- pending tool snapshot 的 approval context hash。
- project root 是否存在。
- project root hash/path hash 是否一致。
- patch 工具的 `git apply --check`。
- tool program run 是否仍处于 `awaiting_approval`。
- tool program locals hash、base snapshot 是否一致。

如果校验失败，会进入 replan 或 end，而不是直接执行旧审批下的危险操作。

## 11. Reviewer 和 Finalizer

### 11.1 Reviewer

`mortyclaw/core/runtime/nodes/reviewer.py` 在 slow structured 模式下评估当前 step。

它会读取：

- 最新 AIMessage。
- 最新 ToolMessage。
- 工具返回中的 `mortyclaw_step_outcome`。
- 工具错误中的 `mortyclaw_error`。

它的输出可能是：

- 当前 step 成功，进入下一 step。
- 当前 step 失败但可重试，回到 slow_agent。
- 当前 step 失败且需要 replan。
- 当前任务完成，进入 finalizer。
- 失败终止。

默认 retry 上限约为 2 次。

### 11.2 Finalizer

`mortyclaw/core/runtime/nodes/finalizer.py` 负责统一汇总：

- goal
- plan steps
- step results
- todo 状态
- 验证命令和结果
- 剩余问题
- 最终回答

finalizer 结束时会清理 session todo 状态，避免下一个用户 turn 继承过期任务。

## 12. ReAct Node

`mortyclaw/core/agent/react_node.py` 是系统最重的运行节点。它连接了工具、上下文、prompt、记忆、压缩、错误恢复和审批。

### 12.1 ReAct Node 前置处理

每次进入 ReAct node 时会执行：

1. 处理最近 tool results 的持久化和 turn 预算。
2. 更新 active todo。
3. 读取 session memory 和 long-term memory。
4. 判断 arxiv passthrough 是否可以直接 final。
5. 根据 fast/slow/current step 构造 active tool scope。
6. 处理 deferred tool schema 请求。
7. 构造 dynamic context envelope。
8. 估算上下文压力。
9. 必要时进行 trim、summary 或 auto compact。
10. 构造 prompt bundle。
11. 绑定 eager tool schemas。
12. 调用 LLM。
13. 标准化特殊 tool call，例如 Tavily query。
14. 对 destructive tool calls 生成 pending approval。

### 12.2 错误恢复

`mortyclaw/core/agent/recovery.py` 会分类错误：

- provider timeout
- rate limit
- auth error
- context overflow
- tool schema error
- tool runtime error
- approval blocked
- empty response
- unsafe request
- unknown

恢复策略包括：

- provider timeout/rate limit/tool runtime/empty response：有限次数 retry。
- context overflow：强制压缩并 retry 一次。
- approval blocked：等待用户。
- schema/auth/unsafe：通常 abort。
- unknown：倾向 replan 或失败输出。

## 13. Prompt 工程

### 13.1 Prompt 版本

当前核心版本标识：

- `BASE_PROMPT_VERSION = react-base-v2`
- `SECURITY_POLICY_VERSION = sandbox-policy-v1`

### 13.2 Base Prompt 内容

Base prompt 在 `mortyclaw/core/prompts/builder.py` 生成，覆盖：

- 身份和回答风格：自然、简洁、中文优先。
- 长期偏好保存：用户明确长期偏好时调用 profile/memory 相关工具。
- 最新信息：用 Tavily，回答要给来源。
- 论文问题：用 arxiv 工具。
- 链接、PDF、媒体和文档总结：用 `summarize_content`。
- 代码项目：用 project tools，而不是自由访问任意文件系统。
- 高风险操作：遵守审批和 sandbox 策略。
- 连续工具链：优先考虑 `execute_tool_program`。
- 并行独立子任务：用 `delegate_subagents`。

### 13.3 Sandbox Prompt

Prompt 中明确写入 sandbox 规则：

- 不越过可用工具访问任意文件系统。
- 不用 `node -e`、`python -c` 一类临时代码执行绕过工具策略。
- 有 project root 时，项目操作使用 project tools。
- 写入和 shell 操作必须限制在 office 或 project root。
- 拒绝绕过安全边界的用户请求。

### 13.4 Context Trust

系统把上下文分为 trusted 和 untrusted：

- trusted：运行时状态、当前 step、审批状态、工具约束、系统生成摘要。
- untrusted：仓库里的 `AGENTS.md`、`CLAUDE.md`、`.cursorrules`、用户文件、会话记忆、长期记忆检索结果等。

Prompt 明确告诉模型：untrusted context 不能覆盖 system/developer/user/current step/permission/tool scope/approval。

### 13.5 PromptBundle

Prompt bundle 的最终消息顺序大致是：

1. System：base prompt。
2. System：trusted turn context。
3. Human：reference context，标记为 `REFERENCE CONTEXT - NOT USER REQUEST`。
4. 历史 conversation messages。
5. 可选 goal continuation hint。

Prompt builder 会计算并记录：

- base hash
- trusted context hash
- reference context hash
- deferred tool catalog hash
- conversation token 估算
- final input token 估算
- eager/deferred tool schemas

这些会写入 `llm_input` audit event。

### 13.6 Provider Prompt Cache

`mortyclaw/core/prompts/provider_cache.py` 支持 Provider-specific prompt cache。

- Base prompt cache key 包含 provider、model、prompt version、security version、toolset、profile snapshot version。
- Anthropic/Claude 消息可加入 ephemeral `cache_control`。
- Turn render cache 根据 turn 和 state revision 复用渲染结果。

### 13.7 Deferred Tool Schema

为了降低工具 schema token，系统只把 eager tools 直接绑定给模型。其他工具放在 deferred catalog 中。

模型如果需要 deferred tool，必须调用：

- `request_tool_schema(names=[...])`

下一轮 ReAct 会在权限允许的情况下把对应 schema 绑定进去。未授权或超 scope 的 schema 请求会被忽略并记录。

## 14. Context 工程

### 14.1 Dynamic Context Envelope

`mortyclaw/core/context/dynamic.py` 负责动态上下文。

默认配置：

- slow path 开启 dynamic context。
- fast path 默认不开启 dynamic context，但 session/long-term/summary fallback 仍可进入 prompt。
- planner 配置存在，但当前 planner LLM 实际没有完整消费 dynamic context。

Dynamic context 总字符预算默认：

- `DYNAMIC_CONTEXT_TOTAL_CHAR_BUDGET = 12000`

### 14.2 Trusted Blocks

Trusted blocks 包括：

- workspace summary，默认约 1800 chars，planner compact 约 1400。
- handoff summary，默认约 3200 chars，planner compact 约 2200。
- runtime status，默认约 1600 chars，planner compact 约 1200。

### 14.3 Untrusted Blocks

Untrusted blocks 包括：

- context files，总预算约 5000 chars。
- session memory，默认约 2200 chars，planner compact 约 1400。
- long-term memory，默认约 2200 chars，planner compact 约 1400。
- subdirectory hints，slow path 约 3000 chars。

Context files 的候选包括：

- `.bytecode.md`
- `BYTECODE.md`
- `AGENTS.md`
- `agents.md`
- `CLAUDE.md`
- `claude.md`
- `.cursorrules`
- `.cursor/rules/*.md`

### 14.4 Trusted Turn Context

`render_trusted_turn_context` 的硬上限约 3500 chars，内容包括：

- route/goal/risk/permission/run status。
- 当前 project path。
- pending approval 信息。
- execution guard 或 tool program 状态。
- compact generation。
- hard session constraints。
- active todo。
- active summary excerpt。
- trusted anchor blocks。
- slow autonomous 或 current structured step 的执行规则。

### 14.5 Context Safety

`mortyclaw/core/context/safety.py` 会扫描 untrusted context：

- 去除隐藏 Unicode。
- 检测 prompt override。
- 检测 system/developer 指令伪装。
- 检测 secret exfiltration。
- 检测 jailbreak。
- 检测 role spoofing。

如果 `CONTEXT_BLOCK_ON_THREATS` 为 true，命中威胁的 context 会被替换为“source blocked”说明，原内容不进入 prompt。

### 14.6 Subdirectory Hints

`mortyclaw/core/context/subdirectory_hints.py` 会在项目工具调用后，根据被访问路径沿父目录向上读取局部 context 文件。

默认行为：

- 最多向上 5 层。
- 保留最近约 12 个 hint block。
- root context 预加载，避免重复。
- 只在 slow path 中作为 untrusted context 注入。

## 15. 上下文压缩、裁剪和阈值

### 15.1 Token 预算

核心配置在 `mortyclaw/core/config.py` 和 `mortyclaw/core/context/window.py`。

默认值：

- `CONTEXT_COMPRESSION_BUDGET_TOKENS = 250000`
- 最小预算：`120000`
- `CONTEXT_LAYER2_TRIGGER_RATIO = 0.7`
- `CONTEXT_LAYER3_TRIGGER_RATIO = 0.7`
- `CONTEXT_TRIM_KEEP_TOKENS = 220000`
- `CONTEXT_OVERFLOW_KEEP_TOKENS = 120000`
- `CONTEXT_NON_MESSAGE_RESERVE_TOKENS = 80000`
- `CONTEXT_SUMMARY_TIMEOUT_SECONDS = 8`

因为 layer2 和 layer3 默认都为 0.7，所以上下文压力达到约 70% 时会直接进入高压处理语义。

### 15.2 Token 估算

系统优先使用 `tiktoken`。不可用时，使用 UTF-8 byte fallback。

估算时会加入 message overhead：

- 普通 message overhead 约 8。
- tool message 额外约 12。

### 15.3 Pressure 计算

上下文压力大致由以下部分构成：

- message token 估算。
- dynamic context / memory / reference context token 估算。
- deferred tool catalog token 估算。
- 非 message reserve，默认 80000 tokens。

pressure = `(messages + extra_text + reserve) / budget`

### 15.4 Trim 逻辑

当 pressure 达到 medium/high 时，ReAct node 会调用 `trim_context_messages`。

默认保留策略：

- 保留第一个 system message。
- 保护开头 3 条非 system message。
- 保护 tail，至少约 8 条。
- 永远保护最新 human message。
- 对齐 AI tool call 和 ToolMessage，避免断裂。
- 缺失 tool result 时插入 repair stub。
- 老旧且非 tail 的 ToolMessage 如果超过约 480 chars，会被压缩为 `[compressed-tool-result]` preview，除非已经持久化 artifact。

### 15.5 Summary

被裁掉的消息会进入摘要流程：

- 优先用 LLM 总结，超时约 8 秒。
- LLM 失败时使用结构化 fallback 提取。
- 摘要写入 state `summary`。
- 摘要也写入 `conversation_summaries`。

### 15.6 Auto Compact

auto compact 的条件比较严格：

- 当前已经有新 summary。
- 没有 pending approval。
- 没有 pending tool calls。
- 没有 pending execution snapshot。
- 最新 human message 有 id。
- 普通模式下 pressure 必须 high。
- overflow retry 模式下只要满足安全条件即可。

auto compact 会删除旧 messages，只保留最新用户输入和摘要状态，并递增 `compact_generation`。

### 15.7 Context Overflow Retry

如果 Provider 报 context overflow：

1. 强制 trim 到 `CONTEXT_OVERFLOW_KEEP_TOKENS = 120000` 附近。
2. 尝试 summary。
3. 满足条件则 auto compact。
4. 最多 retry 一次。

## 16. Tool Result 过长处理

`mortyclaw/core/runtime/tool_results.py` 负责 tool result 持久化和 turn 内预算控制。

### 16.1 默认阈值

- `DEFAULT_RESULT_THRESHOLD = 9000` chars
- `DEFAULT_PREVIEW_CHARS = 2400` chars
- `MAX_TURN_BUDGET_CHARS = 24000` chars

### 16.2 按工具覆盖阈值

当前实现中的典型阈值：

- `run_project_command`：7000 chars
- `run_project_tests`：7000 chars
- `show_git_diff`：7000 chars
- `search_project_code`：8000 chars
- `summarize_content`：8000 chars
- `tavily_web_search`：8000 chars
- `edit_project_file`：12000 chars
- `read_project_file`：14000 chars
- `read_office_file`：14000 chars
- `write_project_file`：14000 chars

### 16.3 持久化格式

超过阈值的 tool result 会写入：

```text
workspace/runtime/artifacts/{thread_id}/{turn_id}/{tool_call_id}.txt
```

进入 message history 的内容会变为 `<persisted-output>` block，包含：

- artifact path。
- 原始长度。
- preview。
- `mortyclaw_artifact` metadata。

### 16.4 Turn Budget

如果一个 turn 中所有 ToolMessage 总字符数超过约 24000 chars，系统会从最大的非持久化 tool result 开始继续持久化，直到降回预算内。

## 17. 记忆系统

### 17.1 记忆层级

MortyClaw 的记忆分为：

- working memory：当前运行图状态中的短期快照。
- session memory：当前会话或项目范围内的偏好和上下文。
- long-term memory：跨会话长期记忆。

长期和会话记忆存储在 `workspace/memory/memory.sqlite3`。

### 17.2 Memory Record

`memory_records` 的核心字段包括：

- `memory_id`
- `layer`
- `scope`
- `type`
- `subject`
- `content`
- `source_kind`
- `source_ref`
- `created_at`
- `updated_at`
- `confidence`
- `status`

状态包括：

- `active`
- `superseded`
- `archived`
- `expired`
- `deleted`

同一 scope/type/subject 的 active long-term 记忆 upsert 时会 supersede 旧记录。

### 17.3 FTS 和中文检索

记忆库使用 FTS5：

- tokenizer 为 unicode61。
- 针对中文会做 bigram term expansion。
- FTS 不可用或无命中时 fallback 到 LIKE 搜索。

### 17.4 Session Memory

每个用户 turn 中，router 会同步 session memory。

当前可抽取的 session memory 包括：

- 当前项目路径。
- 回答语言偏好，例如中文。
- 代码修改策略，例如“不要修改代码，只分析”。
- 操作工作区，例如 office。
- 高风险操作审批偏好。

session memory 使用确定性 id，例如：

```text
session::{scope}::{type}
```

注入 prompt 时默认最多约 5 条。

### 17.5 Long-Term Capture

长期记忆不是每句话都捕获。只有用户输入带有长期偏好信号时才触发，例如：

- `记住`
- `以后`
- `一直`
- `长期`
- `我喜欢`
- `我不喜欢`
- `my preference`

捕获前会走 memory safety，阻止：

- prompt injection
- secret exfiltration
- API key/SSH key 等敏感秘密
- 明显恶意持久化指令

长期记忆类型包括：

- `user_preference`
- `project_fact`
- `workflow_preference`
- `safety_preference`

默认 confidence 约 0.8，写入由 async writer 完成。

### 17.6 Long-Term Recall

长期检索也不是 always-on。只有当前 query 有记忆相关提示时才召回，例如：

- `记住`
- `偏好`
- `喜欢`
- `习惯`
- `之前`
- `以前`
- `还记得`
- `根据我的`
- `我的设置`
- `我的风格`
- `profile`
- `preference`
- `remember`

召回策略：

- 读取 profile snapshot。
- 搜索 long-term FTS，默认 limit 约 4。
- 无 FTS 命中时 fallback latest。
- `MemoryPromptCache` LRU 约 128。

### 17.7 User Profile

`save_user_profile` 会维护 `workspace/memory/user_profile.md`，并将 profile snapshot upsert 为 long-term memory 记录。

### 17.8 Memory Provider 抽象

源码中存在 `MemoryProvider`、`BuiltinMemoryProvider` 和 manager 抽象；主运行链路当前主要仍使用 built-in memory bridge 函数。

## 18. 工具系统总览

工具注册在 `mortyclaw/core/tools/builtins/registry.py` 和 `mortyclaw/core/tools/builtins/*`。

### 18.1 常用内置工具

通用工具：

- `get_current_time`
- `calculator`
- `request_tool_schema`
- `get_system_model_info`

研究工具：

- `tavily_web_search`
- `arxiv_rag_ask`
- `summarize_content`

记忆工具：

- `save_user_profile`

Office 工具：

- list/read/write office files
- `execute_office_shell`

项目工具：

- `read_project_file`
- `write_project_file`
- `edit_project_file`
- `apply_project_patch`
- `search_project_code`
- `show_git_diff`
- `run_project_tests`
- `run_project_command`

程序化和子 Agent：

- `execute_tool_program`
- `delegate_subagent`
- `delegate_subagents`
- `wait_subagents`
- `list_subagents`
- `cancel_subagents`

任务和会话：

- task schedule/list/delete/modify
- `search_sessions`
- `update_todo_list`

### 18.2 Tool Policy

`mortyclaw/core/agent/tool_policy.py` 会根据 route、permission、project root、current step 和 slow mode 裁剪工具。

规则要点：

- fast path 不提供 `update_todo_list`。
- destructive tools 在 `plan` 模式下被阻止。
- `auto` 模式仍阻止 `execute_office_shell`。
- 有 project root 且 slow autonomous 时，会移除 office write/shell，避免绕过项目边界。
- current step 会进一步限制 tool scope。
- deferred tool schema 请求也要通过同一套 scope 检查。

### 18.3 Destructive Tools

典型 destructive tools 包括：

- `edit_project_file`
- `write_project_file`
- `apply_project_patch`
- office write
- `run_project_tests`
- `run_project_command`
- `execute_office_shell`
- `execute_tool_program`
- `delegate_subagent`
- `delegate_subagents`

## 19. 项目文件工具

项目工具集中在 `mortyclaw/core/tools/project/*`。

### 19.1 Project Root

项目 root 通过 markers 推断，例如：

- `.git`
- `pyproject.toml`
- `setup.py`
- `requirements.txt`
- `package.json`
- `Cargo.toml`
- `go.mod`
- `pom.xml`
- `manage.py`
- `README.md`
- `src`
- `tests`
- `docs`
- `agents`
- `web`
- `app`
- `pkg`

如果路径外层只是 wrapper，内部唯一子目录才有 markers，系统会规范到真实项目 root。

### 19.2 安全边界

项目工具只能访问 project root 内部。

敏感文件会被拒绝，例如：

- `.env`
- `.npmrc`
- `.pypirc`
- `id_rsa`
- `id_dsa`
- `id_ecdsa`
- `id_ed25519`
- `.key`
- `.pem`
- `.p12`
- `.pfx`

排除目录包括：

- `.git`
- `.hg`
- `.mypy_cache`
- `.pytest_cache`
- `.ruff_cache`
- `.tox`
- `.venv`
- `__pycache__`
- `build`
- `dist`
- `logs`
- `node_modules`
- `rick`
- `workspace`

### 19.3 读取和搜索

`read_project_file`：

- 默认 `start_line = 1`。
- 默认最多约 240 行。
- `max_lines` 被限制在 1-1000。
- 单文件读取上限约 2MB。
- 输出默认最多约 12000 chars，超过由 tool result 层持久化。

`search_project_code` 支持模式：

- `text`
- `symbol`
- `callers`
- `dependencies`
- `data_flow`
- `entrypoints`

文本搜索优先使用 `rg`。结构化搜索使用 `workspace/code_index.sqlite3`。

### 19.4 写入和 patch

`write_project_file`：

- 整文件写入。
- 新文件必须显式 `create_if_missing`。
- 支持 `expected_hash`。
- 使用 path lock。

`edit_project_file`：

- 支持 `old_text` 精确替换。
- 支持 start/end line 替换。
- 支持 `expected_hash`。
- mismatch 时返回 recovery JSON。

`apply_project_patch`：

- 接收 unified/git diff。
- 校验路径。
- 先做 `git apply --check`。
- 支持 dry run。
- 使用 path lock。
- 返回 changed paths 和 diff stat。

### 19.5 测试和命令

`run_project_tests`：

- shell 模式，但严格白名单。
- 默认如果有 tests 目录则跑 unittest discover，否则 py_compile 前 50 个 Python 文件。
- timeout 5-1800 秒，默认约 180 秒。
- 阻止 `rm`、`mv`、`sudo`、`git reset`、`git checkout`、`git clean`、`python -c`、`node -e`、重定向、分号和 pipe-to-shell 等。

`run_project_command`：

- 不使用 shell，基于 shlex。
- 阻止 shell metachar：`& ; | > < \` $ ( )` 等。
- 阻止 `rm`、`mv`、`sudo`、`bash`、`sh`、`zsh`、`fish` 等。
- 允许安全模式，例如 py_compile、unittest、pytest、ruff、mypy、npm test/build、pnpm/yarn、uv、tox、make test、`rick python` 等。
- timeout 5-1800 秒。

## 20. Office 工具和动态 Skills

### 20.1 Office 工具

Office 工具围绕 `workspace/office`。

- list/read 可以读取 office 内文件；绝对只读路径也有兼容支持。
- write 只能写 office 内。
- `execute_office_shell` 在 office 目录执行，timeout 默认约 60 秒。
- 阻止 `..`、Unix root、`~`、Windows drive/root 等路径逃逸。
- stdout/stderr 输出会截断尾部约 2000 chars。
- `auto` permission mode 仍然阻止 `execute_office_shell`。

### 20.2 Dynamic Skills

动态 skill 位于：

```text
workspace/office/skills/<skill>/SKILL.md
```

或使用 README 作为 fallback。

Skill loader 会提取：

- `name`
- `description`

生成的 skill tool 支持：

- `mode=help`：返回 skill 文档前约 3000 chars。
- `mode=run`：用 `execute_office_shell` 执行命令，并把 `{baseDir}` 替换为 skill 目录。

因此 skill 执行仍受 office shell 安全策略约束。

## 21. Web、论文和总结工具

### 21.1 Tavily

`tavily_web_search`：

- 需要 `TAVILY_API_KEY`。
- `topic` 支持 `general` 和 `news`。
- `search_depth` 支持 `basic` 和 `advanced`。
- `max_results` 限制在 1-10。
- URL fetch timeout 约 30 秒。

### 21.2 arxiv RAG

`arxiv_rag_ask`：

- 调用 `ARXIV_RAG_API_BASE` 或 `FEISHU__API_BASE_URL`。
- 默认本地 `localhost:8001`。
- 路径 `/api/v1/feishu/reply`。
- timeout 可配置，默认约 60 秒。
- 返回带 `_mortyclaw_passthrough` 的 JSON 时，ReAct node 可以直接把结果作为 final answer。

### 21.3 summarize_content

`summarize_content` 支持：

- URL
- local file
- PDF
- 文本文件

限制和行为：

- 拒绝代码/config 扩展和敏感文件名。
- 相对本地路径限制在 office。
- PDF 下载 timeout 约 60 秒。
- PDF/下载大小上限约 50MB。
- 本地文本上限约 5MB。
- 外部 summarize 命令 timeout 约 300 秒。
- 最终输出约 12000 chars。
- 媒体二进制本地文件不直接总结。

## 22. `execute_tool_program` 程序化执行

`execute_tool_program` 是为连续工具链设计的小型安全 DSL，位于 tools/program 相关实现。

### 22.1 启用和用途

默认启用：

- `ENABLE_EXECUTE_TOOL_PROGRAM = true`

适合：

- 批量读取多个文件。
- 搜索后逐个读取。
- 循环执行验证。
- 简单读改测链路。
- 需要减少 LLM 多轮工具调用开销的任务。

### 22.2 SDK Alias

DSL 暴露的常见别名：

- `read_file`
- `search_code`
- `show_diff`
- `edit_file`
- `write_file`
- `apply_patch`
- `run_tests`
- `run_command`
- `emit_result`
- `update_todo`

### 22.3 安全 AST

允许的 builtins：

- `len`
- `str`
- `int`
- `bool`
- `list`
- `dict`
- `range`
- `enumerate`
- `min`
- `max`
- `sum`
- `sorted`

允许的 AST 类型包括：

- module
- assign
- augassign
- expr
- if
- for
- names/constants
- list/tuple/dict
- binop/compare/bool/unary
- subscript/slice
- call/keyword/attribute
- pass
- 简单 list comprehension

禁止：

- import
- function/class definition
- while
- with
- lambda
- del
- global/nonlocal
- raise
- yield
- await

### 22.4 执行限制

- `max_steps` 默认约 40，限制在 1-200。
- wall time 默认约 60 秒，限制在 5-600 秒。
- trace 只保留最近约 12 条返回。

### 22.5 审批恢复

如果 program 中遇到 destructive tool alias：

1. 抛出 `ProgramPauseRequested`。
2. 将 program run 写入 `tool_program_runs`，状态 `awaiting_approval`。
3. 保存 pc、locals、staged tool calls、stdout、normalized IR、metadata。
4. 返回 `needs_approval`。
5. 写入 `pending_execution_snapshot.kind = tool_program`。
6. 用户审批后由 `execution_guard` 校验 approval hash、locals hash、base snapshot。
7. 校验通过后从 pause 点恢复。

## 23. 子 Agent / Worker 编排

### 23.1 总体定位

MortyClaw 的 worker system 用于把独立子任务并行委派出去。当前推荐入口是 blocking `delegate_subagents`，而不是 spawn-only + wait/list 的旧链路。

### 23.2 配置

默认配置：

- `ENABLE_WORKER_SUBAGENTS = true`
- `WORKER_MAX_CONCURRENCY = 4`
- `WORKER_MAX_BATCH_SIZE` 默认等于 concurrency
- `WORKER_DEFAULT_TIMEOUT_SECONDS = 180`

当前源码中 concurrency 由 supervisor 的 executor/信号量约束；`WORKER_MAX_BATCH_SIZE` 是配置项，但 batch submit 路径中没有明显看到同等强度的显式批量数量裁剪逻辑，因此实际治理重点是并发和 timeout。

### 23.3 Worker Roles

支持角色：

- `explore`：探索、阅读、分析。
- `verify`：验证、测试、复查。
- `implement`：实现、修改文件。

默认工具：

- explore：`read_project_file`、`search_project_code`、`show_git_diff`、`update_todo_list`
- verify：explore 工具 + `run_project_tests`、`run_project_command`
- implement：verify 工具 + `edit_project_file`、`write_project_file`、`apply_project_patch`

### 23.4 Toolsets

worker task 可声明 toolsets：

- `project_read`
- `project_write`
- `project_verify`
- `research`
- `project_full`

实际工具集合还会与父 Agent 当前 active tool scope 相交，避免 worker 获得父链路没有授权的工具。

### 23.5 禁用工具

worker 默认会移除高风险或不适合嵌套的工具：

- `delegate_subagent`
- `delegate_subagents`
- `wait_subagents`
- `list_subagents`
- `cancel_subagents`
- `execute_tool_program`
- `request_tool_schema`
- `save_user_profile`
- `search_sessions`
- task tools

### 23.6 `delegate_subagents`

当前 `delegate_subagents` 的输入 task schema 包括：

- `task`
- `role`
- `toolsets`
- `allowed_tools`
- `write_scope`
- `context_brief`
- `deliverables`
- `timeout_seconds`
- `priority`

要求：

- 每个 worker 必须有 `context_brief`。
- 每个 worker 必须有 `deliverables`。
- `implement` worker 必须提供 `write_scope`。

### 23.7 Blocking 返回

`delegate_subagents` 不再是普通 spawn-only 工具。它内部会等待 worker 完成、失败、超时或取消，然后返回 compact summaries。

返回结构包括：

- `success`
- `status`：`completed`、`partial`、`timeout`、`failed`
- `batch_id`
- `worker_ids`
- `workers`
- `completed_count`
- `failed_count`
- `timeout_count`
- `cancelled_count`
- `retry_policy = do_not_auto_retry`
- `debug_only_tools`
- `next_action_hint`

每个 worker summary 只保留主 Agent 汇总需要的信息：

- `summary`
- `changed_files`
- `commands_run`
- `tests_run`
- `blocking_issue`
- `summary_truncated`

### 23.8 Worker 运行模式

explore/verify direct worker：

- 使用隔离 LLM 循环。
- 最多约 6 轮。
- 最多约 10 次 tool call。
- 最多约 8 次 file read。
- 只允许授权工具。
- 注入 project root。

implement worker：

- 使用完整 LangGraph app。
- 开启 worker isolation mode。
- implement/verify 默认 permission mode 为 auto。
- explore 默认偏 plan/read-only。
- slow execution mode 通常为 autonomous。
- 带 parent/batch/task metadata。

### 23.9 Worker 状态

worker status 包括：

- `pending`
- `running`
- `completed`
- `failed`
- `cancelled`
- `timeout`

timeout 使用 `asyncio.wait_for`，会写入 `status="timeout"`，并尽量保留 partial summary、changed files、commands、tests 和 blocking issue。

cancel 是协作式取消：

- 对 running worker 写入 cancel metadata。
- worker event loop 每次 update 后检查 cancel。
- 不强杀正在阻塞的 LLM/tool 调用。

### 23.10 Path Lock

implement worker 如果带有 `write_scope` 和 project root，会通过 `ProjectPathLockManager` 获取路径锁，降低并行写入冲突。

项目 edit/write/patch 工具也会使用 path lock。锁持有者可能是：

- active program
- worker
- thread

## 24. Todo 系统

`update_todo_list` 和 runtime todo state 用于让慢链路维护任务进度。

特征：

- todo 可以被 ReAct node、tool program 和 worker 更新。
- slow structured plan 的 step 完成会同步 todo。
- finalizer 会清理 session todo。
- active todo 会进入 trusted turn context。

## 25. 会话搜索和历史召回

`search_sessions` 使用 conversation store 和 FTS 检索历史会话。

能力：

- 按关键词搜索历史 message。
- 默认排除当前 lineage，避免把当前会话重复召回。
- 可选择 include current。
- 可用 route classifier LLM 对命中片段做总结。
- 总结 timeout 可配置，默认约 45 秒。
- limit 通常 1-5。

`session_recall.py` 会按 lineage root 聚合历史材料：

- 最大 summary sessions 约 3。
- material budget 约 20000 chars。

## 26. Handoff Summary

`mortyclaw/core/context/handoff.py` 生成跨 turn 的结构化 handoff summary。

格式是 JSON v1，主要字段：

- `goal`
- `active_task`
- `completed_steps`
- `pending_steps`
- `files_touched`
- `commands_run`
- `tool_results`
- `todos`
- `context_notes`
- `open_questions`
- `risks`
- `last_user_intent`

默认限制：

- steps：约 6
- files：约 8
- commands：约 6
- tool results：约 6
- notes：约 6
- risks：约 6
- open questions：约 4
- events：约 32
- paths：约 6

handoff summary 不复制长日志，只保留 command、path、result preview、artifact path 等摘要信息。

## 27. 代码索引

`mortyclaw/core/code/index.py` 负责项目代码索引。

默认限制：

- 单文件索引上限约 2MB。
- 最大索引文件数约 3000。
- 排除目录和敏感文件策略与项目工具相近。

当前主要支持 Python AST：

- files
- symbols
- calls
- imports
- entrypoint score

entrypoint scoring 规则示例：

- 文件名含 main/train/run/finetune：加分。
- 引入 argparse/click/typer/hydra：加分。
- main guard：高权重加分。
- 定义 main/train/fit/run/cli：加分。
- 调用 fit/train/Trainer/main：加分。

`search_project_code` 的 symbol、callers、dependencies、data_flow、entrypoints 模式会使用该索引。

## 28. 存储、日志和维护

### 28.1 Conversation Store

conversation writer 是异步队列，写入：

- message
- tool call
- tool result
- FTS 索引
- session 统计
- session title

它支持中文 bigram 搜索，并维护 message count、tool call count 等 session 元数据。

### 28.2 Heartbeat 和任务调度

heartbeat 会周期性处理到期任务：

1. 从 legacy `tasks.json` 导入或同步任务。
2. 查找 due tasks。
3. 向目标 session 的 inbox 写入 `heartbeat_task`。
4. 记录 `task_run`。
5. 对 repeat task 推进下次运行时间。

pacemaker loop 默认间隔约 10 秒。

### 28.3 Maintenance

维护命令包括：

- `doctor`：检查 DB、日志、表数量和状态。
- `gc logs`：默认保留约 14 天或 max 5MB，旧日志归档 tar.gz。
- `gc runtime`：清理 delivered inbox、旧 task runs 等。
- `gc state`：每个 thread 默认保留最新约 30 个 checkpoints，并先备份。
- `reset_thread_state`：删除指定 thread 的 checkpoints/writes。

## 29. Observability

`mortyclaw/core/observability/audit.py` 负责 JSONL 事件日志。

日志事件覆盖：

- LLM 输入，包括 prompt hash、token stats、cache 信息、tool schema 信息。
- tool call。
- tool result。
- AI message。
- query/tool call adjusted。
- system action。
- worker lifecycle。
- tool program lifecycle。
- path lock lifecycle。

这些事件是 `mortyclaw monitor`、问题复盘和性能分析的主要数据来源。

## 30. 安全模型

MortyClaw 的安全边界由多层共同构成：

1. Prompt 层：sandbox policy、context trust、tool discipline。
2. Routing 层：高风险任务强制 slow/approval。
3. Permission 层：ask/plan/auto。
4. Tool scope 层：根据 route/current step 裁剪工具。
5. Deferred schema 层：未授权工具 schema 不直接暴露。
6. Project tool 层：project root 限制、敏感文件拒绝、命令白名单。
7. Office tool 层：office sandbox 和路径逃逸检测。
8. Execution guard 层：审批后执行前复核。
9. Path lock 层：并行写入冲突控制。
10. Context safety 层：untrusted context 注入检测。
11. Memory safety 层：长期记忆写入前检测敏感和恶意内容。

## 31. 测试面

仓库测试覆盖多个子系统，典型测试文件包括：

- `tests/test_agent.py`
- `tests/test_approval.py`
- `tests/test_builtins.py`
- `tests/test_cli.py`
- `tests/test_config_and_skill_loader.py`
- `tests/test_context_advanced.py`
- `tests/test_conversation_store.py`
- `tests/test_dynamic_context.py`
- `tests/test_handoff_summary.py`
- `tests/test_heartbeat.py`
- `tests/test_maintenance.py`
- `tests/test_memory.py`
- `tests/test_planner_prompt.py`
- `tests/test_planning_rules.py`
- `tests/test_program_runtime.py`
- `tests/test_project_tools.py`
- `tests/test_prompt_layers.py`
- `tests/test_sandbox_tools.py`
- `tests/test_summarize_tool.py`
- `tests/test_two_phase_skills.py`
- `tests/test_worker_tools.py`

推荐按变更范围运行测试。当前项目环境中常见命令是：

```bash
./rick/bin/python -m unittest tests/test_worker_tools.py
./rick/bin/python -m unittest tests/test_prompt_layers.py tests/test_dynamic_context.py tests/test_memory.py
./rick/bin/python -m unittest
```

## 32. 关键阈值速查

| 子系统 | 参数 | 默认值/行为 |
| --- | --- | --- |
| Dynamic Context | total char budget | 12000 |
| Dynamic Context | context files | 5000 chars |
| Dynamic Context | subdirectory hints | 3000 chars |
| Trusted Turn Context | hard cap | 约 3500 chars |
| Context Compression | budget | 250000 tokens |
| Context Compression | minimum budget | 120000 tokens |
| Context Compression | layer2 trigger | 0.7 |
| Context Compression | layer3 trigger | 0.7 |
| Context Trim | keep tokens | 220000 |
| Context Overflow | keep tokens | 120000 |
| Context Reserve | non-message reserve | 80000 |
| Summary | timeout | 8 秒 |
| Tool Result | default threshold | 9000 chars |
| Tool Result | preview | 2400 chars |
| Tool Result | turn budget | 24000 chars |
| Project Read | default max lines | 240 |
| Project Read | max lines | 1000 |
| Project Read | max file bytes | 2MB |
| Project Command | timeout | 5-1800 秒 |
| Project Tests | default timeout | 约 180 秒 |
| Tavily | max results | 1-10 |
| Summarize | local text max | 5MB |
| Summarize | PDF/download max | 50MB |
| Summarize | external timeout | 300 秒 |
| Worker | max concurrency | 4 |
| Worker | default timeout | 180 秒 |
| Direct Worker | max rounds | 6 |
| Direct Worker | max tool calls | 10 |
| Direct Worker | max file reads | 8 |
| Tool Program | default max steps | 40 |
| Tool Program | max steps | 200 |
| Tool Program | default wall time | 60 秒 |
| State GC | checkpoints kept | 每 thread 约 30 |
| Logs GC | retention | 约 14 天或 max 5MB |

## 33. 当前实现的几个重要注意点

- `delegate_subagents` 当前已经是 blocking summary 返回路径；`wait_subagents` 和 `list_subagents` 主要用于兼容、调试或异常场景。
- worker result 仍会写 parent inbox，但主 Agent 正常不应再依赖 inbox/wait/list 获取结果。
- `WORKER_MAX_BATCH_SIZE` 是配置项，但当前源码中实际更明显生效的是并发上限和 timeout。
- planner dynamic context 的配置存在，但当前 planner LLM 调用未完整使用 dynamic context 文本。
- long-term memory 不是每轮必召回；它依赖用户 query 中的记忆相关信号。
- fast path 不是“完全不用工具”，它可以用低风险工具和只读项目工具；一旦出现 destructive tool call，会升级到 slow。
- auto compact 不会在 pending approval、pending tool calls 或 pending execution snapshot 存在时运行。
- context files、session memory、long-term memory、subdirectory hints 都是 untrusted context，不能覆盖系统指令。
- `execute_office_shell` 即使在 auto permission 下也被阻止。
- 项目命令和测试命令有白名单；不要把它理解成任意 shell。

## 34. 源码阅读地图

如果要继续深入，建议按下面顺序读源码：

1. `mortyclaw/entry/main.py`：终端主循环和 turn 生命周期。
2. `mortyclaw/core/agent/app.py`：app 创建、工具和 provider 绑定。
3. `mortyclaw/core/runtime/graph.py`：LangGraph 节点和边。
4. `mortyclaw/core/runtime/state.py`：AgentState。
5. `mortyclaw/core/runtime/nodes/router.py`：路由、记忆同步、权限恢复。
6. `mortyclaw/core/runtime/nodes/planner.py`：规划。
7. `mortyclaw/core/agent/react_node.py`：ReAct 主执行节点。
8. `mortyclaw/core/prompts/builder.py`：prompt bundle。
9. `mortyclaw/core/context/dynamic.py` 和 `mortyclaw/core/context/window.py`：上下文注入、裁剪和压缩。
10. `mortyclaw/core/agent/memory_bridge.py`、`mortyclaw/core/memory/store.py`：记忆写入和检索。
11. `mortyclaw/core/runtime/tool_results.py`：超长 tool result 持久化。
12. `mortyclaw/core/tools/project/*`：项目文件、搜索、命令和测试工具。
13. `mortyclaw/core/tools/builtins/workers.py` 和 `mortyclaw/core/runtime/worker_supervisor.py`：子 Agent 编排。
14. `mortyclaw/core/tools/builtins/programs.py` 及 runtime program 相关文件：程序化执行 DSL。
15. `mortyclaw/core/storage/*`：runtime DB、conversation store、worker/tool program store。
16. `mortyclaw/core/observability/audit.py`：事件日志。

## 35. 一句话总览

MortyClaw 当前已经不是一个简单 ReAct Agent，而是一个带路由、规划、审批、工具裁剪、上下文预算、长期记忆、程序化执行、blocking worker 编排、持久化 runtime 和监控面的本地工程 Agent 系统。理解它时，最重要的是把“LLM 调用”看作其中一个节点，而不是整个系统；真正的行为由运行图、状态、工具策略、上下文工程和安全治理共同决定。
