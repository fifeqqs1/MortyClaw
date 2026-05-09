# MortyClaw 项目深度分析报告与简历包装文档

> 目标读者：项目作者、简历筛选人、AI Agent / RAG / 工程系统方向面试官。
>
> 文档定位：这不是 README 复述，而是从系统架构、工程实现、Agent workflow、记忆系统、工具安全、科研助手能力、运行态设计、简历包装和面试讲述角度，对 MortyClaw 做一次系统化拆解。

## 0. 核心结论

MortyClaw 是一个基于 LangGraph 的个人 AI Agent 运行平台。它的核心价值不在于“又做了一个聊天机器人”，而在于把 Agent 的几个工程化难点放进了一套统一架构里：

- 可追踪的图工作流：router、planner、approval gate、fast/slow agent、reviewer 组成显式状态机。
- 可控的工具调用：不同路径和不同步骤限制工具作用域，高风险动作需要用户确认。
- 分层记忆系统：working memory、session memory、long-term memory、context summary 分工明确。
- SQLite-first 运行态：对话状态、会话、任务、任务运行、跨进程 inbox、结构化记忆都落到本地 SQLite。
- 科研/代码协作工具：支持项目文件读取、全文搜索、AST 符号/调用/依赖/入口/数据流分析、patch 修改、diff 查看和测试验证。
- 外部科研助手集成：论文类问题直连本地 arxiv_rag 服务，复用其论文问答和会话逻辑。
- 可观察性：每个 thread_id 写独立 JSONL 审计日志，monitor 可以实时观察 LLM 输入、工具调用、工具结果、系统动作和 AI 回复。

一句话包装：

> MortyClaw 是一个面向个人科研与开发场景的透明可控 AI Agent 终端，基于 LangGraph 构建可观测图工作流，集成分层记忆、SQLite 持久化运行态、任务调度、风险审批、代码项目工具、论文 RAG 转发和实时监控，使 Agent 从黑箱聊天变成可恢复、可审计、可扩展的本地智能工作台。

需要明确的边界：

- 它当前不是严格多 Agent 系统，而是“单 Agent + 多节点图工作流”。
- 记忆检索当前是 SQLite FTS5 + BM25 + 中文二元词增强，不是 embedding 向量检索。
- 它是个人本地 Agent 沙盒，不是生产级强隔离沙盒。
- 当前 planner / router / memory extractor 大多是规则驱动，不是完整 LLM planner 或学习型 memory extractor。

### 0.1 事实与推断边界

本文档尽量基于项目源码和现有文档做事实总结。以下内容属于明确实现事实：

- LangGraph 图工作流、fast/slow path、approval gate、reviewer。
- SQLite checkpointer、runtime store、memory store、code index。
- FTS5 记忆检索、中文二元词增强、long-term memory 冲突 supersede。
- sessions/tasks/task_runs/session_inbox/heartbeat 运行态设计。
- project_tools、sandbox_tools、web_tools、summarize_content、dynamic skills。
- JSONL 审计日志和 Rich monitor。

以下内容属于基于实现做出的“产品化/简历包装推断”，不是代码中显式声明的功能名称：

- “本地 Agent 操作系统雏形”：这是对 CLI、运行态、任务、工具、记忆和监控组合形态的概括性包装。
- “科研助手/开发工作台”：这是根据 arxiv_rag、project_tools、summarize_content、code_index 等能力做出的定位推断。
- “适合岗位方向”：这是面向简历和面试的表达建议，不代表项目已有线上用户或生产部署。

## 1. 项目到底做了什么

### 1.1 项目定位

MortyClaw 可以理解为一个“本地 Agent 操作系统雏形”。它把聊天、工具、任务、记忆、审批、监控和代码协作能力组合成一个终端产品。

普通聊天机器人通常只有：

- 用户输入。
- LLM 回复。
- 可选工具调用。
- 简单会话历史。

MortyClaw 多做了几层工程化封装：

- 在 LLM 前面加 router，判断任务复杂度和风险。
- 在复杂任务前加 planner，将目标拆成步骤。
- 在高风险步骤前加 approval gate，要求用户显式确认。
- 在每一步执行后加 reviewer，决定继续、重试、重规划或结束。
- 对工具调用加步骤级作用域约束，避免模型跳步或乱用工具。
- 用 LangGraph checkpoint 持久化消息和状态。
- 用 SQLite runtime 表管理会话、任务、任务运行和跨进程事件。
- 用结构化 memory 表保存 session 和 long-term memory。
- 用 JSONL 日志和 monitor 让用户看到 Agent 到底在做什么。

因此它更接近“Agent runtime + personal research/dev assistant”，而不是单点功能 Demo。

### 1.2 面向的核心场景

从代码和工具设计看，MortyClaw 主要覆盖四类场景：

| 场景 | 典型问题 | MortyClaw 的处理方式 |
| --- | --- | --- |
| 日常问答 | 现在几点、计算、常识问答 | fast path 直接走 LLM 或基础工具 |
| 个人工作台 | 设置提醒、查看任务、恢复会话 | SQLite tasks + sessions + heartbeat + inbox |
| 科研论文助手 | 查论文、解释 arXiv、研究方法对比 | 论文类 query 直连 arxiv_rag_ask |
| 代码/科研项目协作 | 查代码、找入口、修 bug、跑测试 | project_tools + code_index + patch/diff/test |

项目的差异化价值在于这些能力不是散落脚本，而是都接入了同一个 Agent workflow、同一个 thread_id 隔离体系、同一个审计日志体系和同一套安全策略。

## 2. 总体架构

### 2.1 一句话架构

```text
MortyClaw =
  CLI/交互终端
  + LangGraph 图工作流
  + 多模型 Provider 工厂
  + 工具体系和动态技能
  + SQLite Checkpoint
  + SQLite Runtime Store
  + SQLite Memory Store
  + JSONL Monitor
  + 沙盒和审批安全层
```

### 2.2 分层视图

| 层级 | 关键文件 | 作用 |
| --- | --- | --- |
| Entry/UI 层 | `entry/cli.py`, `entry/main.py`, `entry/monitor.py` | 命令行入口、交互终端、实时监控 |
| Agent 装配层 | `mortyclaw/core/agent.py` | 加载模型、工具、技能，组装 graph 节点 |
| Graph 编排层 | `mortyclaw/core/runtime_graph.py` | 定义 LangGraph 节点和边 |
| 策略层 | `routing.py`, `planning.py`, `approval.py`, `prompt_builder.py` | 路由、计划、审批、prompt、上下文摘要 |
| 记忆层 | `context.py`, `memory.py`, `memory_policy.py` | 工作记忆、会话记忆、长期记忆、FTS 检索、Prompt cache |
| 运行态层 | `runtime_store.py`, `heartbeat.py`, `runtime_context.py` | sessions、tasks、task_runs、session_inbox、当前 thread_id |
| 工具层 | `tools/*.py`, `code_index.py` | 内置工具、项目工具、沙盒工具、搜索和摘要工具 |
| 扩展层 | `skill_loader.py`, `workspace/office/skills` | 动态技能加载，help -> run 两阶段协议 |
| 观测层 | `logger.py`, `logs/*.jsonl`, `entry/monitor.py` | 异步审计日志、实时事件流 |

### 2.3 为什么要这样分层

这个分层解决了一个 Agent 项目的常见问题：如果把所有逻辑都放进一个 prompt 或一个 agent loop，系统会很快变成黑箱。

MortyClaw 的设计把不同责任拆开：

- 路由和风险判断放在 `routing.py`。
- 步骤拆分和工具作用域放在 `planning.py`。
- 高风险确认放在 `approval.py`。
- 记忆抽取、召回和缓存放在 `memory_policy.py`。
- 存储细节放在 `memory.py` 和 `runtime_store.py`。
- LLM 调用和 prompt 拼接放在 `agent.py` / `prompt_builder.py`。
- UI 和交互细节放在 `entry/main.py`。

这种拆分的好处是面试时可以讲清楚“每个模块为什么存在”，也方便后续替换单个策略。例如未来要把规则 router 换成 LLM router，只需要替换 `build_route_decision()`，不用重写整个 runtime。

## 3. Agent Workflow 深度分析

### 3.1 当前图结构

MortyClaw 的 LangGraph 工作流是：

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

### 3.2 为什么不是单 loop

普通 ReAct Agent 通常是：

```text
LLM -> Tool -> LLM -> Tool -> ... -> Answer
```

问题是：

- 用户不知道 Agent 为什么调用某个工具。
- 复杂任务没有显式步骤。
- 高风险工具可能被模型直接调用。
- 步骤失败后没有统一恢复策略。
- 很难审计“执行到哪里了”。

MortyClaw 用图工作流把这些隐式过程显式化：

- router 负责“走快还是走慢”。
- planner 负责“慢任务拆成几步”。
- approval gate 负责“高风险先停住”。
- reviewer 负责“执行后检查是否成功”。
- fast_tools / slow_tools 分开接入工具执行。

这让它相对普通聊天机器人更像一个可控执行系统。

### 3.3 Router

作用：

- 从最新 HumanMessage 中提取 query。
- 基于关键词判断任务复杂度和风险。
- 论文类问题保持 fast path，并在 agent 节点中直接转发 arxiv_rag。
- 同步 session memory。
- 调度 long-term memory 异步写入。

为什么需要：

- 不是所有任务都应该进入复杂流程。普通问答如果也 planner + approval，会显著增加延迟和用户负担。
- 高风险任务不能直接让 LLM 自由执行。
- 记忆捕获适合在每轮入口处做，因为 router 能拿到用户原始输入。

当前实现类型：

- 规则路由：基于 `SLOW_PATH_MULTI_STEP_HINTS`、`SLOW_PATH_HIGH_RISK_HINTS`、`ARXIV_DIRECT_QUERY_HINTS` 等关键词。

优点：

- 可解释。
- 低延迟。
- 不消耗额外 LLM 调用。
- 容易测试，测试中已经覆盖 simple、multi-step、high-risk、arxiv query。

局限：

- 关键词漏召或误判不可避免。
- 对复杂语义理解弱，例如用户隐晦表达“帮我动一下这个文件”可能不一定覆盖。

可优化：

- 保留规则作为 first pass，再加入低成本 LLM classifier 作为 fallback。
- 给路由结果增加 confidence 字段，低置信度时进入保守 slow path。
- 将高风险词和工具风险级别做成配置表，而不是散落在常量里。

### 3.4 Planner

作用：

- 将 slow path 的用户目标拆成 execution plan。
- 为每个步骤标记 `risk_level`。
- 判断当前步骤是否需要 approval。

为什么需要：

- 复杂任务不能让 LLM 一次性自由发挥。
- 多步骤任务需要可恢复的中间状态。
- 每个步骤风险不同，不能只按整体任务风险处理。

当前实现类型：

- 规则 planner：根据“然后、接着、最后、再、then、finally”等连接词切分任务。
- 风险分类也是规则：文件写入、shell、删除、修改任务等归入 high risk。

优点：

- 透明稳定。
- 对常见中文多步骤表达有效。
- 易于和 reviewer 配合做 step-by-step 推进。

局限：

- 不能像 LLM planner 那样根据任务语义生成更合理的步骤。
- 复杂科研任务的步骤质量依赖用户原始表达。

可优化：

- 引入可选 LLM planner，但保留规则 planner 作为安全兜底。
- 给每个 step 增加 `expected_output`、`allowed_tools`、`success_criteria`。
- 对代码修复任务生成标准流程：定位 -> 修改 -> diff -> test -> 总结。

### 3.5 Approval Gate

作用：

- 对 high risk step 做显式用户确认。
- 支持确认、取消、继续等待有效确认。
- 确认后才允许进入 slow_agent 执行。

为什么需要：

- Agent 可能调用写文件、shell、删除任务、修改任务等不可逆或有副作用工具。
- 用户需要知道当前要执行的具体步骤和风险原因。

当前实现类型：

- 规则审批：`risk_level == high` 时要求确认。
- 确认词和拒绝词固定集合，例如“确认执行”“取消”“yes”“no”。

优点：

- 安全边界清晰。
- 用户可控。
- 与 slow path 的步骤计划结合，提示信息具体到当前步骤。

局限：

- 只能判断简短确认词，对复杂自然语言确认支持有限。
- 当前没有 diff preview 或命令 preview 审批，用户确认的是步骤描述，不一定是最终具体 patch 或 shell 命令。

可优化：

- 高风险执行前展示具体 tool call 参数。
- 写文件/patch 前展示 diff dry-run。
- 引入审批策略表，区分 read/write/shell/delete/network 等风险类别。

### 3.6 Fast Agent 与 Slow Agent

作用：

- 两者共享 `_run_react_agent_node()`。
- fast path 面向简单低风险任务。
- slow path 面向复杂或高风险任务，并根据当前 step 限制工具集合。

关键实现点：

- slow path 会调用 `select_tools_for_current_step()`，只绑定当前步骤允许的工具。
- 如果模型仍然发起越界工具调用，`enforce_slow_step_tool_scope()` 会拦截并返回失败信息。

为什么重要：

普通 Agent 很容易出现“用户让它先分析，它直接修改”或“第一步应该读文件，它提前运行 shell”。MortyClaw 把当前 step 的工具作用域显式约束住，降低越界执行风险。

优点：

- 工具控制粒度比单纯 prompt 约束更强。
- 越界工具调用会记录审计日志。
- 适合面试强调“Agent safety 不只靠 prompt”。

局限：

- 工具选择仍然依赖规则关键词。
- 如果步骤文本本身很模糊，可能给错工具。
- 不能完全阻止工具内部逻辑的安全缺陷，因此还需要工具层沙盒。

### 3.7 Reviewer

作用：

- slow_agent 执行后检查当前步骤结果。
- 如果结果看起来失败，先 retry。
- retry 超过预算后 replan。
- 成功后推进下一步，下一步高风险则重新进入 approval gate。

当前实现类型：

- 规则 reviewer：检查输出中是否包含“失败、错误、异常、timeout、traceback、not found”等失败标记。

优点：

- 给 slow path 增加了恢复机制，而不是失败后直接结束。
- 每一步结果写入 `step_results`，工作记忆可以保留最近轨迹。

局限：

- 判断成功/失败比较粗糙，可能误判。
- 没有真正验证工具结果是否满足 step 的 success criteria。

可优化：

- 在 planner 中为每步生成 success criteria。
- reviewer 结合工具退出码、测试结果和 diff 判断。
- 对代码任务强制经过 test/diff gate 后才标记完成。

## 4. 记忆系统分析

### 4.1 记忆层级

MortyClaw 的记忆不是单一聊天历史，而是四层：

| 层级 | 存储位置 | 生命周期 | 作用 |
| --- | --- | --- | --- |
| working memory | LangGraph state | 当前任务/工作流 | 保存目标、计划、步骤、审批、最近工具结果 |
| session memory | `memory_records` | 当前 thread_id | 保存项目路径、语言偏好、只分析不改代码、审批偏好等 |
| long-term memory | `memory_records` + `user_profile.md` | 全局长期 | 保存用户偏好、项目事实、工作流偏好、安全偏好 |
| context summary | LangGraph state | 长对话期间 | 将旧对话压缩成近期上下文摘要 |

### 4.2 为什么这样设计

如果只依赖完整聊天历史，会出现几个问题：

- 上下文越来越长，成本和延迟上升。
- 用户长期偏好和临时任务进度混在一起。
- 换 thread_id 后无法区分哪些该继承，哪些不该继承。
- 冲突偏好无法治理，例如“以后用中文”和“以后用英文”可能同时存在。

MortyClaw 用分层记忆解决：

- working memory 管当前执行状态。
- session memory 管当前会话约束。
- long-term memory 管跨会话偏好和事实。
- context summary 管长对话压缩。

### 4.3 MemoryRecord 数据模型

核心字段：

| 字段 | 作用 |
| --- | --- |
| `memory_id` | 主键 |
| `layer` | working/session/long_term |
| `scope` | session 用 thread_id，长期默认 user_default |
| `type` | 记忆类别 |
| `subject` | 细粒度主题，用于冲突治理 |
| `content` | 记忆内容 |
| `source_kind` | 来源，例如 rule_extractor、manual_tool |
| `source_ref` | 来源文本 |
| `confidence` | 置信度 |
| `status` | active/superseded/archived/expired/deleted |

这比“简单存一句话”更工程化，因为它为检索、过滤、冲突治理和可解释管理留下了结构。

### 4.4 Session Memory

作用：

- 在当前 thread_id 下记住临时上下文。
- 典型内容包括项目路径、中文输出偏好、只分析不改代码、office 工作区、高风险确认偏好。

为什么需要：

- 用户不会每轮都重复项目路径。
- 当前会话可能有特殊约束，不应该污染全局长期记忆。
- 代码项目工具可以从 session memory 中读取 `project_path` 作为默认 project_root。

当前实现：

- `extract_session_memory_records()` 用规则抽取。
- `sync_session_memory_from_query()` 写入 `memory_records`。
- `build_session_memory_prompt()` 将最近 session memory 格式化到 system prompt 的 `[本轮会话约束]`。

优点：

- 简单可控，低延迟。
- 不跨会话污染。
- 对项目路径这类信息非常实用。

局限：

- 抽取规则有限。
- 不能理解复杂自然语言约束。
- 缺少用户可见的 `/session-memory` 管理界面。

### 4.5 Long-term Memory

作用：

- 保存跨会话长期有效的偏好、事实、工作流和安全规则。

当前支持类型：

- `user_preference`
- `project_fact`
- `workflow_preference`
- `safety_preference`
- 兼容旧 `user_preference_note`

当前实现：

- 用户说“记住、以后、一直、长期、我喜欢、我不喜欢、my preference”等，会触发规则抽取。
- 根据 query 规则分类 type 和 subject。
- 异步写入 SQLite，避免阻塞主路径。

优点：

- 结构化程度足够支撑冲突治理。
- 不把所有历史对话都当长期记忆，降低污染。
- 通过 `source_ref` 保留来源，便于解释。

局限：

- 抽取偏规则，复杂偏好容易漏。
- 没有 pending/confirmed 机制，规则命中后就写入。
- 还没有完整 forget/archive UI。

### 4.6 冲突治理

关键机制：

```text
同一 scope + type + subject 下，
新的 long_term active 记忆写入时，
旧 active 记忆自动变为 superseded。
```

例子：

- 旧记忆：`user_preference / response_language / 以后用中文回答`
- 新记忆：`user_preference / response_language / 以后用英文回答`
- 结果：旧记录 `superseded`，新记录 `active`

价值：

- 避免 prompt 中同时注入相互冲突的长期偏好。
- 让“最新明确偏好生效”成为系统规则。
- 这比普通 memory list 更接近产品级记忆系统。

局限：

- 冲突粒度依赖 subject 分类准确性。
- 对“有条件偏好”支持不足，例如“写代码用中文解释，但论文摘要用英文”。

### 4.7 FTS5 检索

当前记忆检索不是向量检索，而是：

- SQLite FTS5 虚拟表 `memory_records_fts`。
- `MATCH` 查询。
- `bm25(memory_records_fts)` 排序。
- 中文二元词增强。
- type / subject 中文别名增强。
- FTS 不可用时 fallback 到 `LIKE`。

为什么不用向量检索：

- 当前记忆多是短文本和结构化偏好，FTS5 足够覆盖大量场景。
- SQLite-first 使部署简单，不需要 embedding 模型或向量库。
- 精确事实、路径、任务、偏好词匹配更可解释。
- 本地个人 Agent 更看重低延迟、低依赖和可维护性。

使用向量检索后的区别：

| 维度 | 当前 FTS5 | 向量/混合检索 |
| --- | --- | --- |
| 精确路径/专有名词 | 强 | 不一定更好 |
| 同义表达 | 较弱 | 更强 |
| 部署复杂度 | 低 | 中到高 |
| 查询成本 | 低 | 需要 embedding 或本地模型 |
| 可解释性 | 高 | 中等 |
| 误召回风险 | 较低 | 更高 |

建议：

- 不建议替换 FTS5。
- 可选做混合检索：FTS5 主路径，向量作为 long-term memory slow path。
- 对安全偏好和项目路径仍应优先结构化字段和 FTS5。

### 4.8 Context Summary

作用：

- 长对话超过阈值时，保留最近完整回合，旧回合压缩成摘要。

当前实现：

- `trim_context_messages(raw_messages, trigger_turns=40, keep_turns=10)`。
- 按 HumanMessage 回合切分，保留工具消息完整性。
- 调用 LLM 生成不超过 150 字上下文摘要。
- 摘要超时或失败时 fallback 到规则摘要。
- 摘要只记录任务进度，不记录静态用户偏好。

价值：

- 控制 prompt 长度。
- 避免长对话完全丢失旧任务进度。
- 与长期记忆职责分离，减少偏好污染。

局限：

- 150 字摘要可能丢细节。
- 摘要质量依赖 LLM。
- 超时 fallback 较粗糙。

## 5. SQLite-first 运行态系统

### 5.1 存储布局

| 文件 | 用途 |
| --- | --- |
| `workspace/state.sqlite3` | LangGraph checkpointer，保存 thread_id 对应状态和消息历史 |
| `workspace/runtime.sqlite3` | sessions、tasks、task_runs、session_inbox |
| `workspace/memory/memory.sqlite3` | memory_records 和 FTS5 索引 |
| `workspace/code_index.sqlite3` | Python 代码索引 |
| `workspace/tasks.json` | 旧任务格式兼容镜像 |
| `logs/{thread_id}.jsonl` | 会话审计日志 |

### 5.2 为什么选择 SQLite

对个人本地 Agent 来说，SQLite 是合理选择：

- 零运维，不需要 Redis/Postgres/Qdrant。
- 适合单机本地工具。
- 能同时承担 checkpoint、runtime、memory、code index。
- 可持久化，进程重启后状态仍存在。
- 便于用户直接查看和备份。

边界：

- 高并发多用户不是 SQLite 的主要优势。
- 多进程任务 claim/lock 还可以增强。
- schema 演进目前是 `CREATE TABLE IF NOT EXISTS`，缺少正式 migration 版本管理。

### 5.3 Sessions

作用：

- 管理 thread_id、provider、model、status、log_file、last_active_at。
- 支撑 `/sessions`、`mortyclaw sessions` 和 monitor latest。

为什么需要：

- Agent 项目一旦有多会话，就需要清楚知道当前状态属于哪个会话。
- thread_id 同时贯穿 checkpoint、memory scope、task、inbox、logs 和 arxiv_rag session_id。

价值：

- 不同会话状态隔离。
- 可以恢复指定 thread_id。
- monitor 能选择最近活跃会话。

### 5.4 Tasks 与 Heartbeat

作用：

- 用户可设置单次或循环定时任务。
- `schedule_task` 写入 SQLite tasks。
- `heartbeat` 独立进程扫描到期任务。
- 到期后写入 `session_inbox`，由对应 run 进程消费。

关键设计：

```text
LLM calls schedule_task
  -> tasks 表持久化
  -> heartbeat 扫描 due tasks
  -> 写入 session_inbox
  -> run 进程轮询 pending event
  -> 转为 HumanMessage 进入 Agent
  -> 标记 delivered
```

为什么不是内存队列：

- heartbeat 和 run 进程不是同一个进程。
- 主程序重启后，内存队列会丢。
- SQLite inbox 可以让任务事件跨进程、可恢复、可查询。

这是项目里非常值得写进简历的工程点，因为它体现了“运行态可靠性”的设计意识。

当前边界：

- 多 heartbeat 并发时还缺任务级 claim/lock。
- `session_inbox` 只有 pending/delivered，没有 retry/dead-letter。
- 对任务执行结果的闭环记录还可以更丰富。

## 6. 工具体系与科研代码协作能力

### 6.1 工具分类

| 类别 | 工具 |
| --- | --- |
| 基础工具 | `get_current_time`, `calculator`, `get_system_model_info` |
| 任务工具 | `schedule_task`, `list_scheduled_tasks`, `modify_scheduled_task`, `delete_scheduled_task` |
| 记忆工具 | `save_user_profile` |
| 沙盒文件/shell | `list_office_files`, `read_office_file`, `write_office_file`, `execute_office_shell` |
| 项目代码工具 | `read_project_file`, `search_project_code`, `apply_project_patch`, `show_git_diff`, `run_project_tests` |
| 外部信息 | `tavily_web_search`, `arxiv_rag_ask`, `summarize_content` |
| 动态技能 | `workspace/office/skills/*/SKILL.md` 自动包装为 StructuredTool |

### 6.2 项目级代码工具

这是 MortyClaw 相比普通个人助手的重要增强。

#### read_project_file

作用：

- 在用户给定 project_root 或 session memory 记录项目路径后，读取项目内文件片段。
- 返回带行号内容。

安全：

- 只能访问 project_root 内文件。
- 默认拒绝 `.env`、私钥、证书等敏感文件。
- 文件大小有限制。

#### search_project_code

支持模式：

- `text`：基于 rg 的全文搜索。
- `symbol`：基于 Python AST 搜索函数/类。
- `callers`：搜索函数/方法调用点。
- `dependencies`：分析模块 imports 和反向依赖。
- `data_flow`：汇总模块 imports、结构、I/O、训练/推理线索。
- `entrypoints`：寻找可能训练/运行入口。

工程价值：

- 让 Agent 具备“读项目”的结构化能力，而不只是全文 grep。
- 对科研代码尤其有用，因为经常需要找训练入口、数据加载、模型定义、评估函数。
- `code_index.py` 使用 SQLite 增量索引 Python 文件，减少重复 AST 扫描。

#### apply_project_patch

作用：

- 接收 unified diff / git diff 格式 patch。
- 用 `git apply --check` 先校验，再正式应用。
- 限制只能修改 project_root 内路径。
- 拒绝敏感文件。

为什么比整文件覆盖更好：

- 改动粒度更小。
- 用户和 Agent 都能通过 diff 看到实际变化。
- 更适合代码 review 和测试闭环。

#### show_git_diff 与 run_project_tests

作用：

- 修改后查看真实 diff。
- 运行测试或静态检查。

安全：

- `run_project_tests` 只允许常见测试/检查命令前缀。
- 拒绝 `rm`、`sudo`、`git reset`、`python -c`、重定向等危险片段。

面试亮点：

> 我没有让 Agent 直接 shell 任意跑命令，而是把项目修改能力拆成 read/search/patch/diff/test 几个工具，并对每个工具加 project_root 边界、敏感文件过滤和命令白名单。这样 Agent 能协作代码，但副作用被限制在可审计流程里。

### 6.3 arxiv_rag 集成

作用：

- 论文类问题直接调用本地 arxiv_rag 的 Feishu reply 接口。
- MortyClaw 传递原始 query 和当前 thread_id 作为 session_id。
- 成功返回后使用 passthrough 机制直接展示结果，不再让外层 LLM 二次改写。

为什么这样设计：

- arxiv_rag 已经有自己的论文检索、问答和会话逻辑。
- MortyClaw 作为上层工作台，不重复实现论文 RAG。
- passthrough 避免外层 LLM 改写论文答案导致引用或细节失真。

边界：

- arxiv_rag 是外部服务，MortyClaw 当前只是转发和集成。
- 如果外部服务不可用，会返回调用失败信息。

### 6.4 Tavily 搜索

作用：

- 处理新闻、实时信息、官网、天气、赛程、价格等外部信息。

关键细节：

- Prompt 中要求模型优先调用 Tavily，不要用 shell 自己联网。
- `normalize_tavily_tool_calls()` 会根据用户原始 query 调整 topic 为 `general` 或 `news`。

价值：

- 防止模型把赛程/天气误当新闻搜索。
- 让外部搜索参数更稳定。

### 6.5 summarize_content

作用：

- 抽取网页、PDF、YouTube、播客、音频、视频、图片或普通文档内容。
- PDF 优先本地 `pypdf` 抽取。
- 网页等通过外部 summarize CLI 的 extract-only 模式抽取。
- 最终摘要由 MortyClaw 当前 Agent 自己完成。

安全边界：

- 不处理代码和项目配置文件。
- 本地相对路径必须在 office 内。
- 本地文本/PDF 有大小限制。
- 本地二进制媒体不做无模型抽取。

为什么重要：

- 把“内容抽取”和“最终总结”分离，避免把外部 CLI 包装成不可控 LLM 摘要器。
- 明确代码分析应该走 project_tools，不走 summarize_content。

### 6.6 动态技能系统

作用：

- 扫描 `workspace/office/skills/<skill>/SKILL.md` 或 README。
- 解析 name/description。
- 包装成 LangChain StructuredTool。
- 支持 `mode='help'` 和 `mode='run'`。

help -> run 协议价值：

- Agent 首次使用技能时先读说明书。
- 降低直接运行未知脚本的风险。
- 技能命令通过 `execute_office_shell` 执行，继承 office 沙盒约束。

局限：

- 技能元数据解析较轻量。
- 没有插件签名、权限声明和版本管理。
- 只适合本地可信技能目录。

## 7. 安全机制分析

MortyClaw 的安全不是单点防护，而是多层防线。

### 7.1 Prompt 层

System prompt 明确规定：

- 写入、删除、shell 只能在 office 或明确 project_root 内。
- 绝对路径只读。
- 科研代码修改必须使用 project tools。
- 修改后必须 diff + test。

价值：

- 给模型行为提供高层约束。

局限：

- Prompt 不是强安全边界，因此后面需要工具层校验。

### 7.2 Router 层

高风险词命中后进入 slow path，例如：

- 写文件、修改代码、运行命令、删除任务、patch、fix、shell。

价值：

- 在执行路径层面把风险任务和普通问答区分开。

### 7.3 Planner / Tool Scope 层

slow path 每一步只允许部分工具。

例子：

- 文件写入步骤才允许 `write_office_file` / `apply_project_patch`。
- 测试步骤才允许 `run_project_tests`。
- shell 步骤才允许 `execute_office_shell`。
- 只读工具默认安全可用。

价值：

- 防止 Agent 提前执行后续操作。
- 防止“当前步骤只该读文件，但模型直接 shell”的情况。

### 7.4 Approval Gate 层

high risk step 必须显式确认。

价值：

- 用户仍然掌握最后执行权。

### 7.5 工具层沙盒

office 工具：

- 写入和 shell 限制在 `workspace/office`。
- 相对路径用 realpath + commonpath 防路径穿越。
- shell 拦截 `..`、绝对路径、home 路径、Windows 盘符等模式。
- shell timeout 60 秒。

project tools：

- 限制在 project_root 内。
- 拒绝敏感文件。
- patch 路径校验。
- 测试命令白名单。

价值：

- 这是实际强约束，不依赖模型自觉。

### 7.6 观测层

每次 LLM 输入、工具调用、工具结果、系统动作都写 JSONL。

价值：

- 出问题后可以追踪 Agent 是如何做出决策的。
- 方便面试演示“透明 Agent”。

### 7.7 当前安全边界

需要如实说明：

- 这不是生产级容器沙盒。
- `calculator` 使用了受限 `eval`，虽然禁用了 builtins，但生产环境最好替换为 AST 表达式解析器。
- shell 正则拦截不是形式化安全证明。
- 还没有命令级审批详情、diff preview 审批、权限策略表。

## 8. 可观察性与可调试性

### 8.1 JSONL 审计日志

`JSONLEventLogger` 使用内存队列和后台线程异步写日志：

- `llm_input`
- `tool_call`
- `tool_result`
- `ai_message`
- `tool_call_adjusted`
- `system_action`

每个事件包含：

- UTC 时间。
- thread_id。
- event 类型。
- 相关参数或摘要。

为什么异步：

- 避免日志 I/O 阻塞 Agent 主流程。
- 退出时通过 atexit flush。

### 8.2 Monitor 终端

`entry/monitor.py` tail 指定会话的 JSONL 文件，并用 Rich 渲染。

价值：

- 用户可以实时看到 Agent 正在“路由、计划、审批、调用工具、收到结果、回复”。
- 对调试 Agent 黑箱行为很有帮助。

可优化：

- Web dashboard。
- 展示 graph state、memory 命中、当前 plan、task_runs、inbox。
- 支持按 event 类型过滤。

## 9. 多模型 Provider 适配

当前支持：

- OpenAI。
- Anthropic。
- 阿里云 DashScope/OpenAI compatible。
- 腾讯混元/OpenAI compatible。
- 智谱 z.ai/OpenAI compatible。
- Ollama。
- other OpenAI compatible。

设计方式：

- `get_provider()` 工厂函数返回 LangChain ChatModel。
- OpenAI compatible provider 统一走 `ChatOpenAI`。
- Anthropic 走 `ChatAnthropic`。
- Ollama 走 `ChatOllama`。

优点：

- Agent workflow 与具体模型解耦。
- CLI config 可以切换 provider/model。
- 适合本地和云模型混合试验。

局限：

- 没有 per-tool model routing。
- 没有模型能力检测，例如是否支持 tool calling。
- 没有 fallback provider 或重试策略。

## 10. 测试覆盖

项目使用 `unittest`，测试覆盖面较广。根据测试文件，已经覆盖：

- Agent app 创建、checkpointer、fast/slow path。
- arxiv passthrough，避免二次 LLM 改写。
- Tavily topic 自动调整。
- 上下文摘要超时 fallback。
- router 路由策略。
- high-risk approval。
- session memory 和 long-term memory 注入。
- 每个高风险步骤需要重新审批。
- slow step 工具越界拦截。
- reviewer retry 和 replan。
- working memory snapshot。
- memory store、FTS 检索、冲突 supersede、prompt cache。
- runtime tasks、heartbeat、session_inbox、重复任务、重启恢复、多会话隔离。
- CLI 新会话 ID。
- sandbox 路径越权、只读绝对路径、写入限制、危险 shell 拦截。
- project tools 注册、搜索、AST 索引、patch、diff、测试命令限制。
- summarize_content 的 PDF、本地文档、代码文件拒绝、timeout、nonzero exit。

面试表达：

> 我没有只做功能 Demo，而是围绕 Agent 路由、审批、记忆、运行态、工具安全和项目代码能力写了回归测试，确保核心 Agent workflow 和 SQLite 持久化逻辑可以被稳定验证。

注意：

- 当前测试主要是单元测试和集成式 mock 测试。
- 还不是完整线上评测体系。
- 对 LLM 真实表现、工具调用质量、长期任务准确率还可以加入 benchmark/eval。

## 11. 相对普通聊天机器人 / 普通 RAG / 普通脚本工具的差异

### 11.1 相对普通聊天机器人

普通聊天机器人：

- 以对话为中心。
- 工具调用不透明。
- 风险控制依赖 prompt。
- 会话状态通常只靠上下文。

MortyClaw：

- 以 Agent runtime 为中心。
- 有 graph state、workflow、reviewer、approval。
- 有显式 memory store 和 runtime store。
- 有工具作用域和沙盒边界。
- 有审计日志和 monitor。

### 11.2 相对普通 RAG

普通 RAG：

- 核心是文档检索 + 回答。
- 重点在知识库召回质量。

MortyClaw：

- RAG 只是外部能力之一。
- 它更关注“Agent 如何安全地执行任务、调用工具、保存状态、恢复任务、记忆偏好、观察过程”。
- arxiv_rag 是被集成的论文能力，不是 MortyClaw 的全部。

### 11.3 相对普通脚本工具

普通脚本工具：

- 单次命令执行。
- 没有会话状态。
- 没有自然语言计划。
- 没有审批和记忆。

MortyClaw：

- 脚本能力通过工具层接入 Agent。
- 工具调用受 workflow 管控。
- 结果写入日志和状态。
- 可结合记忆和项目上下文连续工作。

## 12. 关键工程亮点

### 12.1 图工作流让 Agent 可控可观测

亮点：

- 用 LangGraph 将 Agent 执行拆成 router/planner/approval/agent/tools/reviewer。
- fast path 和 slow path 分离，兼顾响应速度与高风险控制。
- reviewer 提供 retry/replan/step advance。

可写简历：

> 基于 LangGraph 设计单 Agent 图工作流，将任务路由、计划拆分、风险审批、工具执行和结果审查拆解为可追踪节点，实现 fast/slow path 分流和复杂任务逐步执行。

### 12.2 分层记忆与冲突治理

亮点：

- working/session/long-term/context summary 职责分离。
- long-term memory 使用 type + subject 做冲突 supersede。
- FTS5 + 中文二元词 + 字段别名增强低延迟召回。
- Prompt cache 通过 store revision 自动失效。

可写简历：

> 设计 SQLite 分层记忆系统，支持会话级上下文、长期偏好、上下文摘要和 FTS5 检索，并基于 scope/type/subject 实现长期记忆冲突治理，避免过期偏好污染 prompt。

### 12.3 SQLite-first 的本地 Agent 运行态

亮点：

- state.sqlite3 保存 LangGraph checkpoint。
- runtime.sqlite3 保存 sessions/tasks/task_runs/session_inbox。
- memory.sqlite3 保存结构化记忆。
- code_index.sqlite3 保存代码索引。

可写简历：

> 构建 SQLite-first 本地运行态，将会话、任务、跨进程事件、记忆和代码索引统一持久化，实现多会话隔离、任务重启恢复和本地低依赖部署。

### 12.4 session_inbox 跨进程事件投递

亮点：

- heartbeat 独立进程不直接调用 Agent。
- 到期任务写入 session_inbox。
- run 进程按当前 thread_id 轮询消费。

可写简历：

> 设计 session_inbox 机制解耦 heartbeat 调度进程与 Agent 交互进程，使用 SQLite pending/delivered 状态实现定时任务跨进程投递和重启恢复。

### 12.5 工具作用域与沙盒安全

亮点：

- slow step 只绑定当前步骤允许工具。
- 越界工具调用被拦截。
- office 写入/shell 限定目录。
- project patch/test 有 project_root 和命令白名单。

可写简历：

> 实现多层 Agent 工具安全机制，包括高风险审批、步骤级工具白名单、沙盒路径校验、敏感文件拦截、patch 校验和测试命令白名单，降低 LLM 工具误用风险。

### 12.6 科研代码协作能力

亮点：

- 读文件带行号。
- rg 全文搜索。
- AST 符号、调用点、依赖、数据流、入口分析。
- SQLite 增量代码索引。
- patch/diff/test 闭环。

可写简历：

> 构建面向科研代码库的 Agent 工具链，支持代码检索、AST 结构分析、调用链定位、训练入口识别、补丁级修改、diff 检查和测试验证，提升 Agent 在真实项目中的可执行能力。

## 13. 项目难点与解决方案

### 难点 1：Agent 工具调用不可控

问题：

- LLM 可能乱用工具、提前执行、跳过步骤或调用高风险工具。

方案：

- router 将高风险任务送入 slow path。
- planner 拆步骤并标记风险。
- approval gate 要求显式确认。
- slow agent 只绑定当前 step 允许工具。
- enforce_slow_step_tool_scope 拦截越界工具调用。
- 工具内部再做路径和命令校验。

### 难点 2：长期记忆容易污染 prompt

问题：

- 如果把所有用户表达都塞进长期记忆，prompt 会变脏。
- 冲突偏好会同时出现。

方案：

- 只有命中“记住/以后/长期/我喜欢”等触发词才写长期记忆。
- 记忆分类为 user_preference/project_fact/workflow_preference/safety_preference。
- subject 做冲突治理。
- 长期记忆按需召回，不每轮注入。

### 难点 3：长对话上下文过长

问题：

- 长期聊天会让 prompt 爆炸。
- 直接截断会丢失任务进度。

方案：

- 按完整用户回合裁剪，保留工具消息完整性。
- 保留最近 10 回合。
- 旧回合用 LLM 摘要压缩。
- 摘要失败或超时用 fallback。

### 难点 4：定时任务跨进程可靠投递

问题：

- heartbeat 和 run 进程分离，内存队列不可共享。
- 进程重启后任务不能丢。

方案：

- tasks 表持久化任务。
- heartbeat 扫描 due tasks。
- session_inbox 存 pending event。
- run 进程轮询当前 thread_id 的 inbox。
- 消费成功后标记 delivered。

### 难点 5：科研代码 Agent 不能只靠 shell

问题：

- 让 LLM 任意 shell 很危险。
- 代码理解需要结构化检索，不只是打开文件。

方案：

- 提供 project_tools 专用接口。
- read/search/patch/diff/test 分离。
- AST + SQLite code index 提供结构化能力。
- patch 先 git apply --check。
- 测试命令白名单。

## 14. 当前不足与优化方向

### 14.1 架构层

不足：

- 当前不是严格多 Agent，router/planner/reviewer 是状态机节点。
- planner 和 reviewer 多为规则逻辑，复杂任务智能性有限。

优化：

- 引入可选 LLM planner。
- 增加 step success criteria。
- 对 reviewer 加入测试结果和工具结构化结果判断。
- 对 graph state 做可视化 dashboard。

### 14.2 记忆层

不足：

- 长期记忆抽取偏规则。
- FTS5 不是语义检索。
- 缺少记忆管理命令。
- 跨进程 prompt cache revision 感知有限。

优化：

- 加入异步 LLM memory extractor。
- FTS5 + vector hybrid 检索作为可选增强。
- 增加 `/memory`, `/forget`, `/archive-memory`。
- 增加 DB revision 表。
- 增加 pending/confirmed 记忆确认流程。

### 14.3 安全层

不足：

- office shell 正则拦截不是强隔离。
- 没有容器沙盒。
- approval 还没有展示具体 diff/command。

优化：

- 容器化执行。
- 命令白名单。
- 写入前 diff preview。
- 高风险 tool call 参数审批。
- 权限策略表和工具风险分级。

### 14.4 运行态层

不足：

- heartbeat 并发可能需要 claim/lock。
- inbox 没有 retry/dead-letter。
- schema migration 不正式。

优化：

- tasks 增加 claimed_at/claimed_by。
- session_inbox 增加 retry_count/error_message/dead_letter。
- 引入 schema version migration。

### 14.5 产品层

不足：

- monitor 是终端日志流。
- `/sessions` 和 `/tasks` 已有，但记忆、inbox、task_runs 缺 UI。

优化：

- Web dashboard。
- 记忆管理 UI。
- 任务运行历史 UI。
- Agent 执行 timeline。

## 15. 简历包装素材

### 15.1 简洁版项目描述

> MortyClaw 是一个基于 LangGraph 的透明可控个人 AI Agent 终端，面向科研和代码协作场景，集成图工作流编排、分层记忆、SQLite 持久化运行态、风险审批、工具沙盒、定时任务、论文 RAG 转发和实时监控，支持多会话隔离、项目代码分析、补丁级修改与测试验证。

### 15.2 详细版项目描述

> 设计并实现 MortyClaw，一个面向个人科研/开发工作流的 AI Agent 运行平台。项目基于 LangGraph 构建 router/planner/approval gate/reviewer 图工作流，将简单问答、高风险操作和多步骤任务分流处理；基于 SQLite 构建 checkpoint、runtime、memory 和 code index 持久化层；实现 working/session/long-term/context summary 分层记忆、FTS5 检索、中文二元词增强和长期偏好冲突治理；集成项目级代码工具链、Tavily 搜索、arxiv_rag 论文问答、外部内容抽取摘要、定时任务和 JSONL 实时监控，重点解决 Agent 工具调用不可控、长期记忆污染、任务跨进程恢复和执行过程不可观测等问题。

### 15.3 简历 Bullet 版本

可直接选择 4 到 6 条放入简历：

- 基于 LangGraph 设计单 Agent 图工作流，拆分 router、planner、approval gate、fast/slow agent、reviewer 等节点，实现简单任务 fast path 和高风险/多步骤任务 slow path 分流。
- 构建多层 Agent 安全机制：高风险步骤显式审批、slow path 步骤级工具白名单、越界工具调用拦截、office 沙盒路径校验、项目工具敏感文件过滤和测试命令白名单。
- 设计 SQLite 分层记忆系统，支持 working/session/long-term/context summary，使用 FTS5 + BM25 + 中文二元词增强检索，并基于 scope/type/subject 实现长期记忆冲突 supersede。
- 实现 SQLite-first 运行态，使用 `state.sqlite3` 保存 LangGraph checkpoint，`runtime.sqlite3` 管理 sessions/tasks/task_runs/session_inbox，支持多会话隔离、定时任务重启恢复和跨进程事件投递。
- 构建面向科研代码库的 Agent 工具链，支持项目文件读取、rg 搜索、Python AST 符号/调用/依赖/入口/数据流分析、patch 级修改、git diff 查看和测试验证。
- 集成 arxiv_rag 论文问答服务和 Tavily 搜索，论文类 query 保留原始问题直连外部 RAG 并使用 passthrough 返回，避免外层 LLM 二次改写导致学术答案失真。
- 实现 JSONL 异步审计日志和 Rich monitor 终端，按 thread_id 实时展示 LLM 输入、工具调用、工具结果、AI 回复和系统状态机动作，提升 Agent 可观测性。
- 编写覆盖 Agent 路由、审批、记忆、runtime、heartbeat、sandbox、project tools、summarize 工具等核心路径的 unittest 回归测试，保障复杂 Agent workflow 的可验证性。

### 15.4 技术关键词

可以放在简历技能/项目关键词里：

```text
LangGraph, LangChain, ReAct Agent, Tool Calling, Agent Workflow,
Router, Planner, Approval Gate, Reviewer, Human-in-the-loop,
SQLite, AsyncSqliteSaver, Checkpoint, FTS5, BM25, Memory Store,
Working Memory, Session Memory, Long-term Memory, Context Summary,
Prompt Cache, JSONL Audit Log, Rich Monitor, Typer CLI,
Sandbox, Tool Scope Control, Project Code Tools, Python AST,
Code Index, Patch Apply, Git Diff, Test Runner,
Tavily Search, arxiv RAG Integration, Multi-session Runtime,
Task Scheduler, Heartbeat, Session Inbox
```

### 15.5 面试开场讲法

推荐 1 分钟版本：

> 我做的 MortyClaw 不是普通聊天机器人，而是一个本地 Agent runtime。它用 LangGraph 把 Agent 执行拆成 router、planner、approval gate、agent、tools、reviewer 几个节点，让简单任务走 fast path，高风险或多步骤任务走 slow path。系统里有分层记忆，session memory 记当前项目和会话约束，long-term memory 记用户偏好和项目事实，并用 SQLite FTS5 检索和 type/subject 做冲突治理。运行态也全部落在 SQLite，包括 LangGraph checkpoint、sessions、tasks、task_runs 和跨进程 session_inbox。工具层支持论文 RAG 转发、Tavily 搜索、外部内容抽取，以及项目级代码分析、patch 修改和测试验证。为了避免黑箱执行，我还做了高风险审批、步骤级工具作用域、沙盒路径校验和 JSONL monitor，可以实时看到 Agent 每一步在做什么。

### 15.6 面试深入讲法

如果面试官问“这个项目有什么难点”，可以按这条线讲：

1. Agent 工具调用不可控，所以我没有只靠 prompt，而是做了 graph workflow、risk routing、approval gate、step-level tool scope 和工具内部安全校验。
2. 记忆容易污染，所以我把记忆分成 working/session/long-term/context summary，不同生命周期分开管理；长期记忆加 type/subject 冲突治理。
3. 本地任务调度需要跨进程恢复，所以我用 SQLite tasks + session_inbox 解耦 heartbeat 和 run 进程。
4. 科研代码协作不能只靠 shell，所以我做了 project_tools：AST 索引、调用点、依赖、入口分析、patch/diff/test 闭环。
5. Agent 黑箱难调试，所以我做了 JSONL 审计日志和 monitor，把 LLM 输入、工具调用和状态机动作都打出来。

### 15.7 面试问答素材

#### Q1：为什么用 LangGraph？

答：

> 因为我需要的不只是一个 while loop ReAct，而是可恢复、可观测、可插入审批的状态机。LangGraph 可以把 router、planner、approval gate、agent、tools、reviewer 显式建成节点，用条件边控制流转，同时通过 checkpointer 按 thread_id 持久化状态。这样复杂任务执行到一半等待用户确认或进程重启后，都有状态基础可以恢复。

#### Q2：为什么用 SQLite，而不是 Redis/Postgres/向量库？

答：

> 项目定位是个人本地 Agent 终端，低依赖和可迁移比高并发更重要。SQLite 可以同时承载 checkpoint、runtime、memory 和 code index，不需要额外服务。对当前短文本结构化记忆，FTS5 已经能覆盖路径、偏好、项目事实等主要召回场景。后续如果长期记忆规模扩大或同义表达召回不足，可以在 FTS5 之外加向量检索作为 slow path，而不是直接替换。

#### Q3：怎么防止 Agent 乱执行命令？

答：

> 我做了多层控制。第一层 router 把写文件、shell、删除、修改等请求识别为 high risk，进入 slow path。第二层 planner 拆步骤并标风险。第三层 approval gate 要求用户确认。第四层 slow agent 只绑定当前步骤允许的工具，越界 tool call 会被拦截。第五层工具内部做 realpath/commonpath 路径校验、敏感文件拦截和测试命令白名单。也就是说安全不只靠 prompt。

#### Q4：记忆系统怎么避免冲突？

答：

> 长期记忆不是简单 append，而是有 type 和 subject。例如“以后用中文回答”和“以后用英文回答”都会归到 `user_preference / response_language`。当新的 active 长期记忆写入时，系统会把同一 scope/type/subject 下旧 active 记录标记为 superseded，保证 prompt 里只注入最新明确偏好。

#### Q5：为什么论文问题直接 passthrough arxiv_rag？

答：

> 因为 arxiv_rag 已经负责论文检索、问答和会话逻辑，MortyClaw 的定位是上层 Agent 工作台。如果外层 LLM 再二次改写，可能引入引用或结论失真。所以论文类 query 直接把原始问题和 thread_id 传给 arxiv_rag，返回后用 passthrough 直接展示。

#### Q6：你的项目和普通 RAG 有什么区别？

答：

> 普通 RAG 的核心是文档检索和回答，MortyClaw 的核心是 Agent runtime。它关注的是一个 Agent 如何路由任务、调用工具、做风险审批、保存状态、管理记忆、调度任务、分析代码和审计执行过程。arxiv RAG 只是其中一个外部工具能力。

## 16. 推荐简历成稿

### 16.1 偏 AI Agent 平台方向

**MortyClaw：透明可控的个人 AI Agent 运行平台**

- 基于 LangGraph 构建 router/planner/approval gate/reviewer 图工作流，实现简单任务 fast path 与高风险/多步骤任务 slow path 分流，支持步骤级执行、失败重试和重规划。
- 设计 SQLite-first 持久化运行态，使用 checkpoint 保存多会话对话状态，并通过 sessions/tasks/task_runs/session_inbox 支持会话隔离、定时任务、heartbeat 跨进程投递和重启恢复。
- 实现分层记忆系统，支持 working/session/long-term/context summary，采用 SQLite FTS5 + BM25 + 中文二元词增强进行记忆召回，并通过 scope/type/subject 解决长期偏好冲突。
- 构建多层工具安全机制，包括高风险人工审批、步骤级工具白名单、越界调用拦截、沙盒路径校验、敏感文件过滤、patch 校验和测试命令白名单。
- 集成科研代码协作工具链和论文 RAG 转发能力，支持代码检索、AST 调用/依赖/入口分析、补丁级修改、diff/test 验证、Tavily 搜索和 arxiv_rag passthrough。

### 16.2 偏科研助手方向

**MortyClaw：面向科研与代码协作的本地 AI Agent 工作台**

- 构建面向论文问答、项目代码理解和实验辅助的 Agent 工具体系，集成 arxiv_rag 论文服务、Tavily 搜索、外部内容抽取摘要和项目级代码分析工具。
- 实现 Python 代码库 AST 增量索引，支持函数/类检索、调用点定位、模块依赖、数据流摘要和训练入口发现，辅助快速理解科研项目结构。
- 设计 patch/diff/test 闭环，允许 Agent 在 project_root 内执行补丁级修改、查看 git diff 并运行 unittest/pytest/ruff 等安全测试命令。
- 基于 LangGraph 引入显式 workflow 和人工审批机制，避免科研代码修改、shell 执行和任务删除等高风险操作被模型黑箱执行。
- 基于分层记忆记录项目路径、会话约束、用户偏好和工作流偏好，使 Agent 能在多轮科研协作中保留上下文并减少重复输入。

### 16.3 偏工程系统方向

**MortyClaw：SQLite-first 的可观测 Agent Runtime**

- 设计本地 Agent runtime，将 LangGraph checkpoint、sessions、tasks、task_runs、session_inbox、memory_records 和 code_index 统一持久化到 SQLite，实现低依赖可恢复运行。
- 通过 heartbeat + session_inbox 解耦定时任务调度进程与 Agent 交互进程，支持到期任务持久化投递、pending/delivered 状态管理和多会话隔离。
- 实现异步 JSONL 审计日志和 Rich monitor 终端，实时展示 LLM 输入、工具调用、工具结果、AI 回复和系统状态机动作，提升 Agent 行为可观察性。
- 构建规则路由、风险审批、工具作用域控制和沙盒校验组合机制，将 LLM 工具调用副作用限制在可审计、可恢复、可确认的工作流中。

## 17. 最适合展示的项目结构

```text
MortyClaw/
├── entry/
│   ├── cli.py                 # Typer CLI: config/run/monitor/heartbeat/sessions/migrate-tasks
│   ├── main.py                # 交互终端、agent_worker、inbox_poller
│   └── monitor.py             # Rich JSONL 实时监控
├── mortyclaw/core/
│   ├── agent.py               # Agent 装配、LLM 调用、prompt 注入、context trimming
│   ├── runtime_graph.py       # LangGraph 节点和条件边
│   ├── routing.py             # fast/slow/arxiv/Tavily 路由策略
│   ├── planning.py            # 执行计划、风险分类、工具作用域
│   ├── approval.py            # 高风险确认
│   ├── prompt_builder.py      # System prompt、上下文摘要
│   ├── context.py             # AgentState、working memory、上下文裁剪
│   ├── memory.py              # SQLite memory store、FTS5、冲突治理
│   ├── memory_policy.py       # 记忆抽取、召回、Prompt cache
│   ├── runtime_store.py       # sessions/tasks/task_runs/session_inbox
│   ├── heartbeat.py           # 到期任务扫描和投递
│   ├── code_index.py          # Python AST SQLite 增量索引
│   ├── logger.py              # 异步 JSONL 审计日志
│   ├── provider.py            # 多模型 provider 工厂
│   ├── skill_loader.py        # 动态技能加载
│   └── tools/
│       ├── builtins.py        # 基础工具、任务工具、工具注册
│       ├── sandbox_tools.py   # office 文件和 shell 沙盒
│       ├── project_tools.py   # 项目代码 read/search/patch/diff/test
│       ├── web_tools.py       # Tavily + arxiv_rag
│       └── summarize_tool.py  # 外部内容抽取
├── docs/
├── tests/
├── workspace/
└── logs/
```

## 18. 最后的项目定位建议

如果投 AI Agent / LLM 应用工程岗位，建议主打：

- LangGraph workflow。
- Agent tool safety。
- 分层记忆。
- SQLite runtime。
- 可观察性。

如果投科研助手 / RAG 岗位，建议主打：

- arxiv_rag 集成。
- 科研代码理解工具。
- 论文和项目上下文结合。
- 长期偏好和工作流记忆。

如果投后端/平台工程岗位，建议主打：

- SQLite-first 状态管理。
- session_inbox 跨进程投递。
- heartbeat 调度。
- 审计日志和 monitor。
- 工具权限边界。

最推荐的总包装：

> MortyClaw 的亮点不是单个工具，而是把 Agent 的执行控制、记忆管理、任务调度、工具安全、科研代码协作和可观察性统一进一个本地 runtime。这个项目能体现的不只是会调 LLM API，而是具备 Agent 系统设计、状态管理、安全边界、工程落地和产品化思考能力。
