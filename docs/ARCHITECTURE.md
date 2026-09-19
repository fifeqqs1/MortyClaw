# MortyClaw 架构

## 组件边界

| 层 | 主要代码 | 职责 |
| --- | --- | --- |
| 入口 | `entry/main.py`, `integrations/feishu_bot.py`, `observability/heartbeat.py` | CLI、飞书长连接和定时任务 |
| 会话外壳 | `core/harness/runtime.py` | 统一轮次接口、并发、租约、会话映射、记忆注入和历史投影 |
| 对话内核 | `deepseek-harness-sdk==0.1.6a2` | Agent Loop、会话上下文、压缩、Skill、原生子 Agent |
| 工具网关 | `core/harness/gateway.py` | Streamable HTTP MCP、上下文令牌、风险判断和审批暂存 |
| MCP 管理 | `core/integrations/mcp_manager.py` | 飞书、Zotero、Arxiv 的发现、命名隔离和故障降级 |
| 科研检索 | `core/research/` | 按需 Embedding、Qdrant 混合检索、文档增量同步和进程内去重 |
| 持久化 | `core/harness/storage.py`, `core/storage/`, `core/memory/` | session generation、租约、审批、FTS5、长期记忆和审计 |

## 一次普通轮次

```text
入口创建 AgentTurnRequest
  -> 读取 thread_id 对应的 Harness session generation
  -> 同步显式偏好，召回最多 5 条长期记忆
  -> 签发仅保存哈希的 context_token
  -> DeepSeekHarness.run(session_id=...)
  -> Harness 自主管理上下文、Skill、工具与子 Agent
  -> 最终回答和脱敏工具摘要投影到 SQLite
  -> 入口返回回答
```

同一 `thread_id` 使用进程内异步锁串行执行，不同会话受全局并发上限控制。SQLite session lease 防止多个 MortyClaw 进程同时占用同一 Harness session。

SDK 是同步接口，MortyClaw 通过 `asyncio.to_thread()` 调用。SDK 子进程在服务生命周期内保持存活。轮次超时或传输断开后关闭并重建子进程，失败请求不会自动重放。

## 会话与重置

`harness_session_bindings` 保存：

```text
thread_id + generation -> harness_session_id
```

session ID 使用随机盐哈希生成，不包含飞书 `chat_id`。`/reset` 增加 generation，不删除旧 Harness JSONL；长期记忆、FTS5 和用户偏好不变。

Harness JSONL 是当前会话上下文的真源。MortyClaw SQLite 只保存用户消息、最终回答、工具名、脱敏结果摘要和子 Agent 生命周期，用于检索、展示和审计。

## MCP Gateway

Gateway 使用 Streamable HTTP，只监听 `127.0.0.1` 随机端口。每次进程启动生成随机 Bearer Token，URL 与 Token 仅通过子进程环境传入 Harness。

工具名称形如：

```text
mcp__mortyclaw__feishu_*
mcp__mortyclaw__zotero_*
mcp__mortyclaw__arxiv_*
mcp__mortyclaw__memory_*
mcp__mortyclaw__task_*
mcp__mortyclaw__project_*
mcp__mortyclaw__research_*
```

所有工具 Schema 都增加必填 `context_token`。数据库仅保存令牌哈希。令牌映射到 `thread_id`、`turn_id`、来源和工作区，默认两小时过期。它只定位可信运行时上下文，不改变工具风险等级。

## 风险与审批

搜索、读取、列表和导出等低风险工具直接执行。下列能力必须审批：

- 飞书写入；
- Arxiv 下载、监控和索引变更；
- 本地文件写入；
- Shell 或测试命令；
- 定时任务变更；
- 未知元数据工具。

Gateway 收到高风险调用后只写入审批表，不调用底层工具。同一轮相同工具和参数按指纹去重，多项操作合并成一个批次。批准时按保存参数顺序执行，每项最多一次；一次失败后停止其余操作。拒绝、过期或用户开启新任务都会取消未执行项。

定时任务使用 `source=scheduled` 的同一 HarnessRuntime。后台任务不能自批写操作；审批批次与结果会写入所属会话 inbox，并在下一次 CLI 或飞书交互时展示。

## 记忆职责

Harness 管理当前会话历史、上下文压缩、工具结果压缩、子 Agent 上下文和任务状态。

MortyClaw 管理长期用户偏好、项目事实、工作流偏好与跨会话 FTS5 检索。召回内容以“仅供参考的数据上下文”注入，不能覆盖用户指令。用户明确要求“记住”时，规则提取器异步写入长期记忆。

Gateway 还提供 `memory_search_sessions` 与记忆工具，使 Harness 在用户明确引用历史任务时扩大检索。

## Agentic RAG

Harness 不经过规则 Router，而是根据用户问题和当前上下文自主决定是否调用 `research_retrieve`。工具描述和
Skill 要求：普通问答不检索；已有 `[RETRIEVED_EVIDENCE]` 足够时直接使用；只有出现新的证据缺口时才以
`mode=expand` 补充召回。完全相同的查询由两小时进程内缓存去重，不做可能误拦截的语义相似规则。

Qdrant 1.19.1 作为独立本地服务保存文档 chunk、1024 维 multilingual-e5-large dense vector、BM25 sparse
vector 和来源 payload。检索并行获取 dense 与多语言 BM25 候选，使用 RRF 融合，再合并相邻 chunk 并限制
每篇文档最多两个证据片段。Embedding 和 Qdrant 客户端均延迟初始化，因此普通飞书对话不会访问检索层。

SQLite 只增加 `research_documents` 一张来源级同步表，保存 `document_key`、内容 hash、状态和 chunk 数；
不保存正文、chunk、向量、召回日志或缓存。Zotero 可读全文、明确指定的飞书文档以及明确添加或已下载的
arXiv 论文通过统一索引器增量写入 Qdrant。

## Harness Profile 与 Skill

`mortyclaw-sdk` 基于 Harness 的 `sdk` profile，并通过 `configs/mortyclaw-harness.patch.yml`：

- 挂载 MortyClaw MCP Gateway；
- 禁用原生文件写入、Shell、Job、PTC、内部审批和插件管理；
- 禁用遥测；
- 保留读取、Web、compaction、Skill 和原生 subagent。

`.agents/skills/mortyclaw-tools/SKILL.md` 通过 Harness 原生 Skill 机制按需披露工具指导，并规定子 Agent 必须继承当前 `context_token`。

## 数据库兼容

迁移新增 Harness、上下文令牌、审批表和一张轻量 `research_documents` 同步状态表。旧 checkpoint、worker
和会话数据不删除，因此历史搜索仍可使用。新入口不再导入或创建旧执行图。
