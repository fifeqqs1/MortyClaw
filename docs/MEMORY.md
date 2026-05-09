# MortyClaw Memory System

MortyClaw 当前已经是分层记忆架构。它不再只是“长期画像 + 短期摘要”，而是由工作记忆、会话记忆、长期记忆、上下文摘要、FTS5 检索和内存级 Prompt cache 共同组成。

## 记忆层级

| 层级 | 存储位置 | 生命周期 | 作用 |
| --- | --- | --- | --- |
| working memory | LangGraph state | 单次任务/当前工作流 | 保存目标、计划、当前步骤、审批状态、最近工具结果 |
| session memory | `memory_records` | 当前 thread_id | 保存当前会话的项目路径、语言偏好、操作目录、审批偏好 |
| long-term memory | `memory_records` + `user_profile.md` | 全局长期 | 保存用户偏好、项目事实、工作流偏好、安全偏好 |
| context summary | LangGraph state | 长对话期间 | 对旧对话做摘要，减少 token 压力 |

## 数据表

核心表是 `workspace/memory/memory.sqlite3` 中的 `memory_records`：

| 字段 | 说明 |
| --- | --- |
| `memory_id` | 主键 |
| `layer` | `working` / `session` / `long_term` |
| `scope` | 作用域，session memory 用 thread_id，长期记忆默认 `user_default` |
| `type` | 记忆类型 |
| `subject` | 更细粒度主题，用于冲突治理 |
| `content` | 记忆内容 |
| `source_kind` | 来源，例如 rule_extractor、manual_tool |
| `source_ref` | 来源文本或引用 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `confidence` | 置信度 |
| `status` | `active` / `superseded` / `archived` / `expired` / `deleted` |

FTS 表是 `memory_records_fts`，用于低延迟关键词检索。

## 长期记忆类型

当前长期记忆支持四类主类型：

- `user_preference`：用户表达偏好，例如中文/英文、简洁/详细、称呼方式。
- `project_fact`：项目事实，例如项目路径、仓库、模块、服务、数据库。
- `workflow_preference`：工作流偏好，例如先测试再修改、提交前跑回归。
- `safety_preference`：安全偏好，例如只有高风险才确认、不要越权访问。

为了兼容旧数据，系统仍支持 `user_preference_note`。

## subject 设计

`subject` 是记忆冲突治理的关键。例如：

| type | subject | 示例 |
| --- | --- | --- |
| `user_preference` | `response_language` | “以后用中文回答” |
| `user_preference` | `answer_style` | “以后回答简洁一点” |
| `project_fact` | `project_path` | “当前项目在 /mnt/A/...” |
| `workflow_preference` | `testing_workflow` | “修改后先跑 unittest” |
| `safety_preference` | `approval_policy` | “只有高风险步骤需要确认” |

当新的 long-term active 记忆写入时，如果存在相同 `scope + type + subject` 的旧 active 记忆，旧记录会自动变成 `superseded`。

## 召回流程

用户输入进入 router 后，记忆系统会做两类处理：

1. session memory 同步：规则抽取当前项目路径、语言偏好、操作目录、审批偏好。
2. long-term memory 异步捕获：如果用户说“记住”“以后”“我喜欢”等，写入异步队列。

构建 Prompt 时：

```text
query
  -> should_recall_long_term_memory
  -> load user_profile snapshot
  -> search_memories(query, layer=long_term)
  -> fallback list recent memories
  -> format into system prompt
```

只有当 query 命中长期记忆召回提示词时，才会构建 long-term memory prompt。这样可以避免每轮都查询和注入长期记忆，降低延迟和 token 消耗。

## FTS5 检索

`MemoryStore.search_memories()` 优先使用 SQLite FTS5：

- 英文和数字按 token 索引。
- 中文文本会补充二元词，提升中文短语召回。
- type 和 subject 会加入中文别名，例如 `response_language` 会补充“语言 中文 英文 回复 回答”。
- 如果当前 SQLite 不支持 FTS5，会 fallback 到 `LIKE` 检索。

这不是 embedding 语义检索。它的优点是：

- 不需要额外向量库。
- 不需要 embedding 模型。
- 延迟低。
- 和当前 SQLite 架构一致。

它的缺点是：

- 对同义表达的泛化能力有限。
- 复杂语义匹配不如向量检索。

## Prompt Cache

`MemoryPromptCache` 是进程内 LRU cache，用于缓存 session prompt 和 long-term prompt。cache key 包含：

- store db path
- store object id
- store revision
- thread_id 或 query
- profile 文件 mtime
- prompt limit

当同一进程内记忆写入时，`MemoryStore.revision` 增加，cache 自动失效。这样可以减少重复 SQLite 查询和格式化开销，同时避免大多数陈旧缓存问题。

## 上下文摘要

长对话不会无限把所有消息塞给 LLM。系统会按用户回合裁剪上下文：

- 保留 system prompt。
- 保留最近若干完整回合。
- 丢弃的旧回合交给摘要模块压缩。
- 摘要失败或超时时使用 fallback 摘要。

摘要只记录任务进度和对话上下文，不记录静态用户偏好。静态偏好由 long-term memory 负责。

## 当前边界和优化方向

当前记忆系统已经具备工程可用性，但仍有改进空间：

- 复杂偏好抽取仍偏规则，未来可加入低频异步 LLM 抽取器。
- FTS5 不是完整语义检索，未来可选择性加入 embedding 检索作为慢路径。
- 缺少 `/memory`、`/forget`、`/memory archive` 等用户可见管理命令。
- 跨进程写入记忆时，其他进程的内存 cache 不会立即感知 revision 变化，可通过 DB revision 表进一步增强。
- 记忆置信度目前较简单，未来可增加 pending/confirmed 流程。

## 设计原则

- 主路径低延迟优先。
- 长期记忆召回按需触发，不每轮注入。
- 先用 SQLite/FTS5 解决 80% 问题，再考虑向量库。
- 冲突治理优先保证“最新明确偏好生效”。
- 记忆系统不能偷偷覆盖用户意图，重要偏好应可查看、可删除、可解释。
