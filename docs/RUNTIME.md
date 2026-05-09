# MortyClaw Runtime System

本文档描述 MortyClaw 的运行态系统，包括 CLI 命令、SQLite 表、会话生命周期、任务调度、heartbeat、session_inbox 和 monitor 事件流。

## 运行态文件

| 路径 | 说明 |
| --- | --- |
| `workspace/state.sqlite3` | LangGraph checkpointer，保存 thread_id 对应的对话状态 |
| `workspace/runtime.sqlite3` | sessions、tasks、task_runs、session_inbox |
| `workspace/tasks.json` | 旧任务格式兼容镜像 |
| `logs/{thread_id}.jsonl` | 指定会话审计日志 |

默认 workspace 可通过 `MORTYCLAW_WORKSPACE` 覆盖。

## CLI 命令

| 命令 | 作用 |
| --- | --- |
| `mortyclaw config` | 配置 provider、model、API key、base url |
| `mortyclaw run` | 启动默认会话 `local_geek_master` |
| `mortyclaw run --thread-id <id>` | 启动或恢复指定会话 |
| `mortyclaw run --new` | 创建短编号新会话，例如 `session-1`、`session-2` |
| `mortyclaw run --new-session` | `--new` 的兼容别名 |
| `mortyclaw monitor --latest` | 监控最近活跃会话 |
| `mortyclaw monitor --thread-id <id>` | 监控指定会话 |
| `mortyclaw monitor --list-sessions` | 列出可监控会话 |
| `mortyclaw sessions` | 列出最近会话 |
| `mortyclaw heartbeat` | 启动独立心跳进程 |
| `mortyclaw heartbeat --once` | 只扫描一次到期任务 |
| `mortyclaw migrate-tasks` | 从 `tasks.json` 导入 SQLite |

交互终端内部也支持：

- `/sessions`：以当前 UI 风格展示最近会话。
- `/tasks`：展示当前会话 scheduled 任务。
- `/exit` 或 `/quit`：退出。

## SQLite 表

运行态数据库由 `RuntimeStore.ensure_schema()` 初始化。

### sessions

记录会话元数据。

| 字段 | 说明 |
| --- | --- |
| `thread_id` | 主键，LangGraph checkpointer、日志和任务隔离都依赖它 |
| `display_name` | 显示名 |
| `provider` | 当前模型 provider |
| `model` | 当前模型名 |
| `status` | `active` / `idle` 等 |
| `log_file` | 对应 JSONL 日志路径 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `last_active_at` | 最近活跃时间 |
| `metadata_json` | 扩展元数据 |

### tasks

记录定时任务主数据。

| 字段 | 说明 |
| --- | --- |
| `task_id` | 主键 |
| `thread_id` | 任务归属会话 |
| `description` | 到期后要提醒或执行的内容 |
| `target_time` | 目标时间，格式 `YYYY-MM-DD HH:MM:SS` |
| `repeat` | `hourly` / `daily` / `weekly` / NULL |
| `repeat_count` | 循环总次数 |
| `remaining_runs` | 剩余次数，NULL 表示无限或非循环 |
| `status` | `scheduled` / `completed` / `cancelled` / `failed` |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `last_run_at` | 最近触发时间 |

### task_runs

记录每次任务触发。

| 字段 | 说明 |
| --- | --- |
| `run_id` | 主键 |
| `task_id` | 对应任务 |
| `thread_id` | 对应会话 |
| `status` | 触发结果 |
| `triggered_at` | 触发时间 |
| `finished_at` | 完成时间 |
| `result_summary` | 结果摘要 |
| `error_message` | 错误信息 |

### session_inbox

跨进程事件投递表。

| 字段 | 说明 |
| --- | --- |
| `event_id` | 主键 |
| `thread_id` | 目标会话 |
| `event_type` | 事件类型，例如 `heartbeat_task` |
| `payload` | JSON payload |
| `status` | `pending` / `delivered` |
| `created_at` | 创建时间 |
| `delivered_at` | 被 run 进程消费的时间 |

## 会话生命周期

启动 `mortyclaw run` 时：

```text
resolve thread_id
  -> set_active_thread_id
  -> upsert sessions
  -> create_agent_app
  -> AsyncSqliteSaver(state.sqlite3)
  -> start agent_worker + inbox_poller + user_input_loop
```

用户输入会被送入 `task_queue`，由 `agent_worker` 调用 LangGraph `app.astream()`。会话退出时，状态会被标记为 `idle`。

`--new` 会扫描已有短编号会话，生成下一个可用 ID，例如 `session-1`、`session-2`。`--new-session` 仍作为兼容别名保留。旧版时间戳会话 ID 仍可继续恢复，但不会再影响新短编号的生成。

## 任务创建流程

用户说“明天 9 点提醒我开会”时：

```text
LLM calls schedule_task
  -> validate time format
  -> get active thread_id
  -> create_task in runtime.sqlite3
  -> sync tasks.json compatibility mirror
  -> return task id/time/description
```

任务工具外部接口保持不变，但底层主存储已经是 SQLite。

## heartbeat 流程

独立 heartbeat 进程负责扫描到期任务：

```text
mortyclaw heartbeat
  -> process_due_tasks_once
  -> list_due_tasks
  -> enqueue session_inbox event
  -> record task_run
  -> advance_after_dispatch
  -> log system_action
```

对于循环任务：

- `hourly`：下一次目标时间 + 1 小时。
- `daily`：下一次目标时间 + 1 天。
- `weekly`：下一次目标时间 + 7 天。
- 有 `repeat_count` 时会递减 `remaining_runs`。
- 没有剩余次数后任务状态变为 `completed`。

## inbox 消费流程

`entry/main.py` 中的 `inbox_poller` 每秒查询当前 thread_id 的 pending inbox：

```text
list_pending_inbox_events(current_thread_id)
  -> parse payload.content
  -> put into task_queue
  -> agent_worker processes as HumanMessage
  -> mark_inbox_event_delivered
```

这样 heartbeat 不需要和主程序共享内存队列。主程序重启后，只要 inbox 里仍有 pending 事件，就能恢复消费。

## monitor 流程

运行 `mortyclaw monitor --thread-id <id>` 时：

```text
resolve thread_id
  -> build logs/{thread_id}.jsonl
  -> tail file
  -> render event
```

当前支持渲染：

- `llm_input`：发送给模型的上下文条数。
- `tool_call`：工具名和参数。
- `tool_result`：工具返回摘要。
- `ai_message`：AI 最终回复。
- `tool_call_adjusted`：系统自动修正后的工具参数。
- `system_action`：router、planner、reviewer、heartbeat 等系统行为。

## 兼容迁移

旧版任务保存在 `workspace/tasks.json`。迁移方式：

```bash
mortyclaw migrate-tasks
mortyclaw migrate-tasks --source workspace/tasks.json --default-thread-id local_geek_master
```

短期内任务系统会继续双写 `tasks.json`，方便兼容旧工具和人工查看；长期建议以 `runtime.sqlite3` 为准。

## 当前边界和后续优化

- heartbeat 多进程并发时还可以加入任务级 claim/lock，进一步降低重复投递风险。
- `session_inbox` 可以扩展 retry_count、error_message、dead-letter 状态。
- monitor 当前是日志流，不是完整 dashboard。
- UI 当前提供 `/sessions` 和 `/tasks`，后续可加入 `/inbox`、`/task-runs`、`/memory`。
- runtime schema 目前通过 `CREATE TABLE IF NOT EXISTS` 演进，后续可引入正式 migration 版本表。
