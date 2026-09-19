# MortyClaw

MortyClaw 是运行在飞书、本地 CLI 和定时任务中的科研办公助手。当前版本使用 **DeepSeek Harness Python SDK** 作为对话内核，并通过本机 **MortyClaw MCP Gateway** 统一访问飞书、Zotero、Arxiv、记忆、定时任务和项目工具。

## 架构

```text
飞书机器人 / 本地 CLI / 定时任务
                 ↓
       MortyClaw 会话与记忆外壳
                 ↓
 DeepSeek Harness Python SDK 子进程
                 ↓
       MortyClaw MCP Gateway
                 ↓
飞书 / Zotero / Arxiv / Agentic RAG / 记忆 / 项目工具
```

- Harness 负责 Agent Loop、当前会话上下文、压缩、Skill 和原生子 Agent。
- MortyClaw 负责入口适配、跨会话记忆、FTS5 历史检索、工具风险分级、审批和审计。
- 普通问答只进入一个 Harness 轮次，不经过额外的路由、规划或结果复核模型调用。
- Gateway 只监听 `127.0.0.1` 随机端口，使用进程内随机 Bearer Token。
- 文件写入、Shell、飞书写入、Arxiv 下载或监控、定时任务变更等操作先暂存，用户批准后才按保存参数执行一次。

详细设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 环境要求

- Python 3.11 或 3.12
- Windows x64（当前固定使用官方 Windows runtime wheel）
- DeepSeek API Key
- Zotero 7+（仅在使用 Zotero MCP 时需要）

固定内核版本：

```text
deepseek-harness-sdk==0.1.6a2
deepseek-harness-runtime-bin==0.1.6a2
mcp>=1.30,<2
```

运行时不依赖仓库旁的 `deepseek-harness` 源码，也不依赖系统 Node.js。

## 安装

```powershell
git clone https://github.com/fifeqqs1/MortyClaw.git
cd MortyClaw
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[research-mcp]"
# 需要本地科研知识库时再安装：
python -m pip install -e ".[rag]"
Copy-Item .env.example .env
```

当前 `0.1.6a2` 预发布版尚未同步到公共 PyPI。全新环境需要先从 DeepSeek Harness 官方发布产物或内部 wheelhouse 安装同版本的 `deepseek-harness-sdk` 与 `deepseek-harness-runtime-bin` wheel，再执行上面的项目安装命令。本机已经安装这两个固定版本；应用运行时不依赖 Harness 源码目录或系统 Node.js。

配置 DeepSeek 并检查 SDK、runtime 与 Gateway：

```powershell
mortyclaw harness configure
mortyclaw harness status
mortyclaw harness doctor
mortyclaw gateway status
```

使用 DeepSeek 官方地址时无需填写 `DEEPSEEK_BASE_URL`。自定义 DeepSeek 兼容代理才需要该配置。

## 启动

本地 CLI：

```powershell
mortyclaw run
```

飞书机器人：

```powershell
mortyclaw feishu-bot
```

定时任务执行器：

```powershell
mortyclaw heartbeat
mortyclaw heartbeat --once
```

CLI 会话命令：

```text
/sessions  查看会话
/tasks     查看定时任务
/new       创建新会话
/reset     重置当前 Harness 会话
/clear     清屏
/exit      退出
```

`/reset` 只创建新的 Harness session generation；长期记忆、用户偏好和旧会话搜索仍保留。

## 飞书

在飞书开放平台创建并发布应用后执行：

```powershell
mortyclaw feishu-config
mortyclaw feishu-bot
```

机器人使用长连接订阅 `im.message.receive_v1`，无需公网回调地址。私聊直接回复；群聊默认需要先 @机器人。飞书聊天 ID 只以哈希形式进入本地会话映射。

## Zotero 与 Arxiv MCP

```powershell
mortyclaw mcp configure zotero
mortyclaw mcp configure arxiv
mortyclaw mcp status
```

Zotero 使用本机 Local API，并始终过滤所有写工具。请在 Zotero 的高级设置中开启“允许此计算机上的其他应用与 Zotero 通信”。

Arxiv 搜索和读取可直接执行；论文下载、主题监控和索引变更进入审批。论文缓存限定在 `workspace/arxiv-papers`。

## Agentic RAG 科研知识库

Agentic RAG 是可选功能。DeepSeek Harness 自主判断当前问题是否缺少用户论文或文档证据，只有调用
`research_retrieve` 时才加载本地 Embedding 并查询 Qdrant。普通飞书聊天、代码问答、润色、翻译和
对已提供文本的总结不会预先检索，也没有额外的规则 Router 或分类模型。

首次配置：

```powershell
python -m pip install -e ".[rag]"
mortyclaw research configure
mortyclaw research doctor
```

`configure` 会安装并在 `127.0.0.1:6333` 启动官方 Qdrant 1.19.1 Windows x64 进程，下载并验证
`intfloat/multilingual-e5-large` 本地模型；API Key 只保存到 Git 忽略的 `.env`。`doctor` 可随时复查
Qdrant、Dense、BM25 与 RRF 链路。

资料入库：

```powershell
mortyclaw research sync zotero
mortyclaw research sync zotero --query "UAV tracking" --limit 20
mortyclaw research add feishu <document-url>
mortyclaw research add arxiv <paper-id-or-pdf>
mortyclaw research list
mortyclaw research status
```

- Zotero 同步本地可读全文；可用 `--query` 只同步指定主题、标题或作者，Zotero 本身仍严格只读。
- 飞书只索引明确提供的文档或 Wiki 节点，不扫描整个云空间。
- arXiv 只索引明确添加或已下载的论文；实时论文搜索仍走 `arxiv_*` MCP。
- Dense E5 与多语言 BM25 在 Qdrant 中通过 RRF 融合，召回结果保留标题、章节、页码和来源链接。
- 相同会话中的完全相同查询使用两小时进程内缓存；缓存、chunk 和向量均不写 SQLite。

Qdrant 生命周期与维护：

```powershell
mortyclaw research start
mortyclaw research stop
mortyclaw research remove <document-key>
mortyclaw research rebuild --yes
```

`runtime.sqlite3` 只增加 `research_documents` 一张表，用于内容 hash 和同步状态。正文、chunk、向量保存在
`workspace/qdrant`，模型保存在 `workspace/models`；这些目录均被 Git 忽略。

## 审批

当 Harness 请求有副作用的工具时，Gateway 返回 `approval_required`，并把同一轮操作合并为一个批次。批准前工具执行次数为零。

在飞书或 CLI 中回复“确认”或“拒绝”，也可以使用：

```powershell
mortyclaw approvals list
mortyclaw approvals approve <batch-id>
mortyclaw approvals reject <batch-id>
```

审批 15 分钟后过期。执行时按顺序处理；某一步失败后，剩余操作不会继续执行。用户在等待审批期间发送新任务会取消旧批次。

## 会话、记忆与审计

- Harness JSONL 是当前会话上下文的真源。
- `workspace/runtime.sqlite3` 保存会话映射、租约、审批、可搜索消息和脱敏工具摘要。
- `workspace/memory/memory.sqlite3` 保存长期用户偏好、项目事实和工作流偏好。
- 每轮最多召回 5 条相关记忆，总长度不超过 2000 字符，并以非指令数据注入。
- 旧执行内核的数据库内容不删除，仍可通过 FTS5 搜索。
- API Key、飞书密钥、Gateway Token 和完整子进程环境不会写入状态输出或审计日志。

相关命令：

```powershell
mortyclaw sessions
mortyclaw session-search "关键词"
mortyclaw session-show <thread-id>
mortyclaw workers
mortyclaw monitor --latest
```

## Harness Skill

项目内的 `.agents/skills/mortyclaw-tools/SKILL.md` 使用 Harness 原生两阶段 Skill 披露。它说明各 MCP 工具的使用场景、`context_token` 继承规则和审批边界。

Harness profile 禁用了原生文件写入、编辑、Shell、内部交互审批和遥测。所有副作用能力必须经过 MortyClaw Gateway。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -q
mortyclaw harness doctor
mortyclaw gateway status
```

`workspace/harness-home`、论文缓存、数据库、日志和 `.env` 均已加入 Git 忽略规则。
