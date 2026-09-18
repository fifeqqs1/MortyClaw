---
name: mortyclaw-tools
description: Use MortyClaw MCP tools for Feishu, Zotero, Arxiv, memory, task, and project operations.
---

# MortyClaw tools

Use `mcp__mortyclaw__feishu_*` for Feishu documents and messages,
`mcp__mortyclaw__zotero_*` for the local read-only literature library, and
`mcp__mortyclaw__arxiv_*` for public-paper search and reading. Use
`mcp__mortyclaw__memory_*` only when the user refers to prior work, and
`mcp__mortyclaw__task_*` for scheduled work.

Every MortyClaw MCP call must include the exact `context_token` supplied in the current trusted
runtime context. Pass it unchanged to delegated subagents. It locates the current turn and never
bypasses policy.

Writes, downloads, monitoring changes, task changes, shell commands, and file changes are staged
for MortyClaw approval. Do not retry an `approval_required` response and do not use another tool to
bypass approval. Explain the proposed operation and let MortyClaw present the approval batch.

Zotero is always read-only. Search and read before requesting any external modification.
