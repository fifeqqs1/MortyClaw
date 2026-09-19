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

Use `mcp__mortyclaw__research_retrieve` only when an answer needs evidence from the user's
indexed papers or documents and the current context does not already contain enough
`[RETRIEVED_EVIDENCE]`. Do not retrieve for ordinary chat, code questions, rewriting,
translation, or summarizing text already supplied by the user. The presence of words such as
"paper" or "research" alone is not a reason to retrieve. Cite the returned evidence IDs. If an
existing retrieval lacks a new aspect, call with `mode=expand` and its `prior_retrieval_id`;
otherwise use the evidence already in context without another call.

For conceptual or topic-based discovery in the indexed Zotero library, prefer
`mcp__mortyclaw__research_retrieve` with `sources=["zotero"]`. Use direct Zotero search for exact
title, author, year, citation key, collection, or item-key lookup, or when the research index has
no matching evidence. Zotero's optional Chroma semantic-search tool is intentionally unavailable;
MortyClaw uses the shared Qdrant research index for semantic retrieval.

`mcp__mortyclaw__research_index_document`, `research_remove_document`, and
`research_sync_zotero` change the local knowledge index and therefore require approval.

Every MortyClaw MCP call must include the exact `context_token` supplied in the current trusted
runtime context. Pass it unchanged to delegated subagents. It locates the current turn and never
bypasses policy.

Writes, downloads, monitoring changes, task changes, shell commands, and file changes are staged
for MortyClaw approval. Do not retry an `approval_required` response and do not use another tool to
bypass approval. Explain the proposed operation and let MortyClaw present the approval batch.

Zotero is always read-only. Search and read before requesting any external modification.
