from __future__ import annotations

import threading
from typing import Literal

from ..tools.base import mortyclaw_tool
from .indexing import ResearchIndexer
from .retrieval import ResearchRetriever, format_retrieval_result


_retriever: ResearchRetriever | None = None
_indexer: ResearchIndexer | None = None
_lock = threading.Lock()


def _get_retriever() -> ResearchRetriever:
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = ResearchRetriever()
        return _retriever


def _get_indexer() -> ResearchIndexer:
    global _indexer
    with _lock:
        if _indexer is None:
            _indexer = ResearchIndexer()
        return _indexer


@mortyclaw_tool
def research_retrieve(
    query: str,
    sources: list[Literal["zotero", "feishu", "arxiv"]] | None = None,
    document_ids: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    top_k: int = 6,
    mode: Literal["search", "expand"] = "search",
    prior_retrieval_id: str | None = None,
) -> str:
    """检索已经进入 MortyClaw 科研知识库的论文和文档证据。

    仅当回答需要用户文献库、指定文档、论文原文、实验数据、研究结论或引用依据，
    且当前上下文没有足够的 [RETRIEVED_EVIDENCE] 时调用。普通聊天、代码问答、
    润色、翻译，以及总结用户已经提供的完整文本时不要调用。不要仅因问题出现“论文”
    或“研究”就调用。已有证据足够时直接回答；只有问题出现新的研究维度时才使用
    mode=expand，并传入 prior_retrieval_id。回答必须引用返回的 evidence_id；证据不足时
    明确说明，不得用模型常识补写文献结论。
    """
    return format_retrieval_result(_get_retriever().retrieve(
        query=query,
        sources=list(sources or []),
        document_ids=document_ids,
        year_from=year_from,
        year_to=year_to,
        top_k=top_k,
        mode=mode,
        prior_retrieval_id=prior_retrieval_id,
    ))


@mortyclaw_tool
def research_index_document(
    source: Literal["feishu", "arxiv"], locator: str, force: bool = False
) -> dict:
    """将用户明确指定的飞书文档、arXiv ID 或本地论文 PDF 加入科研知识库。

    该操作会写入本地 Qdrant 索引，必须经过 MortyClaw 审批。不会扫描整个飞书云空间，
    也不会建立全量 arXiv 索引。
    """
    return _get_indexer().index_locator(source, locator, force=force)


@mortyclaw_tool
def research_remove_document(document_key: str) -> dict:
    """从本地科研知识库移除一个 document_key；不会删除飞书、Zotero 或原始 PDF。"""
    return _get_indexer().remove(document_key)


@mortyclaw_tool
def research_sync_zotero(
    limit: int = 1000, query: str = "", force: bool = False
) -> dict:
    """把本机 Zotero 中匹配查询且可读取全文的文献增量同步到科研知识库。

    query 留空时按最近条目同步；传入主题、标题或作者时只同步匹配项。该写入需要审批。
    """
    return _get_indexer().sync_zotero(limit=limit, query=query, force=force)


RESEARCH_TOOLS = [
    research_retrieve,
    research_index_document,
    research_remove_document,
    research_sync_zotero,
]

RESEARCH_TOOL_NAMES = frozenset(tool.name for tool in RESEARCH_TOOLS)


__all__ = [
    "RESEARCH_TOOLS",
    "RESEARCH_TOOL_NAMES",
    "research_index_document",
    "research_remove_document",
    "research_retrieve",
    "research_sync_zotero",
]
