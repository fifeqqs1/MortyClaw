from __future__ import annotations

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlparse

from ..config import PROJECT_ROOT
from ..integrations import MCPManager
from .chunking import normalize_text
from .models import ResearchDocument, ResearchSection


_ARXIV_ID = re.compile(r"(?:arxiv:)?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


def _tool_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(
        isinstance(item, dict) and item.get("type") == "text" for item in value
    ):
        return "\n".join(str(item.get("text") or "") for item in value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _decode_json(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and not (
        value and all(isinstance(item, dict) and item.get("type") == "text" for item in value)
    ):
        return value
    text = _tool_text(value).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"(?:\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return value


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _invoke(tool: Any, values: dict[str, Any]) -> Any:
    args = dict(getattr(tool, "args", {}) or {})
    filtered = {key: value for key, value in values.items() if key in args and value is not None}

    # Feishu's OpenAPI MCP exposes transport envelopes (path/params/data)
    # rather than flat business parameters. Populate those envelopes from the
    # same source-neutral values used by the other adapters.
    for envelope in ("path", "params", "data"):
        schema = args.get(envelope)
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties") or {}
        nested = {
            key: value
            for key, value in values.items()
            if key in properties and value is not None
        }
        if nested:
            filtered[envelope] = nested
    if "useUAT" in args:
        filtered["useUAT"] = True

    async def call() -> Any:
        return await asyncio.wait_for(tool.ainvoke(filtered), timeout=60.0)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-source") as pool:
        return pool.submit(asyncio.run, call()).result()


def _find_tool(tools: list[Any], *markers: str) -> Any | None:
    lowered = tuple(marker.lower() for marker in markers)
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").lower()
        if all(marker in name for marker in lowered):
            return tool
    return None


def document_from_pdf(
    path: str | Path,
    *,
    source: str = "arxiv",
    source_id: str = "",
    title: str = "",
    uri: str = "",
) -> ResearchDocument:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError('缺少 PDF 解析依赖，请执行 pip install -e ".[rag]"。') from exc
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    sections: list[ResearchSection] = []
    metadata: dict[str, Any] = {}
    with fitz.open(resolved) as pdf:
        metadata = dict(pdf.metadata or {})
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=True)
            text = "\n\n".join(
                str(block[4]).strip() for block in blocks if len(block) > 4 and str(block[4]).strip()
            )
            if text:
                sections.append(
                    ResearchSection(text=normalize_text(text), heading=f"Page {page_number}", page=page_number)
                )
    resolved_title = title.strip() or str(metadata.get("title") or "").strip() or resolved.stem
    resolved_id = source_id.strip() or resolved.stem
    author_value = str(metadata.get("author") or "").strip()
    authors = tuple(part.strip() for part in re.split(r"[;,]", author_value) if part.strip())
    return ResearchDocument(
        source=source,  # type: ignore[arg-type]
        source_id=resolved_id,
        title=resolved_title,
        sections=tuple(sections),
        uri=uri or resolved.as_uri(),
        authors=authors,
    )


class ResearchSourceLoader:
    def __init__(self, manager: MCPManager | None = None):
        self.manager = manager or MCPManager()

    def load_arxiv(self, locator: str) -> ResearchDocument:
        candidate = Path(locator).expanduser()
        if candidate.is_file():
            return document_from_pdf(candidate, source="arxiv")
        match = _ARXIV_ID.search(locator)
        if not match:
            raise ValueError("无法识别 arXiv ID 或 PDF 路径")
        paper_id = match.group("id")
        storage = Path(PROJECT_ROOT) / "workspace" / "arxiv-papers"
        for suffix in (".pdf", ".md", ".txt"):
            for path in storage.rglob(f"*{paper_id}*{suffix}") if storage.exists() else ():
                if suffix == ".pdf":
                    return document_from_pdf(
                        path,
                        source="arxiv",
                        source_id=paper_id,
                        uri=f"https://arxiv.org/abs/{paper_id}",
                    )
                text = path.read_text(encoding="utf-8", errors="replace")
                return ResearchDocument(
                    source="arxiv",
                    source_id=paper_id,
                    title=self._markdown_title(text) or paper_id,
                    sections=(ResearchSection(text=normalize_text(text)),),
                    uri=f"https://arxiv.org/abs/{paper_id}",
                )
        tools = self.manager.load_tools(strict=True, service="arxiv")
        read_tool = _find_tool(tools, "read_paper")
        download_tool = _find_tool(tools, "download_paper")
        if read_tool is None and download_tool is None:
            raise RuntimeError("Arxiv MCP 未提供论文下载或读取工具")
        content = ""
        if download_tool is not None:
            try:
                content = _tool_text(_invoke(download_tool, {
                    "paper_id": paper_id,
                    "id": paper_id,
                    "force": False,
                    "return_full_text": True,
                    "max_chars": 1_000_000,
                }))
            except Exception:
                content = ""
        if (not normalize_text(content) or content.lstrip().lower().startswith("error")) and read_tool:
            content = _tool_text(_invoke(read_tool, {
                "paper_id": paper_id,
                "id": paper_id,
                "return_full_text": True,
                "max_chars": 1_000_000,
            }))
        if not normalize_text(content) or content.lstrip().lower().startswith("error"):
            raise RuntimeError("Arxiv MCP 未返回可索引的论文正文")
        return ResearchDocument(
            source="arxiv",
            source_id=paper_id,
            title=self._markdown_title(content) or paper_id,
            sections=(ResearchSection(text=normalize_text(content)),),
            uri=f"https://arxiv.org/abs/{paper_id}",
        )

    def load_feishu(self, locator: str) -> ResearchDocument:
        parsed = urlparse(locator)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2 or parts[-2] not in {"docx", "wiki", "docs"}:
            raise ValueError("请提供飞书 docx、docs 或 wiki 文档链接")
        kind, token = parts[-2], parts[-1]
        tools = self.manager.load_tools(strict=True, service="feishu")
        document_id = token
        if kind == "wiki":
            node_tool = _find_tool(tools, "wiki", "getnode") or _find_tool(tools, "wiki", "get_node")
            if node_tool is None:
                raise RuntimeError("飞书 MCP 未提供知识库节点读取工具")
            node = _decode_json(
                _invoke(node_tool, {"token": token, "node_token": token, "wiki_token": token})
            )
            for item in _walk_dicts(node):
                value = item.get("obj_token") or item.get("objToken") or item.get("document_id")
                if value:
                    document_id = str(value)
                    break
        raw_tool = (
            _find_tool(tools, "document", "rawcontent")
            or _find_tool(tools, "document", "raw_content")
            or _find_tool(tools, "docx", "raw")
        )
        if raw_tool is None:
            raise RuntimeError("飞书 MCP 未提供文档纯文本读取工具")
        raw_value = _invoke(
            raw_tool,
            {
                "document_id": document_id,
                "documentId": document_id,
                "doc_token": document_id,
                "token": document_id,
            },
        )
        decoded = _decode_json(raw_value)
        content = ""
        for item in _walk_dicts(decoded):
            value = item.get("content")
            if isinstance(value, str) and normalize_text(value):
                content = value
                break
        content = content or _tool_text(raw_value)
        title = ""
        metadata_tool = _find_tool(tools, "document_get")
        if metadata_tool is not None:
            try:
                metadata = _decode_json(
                    _invoke(metadata_tool, {"document_id": document_id, "documentId": document_id})
                )
                for item in _walk_dicts(metadata):
                    value = item.get("title")
                    if value:
                        title = str(value).strip()
                        break
            except Exception:
                pass
        title = title or self._markdown_title(content) or f"飞书文档 {document_id[:8]}"
        return ResearchDocument(
            source="feishu",
            source_id=document_id,
            title=title,
            sections=(ResearchSection(text=normalize_text(content)),),
            uri=locator,
        )

    def load_zotero_items(
        self, *, limit: int = 1000, query: str = ""
    ) -> list[ResearchDocument]:
        tools = self.manager.load_tools(strict=True, service="zotero")
        normalized_query = normalize_text(query)
        if normalized_query:
            search_tool = _find_tool(tools, "search_items")
        else:
            search_tool = (
                _find_tool(tools, "get_recent")
                or _find_tool(tools, "list_items")
                or _find_tool(tools, "get_items")
                or _find_tool(tools, "search_items")
            )
        if search_tool is None:
            raise RuntimeError("Zotero MCP 未提供条目搜索或列表工具")
        raw_value = _invoke(
            search_tool,
            {
                "query": normalized_query,
                "q": normalized_query,
                "qmode": "everything",
                "item_type": "-attachment",
                "limit": min(max(1, limit), 1000),
            },
        )
        raw = _decode_json(raw_value)
        candidates = list(_walk_dicts(raw))
        if not any(
            (item.get("key") or item.get("item_key") or item.get("itemKey"))
            and item.get("title")
            for item in candidates
        ):
            candidates = self._parse_zotero_item_list(_tool_text(raw_value))
        items = []
        seen: set[str] = set()
        fulltext_tool = (
            _find_tool(tools, "fulltext")
            or _find_tool(tools, "full_text")
            or _find_tool(tools, "read_item")
        )
        for item in candidates:
            key = item.get("key") or item.get("item_key") or item.get("itemKey")
            title = item.get("title")
            if not key or not title or str(key) in seen:
                continue
            seen.add(str(key))
            content = (
                item.get("fulltext")
                or item.get("full_text")
                or item.get("content")
                or item.get("abstractNote")
                or item.get("abstract")
                or ""
            )
            if fulltext_tool is not None:
                try:
                    content = _tool_text(
                        _invoke(
                            fulltext_tool,
                            {"item_key": str(key), "itemKey": str(key), "key": str(key)},
                        )
                    ) or content
                except Exception:
                    pass
            if not normalize_text(str(content)):
                continue
            creators = item.get("creators") or item.get("authors") or []
            authors = []
            for creator in creators if isinstance(creators, list) else []:
                if isinstance(creator, dict):
                    name = creator.get("name") or " ".join(
                        part for part in (creator.get("firstName"), creator.get("lastName")) if part
                    )
                    if name:
                        authors.append(str(name))
                elif creator:
                    authors.append(str(creator))
            year = self._year(item.get("date") or item.get("year"))
            items.append(
                ResearchDocument(
                    source="zotero",
                    source_id=str(key),
                    title=str(title),
                    sections=(ResearchSection(text=normalize_text(str(content))),),
                    uri=str(item.get("url") or f"zotero://select/library/items/{key}"),
                    authors=tuple(authors),
                    year=year,
                )
            )
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _parse_zotero_item_list(text: str) -> list[dict[str, Any]]:
        """Parse the stable Markdown listing returned by zotero-mcp get_recent."""
        results: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            heading = re.match(r"^##\s+\d+\.\s+(.+)$", line)
            if heading:
                if current:
                    results.append(current)
                current = {"title": heading.group(1).strip()}
                continue
            if current is None:
                continue
            field = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", line)
            if not field:
                continue
            name, value = field.group(1).strip().lower(), field.group(2).strip()
            if name == "item key":
                current["key"] = value
            elif name == "date":
                current["date"] = value
            elif name == "authors":
                current["authors"] = [part.strip() for part in value.split(";") if part.strip()]
        if current:
            results.append(current)
        return results

    @staticmethod
    def _markdown_title(text: str) -> str:
        for line in str(text).splitlines():
            value = line.strip().lstrip("#").strip()
            if value:
                return value[:300]
        return ""

    @staticmethod
    def _year(value: Any) -> int | None:
        match = re.search(r"(?:19|20)\d{2}", str(value or ""))
        return int(match.group(0)) if match else None
