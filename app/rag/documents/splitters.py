"""标题感知分割器：输出带元数据的 LangChain Document。"""

import re
from collections.abc import Iterator, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _markdown_paragraphs(text: str) -> Iterator[tuple[list[tuple[int, str]], str]]:
    """提取正文段落与标题路径，围栏代码中的井号不作为标题。"""
    headings: list[tuple[int, str]] = []
    lines: list[str] = []
    fence: tuple[str, int] | None = None

    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            lines.append(line)
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= fence[1]
                and not marker[2].strip()
            ):
                fence = None
            continue
        if marker:
            fence = (marker[1][0], len(marker[1]))
            lines.append(line)
            continue

        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+.*|[ \t]*)$", line)
        if heading or not line.strip():
            if lines:
                body = "\n".join(lines).strip()
                if body:
                    yield headings.copy(), body
                lines = []
            if heading:
                level = len(heading[1])
                # 同级标题替换，子标题保留父级标题，避免跨章节串联。
                headings = [
                    (depth, title) for depth, title in headings if depth < level
                ]
                headings.append((level, line.strip()))
            continue
        lines.append(line)

    if lines:
        body = "\n".join(lines).strip()
        if body:
            yield headings.copy(), body


def split_documents(
    documents: Sequence[Document], size: int = 600, overlap: int = 100
) -> list[Document]:
    """一级标题仅存元数据，章节标题附在正文前；TXT 直接递归切分。"""
    if not 0 <= overlap < size:
        raise ValueError("重叠长度必须非负且小于分块长度")
    result = []
    for document in documents:
        if document.metadata.get("pdf_prechunked"):
            # PDF 已按结构切片；再次递归切分会破坏字符到来源位置的映射。
            result.append(Document(page_content=document.page_content,
                                   metadata={**document.metadata, "chunk": len(result) + 1}))
            continue
        is_markdown = document.metadata.get("format", ".txt") == ".md"
        sections = (
            _markdown_paragraphs(document.page_content)
            if is_markdown
            else [([], document.page_content)]
        )
        chunk_number = 0
        for headings, body in sections:
            # 文档总标题保留用于展示；仅章节标题参与正文向量化。
            heading_path = "\n".join(title for _, title in headings)
            section_path = "\n".join(title for level, title in headings if level > 1)
            document_title = next(
                (title[1:].strip() for level, title in headings if level == 1), ""
            )
            prefix = section_path + "\n\n" if section_path else ""
            budget = size - len(prefix)
            if budget <= 0:
                raise ValueError("标题路径过长，请缩短标题或增大分块长度")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=budget,
                chunk_overlap=min(overlap, budget - 1),
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
                keep_separator="end",
            )
            for content in splitter.split_text(body):
                if not content.strip():
                    continue
                chunk_number += 1
                metadata = {**document.metadata, "chunk": chunk_number}
                if heading_path:
                    metadata["heading_path"] = heading_path
                if document_title:
                    metadata["document_title"] = document_title
                if section_path:
                    metadata["section_path"] = section_path
                result.append(
                    Document(page_content=prefix + content, metadata=metadata)
                )
    return result


def split_into_chunks(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """旧文本接口，委托统一 Document 分割流程。"""
    return [
        doc.page_content
        for doc in split_documents(
            [Document(page_content=text, metadata={"format": ".md"})], size, overlap
        )
    ]
