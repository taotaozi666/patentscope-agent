"""可选的上下文增强；默认规则模式，模型增强失败回退，结果不作事实证据。"""

import json
from hashlib import sha256

from langchain_core.documents import Document
from openai import RateLimitError

from app.models import create_client
from app.prompt.prompt import PDF_CONTEXT_SYSTEM, PDF_CONTEXT_VERSION
from app.schemas import Settings
from app.schemas.pdf import ParsedPdf
from app.storage.pdf_repository import PdfRepository
from app.trace import print_step


def enhance_context(parsed: ParsedPdf, chunks: list[Document], settings: Settings) -> None:
    if settings.pdf_context_mode != "model":
        return
    if not settings.api_key:
        parsed.warnings.append("模型上下文增强没有可用密钥，已回退到结构上下文。")
        return
    parents = {parent.id: parent for parent in parsed.objects}
    repository = PdfRepository(settings.pdf_directory)
    failures = 0
    try:
        with create_client(settings) as client:
            for index, chunk in enumerate(chunks):
                try:
                    parent = parents[chunk.metadata["parent_id"]]
                    # 长父对象只提供有界摘录；片段自身始终完整，避免无限发送整本文件。
                    payload = {"document": parsed.title, "object": parent.title,
                               "parent_excerpt": parent.raw_content[:12000], "fragment": chunk.page_content,
                               "known_context": chunk.metadata["context_text"]}
                    message = json.dumps(payload, ensure_ascii=False)
                    cache_id = sha256(f"{PDF_CONTEXT_VERSION}:{settings.base_url}:{settings.chat_model}:{message}".encode()).hexdigest()
                    context = repository.get_context(cache_id)
                    if context is None:
                        response = client.chat.completions.create(
                            model=settings.chat_model, temperature=0, stream=False,
                            messages=[{"role": "system", "content": PDF_CONTEXT_SYSTEM}, {"role": "user", "content": message}],
                            extra_body={"enable_thinking": False},
                        )
                        context = (response.choices[0].message.content or "").strip()[:500]
                        if not context:
                            raise ValueError("上下文为空")
                        repository.save_context(cache_id, context, settings.chat_model, PDF_CONTEXT_VERSION)
                    chunk.metadata.update(context_text=chunk.metadata["context_text"] + "\n" + context,
                                          context_method="model", context_version=PDF_CONTEXT_VERSION,
                                          context_model=settings.chat_model, context_is_evidence=False)
                    chunk.metadata["pdf_context"] = chunk.metadata["context_text"]
                    chunk.metadata["_pdf_index_text"] = chunk.metadata["context_text"] + "\n" + chunk.page_content
                    chunk.metadata["pdf_chunk_id"] = sha256(f"{chunk.metadata['pdf_chunk_id']}:{cache_id}".encode()).hexdigest()
                except RateLimitError:
                    failures += len(chunks) - index
                    break  # 配额不足时不继续对其余片段发送增强请求。
                except Exception:
                    failures += 1
    except Exception:
        failures = len(chunks)
    # 增强后的 ID 变化，需要更新相邻子块链接。
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.metadata["parent_id"], []).append(chunk)
    for siblings in grouped.values():
        for index, chunk in enumerate(siblings):
            chunk.metadata["previous_chunk_id"] = siblings[index-1].metadata["pdf_chunk_id"] if index else ""
            chunk.metadata["next_chunk_id"] = siblings[index+1].metadata["pdf_chunk_id"] if index+1 < len(siblings) else ""
    if failures:
        parsed.warnings.append(f"{failures} 个片段的模型上下文增强失败，已回退结构上下文。")
    print_step("PDF 上下文增强完成", 片段数=len(chunks), 回退数=failures, 版本=PDF_CONTEXT_VERSION)
