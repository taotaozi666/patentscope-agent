"""编排 PDF 解析、来源保存与既有向量入库之间的交接。"""

from app.schemas import Settings
from app.trace import print_step

from .chunking import build_chunks
from .parser import PARSER_VERSION, parse_pdf
from app.storage.pdf_repository import PdfRepository
from app.storage.quality_repository import QualityRepository
from app.quality import check_pdf
from .context import enhance_context


def prepare_pdf(source: str, content: bytes, settings: Settings):
    parsed = parse_pdf(source, content)
    chunks = build_chunks(parsed, settings.chunk_size, settings.pdf_target_size)
    for chunk in chunks:
        chunk.metadata["pdf_context_mode"] = settings.pdf_context_mode
        chunk.metadata["pdf_context_model"] = settings.chat_model if settings.pdf_context_mode == "model" else ""
    enhance_context(parsed, chunks, settings)
    report = check_pdf(parsed, chunks, settings.chunk_size)
    for chunk in chunks:
        chunk.metadata["quality_status"] = report.status
    PdfRepository(settings.pdf_directory).save(parsed, content, chunks)
    QualityRepository(settings.pdf_directory).save(report)
    print_step("PDF 质量检查完成", 状态=report.status, 问题数=len(report.issues), 报告ID=report.id)
    if report.status == "blocked":
        raise ValueError("PDF 来源映射检查未通过，已保存解析资产和质量报告，请在解析检查页核对后重新导入")
    print_step("PDF 解析完成", 页数=parsed.page_count, 元素数=len(parsed.elements),
               片段数=len(chunks), 警告=parsed.warnings, 解析器=PARSER_VERSION)
    message = "；".join(parsed.warnings[:5])
    if report.status == "review":
        message = f"解析质量检查发现 {len(report.issues)} 项待复核，请到“解析检查”页面查看。" + message
    return chunks, message
