"""历史解析记录检查：恢复结构对象，并将报告绑定到明确的分块集合。"""

from langchain_core.documents import Document

from app.pdf.serialization import restore_pdf
from app.schemas.quality import QualityReport
from app.storage.pdf_repository import PdfRepository
from app.storage.quality_repository import QualityRepository

from .checks import check_pdf


def inspect_saved_pdf(directory: str, run_id: str) -> QualityReport:
    repository = PdfRepository(directory)
    data = repository.get_run(run_id)
    if data is None:
        raise ValueError("解析记录不存在")
    parsed = restore_pdf(data)
    rows = repository.list_chunks(run_id)
    # 新记录按预算区分；历史记录没有预算字段时按默认 600 处理。
    # 只检查最后保存的一套配置，不能把不同预算的分块混成一份报告。
    size = int(rows[-1]["metadata"].get("pdf_chunk_size", 600)) if rows else 600
    def profile(metadata: dict) -> tuple:
        return (metadata.get("pdf_chunk_size", 600), metadata.get("pdf_target_size", 0),
                metadata.get("pdf_context_mode", "rule"), metadata.get("pdf_context_model", ""))
    latest_profile = profile(rows[-1]["metadata"]) if rows else None
    chunks = [Document(page_content=row["text"], metadata=row["metadata"]) for row in rows
              if profile(row["metadata"]) == latest_profile]
    report = check_pdf(parsed, chunks, size)
    QualityRepository(directory).save(report)
    return report
