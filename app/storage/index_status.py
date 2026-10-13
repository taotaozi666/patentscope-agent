"""只读取当前配置对应的 Chroma 集合，检查报告片段是否实际存在。"""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.schemas import Settings
from .storage import collection_name


def pdf_index_status(settings: Settings, run_id: str, chunk_ids: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    if not (Path(settings.chroma_directory) / "chroma.sqlite3").exists():
        return {"语义集合": "未建立", "离线集合": "未建立"}
    client = chromadb.PersistentClient(path=settings.chroma_directory, settings=ChromaSettings(anonymized_telemetry=False))
    available = {collection.name for collection in client.list_collections()}
    expected = set(chunk_ids)
    for offline, label in ((False, "语义集合"), (True, "离线集合")):
        name = collection_name(settings, offline)
        if name not in available:
            result[label] = "未建立"
            continue
        records = client.get_collection(name).get(where={"pdf_parse_id": run_id}, include=["metadatas"])
        actual = {str(item.get("pdf_chunk_id", "")) for item in records["metadatas"] or []}
        found = len(expected & actual)
        result[label] = "未入库" if not found else f"完整（{found}/{len(expected)}）" if expected <= actual else f"不完整（{found}/{len(expected)}）"
    return result
