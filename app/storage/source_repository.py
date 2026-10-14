"""按内容哈希保存源文件，供知识库列表查看和下载。"""

from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4


class SourceRepository:
    def __init__(self, directory: str | Path) -> None:
        # 与已有 PDF 原件共用目录，历史 PDF 无需迁移。
        self.originals = Path(directory) / "originals"

    def _path(self, content_hash: str, file_format: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", content_hash):
            raise ValueError("无效的源文件标识")
        if file_format not in {".txt", ".md", ".pdf"}:
            raise ValueError("不支持该源文件格式")
        return self.originals / f"{content_hash}{file_format}"

    def save(self, content: bytes, file_format: str) -> None:
        path = self._path(sha256(content).hexdigest(), file_format)
        self.originals.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            return
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, content_hash: str, file_format: str) -> bytes:
        return self._path(content_hash, file_format).read_bytes()
