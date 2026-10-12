"""文档加载和切分的公共接口。"""

from .documents import LoaderRegistry, UTF8DocumentLoader, loaders
from .splitters import split_documents, split_into_chunks

__all__ = ["LoaderRegistry", "UTF8DocumentLoader", "loaders", "split_documents", "split_into_chunks"]
