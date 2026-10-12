"""storage 模块的公共接口。"""

from .storage import ChromaKnowledgeStore, KnowledgeStore, collection_name, create_store

__all__ = ["KnowledgeStore", "ChromaKnowledgeStore", "create_store", "collection_name"]
