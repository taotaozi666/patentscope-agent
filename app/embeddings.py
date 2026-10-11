"""嵌入创建入口；演示与千问模型均实现 LangChain Embeddings。"""

import hashlib
import math
import re
from itertools import pairwise

from langchain_core.embeddings import Embeddings

from app.models import SemanticEmbedder
from app.schemas import Settings


class DemoEmbedder(Embeddings):
    """将字符词频哈希到固定维度，用于无需密钥的离线流程演示。"""

    dimensions = 2048

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
            tokens = words + [a + b for a, b in pairwise(words)]
            vector = [0.0] * self.dimensions
            for token in tokens:
                # 使用稳定哈希，跨进程重启后同一文本仍得到相同向量。
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8])
                vector[index % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embeddings(settings: Settings, offline: bool = False) -> Embeddings:
    return DemoEmbedder() if offline else SemanticEmbedder(settings=settings)


def collection_name(settings: Settings, offline: bool = False) -> str:
    """按模型和服务地址分集合，密钥更换不会丢失已有知识库。"""
    identity = (
        "demo-hash-v1-2048"
        if offline
        else f"qwen-compatible:{settings.base_url}:{settings.embedding_model}"
    )
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"{settings.chroma_collection}_{suffix}"
