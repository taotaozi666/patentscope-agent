"""嵌入创建入口；离线演示与语义模型均实现 LangChain Embeddings。"""

import hashlib
import math
import re
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import pairwise

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.config import get_settings
from app.models import create_client
from app.trace import print_step
from app.schemas import Settings


class SemanticEmbedder(Embeddings):
    """保留项目配置的兼容接口地址。"""

    def __init__(
        self, model_name: str | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model
        if not self.settings.api_key:
            raise ValueError("语义检索需要配置模型服务的 API_KEY")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents_with_progress(texts)

    def embed_documents_with_progress(
        self,
        texts: list[str],
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """复用单个 HTTP 客户端分批向量化，并可报告批次进度。"""
        vectors: list[list[float]] = []
        if not texts:
            return vectors
        batch_size = 10
        total_batches = (len(texts) + batch_size - 1) // batch_size
        total_characters = sum(len(text) for text in texts)
        worker_count = min(self.settings.embedding_concurrency, total_batches)
        print_step(
            "向量接口请求开始",
            模型=self.model_name,
            文本数量=len(texts),
            总字符数=total_characters,
            批次数=total_batches,
            每批数量=batch_size,
            并发数=worker_count,
        )
        batches = [
            texts[start : start + batch_size]
            for start in range(0, len(texts), batch_size)
        ]
        vectors = [[] for _ in texts]

        def request_batch(client: OpenAI, batch: list[str]) -> list[list[float]]:
            response = client.embeddings.create(
                model=self.model_name, input=batch, encoding_format="float"
            )
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("向量接口返回的片段索引不完整")
            normalized: list[list[float]] = []
            for item in items:
                norm = math.sqrt(sum(value * value for value in item.embedding))
                if not norm or not math.isfinite(norm):
                    raise ValueError("向量接口返回无效向量")
                normalized.append([value / norm for value in item.embedding])
            return normalized

        # 保持并发中的待处理请求有界，避免超大文件一次创建大量 Future。
        with create_client(self.settings) as client:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending: dict[Future[list[list[float]]], int] = {}
                next_batch = 0
                completed_batches = 0
                completed_texts = 0

                def submit_next() -> bool:
                    nonlocal next_batch
                    if next_batch >= total_batches:
                        return False
                    future = executor.submit(
                        request_batch, client, batches[next_batch]
                    )
                    pending[future] = next_batch
                    next_batch += 1
                    return True

                for _ in range(worker_count):
                    submit_next()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_number = pending.pop(future)
                        start = batch_number * batch_size
                        result = future.result()
                        vectors[start : start + len(result)] = result
                        completed_batches += 1
                        completed_texts += len(batches[batch_number])
                        if on_batch:
                            on_batch(completed_texts, len(texts))
                        # 并发完成顺序可能不同；按完成数量输出稀疏进度。
                        if (
                            completed_batches == 1
                            or completed_batches % 50 == 0
                            or completed_batches == total_batches
                        ):
                            print_step(
                                "向量接口批次进度",
                                已完成批次=completed_batches,
                                总批次=total_batches,
                                已处理文本=completed_texts,
                                总文本=len(texts),
                            )
                        submit_next()
        print_step(
            "向量接口返回",
            向量数量=len(vectors),
            向量长度=len(vectors[0]) if vectors else 0,
            状态="已归一化",
        )
        return vectors


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
