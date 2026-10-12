"""rag 模块的公共接口。"""

from .rag import (RAGService, answer)
from .routing import direct_reply

__all__ = ['RAGService', 'answer', 'direct_reply']
