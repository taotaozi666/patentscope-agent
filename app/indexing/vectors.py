"""在写入向量库前拒绝不完整或无效的向量批次。"""

import math
from numbers import Real


def validate_vectors(vectors):
    dimension = None
    for vector in vectors:
        if not len(vector) or (dimension is not None and len(vector) != dimension):
            raise ValueError("嵌入向量为空或维度不一致")
        dimension = len(vector)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in vector):
            raise ValueError("嵌入向量包含非有限数值")
