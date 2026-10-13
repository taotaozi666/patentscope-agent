"""解析质量检查：规则检查及人工复核编排，不调用模型。"""

from .checks import check_pdf

__all__ = ["check_pdf"]
