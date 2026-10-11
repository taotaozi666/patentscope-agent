"""统一导出聊天历史处理接口。"""

from app.history.history import (
    ChatHistory,
    as_history,
    get_session_history,
)

__all__ = ["ChatHistory", "as_history", "get_session_history"]
