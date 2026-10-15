"""分级日志：用操作编号关联流程，详细内容仅在 DEBUG 时构建。"""

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

_operation_id: ContextVar[str] = ContextVar("operation_id", default="独立调用")
_timezone = ZoneInfo("Asia/Shanghai")
_logger = logging.getLogger(__name__)


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, _timezone).strftime("%H:%M:%S")
        operation = getattr(record, "operation_id", _operation_id.get())
        payload = json.dumps(getattr(record, "details", {}), ensure_ascii=False, default=str)
        message = f"[{timestamp}][{operation}][{record.getMessage()}][{record.levelname}] {payload}"
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return message


class _ConsoleHandler(logging.Handler):
    _application_console = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # 使用当前 stdout；Handler 的锁避免线程同时写入，不逐条强制刷新。
            print(self.format(record))
        except Exception:
            self.handleError(record)


def configure_logging(level: str = "INFO") -> None:
    """只配置应用日志，不改变第三方库日志；重复调用不增加 Handler。"""
    level = level.upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("LOG_LEVEL 必须为 DEBUG、INFO、WARNING 或 ERROR")
    for name in ("app", "frontend"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        if not any(getattr(handler, "_application_console", False) for handler in logger.handlers):
            handler = _ConsoleHandler()
            handler.setFormatter(_Formatter())
            logger.addHandler(handler)


def _log_step(step: str, level: int, details: dict[str, object]) -> None:
    if _logger.isEnabledFor(level):
        _logger.log(level, step, extra={"operation_id": _operation_id.get(), "details": details})


def print_step(step: str, *, level: int = logging.INFO, **details: object) -> None:
    """输出摘要；关闭的级别不序列化数据，调用方不得传入密钥。"""
    _log_step(step, level, details)


def debug_step(step: str, payload: Callable[[], dict[str, object]]) -> None:
    """仅 DEBUG 开启时构造原文、消息和向量预览等详细数据。"""
    if _logger.isEnabledFor(logging.DEBUG):
        _log_step(step, logging.DEBUG, payload())


@contextmanager
def trace_operation(name: str, **details: object) -> Iterator[None]:
    """关联步骤并记录耗时，失败只记录异常类型，不输出第三方异常正文。"""
    token = _operation_id.set(uuid4().hex[:8])
    start = perf_counter()
    _log_step(f"{name}开始", logging.INFO, details)
    try:
        yield
    except Exception as exc:
        print_step(f"{name}失败", level=logging.ERROR, 异常类型=type(exc).__name__)
        raise
    finally:
        print_step(f"{name}结束", 耗时秒=round(perf_counter() - start, 3))
        _operation_id.reset(token)


configure_logging()

