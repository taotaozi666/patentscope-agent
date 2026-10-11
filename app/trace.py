"""后台链路打印：使用操作编号关联同一轮导入或聊天的各个步骤。"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

_operation_id: ContextVar[str] = ContextVar("operation_id", default="独立调用")


def print_step(step: str, **details: object) -> None:
    """打印步骤及结果；调用方不得传入密钥、请求头或完整配置对象。"""
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%H:%M:%S")
    # 一条步骤占一行，正文中的换行以转义形式输出，减少并发时日志混杂。
    payload = json.dumps(details, ensure_ascii=False, default=str)
    print(f"[{timestamp}][{_operation_id.get()}][{step}] {payload}", flush=True)


@contextmanager
def trace_operation(name: str, **details: object) -> Iterator[None]:
    """创建一次操作的打印上下文；异常时打印失败类型并继续抛出异常。"""
    token = _operation_id.set(uuid4().hex[:8])
    start = perf_counter()
    print_step(f"{name}开始", **details)
    try:
        yield
    except Exception as exc:
        # 不打印第三方异常正文，避免其中夹带请求密钥。
        print_step(f"{name}失败", 异常类型=type(exc).__name__)
        raise
    finally:
        print_step(f"{name}结束", 耗时秒=round(perf_counter() - start, 3))
        _operation_id.reset(token)
