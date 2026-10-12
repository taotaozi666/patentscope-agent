"""通过项目命令启动 Streamlit 聊天服务。"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """使用当前 Python 环境启动页面，避免依赖来自其他解释器。"""
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve().parents[2] / "frontend" / "ui.py"),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        # Ctrl+C 会同时通知启动进程和 Streamlit；这是正常退出行为。
        # subprocess.run 会回收它创建的子进程，不继续向外抛中断堆栈。
        return
