"""离线检查当前解释器中的页面与 CLI，可用于源码环境和独立 wheel 环境。"""

import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from unittest.mock import patch

import app
import frontend
from app.cli import main
from app.schemas import Settings
from streamlit.testing.v1 import AppTest


def check_runtime() -> None:
    target = Path(frontend.__file__).parent / "ui.py"
    print(f"Application: {app.__file__}", flush=True)
    print(f"Frontend: {target}", flush=True)
    assert target.is_file(), "安装包缺少前端页面"
    with patch("app.cli.cli.subprocess.run") as run:
        main()
    command = run.call_args.args[0]
    assert Path(command[-1]).resolve() == target.resolve()

    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        settings = Settings(
            api_key="", chat_model="", embedding_model="",
            chroma_directory=str(Path(directory) / "chroma"),
            ingestion_directory=str(Path(directory) / "ingestion"),
            pdf_directory=str(Path(directory) / "pdf"),
        )
        with patch("app.config.get_settings", return_value=settings):
            page = AppTest.from_file(str(target), default_timeout=30).run()
            assert not page.exception, page.exception
            page.chat_input[0].set_value("你好").run()
            assert not page.exception, page.exception
            assert page.session_state["chat_history"].messages()
            for name in ("知识库查询", "解析检查"):
                page.sidebar.radio(key="current_page").set_value(name).run()
                assert not page.exception, page.exception
        print("PASS: page execution, greeting and navigation (offline)", flush=True)

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ)
        env.update(STREAMLIT_SERVER_HEADLESS="true", STREAMLIT_SERVER_ADDRESS="127.0.0.1",
                   STREAMLIT_SERVER_PORT=str(port), STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
        launcher = Path(sys.executable).parent / ("patentscope-agent.exe" if os.name == "nt" else "patentscope-agent")
        assert launcher.is_file(), "缺少项目命令入口"
        log_path = Path(directory) / "server.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(launcher)], cwd=directory, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt")
            print(f"Temporary server PID: {process.pid}", flush=True)
            try:
                opener = build_opener(ProxyHandler({}))
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text(encoding="utf-8"))
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                            if response.status == 200 and response.read() == b"ok":
                                break
                    except (URLError, TimeoutError):
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("Streamlit 启动超时")
                print("PASS: installed CLI launches Streamlit; health endpoint returns 200", flush=True)
            finally:
                # 只清理本检查启动的命令及其 Streamlit 子进程。
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   check=True, stdout=subprocess.DEVNULL)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)


if __name__ == "__main__":
    check_runtime()
