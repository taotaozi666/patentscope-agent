"""检查启动命令指向实际存在的前端页面。"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from app.cli import main


class CLITests(unittest.TestCase):
    def test_ctrl_c_exits_normally(self):
        with patch("app.cli.subprocess.run", side_effect=KeyboardInterrupt):
            main()

    def test_actual_launch_failure_is_not_silenced(self):
        failure = subprocess.CalledProcessError(2, ["streamlit", "run"])
        with (
            patch("app.cli.subprocess.run", side_effect=failure),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            main()

    def test_entry_points_to_existing_frontend(self):
        with patch("app.cli.subprocess.run") as run:
            main()
        target = Path(run.call_args.args[0][-1])
        self.assertTrue(target.is_file())
        self.assertEqual(
            target, Path(__file__).resolve().parents[1] / "frontend" / "ui.py"
        )


if __name__ == "__main__":
    unittest.main()
