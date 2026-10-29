"""pytest 共享配置：把临时目录固定到 tests/.tmp，避免 Windows 沙箱下 TEMP 不可写
导致 tempfile 回退到项目根目录，每次测试都留下一个 tmpXXXX 目录。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TMP = Path(__file__).resolve().parent / ".tmp"
_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP)
os.environ.setdefault("TMP", str(_TMP))
os.environ.setdefault("TEMP", str(_TMP))


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    shutil.rmtree(_TMP, ignore_errors=True)
