"""工作区路径安全：统一解析、边界校验、Windows 大小写/符号链接处理。

所有对用户提供的路径（工具参数、上传目录、附件路径）都应经过这里，
避免 ``terminal(cwd="..")`` 这类越界访问。
"""

from __future__ import annotations

import os
from pathlib import Path


class PathOutsideWorkspace(PermissionError):
    """目标路径不在工作区内。"""


def _norm(path: str | Path) -> str:
    """规范化到真实路径 + 平台大小写规则（Windows 不区分大小写）。"""
    return os.path.normcase(os.path.realpath(str(path)))


def is_within(workspace: str | Path, target: str | Path) -> bool:
    """target 是否等于 workspace 或位于其子目录内。"""
    if not workspace:
        return False
    ws = _norm(workspace)
    tp = _norm(target)
    return tp == ws or tp.startswith(ws + os.sep)


def ensure_workspace(workspace: str) -> str:
    """校验并返回工作区绝对路径（必须是已存在的目录）。"""
    if not workspace or not str(workspace).strip():
        raise ValueError("未设置工作区")
    p = Path(workspace).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"工作区不存在或不是目录: {p}")
    return str(p)


def resolve_workspace_path(
    workspace: str,
    path: str = "",
    *,
    must_exist: bool = False,
) -> Path:
    """把相对/绝对路径解析到工作区内，越界抛 PathOutsideWorkspace。

    - 相对路径基于工作区根目录
    - 绝对路径必须仍位于工作区内
    - ``resolve()`` 会展开符号链接，防止通过链接逃逸
    """
    try:
        ws = Path(workspace).expanduser().resolve()
    except OSError as e:
        raise ValueError(f"工作区无效: {e}") from e
    raw = (path or "").strip()
    p = Path(raw) if raw else ws
    if not p.is_absolute():
        p = ws / p
    try:
        p = p.resolve()
    except OSError as e:
        raise ValueError(f"路径无效: {e}") from e
    if not is_within(ws, p):
        raise PathOutsideWorkspace(f"路径超出工作区，禁止访问: {p}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"路径不存在: {p}")
    return p


def relative_to_workspace(workspace: str, path: str | Path) -> str:
    """返回相对工作区的展示路径；越界时返回原路径。"""
    try:
        return os.path.relpath(_norm(path), _norm(workspace))
    except ValueError:
        return str(path)
