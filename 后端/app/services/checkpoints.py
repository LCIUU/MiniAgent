"""工作区级快照：整仓打包与回滚。

比 ``.agent_backups`` 的单文件备份更进一步：批量脚本 / 大范围替换前建一个
安全点，出问题可整体回退。快照存放于 ``.agent_backups/checkpoints``，
默认保留最近 ``KEEP_CHECKPOINTS`` 个，自动忽略依赖/构建等大目录。
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import PathOutsideWorkspace, resolve_workspace_path

CHECKPOINT_DIR = ".agent_backups/checkpoints"
KEEP_CHECKPOINTS = 10
EXCLUDED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", ".nuxt", "target", ".tox", ".mypy_cache", ".pytest_cache",
    ".agent_logs", ".agent_uploads", ".agent_backups", "coverage", ".turbo",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 300 * 1024 * 1024
MAX_FILES = 20000


def _root(workspace: str) -> Path:
    return Path(workspace).resolve() / CHECKPOINT_DIR


def _iter_files(workspace: str):
    """遍历可快照的文件，返回 (Path, 相对 posix 路径)；大文件/排除目录跳过。"""
    ws = Path(workspace).resolve()
    for root, dirs, names in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".agent_")]
        for name in names:
            path = Path(root) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path, path.relative_to(ws).as_posix()


def create_checkpoint(workspace: str, label: str = "") -> dict[str, Any]:
    """创建整仓快照，返回元信息；工作区过大时返回 ok=False 由调用方降级。"""
    ws = Path(workspace).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    cp_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    root = _root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{cp_id}.zip"

    files = 0
    total = 0
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path, rel in _iter_files(workspace):
                size = path.stat().st_size
                if files + 1 > MAX_FILES or total + size > MAX_TOTAL_BYTES:
                    raise _TooLarge(
                        f"工作区过大（>{MAX_FILES} 个文件或 >{MAX_TOTAL_BYTES // (1024 * 1024)}MB）"
                    )
                zf.write(path, rel)
                files += 1
                total += size
    except _TooLarge as e:
        zip_path.unlink(missing_ok=True)
        return {"ok": False, "error": str(e)}
    except (OSError, zipfile.BadZipFile) as e:
        zip_path.unlink(missing_ok=True)
        return {"ok": False, "error": f"快照创建失败: {e}"}

    meta = {
        "checkpoint_id": cp_id,
        "label": label.strip()[:120],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
        "bytes": total,
        "workspace": str(ws),
    }
    try:
        (root / f"{cp_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass
    _cleanup(root)
    return {"ok": True, **meta}


class _TooLarge(Exception):
    pass


def _cleanup(root: Path) -> None:
    try:
        zips = sorted(root.glob("*.zip"), key=lambda p: p.name, reverse=True)
        for old in zips[KEEP_CHECKPOINTS:]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    except OSError:
        pass


def list_checkpoints(workspace: str, limit: int = 20) -> list[dict[str, Any]]:
    root = _root(workspace)
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for z in sorted(root.glob("*.zip"), key=lambda p: p.name, reverse=True):
        meta_path = z.with_suffix(".json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {"checkpoint_id": z.stem, "created_at": ""}
        meta["size"] = z.stat().st_size
        items.append(meta)
        if len(items) >= limit:
            break
    return items


def restore_checkpoint(workspace: str, checkpoint_id: str = "") -> dict[str, Any]:
    """恢复快照中的文件（覆盖同名文件，不删除快照之外的新文件）。"""
    root = _root(workspace)
    if not root.is_dir():
        return {"ok": False, "error": "没有可用的快照。"}
    if checkpoint_id:
        zip_path = root / f"{checkpoint_id}.zip"
        if not zip_path.is_file():
            return {"ok": False, "error": f"快照不存在: {checkpoint_id}"}
    else:
        zips = sorted(root.glob("*.zip"), key=lambda p: p.name, reverse=True)
        if not zips:
            return {"ok": False, "error": "没有可用的快照。"}
        zip_path = zips[0]

    restored = 0
    failed: list[str] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = info.filename
                try:
                    dest = resolve_workspace_path(workspace, rel)
                except (PathOutsideWorkspace, ValueError):
                    failed.append(f"{rel}（路径越界，已跳过）")
                    continue
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    restored += 1
                except OSError as e:
                    failed.append(f"{rel}: {e}")
    except (OSError, zipfile.BadZipFile) as e:
        return {"ok": False, "error": f"快照读取失败: {e}"}

    return {
        "ok": True,
        "checkpoint_id": zip_path.stem,
        "restored": restored,
        "failed": failed[:10],
        "failed_total": len(failed),
    }


def latest_checkpoint_id(workspace: str) -> str:
    root = _root(workspace)
    if not root.is_dir():
        return ""
    zips = sorted(root.glob("*.zip"), key=lambda p: p.name, reverse=True)
    return zips[0].stem if zips else ""
