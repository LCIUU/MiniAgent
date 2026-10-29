"""文件修改前自动备份 + 恢复（纯本地，不依赖 Git）。

每次 write/edit/apply_patch/delete/move 之前，把原文件复制到工作区
``.agent_backups/<时间戳>/<相对路径>``，同目录写 ``meta.json``。
`restore_backup` 工具可按备份 id 或最近一次备份恢复，给非 Git 项目一个兜底撤销能力。
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_effective
from .paths import PathOutsideWorkspace, is_within, resolve_workspace_path

logger = logging.getLogger(__name__)

BACKUP_DIR = ".agent_backups"
KEEP_BACKUPS = 100


def _enabled() -> bool:
    try:
        return bool(get_effective("enable_file_backups", True))
    except Exception:
        return True


def _root(workspace: str) -> Path:
    return Path(workspace).resolve() / BACKUP_DIR


def _cleanup(root: Path) -> None:
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        for old in dirs[KEEP_BACKUPS:]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def backup_file(workspace: str, path: str | Path) -> str | None:
    """备份单个文件，返回 backup_id；文件不存在或失败时返回 None。"""
    if not _enabled():
        return None
    try:
        p = Path(path).resolve()
        ws = Path(workspace).resolve()
        if not is_within(ws, p) or not p.is_file():
            return None
        rel = p.relative_to(ws)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid.uuid4().hex[:6]
        dest_dir = _root(workspace) / stamp
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        (dest_dir / "meta.json").write_text(
            json.dumps({"backup_id": stamp, "original": rel.as_posix(), "created_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False),
            encoding="utf-8",
        )
        _cleanup(_root(workspace))
        return stamp
    except (OSError, ValueError, PathOutsideWorkspace) as e:
        logger.debug("文件备份失败（跳过）: %s", e)
        return None


def list_backups(workspace: str, limit: int = 20) -> list[dict[str, Any]]:
    root = _root(workspace)
    if not root.is_dir():
        return []
    items = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        meta_path = d / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta["path"] = str(d)
        items.append(meta)
        if len(items) >= limit:
            break
    return items


def restore_backup(workspace: str, backup_id: str = "") -> str:
    """恢复指定/最近一次备份，返回结果说明。

    恢复前先把当前文件备份到 .agent_backups/pre-restore-<ts>/ ，
    避免用户选错备份后当前版本永久丢失。
    """
    root = _root(workspace)
    if not root.is_dir():
        return "没有可恢复的备份。"
    if backup_id:
        target_dir = root / backup_id
        if not target_dir.is_dir():
            return f"备份不存在: {backup_id}"
    else:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        if not dirs:
            return "没有可恢复的备份。"
        target_dir = dirs[0]
    try:
        meta = json.loads((target_dir / "meta.json").read_text(encoding="utf-8"))
        original = str(meta.get("original", ""))
        if not original:
            return "备份元信息损坏，缺少 original 字段。"
        src = target_dir / original
        dest = resolve_workspace_path(workspace, original)
        if not src.is_file():
            return f"备份文件缺失: {src}"

        # 恢复前先备份当前版本
        if dest.is_file():
            import time
            pre_dir = root / f"pre-restore-{int(time.time() * 1000)}"
            pre_dir.mkdir(parents=True, exist_ok=True)
            pre_meta = {"original": original, "ts": time.time()}
            (pre_dir / "meta.json").write_text(json.dumps(pre_meta, ensure_ascii=False), encoding="utf-8")
            pre_dest = pre_dir / original
            pre_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, pre_dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return f"已从备份 {target_dir.name} 恢复: {original}"
    except (OSError, ValueError, PathOutsideWorkspace) as e:
        return f"恢复失败: {e}"
