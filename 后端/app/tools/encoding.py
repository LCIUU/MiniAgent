"""编码体检工具：找出非 UTF-8 / 乱码文件，可选批量转成 UTF-8。

针对历史事故（GBK 文件被当成 UTF-8 处理后整仓语法损坏）：
- 只报告时不动文件
- fix=true 仅转换"UTF-8 解码失败但 GBK 可解码"的文件，且先建快照 + 逐文件备份
- 已是合法 UTF-8 但内容像二次编码乱码的文件只报告，不强改（无法可靠还原）
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from ..services.backups import backup_file
from ..services.checkpoints import create_checkpoint
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec

_IGNORED = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", ".nuxt", "target", ".tox", ".mypy_cache", ".pytest_cache",
    ".agent_logs", ".agent_uploads", ".agent_backups", "coverage", ".turbo",
}
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_FILES = 5000
_SAMPLE_LIMIT = 40
# 合法 UTF-8 但内容像"UTF-8 被按 Latin-1/GBK 再编码"的典型片段
_MOJIBAKE_MARKERS = (
    "锟斤拷", "\ufffd", "ï¿½", "Ã¤", "Ã©", "Ã¶", "Ã¼", "ÃŸ",
    "â€™", "â€œ", "â€\x9d", "Â«", "Â»",
)


def _rel(workspace: str, p: Path) -> str:
    try:
        return os.path.relpath(str(p), workspace).replace("\\", "/")
    except ValueError:
        return str(p)


def _candidates(workspace: str, base: Path, include: str) -> list[Path]:
    if base.is_file():
        return [base]
    files: list[Path] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _IGNORED and not d.startswith(".agent_")]
        for name in names:
            p = Path(root) / name
            rel = _rel(workspace, p)
            if include and not (
                fnmatch.fnmatch(rel, include) or fnmatch.fnmatch(name, include)
            ):
                continue
            try:
                if p.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(p)
            if len(files) >= _MAX_FILES:
                return files
    return files


def _classify(data: bytes) -> str:
    """返回 utf-8 | utf-8-bom | gbk | mojibake | binary | unknown。"""
    if b"\x00" in data[:4096]:
        return "binary"
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            data[3:].decode("utf-8")
            return "utf-8-bom"
        except UnicodeDecodeError:
            return "unknown"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            data.decode("gbk")
            return "gbk"
        except UnicodeDecodeError:
            return "unknown"
    if any(marker in text for marker in _MOJIBAKE_MARKERS):
        return "mojibake"
    return "utf-8"


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".agent-enc-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def make_encoding_tool(ctx: ToolContext):

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=120)
    @tool(parse_docstring=True)
    async def check_encoding(
        path: str = "",
        include: str = "",
        fix: bool = False,
        max_files: int = 2000,
    ) -> str:
        """
        检查文件编码（UTF-8 / UTF-8 BOM / GBK / 乱码），可选把 GBK 文件转成 UTF-8

        适用：遇到中文乱码、跨平台编辑后语法报错、历史项目编码不统一时。默认只检查不改文件；
        fix=true 时只转换"UTF-8 解码失败但 GBK 可解码"的文件，转换前创建整仓快照并逐文件备份。

        已是合法 UTF-8 但内容像二次编码乱码（如"锟斤拷"）的文件只报告、不强改，因为无法可靠
        还原；UTF-8 BOM 文件也只报告（一般无害）。扫描忽略依赖/构建目录，单文件上限 2MB，
        最多扫描 max_files 个文件（上限 5000）。

        Args:
            path: 可选，检查起始目录或文件，相对当前工作区；留空表示整个工作区。
            include: 可选，文件过滤 glob，例如 "*.js" 或 "src/**"。
            fix: 是否把 GBK 文件转换为 UTF-8，默认 False 只检查。
            max_files: 最多扫描的文件数，默认 2000，上限 5000。

        Returns:
            编码分布与问题文件清单；fix=true 时返回转换结果与快照 id。
        """
        ctx.check_stopped()
        try:
            base = resolve_workspace_path(ctx.workspace, path or ".")
        except (PathOutsideWorkspace, ValueError) as e:
            return f"错误：{e}"
        if not base.exists():
            return f"错误：路径不存在: {path}"

        candidates = _candidates(ctx.workspace, base, include)
        limit = max(1, min(int(max_files), _MAX_FILES))
        if len(candidates) > limit:
            candidates = candidates[:limit]

        groups: dict[str, list[Path]] = {}
        for p in candidates:
            ctx.check_stopped()
            try:
                data = p.read_bytes()
            except OSError:
                continue
            kind = _classify(data)
            if kind == "binary":
                continue
            if kind == "utf-8":
                continue
            groups.setdefault(kind, []).append(p)

        if not groups:
            return f"已检查 {len(candidates)} 个文件：全部是合法 UTF-8，未发现编码问题。"

        lines = [f"已检查 {len(candidates)} 个文件，发现 {sum(len(v) for v in groups.values())} 个非标准编码文件："]
        titles = {
            "gbk": "GBK 编码（UTF-8 解码失败，可转换）",
            "utf-8-bom": "UTF-8 带 BOM（一般无害，部分工具不识别 BOM）",
            "mojibake": "疑似乱码（本身是合法 UTF-8，但内容像二次编码，无法自动还原）",
            "unknown": "无法识别编码（既不是 UTF-8 也不是 GBK）",
        }
        for kind in ("gbk", "mojibake", "unknown", "utf-8-bom"):
            items = groups.get(kind) or []
            if not items:
                continue
            lines.append(f"\n{titles[kind]}：{len(items)} 个")
            for p in items[:_SAMPLE_LIMIT]:
                lines.append(f"  {_rel(ctx.workspace, p)}")
            if len(items) > _SAMPLE_LIMIT:
                lines.append(f"  ...（其余 {len(items) - _SAMPLE_LIMIT} 个省略）")

        gbk_files = groups.get("gbk") or []
        if not fix:
            if gbk_files:
                lines.append(
                    '\n如需把 GBK 文件统一转成 UTF-8，用相同参数并设 fix=true（会先建快照并逐文件备份）。'
                )
            return "\n".join(lines)

        if not gbk_files:
            lines.append("\nfix=true 只处理 GBK 文件；本次没有可自动转换的文件。")
            return "\n".join(lines)

        approved = await ctx.approvals.request(
            ctx.emit_event,
            "check_encoding",
            "normal",
            {"command": f"把 {len(gbk_files)} 个 GBK 文件转为 UTF-8", "cwd": ctx.workspace},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了编码转换。"

        snapshot = await asyncio.to_thread(
            create_checkpoint, ctx.workspace, "pre-encoding-fix"
        )
        converted = 0
        failed: list[str] = []
        for p in gbk_files:
            ctx.check_stopped()
            try:
                data = p.read_bytes()
                text = data.decode("gbk")
            except (OSError, UnicodeDecodeError) as e:
                failed.append(f"{_rel(ctx.workspace, p)}: {e}")
                continue
            backup_file(ctx.workspace, p)
            try:
                _write_bytes_atomic(p, text.encode("utf-8"))
                converted += 1
            except OSError as e:
                failed.append(f"{_rel(ctx.workspace, p)}: {e}")

        lines = [f"已把 {converted}/{len(gbk_files)} 个 GBK 文件转换为 UTF-8。"]
        if snapshot.get("ok"):
            lines.append(f"转换前快照：{snapshot['checkpoint_id']}（可用 workspace_checkpoint 回滚）")
        else:
            lines.append(f"快照未创建（{snapshot.get('error')}），每个文件已单独备份到 .agent_backups")
        if failed:
            lines.append(f"失败 {len(failed)} 个：" + "；".join(failed[:10]))
        return "\n".join(lines)

    return check_encoding
