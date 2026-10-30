"""跨文件批量替换：先预览、后落盘，替代模型写一次性修复脚本。

默认 dry_run=True 只返回命中预览；确认后传 dry_run=False 才会写入。
落盘前自动创建工作区快照 + 逐文件备份，出问题可整体回滚。
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import re
from pathlib import Path

from langchain_core.tools import tool

from ..services.backups import backup_file
from ..services.checkpoints import create_checkpoint
from ..services.file_io import (
    BinaryFileError,
    TextDocument,
    async_read_document,
    async_write_document,
)
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
_PREVIEW_LIMIT = 30


def _rel(workspace: str, p: Path) -> str:
    try:
        return os.path.relpath(str(p), workspace).replace("\\", "/")
    except ValueError:
        return str(p)


def _candidates(workspace: str, base: Path, include: str) -> list[Path]:
    files: list[Path] = []
    if base.is_file():
        return [base]
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


async def _emit_diff(ctx: ToolContext, path: Path, before: str, after: str) -> None:
    rel = _rel(ctx.workspace, path)
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
        )
    )
    if not diff:
        return
    await ctx.emit_event({
        "type": "diff",
        "call_id": ctx.call_id,
        "path": rel,
        "diff": diff[:12000],
        "added": sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")),
        "removed": sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")),
    })


def make_bulk_edit_tool(ctx: ToolContext):

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=120)
    @tool(parse_docstring=True)
    async def replace_in_files(
        pattern: str,
        replacement: str = "",
        include: str = "",
        path: str = "",
        regex: bool = False,
        word_boundary: bool = False,
        dry_run: bool = True,
        max_files: int = 200,
    ) -> str:
        """
        在多个文件中批量查找并替换文本（默认只预览，不写入）

        适用：重命名函数/变量、批量改路径、统一替换文案等跨文件重复修改。替代"写一次性
        脚本批量改文件"：先 dry_run 预览命中位置，确认后再传 dry_run=false 落盘。

        默认 dry_run=true，只返回命中文件数、替换次数和前 30 处预览，不修改任何文件。
        落盘前会创建整仓快照并逐文件备份；自动跳过二进制文件和依赖/构建目录；单个文件上限 2MB，
        候选文件数超过 max_files 时直接报错，需用 include 或 path 缩小范围。regex=true 时
        replacement 支持 \\1 等反向引用；word_boundary=true 时按整词匹配。

        Args:
            pattern: 要查找的文本；regex=true 时按正则表达式处理。
            replacement: 替换后的文本；留空表示删除匹配内容。
            include: 文件过滤 glob，例如 "*.py" 或 "src/**/*.tsx"；留空表示所有文本文件。
            path: 搜索起始目录或文件，相对当前工作区；留空表示整个工作区。
            regex: 是否按正则匹配，默认 False 按纯文本。
            word_boundary: 是否要求整词匹配，默认 False；可避免 user 命中 username。
            dry_run: 默认 True 只预览；确认无误后设为 False 才会写入。
            max_files: 最多处理的候选文件数，默认 200，上限 1000。

        Returns:
            预览命中清单，或写入结果（替换文件数、总处数、快照 id）；失败时返回原因。
        """
        ctx.check_stopped()
        if not pattern:
            return "错误：pattern 不能为空。"
        try:
            base = resolve_workspace_path(ctx.workspace, path or ".")
        except (PathOutsideWorkspace, ValueError) as e:
            return f"错误：{e}"
        if not base.exists():
            return f"错误：路径不存在: {path}"

        try:
            if regex or word_boundary:
                source = rf"\b(?:{pattern})\b" if word_boundary else pattern
                compiled = re.compile(source)
            else:
                compiled = None
        except re.error as e:
            return f"正则错误: {e}（如需按纯文本匹配，请设 regex=false 且 word_boundary=false）"

        limit_files = max(1, min(int(max_files), 1000))
        candidates = _candidates(ctx.workspace, base, include)
        if len(candidates) > limit_files:
            return (
                f"匹配到 {len(candidates)} 个候选文件，超过上限 {limit_files}。"
                f"请用 include 或 path 缩小范围，或调大 max_files。"
            )

        previews: list[str] = []
        planned: list[tuple[Path, TextDocument, str, str, int]] = []
        skipped_binary = 0
        for p in candidates:
            ctx.check_stopped()
            try:
                doc = await async_read_document(p)
            except BinaryFileError:
                skipped_binary += 1
                continue
            except (ValueError, OSError):
                continue
            text = doc.text
            if not text:
                continue
            try:
                if compiled is not None:
                    count = len(compiled.findall(text))
                    new_text = compiled.sub(replacement, text)
                else:
                    count = text.count(pattern)
                    new_text = text.replace(pattern, replacement)
            except (re.error, IndexError) as e:
                return f"替换失败：{_rel(ctx.workspace, p)}: {e}"
            if count == 0:
                continue
            planned.append((p, doc, text, new_text, count))
            if len(previews) < _PREVIEW_LIMIT:
                for lineno, line in enumerate(text.splitlines(), 1):
                    if (compiled.search(line) if compiled is not None else pattern in line):
                        previews.append(f"{_rel(ctx.workspace, p)}:{lineno}: {line.strip()[:180]}")
                        if len(previews) >= _PREVIEW_LIMIT:
                            break

        total = sum(item[4] for item in planned)
        if not planned:
            extra = f"（跳过 {skipped_binary} 个二进制文件）" if skipped_binary else ""
            return f"没有找到匹配内容{extra}。"

        head = (
            f"命中 {len(planned)} 个文件，共 {total} 处替换"
            + (f"（跳过 {skipped_binary} 个二进制文件）" if skipped_binary else "")
        )
        body = "\n".join(previews)
        truncated = (
            f"\n...(预览仅显示前 {_PREVIEW_LIMIT} 处)" if len(previews) >= _PREVIEW_LIMIT else ""
        )
        if dry_run:
            return (
                f"【预览】{head}\n{body}{truncated}\n\n"
                "确认无误后，用相同参数并把 dry_run 设为 false 才会写入。"
            )

        approved = await ctx.approvals.request(
            ctx.emit_event,
            "replace_in_files",
            "normal",
            {
                "command": f"批量替换 {len(planned)} 个文件（{total} 处）: {pattern[:60]}",
                "cwd": ctx.workspace,
            },
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了批量替换。"

        snapshot = await asyncio.to_thread(
            create_checkpoint, ctx.workspace, f"pre-replace:{pattern[:60]}"
        )
        changed = 0
        for p, doc, before, after, _count in planned:
            ctx.check_stopped()
            backup_file(ctx.workspace, p)
            try:
                await async_write_document(
                    p, after,
                    encoding=doc.encoding, newline=doc.newline, bom=doc.bom,
                )
            except OSError as e:
                return f"写入失败 {_rel(ctx.workspace, p)}: {e}（已修改 {changed} 个文件）"
            changed += 1
            await _emit_diff(ctx, p, before, after)

        lines = [f"已替换 {changed} 个文件，共 {total} 处。"]
        if snapshot.get("ok"):
            lines.append(f"替换前快照：{snapshot['checkpoint_id']}（可用 workspace_checkpoint 回滚）")
        else:
            lines.append(f"快照未创建（{snapshot.get('error')}），但每个文件均已备份到 .agent_backups")
        return "\n".join(lines)

    return replace_in_files
