"""文件工具：读 / 写 / 编辑 / 列出 / 删除 / 移动 / 复制（均限定在工作区内）。"""

from __future__ import annotations

import difflib
import os
import re
import shutil
from pathlib import Path

from langchain_core.tools import tool

from ..services.backups import backup_file, list_backups, restore_backup
from ..services.file_io import (
    MAX_TEXT_BYTES,
    BinaryFileError,
    async_read_document,
    async_read_text_range,
    async_write_document,
    async_write_new_text,
)
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_HIGH, RISK_MEDIUM, tool_spec

_IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next",
    ".agent_logs", ".agent_uploads", ".agent_backups",
}


def _resolve(ctx: ToolContext, path: str) -> Path:
    """将相对/绝对路径解析到工作区内，防止逃逸。"""
    return resolve_workspace_path(ctx.workspace, path)


def _rel(ctx: ToolContext, p: Path) -> str:
    try:
        return os.path.relpath(str(p), str(ctx.workspace)).replace("\\", "/")
    except ValueError:
        return str(p)


async def _emit_diff(ctx: ToolContext, path: Path, before: str, after: str) -> None:
    """把修改前后的 unified diff 推给前端（工具卡片可展示具体改动）。"""
    if before == after:
        return
    rel = _rel(ctx, path)
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
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


async def _ask_write(ctx: ToolContext, tool: str, action: str, path: Path) -> bool:
    """写操作审批：仅在 command_confirm_mode=all 时真正等待用户确认。"""
    return await ctx.approvals.request(
        ctx.emit_event,
        tool,
        "normal",
        {"command": f"{action} {_rel(ctx, path)}", "cwd": str(path)},
        stop_event=ctx.stop_event,
    )


def make_file_tools(ctx: ToolContext):

    @tool_spec(category="fs", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def read_file(
        path: str,
        offset: int = 1,
        limit: int = 0,
    ) -> str:
        """
        读取文本文件内容

        Args:
            path: 文件路径，相对于工作区的路径
            offset: 可选，起始行号，从 1 开始计数
            limit: 可选，最多读取的行数；0 表示读取全部（默认）

        Returns:
            返回文件内容并带行号，便于后续精准编辑
        """
        ctx.check_stopped()
        try:
            p = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if not p.is_file():
            return f"错误：文件不存在: {path}"
        start = max(1, offset)
        try:
            size = p.stat().st_size
        except OSError as e:
            return f"读取失败: {e}"
        if size > MAX_TEXT_BYTES and (start > 1 or limit > 0):
            # 大文件只按行取所需区间，不把整个文件载入内存
            try:
                rng = await async_read_text_range(p, start, limit)
            except BinaryFileError as e:
                return str(e)
            except (ValueError, OSError) as e:
                return f"读取失败: {e}"
            if start > rng.total_lines and rng.total_lines > 0:
                return (
                    f"文件: {path} | 总行数: {rng.total_lines} | "
                    f"offset={offset} 超出文件末尾 | 编码: {rng.encoding}"
                )
            end = start + len(rng.lines) - 1 if rng.lines else start - 1
            numbered = "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(rng.lines))
            return f"文件: {path} | 总行数: {rng.total_lines} | 显示 {start}-{end} | 编码: {rng.encoding}\n{numbered}"
        try:
            doc = await async_read_document(p)
        except BinaryFileError as e:
            return str(e)
        except (ValueError, OSError) as e:
            return f"读取失败: {e}"
        lines = doc.text.splitlines()
        total = len(lines)
        if start > total and total > 0:
            return (
                f"文件: {path} | 总行数: {total} | "
                f"offset={offset} 超出文件末尾 | 编码: {doc.encoding}"
            )
        end = total if limit <= 0 else min(total, start + limit - 1)
        selected = lines[start - 1 : end]
        numbered = "\n".join(f"{i + start:>5} | {line}" for i, line in enumerate(selected))
        return f"文件: {path} | 总行数: {total} | 显示 {start}-{end} | 编码: {doc.encoding}\n{numbered}"

    @tool_spec(category="fs", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def read_files(
        paths: str,
        max_lines_each: int = 0,
    ) -> str:
        """
        一次读取多个文本文件（每个文件带行号，适合同时查看组件/样式/类型等多个文件）

        Args:
            paths: 文件路径列表，用换行或逗号分隔，相对于工作区（最多 8 个）
            max_lines_each: 可选，每个文件最多读取的行数；0 表示读取全部（默认）

        Returns:
            按文件分段的内容
        """
        ctx.check_stopped()
        items: list[str] = []
        for raw in re.split(r"[\n,;]+", paths or ""):
            item = raw.strip()
            if item and item not in items:
                items.append(item)
        if not items:
            return "错误：paths 不能为空。"
        extra = len(items) - 8
        items = items[:8]
        per_file = max(0, int(max_lines_each))
        blocks: list[str] = []
        for rel_path in items:
            ctx.check_stopped()
            try:
                p = _resolve(ctx, rel_path)
            except (PathOutsideWorkspace, ValueError) as e:
                blocks.append(f"=== {rel_path} ===\n{e}")
                continue
            if not p.is_file():
                blocks.append(f"=== {rel_path} ===\n错误：文件不存在")
                continue
            try:
                doc = await async_read_document(p)
            except BinaryFileError as e:
                blocks.append(f"=== {rel_path} ===\n{e}")
                continue
            except (ValueError, OSError) as e:
                blocks.append(f"=== {rel_path} ===\n读取失败: {e}")
                continue
            lines = doc.text.splitlines()
            total = len(lines)
            selected = lines if per_file <= 0 else lines[:per_file]
            numbered = "\n".join(f"{i:>5} | {line}" for i, line in enumerate(selected, 1))
            more = f"（仅显示前 {per_file} 行）" if per_file > 0 and total > per_file else ""
            blocks.append(f"=== {rel_path}（共 {total} 行，编码 {doc.encoding}）{more} ===\n{numbered}")
        result = "\n\n".join(blocks)
        if extra > 0:
            result += f"\n\n...(另有 {extra} 个文件未读取，单次最多 8 个)"
        return result

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=60)
    @tool(parse_docstring=True)
    async def write_file(
        path: str,
        content: str,
    ) -> str:
        """
        写入文件（覆盖已有内容）

        Args:
            path: 文件路径，相对于工作区的路径，会自动创建父目录
            content: 完整的文件内容，覆盖写入

        Returns:
            返回写入结果和字符数
        """
        ctx.check_stopped()
        try:
            p = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        before = ""
        doc = None
        if p.exists():
            try:
                doc = await async_read_document(p)
                before = doc.text
            except (BinaryFileError, ValueError, OSError):
                doc = None
        if not await _ask_write(ctx, "write_file", "写入", p):
            return "用户拒绝了写入操作。"
        backup_file(ctx.workspace, p)
        try:
            if doc is not None:
                await async_write_document(p, content, encoding=doc.encoding, newline=doc.newline, bom=doc.bom)
            else:
                await async_write_new_text(p, content)
        except OSError as e:
            return f"写入失败: {e}"
        await _emit_diff(ctx, p, before, content)
        return f"【AI 写入】已写入文件 {path}（{len(content)} 字符，编码 {doc.encoding if doc else 'utf-8'}）"

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=60)
    @tool(parse_docstring=True)
    async def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        精准编辑文件，将旧内容替换为新内容

        Args:
            path: 文件路径，相对于工作区的路径
            old_string: 要被替换的旧内容，必须精确匹配文件中的文本
            new_string: 替换后的新内容
            replace_all: 可选，是否替换所有匹配项，默认 False 只替换第一处

        Returns:
            返回编辑结果，替换了几处
        """
        ctx.check_stopped()
        try:
            p = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if not p.is_file():
            return f"错误：文件不存在: {path}"
        try:
            doc = await async_read_document(p)
        except BinaryFileError as e:
            return str(e)
        except (ValueError, OSError) as e:
            return f"读取失败: {e}"
        content = doc.text
        if not old_string:
            return "编辑失败：old_string 不能为空（避免全文件替换）。"
        count = content.count(old_string)
        if count == 0:
            return f"编辑失败：未找到要替换的内容（注意大小写与精确匹配）。文件: {path}"
        if count > 1 and not replace_all:
            return f"编辑失败：找到 {count} 处匹配，请提供更精确的 old_string 或设置 replace_all=true。文件: {path}"
        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        if not await _ask_write(ctx, "edit_file", "编辑", p):
            return "用户拒绝了编辑操作。"
        backup_file(ctx.workspace, p)
        try:
            await async_write_document(p, new_content, encoding=doc.encoding, newline=doc.newline, bom=doc.bom)
        except OSError as e:
            return f"写入失败: {e}"
        await _emit_diff(ctx, p, content, new_content)
        return f"【AI 修改】编辑成功：{path}（替换 {count if replace_all else 1} 处）"

    @tool_spec(category="fs", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def list_dir(
        path: str = "",
        include_hidden: bool = False,
    ) -> str:
        """
        列出目录内容

        Args:
            path: 可选，目录路径，相对于工作区，不传则列出工作区根目录
            include_hidden: 可选，是否包含隐藏文件，默认 False

        Returns:
            返回目录下每个条目的名称、类型和大小
        """
        ctx.check_stopped()
        try:
            p = _resolve(ctx, path or ".")
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if not p.is_dir():
            return f"错误：目录不存在: {path or '.'}"
        entries = []
        try:
            for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if not include_hidden and child.name.startswith("."):
                    continue
                if child.name in _IGNORED_DIRS:
                    continue
                if child.is_dir():
                    entries.append(f"[目录] {child.name}/")
                else:
                    size = child.stat().st_size
                    size_str = f"{size/1024:.1f}KB" if size >= 1024 else f"{size}B"
                    entries.append(f"[文件] {child.name} ({size_str})")
        except PermissionError:
            return f"错误：没有权限访问目录: {path or '.'}"
        total = len(entries)
        shown = entries[:500]
        body = "\n".join(shown) if shown else "(空目录)"
        return f"目录: {_rel(ctx, p)} | 共 {total} 项（显示前 {len(shown)} 项）\n{body}"

    @tool_spec(category="fs", risk_level=RISK_HIGH, timeout=60)
    @tool(parse_docstring=True)
    async def delete_file(path: str, recursive: bool = False) -> str:
        """
        删除工作区内的文件或目录（不可恢复，需谨慎）

        Args:
            path: 要删除的文件或目录路径，相对于工作区
            recursive: 删除目录时是否递归删除其内容，默认 False 只允许空目录

        Returns:
            删除结果
        """
        ctx.check_stopped()
        try:
            p = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if p == Path(ctx.workspace).resolve():
            return "错误：不能删除工作区根目录。"
        if not p.exists():
            return f"错误：路径不存在: {path}"
        # 删除是工具级硬约束：无论 command_confirm_mode 为何，都必须由用户在卡片上确认
        target = _rel(ctx, p) + ("（目录及全部内容）" if p.is_dir() and recursive else "")
        warning = (
            "目录不会自动备份，删除后不可恢复。"
            if p.is_dir()
            else "删除前会备份到 .agent_backups，但不会自动还原。"
        )
        answers = await ctx.questions.ask(
            ctx.emit_event,
            "delete_file",
            f"确认删除 {target}？{warning}",
            ["确认删除", "取消"],
            stop_event=ctx.stop_event,
        )
        if not answers or answers[0] != "确认删除":
            return "用户取消了删除操作（不要绕道用 terminal 重新删除）。"
        if p.is_file():
            backup_file(ctx.workspace, p)
        try:
            if p.is_dir():
                if recursive:
                    shutil.rmtree(p)
                else:
                    p.rmdir()
            else:
                p.unlink()
        except OSError as e:
            return f"删除失败: {e}"
        return f"已删除: {_rel(ctx, p)}"

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=60)
    @tool(parse_docstring=True)
    async def move_file(source: str, destination: str, overwrite: bool = False) -> str:
        """
        移动或重命名工作区内的文件/目录

        Args:
            source: 源路径，相对于工作区
            destination: 目标路径，相对于工作区
            overwrite: 目标已存在时是否覆盖，默认 False

        Returns:
            移动结果
        """
        ctx.check_stopped()
        try:
            src = _resolve(ctx, source)
            dst = _resolve(ctx, destination)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if not src.exists():
            return f"错误：源路径不存在: {source}"
        if dst.exists() and not overwrite:
            return f"错误：目标已存在: {destination}（如需覆盖请传 overwrite=true）"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "move_file",
            "high_risk",
            {"command": f"移动 {_rel(ctx, src)} → {_rel(ctx, dst)}", "cwd": str(dst)},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了移动操作。"
        if src.is_file():
            backup_file(ctx.workspace, src)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.is_file():
                backup_file(ctx.workspace, dst)  # 覆盖前先备份目标，避免不可回滚
                dst.unlink()
            shutil.move(str(src), str(dst))
        except OSError as e:
            return f"移动失败: {e}"
        return f"已移动: {_rel(ctx, src)} → {_rel(ctx, dst)}"

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=60)
    @tool(parse_docstring=True)
    async def copy_file(source: str, destination: str, overwrite: bool = False) -> str:
        """
        复制工作区内的文件

        Args:
            source: 源文件路径，相对于工作区
            destination: 目标文件路径，相对于工作区
            overwrite: 目标已存在时是否覆盖，默认 False

        Returns:
            复制结果
        """
        ctx.check_stopped()
        try:
            src = _resolve(ctx, source)
            dst = _resolve(ctx, destination)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if not src.is_file():
            return f"错误：源文件不存在: {source}"
        if dst.exists() and not overwrite:
            return f"错误：目标已存在: {destination}（如需覆盖请传 overwrite=true）"
        if not await _ask_write(ctx, "copy_file", "复制", src):
            return "用户拒绝了复制操作。"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.is_file():
                backup_file(ctx.workspace, dst)  # 覆盖前先备份目标，避免不可回滚
            shutil.copy2(src, dst)
        except OSError as e:
            return f"复制失败: {e}"
        return f"已复制: {_rel(ctx, src)} → {_rel(ctx, dst)}"

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=30)
    @tool(parse_docstring=True)
    async def restore_file(action: str = "list", backup_id: str = "") -> str:
        """
        查看或恢复文件修改前的自动备份（工作区 .agent_backups）

        Args:
            action: "list" 列出最近备份；"restore" 恢复指定/最近一次备份
            backup_id: 可选，备份 id（list 结果中的 backup_id）；为空时恢复最近一次

        Returns:
            备份列表或恢复结果
        """
        ctx.check_stopped()
        if action == "list":
            items = list_backups(ctx.workspace, limit=20)
            if not items:
                return "没有可用的自动备份。"
            lines = ["最近备份:"]
            for it in items:
                lines.append(f"- {it.get('backup_id')}  {it.get('original')}  {it.get('created_at')}")
            return "\n".join(lines)
        if action != "restore":
            return f"未知 action: {action}（支持 list / restore）"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "restore_file",
            "normal",
            {"command": f"恢复备份 {backup_id or '(最近一次)'}", "cwd": ctx.workspace},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了恢复操作。"
        return restore_backup(ctx.workspace, backup_id)

    return (
        read_file, write_file, edit_file, list_dir,
        delete_file, move_file, copy_file, restore_file, read_files,
    )
