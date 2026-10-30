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


def _diff_counts(before: str, after: str) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


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
        读取工作区内的文本文件，并按行号分段返回内容

        适用：编辑前确认目标代码的当前内容、修改后核对结果、定位报错行附近的上下文。
        读取大文件时先用 offset / limit 只取相关区间，不要全文读取；需要同时查看多个文件时
        改用 read_files，减少往返次数。

        路径限定在当前工作区内；文件超过 8MB 时整份读取会报错，必须用 offset / limit 分段读取；
        二进制文件会直接拒绝。返回值包含总行数、显示区间和文件编码。行号前缀仅用于定位，
        edit_file 的 old_string 不要复制行号，否则会匹配失败。

        Args:
            path: 文件路径，相对当前工作区（也接受位于工作区内的绝对路径）；不存在时返回错误。
            offset: 起始行号，从 1 开始计数，默认 1；超出文件末尾时返回提示。
            limit: 最多读取的行数，默认 0 表示读到文件末尾；查大文件时建议设为 100-400。

        Returns:
            文件元信息（总行数、显示区间、编码）与带行号的内容；失败时返回错误原因。
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
            return (
                f"文件: {path} | 总行数: {rng.total_lines} | 显示 {start}-{end} | 编码: {rng.encoding}"
                f" | 行号仅用于定位，edit_file 的 old_string 不要带行号前缀\n{numbered}"
            )
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
        return (
            f"文件: {path} | 总行数: {total} | 显示 {start}-{end} | 编码: {doc.encoding}"
            f" | 行号仅用于定位，edit_file 的 old_string 不要带行号前缀\n{numbered}"
        )

    @tool_spec(category="fs", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def read_files(
        paths: str,
        max_lines_each: int = 0,
    ) -> str:
        """
        一次读取多个文本文件（每个文件带行号，按文件分段返回）

        适用：同时查看一组相关文件（组件 + 样式 + 类型定义、同目录多个模块），
        比连续调用多次 read_file 更省往返。只做快速浏览时可用 max_lines_each 限制每个文件的行数。

        单次最多读取 8 个文件，超出部分会被截断并提示；某个文件不存在或为二进制时只在该文件
        分段内报错，不影响其他文件。行号仅用于定位，不要带进 edit_file 的 old_string。

        Args:
            paths: 文件路径列表，用换行、逗号或分号分隔，相对当前工作区，最多 8 个。
            max_lines_each: 每个文件最多读取的行数，默认 0 表示读取全部；查看大文件时建议 100-300。

        Returns:
            按 "=== 路径（共 N 行，编码 X）===" 分段的带行号内容；失败的条目附错误原因。
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
        新建文本文件，或在确有必要时整体覆盖已有文件

        优先用于新建文件。目标文件已存在时应先用 read_file 查看当前内容，再用 edit_file 只修改差异；
        整份覆盖会抹掉其他改动，也会让后续审查看不到真实增量。只有用户明确要求整体重写、
        或原内容已完全失效时才允许覆盖。

        路径限定在工作区内，自动创建父目录。写入前会自动备份旧文件到 .agent_backups；
        覆盖时保留原文件的编码、换行符与 BOM，新建文件默认 UTF-8。返回结果会区分
        "已创建 / 内容完全一致 / 已覆盖"，并给出相对写入前的增删行数；若提示内容完全一致，
        说明本次写入没有任何变化，不要重复调用。

        Args:
            path: 文件路径，相对当前工作区；不存在时按新建处理，自动创建父目录。
            content: 要写入的完整文件内容；覆盖已有文件时会整体替换。

        Returns:
            写入状态、字符数与 +新增/-删除 行数；失败时返回错误原因。
        """
        ctx.check_stopped()
        try:
            p = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        existed = p.exists()
        before = ""
        doc = None
        if existed:
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
        added, removed = _diff_counts(before, content) if doc is not None else (0, 0)
        await _emit_diff(ctx, p, before, content)
        if not existed:
            return (
                f"【AI 写入】已创建文件 {path}（{len(content)} 字符）。"
                "后续修改请先用 read_file 读取相关片段确认当前内容，"
                "再用 edit_file 只改差异，不要整份覆盖。"
            )
        if added == 0 and removed == 0:
            return (
                f"【AI 写入】已写入文件 {path}，但内容与写入前完全一致"
                f"（{len(content)} 字符，+0/-0 行）。本次写入没有产生任何变化；"
                "不要重复写入相同内容，请直接基于当前文件继续或给出最终回答。"
            )
        return (
            f"【AI 写入】已覆盖文件 {path}（原 {len(before)} → 新 {len(content)} 字符，"
            f"+{added}/-{removed} 行，编码 {doc.encoding}）。该文件已存在；"
            "后续修改请先用 read_file 读取相关片段确认当前内容，"
            "再用 edit_file 只改差异，不要再次整份覆盖。"
        )

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=60)
    @tool(parse_docstring=True)
    async def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        精确替换现有文件中的一小段内容（修改现有文件的首选工具）

        只需提供要替换的旧片段和新片段，不必输出整个文件。适合修 bug、改样式、调参数、
        替换文案、补一个函数或区块等局部修改；多处或跨文件修改可改用 apply_patch。

        old_string 必须与文件当前内容精确匹配（包括空格与缩进），建议先 read_file 确认；
        从 read_file 复制内容时行号前缀会被自动忽略。匹配失败会返回提示，此时重新读取相关片段
        再试，不要退回整份 write_file。匹配到多处且未设置 replace_all 时会拒绝执行。
        写入前自动备份原文件到 .agent_backups，并推送 diff 供审查。

        Args:
            path: 文件路径，相对当前工作区；文件必须已存在。
            old_string: 要被替换的旧内容，需精确匹配文件中的一段文本；尽量包含上下文以保证唯一。
            new_string: 替换后的新内容；留空表示删除该片段。
            replace_all: 是否替换所有匹配项，默认 False 只替换第一处；替换多处相同文本时设为 True。

        Returns:
            编辑结果与替换处数；未找到或匹配不唯一时返回失败原因。
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
        stripped_numbers = False
        if count == 0:
            # read_file 的输出带 "  12 | " 行号；模型偶尔连行号一起复制。
            # 精确匹配失败时去掉行号前缀再试一次，避免一次复制失误就退回整份重写。
            cleaned = "\n".join(
                re.sub(r"^\s*\d+\s*\|\s?", "", line) for line in old_string.splitlines()
            )
            if cleaned != old_string and cleaned:
                count = content.count(cleaned)
                if count:
                    old_string = cleaned
                    stripped_numbers = True
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
        note = "（old_string 中的行号前缀已自动忽略）" if stripped_numbers else ""
        return f"【AI 修改】编辑成功：{path}（替换 {count if replace_all else 1} 处）{note}"

    @tool_spec(category="fs", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def list_dir(
        path: str = "",
        include_hidden: bool = False,
    ) -> str:
        """
        列出目录下一层的文件与子目录（非递归）

        适用：确认项目结构、查找配置文件或入口文件所在目录。需要按模式递归找文件时用 list_files；
        需要查看某文件内容时用 read_file。自动忽略 .git、node_modules、.venv、dist、build
        等依赖/构建目录；隐藏文件默认不显示。

        只列出当前层，不会递归展开；条目按目录优先、名称排序，最多显示 500 项。

        条目超过 500 项时只返回前 500 项并注明总数；目录很大时改用 list_files 按模式筛选，
        或指定更深的 path 缩小范围。目录不存在或没有读取权限时返回明确错误。

        Args:
            path: 目录路径，相对当前工作区；默认空字符串表示工作区根目录。
            include_hidden: 是否包含以点开头的隐藏文件/目录，默认 False。

        Returns:
            目录路径、条目总数，以及 "[目录] 名称/"、"[文件] 名称 (大小)" 形式的清单。
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
        删除工作区内的文件或目录（高风险操作，必须用户确认）

        仅在用户明确要求删除、或确认目标文件确实不再需要时使用。工具会弹出确认卡，
        无论审批模式如何都必须由用户点选"确认删除"才会执行；用户取消后不要改用 terminal
        或脚本绕过删除，应直接向用户说明。

        文件删除前会备份到 .agent_backups，可通过 restore_file 恢复；目录删除不会备份，
        删除后不可恢复。禁止删除工作区根目录；递归删除目录时必须显式传 recursive=true。

        Args:
            path: 要删除的文件或目录路径，相对当前工作区。
            recursive: 删除目录时是否连同全部子项递归删除，默认 False 只允许空目录。

        Returns:
            删除结果；用户取消、路径不存在或删除失败时返回对应说明。
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
        移动或重命名工作区内的文件或目录

        适用：重构目录结构、重命名文件、把文件移到目标目录。目标目录不存在时自动创建；
        目标已存在时必须显式设置 overwrite=true，否则拒绝执行。

        源文件会在移动前备份到 .agent_backups；覆盖已有目标时目标也会先备份。
        操作需要用户审批（auto 模式直接通过）。移动目录时不会单独备份目录内容。
        移动后请同步更新代码中的引用路径（可用 find_references 检查旧路径是否还有引用）；
        不确定目标位置时先用 list_dir 确认目录结构。

        Args:
            source: 源文件或目录路径，相对当前工作区，必须存在。
            destination: 目标路径，相对当前工作区；可以是新文件名或目标目录下的完整路径。
            overwrite: 目标已存在时是否覆盖，默认 False。

        Returns:
            移动结果（源路径 → 目标路径）；失败时返回原因。
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
        复制工作区内的单个文件（不复制目录）

        适用：创建配置/样式的副本、备份单个文件、复制模板后修改。目标目录不存在时自动创建；
        目标已存在时必须显式设置 overwrite=true，否则拒绝执行。复制保留文件元数据；
        覆盖已有目标前会先备份目标文件。

        需要复制整个目录时，用 terminal 或逐个文件复制；需要移动文件时用 move_file。
        复制操作会触发审批（auto 模式直接通过）；复制后如需修改内容，先用 read_file 读取
        再配合 edit_file 精确修改，避免整文件重写。

        Args:
            source: 源文件路径，相对当前工作区，必须是已存在的文件。
            destination: 目标文件路径，相对当前工作区；建议写完整文件名而非仅目录。
            overwrite: 目标已存在时是否覆盖，默认 False。

        Returns:
            复制结果（源路径 → 目标路径）；失败时返回原因。
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
        查看或恢复自动备份的文件版本（工作区 .agent_backups）

        写入、编辑、删除、批量替换等操作前都会自动备份原文件。本工具用于误改后回滚：
        先用 action="list" 查看最近 20 条备份，再用 action="restore" 恢复指定 backup_id
        或最近一次备份。恢复会覆盖当前文件，执行前需要用户审批。

        这是单文件级备份，不是整仓快照；需要整体回退多处修改时用 workspace_checkpoint。

        Args:
            action: "list" 列出最近备份；"restore" 恢复指定或最近一次备份。
            backup_id: 可选，list 结果中的备份 id；restore 时留空表示恢复最近一次。

        Returns:
            list 返回备份清单（id、原文件、时间）；restore 返回恢复结果。
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
