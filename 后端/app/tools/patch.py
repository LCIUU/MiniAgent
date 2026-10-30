"""apply_patch 工具：标准 unified diff 的多文件、多 hunk 原子应用。

支持：
- ``--- a/path`` / ``+++ b/path`` 头（自动去掉 a/、b/ 前缀）
- ``/dev/null`` 表示新建或删除文件
- ``@@ -old_start,old_count +new_start,new_count @@`` hunk
- 全量校验后再落盘：任一处上下文不匹配则整体失败，不产生半成品

这是 coding agent 的核心编辑工具：比 write_file 精准，比 edit_file 适合多块修改。
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool

from ..services.backups import backup_file
from ..services.file_io import (
    BinaryFileError,
    async_read_document,
    async_write_document,
    async_write_new_text,
)
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec

_HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_delete: bool = False


def _strip_prefix(path: str) -> str | None:
    path = path.strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse_patch(patch: str) -> list[FilePatch]:
    """解析 unified diff；格式错误抛 ValueError。"""
    if not patch or not patch.strip():
        raise ValueError("patch 为空")
    lines = patch.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    files: list[FilePatch] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise ValueError(f"第 {i + 1} 行：--- 头后缺少 +++ 头")
            old = _strip_prefix(line[4:])
            new = _strip_prefix(lines[i + 1][4:])
            fp = FilePatch(old_path=old, new_path=new, is_new=old is None, is_delete=new is None)
            i += 2
            while i < len(lines):
                m = _HUNK_RE.match(lines[i])
                if not m:
                    break
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                hunk = Hunk(old_start, old_count, new_start, new_count)
                i += 1
                while i < len(lines):
                    body = lines[i]
                    if body.startswith(("--- ", "@@ ")) or body.startswith("diff --git "):
                        break
                    if body == "":
                        # 空行表示 hunk 结束（标准 diff 的空上下文行是 " "，不是 ""）
                        break
                    if body[0] not in " +-\\":
                        raise ValueError(f"第 {i + 1} 行：非法 hunk 行 {body[:80]!r}")
                    if body.startswith("\\ No newline at end of file"):
                        i += 1
                        continue
                    hunk.lines.append(body)
                    i += 1
                fp.hunks.append(hunk)
            if not fp.hunks and not fp.is_new and not fp.is_delete:
                raise ValueError(f"文件 {new or old} 没有 hunk")
            files.append(fp)
            continue
        i += 1
    if not files:
        raise ValueError("未找到 --- / +++ 文件头，请使用标准 unified diff")
    return files


def _apply_hunks(original: str, patch: FilePatch) -> str:
    """按 hunk 应用补丁；上下文不匹配时抛 ValueError（带定位）。"""
    trailing_newline = original.endswith("\n") or original == ""
    lines = original.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    out: list[str] = []
    cursor = 0
    for hunk in patch.hunks:
        start = max(0, hunk.old_start - 1)
        if start < cursor:
            raise ValueError(f"hunk 位置重叠或顺序错误：@@ -{hunk.old_start}")
        if start > len(lines):
            raise ValueError(f"hunk 起始行超出文件范围：@@ -{hunk.old_start}")
        out.extend(lines[cursor:start])
        idx = start
        for raw in hunk.lines:
            tag, content = raw[0], raw[1:]
            if tag == " ":
                if idx >= len(lines) or lines[idx] != content:
                    got = lines[idx] if idx < len(lines) else "<EOF>"
                    raise ValueError(
                        f"上下文不匹配（原文件第 {idx + 1} 行）：期望 {content!r}，实际 {got!r}"
                    )
                out.append(lines[idx])
                idx += 1
            elif tag == "-":
                if idx >= len(lines) or lines[idx] != content:
                    got = lines[idx] if idx < len(lines) else "<EOF>"
                    raise ValueError(
                        f"待删除行不匹配（原文件第 {idx + 1} 行）：期望 {content!r}，实际 {got!r}"
                    )
                idx += 1
            elif tag == "+":
                out.append(content)
        cursor = idx
    out.extend(lines[cursor:])
    return "\n".join(out) + ("\n" if trailing_newline else "")


def _rel(workspace: str, p: Path) -> str:
    try:
        return os.path.relpath(str(p), workspace).replace("\\", "/")
    except ValueError:
        return str(p)


def make_patch_tool(ctx: ToolContext):

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=90)
    @tool(parse_docstring=True)
    async def apply_patch(patch: str, dry_run: bool = False) -> str:
        """
        用标准 unified diff 精准修改一个或多个文件（多处/跨文件修改首选）

        适用：同一文件多处修改、一次修改多个文件、需要精确控制增删行。只描述改动片段，
        不必重写整个文件，比整份 write_file 更安全、diff 也更易审查。

        patch 必须是标准 unified diff：---/+++ 文件头加 @@ -旧起,旧数 +新起,新数 @@ hunk，
        上下文行以空格开头，删除行以 - 开头，新增行以 + 开头；新建文件用 --- /dev/null，
        删除文件用 +++ /dev/null。所有文件先全部校验再统一落盘，任何一处上下文不匹配都会
        整体失败、不产生半成品。删除文件会弹出独立确认卡。建议先用 dry_run=true 预览 diff。

        Args:
            patch: 完整 unified diff 文本；文件路径相对工作区，可带 a/、b/ 前缀。
            dry_run: 默认 False；True 时只校验并返回每个文件的 diff 预览，不写入任何文件。

        Returns:
            每个文件的修改结果与 +新增/-删除 行数；校验失败时返回具体位置与原因。
        """
        ctx.check_stopped()
        try:
            file_patches = parse_patch(patch)
        except ValueError as e:
            return f"patch 解析失败: {e}"

        # 先解析路径 + 计算新内容，全部成功后再统一落盘（原子性）
        prepared: list[tuple[FilePatch, Path, str, str, object]] = []
        for fp in file_patches:
            raw_path = fp.new_path or fp.old_path
            if not raw_path:
                return "patch 错误：文件路径为空"
            try:
                p = resolve_workspace_path(ctx.workspace, raw_path)
            except (PathOutsideWorkspace, ValueError) as e:
                return f"patch 错误：{e}"
            if fp.is_delete:
                if not p.exists():
                    return f"patch 错误：要删除的文件不存在: {raw_path}"
                prepared.append((fp, p, "", "", None))
                continue
            doc = None
            before = ""
            if fp.is_new:
                if p.exists():
                    return f"patch 错误：新建文件已存在: {raw_path}"
            else:
                if not p.is_file():
                    return f"patch 错误：文件不存在: {raw_path}"
                try:
                    doc = await async_read_document(p)
                except (BinaryFileError, ValueError, OSError) as e:
                    return f"patch 错误：{raw_path} 读取失败: {e}"
                before = doc.text
            try:
                after = _apply_hunks("" if fp.is_new else before, fp)
            except ValueError as e:
                return f"patch 错误：{raw_path}: {e}"
            prepared.append((fp, p, before, after, doc))

        if dry_run:
            previews: list[str] = []
            for fp, p, before, after, _doc in prepared:
                rel = _rel(ctx.workspace, p)
                if fp.is_delete:
                    previews.append(f"=== {rel}（将删除，共 {len(before.splitlines())} 行）===")
                    continue
                diff = "\n".join(
                    difflib.unified_diff(
                        before.splitlines(), after.splitlines(),
                        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
                    )
                )
                previews.append(f"=== {rel} ===\n{diff or '（无变化）'}")
            body = "\n\n".join(previews)
            return (
                f"【预览】apply_patch 将修改 {len(prepared)} 个文件（未写入）：\n{body[:16000]}\n\n"
                "确认无误后，用相同 patch 并把 dry_run 设为 false 才会写入。"
            )

        # 需要审批的写入/删除操作统一走审批（auto 模式直接通过，all 模式会询问）
        if any(not fp.is_delete for fp, *_ in prepared):
            approved = await ctx.approvals.request(
                ctx.emit_event,
                "apply_patch",
                "normal",
                {
                    "command": f"应用 patch，修改 {len([1 for fp, *_ in prepared if not fp.is_delete])} 个文件",
                    "cwd": ctx.workspace,
                },
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了 patch 应用。"
        if any(fp.is_delete for fp, *_ in prepared):
            del_targets = [_rel(ctx.workspace, p) for fp, p, *_ in prepared if fp.is_delete]
            listing = "\n".join(f"- {t}" for t in del_targets[:20])
            # 删除是工具级硬约束：不受 command_confirm_mode 影响，必须用户点选确认
            answers = await ctx.questions.ask(
                ctx.emit_event,
                "apply_patch",
                f"apply_patch 将删除 {len(del_targets)} 个文件：\n{listing}\n\n确认删除？",
                ["确认删除", "取消"],
                stop_event=ctx.stop_event,
            )
            if not answers or answers[0] != "确认删除":
                return "用户取消了包含删除操作的 patch。"

        results: list[str] = []
        for fp, p, before, after, doc in prepared:
            if fp.is_delete:
                backup_file(ctx.workspace, p)
                try:
                    p.unlink()
                except OSError as e:
                    return f"删除失败 {_rel(ctx.workspace, p)}: {e}"
                await ctx.emit_event({
                    "type": "diff", "call_id": ctx.call_id,
                    "path": _rel(ctx.workspace, p), "diff": f"删除文件 {_rel(ctx.workspace, p)}",
                    "added": 0, "removed": len(before.splitlines()),
                })
                results.append(f"删除 {_rel(ctx.workspace, p)}")
                continue
            try:
                if fp.is_new or doc is None:
                    await async_write_new_text(p, after)
                else:
                    backup_file(ctx.workspace, p)
                    await async_write_document(p, after, encoding=doc.encoding, newline=doc.newline, bom=doc.bom)
            except OSError as e:
                return f"写入失败 {_rel(ctx.workspace, p)}: {e}"
            added = sum(1 for h in fp.hunks for line in h.lines if line.startswith("+"))
            removed = sum(1 for h in fp.hunks for line in h.lines if line.startswith("-"))

            diff = "\n".join(
                difflib.unified_diff(
                    before.splitlines(), after.splitlines(),
                    fromfile=f"a/{_rel(ctx.workspace, p)}", tofile=f"b/{_rel(ctx.workspace, p)}",
                    lineterm="",
                )
            )
            await ctx.emit_event({
                "type": "diff", "call_id": ctx.call_id,
                "path": _rel(ctx.workspace, p), "diff": diff[:12000],
                "added": added, "removed": removed,
            })
            results.append(f"{_rel(ctx.workspace, p)} (+{added}/-{removed})")

        return "apply_patch 完成:\n" + "\n".join(f"- {r}" for r in results)

    return apply_patch
