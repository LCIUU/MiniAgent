"""结构化 JSON 编辑：按 JSON Pointer 精确读写，避免整文件重写破坏格式。

- ``get`` 只读，返回目标节点
- ``set`` / ``delete`` 保留原文件的缩进风格（空格/制表符）、Unicode 字符、末尾换行
- 写入前备份并发送 diff；JSONC（含注释）文件无法安全重排，提示改用 edit_file
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

from langchain_core.tools import tool

from ..services.backups import backup_file
from ..services.file_io import BinaryFileError, async_read_document, async_write_document
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec

_MAX_VALUE_CHARS = 8000
_INDENT_RE = re.compile(r"^([ \t]+)\S", re.MULTILINE)


def _rel(workspace: str, p: Path) -> str:
    try:
        return os.path.relpath(str(p), workspace).replace("\\", "/")
    except ValueError:
        return str(p)


def _detect_indent(text: str) -> str:
    m = _INDENT_RE.search(text)
    if not m:
        return "  "
    indent = m.group(1)
    return "\t" if indent.startswith("\t") else " " * len(indent)


def _looks_like_jsonc(text: str) -> bool:
    """跳过字符串内容后检查是否存在 // 或 /* 注释。"""
    in_str = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] in "/*":
            return True
        i += 1
    return False


def _parse_pointer(pointer: str) -> list[str]:
    if pointer in ("", "/"):
        return [] if pointer == "" else [""]
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer 必须以 / 开头，例如 /scripts/build/command；根节点用空字符串")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _child_key(container: object, part: str) -> object:
    if isinstance(container, list):
        if not part.isdigit():
            raise ValueError(f"数组下标必须是数字，收到 {part!r}")
        idx = int(part)
        if idx >= len(container):
            raise KeyError(f"数组下标越界：{idx} ≥ 长度 {len(container)}")
        return idx
    if isinstance(container, dict):
        if part not in container:
            raise KeyError(f"键不存在：{part}")
        return part
    raise KeyError(f"无法在 {type(container).__name__} 上继续下钻")


def _resolve(data: object, parts: list[str]) -> object:
    node = data
    for part in parts:
        if not isinstance(node, (dict, list)):
            raise KeyError(f"无法在 {type(node).__name__} 上继续下钻")
        node = node[_child_key(node, part)]
    return node


def _format_value(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > _MAX_VALUE_CHARS:
        return text[:_MAX_VALUE_CHARS] + f"\n...[已截断，共 {len(text)} 字符]"
    return text


def _decode_new_value(raw: str) -> tuple[object, bool]:
    """优先按 JSON 解析；失败则当作纯字符串（返回 is_raw=True）。"""
    if raw == "":
        return "", True
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        return raw, True


def make_json_edit_tool(ctx: ToolContext):

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=30)
    @tool(parse_docstring=True)
    async def json_edit(
        path: str,
        pointer: str = "",
        value: str = "",
        action: str = "get",
    ) -> str:
        """
        按 JSON Pointer 读取或修改 JSON 文件（保留缩进与中文）

        比整文件重写更安全：只改动目标节点，其余键顺序和格式保持不变。

        Args:
            path: JSON 文件路径，相对工作区
            pointer: JSON Pointer，根节点留空；例如 /scripts/build/command 或 /dependencies/0
            value: 可选，set 时的新值；优先按 JSON 解析（如 42 / true / [1,2] / {"a":1}），解析失败则按字符串写入
            action: 操作类型：get（默认，只读）/ set / delete

        Returns:
            节点当前值，或修改结果与增删摘要
        """
        ctx.check_stopped()
        action = action.lower().strip()
        if action not in ("get", "set", "delete"):
            return f"错误：不支持的 action {action}（可选 get / set / delete）"
        try:
            p = resolve_workspace_path(ctx.workspace, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return f"错误：{e}"
        if not p.is_file():
            return f"错误：文件不存在: {path}"
        try:
            doc = await async_read_document(p)
        except BinaryFileError as e:
            return f"错误：{e}"
        except (ValueError, OSError) as e:
            return f"读取失败: {e}"

        text = doc.text
        if _looks_like_jsonc(text):
            return (
                f"{path} 含 // 或 /* 注释（JSONC）。标准 JSON 解析会失败，"
                "json_edit 无法保证保留注释——请改用 edit_file 精确替换。"
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return f"JSON 解析失败（第 {e.lineno} 行第 {e.colno} 列）：{e.msg}；如需改文本请用 edit_file。"

        try:
            parts = _parse_pointer(pointer)
        except ValueError as e:
            return f"错误：{e}"

        if action == "get":
            try:
                node = _resolve(data, parts)
            except (KeyError, ValueError) as e:
                return f"读取失败：{e}"
            return f"{pointer or '<根节点>'} =\n{_format_value(node)}"

        # set / delete
        if not parts:
            if action == "delete":
                return "错误：不能删除根节点。如需清空文件请用 write_file。"
            new_data, is_raw = _decode_new_value(value)
        else:
            parent_parts, last = parts[:-1], parts[-1]
            try:
                parent = _resolve(data, parent_parts)
            except (KeyError, ValueError) as e:
                return f"错误：父节点不存在（{e}）；json_edit 不会自动创建中间节点，请先用 set 逐层创建。"
            if isinstance(parent, dict):
                if action == "delete":
                    if last not in parent:
                        return f"错误：键不存在：{last}"
                    del parent[last]
                else:
                    new_data, is_raw = _decode_new_value(value)
                    parent[last] = new_data
                new_data = data
            elif isinstance(parent, list):
                if not last.isdigit():
                    return f"错误：数组下标必须是数字，收到 {last!r}"
                idx = int(last)
                if action == "delete":
                    if idx >= len(parent):
                        return f"错误：数组下标越界：{idx} ≥ 长度 {len(parent)}"
                    del parent[idx]
                else:
                    new_data, is_raw = _decode_new_value(value)
                    if idx > len(parent):
                        return f"错误：数组下标越界：{idx} > 长度 {len(parent)}（追加请用 {len(parent)}）"
                    if idx == len(parent):
                        parent.append(new_data)
                    else:
                        parent[idx] = new_data
                new_data = data
            else:
                return f"错误：父节点是 {type(parent).__name__}，无法设置键；请检查 pointer。"

        indent = _detect_indent(text)
        after = json.dumps(new_data, ensure_ascii=False, indent=indent)
        if text.endswith("\n"):
            after += "\n"

        approved = await ctx.approvals.request(
            ctx.emit_event,
            "json_edit",
            "normal",
            {
                "command": f"json_edit {action} {path}#{pointer or '/'}",
                "cwd": ctx.workspace,
            },
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了 JSON 修改。"

        backup_file(ctx.workspace, p)
        try:
            await async_write_document(
                p, after, encoding=doc.encoding, newline=doc.newline, bom=doc.bom
            )
        except OSError as e:
            return f"写入失败: {e}"

        diff = "\n".join(
            difflib.unified_diff(
                text.splitlines(), after.splitlines(),
                fromfile=f"a/{_rel(ctx.workspace, p)}", tofile=f"b/{_rel(ctx.workspace, p)}",
                lineterm="",
            )
        )
        await ctx.emit_event({
            "type": "diff", "call_id": ctx.call_id,
            "path": _rel(ctx.workspace, p), "diff": diff[:12000],
            "added": sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")),
            "removed": sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")),
        })

        target = pointer or "<根节点>"
        if action == "delete":
            return f"已删除 {path} 的 {target}（已备份原文件到 .agent_backups）。"
        note = "（值未能按 JSON 解析，已按纯字符串写入）" if is_raw else ""
        return f"已更新 {path} 的 {target}{note}，当前值：\n{_format_value(new_data if not parts else _resolve(new_data, parts))}"

    return json_edit
