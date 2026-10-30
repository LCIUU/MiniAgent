"""项目记忆：跨会话保存约定、命令、踩坑记录（工作区 .agent/memory.md）。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from ..services.file_io import async_read_document, async_write_document, async_write_new_text, read_document
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec

MEMORY_REL = ".agent/memory.md"
MAX_MEMORY_CHARS = 20000


def memory_path(ctx: ToolContext) -> Path:
    return resolve_workspace_path(ctx.workspace, MEMORY_REL)


def load_memory(ctx: ToolContext, limit: int = 6000) -> str:
    """读取项目记忆（供系统提示注入）。"""
    try:
        p = memory_path(ctx)
        if not p.is_file():
            return ""
        doc = read_document(p)
        text = doc.text.strip()
        return text[:limit]
    except Exception:
        return ""


def make_memory_tools(ctx: ToolContext):

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=20)
    @tool(parse_docstring=True)
    async def project_memory(action: str = "read", content: str = "") -> str:
        """
        读取或写入项目长期记忆（跨会话保存，位于工作区 .agent/memory.md）

        适用：记录项目约定、常用命令、架构决策、踩坑结论等下次会话仍需知道的信息。
        read 返回当前记忆；append 追加一段（推荐）；replace 覆盖全部内容。
        写入需要用户审批，追加时自动用空行分隔，内容上限 20000 字符（超出会截断）。

        不要把临时任务进度、代码细节或可从文件直接读到的内容写进记忆；
        需要整体重写时才用 replace，日常增量记录用 append。

        Args:
            action: "read" 读取；"append" 追加一段；"replace" 覆盖全部内容。
            content: append/replace 时要写入的文本；read 时忽略。

        Returns:
            读取结果或写入确认；失败时返回原因。
        """
        ctx.check_stopped()
        try:
            p = memory_path(ctx)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        if action == "read":
            if not p.is_file():
                return "项目记忆为空。可用 action='append' 记录项目约定、常用命令、踩坑结论。"
            try:
                doc = await async_read_document(p)
                return doc.text[:MAX_MEMORY_CHARS]
            except Exception as e:
                return f"读取项目记忆失败: {e}"
        if action not in ("append", "replace"):
            return f"未知 action: {action}（支持 read / append / replace）"
        if not content.strip():
            return "错误：content 不能为空。"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "project_memory",
            "normal",
            {"command": f"{action} 项目记忆", "cwd": str(p)},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了写入项目记忆。"
        try:
            if action == "append" and p.is_file():
                old_doc = await async_read_document(p)
                old = old_doc.text
                new = (old.rstrip() + "\n\n" + content.strip() + "\n")[:MAX_MEMORY_CHARS]
                await async_write_document(p, new, encoding="utf-8", newline="\n")
            else:
                await async_write_new_text(p, (content.rstrip() + "\n")[:MAX_MEMORY_CHARS])
        except OSError as e:
            return f"写入项目记忆失败: {e}"
        return f"已{'追加' if action == 'append' else '更新'}项目记忆: {MEMORY_REL}（{len(content)} 字符）"

    return project_memory
