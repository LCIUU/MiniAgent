"""工具聚合：按运行上下文创建全部工具。"""

from __future__ import annotations

from .build_test import make_build_test_tools
from .bulk_edit import make_bulk_edit_tool
from .checkpoint import make_checkpoint_tool
from .code_intel import make_code_intel_tools
from .context import ApprovalManager, QuestionManager, ToolContext
from .db_query import make_db_tools
from .dependencies import make_dependency_tools
from .docker import make_docker_tools
from .encoding import make_encoding_tool
from .env_check import make_env_tools
from .files import make_file_tools
from .formatting import make_format_tool
from .git_tools import make_git_tools
from .http_request import make_http_tools
from .interactive import make_interactive_tools
from .json_edit import make_json_edit_tool
from .memory import make_memory_tools
from .patch import make_patch_tool
from .registry import ensure_spec
from .runtime_tools import make_runtime_tools
from .search import make_search_tools
from .security import make_security_tool
from .terminal import make_process_tool, make_terminal_tool
from .web import make_web_search_tool


def build_tools(
    workspace: str,
    emit,
    approvals: ApprovalManager,
    stop_event,
    mcp_tools: list | None = None,
    questions: QuestionManager | None = None,
) -> list:
    """创建绑定当前运行上下文的完整工具列表。

    构建过程中各工厂通过 @tool_spec 完成元数据注册；
    MCP 等外部工具未显式注册，统一补保守默认 spec（非幂等、不缓存）。
    """
    ctx = ToolContext(
        workspace=workspace,
        emit=emit,
        approvals=approvals,
        stop_event=stop_event,
        questions=questions or QuestionManager(),
    )
    tools = [
        make_terminal_tool(ctx),
        make_process_tool(ctx),
        *make_file_tools(ctx),
        make_patch_tool(ctx),
        make_json_edit_tool(ctx),
        make_bulk_edit_tool(ctx),
        *make_search_tools(ctx),
        *make_code_intel_tools(ctx),
        *make_git_tools(ctx),
        make_memory_tools(ctx),
        make_format_tool(ctx),
        *make_web_search_tool(ctx),
        make_build_test_tools(ctx),
        *make_runtime_tools(ctx),
        *make_interactive_tools(ctx),
        *make_dependency_tools(ctx),
        *make_db_tools(ctx),
        make_http_tools(ctx),
        make_env_tools(ctx),
        *make_docker_tools(ctx),
        make_checkpoint_tool(ctx),
        make_encoding_tool(ctx),
        make_security_tool(ctx),
    ]
    if mcp_tools:
        # 命名空间隔离：MCP 工具若与内置工具同名，重命名为 mcp__<原名>，避免静默覆盖
        used = {t.name for t in tools}
        for t in mcp_tools:
            name = t.name
            if name in used:
                base = f"mcp__{name}"
                name = base
                counter = 2
                while name in used:
                    name = f"{base}_{counter}"
                    counter += 1
                try:
                    t = t.model_copy(update={"name": name})
                except Exception:
                    continue
            used.add(name)
            tools.append(t)
    for t in tools:
        ensure_spec(t.name)
    return tools
