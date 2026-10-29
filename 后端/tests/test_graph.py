"""Agent 图端到端测试（mock LLM，不依赖真实 API）。

验证：execute（ReAct 循环：模型自主 write_todos → 工具执行）→ context 全流程 + 事件流 + 真实工具执行。
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_llm import MockLLM  # noqa: E402


async def main():
    import app.agent.graph as g
    from app.agent.state import AgentState
    from app.tools.build_test import make_build_test_tools
    from app.tools.context import ApprovalManager
    from app.tools.files import make_file_tools
    from app.tools.search import make_search_tools
    from app.tools.terminal import make_terminal_tool

    ws = tempfile.mkdtemp()
    events = []

    async def emit(ev):
        events.append(ev)

    # 构造真实工具（绑定 fake ctx 会因绑定问题出错，这里直接用真实 ToolContext）
    from app.tools.context import ToolContext

    ctx = ToolContext(workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto"))
    tools = [
        make_terminal_tool(ctx),
        *make_file_tools(ctx),
        *make_search_tools(ctx),
        make_build_test_tools(ctx),
    ]

    original = g.init_chat_models
    g.init_chat_models = lambda **kw: MockLLM()
    try:
        initial: AgentState = {
            "request": "创建一个 demo.txt",
            "workspace": ws,
            "messages": [],
            "plan": [],
            "tool_rounds": 0,
            "summary": "",
            "summary_count": 0,
            "run_id": "test_run",
            "session_id": "test_session",
            "stopped": False,
            "error": None,
            "emit": emit,
            "approvals": ApprovalManager(mode="auto"),
            "stop_event": asyncio.Event(),
            "_tools": tools,
        }
        result = await g.graph.ainvoke(initial)
    finally:
        g.init_chat_models = original

    # 验证事件流
    types = [e["type"] for e in events]
    print("event types:", types)
    assert "plan" in types, "应发出 plan 事件"
    assert "token" in types, "应发出 token 事件"
    assert "tool_start" in types, "应发出 tool_start"
    assert "tool_end" in types, "应发出 tool_end"
    assert result["messages"], "应累积消息"

    # 验证工具实际执行：demo.txt 应存在
    demo = Path(ws) / "demo.txt"
    print("demo.txt exists:", demo.exists())
    assert demo.exists(), "write_file 应真正创建文件"

    print("GRAPH E2E TEST PASSED")


asyncio.run(main())
