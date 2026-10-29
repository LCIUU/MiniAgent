"""护栏中断后消息序列合法性：AIMessage.tool_calls 后必须紧跟对应 ToolMessage。

OpenAI 兼容端点在收到「tool_calls 未配对」的历史时会直接返回 400，
曾出现重复读取/重复计划护栏触发后 run 直接报错的问题。
"""
import asyncio
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MockDuplicateRead:
    """读取同一文件 → 重复读取（应被拦截）→ 给出最终回答。"""

    _llm_type = "mock"

    def __init__(self):
        self.n = 0

    async def astream(self, messages, **kwargs):
        self.n += 1
        if self.n == 1:
            yield type("M", (), {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}],
            })()
        elif self.n == 2:
            yield type("M", (), {
                "content": "",
                "tool_calls": [{"id": "c2", "name": "read_file", "args": {"path": "a.txt"}}],
            })()
        else:
            yield type("M", (), {"content": "已读取 a.txt", "tool_calls": []})()

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kw):
        return self


def assert_tool_calls_paired(messages) -> None:
    for i, msg in enumerate(messages):
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            continue
        pending = {tc["id"] for tc in tool_calls}
        for follow in messages[i + 1 :]:
            if not pending:
                break
            assert isinstance(follow, ToolMessage), (
                f"第 {i} 条 AIMessage 的 tool_calls 未配对，后面出现了 {type(follow).__name__}: "
                f"{getattr(follow, 'content', '')[:80]!r}"
            )
            pending.discard(follow.tool_call_id)
        assert not pending, f"第 {i} 条 AIMessage 缺少 ToolMessage: {pending}"


async def main() -> None:
    import app.agent.graph as g
    from app.agent.graph import execute_node
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools

    original = g.init_chat_models
    ws = tempfile.mkdtemp()
    (Path(ws) / "a.txt").write_text("hello", encoding="utf-8")
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    ctx = ToolContext(workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto"))
    tools = list(make_file_tools(ctx))
    state = {
        "request": "看看 a.txt", "workspace": ws, "messages": [], "plan": [],
        "tool_rounds": 0,
        "summary": "", "summary_count": 0, "run_id": "r1", "session_id": "s1",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": tools,
    }
    g.init_chat_models = lambda **kw: MockDuplicateRead()
    try:
        result = await execute_node(state)
    finally:
        g.init_chat_models = original

    assert result.get("error") is None, result.get("error")
    assert any(e.get("status") == "duplicate_read_blocked" for e in events), "应触发重复读取拦截"
    assert_tool_calls_paired(result["messages"])
    print("PASS: 护栏中断后 tool_calls 全部配对，消息序列合法")


async def history_case() -> None:
    """历史里有写文件调用时，本轮“只说不写”仍必须触发落盘提醒。"""
    import app.agent.graph as g
    from app.agent.graph import execute_node
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools
    from langchain_core.messages import AIMessage, HumanMessage

    class MockCodeOnly:
        _llm_type = "mock"

        async def astream(self, messages, **kwargs):
            yield type("M", (), {
                "content": "这是代码：\n```html\n<html>\n<body>hi</body>\n</html>\n```",
                "tool_calls": [],
            })()

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    original = g.init_chat_models
    ws = tempfile.mkdtemp()
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    history = [
        HumanMessage(content="上一轮：创建旧页面"),
        AIMessage(content="", tool_calls=[{
            "id": "old1", "name": "write_file", "args": {"path": "old.html", "content": "x"},
        }]),
        ToolMessage(content="已写入 old.html", tool_call_id="old1", name="write_file"),
        AIMessage(content="已完成"),
    ]
    ctx = ToolContext(workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto"))
    state = {
        "request": "创建一个新页面", "workspace": ws, "messages": history, "plan": [],
        "tool_rounds": 0,
        "summary": "", "summary_count": 0, "run_id": "r2", "session_id": "s2",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": list(make_file_tools(ctx)),
    }
    g.init_chat_models = lambda **kw: MockCodeOnly()
    try:
        result = await execute_node(state)
    finally:
        g.init_chat_models = original

    statuses = [e.get("status") for e in events]
    assert "file_write_nudge" in statuses, f"历史写操作不应掩盖本轮未落盘: {statuses}"
    assert_tool_calls_paired(result["messages"])
    print("PASS: 历史写操作不影响本轮落盘检测")


asyncio.run(main())
asyncio.run(history_case())
