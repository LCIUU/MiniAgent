"""落盘兜底检测验证：软提醒方案（前两次提醒 → 第三次仍不写则放行）。"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_state(ws, emit, wf):
    import asyncio as _a
    from app.tools.context import ApprovalManager

    return {
        "request": "创建一个 html 页面", "workspace": ws, "messages": [], "plan": [],
        "tool_rounds": 0,
        "summary": "", "summary_count": 0, "run_id": "r1", "session_id": "s1",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"),
        "stop_event": _a.Event(), "_tools": [wf],
    }


class MockStubbornNoWrite:
    """始终只贴代码不写文件：前两次应提醒，第三次应放行（不再提醒）。"""

    _llm_type = "mock"

    def __init__(self):
        self.n = 0

    async def astream(self, messages, **kwargs):
        self.n += 1
        yield type("M", (), {
            "content": "这是代码：\n```html\n<!DOCTYPE html>\n<html>\n<body>hi</body>\n</html>\n```",
            "tool_calls": [],
        })()

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kw):
        return self


class MockNudgeThenWrite:
    """第 1 轮：贴代码不写文件（应触发提醒）；第 2 轮：调用 write_file；第 3 轮：结束。"""

    _llm_type = "mock"

    def __init__(self):
        self.n = 0

    async def astream(self, messages, **kwargs):
        self.n += 1
        if self.n == 1:
            yield type("M", (), {
                "content": "这是代码：\n```html\n<!DOCTYPE html>\n<html>\n<body>hi</body>\n</html>\n```",
                "tool_calls": [],
            })()
        elif self.n == 2:
            yield type("M", (), {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "write_file",
                                "args": {"path": "page.html", "content": "<html>hi</html>"}}],
            })()
        else:
            yield type("M", (), {"content": "已写入 page.html", "tool_calls": []})()

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kw):
        return self


async def run_case(name: str, mock_cls):
    from app.agent.graph import execute_node
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools
    import app.agent.graph as g

    original = g.init_chat_models
    g.init_chat_models = lambda **kw: mock_cls()
    ws = tempfile.mkdtemp()
    events = []

    async def emit(ev):
        events.append(ev)

    ctx = __import__("app.tools.context", fromlist=["ToolContext"]).ToolContext(
        workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto")
    )
    wf = make_file_tools(ctx)[1]  # write_file
    state = make_state(ws, emit, wf)
    try:
        s2 = await execute_node(state)
        nudge_events = [e for e in events if e.get("status") == "file_write_nudge"]
        return {
            "name": name,
            "nudge_count": len(nudge_events),
            "file_created": (Path(ws) / "page.html").exists(),
            "error": s2.get("error"),
            "rounds": s2.get("tool_rounds"),
        }
    finally:
        g.init_chat_models = original


async def main():
    # 场景 A：提醒后自觉补写
    r1 = await run_case("提醒后补写", MockNudgeThenWrite)
    print(r1)
    assert r1["nudge_count"] == 1, "应触发一次提醒"
    assert r1["file_created"], "补写后文件应存在"
    assert r1["error"] is None

    # 场景 B：顽固不写 → 落盘提醒两次 + 提前收尾提醒一次后放行
    r2 = await run_case("顽固不写", MockStubbornNoWrite)
    print(r2)
    assert r2["nudge_count"] == 2, "应触发两次落盘提醒"
    assert r2["error"] is None, "应正常结束而非卡死"
    assert r2["rounds"] == 4, "提醒三轮 + 放行一轮"

    print("ALL PASS: 软提醒行为正确（提醒两次 → 提前收尾提醒一次 → 放行）")


asyncio.run(main())
