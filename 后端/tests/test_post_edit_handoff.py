"""验证落盘后的思考交接：不再回填上一轮“重写整个文件”的思考，改为提示先读后改。"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RecordingMockLLM:
    _llm_type = "mock"

    def __init__(self):
        self.calls = 0
        self.seen: list[list] = []

    async def astream(self, messages, **kwargs):
        from langchain_core.messages import AIMessageChunk

        self.calls += 1
        self.seen.append(list(messages))
        if self.calls > 2:
            yield AIMessageChunk(content="完成")
            return

        plan = "上一轮结论：让我重写整个文件" if self.calls == 1 else "再次重写整个文件"
        yield AIMessageChunk(
            content="", additional_kwargs={"reasoning_content": plan},
        )
        payload = json.dumps(
            {
                "path": "demo.html",
                "content": "<html>" + ("x" if self.calls == 1 else "y") * 200 + "</html>",
            }
        )
        yield AIMessageChunk(content="", tool_call_chunks=[
            {"name": "write_file", "args": payload, "id": f"call_{self.calls}", "index": 0},
        ])

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kw):
        return self


async def main():
    import app.agent.graph as g
    from app.agent.graph import execute_node
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools

    mock = RecordingMockLLM()
    original = g.init_chat_models
    g.init_chat_models = lambda **kw: mock

    ws = tempfile.mkdtemp()
    events = []

    async def emit(ev):
        events.append(ev)

    ctx = ToolContext(
        workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto")
    )
    wf = make_file_tools(ctx)[1]  # write_file

    state = {
        "request": "创建一个 html 页面", "workspace": ws, "messages": [], "plan": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "r1", "session_id": "s1", "stopped": False, "error": None,
        "emit": emit, "approvals": ApprovalManager(mode="auto"),
        "stop_event": asyncio.Event(), "_tools": [wf],
    }
    try:
        out = await execute_node(state)
        assert not out.get("error"), out.get("error")
        assert (Path(ws) / "demo.html").is_file(), "文件应已落盘"

        assert len(mock.seen) >= 3, "应至少有三轮模型调用"
        second = mock.seen[1]
        third = mock.seen[2]
        joined2 = "\n".join(str(getattr(m, "content", "") or "") for m in second)
        joined3 = "\n".join(str(getattr(m, "content", "") or "") for m in third)

        for label, joined in (("第二轮", joined2), ("第三轮", joined3)):
            assert "落盘后交接" in joined, f"{label}应收到落盘交接提示"
            assert "edit_file" in joined, f"{label}落盘交接应引导使用 edit_file"
            assert "不要再次整份 write_file" in joined, \
                f"{label}落盘交接应明确禁止再次整份覆盖"
        assert "上一轮结论：让我重写整个文件" not in joined2, \
            "不应再把上一轮“重写整个文件”的思考回填给模型"
        assert "再次重写整个文件" not in joined3, \
            "模型第二轮仍整份重写后，第三轮也不应回填它的重写计划"

        def write_result(messages):
            for m in reversed(messages):
                if type(m).__name__ == "ToolMessage" and getattr(m, "name", "") == "write_file":
                    return str(m.content)
            return ""

        result2 = write_result(second)
        result3 = write_result(third)
        assert "已创建文件" in result2 and "edit_file" in result2, result2
        assert "已覆盖文件" in result3 and "edit_file" in result3, result3
        print("首次写入结果:", result2[:130])
        print("重复写入结果:", result3[:130])
        print("每轮落盘后都改用交接提示，未回填旧的重写计划")
        print("ALL PASS")
    finally:
        g.init_chat_models = original


asyncio.run(main())
