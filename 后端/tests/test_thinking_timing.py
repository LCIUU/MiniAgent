"""验证思考耗时信号：短思考被闸门暂存时仍带回真实时长与 thinking_end。"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MockLLM:
    _llm_type = "mock"  # SummarizationMiddleware 探测 token 计数器时读取

    def __init__(self):
        self.calls = 0

    async def astream(self, messages, **kwargs):
        from langchain_core.messages import AIMessageChunk

        self.calls += 1
        if self.calls > 1:
            yield AIMessageChunk(content="已完成")
            return

        # 低于闸门阈值（240 字）的短思考，正常会被暂存到轮末才下发；
        # 事件必须自带真实时长，否则前端按收到时间计时只有 0ms。
        yield AIMessageChunk(
            content="", additional_kwargs={"reasoning_content": "我先创建演示文件"},
        )
        await asyncio.sleep(0.08)  # 模拟思考耗时
        payload = json.dumps({"path": "demo.txt", "content": "hi"})
        yield AIMessageChunk(content="", tool_call_chunks=[
            {"name": "write_file", "args": payload, "id": "call_1", "index": 0},
        ])

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def with_retry(self, **kwargs):
        return self


async def main():
    import app.agent.graph as g
    from app.agent.graph import execute_node
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools

    original = g.init_chat_models
    g.init_chat_models = lambda **kw: MockLLM()

    ws = tempfile.mkdtemp()
    events = []

    async def emit(ev):
        events.append(ev)

    ctx = ToolContext(workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto"))
    wf = make_file_tools(ctx)[1]  # write_file

    state = {
        "request": "x", "workspace": ws, "messages": [], "plan": [],
        "tool_rounds": 0, "summary": "",
        "summary_count": 0, "run_id": "r", "session_id": "s", "stopped": False, "error": None,
        "emit": emit, "approvals": ApprovalManager(mode="auto"),
        "stop_event": asyncio.Event(), "_tools": [wf],
    }
    try:
        s2 = await execute_node(state)
        assert not s2.get("error")
        thinks = [e for e in events if e["type"] == "thinking"]
        ends = [e for e in events if e["type"] == "thinking_end"]
        print("thinking events:", [
            (len(e["delta"]), e.get("duration_ms")) for e in thinks
        ])
        print("thinking_end:", [e.get("duration_ms") for e in ends])
        assert thinks and thinks[0].get("duration_ms") is not None, \
            "短思考事件缺少后端真实时长"
        assert thinks[0]["duration_ms"] >= 50, thinks[0]["duration_ms"]
        assert ends and ends[0].get("duration_ms") == thinks[0]["duration_ms"], \
            "thinking_end 时长应与思考事件一致"
        types = [e["type"] for e in events]
        assert types.index("thinking") < types.index("thinking_end") < types.index("tool_start"), \
            f"事件顺序错误: {types}"
        print("SUCCESS: 思考事件携带真实耗时，并在工具开始前收口")
    finally:
        g.init_chat_models = original


asyncio.run(main())
