"""验证流式工具调用参数聚合（模拟 OpenAI 增量 chunk）。"""
import asyncio
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
            yield AIMessageChunk(content="已创建 demo.txt")
            return

        # 模拟 OpenAI 流式：分多次增量输出 tool call 参数（JSON 字符串片段）
        yield AIMessageChunk(content="", tool_call_chunks=[
            {"name": "write_file", "args": '{"path": "dem', "id": "call_1", "index": 0},
        ])
        yield AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": 'o.txt", "content": "hi"}', "id": "call_1", "index": 0},
        ])
        yield AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": "", "id": "call_1", "index": 0},
        ])

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def with_retry(self, **kwargs):
        # SummarizationMiddleware 构造时会调用 with_retry()，测试替身原样返回
        return self


async def main():
    from app.agent.graph import execute_node
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools
    import app.agent.graph as g

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
        s2 = await execute_node(state)  # mock 输出 write_file 工具调用
        demo = Path(ws) / "demo.txt"
        print("demo.txt exists:", demo.exists())
        print("error:", s2.get("error"))
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        for ts in tool_starts:
            print("tool_start args:", ts.get("args"))
        if not demo.exists():
            print("!!! 工具参数未正确聚合，write_file 未执行或参数为空")
            return
        print("SUCCESS: 流式工具参数聚合正确，demo.txt 已创建")
    finally:
        g.init_chat_models = original


asyncio.run(main())
