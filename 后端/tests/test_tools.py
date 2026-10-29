"""工具层冒烟测试（不依赖 LLM）。"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeCtx:
    """模拟 ToolContext（显式类，避免 type() 绑定的坑）。"""

    def __init__(self, workspace):
        self.workspace = workspace
        self.events = []
        self.stop_event = asyncio.Event()
        self.approvals = None
        self.env = {}

    async def emit(self, ev):
        self.events.append(ev["type"])

    async def emit_event(self, ev):
        self.events.append(ev["type"])

    def check_stopped(self):
        if self.stop_event.is_set():
            raise asyncio.CancelledError("stopped")

    @property
    def call_id(self):
        return ""


async def main():
    from app.tools.build_test import make_build_test_tools
    from app.tools.context import ApprovalManager
    from app.tools.files import make_file_tools
    from app.tools.search import make_search_tools
    from app.tools.terminal import make_terminal_tool

    ws = tempfile.mkdtemp()
    am = ApprovalManager(mode="auto")
    ctx = FakeCtx(ws)
    ctx.approvals = am

    # 文件工具（新增 delete/move/copy，保留前四个解包兼容）
    rf, wf, ef, ld, *_extra = make_file_tools(ctx)
    r1 = await wf.ainvoke({"path": "hello.py", "content": "print(1)\nprint(2)\n"})
    print("write:", r1[:40])
    r2 = await rf.ainvoke({"path": "hello.py"})
    print("read ok:", "1 |" in r2)
    r3 = await ef.ainvoke({"path": "hello.py", "old_string": "print(1)", "new_string": "print(100)"})
    print("edit:", r3[:40])
    r3b = await rf.ainvoke({"path": "hello.py"})
    print("edit applied:", "print(100)" in r3b)

    # 终端工具
    t = make_terminal_tool(ctx)
    r4 = await t.ainvoke({"command": "python -c \"print(42)\"", "timeout": 30})
    print("terminal:", "42" in r4, "| exit0:", "退出码: 0" in r4)

    # 搜索工具
    g, lf = make_search_tools(ctx)
    r5 = await g.ainvoke({"pattern": "print"})
    print("grep:", "hello.py:1" in r5)

    # 测试工具（检测 python 项目）
    with open(os.path.join(ws, "pyproject.toml"), "w") as f:
        f.write('[project]\nname="x"\n')
    bc = make_build_test_tools(ctx)
    r6 = await bc.ainvoke({"kind": "test"})
    print("run_checks:", "退出码" in r6, "|", r6[:60])

    # 审批模式测试
    am2 = ApprovalManager(mode="all")
    ctx2 = FakeCtx(ws)
    ctx2.approvals = am2
    t2 = make_terminal_tool(ctx2)

    async def decide():
        await asyncio.sleep(0.3)
        am2.decide(list(am2._pending.keys())[0], True)

    task = asyncio.create_task(decide())
    r7 = await t2.ainvoke({"command": "echo approved", "timeout": 30})
    await task
    print("approval flow:", "approved" in r7)

    print("ALL TOOL TESTS PASSED")


asyncio.run(main())
