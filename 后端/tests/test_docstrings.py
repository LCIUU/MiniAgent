"""验证工具 docstring 解析：@tool(parse_docstring=True) 是否生效。"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main():
    from app.tools import build_tools
    from app.tools.context import ApprovalManager, ToolContext

    ws = tempfile.mkdtemp()
    events = []

    async def emit(ev):
        events.append(ev)

    ctx = ToolContext(workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto"))
    tools = build_tools(ws, ctx.emit, ctx.approvals, ctx.stop_event)

    print(f"共 {len(tools)} 个工具\n")
    all_ok = True
    for t in tools:
        args = t.args_schema.model_json_schema()
        props = args.get("properties", {})
        desc = t.description or ""
        print(f"━━ {t.name} ━━")
        print(f"  描述: {desc[:60]}...")
        for pname, pmeta in props.items():
            pd = pmeta.get("description", "")
            print(f"  参数 {pname}: {pd[:60]}{' ✓' if pd else ' ✗ 无描述!'}")
            if not pd:
                all_ok = False
        if not desc:
            print("  ✗ 工具无描述!")
            all_ok = False
        print()
    print("=" * 40)
    print("ALL DOCSTRINGS PARSED:" , all_ok)


asyncio.run(main())
