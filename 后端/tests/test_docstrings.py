"""工具描述质量守卫：每个工具必须提供详细、Google 风格的 docstring。

模型实际看到的是 LangChain 解析后的 description（``Args:`` 之前的部分），
参数说明则来自 ``Args:`` 段。这里按模型视角校验：

- 工具数量不少于 MIN_TOOLS（防止注册意外丢失）
- 工具描述不低于 MIN_DESC_CHARS，且正文包含摘要行之外的说明段落
- 每个参数都有不低于 MIN_ARG_DESC_CHARS 的说明
- ``Args:`` / ``Returns:`` 段不能泄漏进 description
"""
import asyncio
import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIN_TOOLS = 50
MIN_DESC_CHARS = 100
MIN_ARG_DESC_CHARS = 8


async def _collect_tools():
    from app.tools import build_tools
    from app.tools.context import ApprovalManager, ToolContext

    ws = tempfile.mkdtemp()

    async def emit(ev):
        return None

    ctx = ToolContext(workspace=ws, emit=emit, approvals=ApprovalManager(mode="auto"))
    return build_tools(ws, ctx.emit, ctx.approvals, ctx.stop_event)


def _problems(tools):
    problems: list[str] = []
    for t in tools:
        desc = (t.description or "").strip()
        if len(desc) < MIN_DESC_CHARS:
            problems.append(f"{t.name}: 描述过短（{len(desc)} < {MIN_DESC_CHARS} 字符）")
        fn = t.coroutine or t.func
        doc = inspect.getdoc(fn) or ""
        blocks = doc.split("\n\n")
        if not doc:
            problems.append(f"{t.name}: 缺少 docstring")
        elif "\n" in blocks[0]:
            problems.append(f"{t.name}: 摘要行后缺少空行（Google 风格要求摘要独立成段）")
        if any(ch < " " and ch not in "\n\t" for ch in doc):
            problems.append(f"{t.name}: docstring 含控制字符（\\1 等转义可能被错误解析，需写成 \\\\1）")
        if len(blocks) < 2 or blocks[1].startswith(("Args:", "Returns:")):
            problems.append(f"{t.name}: 缺少 Google 风格的扩展说明段落（摘要后需空行分段）")
        if (
            "\n\nArgs:" in doc
            and "\n\nReturns:" in doc
            and doc.index("\n\nArgs:") > doc.index("\n\nReturns:")
        ):
            problems.append(f"{t.name}: Args: 段必须位于 Returns: 之前")
        for marker in ("\nArgs:", "\nReturns:"):
            if marker in desc:
                problems.append(f"{t.name}: description 泄漏了 {marker.strip()}")
        props = t.args_schema.model_json_schema().get("properties", {})
        for pname, pmeta in props.items():
            pd = (pmeta.get("description") or "").strip()
            if len(pd) < MIN_ARG_DESC_CHARS:
                problems.append(
                    f"{t.name}.{pname}: 参数描述缺失或过短（{len(pd)} < {MIN_ARG_DESC_CHARS} 字符）"
                )
    return problems


def test_tool_descriptions_are_detailed():
    tools = asyncio.run(_collect_tools())
    assert len(tools) >= MIN_TOOLS, f"只注册了 {len(tools)} 个工具（预期 ≥ {MIN_TOOLS}）"
    problems = _problems(tools)
    assert not problems, "工具描述不符合规范：\n" + "\n".join(problems)
