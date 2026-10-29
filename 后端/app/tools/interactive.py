"""交互工具：结构化澄清（用户点选/填写后继续），避免模型在正文里自问自答。"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from .context import ToolContext
from .registry import tool_spec


def _split_options(raw: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[\n,;，；]+", raw or ""):
        item = part.strip()
        if item and item not in out:
            out.append(item)
    return out[:8]


def make_interactive_tools(ctx: ToolContext):

    @tool_spec(category="general", timeout=0)
    @tool(parse_docstring=True)
    async def ask_user(question: str, options: str = "", allow_multiple: bool = False) -> str:
        """
        向用户提出一个需要其拍板的问题，并等待选择或回答（不要自己假设答案）

        适用：多个方案二选一、确认删除/覆盖/发布等不可逆操作、缺少关键信息
        （目标目录、连接方式、账号来源）无法继续时。用户回答后工具才返回。
        能用上下文推断的细节不要问；一次只问一个最关键的问题。

        Args:
            question: 问题文本，一句话说清要用户决定什么
            options: 可选，候选答案用换行或逗号分隔（最多 8 个）；留空表示让用户自由填写
            allow_multiple: 可选，是否允许多选（默认 False 单选）

        Returns:
            用户的选择/回答文本；用户取消或超时未答时返回对应说明
        """
        ctx.check_stopped()
        question = (question or "").strip()
        if not question:
            return "错误：question 不能为空。"
        choices = _split_options(options)
        answers = await ctx.questions.ask(
            ctx.emit_event,
            "ask_user",
            question,
            choices,
            allow_multiple=bool(allow_multiple) and len(choices) > 1,
            stop_event=ctx.stop_event,
        )
        if not answers:
            return (
                "用户未回答（已取消或等待超时）。不要假设答案继续执行："
                "请先完成能确定的部分，或在最终回答中列出需要用户确认的事项。"
            )
        return f"用户回答：{'；'.join(answers)}"

    return (ask_user,)
