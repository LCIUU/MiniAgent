"""上下文管理：中英文感知的 token 预算 + 自定义摘要（不依赖 LangChain 私有 API）。

旧实现直接调用 ``SummarizationMiddleware`` 的私有方法（``_should_summarize``、
``_determine_cutoff_index`` 等），LangChain 小版本升级就会断。这里改为：
1. 自己估算 token（中文 1 字 ≈ 1.3 token）
2. 自己决定保留最近 N 条消息，并保证 AI tool_call 与 ToolMessage 不被拆散
3. 用主模型生成结构化交接摘要，作为带标记的 HumanMessage 插到最前面
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..config import get_effective, resolve_model_endpoint

logger = logging.getLogger(__name__)

_CJK_CHAR_RE = re.compile(
    r"[\u4e00-\u9fff"
    r"\u3400-\u4dbf"
    r"\U00020000-\U0002a6df"
    r"\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f"
    r"\U0002b820-\U0002ceaf"
    r"\U0002ceb0-\U0002ebef"
    r"\U00030000-\U0003134f"
    r"\uf900-\ufaff"
    r"]"
)

SUMMARY_MARKER = "【历史上下文摘要】"

SUMMARY_PROMPT = """你正在为一段 AI 编程助手的对话历史生成"上下文交接摘要"。这段摘要将替换较早的原始对话历史，是模型后续可见的长期记忆，必须准确保留继续任务所需的全部关键信息。

输入分两部分：
- <messages>：需要压缩的较早历史；
- <recent_messages>：仍会原样保留在上下文里的最近消息，用于校准最新目标与进展。
两者冲突时，一律以 <recent_messages> 为准。

按以下结构输出（无内容的小节写"无"）：

## 用户目标与约束
用户的最终任务目标；用户明确提出的要求、偏好、纠正与否定过的做法（原文级准确，后续绝对不得违背）。必须反映最近一次用户要求：如果最新要求改变了旧目标（例如数量从 60 改为 70），直接写新目标，不要把旧摘要里的过期目标当作当前目标。

## 文件状态
已读取/创建/修改的文件路径清单，每个文件注明：具体改动内容、当前状态（是否已验证、是否存在遗留问题）。文件路径与关键代码标识必须精确原文保留。

## 关键决策与失败记录
已确定的技术选型、方案取舍及原因；已尝试但失败或被放弃的方案及失败原因（防止重复踩坑）。

## 最近进展
基于 <recent_messages> 用不超过 6 条概括：最近正在做什么、刚完成或正在修改哪些文件、下一步是什么。确保摘要能无缝衔接到仍然保留的最近上下文。

## 当前进度与下一步
任务完成到哪一步；尚未完成的步骤；接下来应该做什么（具体到文件与操作）。

要求：条目化、精炼；命令、配置值、错误信息、版本号等精确内容原文保留，不得改写；省略寒暄与推理过程；只输出摘要本身，不要任何额外说明。

<messages>
{messages}
</messages>

<recent_messages>
{recent_messages}
</recent_messages>"""


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def count_tokens_zh_aware(
    messages: Iterable[Any],
    *,
    cjk_tokens_per_char: float = 1.3,
    non_cjk_chars_per_token: float = 4.0,
    extra_tokens_per_message: float = 3.0,
) -> int:
    """中英文混合场景的近似 token 计数器。"""
    total = 0.0
    for msg in messages:
        text = _message_text(msg)
        cjk = len(_CJK_CHAR_RE.findall(text))
        total += cjk * cjk_tokens_per_char + (len(text) - cjk) / non_cjk_chars_per_token
        total += extra_tokens_per_message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            tc_text = json.dumps(tool_calls, ensure_ascii=False)
            cjk_tc = len(_CJK_CHAR_RE.findall(tc_text))
            total += cjk_tc * cjk_tokens_per_char + (len(tc_text) - cjk_tc) / non_cjk_chars_per_token
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            total += len(tool_call_id) / non_cjk_chars_per_token
    return int(total)


def _is_summary_message(msg: Any) -> bool:
    if isinstance(msg, HumanMessage) and SUMMARY_MARKER in _message_text(msg)[:100]:
        return True
    return bool(getattr(msg, "additional_kwargs", {}).get("is_summary"))


def _eligible_start(messages: list[BaseMessage], keep: int) -> int:
    """计算摘要截点：保留最近 keep 条，但不拆散 AI tool_call 与 ToolMessage。"""
    if len(messages) <= keep:
        return 0
    start = len(messages) - keep
    # 截点落在 ToolMessage 上时回退到产生它的 AIMessage（带 tool_calls）。
    # 回退后的 AIMessage 自身即为合法起点，不能再往前退：更前一条通常是
    # 上一轮的 ToolMessage，那样会让保留段以孤立工具结果开头。
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return max(0, start)


def _tail_budget_start(messages: list[BaseMessage], budget: int, min_keep: int) -> int:
    """按 token 预算倒序累加，返回保留窗口起点（尽量不超预算）。

    固定条数（keep=30）在工具结果很大时会把一堆超长消息留在窗口里，压缩完仍然
    顶在上限之上，于是每轮都重新压缩、每轮都换一份摘要——上下文前缀反复重建，
    隐式缓存全部失效。这里改成按 token 预算收缩：至少保留 min_keep 条（当前任务
    不能整段被摘要掉），其余按预算从近到远保留。
    """
    if not messages:
        return 0
    costs = [count_tokens_zh_aware([m]) for m in messages]
    start = len(messages)
    total = 0
    while start > 0:
        if len(messages) - start >= min_keep and total + costs[start - 1] > budget:
            break
        total += costs[start - 1]
        start -= 1
    # 与 _eligible_start 同一规则：保留段不能以孤立的 ToolMessage 开头
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return start


def _db_ts(msg: Any) -> float | None:
    value = getattr(msg, "additional_kwargs", {}).get("db_ts")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _summary_cutoff(
    summarized: list[BaseMessage], preserved: list[BaseMessage]
) -> float | None:
    """摘要覆盖到的最后一条 DB 消息时间。

    只取严格早于保留窗口首条 DB 消息的最大时间戳：若截点落在某行消息中间，
    该行整体保留（宁可少量重复，也不能丢掉未摘要的内容）。
    """
    summarized_ts = {t for m in summarized if (t := _db_ts(m)) is not None}
    if not summarized_ts:
        return None
    preserved_ts = {t for m in preserved if (t := _db_ts(m)) is not None}
    if not preserved_ts:
        return max(summarized_ts)
    first_kept = min(preserved_ts)
    older = [t for t in summarized_ts if t < first_kept]
    return max(older) if older else None


def extract_summary_cutoff(messages: list[BaseMessage]) -> float | None:
    """取摘要消息携带的 DB 截止时间（供跨轮跳过已摘要历史）。"""
    for msg in messages:
        if _is_summary_message(msg):
            value = getattr(msg, "additional_kwargs", {}).get("db_cutoff")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)
    return None


def _summary_input_max_chars() -> int:
    """摘要输入上限：默认 36000 字符，过小会丢中间历史。"""
    try:
        value = int(get_effective("summary_input_max_chars", 36000) or 0)
    except (TypeError, ValueError):
        value = 36000
    return min(max(value, 8000), 120000)


def _clip_text(text: str, limit: int) -> str:
    """按字符上限保留头尾，避免单条超长消息整条被丢弃。"""
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head)
    return text[:head] + "\n…（超长截断）…\n" + text[-tail:]


def _format_for_summary(messages: list[BaseMessage], max_chars: int | None = None) -> str:
    """把待摘要消息转成紧凑文本；超长时保留头尾，且最近消息必须入选。"""
    if max_chars is None:
        max_chars = _summary_input_max_chars()
    max_chars = max(200, max_chars)
    blocks: list[str] = []
    for msg in messages:
        # 内部提醒是系统自动产生的，不属于对话内容，不交给摘要模型
        if getattr(msg, "additional_kwargs", {}).get("is_nudge"):
            continue
        role = getattr(msg, "type", "unknown")
        text = _message_text(msg)
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            text += "\n[tool_calls] " + json.dumps(msg.tool_calls, ensure_ascii=False)
        if len(text) > 6000:
            text = _clip_text(text, 6000)
        blocks.append(f"[{role}] {text}")

    if sum(len(b) for b in blocks) <= max_chars:
        return "\n\n".join(blocks)

    # 最近的待摘要消息优先入选：从尾部向前填，单条超预算时裁剪而不是整条丢弃
    tail_budget = max(1000, max_chars // 2)
    tail_blocks: list[str] = []
    tail_used = 0
    for block in reversed(blocks):
        if not tail_blocks:
            clipped = _clip_text(block, tail_budget)
            tail_blocks.append(clipped)
            tail_used += len(clipped)
            continue
        if tail_used + len(block) > tail_budget:
            break
        tail_blocks.append(block)
        tail_used += len(block)
    tail_blocks.reverse()
    tail_start = len(blocks) - len(tail_blocks)

    head_budget = max_chars - tail_used
    head_blocks: list[str] = []
    head_used = 0
    for block in blocks[:tail_start]:
        if head_used + len(block) > head_budget:
            break
        head_blocks.append(block)
        head_used += len(block)

    omitted = tail_start - len(head_blocks)
    parts = list(head_blocks)
    if omitted > 0:
        omitted_chars = sum(len(b) for b in blocks[len(head_blocks):tail_start])
        parts.append(f"…（中间省略 {omitted} 条消息，约 {omitted_chars} 字）…")
    parts.extend(tail_blocks)
    return "\n\n".join(parts)


def _is_recent_context_message(msg: BaseMessage) -> bool:
    """最近上下文里只保留真实对话，跳过内部提醒和旧摘要。"""
    if _is_summary_message(msg):
        return False
    return not bool(getattr(msg, "additional_kwargs", {}).get("is_nudge"))


def _format_recent_context(
    preserved: list[BaseMessage],
    max_chars: int = 8000,
    tail_messages: int = 12,
) -> str:
    """保留窗口的紧凑快照，供摘要模型校准最新目标和进展。

    保留窗口本身不会被压缩，但最近一次用户要求和刚发生的操作只存在于这里；
    不把它们交给摘要模型，摘要里的目标就会停留在旧版本。
    """
    if not preserved:
        return ""
    tail = list(preserved[-tail_messages:])
    last_user = next(
        (m for m in reversed(preserved)
         if isinstance(m, HumanMessage) and _is_recent_context_message(m)),
        None,
    )
    header = ""
    if last_user is not None:
        # 最新用户要求单独置顶并预留预算，避免在最近消息过多时被裁掉
        tail = [m for m in tail if m is not last_user]
        user_text = _clip_text(_message_text(last_user), 2000)
        header = f"【最近一次用户消息（摘要目标必须以此为准）】\n[human] {user_text}"
    budget = max(1000, max_chars - len(header) - 2)
    tail_text = _format_for_summary(tail, max_chars=budget)
    return "\n\n".join(part for part in (header, tail_text) if part)


def _resolve_threshold(model: str = "") -> int:
    """摘要阈值：显式配置优先，否则按模型上下文窗口自动计算。"""
    try:
        explicit = int(get_effective("summary_threshold_tokens", 0) or 0)
    except (TypeError, ValueError):
        explicit = 0
    if explicit > 0:
        return explicit

    model = (model or str(get_effective("model", "") or "")).strip()
    window = 0
    if model:
        try:
            window = int(resolve_model_endpoint(model).get("context_window") or 0)
        except Exception:
            window = 0
    if window <= 0:
        return 60000

    try:
        ratio = float(get_effective("summary_threshold_ratio", 0.6))
    except (TypeError, ValueError):
        ratio = 0.6
    ratio = min(max(ratio, 0.1), 0.95)
    return max(8000, int(window * ratio))


def _keep_ratio() -> float:
    """压缩后保留窗口占阈值的比例（0.45 = 留出一半余量再触发下一次压缩）。"""
    try:
        ratio = float(get_effective("summary_keep_ratio", 0.45))
    except (TypeError, ValueError):
        ratio = 0.45
    return min(max(ratio, 0.1), 0.9)


def _min_keep_messages() -> int:
    """保留窗口的条数下限：再省也要留下当前任务的最后几次往返。"""
    try:
        value = int(get_effective("summary_min_keep_messages", 4))
    except (TypeError, ValueError):
        value = 4
    return max(value, 1)


async def maybe_summarize_messages(
    messages: list[BaseMessage],
    llm: Any,
    model: str = "",
    emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> list[BaseMessage]:
    """检查 token 预算，超限则把早期消息摘要成一条带标记的 HumanMessage。

    阈值：显式 summary_threshold_tokens > 0 时优先；否则按模型的 context_window
    × summary_threshold_ratio 自动计算（目录外模型回退 60k）。

    保留窗口先按 summary_keep_messages 条数取，若这部分仍超过
    summary_keep_ratio × 阈值，则继续按 token 预算收缩（下限
    summary_min_keep_messages 条），保证压缩后留出足够余量、不会下一轮立刻再压缩
    ——每次压缩都会改写上下文前缀，重复压缩等于把隐式缓存全部作废。

    emit 传入时，摘要开始/结束会各广播一条事件，供前端展示"正在优化上下文"。
    """
    if not messages:
        return messages
    try:
        keep = int(get_effective("summary_keep_messages", 30))
    except Exception:
        keep = 30
    threshold = _resolve_threshold(model)

    total = count_tokens_zh_aware(messages)
    triggered = total >= threshold
    # 每轮模型调用前打印估算值，终端可实时观察上下文 token 增长
    logger.info(
        "[summarize] %d 条消息，估算 %d tokens（阈值 %d），触发=%s",
        len(messages), total, threshold, triggered,
    )
    if not triggered:
        return messages

    # 保留开头的 SystemMessage（系统提示/项目规则）；旧摘要并入待摘要内容，
    # 避免输出两条摘要（extract_summary 只取第一条，会导致 session.summary 永不更新）
    head: list[BaseMessage] = []
    body = list(messages)
    while body and isinstance(body[0], SystemMessage):
        head.append(body.pop(0))
    previous_summary: BaseMessage | None = None
    if body and _is_summary_message(body[0]):
        previous_summary = body.pop(0)

    # 保留窗口：先按条数（summary_keep_messages），再按 token 预算收缩。
    # 条数口径单独使用时有两个坑：大工具结果会让压缩后仍顶在阈值之上（下一轮立刻
    # 再压缩，每轮换摘要 = 每轮缓存全灭）；消息条数不足 keep 时又完全不压缩，
    # 少量超长消息能把上下文顶穿模型窗口。
    budget = max(1, int(threshold * _keep_ratio()))
    min_keep = max(1, min(_min_keep_messages(), len(body) - 1))
    start = _eligible_start(body, keep)
    tail_tokens = count_tokens_zh_aware(body[start:])
    if start <= 0 and tail_tokens <= budget:
        logger.info("[summarize] 可摘要消息不足（keep=%d）且未超保留预算，跳过", keep)
        return messages
    if tail_tokens > budget:
        start = max(start, _tail_budget_start(body, budget, min_keep))
    if len(body) > 1:
        start = min(start, len(body) - 1)  # 至少留下最后一条（当前用户任务）
    if start <= 0:
        logger.info("[summarize] 保留窗口已覆盖全部消息，跳过")
        return messages
    to_summarize = ([previous_summary] if previous_summary is not None else []) + body[:start]
    preserved = body[start:]
    if not to_summarize:
        return messages
    cutoff = _summary_cutoff(body[:start], preserved)

    started = time.monotonic()
    if emit is not None:
        await emit({
            "type": "summary_start",
            "before_tokens": total,
            "messages_before": len(messages),
        })

    recent_context = _format_recent_context(preserved)
    prompt = SUMMARY_PROMPT.replace(
        "{messages}", _format_for_summary(to_summarize)
    ).replace("{recent_messages}", recent_context or "无")
    summary = ""
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = _message_text(response).strip()
    except Exception as e:
        logger.warning("上下文摘要失败，保留原消息: %s", e)

    result = messages
    if summary:
        summary_msg = HumanMessage(
            content=f"{SUMMARY_MARKER}\n{summary}",
            additional_kwargs={"is_summary": True, "db_cutoff": cutoff},
        )
        result = [*head, summary_msg, *preserved]
        after = count_tokens_zh_aware(result)
        logger.info(
            "[summarize] 完成：%d 条 → 摘要 + %d 条保留，%d → %d tokens（阈值 %d）",
            len(messages), len(preserved), total, after, threshold,
        )

    if emit is not None:
        await emit({
            "type": "summary_done",
            "ok": bool(summary),
            "before_tokens": total,
            "after_tokens": count_tokens_zh_aware(result) if summary else total,
            "messages_before": len(messages),
            "messages_after": len(result),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "summary": summary[:12000],
            "db_cutoff": cutoff,
        })
    return result


def extract_summary(messages: list[BaseMessage]) -> str:
    """从消息列表里取出摘要文本（供持久化到 session.summary）。"""
    for msg in messages:
        if _is_summary_message(msg):
            text = _message_text(msg)
            return text.replace(SUMMARY_MARKER, "", 1).strip()
    return ""


# 供类型检查/测试引用
_ = (AIMessage, ToolMessage)
