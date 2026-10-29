"""Agent 核心：LangGraph 单循环 ReAct 图（规划由模型自主决定）。

图结构（结合 DeepAgents 思路的自定义实现）：
  START → execute_node（工具循环：LLM 决策 → 调用工具 → 观察结果；需要时模型自行
          调用 write_todos 制定/更新计划，直至产出最终答案）
        → context_node（上下文预算检查：超限则自动摘要）
        → END

关键设计：
- 不设独立规划阶段：是否规划、何时规划由模型判断——多步骤任务先调用 write_todos，
  简单问答直接作答，避免被迫先跑一次规划调用（省一轮 LLM、规划不再被锁死）
- write_todos 复用 langchain.agents.middleware.TodoListMiddleware 的工具、系统提示
  片段与状态 schema（todos: [{content, status}]），计划为结构化数据，与思考过程
  彻底分离，杜绝"闲聊文本被解析成计划步骤"的偏差
- 全程异步，事件通过 emit 回调以 SSE 协议流式推送
- 支持用户中途停止（stop_event）
- 工具审批由 ApprovalManager 处理（终端高危命令）
- 消息历史由中间件压缩（context.py）
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import re
import socket
import time
import uuid
import zlib
from typing import Any
from urllib.parse import urlsplit

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ..config import get_effective, resolve_model_endpoint
from ..services import tool_cache as tool_cache_store
from ..services.project_rules import workspace_context
from ..tools.context import current_call_id
from ..tools.registry import spec_of
from .context import extract_summary, maybe_summarize_messages
from .llm import init_chat_models
from .state import AgentState

logger = logging.getLogger(__name__)

# ===== TodoListMiddleware：write_todos 由模型在执行循环中自主调用 =====
# - write_todos 工具：模型通过结构化工具调用创建/更新计划
# - TODOS_PROMPT：中间件自带的使用说明（何时建计划、如何维护状态）
TODO_MIDDLEWARE = TodoListMiddleware()
WRITE_TODOS: BaseTool = TODO_MIDDLEWARE.tools[0]
TODOS_PROMPT: str = TODO_MIDDLEWARE.system_prompt

_VALID_TODO_STATUS = {"pending", "in_progress", "completed"}

SYSTEM_PROMPT = """
你是一位强大的 AI 编程助手，工作区为当前目录。严格遵循以下规则完成任务。

1. 任务识别
   - 用户指令明确具体时，直接执行，不主动探查文件系统。
   - 仅当指令模糊或依赖项目结构（如找不到入口文件、依赖未知）时，才使用 list_dir / list_files / read_file 获取必要信息。
   - 读取文件必须优先局部读取（通过搜索或偏移），禁止全量读取大文件。

2. 执行规划与代码落盘纪律（核心规则，违反视为任务未完成）
   - 是否制定计划由你自主判断：任务需要 3 个及以上步骤、或涉及多个文件/多轮验证时，
     先调用 write_todos 把任务拆解为可勾选的步骤，执行中随时更新状态（完成即标 completed）；
     一两步就能完成的简单任务或闲聊直接处理，不要调用 write_todos。
     计划只能通过 write_todos 工具记录，不要以正文形式罗列计划清单。
   - 多步骤任务须在内部拆解为原子操作，按顺序逐步执行。
   - 新建文件使用 write_file。
   - 修改现有文件优先使用 edit_file 进行精准差异修改，禁止用 write_file 覆盖整个大文件。
   - 修改前先 read_file 确认目标内容与缩进；edit_file 的 old_string 必须与文件内容精确一致且尽量唯一，避免反复试错。
   - ⚠️ 代码修改必须实际落盘：任何代码修改/新增/删除指令，必须调用 edit_file / write_file 直接写入目标文件；严禁只在回复文本中粘贴代码片段而不执行文件写入。回复中引用的代码只能是已写入内容的说明摘录，绝不能作为交付物。
   - 任务完成前必须自查：用户要求的每处修改是否都有对应的工具写入调用？没有则立即补写，不得直接结束。
   - 删除文件/目录一律用 delete_file（工具会向用户弹出确认卡，用户同意后才删除）；
     含删除的 apply_patch 同样会自动弹确认。禁止用 terminal 的 rm / del / Remove-Item 绕道删除文件。
   - 需要用户拍板（多方案取舍、是否覆盖/发布、缺少关键信息）时调用 ask_user 弹出选项卡，
     不要只在正文里提问然后自己猜答案。
   - ⚠️ 操作主体认知：你通过 write_file / edit_file 修改的文件，其内容变化是由你（AI 助手）执行的，不是用户手动修改的。在后续读取和推理时，不得将自己之前的修改归因于"用户操作"或"用户反馈"。

3. 代码质量
   - 修改必须匹配项目现有编码风格（缩进、引号、命名等）。
   - 引入新依赖前，检查项目依赖清单（如 package.json、requirements.txt 等）。若需新增，必须向用户说明库名、版本及安装命令。
   - 只修改任务相关的代码块，不得改动无关函数、配置或格式化空白。

4. 验证与容错
   - 验证要有针对性：只有项目确实存在测试/构建/检查配置时才运行验证。静态页面、纯文档或单文件小改动无需构建。
   - 不要每次小改都重复跑同一条验证命令：关键改动完成后集中验证一次，最终回答前再确认一次即可。
   - 验证命令必须保留完整输出：禁止用管道过滤裁剪错误信息（如 | Select-String、| findstr、| head），
     更不能只看 -Quiet 之类的 true/false 结果——看不到具体错误就无法定位问题，会陷入反复重试。
     输出过长时直接查看末尾或报错段落即可。
   - 验证失败时依据错误日志修复，同类失败最多重试 2 次；仍失败则停止并如实报告根本原因、已尝试方案与建议手动介入点。
   - 没有实际运行验证时，禁止在最终回答中声称"已测试通过/构建成功"，应明确写"未运行验证"及原因。

5. 安全边界
   - 所有文件读写和终端命令必须限定在工作区根目录内，禁止使用 ../ 或绝对路径访问系统目录。
   - 执行危险命令（如 rm -rf、sudo、chmod 777 等）前，必须向用户发出警告并等待明确确认（Y/N），否则禁止执行。

6. 输出规范
   - 内部推理过程不得完整输出。
   - 最终回复使用中文，根据任务复杂度包含：执行摘要、操作结果（文件变更/关键输出）、用户验证指引。
   - 只能陈述真实执行过的操作与真实结果，禁止虚构未执行的操作或未获得的验证结论。

7. 思考与工具调用纪律（重要，直接影响推理质量）
   - 推理过程必须聚焦「本步即将做出的决策」和「上一条工具结果带来的新信息」。
   - ⚠️ 严禁在推理（思考过程）中以任何形式复述、引用或改写用户原始提问的具体内容（例如「用户问我 xxx」「用户的需求是 xxx」「用户让我做 xxx」「用户想要 xxx」
     「根据用户的问题 xxx」等等），也不要总结之前已经思考过或已经完成的步骤。用户消息、历史上下文和你已经得出的结论你都记得，不需要再抄一遍。
   - ⚠️ 严禁思考中自我循环或无限自我纠正：一个决策问题最多自我纠正一次，之后必须收敛到行动。
     禁止反复自问「Wait, should I use X or Y?」「Wait, did I miss something?」这类无进展的循环——
     选一个最可能的方案直接执行，执行结果会给你新信息用于后续调整。
   - ⚠️ 严禁跨轮复述同一段分析：每一轮的思考只针对「上一条工具结果带来的新信息」给出新判断。
     如果你的分析与上一轮几乎相同，说明这条路走不通——立即换一种做法，或直接给出最终回答。
   - 禁止重复调用能返回相同信息的工具：同一个文件不要重复读取同一区间（大文件需要续读时用新的 offset）；同一个目录不要用 list_dir 列两次；已确认的依赖版本不要用 npm list 再查一次。
   - 探索遵循最小必要原则：先用 grep_search / list_files 定位，再局部 read_file；同一信息只获取一次，不要为了"了解项目"做无关的全面巡视。
   - 如果上一条工具结果已经回答了你的问题，直接进入下一步或给出最终回答，不要为了"再确认"而追加冗余调用。
   - 同一轮回复要么只发起工具调用（可带一句简短的过渡说明），要么只给出最终回答；
     不要一边输出大段分析、一边又发起工具调用——有工具调用说明任务还没做完，过程分析只会干扰阅读。
   - dev server 等长驻命令（npm run dev、uvicorn 等）必须用 terminal 工具的 background=true 启动，使服务器持续在后台运行；随后用 process 工具查看日志（action="log"）确认启动成功，需要时用 process(action="stop") 停止。禁止前台直跑长驻命令（会被超时终止，服务器起不来）。

8. 服务与端口管理（硬约束）
   - 仅在确实要启动长驻服务（dev server、uvicorn 等）时才处理端口；不启动服务时禁止做任何端口探测。
   - 启动前最多探测一次目标端口（单条命令即可）。端口空闲 → 直接以 background=true 启动。
   - 端口被占用 → 禁止终止任何已有进程；直接从默认端口+1 尝试下一个端口，启动成功后无需再探测其他端口。
   - dev server 一律用 background=true 启动；随后用 process(action="log") 查看启动日志，确认监听地址与实际端口。
   - 同一个服务只启动一次；需要重启时先用 process(action="stop") 停止自己启动的 PID，再启动。
   - ⚠️ 绝对禁止：启动新服务时自动关闭或终止任何已在运行的现有服务进程（包括用户自己启动的、其他 Agent 启动的、或先前 run 遗留的）。
   - 用户明确指定固定端口时，必须严格遵守；仅当该端口已被占用且用户未授权释放时，才按上述递增策略迁移，并向用户说明原因。

9. 工具选择（优先专用工具，少写一次性脚本）
   - 批量跨文件替换用 replace_in_files（默认 dry_run 预览），确认后再 dry_run=false；不要写脚本遍历改文件。
   - 改 JSON 中某个字段用 json_edit（JSON Pointer 定位），不要整文件重写，避免打乱键顺序与格式。
   - 大范围改动前先 workspace_checkpoint(action="create") 建快照；改坏了用 action="restore" 整体回滚。
   - 查“符号在哪里被引用”“这个文件被谁依赖”用 find_references / find_dependents，不要盲目 grep。
   - 一次要读多个文件用 read_files；跨多文件多处修改用 apply_patch（可先 dry_run=true 预览）。
   - 验证几行代码用 run_code，不要用 terminal 拼 python -c / node -e（引号转义易错）。
   - 跑测试用 run_tests（可只跑相关文件或关键字）；需要构建 / lint 时再用 run_checks。
   - 启动 dev server 后用 wait_for_server 等就绪；端口占用用 list_ports 查，需要进程详情用 port_info(port)（只读），禁止杀无关进程。
   - 交付前可用 secret_scan 自查是否硬编码密钥；文件乱码用 check_encoding 检查（fix=true 才转换）。
   - 需要用户做选择时用 ask_user(question, options)（options 用换行/逗号分隔，留空让用户自由填写），
     一次只问一个最关键的问题；用户取消或超时未答时不要假设答案，先做能确定的部分。

以上规则必须遵守。现在开始处理用户任务。
"""


def _current_tool_names(tools: list[BaseTool]) -> str:
    return ", ".join(t.name for t in tools)


_CREATION_HINT_RE = re.compile(
    r"开发|创建|新建|搭建|做一个|写一个|实现一个|完成一个|设计一个|生成一个|"
    r"编写|重构|修复|添加|新增|改造|升级|迁移|优化一下|加一个",
    re.IGNORECASE,
)
# 视为"实际执行"的工具：只读工具与 write_todos 不算（防止模型只看不写就宣称完成）
_PRODUCTIVE_TOOLS = {
    "write_file", "edit_file", "apply_patch", "delete_file", "move_file", "copy_file", "terminal",
}
# 会改动工作区的工具（落盘纪律检测用）
_FILE_WRITE_TOOLS = {
    "write_file", "edit_file", "apply_patch", "delete_file", "move_file", "copy_file",
}


def _tool_mutates(name: str, args: dict) -> bool:
    """判断本次工具调用是否真的改动了工作区（dry_run / get / 未 fix 不算）。"""
    if name in _FILE_WRITE_TOOLS:
        if name == "apply_patch":
            return not args.get("dry_run", False)
        return True
    if name == "replace_in_files":
        return not args.get("dry_run", True)
    if name == "json_edit":
        return str(args.get("action") or "get").lower() != "get"
    if name == "check_encoding":
        return bool(args.get("fix"))
    return False


def _system_message(state: AgentState) -> SystemMessage:
    """组装系统提示：基础规则 + 工作区 + 工具 + 项目规则 + 长期记忆。"""
    workspace = state.get("workspace", "")
    parts = [
        SYSTEM_PROMPT,
        f"当前工作区: {workspace}",
        f"可用工具: {_current_tool_names(state.get('_tools', []))}",
        TODOS_PROMPT,
    ]
    try:
        extra = workspace_context(workspace)
    except Exception:
        extra = ""
    if extra:
        parts.append(extra)
    return SystemMessage(content="\n\n".join(parts))


def _limit_chars() -> int:
    try:
        return max(1000, int(get_effective("tool_output_max_chars", 12000)))
    except Exception:
        return 12000


# 读取类工具：模型侧、缓存、卡片展示都完整返回，不设上限；
# 其余工具仍按 tool_output_max_chars 截断，避免日志类输出撑爆上下文。
_READ_TOOLS = {"read_file", "read_files"}


def _truncate_result(text: str) -> str:
    """工具结果统一截断：头尾保留，避免直接砍掉错误栈尾部。"""
    limit = _limit_chars()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [工具输出已截断，共 {len(text)} 字符] ...\n\n{tail}"


def _path_key(workspace: str, raw: str) -> str:
    """把工具路径参数规范化为稳定的绝对键，供重复读取检测。"""
    if not raw:
        return ""
    try:
        p = raw
        if not os.path.isabs(p):
            p = os.path.join(workspace, p)
        return os.path.normcase(os.path.realpath(p))
    except Exception:
        return raw


def _read_key(workspace: str, tool_name: str, args: dict) -> str:
    """重复读取检测键：read_file 带 offset/limit，允许分段读取同一文件。"""
    raw = args.get("path") or args.get("directory") or args.get("dir")
    if not raw:
        return ""
    base = _path_key(workspace, str(raw))
    if not base:
        return ""
    if tool_name == "read_file":
        return f"{base}|{args.get('offset', 1)}|{args.get('limit', 0)}"
    return base


def _reasoning_delta(chunk: Any) -> str:
    """从流式 chunk 中提取推理内容（deepseek-reasoner 的 reasoning_content）。"""
    raw = getattr(chunk, "reasoning_content", None)
    if not raw:
        try:
            raw = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
        except Exception:
            raw = None
    return raw if isinstance(raw, str) else ""


def _exception_brief(exc: BaseException) -> str:
    """把异常转成非空可诊断文本：类型名 + 消息/响应体 + HTTP 状态/请求 ID/底层原因链。

    部分 SDK 的流式异常（如 httpx.RemoteProtocolError）str() 为空，直接拼进
    state["error"] 会让用户只看到「调用模型失败: 」，无法定位问题。
    openai 的 APIConnectionError 也只会说 "Connection error."，真正原因
    （DNS / TLS / 连接被重置）藏在 __cause__ 里，必须一并带出才能判断是本机
    网络问题还是上游断流。
    """
    name = type(exc).__name__
    detail = str(exc).strip()
    if not detail:
        detail = repr(exc)
        prefix = f"{name}("
        if detail.startswith(prefix) and detail.endswith(")"):
            detail = detail[len(prefix) : -1].strip()
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(response, "status_code", None)
    body = getattr(response, "text", None)
    if isinstance(body, str) and body.strip():
        detail = f"{detail} | body={body.strip()[:300]}"
    if status:
        detail = f"HTTP {status} {detail}".strip()
    request_id = getattr(exc, "request_id", None)
    if not request_id:
        headers = getattr(response, "headers", None)
        try:
            request_id = headers.get("x-request-id") if headers else None
        except Exception:
            request_id = None
    if request_id:
        detail = f"{detail} | request_id={request_id}"
    chain: list[str] = []
    cause = exc.__cause__ or exc.__context__
    seen_ids = {id(exc)}
    while cause is not None and id(cause) not in seen_ids and len(chain) < 2:
        seen_ids.add(id(cause))
        cname = type(cause).__name__
        cmsg = str(cause).strip()
        if cmsg:
            chain.append(cmsg if cmsg.startswith(cname) else f"{cname}: {cmsg}")
        else:
            chain.append(repr(cause))
        cause = cause.__cause__ or cause.__context__
    if chain:
        joined = " ← ".join(chain)
        detail = f"{detail} | caused by {joined}" if detail else joined
    if not detail:
        return f"{name}（无错误信息）"
    if detail.startswith(name):
        return detail
    return f"{name}: {detail}"


# ----- reasoning 出口级"复述用户提问"兜底过滤 -----
# 即使模型忽略 prompt 纪律，也尽量在输出到前端前把"用户 xxx"引导句和整段
# 复制粘贴用户原始提问的内容剥掉，减少思考区反复出现用户问题的观感。

# 典型引导短语：「用户问我/用户让我/用户的需求是/用户想要我做/根据用户的问题…/用户的任务是」
# 匹配到行首且长度 < 80 字时，整行剥除（保留后续的真实推理）。
_LEADING_REFERRAL_RE = re.compile(
    r"^[ \t\u3000]*(?:用户|需求|任务)[^\n\r\u2028\u2029]{0,80}?[：:是要为的给了提说问想让叫](?:[^\n\r\u2028\u2029]{0,120}?)"
    r"(?:$|[\n\r\u2028\u2029]+)",
    re.MULTILINE,
)

# 流式首块常常只有「用」「用户」两个字，此时正则还匹配不出完整的复述句。
# 若这种行首片段被立即下发并清空缓冲，后半句就再也没有可识别的行首，
# 「用户想要…」会被原样漏到前端 —— 因此疑似行首必须等整行到齐再判定。
_REFERRAL_LEAD_CHARS = frozenset("用需任")
_REFERRAL_PREFIXES = ("用户", "需求", "任务")
_REFERRAL_HOLD_MAX = 240  # 超过这个长度仍未换行，就不再等待（复述句不会这么长）


def _maybe_referral_partial(buf: str) -> bool:
    """缓冲区是否可能是「用户/需求/任务 xxx」复述句的前半截（需暂存）。"""
    s = buf.lstrip(" \t\u3000")
    if not s:
        return True
    if len(s) == 1:
        return s in _REFERRAL_LEAD_CHARS
    return s.startswith(_REFERRAL_PREFIXES) or _LEADING_REFERRAL_RE.match(buf) is not None


# 部分模型会把思考结束标签写进 reasoning/正文（如 </think>、</anttThinking>），
# 直接展示会给用户看到协议碎片，统一剥除
_STRAY_THINK_TAG_RE = re.compile(
    r"</?(?:think|thinking|antthinking|anttthinking|antml:thinking)\s*>",
    re.IGNORECASE,
)


def _strip_referral_prefixes(text: str) -> str:
    """去除 reasoning 开头/段落首出现的"用户 xxx …"复述引导句。"""
    if not text:
        return text
    # 最多连清 5 次循环出现的"用户 xxx … 用户 xxx …"套话
    prev = None
    for _ in range(5):
        if prev == text:
            break
        prev = text
        text = _LEADING_REFERRAL_RE.sub("", text)
        text = text.lstrip(" \t\r\n\u200b\u3000")
    return text


def _overlap_with_request(text: str, request: str) -> str:
    """
    兜底：如果 reasoning 的连续 ≥28 个字符与用户原始提问完全一致，
    就把这部分重抄内容截掉（从命中位置起至结束不再 emit，避免整段原封不动搬运）。
    阈值 28 字：避免"好的/我需要先检查"这类真的巧合被误伤。
    """
    if not text or not request:
        return text
    req = request.strip()
    if len(req) < 16:
        return text
    overlap_len = 28
    stripped: list[str] = []
    i = 0
    n = len(text)
    m = len(req)
    while i < n:
        if n - i < overlap_len:
            stripped.append(text[i:])
            break
        window = text[i : i + overlap_len]
        r = req.find(window)
        found = -1
        best_len = 0
        if r >= 0:
            p, q = i + overlap_len, r + overlap_len
            while p < n and q < m and text[p] == req[q]:
                p += 1
                q += 1
            pl, ql = i, r
            while pl > 0 and ql > 0 and text[pl - 1] == req[ql - 1]:
                pl -= 1
                ql -= 1
            matched_len = p - pl
            if matched_len >= overlap_len:
                found = pl
                best_len = matched_len
        if found >= 0 and best_len >= overlap_len:
            i = found + best_len
            continue
        stripped.append(text[i])
        i += 1
    return "".join(stripped)


def _clean_reasoning(raw: str, user_request: str | None) -> str:
    """对 reasoning 累积文本做统一清洗：去残留思考标签 + 去「用户 xxx」引导句
    + 去整段重抄用户提问。"""
    if not raw:
        return ""
    text = _STRAY_THINK_TAG_RE.sub("", raw)
    text = _strip_referral_prefixes(text)
    if user_request:
        text = _overlap_with_request(text, user_request)
    return text


class ReasoningAcc:
    """流式 reasoning 清洗累加器（增量、O(n)）。

    旧实现每个 delta 都对累计全文重跑正则与长文本重叠检测，长思考时是 O(n²)，
    且会随流式 chunk 数量放大 CPU。这里改为：
    - 完整行：立即清洗并下发
    - 未完成行：若像「用户问我…」引导句则暂存，否则立即下发（不扣留尾巴）
    - 流结束时 flush 残余
    - 同时累计清洗后的全文（text），供跨轮思考重复检测使用
    """

    __slots__ = ("_buf", "_request", "_text")

    def __init__(self, user_request: str | None):
        self._request = user_request or None
        self._buf = ""
        self._text = ""

    @property
    def text(self) -> str:
        """本轮已下发的完整思考文本（清洗后）。"""
        return self._text

    async def push(self, delta: str, send) -> None:  # send: async (delta: str) -> Awaitable[Any]
        if not delta:
            return
        self._buf += delta
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = _clean_reasoning(line, self._request)
            if cleaned:
                self._text += cleaned + "\n"
                await send(cleaned + "\n")
        buf = self._buf
        if not buf:
            return
        # 行首疑似「用户问我…」：等这一行完整后再判断，避免半句泄漏。
        # 判定必须覆盖只有 1-2 个字的行首片段（如首块只来了「用户」），
        # 否则片段会被原样下发，缓冲清空后后半句再也无法清洗。
        if len(buf) < _REFERRAL_HOLD_MAX and _maybe_referral_partial(buf):
            return
        # 其余内容立即下发。旧实现固定扣留末尾 24 字到流结束才 flush，
        # 当模型随后生成大段工具参数（如 write_file 的 content）时，
        # 短思考会长时间停在倒数第 24 个字，看起来像被截断。
        cleaned = _clean_reasoning(buf, self._request)
        self._buf = ""
        if cleaned:
            self._text += cleaned
            await send(cleaned)

    async def flush(self, send) -> None:
        """流结束时把残余文本清洗后下发（避免最后一行丢失）。"""
        if not self._buf:
            return
        cleaned = _clean_reasoning(self._buf, self._request)
        self._buf = ""
        if cleaned:
            self._text += cleaned
            await send(cleaned)



async def _emit(state: AgentState, event: dict[str, Any]) -> None:
    emit = state.get("emit")
    if emit:
        try:
            await emit(event)
        except Exception as e:
            logger.warning("emit 失败: %s", e)


# ---------------------------------------------------------------------------
# todos ↔ plan 转换（TodoListMiddleware 状态 schema → 前端 plan 事件）
# ---------------------------------------------------------------------------

def normalize_todos(raw: Any) -> list[dict[str, str]]:
    """把模型 write_todos 参数规范化为中间件状态 schema：[{content, status}]。

    - 过滤空内容项
    - 非法 status 回退为 pending
    - 纯字符串项视为 pending
    """
    todos: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return todos
    for item in raw:
        if isinstance(item, dict):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending"))
            if content:
                todos.append({
                    "content": content,
                    "status": status if status in _VALID_TODO_STATUS else "pending",
                })
        elif isinstance(item, str) and item.strip():
            todos.append({"content": item.strip(), "status": "pending"})
    return todos


def todos_to_steps(todos: list[dict[str, str]]) -> list[str]:
    """todos → 前端 plan 事件的 steps（保持纯文本步骤列表，由前端渲染序号）。"""
    return [t["content"] for t in todos]


def apply_todos(state: AgentState, raw: Any) -> list[dict[str, str]]:
    """规范化并写入 state（todos/plan），返回规范化后的 todos。"""
    todos = normalize_todos(raw)
    state["todos"] = todos
    state["plan"] = todos_to_steps(todos)
    return todos


def _extract_raw_write_todos(text: str) -> list[dict[str, str]] | None:
    """解析模型把 write_todos 当纯文本输出的情况（未返回结构化 tool_calls）。

    部分模型（如 Qwen）只把 ``write_todos(todos=[...])`` 写成正文文本而不是
    工具调用。这里用 ast.literal_eval 安全解析，解析失败或不是完整调用则返回 None。
    """
    body = (text or "").strip()
    if not body:
        return None
    fenced = re.fullmatch(r"```(?:python|json)?\s*(.*?)```", body, flags=re.S)
    if fenced:
        body = fenced.group(1).strip()
    if not body.startswith(f"{WRITE_TODOS.name}("):
        return None
    try:
        expr = ast.parse(body, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Call):
        return None
    func_name = (
        expr.func.id if isinstance(expr.func, ast.Name)
        else expr.func.attr if isinstance(expr.func, ast.Attribute)
        else ""
    )
    if func_name != WRITE_TODOS.name:
        return None
    arg_node = next((kw.value for kw in expr.keywords if kw.arg == "todos"), None)
    if arg_node is None and expr.args:
        arg_node = expr.args[0]
    if arg_node is None:
        return None
    try:
        raw = ast.literal_eval(arg_node)
    except (ValueError, SyntaxError):
        return None
    todos = normalize_todos(raw)
    return todos or None


# ---------------------------------------------------------------------------
# 工具执行层：并行分组 / 超时治理 / 校验自纠 / 结果缓存（内存 + 会话级 DB）
# ---------------------------------------------------------------------------

def _tool_calls_signature(tool_calls: list[dict]) -> str:
    """仅按工具名+参数判定重复；tool_call id 每轮由模型新生成，不能参与比较。"""
    payload = [
        {"name": tc.get("name", ""), "args": tc.get("args") or {}}
        for tc in tool_calls
    ]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# 跨轮思考重复检测：清洗后的思考 ≥120 字且与上一轮字符 4-gram Jaccard ≥0.8，
# 视为「几乎一模一样的复述」（reasoning 不进历史，模型每轮会从同样的证据重新推导）
_REASONING_REPEAT_MIN_CHARS = 120
_REASONING_REPEAT_SIMILARITY = 0.8


def _reasoning_shingles(text: str, n: int = 4) -> frozenset[int]:
    """压缩空白后取字符 n-gram 的 CRC 指纹集合，作为思考文本的相似度签名。"""
    norm = re.sub(r"\s+", "", text or "")
    if len(norm) < n:
        return frozenset()
    return frozenset(
        zlib.crc32(norm[i:i + n].encode("utf-8")) for i in range(len(norm) - n + 1)
    )


def _reasoning_similarity(a: frozenset[int], b: frozenset[int]) -> float:
    """两个 n-gram 集合的 Jaccard 相似度；任一为空视为不相似。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# 展示级去重：模型每轮都会从同样的工具结果重新推导，短思考常常整段复写上一轮。
# 短思考先暂存，等本轮正文/工具调用开始或缓冲够长再决定是否展示；
# 与上一轮高度相似时整轮不展示（推理仍完整参与后续护栏与衔接摘要）。
_THINK_GATE_FLUSH_CHARS = 240
_THINK_DEDUPE_MIN_CHARS = 20
_THINK_DEDUPE_SIMILARITY = 0.85


def _is_duplicate_reasoning(text: str, prev: str) -> bool:
    a = (text or "").strip()
    b = (prev or "").strip()
    if len(a) < _THINK_DEDUPE_MIN_CHARS or len(b) < _THINK_DEDUPE_MIN_CHARS:
        return False
    return (
        _reasoning_similarity(_reasoning_shingles(a), _reasoning_shingles(b))
        >= _THINK_DEDUPE_SIMILARITY
    )


# 正文回显抑制：中间轮次常把之前对用户说过的旁白逐字再说一遍（未必紧邻上一轮）。
# 本轮正文仍是最近几轮正文的前缀时先暂存（最长 400 字），一旦偏离就立即恢复流式；
# 轮次结束若与其中任一轮逐字相同且后面还有工具调用，就不再重复展示。
_ECHO_MAX_CHARS = 400
_RECENT_TEXT_WINDOW = 4  # 回显对比窗口：最近几轮正文


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


# 单轮思考"原地复读"检测：一条流里反复复写前面的段落时及时中断。
# 不能只看长度——复杂任务正常思考 4000 字并不算多，按长度硬切会把
# 有意义的推理掐死在半路（截断后 reasoning 又不会进入下一轮上下文，
# 模型只能从零重推，于是思考区反复出现同一段分析）。这里用尾部窗口的
# 4-gram 命中率判断是不是在复读，另保留一个很高的绝对上限兜底。
_REASONING_SELF_REPEAT_MIN_CHARS = 2400
_REASONING_SELF_REPEAT_WINDOW = 800
_REASONING_SELF_REPEAT_STEP = 800
_REASONING_SELF_REPEAT_THRESHOLD = 0.92
_REASONING_ABS_LIMIT_CHARS = 12000


def _reasoning_self_repeat(text: str) -> bool:
    """尾部 800 字的 8-gram 有 ≥92% 已出现在更早的思考里 → 判定为原地复读。"""
    norm = re.sub(r"\s+", "", text or "")
    if len(norm) < _REASONING_SELF_REPEAT_MIN_CHARS:
        return False
    body = norm[-7000:-_REASONING_SELF_REPEAT_WINDOW]
    tail = norm[-_REASONING_SELF_REPEAT_WINDOW:]
    if not body:
        return False
    tail_shingles = _reasoning_shingles(tail, n=8)
    if not tail_shingles:
        return False
    body_shingles = _reasoning_shingles(body, n=8)
    return (
        len(tail_shingles & body_shingles) / len(tail_shingles)
        >= _REASONING_SELF_REPEAT_THRESHOLD
    )


# 上一轮思考回填：reasoning 不进消息历史（OpenAI 兼容端点也不接受
# reasoning_content 回传），模型每轮都会从同一份证据重新推导。把上一轮
# 思考压成系统侧衔接摘要回填给本轮，模型才知道自己已经得出了什么结论。
_REASONING_BRIDGE_MAX_CHARS = 800
_REASONING_BRIDGE_MIN_CHARS = 80


def _reasoning_bridge(text: str) -> str:
    """把上一轮思考压成可回填的摘要：保留开头结论与结尾行动意图。"""
    norm = re.sub(r"[ \t]+", " ", (text or "").strip())
    norm = re.sub(r"\n{2,}", "\n", norm)
    if len(norm) < _REASONING_BRIDGE_MIN_CHARS:
        return ""
    if len(norm) <= _REASONING_BRIDGE_MAX_CHARS:
        return norm
    head_len = _REASONING_BRIDGE_MAX_CHARS // 3
    tail_len = _REASONING_BRIDGE_MAX_CHARS - head_len
    return f"{norm[:head_len].rstrip()}\n……（中间省略）……\n{norm[-tail_len:].lstrip()}"


def _public_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """剥离内部标记（_invalid_args 等），只保留 OpenAI 协议要求的字段。"""
    return [
        {"id": tc.get("id") or "", "name": tc.get("name") or "", "args": tc.get("args") or {}}
        for tc in tool_calls
    ]


def _group_real_calls(
    calls: list[tuple[int, dict]],
) -> list[list[tuple[int, dict]]]:
    """把真实工具调用切成连续执行段：连续幂等调用并为一段（并行执行），
    非幂等调用单独成段（串行）。保证"读不越过写、写不越过读"的顺序语义。"""
    groups: list[list[tuple[int, dict]]] = []
    last_parallel = False
    for item in calls:
        idem = spec_of(item[1]["name"]).idempotent
        if idem and groups and last_parallel:
            groups[-1].append(item)
        else:
            groups.append([item])
        last_parallel = idem
    return groups


def _format_validation_error(tool: BaseTool, exc: ValidationError) -> str:
    """把 Pydantic 校验错误转成模型可自纠的结构化反馈（字段级定位 + 参数清单）。"""
    lines = []
    empty_input = False
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", []))
        received = str(err.get("input"))[:80]
        if isinstance(err.get("input"), dict) and not err["input"]:
            empty_input = True
        lines.append(f"- 参数 {loc or '?'}: {err.get('msg', 'invalid')}（收到: {received}）")
    schema_hint = ""
    try:
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        parts = []
        for k, v in props.items():
            desc = " ".join(str(v.get("description") or "").split())[:60]
            mark = "必填" if k in required else "可选"
            parts.append(f"{k}({v.get('type', 'any')}，{mark}{'，' + desc if desc else ''})")
        schema_hint = "；".join(parts)
    except Exception:
        pass
    head = (
        f"【参数缺失】工具 {tool.name} 这次没有传任何参数（收到 {{}}）。\n"
        if empty_input
        else f"【参数校验失败】工具 {tool.name} 的参数不符合要求：\n"
    )
    return (
        head
        + "\n".join(lines)
        + (f"\n该工具的参数列表: {schema_hint}" if schema_hint else "")
        + "\n请修正参数后重新调用该工具。"
    )


class ToolResultCache:
    """run 级内存缓存 + 会话级 DB 缓存（仅幂等工具结果持久化）。

    - 命中：内存 → DB（限 TTL），命中后跳过真实执行
    - 写入：成功结果写内存；幂等工具额外写 DB 供跨 run 复用
    - 写失效：任何非幂等工具成功后清空（内存 + DB），防止脏读
    - 键包含工作区与文件 mtime/size，避免外部修改后的脏读
    """

    def __init__(self, session_id: str, workspace: str = ""):
        self._mem: dict[str, str] = {}
        self._session_id = session_id
        self._workspace = workspace
        self._db: Any = None
        # 并行幂等工具共用同一个 AsyncSession，必须串行化 DB 访问
        self._db_lock = asyncio.Lock()

    async def _get_db(self) -> Any:
        if self._db is None:
            from ..database import SessionLocal
            self._db = SessionLocal()
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                pass
            self._db = None

    async def get(self, name: str, args: dict, idempotent: bool) -> str | None:
        key = tool_cache_store.make_cache_key(name, args, self._workspace)
        if key in self._mem:
            return self._mem[key]
        if idempotent:
            async with self._db_lock:
                db = await self._get_db()
                hit = await tool_cache_store.get_cached(
                    self._session_id, name, args, self._workspace, db=db
                )
            if hit is not None:
                self._mem[key] = hit
                return hit
        return None

    async def put(self, name: str, args: dict, result: str, idempotent: bool) -> None:
        if not idempotent:
            return
        self._mem[tool_cache_store.make_cache_key(name, args, self._workspace)] = result
        async with self._db_lock:
            db = await self._get_db()
            await tool_cache_store.put_cached(
                self._session_id, name, args, result, self._workspace, db=db
            )

    async def invalidate(self) -> None:
        self._mem.clear()
        async with self._db_lock:
            db = await self._get_db()
            await tool_cache_store.invalidate_session(self._session_id, db=db)


def _cache_payload(text: str, limit: int | None = None) -> str:
    """缓存结果上限：防止把超大工具输出写进 DB；limit<=0 表示不截断。"""
    if limit is None:
        try:
            limit = max(2000, int(get_effective("tool_cache_max_chars", 20000)))
        except Exception:
            limit = 20000
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [缓存结果已截断，原长 {len(text)} 字符] ..."


async def _run_one_tool(
    state: AgentState,
    tc: dict,
    tool: BaseTool | None,
    cache: ToolResultCache,
) -> ToolMessage:
    """执行单个真实工具调用：缓存 → 超时治理 → 执行 → 结果截断/缓存/写失效。

    事件（tool_start/tool_end）在本函数内发射，天然支持并行段的事件交错；
    status 取值保持 success/error（前端兼容），校验失败与超时以 content 前缀区分。
    """
    spec = spec_of(tc["name"])
    if tool is None:
        return ToolMessage(content=f"错误：未知工具 {tc['name']}", tool_call_id=tc["id"])

    # 参数是半截 JSON（流被停止/截断）：不执行，回一条可自纠的错误，避免写入残缺内容
    if tc.get("_invalid_args"):
        return ToolMessage(
            content=(
                f"【参数解析失败】工具 {tc['name']} 这次调用的参数不是合法 JSON，已取消执行。\n"
                f"收到（截断）: {tc['_invalid_args']}\n"
                "请重新调用该工具，并提供完整、合法的 JSON 参数。"
            ),
            tool_call_id=tc["id"], name=tc["name"],
        )

    # 去重缓存：同一工具+同一参数/同一文件状态 → 复用上一次结果（跳过真实执行）
    cacheable = bool(spec.idempotent and spec.cacheable)
    cached = await cache.get(tc["name"], tc["args"], cacheable)
    if cached is not None:
        hint = f"【重复调用拦截】该工具+参数组合已在之前执行过，直接复用上次结果（{len(cached)} 字符）。"
        logger.warning(
            "[run:%s] 拦截重复工具调用: %s args=%s（复用缓存）",
            state.get("run_id", ""), tc["name"],
            json.dumps(tc["args"], ensure_ascii=False)[:200],
        )
        return ToolMessage(
            content=f"{hint}\n\n--- 上次执行结果 ---\n{cached}",
            tool_call_id=tc["id"], name=tc["name"],
        )

    started = time.time()
    started_at = time.strftime("%H:%M:%S")
    logger.info(
        "[run:%s] 调用工具 %s 于 %s args=%s",
        state.get("run_id", ""), tc["name"], started_at,
        json.dumps(tc["args"], ensure_ascii=False)[:300],
    )
    await _emit(state, {
        "type": "tool_start", "tool": tc["name"], "args": tc["args"], "call_id": tc["id"],
        "started_at": started_at,
    })
    token = current_call_id.set(tc["id"])
    try:
        # 超时治理：spec.timeout<=0 表示工具自管理超时（如 terminal），外层不截断
        if spec.timeout and spec.timeout > 0:
            result = await asyncio.wait_for(tool.ainvoke(tc["args"]), timeout=spec.timeout)
        else:
            result = await tool.ainvoke(tc["args"])
        status = "success"
    except asyncio.CancelledError:
        await _emit(state, {"type": "status", "status": "stopped"})
        raise
    except TimeoutError:
        result = (
            f"工具执行超时（超过 {spec.timeout:.0f}s），已被强制终止。"
            "请调整策略后重试（如缩小范围、简化参数）。"
        )
        status = "error"
    except ValidationError as e:
        result = _format_validation_error(tool, e)
        status = "error"
    except Exception as e:
        result = f"工具执行出错: {e}"
        status = "error"
    finally:
        current_call_id.reset(token)
    duration_ms = int((time.time() - started) * 1000)
    result_str = str(result)
    read_tool = tc["name"] in _READ_TOOLS

    # 成功结果写入缓存（错误不缓存，避免屏蔽偶发重试）；
    # 非幂等工具（写操作）成功后使会话缓存失效，防止脏读
    if status == "success":
        cache_limit = 0 if read_tool else None
        await cache.put(tc["name"], tc["args"], _cache_payload(result_str, cache_limit), cacheable)
        if not spec.idempotent:
            await cache.invalidate()

    logger.info(
        "[run:%s] 工具 %s 完成: status=%s 耗时=%dms",
        state.get("run_id", ""), tc["name"], status, duration_ms,
    )
    # 读取类工具：卡片与模型都拿到完整内容；其余工具卡片只回传 4000 字符预览
    display_result = result_str if read_tool else result_str[:4000]
    await _emit(state, {
        "type": "tool_end", "tool": tc["name"], "call_id": tc["id"],
        "result": display_result, "status": status, "duration_ms": duration_ms,
        "started_at": started_at,
    })
    return ToolMessage(
        content=result_str if read_tool else _truncate_result(result_str),
        tool_call_id=tc["id"], name=tc["name"],
    )


class _StreamAccumulator:
    """按 index 拼装流式工具调用，替代 AIMessageChunk 的逐块相加。

    chunk 相加时 langchain 会把已累计的全部参数重新做一次 parse_partial_json，
    参数越长每次拼接越慢：实测 320 块 0.87s、640 块 3.17s、1280 块 13.7s。
    这段 CPU 跑在事件循环里，写大文件时会把所有会话的流一起卡住。
    这里只做字符串拼接，流结束后一次性解析参数。
    """

    def __init__(self) -> None:
        self._slots: dict[int, dict[str, str]] = {}
        self._complete: list[dict] = []
        self.usage_chunk: Any = None

    def add(self, chunk: Any) -> None:
        if getattr(chunk, "usage_metadata", None) is not None:
            self.usage_chunk = chunk
        # 有的端点/测试替身一次下发完整 tool_calls，没有增量分片。
        # 注意 DashScope 流式响应的第一个工具分片就带 name+id、参数为空，
        # langchain 会把它合成为一条 args={} 的"完整"调用；后续分片才带着
        # 真实参数。因此同 id 的完整调用必须允许被后到的非空参数覆盖，
        # 否则会误判为无参调用（read_file args={} → 校验失败）。
        for tc in getattr(chunk, "tool_calls", None) or []:
            if not isinstance(tc, dict) or not tc.get("name"):
                continue
            args = tc.get("args")
            call = {
                "id": tc.get("id") or "",
                "name": tc["name"],
                "args": args if isinstance(args, dict) else {},
            }
            matched = False
            if call["id"]:
                for i, old in enumerate(self._complete):
                    if old["id"] == call["id"] and old["name"] == call["name"]:
                        if call["args"] or not old["args"]:
                            self._complete[i] = call
                        matched = True
                        break
            if not matched:
                self._complete.append(call)
        for tcc in getattr(chunk, "tool_call_chunks", None) or []:
            idx = tcc.get("index")
            slot = self._slots.setdefault(
                idx if isinstance(idx, int) else 0, {"id": "", "name": "", "args": ""}
            )
            slot["id"] = slot["id"] or (tcc.get("id") or "")
            slot["name"] = slot["name"] or (tcc.get("name") or "")
            slot["args"] += tcc.get("args") or ""

    def tool_calls(self) -> list[dict]:
        """合并分片累积与端点一次性下发：同一 id 优先使用分片拼接出的参数。"""
        complete_by_id = {c["id"]: c for c in self._complete if c["id"]}
        calls: list[dict] = []
        used_ids: set[str] = set()
        for idx in sorted(self._slots):
            slot = self._slots[idx]
            if not slot["name"]:
                continue
            raw = slot["args"].strip()
            args: dict | None = None
            invalid_raw = ""
            if raw:
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    # 流被中断/截断：参数不是完整 JSON。绝不能"部分修复"后执行——
                    # write_file/edit_file 会落下残缺内容；标记非法让模型重发。
                    parsed = None
                    invalid_raw = raw[:300]
                    logger.warning(
                        "工具调用参数不完整（流可能被截断）: name=%s raw=%s",
                        slot["name"], raw[:200],
                    )
                if isinstance(parsed, dict):
                    args = parsed
            if args is None:
                fallback = None if invalid_raw else complete_by_id.get(slot["id"])
                args = fallback["args"] if fallback else {}
            call = {"id": slot["id"], "name": slot["name"], "args": args}
            if invalid_raw:
                call["_invalid_args"] = invalid_raw
                call["args"] = {}
            calls.append(call)
            if slot["id"]:
                used_ids.add(slot["id"])
        for c in self._complete:
            if c["id"] and c["id"] in used_ids:
                continue
            calls.append(dict(c))
        return calls


def _accumulate_usage(state: AgentState, response_chunk: Any) -> None:
    """从聚合响应提取 token 用量并累计到 state（模型未返回时静默跳过）。"""
    if response_chunk is None:
        return
    usage = getattr(response_chunk, "usage_metadata", None)
    if not isinstance(usage, dict):
        return
    input_t = usage.get("input_tokens") or 0
    output_t = usage.get("output_tokens") or 0
    if not input_t and not output_t:
        return
    acc = state.get("token_usage") or {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    acc["input_tokens"] += input_t
    acc["output_tokens"] += output_t
    acc["llm_calls"] += 1
    state["token_usage"] = acc


async def _make_llm(state: AgentState) -> Any:
    """创建模型；测试可通过 state['_llm'] 注入替身。

    思考强度档位（reasoning_effort）随每次请求传入，未指定时走模型默认。
    """
    override = state.get("_llm")
    if override is not None:
        return override
    return init_chat_models(
        reasoning_effort=state.get("reasoning_effort"),
        model=str(state.get("model") or ""),
    )


_CONN_ERROR_NAMES = {
    "APIConnectionError",
    "APIConnectionTimeoutError",
    "ConnectError",
    "ConnectionError",
    "ConnectTimeout",
    "ReadError",
    "WriteError",
    "RemoteProtocolError",
    "StreamChunkTimeoutError",
    "IncompleteRead",
    "BadStatusLine",
}

# 连接类错误的退避表：本机网络抖个十几秒很常见（切 Wi-Fi、DNS 抽风、代理重启），
# 只等 1s/2s 就放弃会让整轮任务白跑，所以给到 2→5→10→20 秒的容忍窗口。
_RETRY_BACKOFF = (2.0, 5.0, 10.0, 20.0)


def _is_connection_error(exc: BaseException) -> bool:
    """是否属于「重试有可能成功」的网络/连接类错误（含被 SDK 包装后的外层异常）。"""
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _CONN_ERROR_NAMES or isinstance(
            cur, (ConnectionError, socket.gaierror)
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _retry_delay(attempt: int) -> float:
    """第 attempt 次失败后的等待秒数：退避表 + ±20% 抖动，避免与上游重试节奏共振。"""
    base = _RETRY_BACKOFF[min(attempt - 1, len(_RETRY_BACKOFF) - 1)]
    return round(base * (0.8 + 0.4 * random.random()), 1)


def _retry_budget(exc: BaseException, max_attempts: int) -> int:
    """按错误类别分配重试次数。

    空闲超时意味着已经白等了整个超时窗口，再反复空等最伤体验，只给一次机会；
    连接类错误给满退避表；其余（4xx 之类重试也不会好）保持较少次数。
    """
    if max_attempts <= 1:
        return max_attempts
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or name == "StreamChunkTimeoutError":
        return min(max_attempts, 2)
    if _is_connection_error(exc):
        return max_attempts
    return min(max_attempts, 3)


async def _network_hint(state: AgentState) -> str:
    """连接类失败时探一下上游是否可达，用来区分「本机断网」和「上游中途掐流」。

    只做 DNS + TCP 握手，不发请求、不读密钥，失败也不影响主流程。
    """
    try:
        host = urlsplit(resolve_model_endpoint(str(state.get("model") or "")).get("base_url", "")).hostname
        if not host:
            return ""
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM), timeout=2.0
            )
        except Exception:
            return f"本机此刻解析不了 {host}（网络/DNS 中断）"
        writer = None
        try:
            _, writer = await asyncio.wait_for(
                loop.create_connection(lambda: asyncio.Protocol(), host, 443), timeout=3.0
            )
        except Exception:
            return f"本机此刻连不上 {host}:443（网络中断）"
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
        return f"本机到 {host} 连通正常，判定为上游中途断流"
    except Exception:
        return ""


async def _rebuild_llm_for_retry(state: AgentState) -> tuple[Any, str]:
    """连接类失败后重建模型客户端，避免复用已卡死的 HTTP 连接池。

    配置了 fallback_model 时直接切换到备用模型；否则用主模型重建一个
    全新的客户端（新的连接池）。测试注入的 _llm 替身保持原样。
    返回 (llm, 实际使用的模型名)。
    """
    override = state.get("_llm")
    if override is not None:
        return override, str(state.get("model") or "")
    primary = str(state.get("model") or "") or str(get_effective("model", ""))
    fallback = str(get_effective("fallback_model", "") or "").strip()
    if fallback and fallback != primary:
        try:
            llm = init_chat_models(
                model=fallback,
                reasoning_effort=state.get("reasoning_effort"),
                use_fallback=False,
            )
            logger.warning(
                "[run:%s] 连接异常，改用备用模型 %s 重试", state.get("run_id", ""), fallback
            )
            return llm, fallback
        except Exception as e:
            logger.warning(
                "[run:%s] 备用模型 %s 不可用，改为重建主模型: %s",
                state.get("run_id", ""), fallback, e,
            )
    llm = init_chat_models(model=primary, reasoning_effort=state.get("reasoning_effort"))
    logger.info("[run:%s] 连接异常，重建模型客户端后重试", state.get("run_id", ""))
    return llm, primary


def _blocked_tool_messages(tool_calls: list[dict], content: str) -> list[ToolMessage]:
    """本轮工具被护栏取消时，为每个 tool_call 补齐 ToolMessage。

    OpenAI 兼容协议要求 AIMessage.tool_calls 后必须紧跟对应 ToolMessage，
    否则下一轮请求会被端点直接拒绝（400），所以护栏 continue 前必须补位。
    """
    return [
        ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"])
        for tc in tool_calls
    ]


def _nudge(content: str) -> SystemMessage:
    """内部提醒：必须以系统消息下发，否则模型会把护栏提醒当成用户发言；
    持久化与摘要时按 is_nudge 过滤。"""
    return SystemMessage(content=content, additional_kwargs={"is_nudge": True})


async def execute_node(state: AgentState) -> AgentState:
    """执行节点：工具循环（ReAct），期间模型可调用 write_todos 动态更新计划。"""
    if state.get("error"):
        return state
    cache = ToolResultCache(state.get("session_id", ""), state.get("workspace", ""))
    try:
        return await _execute_node(state, cache)
    finally:
        # 取消/异常路径同样要释放缓存持有的 AsyncSession，
        # 否则反复点「停止」会把连接池耗光，之后所有 DB 操作排队超时
        with contextlib.suppress(Exception):
            await asyncio.shield(cache.close())


async def _execute_node(state: AgentState, tool_result_cache: ToolResultCache) -> AgentState:
    """执行节点主体。"""
    try:
        llm = await _make_llm(state)
    except Exception as e:
        state["error"] = _exception_brief(e)
        await _emit(state, {"type": "error", "message": state["error"]})
        return state
    tools = state["_tools"]
    messages = list(state.get("messages", []))
    rounds = 0
    active_model = str(state.get("model") or "") or str(get_effective("model", ""))

    # 每轮 run 重新注入系统提示（历史里若首条不是系统提示，如从摘要恢复的会话）
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [_system_message(state), *messages]
    await _emit(state, {"type": "status", "status": "executing"})

    # 路径追踪：防止未修改文件被重复读取（纪律"禁止重复查询"的 runtime 护栏）
    _file_read_round: dict[str, int] = {}   # path → 最后 read 的轮次
    _file_modified: set[str] = set()         # 被 write/edit 过的路径（修改后可重读）
    # 本轮（execute 阶段）是否实际执行过操作：只看本轮派发的工具调用，
    # 不能扫描全部 messages——历史里上一轮写过文件会让本轮“只说不做”被误判为已完成。
    _did_productive = False
    _did_write = False
    # 上一轮思考（仅用于下一轮系统侧衔接，不写入消息历史）
    _prev_reasoning = ""
    # 最近几轮已展示正文（规范化）：模型跨轮复述同一段旁白时用于回显抑制
    _recent_texts: list[str] = []

    # 不设工具轮数上限：模型自行决定何时结束（无工具调用即收尾），
    # 仅受停止信号与 loop_guard（重复相同工具调用）约束
    while True:
        rounds += 1
        state["tool_rounds"] = rounds
        if state.get("stopped"):
            logger.info("[run:%s] execute_node 退出: 收到停止信号 (round %d)", state.get("run_id",""), rounds)
            break
        state["stopped"] = bool(state.get("stop_event") and state["stop_event"].is_set())
        if state["stopped"]:
            await _emit(state, {"type": "status", "status": "stopped"})
            logger.info("[run:%s] execute_node 退出: stop_event 被设置 (round %d)", state.get("run_id",""), rounds)
            break

        # SummarizationMiddleware：每轮模型调用前检查上下文预算
        messages = await maybe_summarize_messages(
            messages, llm, state.get("model", ""), emit=lambda e, _s=state: _emit(_s, e)
        )

        # 流式调用模型（默认绑定真实工具 + write_todos；loop_guard 总结模式下不绑定工具）
        _no_tools = bool(state.get("_loop_summary_mode"))
        model = llm if _no_tools else llm.bind_tools([*tools, WRITE_TODOS])
        # 上一轮思考回填：只作为本轮调用的临时系统消息，不写回 messages
        #（不持久化、不参与上下文摘要），避免历史里堆满思考副本。
        call_messages = messages
        if not _no_tools:
            _bridge = _reasoning_bridge(_prev_reasoning)
            if _bridge:
                call_messages = [
                    *messages,
                    _nudge(
                        "【上一轮思考 · 系统自动记录，不是用户留言】\n"
                        f"{_bridge}\n"
                        "以上是你在上一轮已经完成的判断。本轮直接基于它继续：只补充最新工具"
                        "结果带来的新信息，不要重复这段分析；若新结果与之冲突，以新结果为准。"
                    ),
                ]
        stream_text: list[str] = []
        tool_calls_acc: list[dict] = []
        _max_attempts = 5  # 连接类错误按退避表最多试 5 次，其余类别由 _retry_budget 收窄
        _stream_failed = False
        _round_reasoning: list[str] = []  # 本轮各次尝试的完整思考文本

        for _attempt in range(1, _max_attempts + 1):
            stream_text = []
            tool_calls_acc = []
            acc = _StreamAccumulator()
            reasoning = ReasoningAcc(state.get("request"))
            _reasoning_chars = 0  # 累计思考字符数，用于检测推理死循环
            _reasoning_limit = _REASONING_ABS_LIMIT_CHARS  # 绝对上限，仅极端情况兜底
            _reasoning_cut = ""  # "self_repeat" | "too_long"：本轮流式被推理护栏中断
            _next_repeat_check = _REASONING_SELF_REPEAT_MIN_CHARS
            # Qwen 偶尔在正文之后才下发最后一句 reasoning（"所有步骤已完成，让我给出
            # 最终回答"）。正文开始后的思考先暂存：若本轮还有工具调用则补发；若本轮
            # 就是最终回答，则丢弃，避免答案下方多出一条无意义的 Think。
            _think = {"started": False, "pending": []}
            # 思考去重闸门：短思考先暂存，等正文开始/本轮结束/缓冲够长时再决定是否展示。
            # 与上一轮思考/最近几轮正文高度相似时整轮隐藏，避免思考区每轮复写同一段内容。
            _gate = {"decided": False, "shown": False, "buf": [], "chars": 0}
            _echo = {"active": bool(_recent_texts), "buf": [], "cands": tuple(_recent_texts)}

            async def _decide_gate(
                _s=state, _g=_gate, _t=_think, _prev=_prev_reasoning, _cands=_echo["cands"],
            ) -> None:
                if _g["decided"]:
                    return
                _g["decided"] = True
                text = "".join(_g["buf"]).strip()
                _g["buf"].clear()
                _g["chars"] = 0
                if not text:
                    return
                if _prev and _is_duplicate_reasoning(text, _prev):
                    logger.info(
                        "[run:%s] 本轮思考与上一轮高度重复，已隐藏 (%d chars)",
                        _s.get("run_id", ""), len(text),
                    )
                    return
                # 模型常把之前对用户说过的正文原句当成新思考再发一遍，同样隐藏
                if any(_is_duplicate_reasoning(text, c) for c in _cands):
                    logger.info(
                        "[run:%s] 本轮思考复写了最近轮次的正文，已隐藏 (%d chars)",
                        _s.get("run_id", ""), len(text),
                    )
                    return
                _g["shown"] = True
                if _t["started"]:
                    _t["pending"].append(text)
                else:
                    await _emit(_s, {"type": "thinking", "delta": text})

            async def _send_thinking(delta: str, _s=state, _t=_think, _g=_gate) -> None:
                if _g["decided"] and not _g["shown"]:
                    return  # 本轮思考已判定为重复，整轮不再展示
                if _t["started"]:
                    _t["pending"].append(delta)
                    return
                if _g["decided"]:
                    await _emit(_s, {"type": "thinking", "delta": delta})
                    return
                _g["buf"].append(delta)
                _g["chars"] += len(delta)
                if _g["chars"] >= _THINK_GATE_FLUSH_CHARS:
                    await _decide_gate()

            async def _emit_tokens(deltas, _s=state, _t=_think) -> None:
                # 正文真正展示前，先把正文后产生的思考按原顺序补发
                if _t["pending"]:
                    await _emit(_s, {"type": "thinking", "delta": "".join(_t["pending"])})
                    _t["pending"].clear()
                for d in deltas:
                    if d:
                        await _emit(_s, {"type": "token", "delta": d})

            async def _on_content(delta: str, _e=_echo) -> None:
                if not _e["active"]:
                    await _emit_tokens([delta])
                    return
                _e["buf"].append(delta)
                cur = _norm_ws("".join(_e["buf"]))
                if cur and len(cur) < _ECHO_MAX_CHARS and any(
                    c.startswith(cur) for c in _e["cands"]
                ):
                    return  # 仍可能是最近轮次正文的逐字回显，继续暂存
                _e["active"] = False
                buf, _e["buf"] = _e["buf"], []
                await _emit_tokens(buf)

            try:
                # 用 LangChain 原生流式合并工具调用，避免手动累积参数出错
                async for chunk in model.astream(call_messages):
                    # 停止信号：流式过程中也要能立刻退出，不能等本轮 LLM 输出完
                    if state.get("stop_event") and state["stop_event"].is_set():
                        state["stopped"] = True
                        break
                    delta = getattr(chunk, "content", "")
                    if isinstance(delta, str) and delta:
                        # 正文开始前先把暂存的思考定稿，保证 Think 事件仍排在正文之前
                        await _decide_gate()
                        _think["started"] = True
                        stream_text.append(delta)
                        await _on_content(delta)
                    r = _reasoning_delta(chunk)
                    if r:
                        _reasoning_chars += len(r)
                        await reasoning.push(r, _send_thinking)
                        # 正文已经开始输出时不再切断，避免把正在生成的回答截断
                        if not stream_text:
                            # 原地复读：尾部窗口几乎全部命中更早内容 → 立刻中断
                            if (
                                _reasoning_chars >= _next_repeat_check
                                and _reasoning_self_repeat(reasoning.text)
                            ):
                                _reasoning_cut = "self_repeat"
                                logger.warning(
                                    "[run:%s] 推理自重复护栏触发: reasoning=%d chars，"
                                    "中断流式输出，强制收敛到行动",
                                    state.get("run_id", ""), _reasoning_chars,
                                )
                                await _emit(state, {
                                    "type": "status",
                                    "status": "reasoning_self_repeat",
                                    "detail": "思考内容开始重复，已中断并要求直接行动",
                                })
                                break
                            if _reasoning_chars >= _next_repeat_check:
                                _next_repeat_check += _REASONING_SELF_REPEAT_STEP
                            # 绝对长度兜底：正常长推理允许，只有极端情况才触发
                            if _reasoning_chars > _reasoning_limit:
                                _reasoning_cut = "too_long"
                                logger.warning(
                                    "[run:%s] 推理长度兜底触发: reasoning=%d chars > %d, "
                                    "中断流式输出，强制收敛",
                                    state.get("run_id", ""), _reasoning_chars, _reasoning_limit,
                                )
                                await _emit(state, {
                                    "type": "status",
                                    "status": "reasoning_too_long",
                                    "detail": f"推理过程已超过 {_reasoning_limit} 字，强制收敛到行动",
                                })
                                break  # 提前终止流，用已有部分结果继续
                    # 累积增量工具调用（只拼接参数字符串，流结束后统一解析）
                    acc.add(chunk)
                _accumulate_usage(state, acc.usage_chunk)
                tool_calls_acc = acc.tool_calls()
                break
            except asyncio.CancelledError:
                await _emit(state, {"type": "status", "status": "stopped"})
                raise
            except Exception as e:
                # 先把已缓冲的思考下发，再插重试提示，保证前端顺序正确
                await reasoning.flush(_send_thinking)
                brief = _exception_brief(e)
                logger.error(
                    "[run:%s] execute_node 模型调用异常 (round %d, attempt %d/%d): %s",
                    state.get("run_id", ""), rounds, _attempt, _max_attempts, brief,
                    exc_info=True,
                )
                # 工具调用只有在流结束后才会派发，半截 tool_call 没有副作用，
                # 因此只要还没向用户输出正文就可以安全重试。
                # 重试次数按错误类别分配：空闲超时已经白等了一整个超时窗口，
                # 再反复空等最伤体验；纯连接错误给满退避表。
                attempts_allowed = _retry_budget(e, _max_attempts)
                if _attempt < attempts_allowed and not "".join(stream_text).strip():
                    delay = _retry_delay(_attempt)
                    try:
                        llm, retry_model = await _rebuild_llm_for_retry(state)
                        model = llm if _no_tools else llm.bind_tools([*tools, WRITE_TODOS])
                    except Exception as rebuild_err:
                        logger.warning(
                            "[run:%s] 重建模型失败，沿用原客户端: %s",
                            state.get("run_id", ""), rebuild_err,
                        )
                        retry_model = active_model
                    switch_note = (
                        f"，改用备用模型 {retry_model}"
                        if retry_model and retry_model != active_model
                        else ""
                    )
                    active_model = retry_model or active_model
                    hint = await _network_hint(state) if _is_connection_error(e) else ""
                    # 先把本次尝试已产生的思考定稿，避免重试提示插在思考前面
                    await _decide_gate()
                    await _emit(state, {
                        "type": "thinking",
                        "delta": f"\n\n—— 模型连接中断，{delay:.0f} 秒后自动重试"
                                 f"（第{_attempt}/{attempts_allowed}次{switch_note}"
                                 f"{'，' + hint if hint else ''}，{brief}）——\n\n",
                    })
                    await asyncio.sleep(delay)
                    continue
                # 放弃重试：把已经流给用户的半截正文写回历史，避免下一轮
                # 模型看不到自己刚说过什么（用户侧这段文字仍然保留着）
                partial_text = "".join(stream_text)
                if partial_text.strip():
                    messages.append(AIMessage(content=partial_text))
                state["error"] = f"执行阶段调用模型失败: {brief}"
                await _emit(state, {"type": "error", "message": state["error"]})
                _stream_failed = True
                break
            finally:
                await reasoning.flush(_send_thinking)
                _round_reasoning.append(reasoning.text)
        if not _stream_failed:
            # 短思考到轮次结束仍未定稿（没有正文触发）：此时再决定是否展示。
            # 必须在更新 _prev_reasoning 之前判定，比较对象是上一轮的思考。
            await _decide_gate()
        # 记录本轮思考供下一轮衔接（多尝试时取最后一次非空结果）
        _last_round_reasoning = next(
            (t for t in reversed(_round_reasoning) if t.strip()), "",
        )
        if _last_round_reasoning.strip():
            _prev_reasoning = _last_round_reasoning
        # 中间旁白的逐字回显：整轮正文与最近几轮中任一轮完全相同且后面还有工具调用时
        # 不再重复展示（消息历史仍保留这段正文，模型知道自己已经说过）
        if _echo["active"]:
            _echo["active"] = False
            cur_norm = _norm_ws("".join(_echo["buf"]))
            if cur_norm and cur_norm in _echo["cands"] and tool_calls_acc:
                logger.info(
                    "[run:%s] 本轮正文复写了最近轮次的正文，已隐藏 (round %d, %d chars)",
                    state.get("run_id", ""), rounds, len(cur_norm),
                )
            else:
                for d in _echo["buf"]:
                    if d:
                        await _emit(state, {"type": "token", "delta": d})
            _echo["buf"] = []
        if not _stream_failed and _think["pending"]:
            if tool_calls_acc:
                # 正文后还有工具调用：思考仍属于本轮过程，补发在工具行之前
                await _emit(state, {
                    "type": "thinking", "delta": "".join(_think["pending"]),
                })
            else:
                logger.info(
                    "[run:%s] 丢弃正文后的收尾思考 %d 字（已给出最终回答）",
                    state.get("run_id", ""), sum(len(x) for x in _think["pending"]),
                )
            _think["pending"].clear()
        if _stream_failed:
            break

        # 流式中途收到停止：保留已产生的文本，不继续执行工具
        if state.get("stopped") or (state.get("stop_event") and state["stop_event"].is_set()):
            state["stopped"] = True
            text = "".join(stream_text)
            if text:
                messages.append(AIMessage(content=text))
            await _emit(state, {"type": "status", "status": "stopped"})
            break

        text = "".join(stream_text)
        if text.strip():
            _norm = _norm_ws(text)
            if _norm and _norm not in _recent_texts:
                _recent_texts.append(_norm)
                if len(_recent_texts) > _RECENT_TEXT_WINDOW:
                    _recent_texts.pop(0)
        tool_calls = tool_calls_acc
        # 模型偶尔把 write_todos 当纯文本输出：解析为计划更新后继续循环，
        # 避免把工具调用文本当成最终回答（最多兜底 2 次，防止原地循环）
        if not tool_calls:
            raw_todos = _extract_raw_write_todos(text)
            _raw_count = int(state.get("_raw_plan_count", 0))
            if raw_todos and _raw_count < 2:
                state["_raw_plan_count"] = _raw_count + 1
                call_id = f"raw-plan-{uuid.uuid4().hex[:12]}"
                messages.append(AIMessage(content="", tool_calls=[{
                    "id": call_id, "name": WRITE_TODOS.name,
                    "args": {"todos": raw_todos},
                }]))
                todos = apply_todos(state, raw_todos)
                messages.append(ToolMessage(
                    content=f"Updated todo list to {todos}",
                    tool_call_id=call_id, name=WRITE_TODOS.name,
                ))
                await _emit(state, {"type": "plan", "steps": state["plan"], "todos": todos})
                logger.warning(
                    "[run:%s] 执行阶段模型把 write_todos 输出为纯文本，已解析为计划并继续 (round %d)",
                    state.get("run_id", ""), rounds,
                )
                continue
        if text or tool_calls:
            messages.append(AIMessage(
                content=text or "",
                tool_calls=_public_tool_calls(tool_calls) if tool_calls else [],
            ))

        # 推理死循环护栏（content 版）：某些模型把思考放在 content 而非 reasoning 字段里。
        # 如果 text 很长（说明写了大量思考内容）但 tool_calls 为空（没有行动） → 强制收敛。
        # 总结模式下模型只输出最终回答，不再用"思考过长"类护栏打断（否则会反复收敛循环）
        _summary_mode = bool(state.get("_loop_summary_mode"))
        _text_too_long = len(text) > 2000 and not tool_calls and not _summary_mode
        # 推理死循环护栏（reasoning 版）：流式阶段已被中断（原地复读/超长）且无 tool_calls
        _reasoning_guard_triggered = bool(_reasoning_cut) and not tool_calls and not _summary_mode
        if _text_too_long or _reasoning_guard_triggered:
            if _text_too_long:
                reason = "content 过长且无工具调用"
            elif _reasoning_cut == "self_repeat":
                reason = "思考原地重复且无工具调用"
            else:
                reason = "reasoning 过长且无工具调用"
            _converge = int(state.get("_converge_nudges", 0)) + 1
            state["_converge_nudges"] = _converge
            if _converge >= 2:
                if text.strip():
                    # 两次超长且仍无工具调用：按最终回答收尾，避免把长回答反复丢弃造成死循环
                    logger.warning(
                        "[run:%s] 推理死循环护栏第二次触发，按最终回答收尾: %s, text=%d, reasoning=%d",
                        state.get("run_id", ""), reason, len(text), _reasoning_chars,
                    )
                    await _emit(state, {
                        "type": "status",
                        "status": "reasoning_loop_blocked",
                        "detail": f"推理死循环拦截: {reason}，按最终回答收尾",
                    })
                    break
                state["_loop_summary_mode"] = True
                state["_loop_guard_stage"] = 2
                messages.append(_nudge(
                    "【系统自动保护 · 不是用户留言】你连续多轮只输出思考，没有调用工具也没有给出回答。"
                    "系统已自动切换为总结模式，请直接给出简洁的最终回答。"
                ))
                logger.warning(
                    "[run:%s] 推理死循环护栏第二次触发且无正文，切换为总结模式 (reasoning=%d)",
                    state.get("run_id", ""), _reasoning_chars,
                )
                await _emit(state, {
                    "type": "status",
                    "status": "reasoning_loop_blocked",
                    "detail": f"推理死循环拦截: {reason}，已切换为总结模式",
                })
                continue
            if _reasoning_cut == "self_repeat":
                cut_detail = "你刚才的思考开始重复同一段分析，已被系统中断。"
            elif _reasoning_guard_triggered:
                cut_detail = f"你刚才的思考过程超过了 {_reasoning_chars} 字，已被系统截断。"
            else:
                cut_detail = f"你刚才输出了超过 {len(text)} 字的分析，但没有采取行动。"
            nudge = (
                f"【强制收敛 · 推理死循环检测 · 系统自动提醒】{cut_detail}"
                "你还没有发起工具调用或给出最终回答。已有分析会保留到下一轮，"
                "本轮直接基于它行动：调用一个工具去执行/验证，或给出当前能给出的最佳回答。"
                "不要从头重新分析，也不要重复之前的推理。"
            )
            messages.append(_nudge(nudge))
            logger.warning(
                "[run:%s] 推理死循环护栏触发: %s, text=%d, reasoning=%d, tool_calls=%d",
                state.get("run_id", ""), reason, len(text), _reasoning_chars, len(tool_calls),
            )
            await _emit(state, {
                "type": "status",
                "status": "reasoning_loop_blocked",
                "detail": f"推理死循环拦截: {reason}",
            })
            continue  # 进入下一轮循环，让模型看到这条强提醒后收敛

        # 跨轮思考重复检测：reasoning 不进入消息历史，模型每轮都从同样的工具结果
        # 重新推导，容易反复输出几乎一样的分析却不推进任务。连续复述时先提醒换思路，
        # 继续复述则关闭工具强制收敛（与 loop_guard 的总结模式一致）。
        _reasoning_nudge = ""
        _reasoning_text = "".join(_round_reasoning).strip()
        if len(_reasoning_text) >= _REASONING_REPEAT_MIN_CHARS:
            _shingles = _reasoning_shingles(_reasoning_text)
            _prev_shingles = state.get("_last_reasoning_shingles") or frozenset()
            if _reasoning_similarity(_shingles, _prev_shingles) >= _REASONING_REPEAT_SIMILARITY:
                state["_reasoning_repeat_count"] = int(state.get("_reasoning_repeat_count", 0)) + 1
            else:
                state["_reasoning_repeat_count"] = 0
            state["_last_reasoning_shingles"] = _shingles
        else:
            # 本轮思考过短不足以比较：打断"连续重复"，避免残留计数误触总结
            state["_reasoning_repeat_count"] = 0
        _reasoning_repeat_count = int(state.get("_reasoning_repeat_count", 0))
        if not _summary_mode and tool_calls and _reasoning_repeat_count == 2:
            _reasoning_nudge = (
                "【思考重复提醒】你的思考内容与前几轮几乎完全相同，说明你一直在原地分析，"
                "没有产生新的判断。停止复述这段分析，只看已有的工具结果决定下一步："
                "要么换一种与之前不同的做法行动，要么直接给出当前能给出的最佳回答。"
            )
            logger.warning(
                "[run:%s] 检测到跨轮思考重复，已提醒模型换思路 (round %d, repeat=%d)",
                state.get("run_id", ""), rounds, _reasoning_repeat_count,
            )
            await _emit(state, {
                "type": "status",
                "status": "reasoning_repeat_nudge",
                "detail": "思考内容与前几轮重复，已提醒模型换一种思路",
            })
        if not _summary_mode and tool_calls and _reasoning_repeat_count >= 3:
            state["_loop_summary_mode"] = True
            state["_loop_guard_stage"] = 2
            messages.extend(
                _blocked_tool_messages(
                    tool_calls, "【思考重复检测】本轮工具调用已取消（工具已暂时禁用）。"
                )
            )
            messages.append(_nudge(
                "【思考重复检测 · 系统自动提醒，不是用户留言】你已连续多轮重复同一段思考，且没有带来任何进展。"
                "本轮工具调用已取消，接下来工具将暂时禁用。\n"
                "请立即停止分析，直接基于已经获得的信息给出最终回答："
                "说明已完成的内容、未完成的部分及原因、以及建议用户下一步怎么做。"
            ))
            logger.warning(
                "[run:%s] 思考重复触发总结模式 (round %d, repeat=%d)",
                state.get("run_id", ""), rounds, _reasoning_repeat_count,
            )
            await _emit(state, {
                "type": "status",
                "status": "reasoning_repeat_blocked",
                "detail": "连续多轮思考内容重复，已切换为总结模式",
            })
            continue

        # 重复 read_file / list_dir 检测：模型要读一个之前读过、且中间没被修改的路径
        # → 追加强提醒，终止本轮工具执行，让模型收到提醒后下一轮调整
        _dup_read_blocked = False
        _dup_nudge = ""
        if tool_calls:
            for tc in tool_calls:
                if tc.get("name") in ("read_file", "list_dir", "list_files"):
                    args = tc.get("args") or {}
                    raw_path = args.get("path") or args.get("directory") or args.get("dir")
                    path = _read_key(state.get("workspace", ""), tc.get("name", ""), args)
                    base = _path_key(state.get("workspace", ""), str(raw_path)) if raw_path else ""
                    if path and path in _file_read_round and base not in _file_modified:
                        _dup_nudge = (
                            f"【纪律提醒】你已经在第 {_file_read_round[path]} 轮读取过 `{raw_path}`，"
                            f"该路径自上次读取以来未被任何 write_file / edit_file / apply_patch 修改。"
                            f"请直接基于已掌握的信息继续推理，不要重复读取。"
                            f"如果文件确实被修改过，说明中间有写操作漏掉了——请先补上修改，"
                            f"或者确认后告诉我你需要什么新信息。"
                        )
                        logger.warning(
                            "[run:%s] 拦截重复读取未修改文件: %s (上次 round=%d)",
                            state.get("run_id", ""), path, _file_read_round[path],
                        )
                        await _emit(state, {
                            "type": "status",
                            "status": "duplicate_read_blocked",
                            "detail": f"重复读取未修改文件: {raw_path}",
                        })
                        _dup_read_blocked = True
                        break
        if _dup_read_blocked:
            messages.extend(
                _blocked_tool_messages(
                    tool_calls,
                    "【重复调用拦截】本轮所有工具调用已取消：检测到对未修改文件的重复读取。",
                )
            )
            messages.append(_nudge(_dup_nudge))
            # 已提醒过一次：放行下一次读取，避免"拦截→重试→再拦截"本身形成死循环
            _file_read_round.pop(path, None)
            continue

        # 没有工具调用 → 完成前的落盘兜底检测：
        # 最终回答含代码块，但本轮 run 从未调用 write_file / edit_file →
        # 模型把代码"贴"在回复里当成交付物，违反落盘纪律。
        # 追加提醒让模型补写（最多两次，防死循环；第三次仍不写则放行并记录）
        _nudge_count = state.get("_file_write_nudged", 0)
        if (
            not tool_calls
            and not _summary_mode
            and _nudge_count < 2
            and not _did_write
            and re.search(r"```[\s\S]*?```", text)
            and text.count("\n") >= 3
        ):
            state["_file_write_nudged"] = _nudge_count + 1
            nudge = (
                "【落盘纪律提醒】你的回答包含代码块，但本次任务尚未调用 write_file 或 edit_file "
                "将代码写入文件。用户明确要求：代码修改必须实际落盘，禁止仅在回复中粘贴代码片段作为交付。\n"
                "请立即调用相应工具将上述代码写入正确文件，完成后再给出最终确认。"
            )
            if _nudge_count == 1:
                nudge += "\n【第二次警告】若你无法或不愿调用工具写文件，请直接说明原因，不要输出代码块。"
            messages.append(_nudge(nudge))
            logger.warning("[run:%s] 检测到代码未落盘，已提醒模型补写（第%d次）", state.get("run_id", ""), _nudge_count + 1)
            await _emit(state, {
                "type": "status",
                "status": "file_write_nudge",
                "detail": f"检测到代码未写入文件，已提醒模型补写（第{_nudge_count + 1}次）",
            })
            continue
        # 提前收尾护栏：创建/修改类任务，模型一次操作性工具都没调就想结束 →
        # 模型可能在"假装已完成"（只输出总结/道歉/计划文字）。注入提醒强制实际执行（最多一次）。
        if (
            not tool_calls
            and not _summary_mode
            and not state.get("_premature_done_nudged")
            and _CREATION_HINT_RE.search(state.get("request", "") or "")
            and not _did_productive
        ):
            state["_premature_done_nudged"] = True
            messages.append(_nudge(
                "【执行纪律提醒】这是一个开发/修改类任务，但本次运行你还没有调用任何实际操作工具"
                "（write_file / edit_file / terminal 等）。禁止仅用文字声称任务已完成。\n"
                "请立即调用相应工具实际执行任务；若任务确实无需工具，请直接说明原因。"
            ))
            logger.warning(
                "[run:%s] 检测到未执行任何操作就想收尾，已提醒模型实际执行 (round %d)",
                state.get("run_id", ""), rounds,
            )
            await _emit(state, {
                "type": "status",
                "status": "premature_done_blocked",
                "detail": "模型未执行任何操作就想结束，已提醒其实际执行",
            })
            continue
        # 计划未完成护栏：仍有 pending（尚未开始）的步骤却想直接收尾 →
        # 提醒模型继续执行或显式更新计划状态（最多一次，避免死循环）。
        if (
            not tool_calls
            and not _summary_mode
            and not state.get("_incomplete_todos_nudged")
            and any(t.get("status") == "pending" for t in (state.get("todos") or []))
        ):
            pending = [t for t in state.get("todos") or [] if t.get("status") == "pending"]
            remaining = "、".join(str(t.get("content", ""))[:40] for t in pending[:5])
            state["_incomplete_todos_nudged"] = True
            messages.append(_nudge(
                f"【计划未完成提醒】执行计划中仍有未开始的步骤：{remaining}。当前不能直接收尾。\n"
                "请继续执行剩余步骤；若这些步骤确实已不再需要，请调用 write_todos "
                "把对应项标记为 completed（或移除），再给出最终回答。"
            ))
            logger.warning(
                "[run:%s] 计划仍有 pending 步骤却准备收尾，已提醒模型 (round %d, pending=%d)",
                state.get("run_id", ""), rounds, len(pending),
            )
            await _emit(state, {
                "type": "status",
                "status": "plan_incomplete_blocked",
                "detail": f"计划仍有 {len(pending)} 个未开始步骤，已提醒模型继续执行",
            })
            continue
        if not tool_calls:
            logger.info("[run:%s] execute_node 退出: 模型无工具调用，正常给出最终回答 (round %d, text=%d chars)", state.get("run_id",""), rounds, len(text))
            break

        # 总结模式已解绑工具：模型若仍返回工具调用，一律不执行；
        # 连续两轮仍如此则按系统自动终止收尾，避免无工具可调时原地打转。
        if _summary_mode:
            _summary_tool_rounds = int(state.get("_summary_tool_rounds", 0)) + 1
            state["_summary_tool_rounds"] = _summary_tool_rounds
            messages.extend(_blocked_tool_messages(
                tool_calls, "【系统自动保护】总结模式下工具已禁用，本轮调用未执行。"
            ))
            if _summary_tool_rounds >= 2:
                hint = "系统自动终止：总结模式下仍返回工具调用，无法继续执行。请换一种方式描述任务，或缩小范围后重试。"
                messages.append(AIMessage(content=hint))
                await _emit(state, {"type": "status", "status": "loop_guard", "detail": hint})
                logger.warning(
                    "[run:%s] execute_node 退出: 总结模式下仍返回工具调用 (round %d)",
                    state.get("run_id", ""), rounds,
                )
                break
            messages.append(_nudge(
                "【系统自动保护 · 不是用户留言】总结模式下工具已禁用，请不要再调用工具，"
                "直接基于已有信息给出最终回答。"
            ))
            await _emit(state, {
                "type": "status",
                "status": "summary_tool_blocked",
                "detail": "总结模式下工具调用被拦截",
            })
            continue

        # 重复检测：连续多轮完全相同且无进展的调用 → 分阶段处理。
        # 先取消本轮并提醒换做法继续（不结束任务）；仍然重复才切总结模式。
        sig = _tool_calls_signature(tool_calls)
        if sig == state.get("_last_tool_sig"):
            state["_repeat_count"] = state.get("_repeat_count", 0) + 1
        else:
            state["_last_tool_sig"] = sig
            state["_repeat_count"] = 0
            # 换了新做法：重新给一次"先提醒后升级"的机会
            state["_loop_guard_stage"] = 0
        if state["_repeat_count"] >= 3:
            _loop_stage = int(state.get("_loop_guard_stage", 0))
            _loop_blocks = int(state.get("_loop_guard_blocks", 0)) + 1
            state["_loop_guard_blocks"] = _loop_blocks
            if state["_repeat_count"] >= 6 or _loop_blocks >= 5:
                # 反复换签名但一直绕圈时同样收敛，保证运行有界
                state["_loop_guard_stage"] = 2
                state["_loop_summary_mode"] = True
                hint = (
                    "【系统自动保护 · 不是用户留言】系统已多次拦截重复的工具调用，任务没有实质进展。"
                    "这是系统自动切换为总结模式，不是用户要求你停止。\n"
                    "本轮工具调用已取消，接下来工具将暂时禁用。请立即基于已经获得的信息给出最终回答："
                    "说明已完成的内容、未完成的部分及原因、以及建议用户下一步怎么做。"
                )
                messages.extend(_blocked_tool_messages(tool_calls, "【系统自动保护】本轮工具调用已取消（工具已暂时禁用）。"))
                messages.append(_nudge(hint))
                await _emit(state, {
                    "type": "status",
                    "status": "loop_guard",
                    "detail": "重复工具调用仍未停止，已切换为总结模式",
                })
                logger.warning(
                    "[run:%s] loop_guard 第二阶段：切换为总结模式 (round %d, repeat_count=%d, blocks=%d)",
                    state.get("run_id", ""), rounds, state["_repeat_count"], _loop_blocks,
                )
                continue
            if _loop_stage == 0:
                state["_loop_guard_stage"] = 1
                hint = (
                    "【系统自动保护 · 不是用户留言】这是系统检测到连续多轮完全相同的工具调用后自动发出的提醒，"
                    "不是用户的要求，用户没有让你停止任务，任务也没有完成。本轮重复调用已取消。\n"
                    "请换一种做法继续完成任务，例如：修改参数或路径、换用其他工具、先缩小范围确认问题所在。"
                    "不要因此给出结束语，也不要声称用户要求你停止；只有任务确实完成或确实无法继续时才可以收尾。"
                )
                messages.extend(_blocked_tool_messages(
                    tool_calls, "【系统自动保护】本轮工具调用已取消：与之前几轮完全相同，没有产生新信息。"
                ))
                messages.append(_nudge(hint))
                await _emit(state, {
                    "type": "status",
                    "status": "loop_guard",
                    "detail": "检测到重复工具调用，本轮调用已取消，已提醒换一种做法继续",
                })
                logger.warning(
                    "[run:%s] loop_guard 第一阶段：已拦截重复调用并提醒换做法 (round %d, repeat_count=%d)",
                    state.get("run_id", ""), rounds, state["_repeat_count"],
                )
                continue
            # 已提醒过换做法，模型仍原样重复：继续拦截，但再给几次调整机会
            messages.extend(_blocked_tool_messages(
                tool_calls, "【系统自动保护】本轮工具调用已取消：与上一轮完全相同，没有产生新信息。"
            ))
            messages.append(_nudge(
                "【系统自动保护 · 不是用户留言】你仍然在重复完全相同的工具调用。"
                "请立即换一种做法（改参数、换工具、缩小范围），继续完成任务；"
                "用户没有让你停止，不要给出结束语。"
            ))
            await _emit(state, {
                "type": "status",
                "status": "loop_guard",
                "detail": "重复工具调用再次被拦截，继续要求换做法",
            })
            logger.warning(
                "[run:%s] loop_guard 持续拦截重复调用 (round %d, repeat_count=%d)",
                state.get("run_id", ""), rounds, state["_repeat_count"],
            )
            continue

        # write_todos 重复检测：连续多轮只调用 write_todos 且内容相同 → 强制推进。
        # 第一次已写入计划，第二次提交完全相同的内容没有任何意义，直接拦截。
        _write_todos_calls = [tc for tc in tool_calls if tc["name"] == WRITE_TODOS.name]
        if _write_todos_calls and len(_write_todos_calls) == len(tool_calls):
            _wt_sig = json.dumps([tc["args"] for tc in _write_todos_calls], sort_keys=True, ensure_ascii=False)
            if _wt_sig == state.get("_last_wt_sig"):
                state["_wt_repeat_count"] = state.get("_wt_repeat_count", 0) + 1
            else:
                state["_last_wt_sig"] = _wt_sig
                state["_wt_repeat_count"] = 0
            if state["_wt_repeat_count"] >= 1:
                nudge = (
                    "【计划重复检测】你已连续多轮调用 write_todos 提交相同的计划，但没有执行任何实际工具操作。"
                    "计划已经制定好了，现在请立即开始执行：调用具体的工具（write_file / edit_file / terminal 等）"
                    "来完成计划中的第一个任务。不要再调用 write_todos。"
                )
                messages.extend(
                    _blocked_tool_messages(
                        tool_calls, "【计划重复检测】本轮 write_todos 调用已取消，请直接开始执行计划。"
                    )
                )
                messages.append(_nudge(nudge))
                logger.warning(
                    "[run:%s] write_todos 重复检测触发 (round %d, repeat=%d)",
                    state.get("run_id", ""), rounds, state["_wt_repeat_count"],
                )
                await _emit(state, {
                    "type": "status",
                    "status": "write_todos_loop_blocked",
                    "detail": "write_todos 重复调用拦截，已提醒模型开始执行",
                })
                continue

        # 文本输出重复检测：连续多轮输出相同文本且无工具调用进展 → 强制终止
        _text_sig = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        if _text_sig and _text_sig == state.get("_last_text_sig"):
            state["_text_repeat_count"] = state.get("_text_repeat_count", 0) + 1
        else:
            state["_last_text_sig"] = _text_sig
            state["_text_repeat_count"] = 0
        if state["_text_repeat_count"] >= 2 and tool_calls:
            nudge = (
                "【输出重复检测】你的回复内容与上一轮几乎完全相同，但任务尚未完成。"
                "请基于已有的工具结果直接行动，不要重复相同的分析和计划。"
                "如果计划已经制定好，立即调用工具执行；如果已完成，直接给出最终回答。"
            )
            messages.extend(
                _blocked_tool_messages(
                    tool_calls, "【输出重复检测】本轮工具调用已取消，请基于已有结果直接行动。"
                )
            )
            messages.append(_nudge(nudge))
            logger.warning(
                "[run:%s] 文本输出重复检测触发 (round %d, repeat=%d)",
                state.get("run_id", ""), rounds, state["_text_repeat_count"],
            )
            await _emit(state, {
                "type": "status",
                "status": "text_repeat_blocked",
                "detail": "文本输出重复拦截",
            })
            continue

        # 执行工具：write_todos 即时处理；真实工具按"连续幂等段并行、写操作串行"调度。
        # ToolMessage 先写入 slots 保持与 tool_calls 的顺序对应，全部完成后统一入列。
        state["_converge_nudges"] = 0  # 有实际工具进展，长文本收敛计数重新开始
        name_to_tool = {t.name: t for t in tools}
        slots: list[ToolMessage | None] = [None] * len(tool_calls)
        real_calls: list[tuple[int, dict]] = []

        for idx, tc in enumerate(tool_calls):
            # TodoListMiddleware：拦截计划更新工具（不进入真实工具表）
            if tc["name"] == WRITE_TODOS.name:
                raw = tc["args"].get("todos") if isinstance(tc["args"], dict) else None
                normalized = normalize_todos(raw)
                if not normalized:
                    # 空计划会清空前端进度并丢失上下文，直接拒绝并让模型自纠
                    started_at = time.strftime("%H:%M:%S")
                    message = (
                        "错误：todos 不能为空。请提供至少 1 项步骤，"
                        "每项包含 content 与 status（pending | in_progress | completed）。"
                    )
                    await _emit(state, {
                        "type": "tool_start", "tool": WRITE_TODOS.name, "args": tc["args"],
                        "call_id": tc["id"], "started_at": started_at,
                    })
                    await _emit(state, {
                        "type": "tool_end", "tool": WRITE_TODOS.name, "call_id": tc["id"],
                        "result": message, "status": "error",
                        "duration_ms": 0, "started_at": started_at,
                    })
                    slots[idx] = ToolMessage(
                        content=message, tool_call_id=tc["id"], name=WRITE_TODOS.name,
                    )
                    continue
                todos = apply_todos(state, raw)
                started_at = time.strftime("%H:%M:%S")
                logger.info(
                    "[run:%s] 计划更新（write_todos）: %d/%d 完成",
                    state.get("run_id", ""),
                    sum(1 for t in todos if t["status"] == "completed"),
                    len(todos),
                )
                await _emit(state, {
                    "type": "tool_start", "tool": WRITE_TODOS.name, "args": tc["args"],
                    "call_id": tc["id"], "started_at": started_at,
                })
                # 确认文案与中间件 _write_todos 的 ToolMessage 格式一致；
                # 状态更新已在 apply_todos 中等价完成，无需真实调用工具
                reminder = f"Updated todo list to {todos}"
                await _emit(state, {
                    "type": "tool_end", "tool": WRITE_TODOS.name, "call_id": tc["id"],
                    "result": reminder, "status": "success",
                    "duration_ms": 0, "started_at": started_at,
                })
                await _emit(state, {
                    "type": "plan", "steps": state["plan"], "todos": todos,
                })
                slots[idx] = ToolMessage(
                    content=reminder, tool_call_id=tc["id"], name=WRITE_TODOS.name,
                )
                continue
            real_calls.append((idx, tc))

        # 同一轮内去重：模型在批量创建文件时偶尔会把同一调用重复输出两次，
        # 写操作非幂等会被真实执行两遍。按 name+args 只执行第一次，后续补 ToolMessage。
        first_by_key: dict[str, int] = {}
        unique_calls: list[tuple[int, dict]] = []
        duplicate_calls: list[tuple[int, int]] = []
        for idx, tc in real_calls:
            key = json.dumps(
                [tc.get("name", ""), tc.get("args") or {}],
                sort_keys=True, ensure_ascii=False, default=str,
            )
            if key in first_by_key:
                duplicate_calls.append((idx, first_by_key[key]))
                logger.warning(
                    "[run:%s] 拦截同轮重复工具调用: %s args=%s（首次 call_id=%s）",
                    state.get("run_id", ""), tc.get("name", ""),
                    json.dumps(tc.get("args") or {}, ensure_ascii=False)[:200],
                    tool_calls[first_by_key[key]]["id"],
                )
            else:
                first_by_key[key] = idx
                unique_calls.append((idx, tc))

        # 分组执行：长度>1 的组必为连续幂等段（asyncio.gather 并行）；单点串行
        try:
            for group in _group_real_calls(unique_calls):
                if len(group) == 1:
                    idx, tc = group[0]
                    slots[idx] = await _run_one_tool(
                        state, tc, name_to_tool.get(tc["name"]), tool_result_cache,
                    )
                else:
                    logger.info(
                        "[run:%s] 并行执行 %d 个幂等工具: %s",
                        state.get("run_id", ""), len(group),
                        [tc["name"] for _, tc in group],
                    )
                    results = await asyncio.gather(*(
                        _run_one_tool(state, tc, name_to_tool.get(tc["name"]), tool_result_cache)
                        for _, tc in group
                    ))
                    for (idx, _), msg in zip(group, results, strict=True):
                        slots[idx] = msg
        except asyncio.CancelledError:
            # 用户停止：工具内部 check_stopped 会抛 CancelledError，
            # 这里优雅退出循环并保留已产生的消息，而不是让整个 run 被取消
            if state.get("stop_event") and state["stop_event"].is_set():
                state["stopped"] = True
                await _emit(state, {"type": "status", "status": "stopped"})
                break
            raise

        # 重复调用不重复执行：补一条 ToolMessage 引用首次结果，保持
        # AIMessage.tool_calls 与 ToolMessage 一一对应（否则下一轮请求会被端点拒绝）
        for dup_idx, first_idx in duplicate_calls:
            dup_tc = tool_calls[dup_idx]
            first_msg = slots[first_idx]
            slots[dup_idx] = ToolMessage(
                content=(
                    f"【重复调用拦截】本轮已存在完全相同的 {dup_tc['name']} 调用"
                    f"（call_id={tool_calls[first_idx]['id']}），本次未重复执行。\n\n"
                    f"--- 首次调用结果 ---\n{first_msg.content if first_msg else ''}"
                ),
                tool_call_id=dup_tc["id"],
                name=dup_tc["name"],
            )

        for msg in slots:
            if msg is not None:
                messages.append(msg)

        # 思考重复提醒必须放在 ToolMessage 之后：AIMessage(tool_calls) 与 ToolMessage
        # 之间不能插入其他消息，否则下一轮请求会被端点拒绝
        if _reasoning_nudge:
            messages.append(_nudge(_reasoning_nudge))

        # 记录本轮实际派发过的工具（落盘/收尾护栏只看本轮，不跨历史累计）
        for _, tc in real_calls:
            _tc_name = tc.get("name", "")
            _tc_args = tc.get("args") or {}
            if _tool_mutates(_tc_name, _tc_args):
                _did_productive = True
                _did_write = True
            elif _tc_name in _PRODUCTIVE_TOOLS:
                _did_productive = True

        # 更新路径追踪器：本轮执行的工具结果已经入列，下一轮检测才能正确判断"是否修改过"
        for tc in tool_calls:
            args = tc.get("args") or {}
            tname = tc.get("name", "")
            if tname in ("write_file", "edit_file", "delete_file", "move_file"):
                for key in ("path", "file_path", "target_path", "source", "destination"):
                    path = args.get(key)
                    if path:
                        _file_modified.add(_path_key(state.get("workspace", ""), str(path)))
            elif tname == "apply_patch":
                # 从 patch 文本里提取被修改的文件路径
                patch_text = str(args.get("patch") or "")
                for line in patch_text.splitlines():
                    if line.startswith("+++ ") or line.startswith("--- "):
                        raw = line[4:].strip()
                        if raw and raw != "/dev/null":
                            raw = raw[2:] if raw.startswith(("a/", "b/")) else raw
                            _file_modified.add(_path_key(state.get("workspace", ""), raw))
            elif tname in ("read_file", "list_dir", "list_files"):
                path = _read_key(state.get("workspace", ""), tname, args)
                if path:
                    _file_read_round[path] = rounds

    state["messages"] = messages
    return state


async def context_node(state: AgentState) -> AgentState:
    """上下文管理节点：超限自动摘要，并把摘要持久化到 session.summary。"""
    # 出口处广播累计 token 用量
    usage = state.get("token_usage")
    if usage:
        await _emit(state, {"type": "usage", **usage})
    if state.get("error") or state.get("stopped"):
        return state
    messages = list(state.get("messages", []))
    try:
        llm = await _make_llm(state)
    except Exception:
        return state
    new_messages = await maybe_summarize_messages(
        messages, llm, state.get("model", ""), emit=lambda e, _s=state: _emit(_s, e)
    )
    if len(new_messages) != len(messages):
        state["messages"] = new_messages
    # 执行循环中途压缩产生的摘要也要同步进 state（并持久化到 session.summary）；
    # 未发生新压缩时摘要与 state 相同，不重复推送。
    summary = extract_summary(new_messages)
    if summary and summary != state.get("summary"):
        state["summary"] = summary
        state["summary_count"] = int(state.get("summary_count", 0)) + 1
        await _emit(state, {"type": "summary", "summary": summary[:500]})
    return state


def build_graph() -> Any:
    """构建并编译 LangGraph：START → execute（ReAct 循环）→ context → END。"""
    g = StateGraph(AgentState)

    g.add_node("execute", execute_node)
    g.add_node("context", context_node)

    g.add_edge(START, "execute")
    g.add_edge("execute", "context")
    g.add_edge("context", END)

    return g.compile()


graph = build_graph()
