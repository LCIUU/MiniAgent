"""工具发现 API：输出已注册工具的 manifest（名称/类别/风险/超时/幂等/描述）。

注册发生在工具工厂执行时（@tool_spec 装饰器）；若服务启动后尚未产生任何
Agent 运行，首次请求会用空上下文触发一次惰性构建以完成注册。
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from ..config import get_effective
from ..tools import build_tools
from ..tools.context import ApprovalManager
from ..tools.registry import manifest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tools", tags=["tools"])


def _ensure_registered() -> None:
    """注册表为空时用空上下文构建一次工具，触发 @tool_spec 注册。"""
    if manifest():
        return
    try:
        build_tools(
            workspace=get_effective("workspace", ".") or ".",
            emit=lambda ev: asyncio.sleep(0),
            approvals=ApprovalManager(),
            stop_event=asyncio.Event(),
        )
    except Exception as e:
        logger.warning("工具清单惰性注册失败: %s", e)


@router.get("")
async def list_tools() -> dict:
    """返回全部已注册工具的元数据清单。"""
    _ensure_registered()
    return {"tools": manifest()}
