"""工作区快照工具：整仓创建 / 列出 / 回滚（批量改动前的安全点）。"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from ..services import checkpoints
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec


def make_checkpoint_tool(ctx: ToolContext):

    @tool_spec(category="fs", risk_level=RISK_MEDIUM, timeout=300)
    @tool(parse_docstring=True)
    async def workspace_checkpoint(
        action: str = "create",
        checkpoint_id: str = "",
        label: str = "",
    ) -> str:
        """
        创建、查看或恢复整个工作区的快照（批量修改前的安全点，可整体回滚）

        适用：执行批量替换、批量编码转换、大规模重构前先创建快照；出问题时可整体恢复。
        快照保存到 .agent_backups/checkpoints，自动忽略 .git、node_modules、.venv 等目录，
        单文件上限 5MB、总大小上限 300MB、最多 20000 个文件，默认保留最近 10 个快照。

        create 返回快照 id 与文件数/体积；list 查看最近快照；restore 需要用户审批，
        恢复前会自动再创建一个 pre-restore 快照，选错时可再次退回。工作区过大导致快照
        创建失败时，写操作仍会逐文件备份到 .agent_backups。

        Args:
            action: "create" 创建快照；"list" 列出最近快照；"restore" 恢复到指定快照。
            checkpoint_id: 可选，restore 时的快照 id；留空表示恢复最近一个。
            label: 可选，create 时给快照加说明，例如 "批量替换前"。

        Returns:
            创建/列表/恢复结果；失败时返回原因。
        """
        ctx.check_stopped()
        action = (action or "create").strip().lower()

        if action == "create":
            result = await asyncio.to_thread(checkpoints.create_checkpoint, ctx.workspace, label)
            if not result.get("ok"):
                return f"快照未创建：{result.get('error')}（本次改动仍会逐文件备份到 .agent_backups）"
            size_mb = result["bytes"] / 1024 / 1024
            note = f"，说明：{result['label']}" if result.get("label") else ""
            return (
                f"已创建快照 {result['checkpoint_id']}：{result['files']} 个文件，"
                f"{size_mb:.1f}MB{note}"
            )

        if action == "list":
            items = await asyncio.to_thread(checkpoints.list_checkpoints, ctx.workspace)
            if not items:
                return "还没有快照。可先调用 workspace_checkpoint(action=\"create\") 创建。"
            lines = ["最近快照:"]
            for it in items:
                size_mb = it.get("size", 0) / 1024 / 1024
                label = f"  {it['label']}" if it.get("label") else ""
                lines.append(
                    f"- {it.get('checkpoint_id')}  {it.get('created_at', '')}  "
                    f"{it.get('files', 0)} 个文件  {size_mb:.1f}MB{label}"
                )
            return "\n".join(lines)

        if action != "restore":
            return f"未知 action: {action}（支持 create / list / restore）"

        approved = await ctx.approvals.request(
            ctx.emit_event,
            "workspace_checkpoint",
            "high_risk",
            {
                "command": f"恢复工作区快照 {checkpoint_id or '(最近一个)'}",
                "cwd": ctx.workspace,
            },
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了快照恢复。"

        # 先确定恢复目标再打安全快照：checkpoint_id 为空时 restore 取的是"最新快照"，
        # 若先创建 pre-restore 快照，恢复目标就变成了刚创建的安全快照（原地打转仍报成功）
        target_id = checkpoint_id or await asyncio.to_thread(
            checkpoints.latest_checkpoint_id, ctx.workspace
        )
        if not target_id:
            return "还没有可恢复的快照，请先创建快照。"
        # 恢复前先给当前状态再打一个快照，选错时可以退回来
        safety = await asyncio.to_thread(
            checkpoints.create_checkpoint, ctx.workspace, "pre-restore"
        )
        result = await asyncio.to_thread(
            checkpoints.restore_checkpoint, ctx.workspace, target_id
        )
        if not result.get("ok"):
            return f"恢复失败：{result.get('error')}"
        lines = [
            f"已恢复快照 {result['checkpoint_id']}：{result['restored']} 个文件写回工作区",
        ]
        if safety.get("ok"):
            lines.append(f"恢复前的状态已另存为快照 {safety['checkpoint_id']}，可再次回退")
        else:
            lines.append("恢复前状态快照创建失败（原状态已无法整体回退，请留意）")
        if result.get("failed_total"):
            lines.append(f"失败 {result['failed_total']} 个：" + "；".join(result["failed"]))
        return "\n".join(lines)

    return workspace_checkpoint
