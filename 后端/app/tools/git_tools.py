"""Git 工具：状态 / diff / 日志 / 提交。

只读操作（status/diff/log）可并行、可缓存；提交需审批，且只提交当前工作区。
所有命令通过 ``create_subprocess_exec`` 执行，不拼接 shell 字符串，避免注入。
"""

from __future__ import annotations

import asyncio
import shutil

from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import RISK_HIGH, RISK_MEDIUM, tool_spec

_MAX = 12000


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _truncate(text: str) -> str:
    limit = int(get_effective("tool_output_max_chars", 12000))
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n...[已截断，共 {len(text)} 字符]...\n" + text[-limit // 2 :]


async def _git(ctx: ToolContext, *args: str, check: bool = False) -> tuple[int, str]:
    if shutil.which("git") is None:
        return 127, "未找到 git 可执行文件，请先安装 Git。"
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=ctx.workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
    except TimeoutError:
        proc.kill()
        return 124, "git 命令超时。"
    code = proc.returncode or 0
    text = _decode(out).strip()
    if check and code != 0:
        raise RuntimeError(text or f"git 退出码 {code}")
    return code, text


def _not_repo(code: int, text: str) -> bool:
    return code == 128 and ("not a git repository" in text.lower())


def make_git_tools(ctx: ToolContext):

    @tool_spec(category="git", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def git_status() -> str:
        """
        查看当前工作区的 Git 状态（分支、已修改/已暂存/未跟踪文件）

        Returns:
            git status --short --branch 的输出
        """
        ctx.check_stopped()
        code, text = await _git(ctx, "status", "--short", "--branch")
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        return _truncate(text) if text else "工作区干净（无改动）。"

    @tool_spec(category="git", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def git_diff(path: str = "", staged: bool = False, stat: bool = False) -> str:
        """
        查看 Git 差异

        Args:
            path: 可选，只看某个路径的差异
            staged: 是否查看已暂存（--cached）的差异
            stat: 是否只显示统计信息（--stat）

        Returns:
            git diff 输出
        """
        ctx.check_stopped()
        args = ["diff"]
        if staged:
            args.append("--cached")
        if stat:
            args.append("--stat")
        if path:
            args += ["--", path]
        code, text = await _git(ctx, *args)
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        return _truncate(text) if text else "没有差异。"

    @tool_spec(category="git", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def git_log(limit: int = 10, path: str = "") -> str:
        """
        查看最近的 Git 提交

        Args:
            limit: 返回条数，1-50
            path: 可选，只看某个路径的历史

        Returns:
            git log --oneline 输出
        """
        ctx.check_stopped()
        limit = max(1, min(int(limit), 50))
        args = ["log", f"-{limit}", "--oneline", "--decorate"]
        if path:
            args += ["--", path]
        code, text = await _git(ctx, *args)
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        return _truncate(text) if text else "没有提交记录。"

    @tool_spec(category="git", risk_level=RISK_HIGH, timeout=120)
    @tool(parse_docstring=True)
    async def git_commit(message: str, add_all: bool = True) -> str:
        """
        提交当前工作区的改动（需要用户审批）

        Args:
            message: 提交信息
            add_all: 是否先把所有改动加入暂存区（git add -A），默认 True

        Returns:
            提交结果（提交号与摘要）
        """
        ctx.check_stopped()
        if not message.strip():
            return "错误：提交信息不能为空。"
        code, text = await _git(ctx, "status", "--short")
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        if not text.strip():
            return "没有需要提交的改动。"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "git_commit",
            "high_risk",
            {"command": f"git commit -m {message!r}", "cwd": ctx.workspace},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了提交操作。"
        if add_all:
            code, text = await _git(ctx, "add", "-A")
            if code != 0:
                return f"git add 失败: {text}"
        code, text = await _git(ctx, "commit", "-m", message)
        if code != 0:
            return f"提交失败: {text}"
        _, summary = await _git(ctx, "log", "-1", "--oneline")
        return f"提交成功: {summary}"

    @tool_spec(category="git", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def git_branch(action: str = "list", name: str = "") -> str:
        """
        管理 Git 分支：列出、创建、删除分支

        Args:
            action: "list" 列出所有分支；"create" 创建新分支；"delete" 删除分支
            name: 分支名（create/delete 时必填）

        Returns:
            操作结果
        """
        ctx.check_stopped()
        if action == "list":
            code, text = await _git(ctx, "branch", "-a", "--sort=-committerdate")
            if _not_repo(code, text):
                return "当前工作区不是 Git 仓库。"
            return _truncate(text) if text else "没有分支。"
        if action == "create":
            if not name.strip():
                return "错误：创建分支需要指定 name。"
            code, text = await _git(ctx, "branch", name.strip())
            if code != 0:
                return f"创建分支失败: {text}"
            return f"分支已创建: {name.strip()}"
        if action == "delete":
            if not name.strip():
                return "错误：删除分支需要指定 name。"
            approved = await ctx.approvals.request(
                ctx.emit_event,
                "git_branch",
                "high_risk",
                {"command": f"git branch -d {name.strip()}", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了删除分支操作。"
            code, text = await _git(ctx, "branch", "-d", name.strip())
            if code != 0:
                return f"删除分支失败: {text}"
            return f"分支已删除: {name.strip()}"
        return f"未知 action: {action}（支持 list / create / delete）"

    @tool_spec(category="git", risk_level=RISK_HIGH, timeout=60, idempotent=False)
    @tool(parse_docstring=True)
    async def git_checkout(branch_or_path: str, create: bool = False) -> str:
        """
        切换分支或恢复文件

        Args:
            branch_or_path: 分支名或文件路径
            create: 是否创建并切换到新分支（相当于 git checkout -b），默认 False

        Returns:
            切换结果
        """
        ctx.check_stopped()
        if not branch_or_path.strip():
            return "错误：branch_or_path 不能为空。"
        # checkout 路径会丢弃未提交改动；切分支也可能因冲突/覆盖丢失工作区状态。
        # 创建新分支（-b）不具破坏性，无需打断用户。
        if not create:
            approved = await ctx.approvals.request(
                ctx.emit_event,
                "git_checkout",
                "high_risk",
                {"command": f"git checkout {branch_or_path.strip()}", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了 checkout 操作。"
        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(branch_or_path.strip())
        code, text = await _git(ctx, *args)
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        if code != 0:
            return f"checkout 失败: {text}"
        return _truncate(text) if text else f"已切换到: {branch_or_path.strip()}"

    @tool_spec(category="git", risk_level=RISK_HIGH, timeout=120)
    @tool(parse_docstring=True)
    async def git_merge(branch: str, no_ff: bool = False) -> str:
        """
        合并指定分支到当前分支（需要用户审批）

        Args:
            branch: 要合并的分支名
            no_ff: 是否强制创建合并提交（--no-ff），默认 False

        Returns:
            合并结果
        """
        ctx.check_stopped()
        if not branch.strip():
            return "错误：branch 不能为空。"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "git_merge",
            "high_risk",
            {"command": f"git merge {branch.strip()}", "cwd": ctx.workspace},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了合并操作。"
        args = ["merge"]
        if no_ff:
            args.append("--no-ff")
        args.append(branch.strip())
        code, text = await _git(ctx, *args)
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        if code != 0:
            if "CONFLICT" in text:
                return f"合并冲突:\n{text}\n请手动解决冲突后提交。"
            return f"合并失败: {text}"
        return _truncate(text) if text else f"合并成功: {branch.strip()}"

    @tool_spec(category="git", risk_level=RISK_HIGH, timeout=120)
    @tool(parse_docstring=True)
    async def git_rebase(branch: str, abort: bool = False, continue_rebase: bool = False) -> str:
        """
        变基当前分支到指定分支之上（需要用户审批）

        Args:
            branch: 目标分支名（abort/continue 时忽略）
            abort: 是否中止当前变基（git rebase --abort）
            continue_rebase: 解决冲突后是否继续变基（git rebase --continue）

        Returns:
            变基结果
        """
        ctx.check_stopped()
        if abort:
            code, text = await _git(ctx, "rebase", "--abort")
            if code != 0:
                return f"中止变基失败: {text}"
            return "变基已中止。"
        if continue_rebase:
            code, text = await _git(ctx, "rebase", "--continue")
            if code != 0:
                return f"继续变基失败: {text}"
            return _truncate(text) if text else "变基继续完成。"
        if not branch.strip():
            return "错误：branch 不能为空。"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "git_rebase",
            "high_risk",
            {"command": f"git rebase {branch.strip()}", "cwd": ctx.workspace},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了变基操作。"
        code, text = await _git(ctx, "rebase", branch.strip())
        if _not_repo(code, text):
            return "当前工作区不是 Git 仓库。"
        if code != 0:
            if "CONFLICT" in text:
                return f"变基冲突:\n{text}\n请手动解决冲突后用 git_rebase(continue_rebase=true) 继续，或用 git_rebase(abort=true) 中止。"
            return f"变基失败: {text}"
        return _truncate(text) if text else f"变基成功到: {branch.strip()}"

    @tool_spec(category="git", risk_level=RISK_MEDIUM, timeout=60)
    @tool(parse_docstring=True)
    async def git_stash(action: str = "list", message: str = "", index: int = 0) -> str:
        """
        管理 Git stash：暂存、列出、查看、恢复、删除

        适合在切换分支或对比改动前临时保存工作区，避免改动丢失。

        Args:
            action: "save" 暂存当前改动（含未跟踪文件）；"list" 列出记录；
                "show" 查看指定记录的内容；"pop" 应用并删除记录；"apply" 应用但保留记录；
                "drop" 删除记录（需要审批）
            message: 可选，save 时的说明文字
            index: 可选，操作第几条 stash，默认 0（最近一条）

        Returns:
            操作结果
        """
        ctx.check_stopped()
        action = action.lower().strip()
        index = max(0, int(index))
        ref = f"stash@{{{index}}}"

        if action == "list":
            code, text = await _git(ctx, "stash", "list")
            if _not_repo(code, text):
                return "当前工作区不是 Git 仓库。"
            return _truncate(text) if text else "没有 stash 记录。"

        if action == "save":
            args = ["stash", "push", "-u"]
            if message.strip():
                args += ["-m", message.strip()]
            approved = await ctx.approvals.request(
                ctx.emit_event,
                "git_stash",
                "normal",
                {"command": "git stash push -u" + (f" -m {message.strip()!r}" if message.strip() else ""), "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了暂存操作。"
            code, text = await _git(ctx, *args)
            if _not_repo(code, text):
                return "当前工作区不是 Git 仓库。"
            if code != 0:
                return f"stash 失败: {text}"
            return f"已暂存改动。\n{text}" if text else "已暂存改动。"

        if action == "show":
            code, text = await _git(ctx, "stash", "show", "-p", ref)
            if code != 0:
                return f"查看 {ref} 失败: {text}"
            return _truncate(text) if text else f"{ref} 没有内容。"

        if action in ("pop", "apply"):
            approved = await ctx.approvals.request(
                ctx.emit_event,
                "git_stash",
                "normal",
                {"command": f"git stash {action} {ref}", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return f"用户拒绝了 stash {action} 操作。"
            code, text = await _git(ctx, "stash", action, ref)
            if code != 0:
                if "conflict" in text.lower():
                    return f"恢复 {ref} 时发生冲突:\n{text}\n请解决冲突后手动清理 stash（git stash drop）。"
                return f"stash {action} 失败: {text}"
            kept = "（冲突或失败时记录会保留）" if action == "pop" else ""
            return _truncate(f"已恢复 {ref}{kept}\n{text}") if text else f"已恢复 {ref}。"

        if action == "drop":
            code, text = await _git(ctx, "stash", "list")
            if _not_repo(code, text):
                return "当前工作区不是 Git 仓库。"
            if not text.strip():
                return "没有 stash 记录。"
            approved = await ctx.approvals.request(
                ctx.emit_event,
                "git_stash",
                "high_risk",
                {"command": f"git stash drop {ref}（删除后无法恢复）", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了删除 stash 操作。"
            code, text = await _git(ctx, "stash", "drop", ref)
            if code != 0:
                return f"删除 {ref} 失败: {text}"
            return f"已删除 {ref}。"

        return f"未知 action: {action}（支持 save / list / show / pop / apply / drop）"

    return git_status, git_diff, git_log, git_commit, git_branch, git_checkout, git_merge, git_rebase, git_stash
