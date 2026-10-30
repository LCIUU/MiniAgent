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
        查看当前工作区的 Git 状态：分支、已暂存/已修改/未跟踪文件

        适用：提交前确认改动范围、排查冲突或确认工作区是否干净。只读，不会修改仓库；
        输出为 git status --short --branch 的简写格式。工作区不是 Git 仓库时返回明确提示。

        需要查看具体差异用 git_diff，查看历史用 git_log；提交前如发现意外改动，先确认再继续操作。

        清单沿用 git 的短格式标记：第一列是暂存区状态、第二列是工作区状态，?? 表示未跟踪文件，
        UU/AA 等标记表示存在冲突；首行 ## 显示当前分支以及有上游时的领先/落后情况。
        写入或还原文件后可随时重复调用确认结果，开销很小。

        Returns:
            当前分支与改动文件清单；工作区干净时返回"工作区干净"。
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
        查看 Git 差异（默认查看工作区未暂存的改动）

        适用：审查自己的修改、确认某个文件改了什么、生成提交前的变更摘要。只读。
        staged=true 查看已暂存（--cached）的差异；stat=true 只看统计（--stat）；
        path 可限定单个文件或目录。输出过长时保留头尾并标注截断。
        只读操作，不会暂存、提交或修改任何文件；staged 与未暂存改动需分别查看，
        两者都看时建议先跑一次默认调用再跑一次 staged=true。需要结合代码上下文时，
        用 read_file 读取对应文件片段；需要整体改动概览时改用 stat=true。

        Args:
            path: 可选，只看某个文件或目录的差异，相对当前工作区；留空表示全部改动。
            staged: 是否查看已暂存的差异（git diff --cached），默认 False 查看未暂存改动。
            stat: 是否只显示增删统计，默认 False 显示完整 diff。

        Returns:
            git diff 输出；没有差异或非 Git 仓库时返回对应提示。
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
        查看最近的 Git 提交记录（单行摘要）

        适用：了解最近的改动历史、确认提交信息风格、查找某次修改。只读。
        输出为 git log --oneline --decorate；limit 控制条数，path 可只看某个文件/目录的历史。
        返回内容只有单行摘要，不包含具体 diff；想查看某次提交的完整改动，可复制提交号后用
        terminal 执行 git show，或用 git_diff 对比工作区。历史很长时先用小 limit 看最近几条，
        再用 path 定位特定文件，避免一次拉取过多输出。

        Args:
            limit: 返回的提交条数，默认 10，范围 1-50。
            path: 可选，只看某个文件或目录的提交历史，相对当前工作区。

        Returns:
            单行提交列表（提交号 + 分支/tag + 摘要）；无记录或非 Git 仓库时返回提示。
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

        适用：用户明确要求提交时使用。执行前会检查是否是 Git 仓库、是否有改动；
        没有改动时直接返回提示。add_all=true 会先执行 git add -A 再提交，否则只提交
        已暂存内容。提交信息不能为空，建议用一行简短说明"为什么改"。

        未经用户明确要求不要主动提交；也不要提交密钥、.env 等敏感文件。
        提交成功后返回提交号与摘要。

        本工具只把提交写入本地仓库，不会 push；需要推送时先征求用户同意，再用 terminal 执行
        git push。提交前建议先用 git_diff 复核改动、用 run_checks 确认测试通过，避免提交半成品。

        Args:
            message: 提交信息，不能为空；例如 "修复登录接口超时问题"。
            add_all: 是否先把所有改动加入暂存区（git add -A），默认 True。

        Returns:
            提交成功后的提交号与摘要；用户拒绝或提交失败时返回原因。
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
        管理 Git 分支：列出、创建、删除

        适用：查看已有分支、创建功能分支、删除已合并的临时分支。list 只读；
        create 仅创建分支不切换（切换用 git_checkout）；delete 需要用户审批，
        且只能删除已合并分支（git branch -d），删除失败时先合并或改用 terminal。
        创建后如需立即在新分支上工作，需再调用 git_checkout；删除前请先确认该分支
        已合并或不再需要，未合并的提交不会自动迁移到其他分支。

        Args:
            action: "list" 列出所有分支；"create" 创建新分支；"delete" 删除分支。
            name: 分支名，create/delete 时必填；例如 "feature/login"。

        Returns:
            分支清单或操作结果；失败时返回 git 的原始错误信息。
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
        切换分支或恢复文件（可能丢弃未提交改动，需要用户审批）

        适用：切换到已有分支、创建并切换到新分支（create=true）、或把某个文件恢复到
        指定提交版本。非 create 的操作会触发高风险管理审批；执行前应确认工作区没有
        需要保留的未提交改动，否则先用 git_stash 暂存。

        注意：branch_or_path 是文件路径时，git checkout 会丢弃该文件的未提交修改；
        是分支名时会切换 HEAD。create=true 相当于 git checkout -b，不具破坏性、无需审批。

        Args:
            branch_or_path: 分支名或文件路径；不能为空。
            create: 是否创建并切换到新分支（git checkout -b），默认 False。

        Returns:
            切换/恢复结果；用户拒绝或失败时返回原因。
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

        适用：把功能分支合入主分支或反向同步。执行前需确认工作区状态，必要时先提交或
        stash。no_ff=true 时强制生成合并提交（--no-ff），保留分支历史。

        发生冲突时 git 会留下冲突标记，需解决后手动提交；本工具不会自动解决冲突。
        合并前建议先用 git_status 确认工作区干净，有未提交改动时用 git_stash 暂存；
        合并完成后建议运行项目测试（run_checks 或 run_tests）验证结果未被破坏。

        Args:
            branch: 要合并的分支名，不能为空。
            no_ff: 是否强制创建合并提交（--no-ff），默认 False（可快进时直接快进）。

        Returns:
            合并结果；冲突时返回冲突信息与后续处理建议。
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
        把当前分支变基到指定分支之上（需要用户审批）

        适用：让功能分支基于最新主干重新应用提交，保持线性历史。变基会改写提交历史，
        已推送的分支慎用。发生冲突时先解决冲突，再用 continue_rebase=true 继续，
        或 abort=true 中止并回到变基前状态；abort/continue 不需要再次审批。
        变基前建议记录当前提交号或创建备份分支，以便出错时回退；冲突解决后需要先
        git add 标记已解决，再用 continue_rebase=true 继续，本工具不会自动暂存冲突文件。

        Args:
            branch: 目标分支名；abort/continue 时忽略。
            abort: 是否中止当前变基（git rebase --abort），默认 False。
            continue_rebase: 解决冲突后是否继续变基（git rebase --continue），默认 False。

        Returns:
            变基结果；冲突时返回冲突信息与继续/中止建议。
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

        适用：切换分支或对比改动前临时保存工作区，避免未提交改动丢失。save 会包含
        未跟踪文件（git stash push -u）；pop 应用并删除记录，apply 应用但保留记录；
        drop 会删除记录且无法恢复，需要用户审批。show 查看某条记录的具体 diff。

        index 指定操作第几条 stash（0 为最近一条）。恢复时发生冲突会保留记录并提示手动解决；
        工作区此时仍保留冲突标记，解决后需自行清理 stash。

        Args:
            action: save / list / show / pop / apply / drop，默认 list。
            message: 可选，save 时的说明文字。
            index: 可选，操作第几条 stash，默认 0（最近一条）。

        Returns:
            操作结果或 stash 内容；失败时返回原因。
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
