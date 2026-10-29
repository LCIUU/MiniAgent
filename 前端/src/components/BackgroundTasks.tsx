// 顶栏"后台任务"按钮 + 下拉面板
// 从当前会话消息时间轴中解析 terminal 后台启动（PID/命令/日志）与 process 停止记录

import { useEffect, useMemo, useRef, useState } from "react";
import type { StreamMessage } from "../types";
import { api } from "../api/client";
import { useStore } from "../useAgent";

interface BgTask {
  pid: number | null;
  command: string;
  log: string;
  status: "running" | "stopped" | "failed";
  exitCode?: number;
}

function deriveTasks(messages: StreamMessage[]): BgTask[] {
  const tasks: BgTask[] = [];
  const failRows: BgTask[] = [];
  const stopped = new Set<number>();

  const cmdOf = (args: Record<string, unknown>) =>
    typeof args?.command === "string" ? args.command : "";

  for (const m of messages) {
    for (const it of m.timeline ?? []) {
      if (it.kind !== "tool" || !it.result) continue;
      if (it.name === "terminal") {
        const start = it.result.match(/已在后台启动，PID:\s*(\d+)/);
        if (start) {
          tasks.push({
            pid: Number(start[1]),
            command: cmdOf(it.args),
            log: it.result.match(/日志:\s*(\S+)/)?.[1] ?? "",
            status: "running",
          });
          continue;
        }
        const fail = it.result.match(/后台命令启动后立即退出（退出码\s*(\d+)）/);
        if (fail) {
          failRows.push({
            pid: null,
            command: cmdOf(it.args),
            log: it.result.match(/日志:\s*(\S+)/)?.[1] ?? "",
            status: "failed",
            exitCode: Number(fail[1]),
          });
        }
      } else if (it.name === "process") {
        for (const s of it.result.matchAll(/已停止进程树 PID\s*(\d+)/g)) {
          stopped.add(Number(s[1]));
        }
      }
    }
  }

  for (const t of tasks) {
    if (t.pid != null && stopped.has(t.pid)) t.status = "stopped";
  }
  return [...tasks, ...failRows];
}

export function BackgroundTasks() {
  const { messages } = useStore();
  const [open, setOpen] = useState(false);
  const [stopping, setStopping] = useState<Set<number>>(new Set());
  const [uiStopped, setUiStopped] = useState<Set<number>>(new Set());
  const [deadPids, setDeadPids] = useState<Set<number>>(new Set());
  const [stopError, setStopError] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);
  const tasks = useMemo(() => deriveTasks(messages), [messages]);

  // 解析出运行中的 pid 集合（字符串形式，仅在集合变化时触发存活检测）
  const pidsKey = useMemo(
    () =>
      tasks
        .filter((t) => t.status === "running" && t.pid != null)
        .map((t) => t.pid!)
        .sort((a, b) => a - b)
        .join(","),
    [tasks],
  );

  // 向后端核实 pid 是否真的存活：处理刷新后状态丢失、进程自行崩溃等情况
  useEffect(() => {
    if (!pidsKey) return;
    let cancelled = false;
    api
      .aliveProcesses(pidsKey)
      .then((r) => {
        if (cancelled) return;
        const alive = new Set(r.alive);
        setDeadPids((prev) => {
          const next = new Set(prev);
          for (const pid of pidsKey.split(",").map(Number)) {
            if (!alive.has(pid)) next.add(pid);
            else next.delete(pid);
          }
          return next;
        });
      })
      .catch(() => {
        /* 检测失败时保持原状态 */
      });
    return () => {
      cancelled = true;
    };
  }, [pidsKey]);

  const handleStop = async (pid: number) => {
    setStopping((s) => new Set(s).add(pid));
    setStopError("");
    try {
      await api.stopProcess(pid);
      setUiStopped((s) => new Set(s).add(pid));
    } catch (e) {
      setStopError(e instanceof Error ? e.message : String(e));
    } finally {
      setStopping((s) => {
        const next = new Set(s);
        next.delete(pid);
        return next;
      });
    }
  };

  // 点击面板外部关闭
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  if (tasks.length === 0) return null;

  // 只显示运行中且确实存活的任务：已停止/已退出的不再展示
  const runningTasks = tasks.filter(
    (t) =>
      t.status === "running" &&
      !(t.pid != null && (uiStopped.has(t.pid) || deadPids.has(t.pid))),
  );
  if (runningTasks.length === 0) return null;

  return (
    <div className="bg-tasks" ref={boxRef}>
      <button
        className={`bg-tasks-btn${open ? " open" : ""}`}
        onClick={() => setOpen((o) => !o)}
        aria-label="当前会话的后台任务"
      >
        后台任务
      </button>
      {open && (
        <div className="bg-tasks-panel">
          <div className="bg-tasks-head">当前会话的后台任务</div>
          {runningTasks.map((t, i) => {
            const busy = t.pid != null && stopping.has(t.pid);
            return (
              <div className="bg-task-row" key={i}>
                <span className="dot running" />
                <div className="bg-task-info">
                  <div className="bg-task-cmd" title={t.command}>
                    {t.command || "（未知命令）"}
                  </div>
                  <div className="bg-task-meta">
                    {t.pid != null ? `PID ${t.pid}` : "未获取 PID"} · 运行中
                  </div>
                </div>
                {t.pid != null && (
                  <button
                    className="bg-task-stop"
                    disabled={busy}
                    onClick={() => void handleStop(t.pid!)}
                  >
                    {busy ? "停止中…" : "停止"}
                  </button>
                )}
              </div>
            );
          })}
          {stopError && <div className="bg-task-error">{stopError}</div>}
        </div>
      )}
    </div>
  );
}
