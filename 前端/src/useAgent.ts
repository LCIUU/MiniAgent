// ===== Agent 交互核心逻辑 =====
// 发送消息、消费 SSE 事件流、审批、停止。

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import { openEventStream, type StreamHandle } from "./api/sse";
import { wsControl } from "./api/ws";
import { getState, setState, subscribe } from "./store";
import type { AppState } from "./store";
import type { AgentEvent, ApprovalRequest, AttachmentMeta, StreamMessage, TimelineItem, ToolCall } from "./types";

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

/** DB 消息 → 前端 StreamMessage：持久化的思考时间轴补齐前端运行时字段（id / start）。 */
function mapHistoryMsgs(msgs: Awaited<ReturnType<typeof api.listMessages>>, sessionId: string): StreamMessage[] {
  return msgs.map((m) => ({
    id: m.id,
    role: m.role,
    content: m.content,
    attachments: m.attachments?.length ? m.attachments : undefined,
    tool_calls: m.tool_calls ?? [],
    status: "done" as const,
    session_id: sessionId,
    pendingApproval: null,
    duration_ms: m.duration_ms ?? 0,
    timeline: (m.timeline ?? []).length
      ? m.timeline!.map((it, i) => ({
          ...it,
          id: `${m.id}-tl-${i}`,
          start: it.start ?? 0,
        }))
      : undefined,
  }));
}

/** 追加思考增量：连续思考合并进最后一个思考条目；阶段切换（forceNew）时开启新段。
 *  durationMs 由后端随延迟下发的整段思考带回，直接定稿，避免按收到时间算成 0ms。 */
function withThinking(
  timeline: TimelineItem[],
  delta: string,
  forceNew: boolean,
  durationMs?: number,
): TimelineItem[] {
  const tl = [...timeline];
  const last = tl[tl.length - 1];
  if (last && last.kind === "thinking" && !forceNew) {
    tl[tl.length - 1] = {
      ...last,
      text: last.text + delta,
      duration_ms: durationMs ?? last.duration_ms,
    };
  } else {
    tl.push({ kind: "thinking", id: uid(), start: Date.now(), text: delta, duration_ms: durationMs });
  }
  return tl;
}

/** 追加阶段正文增量：连续文本合并；阶段切换（forceNew）时新开条目。 */
function withText(timeline: TimelineItem[], delta: string, forceNew: boolean): TimelineItem[] {
  const tl = [...timeline];
  const last = tl[tl.length - 1];
  if (last && last.kind === "text" && !forceNew) {
    tl[tl.length - 1] = { ...last, text: last.text + delta };
  } else {
    tl.push({ kind: "text", id: uid(), start: Date.now(), text: delta });
  }
  return tl;
}

/** 思考段封口：给最后一个未计时的思考条目补时长（被 token/tool_start/plan 终止时调用）。
 *  后端 thinking_end 带真实时长时优先使用，避免把后续工具参数生成时间算进思考耗时。 */
function closeThinking(tl: TimelineItem[], durationMs?: number): TimelineItem[] {
  const last = tl[tl.length - 1];
  if (last && last.kind === "thinking" && last.duration_ms == null) {
    const next = [...tl];
    next[next.length - 1] = { ...last, duration_ms: durationMs ?? Date.now() - last.start };
    return next;
  }
  return tl;
}

/** 更新执行计划条目：已有计划条目则覆盖为最新步骤，否则追加。 */
function withPlan(timeline: TimelineItem[], steps: string[], todos?: { content: string; status: string }[]): TimelineItem[] {
  const tl = [...timeline];
  for (let i = tl.length - 1; i >= 0; i--) {
    const item = tl[i];
    if (item.kind === "plan") {
      tl[i] = { ...item, steps, todos };
      return tl;
    }
  }
  tl.push({ kind: "plan", id: uid(), start: Date.now(), steps, todos });
  return tl;
}

/** 追加工具调用条目。 */
function withToolStart(
  timeline: TimelineItem[],
  ev: Extract<AgentEvent, { type: "tool_start" }>,
): TimelineItem[] {
  return [
    ...timeline,
    {
      kind: "tool",
      id: uid(),
      call_id: ev.call_id,
      name: ev.tool,
      args: ev.args ?? {},
      status: "running",
      start: Date.now(),
      started_at: ev.started_at,
    },
  ];
}

/** 回填工具结果：按 call_id 匹配最近一个运行中的工具条目。 */
function withToolEnd(
  timeline: TimelineItem[],
  ev: Extract<AgentEvent, { type: "tool_end" }>,
): TimelineItem[] {
  for (let i = timeline.length - 1; i >= 0; i--) {
    const item = timeline[i];
    if (item.kind === "tool" && item.call_id === ev.call_id && item.status === "running") {
      const tl = [...timeline];
      tl[i] = {
        ...item,
        result: ev.result ?? "",
        status: ev.status ?? "success",
        duration_ms: ev.duration_ms ?? 0,
      };
      return tl;
    }
  }
  return timeline;
}

/** 上下文压缩开始：新开一条压缩条目，并把遗留的运行中条目标记为完成。 */
function withCompressStart(
  timeline: TimelineItem[],
  ev: Extract<AgentEvent, { type: "summary_start" }>,
): TimelineItem[] {
  const tl = timeline.map((item) =>
    item.kind === "compress" && item.status === "running"
      ? { ...item, status: "done" as const, duration_ms: item.duration_ms ?? Date.now() - item.start }
      : item,
  );
  tl.push({
    kind: "compress",
    id: uid(),
    start: Date.now(),
    status: "running",
    before_tokens: ev.before_tokens,
  });
  return tl;
}

/** 上下文压缩结束：回填耗时与压缩后的 token 估算。 */
function withCompressDone(
  timeline: TimelineItem[],
  ev: Extract<AgentEvent, { type: "summary_done" }>,
): TimelineItem[] {
  for (let i = timeline.length - 1; i >= 0; i--) {
    const item = timeline[i];
    if (item.kind === "compress" && item.status === "running") {
      const tl = [...timeline];
      tl[i] = {
        ...item,
        status: "done",
        failed: !ev.ok,
        duration_ms: ev.duration_ms ?? Date.now() - item.start,
        after_tokens: ev.after_tokens,
        summary: ev.summary,
      };
      return tl;
    }
  }
  return timeline;
}

export function useStore(): AppState;
export function useStore<T>(selector: (state: AppState) => T): T;
export function useStore<T>(selector?: (state: AppState) => T) {
  const [, setTick] = useState(0);
  const selectorRef = useRef(selector);
  selectorRef.current = selector;
  const prevRef = useRef(selector ? selector(getState()) : undefined);

  useEffect(() => {
    if (!selector) {
      // 无 selector：全量订阅（兼容旧用法）
      return subscribe(() => setTick((t) => t + 1));
    }
    // 有 selector：仅当选中字段变化时才触发重渲染
    return subscribe(() => {
      const next = selectorRef.current!(getState());
      if (next !== prevRef.current) {
        prevRef.current = next;
        setTick((t) => t + 1);
      }
    });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return selector ? selector(getState()) : getState();
}

// 这些可变引用必须跨 useAgent() 实例共享：App / ChatView / WorkspaceModal 各自调用
// useAgent()，若随实例创建，切换会话/工作区时无法中断另一实例持有的 SSE 连接，
// "加载更早消息"的翻页 offset 也会各自为政。全局只有一个活动运行，用模块级单例。
const streamRef: { current: StreamHandle | null } = { current: null };
const currentMsgRef: { current: StreamMessage | null } = { current: null };
// 阶段分隔标志（思考与阶段正文各自独立）：plan / tool_* / token 事件之后
// 下一段思考、下一段正文均强制新开条目（与后端 _timeline_from_history 规则一致）
const forceNewThinkingRef = { current: false };
const forceNewTextRef = { current: false };
// 流式增量合帧缓冲：SSE 事件突发到达时（一帧多个 token），合并到每帧一次渲染，
// 避免 Markdown 高频全量重解析导致输出成块上屏
const pendingOpsRef: {
  current: { kind: "thinking" | "token"; delta: string; duration_ms?: number }[];
} = { current: [] };
const flushRafRef: { current: number | null } = { current: null };
// 会话加载序号：快速连切会话时丢弃过期响应（防旧数据覆盖新会话）
const loadSeqRef = { current: 0 };
// 已加载的历史消息条数（相对最新消息的 offset，用于向上翻页）
const loadedHistoryRef = { current: 0 };
// 停止后的兜底定时器：等待后端 run_done，超时才强制断开
const stopTimerRef: { current: number | null } = { current: null };
// 已收到事件的最后一个 seq：SSE 断线重连时只回放缺口，避免内容重复
const lastSeqRef = { current: 0 };

export function useAgent() {
  const upsertCurrent = useCallback((patch: Partial<StreamMessage>) => {
    const cur = currentMsgRef.current;
    if (!cur) return;
    Object.assign(cur, patch);
    setState({
      currentStream: { ...cur },
      messages: getState().messages.map((m) => (m.id === cur.id ? { ...cur } : m)),
    });
  }, []);

  // 把缓冲的思考/正文增量按到达顺序一次性合并进当前消息
  const flushPending = useCallback(() => {
    if (flushRafRef.current != null) {
      cancelAnimationFrame(flushRafRef.current);
      flushRafRef.current = null;
    }
    const ops = pendingOpsRef.current;
    if (!ops.length) return;
    pendingOpsRef.current = [];
    const cur = currentMsgRef.current;
    if (!cur) return;
    let tl = cur.timeline ?? [];
    let reasoning = cur.reasoning ?? "";
    let content = cur.content;
    for (const op of ops) {
      if (op.kind === "thinking") {
        tl = withThinking(tl, op.delta, forceNewThinkingRef.current, op.duration_ms);
        forceNewThinkingRef.current = false;
        reasoning += op.delta;
      } else {
        tl = closeThinking(tl);
        tl = withText(tl, op.delta, forceNewTextRef.current);
        forceNewTextRef.current = false;
        content += op.delta;
      }
    }
    upsertCurrent({ reasoning, content, timeline: tl });
  }, [upsertCurrent]);

  // 流式增量只排程不立即渲染：每帧最多 flush 一次
  const scheduleFlush = useCallback(() => {
    if (flushRafRef.current != null) return;
    flushRafRef.current = requestAnimationFrame(flushPending);
  }, [flushPending]);

  const clearStopTimer = useCallback(() => {
    if (stopTimerRef.current != null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
  }, []);

  // 运行收尾：把仍在运行的条目落定为终态，避免刷新前工具卡片/压缩行一直转圈
  const settleStreamingItems = useCallback(() => {
    const cur = currentMsgRef.current;
    if (!cur) return;
    const timeline = (cur.timeline ?? []).map((it) => {
      if (it.kind === "tool" && it.status === "running") return { ...it, status: "cancelled" };
      if (it.kind === "compress" && it.status === "running") {
        return { ...it, status: "done" as const, failed: true };
      }
      // 运行结束时尚未收到 thinking_end 的思考补一个兜底耗时，避免"已完成"却无时间
      if (it.kind === "thinking" && it.duration_ms == null) {
        return { ...it, duration_ms: Date.now() - it.start };
      }
      return it;
    });
    const tool_calls = (cur.tool_calls ?? []).map((tc) =>
      tc.status === "running" ? { ...tc, status: "cancelled" } : tc,
    );
    upsertCurrent({ timeline, tool_calls });
  }, [upsertCurrent]);

  // 结束本次运行计时（run_done / error / 手动停止时调用）
  const finalizeRun = useCallback((durationMs?: number) => {
    const st = getState();
    setState({
      runStartedAt: null,
      runDurationMs:
        durationMs ?? (st.runStartedAt ? Date.now() - st.runStartedAt : st.runDurationMs),
    });
  }, []);

  const handleEvent = useCallback(
    (ev: AgentEvent) => {
      // 记录事件序号（后端每条事件都带 seq），断线重连时作为 since 增量回放；
      // 旧连接与新连接短暂并存时可能重发同一事件，按 seq 幂等丢弃，避免正文重复追加
      const seq = (ev as AgentEvent & { seq?: number }).seq;
      if (typeof seq === "number") {
        if (seq <= lastSeqRef.current) return;
        lastSeqRef.current = seq;
      }
      const cur = currentMsgRef.current;
      switch (ev.type) {
        case "thinking": {
          if (!cur) return;
          pendingOpsRef.current.push({
            kind: "thinking",
            delta: ev.delta,
            duration_ms: ev.duration_ms,
          });
          scheduleFlush();
          break;
        }
        case "thinking_end": {
          if (!cur) return;
          flushPending();
          const curNow = currentMsgRef.current;
          upsertCurrent({
            timeline: closeThinking(curNow?.timeline ?? [], ev.duration_ms),
          });
          break;
        }
        case "token": {
          if (!cur) return;
          forceNewThinkingRef.current = true;
          pendingOpsRef.current.push({ kind: "token", delta: ev.delta });
          scheduleFlush();
          break;
        }
        case "plan": {
          flushPending(); // 先落盘缓冲的流式增量，保证 timeline 顺序
          setState({ connecting: false, running: true });
          forceNewThinkingRef.current = true;
          forceNewTextRef.current = true;
          const curNow = currentMsgRef.current;
          if (curNow) {
            const closedTl = closeThinking(curNow.timeline ?? []);
            upsertCurrent({
              plan: ev.steps,
              timeline:
                ev.steps.length > 0 ? withPlan(closedTl, ev.steps, ev.todos) : closedTl,
            });
          }
          break;
        }
        case "run_start": {
          setState({ runStartedAt: (ev.started_at ?? Date.now()) * (ev.started_at ? 1000 : 1) });
          break;
        }
        case "progress": {
          if (!cur) return;
          upsertCurrent({ progress: ev.detail });
          break;
        }
        case "tool_start": {
          if (!cur) return;
          flushPending(); // 先落盘缓冲的流式增量，保证 timeline 顺序
          forceNewThinkingRef.current = true;
          forceNewTextRef.current = true;
          const curNow = currentMsgRef.current;
          const closedTl = closeThinking(curNow?.timeline ?? []);
          const toolCall: ToolCall = {
            name: ev.tool,
            args: ev.args ?? {},
            result: "",
            status: "running",
            started_at: ev.started_at,
            call_id: ev.call_id,
          };
          const updated = [...(cur.tool_calls ?? []), toolCall];
          upsertCurrent({
            tool_calls: updated,
            timeline: withToolStart(closedTl, ev),
            progress: null,
          });
          break;
        }
        case "tool_end": {
          if (!cur) return;
          flushPending(); // 先落盘缓冲的流式增量，保证 timeline 顺序
          forceNewThinkingRef.current = true;
          forceNewTextRef.current = true;
          const curNow = currentMsgRef.current;
          const updated = (curNow?.tool_calls ?? []).map((tc, i) => {
            const isLast = i === (curNow?.tool_calls?.length ?? 0) - 1;
            const match = ev.call_id ? tc.call_id === ev.call_id : isLast && tc.status === "running";
            if (match && (tc.status === "running" || ev.call_id)) {
              return {
                ...tc,
                result: ev.result ?? "",
                status: ev.status ?? "success",
                duration_ms: ev.duration_ms ?? 0,
                started_at: ev.started_at ?? tc.started_at,
              };
            }
            return tc;
          });
          upsertCurrent({
            tool_calls: updated,
            timeline: withToolEnd(curNow?.timeline ?? [], ev),
          });
          break;
        }
        case "tool_output": {
          if (!cur) return;
          const curNow = currentMsgRef.current;
          // stream 是通道名（stdout/stderr），delta 是本批次增量，按顺序追加
          const updated = (curNow?.tool_calls ?? []).map((tc) => {
            if (tc.call_id === ev.call_id) {
              return { ...tc, result: (tc.result ?? "") + ev.delta };
            }
            return tc;
          });
          const timeline = (curNow?.timeline ?? []).map((item) => {
            if (item.kind === "tool" && item.call_id === ev.call_id) {
              return { ...item, result: (item.result ?? "") + ev.delta };
            }
            return item;
          });
          upsertCurrent({ tool_calls: updated, timeline });
          break;
        }
        case "diff": {
          if (!cur) return;
          const curNow = currentMsgRef.current;
          const updated = (curNow?.tool_calls ?? []).map((tc, i) => {
            const isLast = i === (curNow?.tool_calls?.length ?? 0) - 1;
            if ((ev.call_id && tc.call_id === ev.call_id) || (!ev.call_id && isLast)) {
              return { ...tc, diff: ev.diff };
            }
            return tc;
          });
          // 预先计算最后一个 tool 的下标，避免在 map 内重复扫描
          const tl = curNow?.timeline ?? [];
          let lastToolIdx = -1;
          for (let i = tl.length - 1; i >= 0; i--) {
            if (tl[i].kind === "tool") { lastToolIdx = i; break; }
          }
          const timeline = tl.map((item, i) => {
            if (item.kind !== "tool") return item;
            if ((ev.call_id && item.call_id === ev.call_id) || (!ev.call_id && i === lastToolIdx)) {
              return { ...item, diff: ev.diff };
            }
            return item;
          });
          upsertCurrent({ tool_calls: updated, timeline });
          break;
        }
        case "summary_start": {
          if (!cur) return;
          flushPending(); // 先落盘缓冲的流式增量，保证 timeline 顺序
          forceNewThinkingRef.current = true;
          forceNewTextRef.current = true;
          const curNow = currentMsgRef.current;
          const closedTl = closeThinking(curNow?.timeline ?? []);
          upsertCurrent({ timeline: withCompressStart(closedTl, ev) });
          break;
        }
        case "summary_done": {
          if (!cur) return;
          flushPending();
          const curNow = currentMsgRef.current;
          upsertCurrent({ timeline: withCompressDone(curNow?.timeline ?? [], ev) });
          break;
        }
        case "usage":
          break;
        case "approval": {
          const approval: ApprovalRequest = {
            approval_id: ev.approval_id,
            tool: ev.tool,
            reason: ev.reason,
            command: ev.command,
            cwd: ev.cwd,
          };
          upsertCurrent({ pendingApproval: approval });
          break;
        }
        case "approval_done": {
          // 已批准/已拒绝/超时（含刷新后重放历史）：收起对应审批卡片
          if (cur?.pendingApproval?.approval_id === ev.approval_id) {
            upsertCurrent({ pendingApproval: null });
          }
          break;
        }
        case "question": {
          if (!cur) return;
          flushPending(); // 先落盘缓冲的流式增量，保证卡片出现在正确位置
          upsertCurrent({
            pendingQuestion: {
              question_id: ev.question_id,
              tool: ev.tool,
              question: ev.question,
              options: ev.options ?? [],
              allow_multiple: Boolean(ev.allow_multiple),
            },
          });
          break;
        }
        case "question_done": {
          // 已回答/已取消（含刷新后重放历史）：收起对应卡片
          if (cur?.pendingQuestion?.question_id === ev.question_id) {
            upsertCurrent({ pendingQuestion: null });
          }
          break;
        }
        case "summary":
        case "mcp_loaded":
          break;
        case "status":
          // 重连时回放被截断：提示用户尽早查看结果，事件流本身继续
          if (ev.status === "replay_truncated" && ev.detail) {
            setState({ error: ev.detail });
          }
          break;
        case "message_done": {
          // 后端落库后返回真实 message_id，回写到当前消息（替换前端临时 uid）
          if (cur && ev.message_id && cur.id !== ev.message_id) {
            const oldId = cur.id;
            cur.id = ev.message_id;
            setState({
              currentStream: { ...cur },
              messages: getState().messages.map((m) => (m.id === oldId ? { ...cur } : m)),
            });
          }
          break;
        }
        case "run_done": {
          clearStopTimer();
          flushPending(); // 收尾前先渲染尾部增量
          settleStreamingItems();
          if (currentMsgRef.current) upsertCurrent({ status: "done", pendingQuestion: null, pendingApproval: null, progress: null });
          finalizeRun(ev.duration_ms);
          // 不依赖 SSE 连接关闭事件：run_done 到达即解除运行态，
          // 否则连接收尾稍慢时输入框会一直保持禁用
          setState({ running: false, connecting: false, runId: null });
          break;
        }
        case "error": {
          clearStopTimer();
          flushPending();
          if (ev.fatal === false) {
            // 非致命错误（如 MCP 加载失败）：只提示，不中断事件流
            setState({ error: ev.message });
            break;
          }
          settleStreamingItems();
          upsertCurrent({ status: "error", pendingQuestion: null, pendingApproval: null, progress: null });
          finalizeRun();
          setState({ error: ev.message, running: false, connecting: false, runId: null });
          // 断开 SSE/WS，避免后端继续推送事件
          streamRef.current?.abort();
          streamRef.current = null;
          wsControl.disconnect();
          break;
        }
      }
    },
    [upsertCurrent, finalizeRun, flushPending, scheduleFlush, clearStopTimer, settleStreamingItems],
  );

  /** 断开当前 SSE/WS 并清空运行期引用：切换、新建会话前调用，防止旧 run 事件写进新会话。 */
  const detachStream = useCallback(() => {
    clearStopTimer();
    loadedHistoryRef.current = 0;
    streamRef.current?.abort();
    streamRef.current = null;
    wsControl.disconnect();
    currentMsgRef.current = null;
    pendingOpsRef.current = [];
    if (flushRafRef.current != null) {
      cancelAnimationFrame(flushRafRef.current);
      flushRafRef.current = null;
    }
    forceNewThinkingRef.current = false;
    forceNewTextRef.current = false;
  }, [clearStopTimer]);

  /** 连接/重连某个 run 的 SSE 事件流（按 seq 增量回放，断线重连可补齐缺口且不重复）。 */
  const attachStream = useCallback(
    (runId: string) => {
      lastSeqRef.current = 0; // 新接管：从头全量回放重建过程
      let attempts = 0;
      const connect = () => {
        const handle: StreamHandle = openEventStream(
          runId,
          handleEvent,
          () => {
            clearStopTimer();
            flushPending(); // 收尾前先渲染尾部增量
            settleStreamingItems(); // 连接提前关闭时也不让卡片停在"运行中"
            upsertCurrent({ status: "done", pendingQuestion: null, pendingApproval: null });
            finalizeRun();
            setState({ running: false, connecting: false, runId: null });
            wsControl.disconnect();
            refreshSessions();
          },
          {
            since: lastSeqRef.current,
            onDrop: () => {
              // 主动 abort / 已切换会话：不再重连
              if (streamRef.current !== handle) return;
              if (attempts >= 5) {
                // 重连无望：收尾为错误态，避免界面永远显示"运行中"
                flushPending();
                settleStreamingItems();
                upsertCurrent({ status: "error", pendingQuestion: null, pendingApproval: null, progress: null });
                finalizeRun();
                setState({
                  error: "事件流连接中断，自动重连失败（可刷新页面查看最终结果）",
                  running: false,
                  connecting: false,
                  runId: null,
                });
                streamRef.current = null;
                wsControl.disconnect();
                refreshSessions();
                return;
              }
              attempts += 1;
              setState({ connecting: true });
              window.setTimeout(() => {
                if (streamRef.current !== handle) return;
                connect();
              }, 1500 * attempts);
            },
          },
        );
        streamRef.current = handle;
      };
      connect();
    },
    [handleEvent, upsertCurrent, finalizeRun, flushPending, clearStopTimer, settleStreamingItems],
  );

  const sendMessage = useCallback(
    async (text: string, attachments: AttachmentMeta[] = [], reasoningEffort: string | null = null) => {
      // 防重复发送：如果正在运行中，直接忽略
      if (getState().running) return;

      const st = getState();
      const sessionId = st.activeSessionId ?? undefined;
      // 当前工作区：优先取活动会话所属工作区，做到“每个工作区的会话各自独立”
      let workspace = st.workspace;
      if (sessionId) {
        const ses = st.sessions.find((s) => s.id === sessionId);
        if (ses && ses.workspace) workspace = ses.workspace;
      }

      // 本地立即显示用户消息：attachments 独立字段（图片渲染缩略图），content 保持纯文本
      const userMsg: StreamMessage = {
        id: uid(),
        role: "user",
        content: text,
        attachments: attachments.length ? attachments : undefined,
        tool_calls: [],
        status: "done",
        session_id: sessionId ?? "",
      };
      // 新建 assistant 占位
      const assistantMsg: StreamMessage = {
        id: uid(),
        role: "assistant",
        content: "",
        tool_calls: [],
        status: "streaming",
        session_id: sessionId ?? "",
        pendingApproval: null,
        reasoning: "",
      };
      currentMsgRef.current = assistantMsg;
      setState({
        messages: [...st.messages, userMsg, assistantMsg],
        running: true,
        connecting: true,
        error: null,
        runStartedAt: Date.now(),
        runDurationMs: null,
      });

      try {
        const run = await api.startChat({
          message: text,
          session_id: sessionId,
          workspace,
          attachments: attachments.length ? attachments : undefined,
          reasoning_effort: reasoningEffort,
        });
        setState({ runId: run.run_id, activeSessionId: run.session_id });
        currentMsgRef.current!.session_id = run.session_id;
        wsControl.connect(run.run_id);
        // 新创建的会话按首条消息自动命名
        const cur = getState().sessions.find((s) => s.id === run.session_id);
        if (cur && (!cur.title || cur.title === "新会话")) {
          void api.renameSession(run.session_id, text.slice(0, 30)).then(() => refreshSessions());
        }

        attachStream(run.run_id);
      } catch (err) {
        upsertCurrent({ status: "error" });
        finalizeRun();
        setState({
          error: (err as Error).message,
          running: false,
          connecting: false,
        });
        currentMsgRef.current = null;
      }
    },
    [handleEvent, upsertCurrent, finalizeRun, attachStream],
  );

  const stopRun = useCallback(() => {
    const st = getState();
    if (!st.runId) return;
    wsControl.sendStop();
    api.stopRun(st.runId).catch(() => undefined);
    // 不立即断开 SSE：等待后端优雅停止并回传 run_done，保留停止前的完整输出。
    // 5 秒仍未结束则兜底强制收尾，避免界面卡在"运行中"。
    clearStopTimer();
    stopTimerRef.current = window.setTimeout(() => {
      stopTimerRef.current = null;
      streamRef.current?.abort();
      streamRef.current = null;
      flushPending();
      settleStreamingItems();
      if (currentMsgRef.current) {
        upsertCurrent({ status: "done", pendingQuestion: null, pendingApproval: null });
      }
      finalizeRun();
      setState({ running: false, connecting: false, runId: null });
      wsControl.disconnect();
      refreshSessions();
    }, 5000);
  }, [upsertCurrent, finalizeRun, flushPending, clearStopTimer, settleStreamingItems]);

  const decideApproval = useCallback(
    (approvalId: string, approve: boolean) => {
      const st = getState();
      if (st.runId) {
        wsControl.sendApproval(approvalId, approve);
        api.decideApproval(st.runId, approvalId, approve).catch(() => undefined);
      }
      upsertCurrent({ pendingApproval: null });
    },
    [upsertCurrent],
  );

  const answerQuestion = useCallback(
    (questionId: string, answers: string[]) => {
      const st = getState();
      if (st.runId) {
        wsControl.sendAnswer(questionId, answers);
        api.answerQuestion(st.runId, questionId, answers).catch(() => undefined);
      }
      upsertCurrent({ pendingQuestion: null });
    },
    [upsertCurrent],
  );

  const loadSession = useCallback(
    async (sessionId: string) => {
      const seq = ++loadSeqRef.current;
      // 切换会话：断开旧流、清空运行期缓冲，防止旧 run 的流式事件写进新会话
      detachStream();
      try {
        const PAGE = 100;
        const msgs = await api.listMessages(sessionId, PAGE + 1, 0);
        if (seq !== loadSeqRef.current) return; // 期间又切走，丢弃过期响应
        const hasMore = msgs.length > PAGE;
        const visibleMsgs = hasMore ? msgs.slice(-PAGE) : msgs; // 正序：取最近 PAGE 条
        // 切换会话时，把全局工作区切到该会话所属工作区，保证 agent 在正确目录运行
        const ses = getState().sessions.find((s) => s.id === sessionId);
        const workspace = ses?.workspace || getState().workspace;
        // 先落地历史，并把运行态全部归零：无活动运行的会话绝不残留"暂停"按钮/计时
        setState({
          activeSessionId: sessionId,
          messages: mapHistoryMsgs(visibleMsgs, sessionId),
          error: null,
          workspace,
          running: false,
          connecting: false,
          runId: null,
          currentStream: null,
          runStartedAt: null,
          runDurationMs: null,
          hasMoreEarlierMessages: hasMore,
        });
        loadedHistoryRef.current = visibleMsgs.length;
        // 该会话存在进行中的运行 → 重建流式占位并接管 SSE（后端全量回放补齐过程）
        const active = await api.getActiveRun(sessionId).catch(() => ({ run_id: null as string | null }));
        if (seq !== loadSeqRef.current || !active.run_id) return;
        const assistantMsg: StreamMessage = {
          id: uid(),
          role: "assistant",
          content: "",
          tool_calls: [],
          status: "streaming",
          session_id: sessionId,
          pendingApproval: null,
          reasoning: "",
        };
        currentMsgRef.current = assistantMsg;
        setState({
          messages: [...getState().messages, assistantMsg],
          running: true,
          connecting: true,
          runId: active.run_id,
          runStartedAt:
            "started_at" in active && active.started_at ? active.started_at * 1000 : Date.now(),
        });
        wsControl.connect(active.run_id);
        attachStream(active.run_id);
      } catch (err) {
        setState({ error: (err as Error).message });
      }
    },
    [detachStream, attachStream],
  );

  const newSession = useCallback(async (workspace?: string) => {
    ++loadSeqRef.current; // 使进行中的 loadSession 响应失效
    detachStream();
    const st = getState();
    const target = (workspace ?? st.workspace).trim();
    if (!target) {
      setState({
        activeSessionId: null,
        messages: [],
        runId: null,
        error: null,
        running: false,
        connecting: false,
        currentStream: null,
        workspace: "",
        runStartedAt: null,
        runDurationMs: null,
      });
      return;
    }
    try {
      // 立即创建会话（挂到指定工作区），让侧栏马上出现新对话
      const s = await api.createSession(target);
      if (getState().activeSessionId !== st.activeSessionId && getState().activeSessionId !== null) {
        // 创建期间用户已切换会话——不抢占，但仍刷新侧栏
        refreshSessions();
        return;
      }
      setState({
        activeSessionId: s.id,
        messages: [],
        runId: null,
        error: null,
        running: false,
        connecting: false,
        currentStream: null,
        workspace: s.workspace,
        runStartedAt: null,
        runDurationMs: null,
      });
      refreshSessions();
    } catch (err) {
      setState({ error: (err as Error).message });
    }
  }, [detachStream]);

  // 选择工作区：切到该工作区，并打开它最近一次会话（无会话则刷新空白新对话）
  const selectWorkspace = useCallback(
    async (wsKey: string) => {
      const st = getState();
      const wsSessions = st.sessions.filter((s) => (s.workspace || "") === wsKey);
      // 先递增序号，防止旧 loadSession 完成后覆盖新工作区
      ++loadSeqRef.current;
      setState({ workspace: wsKey });
      if (wsSessions.length > 0) {
        await loadSession(wsSessions[0].id);
      } else {
        ++loadSeqRef.current;
        detachStream();
        setState({
          activeSessionId: null,
          messages: [],
          runId: null,
          error: null,
          running: false,
          connecting: false,
          currentStream: null,
          runStartedAt: null,
          runDurationMs: null,
        });
      }
    },
    [loadSession, detachStream],
  );

  // 删除工作区：删除该工作区下全部会话；若是当前工作区，一并清空选中状态
  const deleteWorkspace = useCallback(async (wsKey: string) => {
    const st = getState();
    const wsSessions = st.sessions.filter((s) => (s.workspace || "") === wsKey);
    for (const s of wsSessions) {
      await api.deleteSession(s.id).catch(() => undefined);
    }
    if (st.workspace === wsKey) {
      ++loadSeqRef.current;
      detachStream();
      setState({
        workspace: "",
        activeSessionId: null,
        messages: [],
        error: null,
        running: false,
        connecting: false,
        runId: null,
        currentStream: null,
        runStartedAt: null,
        runDurationMs: null,
      });
    }
    refreshSessions();
  }, [detachStream]);

  // 挂载后：对当前活动会话执行一次完整接管（刷新页面后，若该会话存在
  // 进行中的运行则自动重连 SSE 补齐过程）；initApp 异步就绪后触发。
  useEffect(() => {
    let done = false;
    const tryResume = () => {
      if (done) return;
      const sid = getState().activeSessionId;
      if (!sid) return;
      done = true;
      void loadSession(sid);
    };
    tryResume();
    const unsub = subscribe(tryResume);
    return unsub;
  }, [loadSession]);

  /** 加载更早的历史消息（按 offset 从最新向前翻页，避免重复/漏读）。 */
  const loadEarlierMessages = useCallback(async () => {
    const { activeSessionId, messages, hasMoreEarlierMessages } = getState();
    if (!activeSessionId || !hasMoreEarlierMessages) return;
    const PAGE = 100;
    try {
      // 后端按 created_at 倒序返回：offset 跳过已加载的最近消息，取下一页更早的
      const older = await api.listMessages(activeSessionId, PAGE, loadedHistoryRef.current);
      if (!older.length) {
        setState({ hasMoreEarlierMessages: false });
        return;
      }
      const existing = new Set(messages.map((m) => m.id));
      const fresh = older.filter((m) => !existing.has(m.id));
      loadedHistoryRef.current += older.length;
      setState({
        messages: [...mapHistoryMsgs(fresh, activeSessionId), ...messages],
        hasMoreEarlierMessages: older.length === PAGE,
      });
    } catch {
      // 加载失败静默忽略，不影响当前视图
    }
  }, []);

  return { sendMessage, stopRun, decideApproval, answerQuestion, loadSession, newSession, selectWorkspace, deleteWorkspace, loadEarlierMessages };
}

let _sessionTimer: number | null = null;

export function refreshSessions() {
  if (_sessionTimer) window.clearTimeout(_sessionTimer);
  _sessionTimer = window.setTimeout(async () => {
    try {
      const sessions = await api.listSessions();
      setState({ sessions });
    } catch {
      /* ignore */
    }
  }, 400);
}

export function initApp() {
  void (async () => {
    try {
      const [settings, sessions] = await Promise.all([api.getSettings(), api.listSessions()]);
      setState({ settings, sessions });
      const ws = sessions[0]?.workspace || settings.workspace || "";
      setState({ workspace: ws });
      if (sessions.length > 0) {
        // 只定位活动会话：消息加载与活动运行接管由 useAgent 的 resume effect 统一走 loadSession
        const first = sessions[0];
        setState({ activeSessionId: first.id, workspace: first.workspace || ws });
      }
    } catch (err) {
      setState({ error: `初始化失败: ${(err as Error).message}` });
    }
  })();
}
