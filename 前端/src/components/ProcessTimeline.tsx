// ===== 执行过程流（DeepSeek 风格：思考/工具/计划/阶段正文按时间顺序内联交错）=====
// - 每个阶段的思考以单行「Think」摘要展示，点击展开全文
// - 工具调用以单行「工具名 · 参数摘要」展示，点击展开参数/结果详情
// - 工具轮之间的说明文本作为正文段落直接渲染
// - 最后一个正文条目即最终回答，由 MessageView 单独用 Markdown 渲染（answerOf 提取）
// 历史消息无 timeline 时，从 reasoning / tool_calls 合成静态流。

import { useEffect, memo, useRef, useState } from "react";
import { Markdown } from "./Markdown";
import type { StreamMessage, TimelineItem } from "../types";
import { splitTimeline } from "../utils/timeline";
import thinkIconUrl from "../assets/Think-icon.svg";
import toolIconUrl from "../assets/Tool-icon.svg";
import terminalIconUrl from "../assets/terminal-icon.svg";

const STATUS_LABEL: Record<string, string> = {
  running: "调用中",
  success: "已完成",
  error: "失败",
  rejected: "已拒绝",
  cancelled: "已取消",
};

function fmtDuration(ms?: number): string {
  if (ms == null) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** 实时耗时文案：1 秒内毫秒位逐帧变化太吵，按 0.5 秒步进；满 1 秒后逐帧平滑滚动。 */
function liveDurationText(start: number, now: number): string {
  const elapsed = now - start;
  return elapsed < 1000
    ? fmtDuration(Math.floor(elapsed / 500) * 500) // 0 ms / 500 ms
    : fmtDuration(elapsed);
}

/** 实时耗时：计时器独立于此组件，按帧直接写 textContent，
 *  不用 state 是为了避免每次 tick 重渲染整行（含展开的 Markdown 正文）。 */
function LiveDuration({ start }: { start: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let raf = 0;
    let last = "";
    const tick = () => {
      const el = ref.current;
      if (el) {
        const text = liveDurationText(start, Date.now());
        if (text !== last) {
          el.textContent = text;
          last = text;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [start]);
  return (
    <span className="pflow-dur" ref={ref}>
      {liveDurationText(start, Date.now())}
    </span>
  );
}

function truncate1(s: string, max: number): string {
  const flat = s.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/** 多行文本（终端输出 / diff）按字符截断，保留换行与缩进。 */
function truncateMultiline(s: string, max: number): string {
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

/** 同轮「思考后正文复述」去重：模型常先想一句结论，紧接着在正文里原样再说一遍。
 *  紧随其后的正文以该思考开头时，隐藏思考行（正文已完整包含，避免文字重复）。 */
function dropEchoedThinking(items: TimelineItem[]): TimelineItem[] {
  return items.filter((it, i) => {
    const next = items[i + 1];
    if (it.kind !== "thinking" || next?.kind !== "text") return true;
    const a = it.text.replace(/\s+/g, "");
    const b = next.text.replace(/\s+/g, "");
    return a.length < 12 || !b.startsWith(a);
  });
}

/** 历史消息合成：无实时 timeline 时由 reasoning + tool_calls 推导静态条目。 */
function synthesize(msg: StreamMessage): TimelineItem[] {
  const items: TimelineItem[] = [];
  if (msg.reasoning) {
    items.push({ kind: "thinking", id: `${msg.id}-th`, start: 0, text: msg.reasoning });
  }
  (msg.tool_calls ?? []).forEach((tc, i) => {
    items.push({
      kind: "tool",
      id: `${msg.id}-tool-${i}`,
      call_id: `hist-${i}`,
      name: tc.name,
      args: tc.args ?? {},
      status: tc.status,
      result: tc.result,
      duration_ms: tc.duration_ms,
      start: 0,
      started_at: tc.started_at,
    });
  });
  return items;
}

/* ---------- 线性 SVG 图标 ---------- */

function ThinkIcon() {
  return <img className="pflow-ic" src={thinkIconUrl} alt="" width={25} height={25} />;
}

function ToolIcon({ name }: { name: string }) {
  const src = name.startsWith("terminal") ? terminalIconUrl : toolIconUrl;
  return <img className="pflow-ic" src={src} alt="" width={25} height={25} />;
}

/** 深色代码卡片顶部圆点（Uiverse by emmanuelh-dev）。 */
function CodeDots() {
  return (
    <span className="code-dots" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}

/** 工具参数 / 结果 / 改动：深色代码卡片。 */
function ToolCodeBlock({
  label,
  code,
  error = false,
}: {
  label: string;
  code: string;
  error?: boolean;
}) {
  return (
    <div className={`pflow-code-card${error ? " err" : ""}`}>
      <div className="code-card-head">
        <CodeDots />
        <span className="code-card-name">{label}</span>
      </div>
      <pre className="pflow-code">{code}</pre>
    </div>
  );
}

function Caret({ open }: { open: boolean }) {
  return (
    <svg
      className={`pflow-caret${open ? " open" : ""}`}
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <polyline
        points="9 18 15 12 9 6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/* ---------- 思考行 ---------- */

function ThinkLine({
  text,
  live,
  start,
  duration,
}: {
  text: string;
  live: boolean;
  start?: number;
  duration?: number;
}) {
  const [open, setOpen] = useState(live); // 思考中默认展开，实时跟随流式输出
  const manualRef = useRef(false); // 用户手动点击后不再自动切换
  useEffect(() => {
    // 思考结束自动收起为摘要（用户未手动干预时），保持历史列表整洁
    if (!live && !manualRef.current) setOpen(false);
  }, [live]);
  return (
    <div className="pflow-line">
      <button
        className="pflow-row"
        type="button"
        onClick={() => {
          manualRef.current = true;
          setOpen(!open);
        }}
      >
        <ThinkIcon />
        <span className="pflow-label">Think</span>
        {/* 展开时正文已完整显示，再放预览会出现同一段文字两边重复 */}
        {!open && <span className="pflow-brief">{truncate1(text, 96)}</span>}
        <span className="pflow-meta">
          {live ? (
            <>
              {start ? <LiveDuration start={start} /> : null}
              <span className="tp-badge running">思考中</span>
            </>
          ) : (
            <>
              {duration != null && <span className="pflow-dur">{fmtDuration(duration)}</span>}
              <span className="tp-badge success">已完成</span>
            </>
          )}
          <Caret open={open} />
        </span>
      </button>
      {open && <div className="pflow-detail"><Markdown content={text} streaming={live} /></div>}
    </div>
  );
}

/* ---------- 上下文压缩行（沿用 Think 样式，可展开查看摘要） ---------- */

function CompressLine({ item }: { item: Extract<TimelineItem, { kind: "compress" }> }) {
  const live = item.status === "running";
  const [open, setOpen] = useState(false);

  return (
    <div className={`pflow-line compress-${item.status}`}>
      <button
        className="pflow-row"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <ThinkIcon />
        <span className="pflow-label">Context Windows</span>
        <span className="pflow-brief">优化上下文</span>
        <span className="pflow-meta">
          {live ? (
            <LiveDuration start={item.start} />
          ) : (
            <>
              {item.duration_ms != null && <span className="pflow-dur">{fmtDuration(item.duration_ms)}</span>}
              <span className={`tp-badge ${item.failed ? "error" : "success"}`}>
                {item.failed ? "已跳过" : "已完成"}
              </span>
            </>
          )}
          <Caret open={open} />
        </span>
      </button>
      {open && (
        <div className="pflow-detail">
          {item.summary ? (
            <Markdown content={item.summary} />
          ) : (
            <div className="pflow-empty">{live ? "正在生成摘要…" : "无摘要内容"}</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------- 工具行 ---------- */

function argBrief(args: Record<string, unknown>): string {
  for (const v of Object.values(args ?? {})) {
    if (typeof v === "string" && v.trim()) return truncate1(v, 64);
    if (typeof v === "number" || typeof v === "boolean") return String(v);
  }
  try {
    const s = JSON.stringify(args ?? {});
    return s && s !== "{}" ? truncate1(s, 64) : "";
  } catch {
    return "";
  }
}

function ToolLine({ item }: { item: Extract<TimelineItem, { kind: "tool" }> }) {
  const [open, setOpen] = useState(false);
  const brief = argBrief(item.args);
  const running = item.status === "running";
  return (
    <div className={`pflow-line tool-${item.status}`}>
      <button className="pflow-row" type="button" onClick={() => setOpen(!open)}>
        <ToolIcon name={item.name} />
        <span className="pflow-label">{item.name}</span>
        {brief && <span className="pflow-brief">{brief}</span>}
        <span className="pflow-meta">
          {item.duration_ms != null && !running && (
            <span className="pflow-dur">{fmtDuration(item.duration_ms)}</span>
          )}
          {running ? (
            <span className="tp-badge running">调用中</span>
          ) : (
            <span className={`tp-badge ${item.status}`}>{STATUS_LABEL[item.status] ?? item.status}</span>
          )}
          <Caret open={open} />
        </span>
      </button>
      {open && (
        <div className="pflow-detail">
          <ToolCodeBlock label="参数" code={JSON.stringify(item.args ?? {}, null, 2)} />
          {item.result && item.result.trim() && (
            <ToolCodeBlock
              label="结果"
              code={item.result}
              error={item.status !== "success"}
            />
          )}
          {item.diff && (
            <ToolCodeBlock label="改动" code={truncateMultiline(item.diff, 5000)} />
          )}
        </div>
      )}
    </div>
  );
}

/* ---------- 阶段正文段落 ---------- */

function StageText({ text, streaming }: { text: string; streaming: boolean }) {
  if (!text) return null;
  return (
    <div className="pflow-text">
      <Markdown content={text} streaming={streaming} />
    </div>
  );
}

/* ---------- 主组件：内联交错流 ---------- */

export const ProcessTimeline = memo(function ProcessTimeline({ msg }: { msg: StreamMessage }) {
  const streaming = msg.status === "streaming";
  const tl = msg.timeline && msg.timeline.length > 0 ? msg.timeline : null;

  // 渲染条目 + 最终回答：由 splitTimeline 统一切分（最终回答 = 最后一个 text 条目，
  // 内联流渲染其余条目）；老消息无 timeline 时从 reasoning/tool_calls 合成。
  let items: TimelineItem[];
  if (tl) {
    items = splitTimeline(tl).items;
  } else {
    items = synthesize(msg);
  }
  items = dropEchoedThinking(items);
  if (items.length === 0) return null;

  // live 思考 = 流式进行中且 timeline 最后一个条目是仍在生成的思考（避免误标已结束的思考行）
  const lastLive = streaming && tl ? tl[tl.length - 1] : null;
  const liveId = lastLive && lastLive.kind === "thinking" && lastLive.duration_ms == null
    ? lastLive.id
    : null;

  // 只有最后一个 text 条目才真正在流式输出，其余中间文本块均已定稿
  const lastTextIdx = streaming
    ? (() => { for (let i = items.length - 1; i >= 0; i--) { if (items[i].kind === "text") return i; } return -1; })()
    : -1;

  return (
    <div className="pflow">
      {items.map((item, idx) => {
        if (item.kind === "thinking") {
          return (
            <ThinkLine
              key={item.id}
              text={item.text}
              live={item.id === liveId}
              start={item.start}
              duration={item.duration_ms}
            />
          );
        }
        if (item.kind === "compress") {
          return <CompressLine key={item.id} item={item} />;
        }
        if (item.kind === "tool") {
          return <ToolLine key={item.id} item={item} />;
        }
        if (item.kind === "plan") return null;
        return <StageText key={item.id} text={item.text} streaming={idx === lastTextIdx} />;
      })}
    </div>
  );
});
