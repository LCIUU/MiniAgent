import { memo, useEffect, useRef, useCallback, useState } from "react";
import { useStore, useAgent } from "../useAgent";
import type { AttachmentMeta, StreamMessage } from "../types";
import { ApprovalCard, QuestionCard } from "./ToolCard";
import { Markdown } from "./Markdown";
import { ProcessTimeline, answerOf, hasLiveItem } from "./ProcessTimeline";
import { BASE } from "../api/client";
import { copyText } from "../utils/clipboard";

/** 附件展示：图片渲染缩略图（点击放大），非图片附件显示文件名 chip。 */
function AttachmentRow({ atts }: { atts: AttachmentMeta[] }) {
  const images = atts.filter((a) => a.kind === "image");
  const files = atts.filter((a) => a.kind !== "image");
  if (!images.length && !files.length) return null;
  return (
    <div className="msg-attachments">
      {images.map((a, i) => (
        <a
          key={a.path || i}
          href={`${BASE}/api/agent/attachment_file?path=${encodeURIComponent(a.path)}`}
          target="_blank"
          rel="noreferrer"
          className="msg-attach-img"
          title={a.name}
        >
          <img
            src={`${BASE}/api/agent/attachment_file?path=${encodeURIComponent(a.path)}`}
            alt={a.name}
          />
        </a>
      ))}
      {files.map((a, i) => (
        <span key={a.path || i} className="msg-attach-file" title={a.name}>
          📎 {a.name}
        </span>
      ))}
    </div>
  );
}

/** 用户消息复制按钮：悬浮消息时出现，复制后图标变为对勾。 */
function CopyMessageButton({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
    },
    [],
  );

  const onCopy = useCallback(async () => {
    const ok = await copyText(text);
    setState(ok ? "copied" : "failed");
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setState("idle"), 1600);
  }, [text]);

  const label =
    state === "copied" ? "已复制" : state === "failed" ? "复制失败，点击重试" : "复制这条消息";

  return (
    <button
      type="button"
      className={`msg-copy-btn${state === "copied" ? " copied" : state === "failed" ? " failed" : ""}`}
      onClick={() => void onCopy()}
      aria-label={label}
    >
      {state === "copied" ? (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M20 6 9 17l-5-5" />
        </svg>
      ) : state === "failed" ? (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      ) : (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}

// 判定"用户在底部附近"的阈值（像素）。
// 小于该值视为在底部，新内容到来时自动跟随；否则尊重用户的上滑操作。
const BOTTOM_THRESHOLD = 80;
// 判定思考详情"在底部附近"的阈值（详情高度较小，阈值稍小以减少误触）
const DETAIL_BOTTOM_THRESHOLD = 24;

function isNearBottom(el: HTMLElement, threshold = BOTTOM_THRESHOLD): boolean {
  const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
  return distance <= threshold;
}

function scrollToBottom(el: HTMLElement) {
  el.scrollTo({ top: el.scrollHeight, behavior: "auto" });
}

/**
 * 获取聊天容器内所有当前处于"展开/可滚动"状态的思考详情元素。
 * 只选择其内容超出容器（scrollHeight > clientHeight）的元素，避免无意义的滚动操作。
 */
function getScrollableDetails(root: HTMLElement): HTMLDivElement[] {
  const nodes = root.querySelectorAll<HTMLDivElement>(".pflow-detail");
  const list: HTMLDivElement[] = [];
  nodes.forEach((n) => {
    if (n.scrollHeight > n.clientHeight + 1) list.push(n);
  });
  return list;
}

// memo：流式期间 store 全量刷新，未变化的消息（对象引用不变）直接跳过重渲，
// 避免长对话时整列表重排导致卡顿。
export const MessageView = memo(function MessageView({
  msg,
  onApproval,
  onAnswer,
}: {
  msg: StreamMessage;
  onApproval: (id: string, approve: boolean) => void;
  onAnswer: (id: string, answers: string[]) => void;
}) {
  if (msg.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble">
          {msg.attachments && msg.attachments.length > 0 && (
            <AttachmentRow atts={msg.attachments} />
          )}
          {msg.content}
        </div>
        {msg.content.trim().length > 0 && <CopyMessageButton text={msg.content} />}
      </div>
    );
  }

  const streaming = msg.status === "streaming";
  const hasContent = msg.content.length > 0;
  const hasTools = (msg.tool_calls ?? []).length > 0;
  const hasReasoning = (msg.reasoning ?? "").length > 0;
  const hasTimeline =
    (msg.timeline ?? []).length > 0 || hasReasoning || hasTools;
  // 最终回答：timeline 模式下为最后一个正文条目（与过程流分离，避免重复渲染）
  const answer = answerOf(msg);

  if (!hasContent && !hasTimeline && streaming) {
    return (
      <div className="msg msg-assistant">
        <div className="status-line">
          <span className="spinner" />
          思考中…
        </div>
      </div>
    );
  }

  return (
    <div className="msg msg-assistant">
      <ProcessTimeline msg={msg} />
      {streaming && !hasLiveItem(msg) && (
        <div className="status-line">
          <span className="spinner" />
          智能体正在规划
        </div>
      )}
      {msg.pendingApproval && (
        <ApprovalCard
          approval={msg.pendingApproval}
          onDecide={(approve) => onApproval(msg.pendingApproval!.approval_id, approve)}
        />
      )}
      {msg.pendingQuestion && (
        <QuestionCard
          request={msg.pendingQuestion}
          onAnswer={(answers) => onAnswer(msg.pendingQuestion!.question_id, answers)}
        />
      )}
      {answer && <Markdown content={answer} streaming={streaming} />}
    </div>
  );
});

export function ChatView({
  onApproval,
  onAnswer,
}: {
  onApproval: (id: string, approve: boolean) => void;
  onAnswer: (id: string, answers: string[]) => void;
}) {
  const { messages, running, error, activeSessionId, hasMoreEarlierMessages } = useStore((s) => ({
    messages: s.messages,
    running: s.running,
    error: s.error,
    activeSessionId: s.activeSessionId,
    hasMoreEarlierMessages: s.hasMoreEarlierMessages,
  }));
  const { loadEarlierMessages } = useAgent();
  const scrollRef = useRef<HTMLDivElement>(null);
  // 用 ref 保持"是否跟随底部"的实时值，避免 scroll 事件频繁 re-render
  const nearBottomRef = useRef(true);
  // 仅用于控制"回到底部"按钮的显隐（跨阈值时才变化）
  const [atBottom, setAtBottom] = useState(true);
  // rAF 节流标记，避免同一帧内多次 scrollTo
  const rafRef = useRef<number | null>(null);
  // "回到底部"的滑动动画句柄；非 null 表示正在程序化滚动
  const jumpAnimRef = useRef<number | null>(null);
  // 递归事件锁：主容器驱动子容器时设为 true，子容器的 wheel 回调据此跳过反向驱动，防止互触发死锁
  const syncLockRef = useRef(false);
  // 主容器"触底边缘"：只在 nearBottom 从 false→true 的跳转帧触发详情同步
  // 避免主容器持续处于底部时每次 onScroll 都重复覆盖用户的手动滚动（跳回底部）
  const prevNearBottomRef = useRef(true);

  const USER_CTRL_ATTR = "data-detail-manual";

  /**
   * 同步思考详情到底部：仅同步"未被用户手动接管"的详情。
   *
   * 接管规则（与 wheel 回调配合，保证用户操作优先）：
   *  - 用户在某个详情内向上滚动（想要回看）→ 给该详情打上 manual 标记 → 从此不受自动同步影响
   *  - 用户把该详情滚到底部（程序或手动）→ 清除 manual 标记，恢复自动跟随
   */
  const syncDetailsToBottom = useCallback(() => {
    const root = scrollRef.current;
    if (!root) return;
    const details = getScrollableDetails(root);
    for (const d of details) {
      // 用户已手动接管该详情 → 尊重，不碰
      if (d.getAttribute(USER_CTRL_ATTR) === "1") continue;
      if (!isNearBottom(d, DETAIL_BOTTOM_THRESHOLD)) scrollToBottom(d);
    }
  }, [USER_CTRL_ATTR]);

  const cancelJump = useCallback(() => {
    if (jumpAnimRef.current != null) {
      cancelAnimationFrame(jumpAnimRef.current);
      jumpAnimRef.current = null;
    }
  }, []);

  const cancelJumpByUser = useCallback(() => {
    if (jumpAnimRef.current == null) return;
    cancelJump();
    const el = scrollRef.current;
    if (!el) return;
    const now = isNearBottom(el);
    nearBottomRef.current = now;
    prevNearBottomRef.current = now;
    setAtBottom(now);
  }, [cancelJump]);

  // 卸载时终止未完成的滑动动画
  useEffect(() => () => cancelJump(), [cancelJump]);

  // 监听主容器滚动：采用"边缘触发"（nearBottom=false → true）才驱动详情同步
  // 持续处于底部时不再反复同步，避免用户在详情里上滑被主容器 scroll 事件立即拉回
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    if (target !== scrollRef.current) return;
    // 滑动动画期间由动画帧接管，忽略中间位置的滚动事件，避免按钮闪回
    if (jumpAnimRef.current != null) return;

    const wasNearBottom = prevNearBottomRef.current;
    const isNow = isNearBottom(target);
    prevNearBottomRef.current = isNow;
    nearBottomRef.current = isNow;
    setAtBottom(isNow);

    // 边缘：用户刚把主容器滚到底部 → 同步详情（比如连续下滚后触底）
    if (!wasNearBottom && isNow && !syncLockRef.current) {
      syncLockRef.current = true;
      syncDetailsToBottom();
      requestAnimationFrame(() => {
        syncLockRef.current = false;
      });
    }
  }, [syncDetailsToBottom]);

  // Wheel 接力代理 + 用户手动接管检测
  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;

    const onWheel = (e: WheelEvent) => {
      if (syncLockRef.current) return;
      // 用户重新接管滚动：立即中断"回到底部"动画
      cancelJumpByUser();
      const t = e.target as HTMLElement | null;
      if (!t) return;
      const d = t.closest<HTMLDivElement>(".pflow-detail");
      if (!d || !(d.scrollHeight > d.clientHeight + 1)) return;

      // ① 用户在思考详情里向上滚动（想回看内容）→ 标记手动接管，
      //    从此自动同步不再覆盖该详情，直到用户亲手把它滚回绝对底部
      if (e.deltaY < 0) {
        d.setAttribute(USER_CTRL_ATTR, "1");
      }

      // ② 详情已到绝对底部 + 用户继续下滚 → 说明用户想往下读后面的内容，
      //    交给主容器接管本帧 delta（接力滚动）
      if (e.deltaY > 0) {
        const atAbsoluteBottom = d.scrollHeight - d.scrollTop - d.clientHeight <= 0;
        if (atAbsoluteBottom) {
          // 明确滚到底部 → 解除手动接管（恢复后续自动跟随）
          d.removeAttribute(USER_CTRL_ATTR);
          const mainEl = scrollRef.current;
          if (mainEl && !isNearBottom(mainEl)) {
            e.preventDefault();
            mainEl.scrollBy({ top: e.deltaY, behavior: "auto" });
          }
          return;
        }
      }

      // ③ 详情已到顶 + 用户继续上滚 → 主容器接管上滚（接力）
      if (e.deltaY < 0 && d.scrollTop <= 0) {
        const mainEl = scrollRef.current;
        if (mainEl && mainEl.scrollTop > 0) {
          e.preventDefault();
          mainEl.scrollBy({ top: e.deltaY, behavior: "auto" });
        }
      }
    };

    // 详情 scroll 事件的辅助：用户通过触摸/拖动滚动条/键盘 手动滚到绝对底部时，也清除接管标记
    const onDetailScroll = (e: Event) => {
      const d = e.target as HTMLDivElement;
      if (d.scrollHeight - d.scrollTop - d.clientHeight <= 0) {
        d.removeAttribute(USER_CTRL_ATTR);
      }
    };
    root.addEventListener("scroll", onDetailScroll, true);

    const onTouchStart = () => {
      cancelJumpByUser();
    };
    root.addEventListener("wheel", onWheel, { passive: false });
    root.addEventListener("touchstart", onTouchStart, { passive: true });
    return () => {
      root.removeEventListener("wheel", onWheel);
      root.removeEventListener("touchstart", onTouchStart);
      root.removeEventListener("scroll", onDetailScroll, true);
    };
  }, [USER_CTRL_ATTR, cancelJumpByUser]);

  // 新内容到来时：仅当主容器仍在底部附近才自动跟随（流式 token 合并到同一帧）
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!nearBottomRef.current) return;

    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(() => {
      // rAF 回调内二次确认：用户在 effect 触发后、rAF 执行前上滑了 → 取消本次自动滚动
      if (!nearBottomRef.current) {
        rafRef.current = null;
        return;
      }
      scrollToBottom(el);
      syncDetailsToBottom();
      rafRef.current = null;
    });
  }, [messages, running, syncDetailsToBottom]);

  // 切会话：强制滚到底部，重置跟随状态 + 清空所有 manual 标记
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    cancelJump();
    nearBottomRef.current = true;
    prevNearBottomRef.current = true;
    setAtBottom(true);
    requestAnimationFrame(() => {
      scrollToBottom(el);
      // 切会话后新内容全部默认"跟随"
      el.querySelectorAll<HTMLDivElement>(".pflow-detail").forEach((d) =>
        d.removeAttribute(USER_CTRL_ATTR),
      );
      syncDetailsToBottom();
    });
  }, [activeSessionId, cancelJump, syncDetailsToBottom, USER_CTRL_ATTR]);

  const jumpToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setAtBottom(true);
    cancelJump();

    const finish = () => {
      nearBottomRef.current = true;
      prevNearBottomRef.current = true;
      setAtBottom(true);
      syncDetailsToBottom();
    };

    const start = el.scrollTop;
    if (el.scrollHeight - el.clientHeight - start <= 1) {
      finish();
      return;
    }
    // 后台标签页 rAF 会被挂起，直接定位，避免动画永远停在半路
    if (document.visibilityState === "hidden") {
      scrollToBottom(el);
      finish();
      return;
    }

    // easeOutCubic：先快后慢地滑到底部，约 380ms
    const duration = 380;
    const startedAt = performance.now();
    const step = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      // 每帧重新取目标高度：流式输出期间内容还在变长，动画尾巴也要贴住最新底部
      const target = el.scrollHeight - el.clientHeight;
      el.scrollTop = start + (target - start) * eased;
      if (progress < 1) {
        jumpAnimRef.current = requestAnimationFrame(step);
      } else {
        jumpAnimRef.current = null;
        finish();
      }
    };
    jumpAnimRef.current = requestAnimationFrame(step);
  }, [cancelJump, syncDetailsToBottom]);

  const empty = messages.length === 0 && !running;
  const showJumpToBottom = !atBottom && messages.length > 0;

  return (
    <div className="chat-shell">
      <div className="chat" ref={scrollRef} onScroll={handleScroll}>
        <div className="chat-inner">
          {error && <div className="error-banner">⚠ {error}</div>}
          {empty && (
            <div className="empty-state">
              <h2>
                AI <span className="accent">编程助手</span>
              </h2>
              <p>
                告诉我你想实现的功能，我会在工作区内帮你规划、写代码、运行测试。
                <br />
                支持：终端命令 · 文件编辑 · 代码搜索 · 网络检索 · MCP 工具
              </p>
            </div>
          )}
          {hasMoreEarlierMessages && (
            <div className="load-earlier-bar">
              <button type="button" onClick={() => void loadEarlierMessages()}>
                加载更早的消息
              </button>
            </div>
          )}
          {messages.map((m) => (
            <MessageView
              key={m.id}
              msg={m}
              onApproval={onApproval}
              onAnswer={onAnswer}
            />
          ))}
        </div>
      </div>
      {showJumpToBottom && (
        <button
          type="button"
          className="jump-bottom"
          onClick={jumpToBottom}
          aria-label="回到底部"
        >
          <svg
            className="svgIcon"
            viewBox="0 0 384 512"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <path d="M169.4 470.6c12.5 12.5 32.8 12.5 45.3 0l160-160c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L224 370.8 224 64c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 306.8L54.6 265.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l160 160z" />
          </svg>
        </button>
      )}
    </div>
  );
}
