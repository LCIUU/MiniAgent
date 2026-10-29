// ===== Markdown 渲染组件（DeepSeek 风格）=====
//
// 功能：完整 Markdown 解析（标题/列表/表格/代码高亮/引用/链接/图片/删除线等）
// 交互：代码块一键复制、图片点击放大（灯箱）、<details> 折叠、外链新标签页
// 性能：markdown-it 单例 + useDeferredValue，流式输出时不阻塞输入
// 安全：html 输出统一经 DOMPurify 消毒

import {
  memo,
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import taskLists from "markdown-it-task-lists";
import { copyText } from "../utils/clipboard";

/* ---------- markdown-it 单例（模块级构建一次） ---------- */

const escapeHtml = (s: string) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

/* 允许透传的 HTML 标签（折叠块/排版辅助）；其余一律剥壳留内容。
   模型偶尔会把答案包在 <answer>/<think>/<updated> 等自定义标签里：
   markdown-it(html:true) 会将整段按 HTML 块透传（内部 markdown 不解析），
   DOMPurify 再剥掉标签后各段文本合并 → 呈现为无格式大块文本。
   此处提前剥壳，让标签内的内容回到正常 markdown 解析管线。 */
const ALLOWED_HTML_TAGS = new Set([
  "details", "summary", "br", "hr", "img", "a", "sub", "sup", "kbd", "mark", "abbr",
]);

function stripForeignHtml(src: string): string {
  // 按 fence 代码块分段：代码块内的 HTML 示例原样保留，只处理代码块外的部分
  const fenceRe = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;
  let out = "";
  let last = 0;
  for (let m = fenceRe.exec(src); m !== null; m = fenceRe.exec(src)) {
    out += cleanSegment(src.slice(last, m.index)) + m[1];
    last = m.index + m[1].length;
  }
  out += cleanSegment(src.slice(last));
  return out;

  function cleanSegment(seg: string): string {
    return seg
      // 思考块整体移除（思考过程已由过程流的 Think 行承载，不进正文）
      .replace(/<think[\s\S]*?<\/think\s*>/gi, "")
      .replace(/<think[^>]*>[\s\S]*$/gi, "")
      // HTML 注释移除
      .replace(/<!--[\s\S]*?-->/g, "")
      // 白名单标签保留，其余标签剥壳（保留内部文本，闭合标签直接删除）
      .replace(/<(\/?)([a-zA-Z][a-zA-Z0-9-]*)((?:\s[^<>]*)?)>/g, (all, _slash, name) =>
        ALLOWED_HTML_TAGS.has(name.toLowerCase()) ? all : "",
      );
  }
}

// 流式尾块渲染开关：hljs 全量高亮是长代码场景每帧最大开销，
// 流式进行中临时关闭（代码以纯文本等宽展示），完成块与流式结束后照常高亮。
let hljsEnabled = true;

function renderMarkdown(src: string, highlight = true): string {
  hljsEnabled = highlight;
  try {
    return DOMPurify.sanitize(md.render(stripForeignHtml(src)), { ADD_ATTR: ["target"] });
  } finally {
    hljsEnabled = true;
  }
}

const md = new MarkdownIt({
  html: true, // 允许 <details> 等折叠标签，输出统一经 DOMPurify 消毒
  linkify: true,
  breaks: true, // 单个换行渲染为换行（贴合 LLM 输出习惯）
  langPrefix: "hljs language-", // fence 代码块补 hljs 基类，便于统一样式
  highlight(code, lang) {
    if (hljsEnabled && lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
      } catch {
        /* 高亮失败降级为纯文本 */
      }
    }
    return escapeHtml(code);
  },
});

/* 任务列表（- [ ] / - [x]），复选框只读 */
md.use(taskLists, { enabled: false, label: true });

/* 代码块：外包头部栏（语言标签 + 复制按钮） */
const defaultFence = md.renderer.rules.fence!;
md.renderer.rules.fence = (tokens, idx, opts, env, self) => {
  const lang = (tokens[idx].info || "").trim().split(/\s+/)[0] || "text";
  const body = defaultFence(tokens, idx, opts, env, self);
  return (
    `<div class="md-code" data-lang="${escapeHtml(lang)}">` +
    `<div class="md-code-header code-card-head">` +
    `<span class="code-dots" aria-hidden="true"><i></i><i></i><i></i></span>` +
    `<span class="code-card-name">${escapeHtml(lang)}</span>` +
    `<button class="md-copy-btn" type="button" aria-label="复制代码">复制代码</button>` +
    `</div>${body}</div>`
  );
};

/* 链接：新标签页打开 */
const defaultLinkOpen =
  md.renderer.rules.link_open ?? ((t, i, o, _e, s) => s.renderToken(t, i, o));
md.renderer.rules.link_open = (tokens, idx, opts, env, self) => {
  tokens[idx].attrSet("target", "_blank");
  tokens[idx].attrSet("rel", "noopener noreferrer");
  return defaultLinkOpen(tokens, idx, opts, env, self);
};

/* 图片：懒加载 */
const defaultImage = md.renderer.rules.image!;
md.renderer.rules.image = (tokens, idx, opts, env, self) => {
  tokens[idx].attrSet("loading", "lazy");
  return defaultImage(tokens, idx, opts, env, self);
};

/* 表格：外包横向滚动容器 */
md.renderer.rules.table_open = () => `<div class="md-table-wrap"><table>`;
md.renderer.rules.table_close = () => `</table></div>`;

/* ---------- 图片灯箱 ---------- */

interface LightboxInfo {
  src: string;
  alt: string;
}

function ImageLightbox({ info, onClose }: { info: LightboxInfo; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div className="md-lightbox" onClick={onClose} role="dialog" aria-modal="true" aria-label={info.alt || "图片预览"}>
      <button className="md-lightbox-close" type="button" aria-label="关闭预览">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>
      <img src={info.src} alt={info.alt} onClick={(e) => e.stopPropagation()} />
    </div>,
    document.body,
  );
}

/* ---------- 流式增量渲染 ---------- */

/* 把 Markdown 源码切成「已完成块 + 尾块」：空行为块边界，代码围栏内部不切。
   流式期间已完成块内容稳定（只增不改），渲染结果可缓存，每帧仅重解析尾块，
   避免长文本下「全量解析 + 全量重建 DOM + 全量布局测量」导致的卡顿与成块上屏。 */
function splitMdBlocks(src: string): { done: string[]; tail: string } {
  const lines = src.split("\n");
  const done: string[] = [];
  let cur: string[] = [];
  let inFence = false;
  for (const line of lines) {
    const isFence = /^\s*(```|~~~)/.test(line);
    if (isFence && !inFence) {
      // 代码围栏开始：前置段落切为稳定块，未闭合代码块独立成尾块
      // （避免 fence 未闭合期间前文反复重渲染；尾块仅剩代码，配纯文本渲染开销最小）
      const pre = cur.join("\n").trim();
      if (pre) done.push(pre);
      cur = [];
      inFence = true;
    } else if (isFence && inFence) {
      inFence = false;
    }
    cur.push(line);
    if (!inFence && line.trim() === "") {
      const block = cur.join("\n").trim();
      if (block) done.push(block);
      cur = [];
    }
  }
  return { done, tail: cur.join("\n") };
}

/* 单块渲染：memo 后已完成块跳过重渲染（不重复设置 innerHTML） */
const MdBlock = memo(function MdBlock({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
});

/* ---------- 主组件 ---------- */

export const Markdown = memo(function Markdown({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  // 非流式：低优先级值平滑渲染；流式：直接用最新值（配合块级增量，跟手不积压，
  // 避免 useDeferredValue 在繁忙时持续落后导致的「卡住后冒出一大段」）
  const deferred = useDeferredValue(content);
  const html = useMemo(() => (streaming ? "" : renderMarkdown(deferred)), [streaming, deferred]);

  // 流式增量：按块切分，已完成块渲染结果缓存（每块一生只解析一次），每帧仅重解析尾块
  const streamParts = useMemo(() => (streaming ? splitMdBlocks(content) : null), [streaming, content]);
  const blockCacheRef = useRef<{ src: string[]; html: string[] }>({ src: [], html: [] });
  const streamHtml = useMemo(() => {
    if (!streamParts) return null;
    const cache = blockCacheRef.current;
    const doneHtml = streamParts.done.map((b, i) => (cache.src[i] === b ? cache.html[i] : renderMarkdown(b)));
    blockCacheRef.current = { src: streamParts.done, html: doneHtml };
    // 尾块每帧重渲染：关闭 hljs 高亮（长代码场景每帧全量高亮是最大开销），
    // 代码以纯文本等宽展示；fence 闭合进完成块或流式结束后自动恢复高亮
    return { doneHtml, tailHtml: renderMarkdown(streamParts.tail, false) };
  }, [streamParts]);

  const rootRef = useRef<HTMLDivElement>(null);
  const [lightbox, setLightbox] = useState<LightboxInfo | null>(null);

  // 代码块全宽：抵消列表/引用/折叠等任意深度嵌套缩进，始终撑满内容区。
  // 采用「显式宽度 + 左偏移抵消」而非 auto 宽度 + 负边距：
  //   右缘 = 根左缘 + 根宽，恒等于根右缘，不受祖先宽度变化影响。
  // 三步分批（重置 → 读取 → 写入），既避免旧内联样式污染测量基准，也减少重排次数。
  const measureCodes = useCallback(() => {
    const root = rootRef.current;
    if (!root) return;
    const codes = Array.from(root.querySelectorAll<HTMLElement>(".md-code"));
    if (!codes.length) return;

    // 1) 重置内联样式，恢复自然文档流位置
    for (const c of codes) {
      c.style.marginLeft = "";
      c.style.width = "";
    }

    // 2) 批量读取自然位置
    const rootRect = root.getBoundingClientRect();
    const rects = codes.map((c) => c.getBoundingClientRect());

    // 3) 批量写入：仅嵌套（left > 0）的代码块需要全出血，顶层保持纯 CSS 全宽
    for (let i = 0; i < codes.length; i++) {
      if (rects[i].width === 0) continue; // 折叠容器内暂不可见，展开后由 summary 点击触发重测
      const left = rects[i].left - rootRect.left;
      if (left > 0.5) {
        codes[i].style.marginLeft = `${-left}px`;
        codes[i].style.width = `${rootRect.width}px`;
      }
    }
  }, []);

  // 流式期间不做代码块全宽测量：streamTailHtml 每帧变化 → 每帧「重置样式+强制同步布局+重写」
  // 两次全文档 reflow，文本越长越卡。嵌套代码块的全宽样式待流式结束（streaming=false）后统一测量。
  useLayoutEffect(() => {
    if (streaming) return;
    measureCodes();
  }, [streaming, deferred, measureCodes]);

  // 根容器尺寸变化（滚动条出现/侧栏收起动画/窗口缩放）时自动校正，
  // 防止旧偏移失效导致错位；rAF 合并连续回调。
  // 流式期间不挂载：文本增长使根容器每帧变尺寸，会导致每帧触发全量布局测量。
  useEffect(() => {
    if (streaming) return;
    const root = rootRef.current;
    if (!root || typeof ResizeObserver === "undefined") return;
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => measureCodes());
    });
    ro.observe(root);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [streaming, measureCodes]);

  // 事件委托：代码复制 / 图片放大（innerHTML 内的元素无法直接绑事件）
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;

      // 代码块复制
      const btn = target.closest<HTMLButtonElement>(".md-copy-btn");
      if (btn) {
        e.preventDefault();
        const code = btn.closest(".md-code")?.querySelector("pre code");
        if (code) {
          void copyText(code.textContent ?? "").then((ok) => {
            btn.classList.add("copied");
            btn.textContent = ok ? "已复制" : "复制失败";
            window.setTimeout(() => {
              btn.classList.remove("copied");
              btn.textContent = "复制代码";
            }, 1600);
          });
        }
        return;
      }

      // 图片点击放大
      const img = target.closest("img");
      if (img) {
        e.preventDefault();
        const src = img.getAttribute("src") ?? "";
        if (src) setLightbox({ src, alt: img.alt });
        return;
      }

      // 折叠块展开/收起后重新测量代码块宽度
      if (target.closest("summary")) {
        requestAnimationFrame(() => measureCodes());
      }
    };
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, [measureCodes]);

  // 空内容也必须先执行全部 Hook，提前 return 会导致 Hook 顺序随 props 变化
  if (!content) return null;

  // 流式：块级增量结构（已完成块 memo 不重渲染，仅尾块随增量更新）
  if (streamHtml) {
    return (
      <>
        <div ref={rootRef} className="markdown-body streaming">
          {streamHtml.doneHtml.map((h, i) => (
            <MdBlock key={i} html={h} />
          ))}
          <MdBlock html={streamHtml.tailHtml} />
        </div>
        {lightbox && <ImageLightbox info={lightbox} onClose={() => setLightbox(null)} />}
      </>
    );
  }

  return (
    <>
      <div
        ref={rootRef}
        className={`markdown-body${streaming ? " streaming" : ""}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {lightbox && <ImageLightbox info={lightbox} onClose={() => setLightbox(null)} />}
    </>
  );
});
