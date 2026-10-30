import { useEffect, useRef, useState, type CSSProperties } from "react";
import { useStore } from "../useAgent";
import { getState, setState } from "../store";
import { api } from "../api/client";
import type { AttachmentMeta, TimelineItem } from "../types";
import { Loading } from "./Loading";

/** 毫秒格式化为 mm:ss / h:mm:ss。 */
function formatMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** 字节数格式化为可读大小。 */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** 允许上传的附件类型：PDF / Word / 图片 / 常见文本 */
const ACCEPT = ".pdf,.doc,.docx,.txt,.md,.py,.js,.ts,.json,.csv,.log,.png,.jpg,.jpeg,.webp,.gif";

/** 粘贴图片：允许的图片 MIME 类型与单张大小上限 */
const PASTE_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"]);
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10 MB

/** 粘贴图片条目：本地预览占位 → 即时上传 → 完成后携带正式元信息 */
interface PastedImage {
  id: string;
  previewUrl: string; // URL.createObjectURL 生成的本地预览
  status: "uploading" | "done" | "error";
  meta?: AttachmentMeta; // 上传完成后后端返回的正式附件记录
  error?: string;
}

let pasteSeq = 0;
const nextPasteId = () => `paste-${Date.now()}-${++pasteSeq}`;

/** 思考强度档位（value 为 reasoning_effort 参数，null=跟随模型默认；展示用英文） */
const EFFORT_OPTIONS: { value: string | null; label: string; desc: string }[] = [
  { value: null, label: "Default", desc: "跟随模型默认思考强度" },
  { value: "off", label: "Off", desc: "关闭思考，响应最快" },
  { value: "low", label: "Low", desc: "轻量思考，快速响应" },
  { value: "medium", label: "Medium", desc: "平衡思考深度与速度" },
  { value: "xhigh", label: "Max", desc: "最深思考，适合复杂任务" },
];

/** 终端命令确认模式（value 与后端设置一致） */
type ModeValue = "auto" | "high_risk" | "all";
const MODE_OPTIONS: { value: ModeValue; label: string; desc: string }[] = [
  { value: "auto", label: "自动执行", desc: "终端命令直接执行，不确认" },
  { value: "high_risk", label: "高危命令需确认", desc: "rm -rf / git push 等高危命令需确认" },
  { value: "all", label: "所有命令都需确认", desc: "每条终端命令执行前都需要确认" },
];

/** 发送栏左侧：模型执行总耗时（常驻显示：运行中实时跳动，结束后定格，无记录时为 00:00）。 */
function RunStatus() {
  const running = useStore((s) => s.running);
  const runStartedAt = useStore((s) => s.runStartedAt);
  const runDurationMs = useStore((s) => s.runDurationMs);
  const messages = useStore((s) => s.messages);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [running]);

  // 刷新/切换会话后 runDurationMs 归零：回退到最近一条 assistant 消息持久化的耗时
  let lastRunMs: number | null = runDurationMs;
  if (!running && lastRunMs == null) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.duration_ms) {
        lastRunMs = m.duration_ms;
        break;
      }
    }
  }

  const ms = running && runStartedAt ? now - runStartedAt : (lastRunMs ?? 0);
  const idle = !running && lastRunMs == null;
  return (
    <span className={`run-status-text ${running ? "live" : ""}`}>
      {running ? "Deep Action" : idle ? "与 LCIU CODEX 对话进行任务" : "总耗时"}
      {!idle && (
        <>
          {"\u00A0\u00A0"}·{"\u00A0\u00A0"}
          {formatMs(ms)}
        </>
      )}
    </span>
  );
}

/** 执行规划弹窗：与 RunStatus 同行，靠右显示。 */
function PlanPopup() {
  const running = useStore((s) => s.running);
  const messages = useStore((s) => s.messages);
  const [open, setOpen] = useState(false);

  // 运行中只看本轮（最后一条 assistant）的规划：新一轮规划尚未产生时保持为空，
  // 不回退上一轮；空闲/刷新后回退到最近一条带规划的历史消息（保持刷新后可见）
  let planItem: Extract<TimelineItem, { kind: "plan" }> | undefined;
  if (running) {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") {
        planItem = messages[i].timeline?.find((t) => t.kind === "plan");
        break;
      }
    }
  } else {
    for (let i = messages.length - 1; i >= 0; i--) {
      const it = messages[i].timeline?.find((t) => t.kind === "plan");
      if (it) {
        planItem = it;
        break;
      }
    }
  }

  const steps = planItem?.steps ?? [];
  const todos = planItem?.todos;
  const completedCount = todos ? todos.filter((t) => t.status === "completed").length : 0;
  const totalCount = todos ? todos.length : steps.length;
  const planKey = steps.join("\n");
  // 本轮规划未产生但历史有过规划时，保留按钮（仅显示"任务规划"，不带上一轮进度）
  const showButton =
    !!planItem ||
    (running && messages.some((m) => m.timeline?.some((t) => t.kind === "plan")));

  // 新规划到达时自动展开；发送新消息（running 变 true）时先收起，展示本轮规划
  useEffect(() => {
    if (planKey) setOpen(true);
  }, [planKey]);

  useEffect(() => {
    if (running) setOpen(false);
  }, [running]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Element;
      if (!target.closest(".plan-popup-container")) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  if (!showButton) return null;

  return (
    <div className="plan-popup-container">
      <button
        type="button"
        className="plan-toggle-btn"
        onClick={() => setOpen(!open)}
      >
        任务规划
        {planItem && (
          <>
            {"\u00A0\u00A0"}·{"\u00A0\u00A0"}
            {completedCount} / {totalCount}
          </>
        )}
      </button>
      {open && planItem && (
        <div className="plan-popup">
          <div className="plan-popup-header">
            <span className="plan-dot plan-dot-red"></span>
            <span className="plan-dot plan-dot-yellow"></span>
            <span className="plan-dot plan-dot-green"></span>
          </div>
          <div className="plan-popup-content">
            <ul>
              {steps.map((s, i) => {
                const todo = todos?.[i];
                const status = todo?.status ?? "pending";
                return (
                  <li key={i} className={`plan-step plan-step-${status}`}>
                    <span className={`step-heart step-heart-${status}`}>
                      <svg viewBox="0 0 24 24" className="heart-outline" xmlns="http://www.w3.org/2000/svg">
                        <path d="M17.5,1.917a6.4,6.4,0,0,0-5.5,3.3,6.4,6.4,0,0,0-5.5-3.3A6.8,6.8,0,0,0,0,8.967c0,4.547,4.786,9.513,8.8,12.88a4.974,4.974,0,0,0,6.4,0C19.214,18.48,24,13.514,24,8.967A6.8,6.8,0,0,0,17.5,1.917Zm-3.585,18.4a2.973,2.973,0,0,1-3.83,0C4.947,16.006,2,11.87,2,8.967a4.8,4.8,0,0,1,4.5-5.05A4.8,4.8,0,0,1,11,8.967a1,1,0,0,0,2,0,4.8,4.8,0,0,1,4.5-5.05A4.8,4.8,0,0,1,22,8.967C22,11.87,19.053,16.006,13.915,20.313Z" />
                      </svg>
                      <svg viewBox="0 0 24 24" className="heart-filled" xmlns="http://www.w3.org/2000/svg">
                        <path d="M17.5,1.917a6.4,6.4,0,0,0-5.5,3.3,6.4,6.4,0,0,0-5.5-3.3A6.8,6.8,0,0,0,0,8.967c0,4.547,4.786,9.513,8.8,12.88a4.974,4.974,0,0,0,6.4,0C19.214,18.48,24,13.514,24,8.967A6.8,6.8,0,0,0,17.5,1.917Z" />
                      </svg>
                    </span>
                    <span className="plan-step-text">{s.replace(/^\s*(?:\d{1,2}[.、)）]|[（(]\d{1,2}[)）]|[一二三四五六七八九十]{1,3}[、.])\s*/, "")}</span>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

/** 发送栏头部行：左侧 RunStatus + 右侧 PlanPopup，flex 两端对齐。 */
function InputHeader() {
  return (
    <div className="input-header">
      <RunStatus />
      <PlanPopup />
    </div>
  );
}

/** 回形针图标（线性风格，与整体图标体系一致）。 */
function ClipIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  );
}

/** 垃圾桶图标（用于移除已选附件）。 */
function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
}

/** 下箭头（展开指示）。 */
function ChevronDownIcon() {
  return (
    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

/** 右箭头（子菜单入口指示）。 */
function ChevronRightIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

/** 对勾（选中指示）。 */
function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12l5 5L20 6" />
    </svg>
  );
}

/* 厂商官方标识（Simple Icons 收录的品牌原始图形） */
const QWEN_LOGO_PATH =
  "M23.919 14.545 20.817 9.17l1.47-2.544a.56.56 0 0 0 0-.566l-1.633-2.83a.57.57 0 0 0-.49-.283h-6.207L12.487.402a.57.57 0 0 0-.49-.284H8.732a.56.56 0 0 0-.49.284L5.139 5.775h-2.94a.56.56 0 0 0-.49.284L.077 8.887a.56.56 0 0 0 0 .567L3.18 14.83l-1.47 2.545a.56.56 0 0 0 0 .566l1.634 2.83a.57.57 0 0 0 .49.283h6.205l1.47 2.545a.57.57 0 0 0 .49.284h3.266a.57.57 0 0 0 .49-.284l3.104-5.375h2.94a.57.57 0 0 0 .49-.283l1.634-2.828a.55.55 0 0 0-.004-.568M8.733.686l1.634 2.828-1.634 2.828H21.8L20.164 9.17H7.425L5.63 6.06Zm1.306 19.801-6.205-.002 1.634-2.83h3.265L2.201 6.344h3.267q3.182 5.517 6.367 11.032zm10.124-5.66L18.53 12l-6.532 11.315-1.634-2.83c2.129-3.673 4.25-7.351 6.373-11.028h3.592l3.102 5.374z";
const DEEPSEEK_LOGO_PATH =
  "M23.748 4.651c-.254-.124-.364.113-.512.233-.051.04-.094.09-.137.137-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.155-.708-.311-.955-.65-.172-.24-.219-.509-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.094.172.187.129.323-.082.28-.18.553-.266.833-.055.179-.137.218-.328.14a5.5 5.5 0 0 1-1.737-1.179c-.857-.828-1.631-1.743-2.597-2.46a12 12 0 0 0-.689-.47c-.985-.957.13-1.743.387-1.836.27-.098.094-.433-.778-.428-.872.003-1.67.295-2.687.685a3 3 0 0 1-.465.136 9.6 9.6 0 0 0-2.883-.101c-1.885.21-3.39 1.1-4.497 2.622C.082 8.776-.231 10.854.152 13.02c.403 2.284 1.568 4.175 3.36 5.653 1.857 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.132-.284 4.994-1.86.47.234.962.328 1.78.398.629.058 1.235-.031 1.705-.129.735-.155.684-.836.418-.961-2.155-1.004-1.682-.595-2.112-.926 1.095-1.295 2.768-3.598 3.284-6.733.05-.346.115-.834.108-1.114-.004-.171.035-.238.23-.257a4.2 4.2 0 0 0 1.545-.475c1.397-.763 1.96-2.016 2.093-3.517.02-.23-.004-.467-.247-.588M11.58 18.168c-2.088-1.642-3.101-2.183-3.52-2.16-.39.024-.32.472-.234.763.09.288.207.487.371.74.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.168-1.361-.801-2.5-1.86-3.301-3.306-.775-1.393-1.225-2.888-1.299-4.482-.02-.385.094-.522.477-.592a4.7 4.7 0 0 1 1.53-.038c2.131.311 3.946 1.264 5.467 2.774.868.86 1.525 1.887 2.202 2.89.72 1.066 1.494 2.082 2.48 2.915.348.291.626.513.892.677-.802.09-2.14.109-3.055-.615zm1.001-6.44a.306.306 0 0 1 .415-.287.3.3 0 0 1 .113.074.3.3 0 0 1 .086.214c0 .17-.136.307-.308.307a.303.303 0 0 1-.306-.307m3.11 1.596c-.2.081-.4.151-.591.16a1.25 1.25 0 0 1-.798-.254c-.274-.23-.47-.358-.551-.758a1.7 1.7 0 0 1 .015-.588c.07-.327-.007-.537-.238-.727-.188-.156-.426-.199-.689-.199a.6.6 0 0 1-.254-.078.253.253 0 0 1-.114-.358 1 1 0 0 1 .192-.21c.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.392.451.462.576.685.915.176.264.336.536.446.848.066.194-.02.353-.25.45";

/** 模型图标（玻璃卡片用）：按厂商显示官方标识，未知模型回退芯片图标。 */
function ModelGlyph({ model }: { model?: string }) {
  const key = (model ?? "").toLowerCase();
  const path = key.includes("deepseek")
    ? DEEPSEEK_LOGO_PATH
    : key.includes("qwen")
      ? QWEN_LOGO_PATH
      : null;
  if (path) {
    return (
      <svg className="glass-glyph brand-logo" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d={path} />
      </svg>
    );
  }
  return (
    <svg
      className="glass-glyph"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="6" y="6" width="12" height="12" rx="2.5" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" />
    </svg>
  );
}

/** 推理强度信号条（分段刻度用）：bars 为点亮的条数。 */
function EffortGlyph({ bars }: { bars: number }) {
  return (
    <svg className="effort-bars" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="4" y="14" width="4" height="6" rx="1" opacity={bars >= 1 ? 1 : 0.25} />
      <rect x="10" y="10" width="4" height="10" rx="1" opacity={bars >= 2 ? 1 : 0.25} />
      <rect x="16" y="6" width="4" height="14" rx="1" opacity={bars >= 3 ? 1 : 0.25} />
    </svg>
  );
}

/** 各推理档位点亮的信号条数量（Default 视为中等）。 */
const effortBars = (value: string | null): number => {
  if (value === "off") return 0;
  if (value === "low") return 1;
  if (value === "xhigh") return 3;
  return 2;
};

/** 终端确认模式图标（终端窗口 + 提示符线性风格）。 */
function TerminalModeIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2.5" y="4" width="19" height="16" rx="3" />
      <path d="M6.5 9.5l3.5 3-3.5 3" />
      <path d="M12.5 15.5h5" />
    </svg>
  );
}

/** 爱心勾选（Uiverse by catraco）：未选中为描边心，选中后实心并迸出火花。 */
function HeartCheck({ checked, onPick }: { checked: boolean; onPick: () => void }) {
  return (
    <span className="heart-container">
      <input
        type="checkbox"
        className="heart-checkbox"
        checked={checked}
        readOnly
        onClick={(e) => {
          e.preventDefault();
          onPick();
        }}
      />
      <span className="svg-container">
        <svg viewBox="0 0 24 24" className="svg-outline" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <path d="M17.5,1.917a6.4,6.4,0,0,0-5.5,3.3,6.4,6.4,0,0,0-5.5-3.3A6.8,6.8,0,0,0,0,8.967c0,4.547,4.786,9.513,8.8,12.88a4.974,4.974,0,0,0,6.4,0C19.214,18.48,24,13.514,24,8.967A6.8,6.8,0,0,0,17.5,1.917Zm-3.585,18.4a2.973,2.973,0,0,1-3.83,0C4.947,16.006,2,11.87,2,8.967a4.8,4.8,0,0,1,4.5-5.05A4.8,4.8,0,0,1,11,8.967a1,1,0,0,0,2,0,4.8,4.8,0,0,1,4.5-5.05A4.8,4.8,0,0,1,22,8.967C22,11.87,19.053,16.006,13.915,20.313Z" />
        </svg>
        <svg viewBox="0 0 24 24" className="svg-filled" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <path d="M17.5,1.917a6.4,6.4,0,0,0-5.5,3.3,6.4,6.4,0,0,0-5.5-3.3A6.8,6.8,0,0,0,0,8.967c0,4.547,4.786,9.513,8.8,12.88a4.974,4.974,0,0,0,6.4,0C19.214,18.48,24,13.514,24,8.967A6.8,6.8,0,0,0,17.5,1.917Z" />
        </svg>
        <svg className="svg-celebrate" width="100" height="100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <polygon points="10,10 20,20" />
          <polygon points="10,50 20,50" />
          <polygon points="20,80 30,70" />
          <polygon points="90,10 80,20" />
          <polygon points="90,50 80,50" />
          <polygon points="80,80 70,70" />
        </svg>
      </span>
    </span>
  );
}

export function InputBox({
  disabled,
  placeholder,
  onSend,
  running,
  onStop,
}: {
  disabled?: boolean;
  placeholder?: string;
  onSend: (text: string, attachments: AttachmentMeta[], reasoningEffort: string | null) => void;
  /** 模型执行中：发送按钮变为停止按钮 */
  running?: boolean;
  /** 点击停止按钮时触发 */
  onStop?: () => void;
}) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [images, setImages] = useState<PastedImage[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [effortOpen, setEffortOpen] = useState(false);
  const [subOpen, setSubOpen] = useState<"model" | "effort" | null>(null);
  const [modeOpen, setModeOpen] = useState(false);
  const { settings } = useStore((s) => ({ settings: s.settings }));
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const effortRef = useRef<HTMLDivElement>(null);
  const modeRef = useRef<HTMLDivElement>(null);

  // 运行结束后重新聚焦输入框，避免用户需要再次点击才能继续输入
  useEffect(() => {
    if (!disabled) {
      // 延迟聚焦，等待 React 完成渲染
      const timer = setTimeout(() => taRef.current?.focus(), 50);
      return () => clearTimeout(timer);
    }
  }, [disabled]);

  // 当前模型与可选清单（阿里百炼 Qwen 系列，来自后端设置）
  const model = settings?.model || "qwen3.7-flash";
  const modelOptions = settings?.model_options?.length
    ? settings.model_options
    : [model];
  // 推理等级：直接从持久化设置读取（刷新后恢复上次选择）
  const effort = settings?.reasoning_effort ?? null;
  // 终端命令确认模式：从持久化设置读取（默认自动执行）
  const confirmMode: ModeValue = settings?.command_confirm_mode ?? "auto";

  // 点击外部关闭思考强度下拉
  useEffect(() => {
    if (!effortOpen) return;
    const onDown = (e: MouseEvent) => {
      if (effortRef.current && !effortRef.current.contains(e.target as Node)) {
        setEffortOpen(false);
        setSubOpen(null);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [effortOpen]);

  // 点击外部关闭终端确认模式下拉
  useEffect(() => {
    if (!modeOpen) return;
    const onDown = (e: MouseEvent) => {
      if (modeRef.current && !modeRef.current.contains(e.target as Node)) {
        setModeOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [modeOpen]);

  const currentEffort = EFFORT_OPTIONS.find((o) => o.value === effort) ?? EFFORT_OPTIONS[0];
  const currentMode = MODE_OPTIONS.find((o) => o.value === confirmMode) ?? MODE_OPTIONS[0];

  /** 切换模型：走设置接口持久化，成功后同步本地 store。 */
  const pickModel = async (m: string) => {
    setSubOpen(null);
    setEffortOpen(false);
    if (m === model) return;
    try {
      const s = await api.updateSettings({ model: m });
      setState({ settings: s });
    } catch {
      /* 设置失败静默忽略（下次打开设置页可见真实值） */
    }
  };

  /** 切换推理等级：同样走设置接口持久化（显式传 null 恢复 Default）。 */
  const pickEffort = async (v: string | null) => {
    setSubOpen(null);
    setEffortOpen(false);
    if (v === effort) return;
    try {
      const s = await api.updateSettings({ reasoning_effort: v });
      setState({ settings: s });
    } catch {
      /* 设置失败静默忽略 */
    }
  };

  /** 切换终端命令确认模式：走设置接口持久化；菜单保持停留，不自动关闭。 */
  const pickMode = async (v: ModeValue) => {
    if (v === confirmMode) return;
    try {
      const s = await api.updateSettings({ command_confirm_mode: v });
      setState({ settings: s });
    } catch {
      /* 设置失败静默忽略 */
    }
  };

  const pickFiles = () => fileRef.current?.click();

  const onFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files ?? []);
    if (picked.length) {
      // 客户端文件大小限制（20MB，与后端一致）
      const oversized = picked.find((f) => f.size > 20 * 1024 * 1024);
      if (oversized) {
        setUploadError(`文件过大: ${oversized.name}（最大 20MB）`);
        e.target.value = "";
        return;
      }
      setFiles((prev) => {
        // 按名称去重后合并
        const seen = new Set(prev.map((f) => f.name));
        return [...prev, ...picked.filter((f) => !seen.has(f.name))];
      });
      setUploadError("");
    }
    e.target.value = ""; // 允许重复选择同一文件
  };

  const removeFile = (name: string) => setFiles((prev) => prev.filter((f) => f.name !== name));

  /** 粘贴图片：本地预览后立即上传，完成后替换为正式元信息 */
  const addPastedImage = async (file: File) => {
    // 格式校验（兼容 Windows 剪贴板可能带大写 MIME 的情况）
    const mime = file.type.toLowerCase();
    if (!PASTE_IMAGE_TYPES.has(mime)) {
      setUploadError(`不支持的图片格式：${mime || "未知"}（仅支持 PNG、JPG、GIF、WebP、BMP）`);
      return;
    }
    if (file.size > MAX_IMAGE_SIZE) {
      setUploadError(`图片超出大小限制（${formatSize(MAX_IMAGE_SIZE)}），当前 ${formatSize(file.size)}`);
      return;
    }
    setUploadError("");

    // 生成可识别文件名（避免所有图片都叫 image.png 冲突）
    const extMap: Record<string, string> = {
      "image/png": ".png",
      "image/jpeg": ".jpg",
      "image/gif": ".gif",
      "image/webp": ".webp",
      "image/bmp": ".bmp",
    };
    const ext = extMap[mime] ?? ".png";
    const name = `pasted-${Date.now()}${ext}`;
    const namedFile = new File([file], name, { type: file.type });

    const id = nextPasteId();
    const previewUrl = URL.createObjectURL(namedFile);
    setImages((prev) => [...prev, { id, previewUrl, status: "uploading" }]);

    try {
      const [meta] = await api.uploadFiles(getState().workspace, [namedFile]);
      setImages((prev) => prev.map((im) => (im.id === id ? { ...im, status: "done", meta } : im)));
    } catch (err) {
      const msg = (err as Error).message || "上传失败";
      setImages((prev) => prev.map((im) => (im.id === id ? { ...im, status: "error", error: msg } : im)));
    }
  };

  const removeImage = (id: string) => {
    setImages((prev) => {
      const im = prev.find((x) => x.id === id);
      if (im) URL.revokeObjectURL(im.previewUrl);
      return prev.filter((x) => x.id !== id);
    });
  };

  /** 组件卸载时清理所有本地预览 blob URL */
  useEffect(() => {
    const imgs = images;  // 捕获当前引用
    return () => {
      imgs.forEach((im) => URL.revokeObjectURL(im.previewUrl));
    };
  }, [images]);

  /** 粘贴事件：支持 Ctrl/Cmd + V 剪贴板图片直接粘贴到输入区 */
  const onPaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    if (disabled || uploading) return;
    const items = e.clipboardData?.items;
    if (!items) return;
    const imageItems: DataTransferItem[] = [];
    let sawNonImageFile = false;
    for (const item of Array.from(items)) {
      if (item.kind === "file") {
        if (item.type.toLowerCase().startsWith("image/")) imageItems.push(item);
        else sawNonImageFile = true;
      }
    }
    if (!imageItems.length) {
      // 纯文本走默认行为；发现非图片文件时给出明确提示
      if (sawNonImageFile) {
        e.preventDefault();
        setUploadError("剪贴板内容不是图片，仅支持粘贴图片");
      }
      return;
    }
    // 有图片时接管粘贴（避免文件名等文本插入 textarea）
    e.preventDefault();
    setUploadError("");
    for (const it of imageItems) {
      const f = it.getAsFile();
      if (f) void addPastedImage(f);
    }
  };

  const submit = async () => {
    const t = text.trim();
    if (!t || disabled || uploading) return;
    const pendingImages = images.filter((im) => im.status === "uploading");
    if (pendingImages.length) {
      setUploadError("图片上传中，请稍候…");
      return;
    }
    const readyImages = images.filter((im) => im.status === "done" && im.meta);
    setUploading(true);
    setUploadError("");
    let atts: AttachmentMeta[] = [];
    try {
      if (files.length) {
        const workspace = getState().workspace;
        atts = await api.uploadFiles(workspace, files);
      }
      atts = [...atts, ...readyImages.map((im) => im.meta!)];
      onSend(t, atts, effort);
      setText("");
      setFiles([]);
      images.forEach((im) => URL.revokeObjectURL(im.previewUrl));
      setImages([]);
      if (taRef.current) taRef.current.style.height = "auto";
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    // 中文/日文输入法确认候选词时的 Enter 不能当作发送
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submit();
    }
  };

  const autoResize = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  };

  return (
    <div className="input-area">
      <InputHeader />
      <div className="input-box">
        <textarea
          ref={taRef}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            autoResize();
          }}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          placeholder={disabled ? "正在执行，请稍候…" : placeholder ?? "输入任务描述，例如：帮我创建一个 FastAPI 项目…"}
          rows={1}
          disabled={disabled}
        />
        {images.length > 0 && (
          <div className="paste-previews">
            {images.map((im) => (
              <div className={`paste-item ${im.status}`} key={im.id}>
                <img src={im.previewUrl} alt="粘贴图片预览" />
                {im.status === "uploading" && (
                  <span className="paste-mask">
                    <Loading />
                    上传中
                  </span>
                )}
                {im.status === "error" && (
                  <span className="paste-mask error" title={im.error}>
                    上传失败
                  </span>
                )}
                <button
                  className="paste-del"
                  onClick={() => removeImage(im.id)}
                  aria-label="移除"
                  type="button"
                >
                  <TrashIcon />
                </button>
              </div>
            ))}
          </div>
        )}
        {files.length > 0 && (
          <div className="attach-chips">
            {files.map((f) => (
              <span className="attach-chip" key={f.name} title={`${f.name}（${formatSize(f.size)}）`}>
                <span className="chip-name">{f.name}</span>
                <span className="chip-size">{formatSize(f.size)}</span>
                <button
                  className="chip-del"
                  onClick={() => removeFile(f.name)}
                  aria-label="移除"
                  type="button"
                >
                  <TrashIcon />
                </button>
              </span>
            ))}
          </div>
        )}
        {uploadError && <div className="attach-error">附件上传失败：{uploadError}</div>}
        <div className="input-actions">
          <button
            className="attach-btn"
            onClick={pickFiles}
            disabled={disabled || uploading}
            aria-label="上传附件"
            type="button"
          >
            <ClipIcon />
          </button>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept={ACCEPT}
            onChange={onFilesChange}
            style={{ display: "none" }}
          />
          <div className="mode-wrap" ref={modeRef}>
            <button
              className="attach-btn"
              onClick={() => setModeOpen((v) => !v)}
              disabled={disabled}
              aria-label={`终端命令确认模式：${currentMode.label}`}
              type="button"
            >
              <TerminalModeIcon />
            </button>
            {modeOpen && (
              <div className="effort-menu mode-menu">
                {MODE_OPTIONS.map((o) => (
                  <label
                    key={o.value}
                    className={`mode-item${o.value === confirmMode ? " selected" : ""}`}
                  >
                    <HeartCheck
                      checked={o.value === confirmMode}
                      onPick={() => void pickMode(o.value)}
                    />
                    <span className="mode-item-label">{o.label}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
          <div className="effort-wrap" ref={effortRef}>
            <button
              className="effort-btn"
              onClick={() =>
                setEffortOpen((v) => {
                  if (!v) setSubOpen(null); // 重新打开时回到主菜单
                  return !v;
                })
              }
              disabled={disabled}
              aria-label="模型与推理等级"
              type="button"
            >
              <span className="effort-model">{model}</span>
              <span className="effort-level">{currentEffort.label}</span>
              <ChevronDownIcon />
            </button>
            {effortOpen && (
              <div className={`effort-menu ${subOpen === "model" ? "glass-menu" : subOpen === "effort" ? "scale-menu" : ""}`}>
                {subOpen === null && (
                  <>
                    <div className="effort-row" onClick={() => setSubOpen("model")}>
                      <span className="effort-row-label">模型</span>
                      <span className="effort-row-value">{model}</span>
                      <ChevronRightIcon />
                    </div>
                    <div className="effort-row" onClick={() => setSubOpen("effort")}>
                      <span className="effort-row-label">推理等级</span>
                      <span className="effort-row-value">{currentEffort.label}</span>
                      <ChevronRightIcon />
                    </div>
                  </>
                )}
                {subOpen === "model" && (
                  <div className="glass-fan" role="group" aria-label="选择模型">
                    {modelOptions.map((m, i) => (
                      <button
                        key={m}
                        className={`glass-option ${m === model ? "selected" : ""}`}
                        style={{ "--r": (i - (modelOptions.length - 1) / 2) * 7 } as CSSProperties}
                        onClick={() => void pickModel(m)}
                        aria-label={m}
                        aria-pressed={m === model}
                        type="button"
                        data-text={m}
                      >
                        <ModelGlyph model={m} />
                        {m === model && (
                          <span className="glass-check">
                            <CheckIcon />
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
                {subOpen === "effort" && (
                  <div className="effort-scale" role="group" aria-label="选择推理等级">
                    {EFFORT_OPTIONS.map((o) => (
                      <button
                        key={o.label}
                        className={`effort-scale-opt ${o.value === effort ? "selected" : ""}`}
                        onClick={() => void pickEffort(o.value)}
                        aria-label={o.label}
                        aria-pressed={o.value === effort}
                        type="button"
                      >
                        <EffortGlyph bars={effortBars(o.value)} />
                        <span>{o.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
          {running ? (
            <button className="send-btn stop-mode" onClick={onStop} aria-label="停止" type="button">
              <svg viewBox="0 0 320 512" height="1em" xmlns="http://www.w3.org/2000/svg" className="stop-icon pause">
                <path d="M48 64C21.5 64 0 85.5 0 112V400c0 26.5 21.5 48 48 48H80c26.5 0 48-21.5 48-48V112c0-26.5-21.5-48-48-48H48zm192 0c-26.5 0-48 21.5-48 48V400c0 26.5 21.5 48 48 48h32c26.5 0 48-21.5 48-48V112c0-26.5-21.5-48-48-48H240z" />
              </svg>
            </button>
          ) : (
            <button className="send-btn" onClick={() => void submit()} disabled={disabled || uploading || !text.trim()} aria-label="发送">
              <div className="svg-wrapper-1">
                <div className="svg-wrapper">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="none" d="M0 0h24v24H0z"></path>
                    <path
                      fill="currentColor"
                      d="M1.946 9.315c-.522-.174-.527-.455.01-.634l19.087-6.362c.529-.176.832.12.684.638l-5.454 19.086c-.15.529-.455.547-.679.045L12 14l6-8-8 6-8.054-2.685z"
                    ></path>
                  </svg>
                </div>
              </div>
            </button>
          )}
        </div>
      </div>
      <div className="input-hint">Enter 发送 · Shift+Enter 换行 · Ctrl/Cmd+V 粘贴图片</div>
    </div>
  );
}
