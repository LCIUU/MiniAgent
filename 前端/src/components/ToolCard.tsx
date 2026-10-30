import { useEffect, useState } from "react";
import type { ApprovalRequest, QuestionRequest } from "../types";

export function ApprovalCard({
  approval,
  onDecide,
}: {
  approval: ApprovalRequest;
  onDecide: (approve: boolean) => void;
}) {
  return (
    <div className="approval-card">
      <div className="title">
        ⚠ 需要确认 · {approval.reason === "high_risk" ? "高危命令" : "命令审批"}
      </div>
      <div className="cmd-card">
        <div className="code-card-head">
          <span className="code-dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span className="code-card-name">{approval.tool}</span>
        </div>
        <div className="cmd">
          {approval.command ?? (approval.tool === "terminal" ? "执行命令" : `调用工具 ${approval.tool}`)}
        </div>
      </div>
      {approval.cwd && <div style={{ fontSize: 11, color: "var(--text-40)" }}>目录: {approval.cwd}</div>}
      <div className="actions">
        <button className="approve" onClick={() => onDecide(true)}>
          允许
        </button>
        <button className="reject" onClick={() => onDecide(false)}>
          拒绝
        </button>
      </div>
    </div>
  );
}

/** 选项自身带的编号/列表符号：展示时用 A/B/C 徽标替代，避免“A 1.”重复。 */
const OPTION_PREFIX = /^\s*(?:[-*•]\s+|\d{1,2}[.、)）]\s*|[A-Za-z][.、)）]\s*)/;
const LIST_LINE = /^\s*(?:[-*•]|\d{1,2}[.、)）])\s+(.+)$/;

const cleanOption = (text: string) => text.replace(OPTION_PREFIX, "").trim();

/**
 * 模型有时把候选方向写进问题正文（“1. A  2. B …”），options 为空。
 * 把尾部连续的列表还原成选项按钮，正文只保留引导句。
 */
function splitInlineOptions(question: string): { question: string; options: string[] } {
  const lines = question.split("\n");
  const picked: string[] = [];
  let start = lines.length;
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line.trim()) continue;
    const m = line.match(LIST_LINE);
    if (!m) break;
    const label = m[1].trim();
    if (!label || label.length > 40) return { question, options: [] };
    picked.unshift(label);
    start = i;
  }
  if (picked.length < 2) return { question, options: [] };
  const head = lines.slice(0, start).join("\n").trim();
  if (!head) return { question, options: [] };
  return { question: head, options: picked };
}

/** 结构化提问卡片：选项点一下就提交（单选直接发送），也随时可以打字描述。 */
export function QuestionCard({
  request,
  onAnswer,
}: {
  request: QuestionRequest;
  onAnswer: (answers: string[]) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [sent, setSent] = useState(false);
  const [submitted, setSubmitted] = useState<string[]>([]);
  const inline = splitInlineOptions(request.question ?? "");
  const declared = (request.options ?? []).map(cleanOption).filter(Boolean);
  const options = declared.length ? declared : inline.options;
  // 正文尾部列表与 options 重复（模型把候选写了两遍）时，只保留按钮并剥掉正文列表
  const useInlineBody =
    options.length > 0 && inline.options.length > 0 && inline.options.every((o) => options.includes(o));
  const body = useInlineBody ? inline.question : request.question;
  const multiple = request.allow_multiple && options.length > 1;

  useEffect(() => {
    setSelected([]);
    setText("");
    setSent(false);
    setSubmitted([]);
  }, [request.question_id]);

  const submit = (answers: string[]) => {
    if (sent) return;
    const picked = answers.map((a) => a.trim()).filter(Boolean);
    if (!picked.length) return;
    setSubmitted(picked);
    setSent(true);
    onAnswer(picked);
  };

  const canSubmit = !sent && (text.trim().length > 0 || (multiple && selected.length > 0));
  const submitFree = () => submit(multiple ? [...selected, text] : [text]);

  if (sent) {
    return (
      <div className="question-card sent">
        <div className="q-head">
          <span className="q-sent-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 13l4 4L19 7" />
            </svg>
          </span>
          <span className="q-title">已提交回答</span>
        </div>
        <div className="q-answer">{submitted.join("；")}</div>
      </div>
    );
  }

  return (
    <div className="question-card">
      <div className="q-head">
        <span className="q-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l2.2-3.1A8.5 8.5 0 1 1 21 11.5Z" />
            <path d="M9.6 9.2a2.5 2.5 0 1 1 3.4 2.3c-.7.3-1 .9-1 1.6v.4" />
            <path d="M12 16.6h.01" />
          </svg>
        </span>
        <span className="q-title">需要你确认</span>
        {options.length > 0 && <span className="q-tag">{multiple ? "可多选" : "单选"}</span>}
      </div>
      <div className="q-text">{body}</div>

      {options.length > 0 && (
        <div className="q-options" role={multiple ? "group" : undefined}>
          {options.map((opt, i) => {
            const checked = selected.includes(opt);
            return (
              <button
                key={`${i}-${opt}`}
                type="button"
                className={`q-option${checked ? " checked" : ""}`}
                aria-pressed={multiple ? checked : undefined}
                onClick={() => {
                  if (multiple) {
                    setSelected((prev) =>
                      prev.includes(opt) ? prev.filter((x) => x !== opt) : [...prev, opt],
                    );
                  } else {
                    submit([opt]);
                  }
                }}
              >
                <span className="q-key">{String.fromCharCode(65 + i)}</span>
                <span className="q-label">{opt}</span>
              </button>
            );
          })}
        </div>
      )}

      <div className={`q-free${options.length ? "" : " solo"}`}>
        <div className="q-free-label">
          {options.length > 0 ? "其他答案 · 也可以自行描述" : "你的回答"}
        </div>
        <div className="q-free-row">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={multiple && selected.length ? "补充说明，提交时会连同已选项一起发送…" : "直接输入你的答案…"}
            rows={1}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitFree();
              }
            }}
          />
          <button type="button" className="q-submit" disabled={!canSubmit} onClick={submitFree}>
            {multiple && selected.length > 1 ? `提交（${selected.length}）` : "提交"}
          </button>
        </div>
      </div>
    </div>
  );
}
