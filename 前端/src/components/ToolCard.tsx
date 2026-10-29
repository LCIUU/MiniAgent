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

/** 结构化提问卡片：单选点选即提交；多选/自由输入需点提交。 */
export function QuestionCard({
  request,
  onAnswer,
}: {
  request: QuestionRequest;
  onAnswer: (answers: string[]) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [text, setText] = useState("");
  const options = request.options ?? [];
  const multiple = request.allow_multiple && options.length > 1;

  useEffect(() => {
    setSelected([]);
    setText("");
  }, [request.question_id]);

  const submit = (answers: string[]) => {
    const picked = answers.map((a) => a.trim()).filter(Boolean);
    if (picked.length) onAnswer(picked);
  };

  return (
    <div className="question-card">
      <div className="title">需要你确认 · {request.tool}</div>
      <div className="q-text">{request.question}</div>
      {options.length === 0 ? (
        <div className="q-free">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="输入你的回答…"
            rows={2}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit([text]);
              }
            }}
          />
          <button type="button" disabled={!text.trim()} onClick={() => submit([text])}>
            提交
          </button>
        </div>
      ) : multiple ? (
        <>
          <div className="q-options">
            {options.map((opt) => {
              const checked = selected.includes(opt);
              return (
                <label key={opt} className={`q-option${checked ? " checked" : ""}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() =>
                      setSelected((prev) =>
                        prev.includes(opt) ? prev.filter((x) => x !== opt) : [...prev, opt],
                      )
                    }
                  />
                  {opt}
                </label>
              );
            })}
          </div>
          <div className="actions">
            <button type="button" disabled={!selected.length} onClick={() => submit(selected)}>
              提交
            </button>
          </div>
        </>
      ) : (
        <div className="actions">
          {options.map((opt) => (
            <button key={opt} type="button" onClick={() => submit([opt])}>
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
