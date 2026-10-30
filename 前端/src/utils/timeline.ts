// 时间轴切分与派生查询。独立于渲染组件，避免组件文件导出非组件导致 Fast Refresh 失效。
import type { StreamMessage, TimelineItem } from "../types";

/** 判断一段文本是否为"噪音碎片"（仅标点/空白，无实质内容）。
 * 模型在 reasoning 模式下 content chunk 偶尔只有 "." 这样的标点，
 * 被作为独立 token emit 后会变成孤立的 text 条目，渲染出来就是一个孤零零的标点。 */
function isNoiseText(s: string): boolean {
  // 去除空白后，如果只剩标点符号（不含字母/数字/中文字符），视为噪音
  const stripped = s.replace(/\s+/g, "");
  if (!stripped) return true;
  // 只要包含一个字母、数字、中文字符或 emoji，就不是噪音
  return !/[A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff]/.test(stripped);
}

/**
 * 切分时间轴：最终回答 = 最后一个 text 条目，且它必须是 timeline 的最后一项。
 * 如果最后一个 text 后面还有 thinking/tool/plan 等条目，说明该 text 只是阶段性正文，
 * 保留在内联流中，不单独提取为最终回答——否则新思考会被"挤到"已输出文本上方。
 * 同时过滤掉噪音 text 条目（仅标点/空白的碎片）。
 * answerOf 与 ProcessTimeline 共用此函数，保证「底部最终回答」与「内联过程流」严格互补。
 */
export function splitTimeline(tl: TimelineItem[]): { items: TimelineItem[]; answer: string } {
  // 尾部可能有收尾压缩（plan/compress）：跳过它们，找到真正的末尾内容条目，
  // 否则"回答后压缩"会把最终回答误判为阶段正文
  let lastIdx = tl.length - 1;
  while (lastIdx >= 0 && (tl[lastIdx].kind === "plan" || tl[lastIdx].kind === "compress")) {
    lastIdx--;
  }
  if (lastIdx < 0) {
    return {
      items: tl.filter((it) => !(it.kind === "text" && isNoiseText(it.text))),
      answer: "",
    };
  }
  const last = tl[lastIdx];
  if (last.kind !== "text") {
    // 最后内容不是 text：过滤掉中间的噪音 text 条目
    return {
      items: tl.filter((it) => !(it.kind === "text" && isNoiseText(it.text))),
      answer: "",
    };
  }
  // 末尾（跳过尾部压缩后）是 text：提取为最终回答，同时过滤掉中间的噪音 text 条目
  const answer = isNoiseText(last.text) ? "" : last.text;
  return {
    items: tl
      .filter((_, i) => i !== lastIdx)
      .filter((it) => !(it.kind === "text" && isNoiseText(it.text))),
    answer,
  };
}

/** 提取最终回答：有 timeline 时为最后一个 text 条目；时间轴末尾不是正文
 *  （中断/停止）时回退到 msg.content，否则错误提示或最终回答会整体丢失。 */
export function answerOf(msg: StreamMessage): string {
  const tl = msg.timeline;
  if (tl && tl.length > 0) {
    const { answer } = splitTimeline(tl);
    if (answer) return answer;
    // 流式期间 content 与内联阶段正文重叠，不能回退（会重复渲染）
    if (msg.status === "streaming") return "";
    // 中断提示追加在正文之后：去掉与内联末段正文重复的前缀，只展示新增部分
    let fallback = msg.content;
    for (let i = tl.length - 1; i >= 0; i--) {
      const it = tl[i];
      if (it.kind === "text" && it.text) {
        if (fallback.startsWith(it.text)) fallback = fallback.slice(it.text.length).trim();
        break;
      }
    }
    return fallback;
  }
  return msg.content;
}

/**
 * 时间轴末尾是否存在"正在进行"的条目（思考 / 运行中的工具 / 流式输出）。
 * 返回 false 表示模型正在准备下一轮输出（包括正在生成工具参数），
 * 此时需要补一行状态提示，以免看起来卡住。
 */
export function hasLiveItem(msg: StreamMessage): boolean {
  if (msg.status !== "streaming") return false;
  const tl = msg.timeline ?? [];
  const last = tl[tl.length - 1];
  if (!last) return false;
  if (last.kind === "thinking") return last.duration_ms == null;
  if (last.kind === "tool") return last.status === "running";
  if (last.kind === "compress") return last.status === "running";
  if (last.kind === "plan") return false; // 计划已定义，等待下一轮模型输出
  return true; // text（最终回答流式输出）
}
