/** 运行状态指示：心电图折线扫描（Uiverse.io by milley69）。 */
export function Loading() {
  return (
    <span className="loader-line" aria-hidden="true">
      <svg viewBox="0 0 64 48">
        <polyline className="back" points="0.157 23.954, 14 23.954, 21.843 48, 43 0, 50 24, 64 24" />
        <polyline className="front" points="0.157 23.954, 14 23.954, 21.843 48, 43 0, 50 24, 64 24" />
      </svg>
    </span>
  );
}
