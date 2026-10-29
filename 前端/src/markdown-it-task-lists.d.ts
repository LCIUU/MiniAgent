// markdown-it-task-lists 无官方类型，手动声明
declare module "markdown-it-task-lists" {
  import type MarkdownIt from "markdown-it";

  interface TaskListsOptions {
    /** true 时复选框可交互 */
    enabled?: boolean;
    /** true 时用 <label> 包裹文本 */
    label?: boolean;
    lineNumber?: boolean;
  }

  const taskLists: (md: MarkdownIt, options?: TaskListsOptions) => void;
  export default taskLists;
}
