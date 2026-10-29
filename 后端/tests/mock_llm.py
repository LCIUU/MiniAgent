"""Mock LLM（测试用）。"""

_GLOBAL_CALL = {"count": 0}


class MockMessage:
    """模拟 langchain AIMessage chunk。"""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockLLM:
    """模拟模型（全局调用计数，因为每轮模型调用都可能新建实例）：

    第 1 次调用：模型自主调用 write_todos 制定结构化计划
    第 2 次调用：调用 write_file 工具
    第 3+ 次调用：输出最终回答
    """

    _llm_type = "mock"  # SummarizationMiddleware 探测 token 计数器时读取

    async def astream(self, messages, **kwargs):
        _GLOBAL_CALL["count"] += 1
        n = _GLOBAL_CALL["count"]
        if n == 1:
            # 模型首轮自主调用 write_todos 制定结构化计划
            yield MockMessage(
                content="",
                tool_calls=[{
                    "id": "call_plan",
                    "name": "write_todos",
                    "args": {"todos": [
                        {"content": "创建工作文件", "status": "in_progress"},
                        {"content": "完成", "status": "pending"},
                    ]},
                }],
            )
        elif n == 2:
            yield MockMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "write_file",
                    "args": {"path": "demo.txt", "content": "hello from agent"},
                }],
            )
        else:
            yield MockMessage(content="已完成，创建了 demo.txt")

    async def ainvoke(self, messages, **kwargs):
        return MockMessage(content="summary text")

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def with_retry(self, **kwargs):
        # SummarizationMiddleware 构造时会调用 with_retry()，测试替身原样返回
        return self
