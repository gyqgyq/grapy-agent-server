from langchain.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from src.agent.config import llm
from src.agent.tools import tools, tools_by_name
from src.agent.state import MessagesState, MemoryContext

# 绑定工具到模型（节点内复用）
model_with_tools = llm.bind_tools(tools)

async def llm_call(state: MessagesState, runtime: Runtime[MemoryContext]):
    """LLM 决策是否调用工具，并结合用户历史记忆生成回复"""

    # 1. 获取最新用户消息，防止消息列表为空
    messages = state.get("messages", [])
    if not messages:
        latest_msg = ""
    else:
        latest_msg = messages[-1].content

    # 从运行时上下文取用户ID，构造记忆命名空间
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    # 异步检索历史记忆
    memory_text = ""
    try:
        memories = await runtime.store.asearch(
            namespace=namespace,
            query=latest_msg,
            limit=3
        )
        memory_text = "\n".join([d.value.get("memory", "") for d in memories])
    except Exception as e:
        # 记忆检索异常兜底，不中断主流程
        print(f"记忆检索异常: {e}")

    # 4. 拼接系统提示词 + 历史记忆
    system_prompt = (
        "You are a helpful assistant tasked with performing arithmetic on a set of inputs.\n"
        f"Refer to the user's historical memories below:\n{memory_text}"
    )
    # 5. 使用异步 ainvoke 调用 LLM，不阻塞事件循环
    response = await model_with_tools.ainvoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tool_node(state: dict):
    """执行 LLM 返回的工具调用。"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}