from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.agent.state import MessagesState, MemoryContext
from src.agent.nodes import llm_call, tool_node
from src.agent.edges import should_continue
from src.agent.memory import update_memory

checkpointer = InMemorySaver()
store = InMemoryStore()

# 1. 初始化图
builder = StateGraph(MessagesState, context_schema=MemoryContext)

# 2. 注册节点
builder.add_node("update_memory", update_memory)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

# 3. 配置边与条件路由
builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    source="llm_call",
    path=should_continue,
    path_map={
        "tool_node": "tool_node",
        "finish": "update_memory"
    }
)
builder.add_edge("tool_node", "llm_call")
builder.add_edge("update_memory", END)

# 4. 编译图（全局单例智能体）
agent = builder.compile(
    name="calculator_agent",
    checkpointer=checkpointer,
    store=store
)