from langgraph.graph import StateGraph, START, END

from src.agent.state import MessagesState, MemoryContext
from src.agent.nodes import llm_call, tool_node
from src.agent.edges import should_continue
from src.agent.memory import update_memory


def build_agent(checkpointer, store):
    builder = StateGraph(MessagesState, context_schema=MemoryContext)

    builder.add_node("update_memory", update_memory)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        source="llm_call",
        path=should_continue,
        path_map={
            "tool_node": "tool_node",
            "finish": "update_memory",
        },
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("update_memory", END)

    return builder.compile(
        name="calculator_agent",
        checkpointer=checkpointer,
        store=store,
    )
