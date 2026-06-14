from typing import Literal
from src.agent.state import MessagesState

def should_continue(state: MessagesState) -> Literal["tool_node", "finish"]:
    """路由判断：有工具调用则走工具节点，否则结束"""
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tool_node"
    return "finish"