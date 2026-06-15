import uuid

from langgraph.runtime import Runtime

from src.agent.state import MemoryContext, MessagesState
from src.core.settings import settings


def _serialize_state_for_memory(state: MessagesState) -> str:
    """将对话状态序列化为可 JSON 存储的文本摘要。"""
    lines = []
    for msg in state.get("messages", []):
        lines.append(f"{msg.type}: {msg.content}")
    llm_calls = state.get("llm_calls")
    if llm_calls is not None:
        lines.append(f"llm_calls: {llm_calls}")
    return "\n".join(lines)


async def update_memory(state: MessagesState, runtime: Runtime[MemoryContext]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    await runtime.store.aput(
        namespace,
        memory_id,
        {settings.MEMORY_VALUE_KEY: _serialize_state_for_memory(state)},
    )