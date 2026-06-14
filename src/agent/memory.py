import uuid

from langgraph.runtime import Runtime

from src.agent.state import MemoryContext, MessagesState
from src.core.settings import settings



async def update_memory(state: MessagesState, runtime: Runtime[MemoryContext]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    await runtime.store.aput(
        namespace,
        memory_id,
        {settings.MEMORY_VALUE_KEY: state},
    )