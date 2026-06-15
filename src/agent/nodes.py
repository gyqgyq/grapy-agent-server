from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from src.agent.config import llm
from src.agent.tools import tools, tools_by_name
from src.agent.memory import memory_namespace, should_load_cross_thread_memory
from src.agent.state import MemoryContext, MessagesState
from src.core.logging import get_logger
from src.core.settings import settings

logger = get_logger(__name__)

model_with_tools = llm.bind_tools(tools)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_system_prompt(memory_text: str) -> str:
    truncated = _truncate_text(memory_text, settings.MEMORY_MAX_CHARS)
    return settings.AGENT_SYSTEM_PROMPT.format(memory=truncated or "（无）")


def _build_llm_messages(
    system_prompt: str,
    messages: list[AnyMessage],
) -> list[AnyMessage]:
    return [SystemMessage(content=system_prompt), *messages]


async def _load_memory_text(
    runtime: Runtime[MemoryContext],
    namespace: tuple[str, str],
    query: str,
) -> str:
    try:
        memories = await runtime.store.asearch(
            namespace,
            query=query,
            limit=settings.MEMORY_SEARCH_LIMIT,
        )
        parts = []
        for item in memories:
            value = item.value.get(settings.MEMORY_VALUE_KEY, "")
            if value:
                parts.append(str(value))
        return "\n".join(parts)
    except Exception:
        logger.exception("记忆检索失败: user_id=%s", runtime.context.user_id)
        return ""


async def llm_call(state: MessagesState, runtime: Runtime[MemoryContext]):
    """LLM 决策是否调用工具，并结合用户历史记忆生成回复"""
    messages = state.get("messages") or []
    latest_msg = str(messages[-1].content) if messages else ""

    memory_text = ""
    if should_load_cross_thread_memory(messages):
        namespace = memory_namespace(runtime.context.user_id)
        memory_text = await _load_memory_text(runtime, namespace, latest_msg)

    system_prompt = _build_system_prompt(memory_text)

    response = await model_with_tools.ainvoke(
        _build_llm_messages(system_prompt, messages)
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def tool_node(state: MessagesState):
    """执行 LLM 返回的工具调用。"""
    messages = state.get("messages") or []
    if not messages:
        logger.warning("tool_node 收到空消息列表，跳过工具调用")
        return {"messages": []}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        logger.warning("tool_node 未找到 tool_calls，跳过工具调用")
        return {"messages": []}

    result: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        tool = tools_by_name.get(tool_name)
        if tool is None:
            logger.error("未知工具: %s", tool_name)
            result.append(
                ToolMessage(
                    content=f"未知工具: {tool_name}",
                    tool_call_id=tool_call_id,
                )
            )
            continue

        try:
            observation = await tool.ainvoke(tool_call["args"])
        except Exception as exc:
            logger.exception("工具调用失败: tool=%s", tool_name)
            observation = f"工具执行失败: {exc}"

        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    return {"messages": result}
