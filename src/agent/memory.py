import json
import uuid

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from src.agent.config import llm
from src.agent.state import MemoryContext, MessagesState
from src.core.logging import get_logger
from src.core.settings import settings

logger = get_logger(__name__)


def memory_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "memories")


def should_load_cross_thread_memory(messages: list[AnyMessage]) -> bool:
    """同 thread 续聊时 Checkpointer 已含完整历史，无需再查 Store。"""
    return len(messages) <= 1


def _format_messages(messages: list[AnyMessage]) -> str:
    lines = []
    for msg in messages:
        lines.append(f"{msg.type}: {msg.content}")
    return "\n".join(lines)


def _format_recent_exchange(messages: list[AnyMessage]) -> str:
    """取最后一轮用户消息及其后的助手/工具回复，用于提炼新事实。"""
    last_human_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].type == "human":
            last_human_idx = idx
            break
    if last_human_idx < 0:
        return _format_messages(messages)
    return _format_messages(messages[last_human_idx:])


def _parse_facts(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("记忆提炼结果不是合法 JSON: %s", text[:200])
        return []

    if not isinstance(parsed, list):
        return []

    facts = []
    for item in parsed:
        if isinstance(item, str):
            fact = item.strip()
            if fact:
                facts.append(fact)
    return facts


async def _extract_facts(messages: list[AnyMessage]) -> list[str]:
    exchange = _format_recent_exchange(messages)
    if not exchange.strip():
        return []

    response = await llm.ainvoke(
        [
            SystemMessage(content=settings.MEMORY_EXTRACT_PROMPT),
            HumanMessage(content=exchange),
        ]
    )
    return _parse_facts(str(response.content))


async def update_memory(state: MessagesState, runtime: Runtime[MemoryContext]):
    messages = state.get("messages") or []
    if not messages:
        return

    facts = await _extract_facts(messages)
    if not facts:
        logger.debug("本轮对话无新增长期记忆: user_id=%s", runtime.context.user_id)
        return

    namespace = memory_namespace(runtime.context.user_id)
    for fact in facts:
        memory_id = str(uuid.uuid4())
        await runtime.store.aput(
            namespace,
            memory_id,
            {settings.MEMORY_VALUE_KEY: fact},
        )
    logger.info(
        "已写入 %s 条长期记忆: user_id=%s",
        len(facts),
        runtime.context.user_id,
    )
