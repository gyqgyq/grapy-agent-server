from langchain.chat_models import init_chat_model

from src.core.settings import settings

# 初始化 LLM 实例（全局单例）
llm = init_chat_model(
    model=settings.AGENT_CHAT_MODEL,
    api_key=settings.AGENT_CHAT_API_KEY,
    temperature=settings.AGENT_CHAT_TEMPERATURE
)