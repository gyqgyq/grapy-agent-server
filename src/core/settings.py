import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    AGENT_CHAT_API_KEY: str = os.getenv("AGENT_CHAT_API_KEY")
    AGENT_CHAT_MODEL: str = os.getenv("AGENT_CHAT_MODEL")
    AGENT_CHAT_TEMPERATURE: float = 0
    ASYNC_DATABASE_URL: str = os.getenv("ASYNC_DATABASE_URL", "")
    PG_POOL_MIN_SIZE: int = int(os.getenv("PG_POOL_MIN_SIZE", "2"))
    PG_POOL_MAX_SIZE: int = int(os.getenv("PG_POOL_MAX_SIZE", "10"))
    AGENT_SYSTEM_PROMPT: str = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        "你是一个帮助用户完成算术运算的助手。\n请参考以下用户历史记忆：\n{memory}",
    )
    MEMORY_VALUE_KEY: str = os.getenv("MEMORY_VALUE_KEY", "memory")
    MEMORY_MAX_CHARS: int = int(os.getenv("MEMORY_MAX_CHARS", "2000"))
    MEMORY_SEARCH_LIMIT: int = int(os.getenv("MEMORY_SEARCH_LIMIT", "3"))


settings = Settings()
