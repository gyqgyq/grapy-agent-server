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
    MEMORY_EMBEDDING_MODEL: str = os.getenv("MEMORY_EMBEDDING_MODEL", "")
    MEMORY_EMBEDDING_DIMS: int = int(os.getenv("MEMORY_EMBEDDING_DIMS", "1536"))
    MEMORY_EMBEDDING_API_KEY: str = os.getenv(
        "MEMORY_EMBEDDING_API_KEY", os.getenv("AGENT_CHAT_API_KEY", "")
    )
    MEMORY_EXTRACT_PROMPT: str = os.getenv(
        "MEMORY_EXTRACT_PROMPT",
        "从对话中提取值得长期记住的用户信息（如姓名、偏好、常用习惯等）。"
        "不要记录具体的算术题目和计算结果。"
        "若无新信息，只输出 JSON 数组：[]。"
        "否则输出 JSON 数组，每项为一条简短中文事实字符串。只输出 JSON，不要其他文字。",
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


settings = Settings()
