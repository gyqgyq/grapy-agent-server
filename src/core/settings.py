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


settings = Settings()
