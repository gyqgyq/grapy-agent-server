import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    AGENT_CHAT_API_KEY: str = os.getenv("AGENT_CHAT_API_KEY")
    AGENT_CHAT_MODEL: str = os.getenv("AGENT_CHAT_MODEL")
    AGENT_CHAT_TEMPERATURE: float = 0
settings = Settings()