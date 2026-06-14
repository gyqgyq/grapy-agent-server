import asyncio
import selectors
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.logging import setup_logging
from src.agent.graph import build_agent
from src.api.routers import router
from src.db.postgres import postgres_persistence

setup_logging()


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Windows 下供 uvicorn 使用：`--loop src.main:selector_loop_factory`。"""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.SelectorEventLoop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化 Agent，关闭时释放 PostgreSQL 连接池。"""
    async with postgres_persistence() as (checkpointer, store):
        app.state.agent = build_agent(checkpointer, store)
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
