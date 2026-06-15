from contextlib import asynccontextmanager

from langchain.embeddings import init_embeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from psycopg import InterfaceError, OperationalError
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.core.logging import get_logger
from src.core.settings import settings

logger = get_logger(__name__)


class PostgresPersistenceError(RuntimeError):
    """PostgreSQL 持久化初始化失败时抛出。"""


def normalize_database_url(url: str) -> str:
    scheme, separator, rest = url.partition("://")
    if separator and scheme.startswith("postgresql+"):
        return f"postgresql://{rest}"
    return url


def build_store_index() -> PostgresIndexConfig | None:
    """构建 Store 向量索引配置；未配置 embedding 模型时返回 None。"""
    model = settings.MEMORY_EMBEDDING_MODEL.strip()
    if not model:
        logger.info(
            "未配置 MEMORY_EMBEDDING_MODEL，Store 语义检索已禁用。"
            "可在 .env 中设置，例如 openai:text-embedding-3-small"
        )
        return None

    embed_kwargs: dict = {}
    if settings.MEMORY_EMBEDDING_API_KEY:
        embed_kwargs["api_key"] = settings.MEMORY_EMBEDDING_API_KEY

    embeddings = init_embeddings(model, **embed_kwargs)
    return PostgresIndexConfig(
        embed=embeddings,
        dims=settings.MEMORY_EMBEDDING_DIMS,
        fields=[settings.MEMORY_VALUE_KEY],
    )


def get_database_url() -> str:
    url = settings.ASYNC_DATABASE_URL.strip()
    if not url:
        raise PostgresPersistenceError(
            "未配置 ASYNC_DATABASE_URL。"
            "示例：postgresql://user:password@localhost:5432/dbname"
        )
    return normalize_database_url(url)


class PostgresPersistence:
    """应用级 PostgreSQL 持久化，使用共享异步连接池。"""

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int,
        max_size: int,
    ) -> None:
        self._database_url = database_url
        self._min_size = min_size
        self._max_size = max_size
        self._pool: AsyncConnectionPool | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None

    @property
    def is_started(self) -> bool:
        return self._pool is not None

    async def start(self) -> tuple[AsyncPostgresSaver, AsyncPostgresStore]:
        if self._checkpointer is not None and self._store is not None:
            return self._checkpointer, self._store

        pool = AsyncConnectionPool(
            self._database_url,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

        try:
            await pool.open(wait=True)
            store = AsyncPostgresStore(conn=pool, index=build_store_index())
            checkpointer = AsyncPostgresSaver(conn=pool)
            await store.setup()
            await checkpointer.setup()
        except (OperationalError, InterfaceError, OSError, TimeoutError) as exc:
            await pool.close()
            logger.exception("PostgreSQL 持久化初始化失败")
            raise PostgresPersistenceError(
                "无法连接 PostgreSQL 或执行持久化 setup，"
                "请检查 ASYNC_DATABASE_URL、网络连通性与数据库权限。"
            ) from exc
        except Exception:
            await pool.close()
            logger.exception("PostgreSQL 持久化 setup 过程中发生未知错误")
            raise

        self._pool = pool
        self._store = store
        self._checkpointer = checkpointer
        logger.info(
            "PostgreSQL 持久化已就绪（连接池 min=%s max=%s）",
            self._min_size,
            self._max_size,
        )
        return checkpointer, store

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.close()
        self._pool = None
        self._checkpointer = None
        self._store = None


@asynccontextmanager
async def postgres_persistence():
    """初始化并在应用生命周期内持有 PostgreSQL checkpointer 与 store。"""
    persistence = PostgresPersistence(
        get_database_url(),
        min_size=settings.PG_POOL_MIN_SIZE,
        max_size=settings.PG_POOL_MAX_SIZE,
    )
    try:
        checkpointer, store = await persistence.start()
        yield checkpointer, store
    finally:
        await persistence.stop()
