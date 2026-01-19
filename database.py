import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from models import Base

logger = logging.getLogger(__name__)

# データベースURLの設定
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./plant_diagnosis.db",
)

# SQLiteの場合は特別な接続プール設定を使用
_is_sqlite = SQLALCHEMY_DATABASE_URL.startswith("sqlite")

# 非同期エンジンの作成（接続プール設定付き）
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,
    future=True,
    # SQLiteの場合はStaticPool（シングル接続）を使用
    # PostgreSQL等の場合はpool_pre_ping/pool_recycleを設定
    **({
        "poolclass": StaticPool,
        "connect_args": {"check_same_thread": False},
    } if _is_sqlite else {
        "pool_pre_ping": True,  # 古い接続を検出
        "pool_recycle": 3600,   # 1時間で接続をリサイクル
    })
)

# 非同期セッションの設定
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """
    データベースの初期化を行う関数
    テーブルが存在しない場合は作成する
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """
    データベースセッションを取得するための非同期ジェネレータ

    Yields:
        AsyncSession: データベースセッション

    Notes:
        - yield前の例外はそのまま伝播
        - yield後の例外はロールバック後に再送出
        - 正常終了時はコミット
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            logger.exception("Database transaction rolled back due to error")
            raise
        else:
            # 例外がなければコミット
            await session.commit()
        finally:
            await session.close()
