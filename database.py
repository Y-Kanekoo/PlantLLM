from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from models import Base

# データベースURLの設定
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./plant_diagnosis.db"

# 非同期エンジンの作成
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,  # SQLログを出力
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
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
