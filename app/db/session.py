from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

engine = create_async_engine(str(settings.DATABASE_URL), echo=True)
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async_session = SessionLocal

async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
