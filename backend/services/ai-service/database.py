import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from base import Base

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DB_URI")

if not DATABASE_URL:
    raise ValueError("DB_URI environment variable not set")

# Ensure the URL uses the asyncpg driver scheme if it's a standard postgresql:// URL
# SQLAlchemy 2.x prefers postgresql+asyncpg://
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif not DATABASE_URL.startswith("postgresql+asyncpg://"):
    raise ValueError(f"Unsupported database URL scheme: {DATABASE_URL}. Expected postgresql:// or postgresql+asyncpg://")

# Create the async engine
# echo=True will log SQL statements, useful for debugging, disable in production
engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    pool_recycle=3600,
    pool_pre_ping=True,
)


# Create an async session factory
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # Recommended for async sessions
    class_=AsyncSession
)

# Dependency to get DB session in FastAPI endpoints (example)
async def get_db_session() -> AsyncSession:
    async with AsyncSessionFactory() as session:
        try:
            yield session
        finally:
            await session.commit()
            await session.close()
# where should i commit this?

# Example of how to initialize tables (call this once at startup, e.g., in main.py)
async def init_db():
    async with engine.begin() as conn:
        #await conn.run_sync(Base.metadata.drop_all) # Use with caution
        await conn.run_sync(Base.metadata.create_all)
