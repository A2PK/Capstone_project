import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from .model import AIModelSQL, AIModelPydantic # Assuming Pydantic model needed for return typing

logger = logging.getLogger(__name__)

class ModelRepository(ABC):
    """Interface for AI Model data access."""

    @abstractmethod
    async def create(self, model_data: AIModelSQL) -> AIModelSQL:
        pass

    @abstractmethod
    async def get_by_id(self, model_id: UUID) -> Optional[AIModelSQL]:
        pass

    @abstractmethod
    async def get_by_name_and_version(self, name: str, version: str) -> Optional[AIModelSQL]:
        pass

    @abstractmethod
    async def list_all(self, offset: int, limit: int) -> Tuple[List[AIModelSQL], int]:
        pass

    @abstractmethod
    async def list_by_station(self, station_id: UUID, offset: int, limit: int) -> Tuple[List[AIModelSQL], int]:
        pass

    @abstractmethod
    async def update(self, model_id: UUID, update_data: dict) -> Optional[AIModelSQL]:
        pass

    @abstractmethod
    async def delete(self, model_id: UUID) -> bool:
        pass # Simple delete, no soft/hard distinction for now

    @abstractmethod
    async def find_newest_by_station_and_type(self, station_id: UUID, model_type_pattern: str) -> Optional[AIModelSQL]:
        pass


class SQLAlchemyModelRepository(ModelRepository):
    """SQLAlchemy implementation of the ModelRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, model_sql: AIModelSQL) -> AIModelSQL:
        try:
            self.session.add(model_sql)
            await self.session.flush() # Flush to get potential errors like unique constraints
            await self.session.refresh(model_sql) # Refresh to get ID, defaults, etc.
            logger.info(f"Created AI Model record with ID: {model_sql.id}")
            return model_sql
        except IntegrityError as e: # Catch unique constraint violations, etc.
            await self.session.rollback()
            logger.error(f"Database integrity error creating model: {e}")
            # Consider raising a custom RepositoryError or UseCaseError
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating model record: {e}")
            raise

    async def get_by_id(self, model_id: UUID) -> Optional[AIModelSQL]:
        try:
            result = await self.session.get(AIModelSQL, model_id)
            return result
        except NoResultFound:
            return None
        except Exception as e:
            logger.error(f"Error retrieving model by ID {model_id}: {e}")
            raise

    async def get_by_name_and_version(self, name: str, version: str) -> Optional[AIModelSQL]:
        try:
            stmt = select(AIModelSQL).where(AIModelSQL.name == name, AIModelSQL.version == version)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except NoResultFound:
            return None # Should be handled by scalar_one_or_none, but belt-and-suspenders
        except Exception as e:
            logger.error(f"Error retrieving model by name '{name}' version '{version}': {e}")
            raise

    async def _count_all(self, base_stmt) -> int:
        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        result = await self.session.execute(count_stmt)
        return result.scalar_one()

    async def list_all(self, offset: int, limit: int) -> Tuple[List[AIModelSQL], int]:
        try:
            base_stmt = select(AIModelSQL)
            total = await self._count_all(base_stmt)

            stmt = base_stmt.offset(offset).limit(limit).order_by(AIModelSQL.created_at.desc())
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            return models, total
        except Exception as e:
            logger.error(f"Error listing all models: {e}")
            raise

    async def list_by_station(self, station_id: UUID, offset: int, limit: int) -> Tuple[List[AIModelSQL], int]:
        try:
            base_stmt = select(AIModelSQL).where(AIModelSQL.station_id == station_id)
            total = await self._count_all(base_stmt)

            stmt = base_stmt.offset(offset).limit(limit).order_by(AIModelSQL.created_at.desc())
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            return models, total
        except Exception as e:
            logger.error(f"Error listing models for station {station_id}: {e}")
            raise

    async def update(self, model_id: UUID, update_data: dict) -> Optional[AIModelSQL]:
        # Filter out None values from update_data
        update_values = {k: v for k, v in update_data.items() if v is not None}
        if not update_values:
            # If nothing to update, maybe fetch and return the existing?
            return await self.get_by_id(model_id)

        try:
            stmt = (
                update(AIModelSQL)
                .where(AIModelSQL.id == model_id)
                .values(**update_values)
                .returning(AIModelSQL) # Return the updated row
            )

            result = await self.session.execute(stmt)
            await self.session.flush()
            updated_model = result.scalar_one_or_none()
            if updated_model:
                 await self.session.refresh(updated_model) # Ensure relationships etc. are loaded if needed
                 logger.info(f"Updated model ID: {model_id}")
                 return updated_model
            else:
                logger.warn(f"Attempted to update non-existent model ID: {model_id}")
                return None
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Database integrity error updating model {model_id}: {e}")
            raise
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating model {model_id}: {e}")
            raise

    async def delete(self, model_id: UUID) -> bool:
        # Currently implementing hard delete for simplicity
        # Soft delete would involve setting deleted_at
        try:
            stmt = delete(AIModelSQL).where(AIModelSQL.id == model_id)
            result = await self.session.execute(stmt)
            await self.session.flush()
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"Deleted model ID: {model_id}")
                return True
            else:
                logger.warn(f"Attempted to delete non-existent model ID: {model_id}")
                return False
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting model {model_id}: {e}")
            raise

    async def find_newest_by_station_and_type(self, station_id: UUID, model_type_pattern: str) -> Optional[AIModelSQL]:
        try:
            # Assuming model name format like "<type>_<datetag>"
            name_pattern = f"{model_type_pattern}%"
            stmt = select(AIModelSQL)\
                .where(AIModelSQL.station_id == station_id, AIModelSQL.name.like(name_pattern))\
                .order_by(AIModelSQL.created_at.desc())\
                .limit(1)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error finding newest model for station {station_id} type pattern {model_type_pattern}: {e}")
            raise
