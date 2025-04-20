import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from uuid import UUID
from datetime import datetime

from .model import AIModelSQL, AIModelPydantic, AIModelCreate, AIModelUpdate
from .model_repo import ModelRepository

logger = logging.getLogger(__name__)

# Custom Exception for Use Case layer
class UseCaseError(Exception):
    pass

class NotFoundError(UseCaseError):
    pass

class ModelUseCase(ABC):
    """Interface for AI Model business logic."""

    @abstractmethod
    async def create_model(self, model_create: AIModelCreate) -> AIModelPydantic:
        pass

    @abstractmethod
    async def get_model_by_id(self, model_id: UUID) -> Optional[AIModelPydantic]:
        pass

    @abstractmethod
    async def get_model_by_name_and_version(self, name: str, version: str) -> Optional[AIModelPydantic]:
        pass

    @abstractmethod
    async def list_models(self, offset: int, limit: int) -> Tuple[List[AIModelPydantic], int]:
        pass

    @abstractmethod
    async def list_models_by_station(self, station_id: UUID, offset: int, limit: int) -> Tuple[List[AIModelPydantic], int]:
        pass

    @abstractmethod
    async def update_model(self, model_id: UUID, model_update: AIModelUpdate) -> Optional[AIModelPydantic]:
        pass

    @abstractmethod
    async def delete_model(self, model_id: UUID) -> bool:
        pass

    @abstractmethod
    async def find_newest_model_for_station(self, station_id: UUID, model_type: str) -> Optional[AIModelPydantic]:
        pass


class AIModelService(ModelUseCase):
    """Implementation of the AI Model business logic."""

    def __init__(self, repository: ModelRepository):
        self.repository = repository

    async def create_model(self, model_create: AIModelCreate) -> AIModelPydantic:
        logger.info(f"Use case: Creating model {model_create.name} v{model_create.version}")
        # Basic validation example (more complex validation could go here)
        existing = await self.repository.get_by_name_and_version(model_create.name, model_create.version)
        if existing:
            raise UseCaseError(f"Model with name '{model_create.name}' and version '{model_create.version}' already exists.")

        # In a real scenario, file_path would be determined by saving the model file
        # For now, we assume it's somehow known or set later.
        # Let's default it to something placeholder if not provided, though the DB requires it.
        # This mapping assumes the existence of a file upload/management mechanism elsewhere.
        model_sql = AIModelSQL(
            name=model_create.name,
            version=model_create.version,
            description=model_create.description,
            trained_at=model_create.trained_at,
            station_id=model_create.station_id,
            availability=model_create.availability,
            file_path=model_create.file_path, # Placeholder!
            parameter_list=model_create.parameter_list
        )
        created_sql = await self.repository.create(model_sql)
        return AIModelPydantic.model_validate(created_sql)

    async def get_model_by_id(self, model_id: UUID) -> Optional[AIModelPydantic]:
        logger.info(f"Use case: Getting model by ID {model_id}")
        model_sql = await self.repository.get_by_id(model_id)
        if not model_sql:
            return None
        return AIModelPydantic.model_validate(model_sql)

    async def get_model_by_name_and_version(self, name: str, version: str) -> Optional[AIModelPydantic]:
        logger.info(f"Use case: Getting model by name '{name}' version '{version}'")
        model_sql = await self.repository.get_by_name_and_version(name, version)
        if not model_sql:
            return None
        return AIModelPydantic.model_validate(model_sql)

    async def list_models(self, offset: int, limit: int) -> Tuple[List[AIModelPydantic], int]:
        logger.info(f"Use case: Listing models, offset={offset}, limit={limit}")
        models_sql, total = await self.repository.list_all(offset=offset, limit=limit)
        models_pydantic = [AIModelPydantic.model_validate(m) for m in models_sql]
        return models_pydantic, total

    async def list_models_by_station(self, station_id: UUID, offset: int, limit: int) -> Tuple[List[AIModelPydantic], int]:
        logger.info(f"Use case: Listing models for station {station_id}, offset={offset}, limit={limit}")
        models_sql, total = await self.repository.list_by_station(station_id=station_id, offset=offset, limit=limit)
        models_pydantic = [AIModelPydantic.model_validate(m) for m in models_sql]
        return models_pydantic, total

    async def update_model(self, model_id: UUID, model_update: AIModelUpdate) -> Optional[AIModelPydantic]:
        logger.info(f"Use case: Updating model ID {model_id}")
        # Convert Pydantic update model to dict, excluding unset fields
        update_data = model_update.model_dump(exclude_unset=True)

        if not update_data:
             logger.warn(f"Update called for model ID {model_id} with no fields to update.")
             # Optionally fetch and return the existing model if no update data is provided
             existing_model = await self.get_model_by_id(model_id)
             if not existing_model:
                 raise NotFoundError(f"Model with ID '{model_id}' not found for update.")
             return existing_model

        updated_sql = await self.repository.update(model_id, update_data)
        if not updated_sql:
             raise NotFoundError(f"Model with ID '{model_id}' not found for update.")

        return AIModelPydantic.model_validate(updated_sql)

    async def delete_model(self, model_id: UUID) -> bool:
        logger.info(f"Use case: Deleting model ID {model_id}")
        deleted = await self.repository.delete(model_id)
        if not deleted:
            raise NotFoundError(f"Model with ID '{model_id}' not found for deletion.")
        return deleted

    async def find_newest_model_for_station(self, station_id: UUID, model_type: str) -> Optional[AIModelPydantic]:
        logger.info(f"Use case: Finding newest model for station {station_id} type {model_type}")
        # We pass the model_type directly as the pattern start
        model_sql = await self.repository.find_newest_by_station_and_type(station_id, model_type)
        if not model_sql:
            logger.warning(f"No model found for station {station_id} type {model_type}")
            return None
        # Ensure availability if needed - adding check here
        if not model_sql.availability:
             logger.warning(f"Newest model found for station {station_id} type {model_type} (ID: {model_sql.id}) is not available.")
             return None
        return AIModelPydantic.model_validate(model_sql)
