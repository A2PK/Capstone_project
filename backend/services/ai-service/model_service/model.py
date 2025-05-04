from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Text, DateTime, Boolean, Uuid, ARRAY, func
from base import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, List

# Represents the Go pkg/core/entity/BaseEntity as SQLAlchemy Base
# Using timezone=True for aware timestamps (stored typically as UTC in DB)
# Using server_default/server_onupdate for DB-generated timestamps
class BaseEntityMapped:
    id = Column(Uuid, primary_key=True, default=uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)

# SQLAlchemy model for AIModel
class AIModelSQL(Base, BaseEntityMapped):
    __tablename__ = 'ai_models' # Choose an appropriate table name

    name = Column(String, nullable=False, index=True) # Index for find by name
    version = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    trained_at = Column(DateTime(timezone=True), nullable=False)
    station_id = Column(Uuid, nullable=False, index=True) # Index for find by station
    availability = Column(Boolean, nullable=False, default=True)
    parameter_list = Column(ARRAY(String), nullable=True) # Store list of parameters used for training

    # Add unique constraint if needed ( handled by GORM hook in Go, can be added here)
    # from sqlalchemy import UniqueConstraint
    # __table_args__ = (UniqueConstraint('name', 'version', name='uq_model_name_version'),)

# --- Pydantic Models --- (Keep existing ones)

# Represents the Go pkg/core/entity/BaseEntity
class BaseEntityModel(BaseModel):
    """Base model reflecting the common fields from Go's BaseEntity."""
    id: UUID = Field(..., description="Unique identifier (UUID format)")
    created_at: datetime = Field(..., description="Timestamp when the entity was created")
    updated_at: datetime = Field(..., description="Timestamp when the entity was last updated")
    deleted_at: Optional[datetime] = Field(None, description="Timestamp when the entity was soft-deleted (null if not deleted)")

    # Configuration for Pydantic model behavior
    model_config = {
        "populate_by_name": True,  # Allows using field names or aliases
        "from_attributes": True,   # Allows creating model from class attributes (e.g., ORM objects)
    }

# Represents the Go services/model-service/internal/entity/AIModel
class AIModelPydantic(BaseEntityModel):
    """Pydantic model representing an AI Model, inheriting base fields."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    file_path: str = Field(..., description="Path to the model file in storage (set by the system)")
    description: Optional[str] = Field(None, description="Optional description of the model")
    trained_at: datetime = Field(..., description="Timestamp when the model was trained")
    station_id: UUID = Field(..., description="UUID of the station this model is associated with")
    availability: bool = Field(True, description="Whether the model is currently available for use (defaults to True)")
    parameter_list: Optional[List[str]] = Field(None, description="List of parameters/features used to train this model")

# Pydantic model for creating a new AIModel (input)
class AIModelCreate(BaseModel):
    name: str
    version: str
    file_path: str
    description: Optional[str] = None
    trained_at: datetime
    station_id: UUID
    availability: bool = True
    parameter_list: List[str] = Field(..., description="List of parameters/features used for training")

# Pydantic model for updating an AIModel (input)
# All fields are optional for update
class AIModelUpdate(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    file_path: Optional[str] = None
    description: Optional[str] = None
    trained_at: Optional[datetime] = None
    station_id: Optional[UUID] = None
    availability: Optional[bool] = None
    parameter_list: Optional[List[str]] = Field(None, description="Updated list of parameters/features used for training")
