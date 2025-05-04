import logging
from typing import List, Optional, Tuple
from uuid import UUID
import os
import grpc # Added for gRPC
import pandas as pd # Added for data manipulation
import io # Added for in-memory file handling
from datetime import datetime # Added for timestamp handling
import tempfile # Added for creating temporary files if needed (alternative to StringIO)
import shutil # Added for directory cleanup
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from datetime import timezone, timedelta

from proto.water_quality_service import water_quality_pb2
from proto.water_quality_service import water_quality_pb2_grpc
# Import core proto if needed
from proto.core import common_pb2
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp


from database import AsyncSession, get_db_session # Changed to absolute import
from .model import AIModelPydantic, AIModelCreate, AIModelUpdate, BaseModel
from .model_repo import ModelRepository, SQLAlchemyModelRepository
from .model_usecase import ModelUseCase, AIModelService, NotFoundError, UseCaseError
from MLCodeForAPI import trainWithMLModel, predictWithMLModel, readCSVfile
from DLCodeForAPI import trainWithDLModel, predictWithDLModel
import numpy as np # Added for convert_np function
from wqi_calculation import calculate_WQI # Import WQI calculation function


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/models", # Prefix for all routes in this controller
)

# --- Constants ---
AI_PREDICTION_SCHEMA_NAME = "ai-service-predictions"
AI_PREDICTION_SCHEMA_IDENTIFIER = "ai-service"


# --- Dependency Providers --- (Can be moved to a separate dependencies.py)

def get_model_repository(session: AsyncSession = Depends(get_db_session)) -> ModelRepository:
    """Provides an instance of the SQLAlchemyModelRepository."""
    return SQLAlchemyModelRepository(session)

def get_model_use_case(repository: ModelRepository = Depends(get_model_repository)) -> ModelUseCase:
    """Provides an instance of the AIModelService use case."""
    return AIModelService(repository)

# --- Response Models ---
# Define a response model for list operations to include pagination info
class PaginatedAIModelResponse(BaseModel):
    items: List[AIModelPydantic]
    total: int
    offset: int
    limit: int

# Re-add PredictionResult definition
class PredictionResult(BaseModel):
     station_id: UUID
     model_id: Optional[UUID] = None
     model_name: Optional[str] = None
     model_version: Optional[str] = None
     prediction_output: Optional[dict] = None
     error: Optional[str] = None
     predicted_datapoints_created: Optional[int] = 0 # Default to 0
     schema_id_used: Optional[str] = None

# --- Helper Functions ---

def _convert_data_points_to_dataframe(data_points: List[water_quality_pb2.DataPoint]) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[str], Optional[str]]:
    """Converts gRPC DataPoint list to a pandas DataFrame suitable for training."""
    if not data_points:
        return None, None, None, None

    records = []
    feature_names = set()
    date_col = "monitoring_time" # Assume based on proto
    place_col = "station_id" # Assume based on proto

    for dp in data_points:
        record = {}
        # Convert protobuf timestamp to python datetime
        if dp.HasField("monitoring_time"):
            record[date_col] = dp.monitoring_time.ToDatetime()
        else:
            record[date_col] = None # Handle missing time if necessary

        record[place_col] = dp.station_id

        # Extract features - assuming numerical 'value' is primary
        dp_features = {}
        for feature in dp.features:
            # --- Filter by purpose ---
            if feature.purpose != water_quality_pb2.INDICATOR_PURPOSE_PREDICTION:
                # logger.debug(f"Skipping feature '{feature.name}' with purpose {feature.purpose} (not PREDICTION)")
                continue
            # --- End Filter ---

            # Prioritize numeric value, fall back to textual if needed and convertible
            feature_value = None
            if feature.HasField("value"):
                feature_value = feature.value
            elif feature.HasField("textual_value"):
                try:
                    feature_value = float(feature.textual_value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert textual_value '{feature.textual_value}' for feature '{feature.name}' to float. Skipping.")
                    continue # Skip this feature for this datapoint if not convertible

            if feature_value is not None:
                dp_features[feature.name] = feature_value
                feature_names.add(feature.name) # Collect all unique feature names

        record.update(dp_features)
        records.append(record)

    if not records:
        return None, None, None, None

    df = pd.DataFrame(records)
    # Ensure date column is datetime type
    df[date_col] = pd.to_datetime(df[date_col])
    # Ensure all feature columns are numeric, fill missing with appropriate strategy (e.g., mean, median, 0)
    numeric_feature_names = list(feature_names)
    for col in numeric_feature_names:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Convert non-numeric to NaN
        # Simple fillna strategy - consider more sophisticated methods
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True) # Or median(), or 0

    return df, numeric_feature_names, date_col, place_col

class TempUploadFile:
    """A wrapper to mimic fastapi.UploadFile from an in-memory buffer."""
    def __init__(self, content: bytes, filename: str):
        self.file = io.BytesIO(content)
        self.filename = filename
        # Add other methods/attributes if trainWithMLModel needs them (e.g., content_type)
        self.content_type = "text/csv" # Assume CSV

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    async def seek(self, offset: int) -> int:
        return self.file.seek(offset)

    async def close(self):
        self.file.close()


# --- Local Model File Management Endpoints (for Debugging) ---

@router.get("/local_models", response_model=List[str], tags=["Local Model Management", "Debugging"])
async def list_local_models():
    """Lists the files currently stored in the local 'saved_models' directory."""
    model_dir = "saved_models"
    if not os.path.isdir(model_dir):
        logger.info(f"Local model directory '{model_dir}' does not exist.")
        return []
    try:
        files = os.listdir(model_dir)
        logger.info(f"Listing files in '{model_dir}': {files}")
        return files
    except Exception as e:
        logger.error(f"Error listing files in {model_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list local models: {str(e)}")

@router.delete("/local-models", status_code=status.HTTP_204_NO_CONTENT, tags=["Local Model Management", "Debugging"])
async def clear_local_models():
    """
    Deletes all files and subdirectories within the local 'saved_models' directory.
    Use with caution! This is primarily for cleaning up during development/debugging.
    """
    model_dir = "saved_models"
    deleted_count = 0
    failed_count = 0
    logger.warning(f"Received request to clear local model directory: {model_dir}")
    if not os.path.isdir(model_dir):
        logger.info(f"Local model directory '{model_dir}' does not exist. Nothing to clear.")
        return # Return 204 even if dir doesn't exist

    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                logger.info(f"Deleted file: {file_path}")
                deleted_count += 1
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Recursively delete directory
                logger.info(f"Deleted directory: {file_path}")
                deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")
            failed_count += 1

    logger.info(f"Local model directory cleanup complete. Deleted: {deleted_count}, Failed: {failed_count}")
    if failed_count > 0:
        # Although we return 204, log that some deletions failed
        logger.error(f"{failed_count} items could not be deleted from {model_dir}.")
        # Optionally, could raise an exception here if partial success is unacceptable
        # raise HTTPException(status_code=500, detail=f"Failed to delete {failed_count} items.")

    return # Return None for 204 status code

# --- API Endpoints --- CRUD, Training, Prediction

@router.post("", response_model=AIModelPydantic, status_code=status.HTTP_201_CREATED, tags=["AI Models CRUD"])
async def create_ai_model(
    model_in: AIModelCreate,
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Creates a new AI Model metadata record."""
    logger.info(f"API: Received request to create model: {model_in.name} v{model_in.version}")
    try:
        created_model = await use_case.create_model(model_in)
        return created_model
    except UseCaseError as e:
        logger.error(f"API Error creating model: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"API Unexpected error creating model: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/", response_model=PaginatedAIModelResponse, tags=["AI Models CRUD"])
async def list_ai_models(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Lists all AI Models with pagination."""
    logger.info(f"API: Received request to list models, offset={offset}, limit={limit}")
    try:
        models, total = await use_case.list_models(offset=offset, limit=limit)
        return PaginatedAIModelResponse(items=models, total=total, offset=offset, limit=limit)
    except Exception as e:
        logger.error(f"API Unexpected error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/find/by_name", response_model=AIModelPydantic, tags=["AI Models CRUD"])
async def get_ai_model_by_name_version(
    name: str = Query(...),
    version: str = Query(...),
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Retrieves a specific AI Model by its name and version."""
    logger.info(f"API: Received request to find model by name='{name}', version='{version}'")
    model = await use_case.get_model_by_name_and_version(name, version)
    if model is None:
        logger.warn(f"API: Model not found for name='{name}', version='{version}'")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{name}' v'{version}' not found")
    return model

@router.get("/station/{station_id}", response_model=PaginatedAIModelResponse, tags=["AI Models CRUD"])
async def list_ai_models_by_station(
    station_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Lists AI Models for a specific station with pagination."""
    logger.info(f"API: Received request to list models for station {station_id}, offset={offset}, limit={limit}")
    try:
        models, total = await use_case.list_models_by_station(station_id=station_id, offset=offset, limit=limit)
        return PaginatedAIModelResponse(items=models, total=total, offset=offset, limit=limit)
    except Exception as e:
        logger.error(f"API Unexpected error listing models for station {station_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{model_id}", response_model=AIModelPydantic, tags=["AI Models CRUD"])
async def get_ai_model(
    model_id: UUID,
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Retrieves a specific AI Model by its ID."""
    logger.info(f"API: Received request to get model ID: {model_id}")
    model = await use_case.get_model_by_id(model_id)
    if model is None:
        logger.warn(f"API: Model not found for ID: {model_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with ID '{model_id}' not found")
    return model

@router.put("/{model_id}", response_model=AIModelPydantic, tags=["AI Models CRUD"])
async def update_ai_model(
    model_id: UUID,
    model_in: AIModelUpdate,
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Updates an existing AI Model metadata record."""
    logger.info(f"API: Received request to update model ID: {model_id}")
    try:
        updated_model = await use_case.update_model(model_id, model_in)
        return updated_model
    except NotFoundError as e:
        logger.warn(f"API: Model not found for update: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except UseCaseError as e: # Catch other potential use case errors (like validation)
        logger.error(f"API Error updating model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"API Unexpected error updating model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["AI Models CRUD"])
async def delete_ai_model(
    model_id: UUID,
    use_case: ModelUseCase = Depends(get_model_use_case)
):
    """Deletes an AI Model metadata record."""
    logger.info(f"API: Received request to delete model ID: {model_id}")
    try:
        await use_case.delete_model(model_id)
        return # Return None for 204 status code
    except NotFoundError as e:
        logger.warn(f"API: Model not found for deletion: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"API Unexpected error deleting model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/train-station-models", response_model=List[AIModelPydantic], tags=["AI Models Training/Prediction"])
async def training_api(
    use_case: ModelUseCase = Depends(get_model_use_case), # Inject use case
    train_test_ratio: float = Form(0.7, ge=0.1, le=1.0),
    place_ids: List[str] = Form(..., description="List of Station IDs (UUIDs) to train models for."),
    date_tag: str = Form(datetime.now().strftime("%d%m%y"), description="A tag, often date-based (e.g., ddmmyy), to version the models."),
):
    """
    Fetches data for specified station IDs from the water-quality gRPC service,
    trains a model for each station, saves the model file locally, and creates a
    metadata record in the database. Model files are saved in './saved_models'.
    """
    created_models: List[AIModelPydantic] = []
    grpc_options = [ # Consider adding security/credentials if needed
        ('grpc.enable_retries', 1),
        ('grpc.keepalive_time_ms', 10000)
    ]
    # Establish gRPC connection (consider managing the channel lifecycle more robustly in production)
    # parsed_elements_list = [e.strip() for e in elements_list[0].split(",") if e.strip()]
    if place_ids is not None:
        place_ids = [e.strip() for e in place_ids[0].split(",") if e.strip()]
    GRPC_TARGET_ADDRESS = os.getenv("WATER_QUALITY_GRPC_ADDRESS")
    model_dir = "saved_models" # Hardcoded model directory

    try:
        async with grpc.aio.insecure_channel(GRPC_TARGET_ADDRESS, options=grpc_options) as channel:
            stub = water_quality_pb2_grpc.WaterQualityServiceStub(channel)
            logger.info(f"gRPC channel opened to {GRPC_TARGET_ADDRESS}")

            for station_id_str in place_ids:
                logger.info(f"Processing station ID: {station_id_str}")
                try:
                    # Validate station_id format (optional but recommended)
                    try:
                        station_uuid = UUID(station_id_str)
                    except ValueError:
                        logger.warning(f"Invalid UUID format for station ID: {station_id_str}. Skipping.")
                        continue

                    # 1. Fetch data via gRPC using POST and filtering for ACTUAL
                    # --- Prepare filters --- 
                    filters = make_proto_value_map({
                        "observation_type": "actual" # Filter for actual observations
                    })
                    filter_options = common_pb2.FilterOptions(
                        limit=10000, # Keep large limit
                        filters=filters
                    )
                    request = water_quality_pb2.ListDataPointsByStationRequest(
                        station_id=station_id_str,
                        options=filter_options
                    )
                    logger.info(f"Calling ListDataPointsByStationPost for station: {station_id_str} with filters")
                    # --- Make the POST call --- 
                    grpc_response = await stub.ListDataPointsByStationPost(request, timeout=60.0) # Increased timeout
                    logger.info(f"Received {len(grpc_response.data_points)} data points (filtered for actual) for station {station_id_str}")

                    if not grpc_response.data_points:
                        logger.warning(f"No ACTUAL data points found for station {station_id_str}. Skipping training.")
                        continue

                    # 2. Convert data to DataFrame (Now only contains ACTUAL data points)
                    df, features_list, date_col, place_col = _convert_data_points_to_dataframe(grpc_response.data_points)

                    if df is None or not features_list or not date_col or not place_col:
                        logger.warning(f"Could not process ACTUAL data into DataFrame for station {station_id_str}. Skipping.")
                        continue

                    if df.empty:
                         logger.warning(f"DataFrame is empty after processing for station {station_id_str}. Skipping.")
                         continue

                    logger.info(f"DataFrame created for station {station_id_str} with {len(df)} rows and features: {features_list}")


                    logger.info(f"Starting model training for station {station_id_str}...")
                    # Train model - Assume trainWithMLModel returns dict for paths, list for types
                    # model_paths, eval_dict, model_type_list = trainWithMLModel(
                    #     file=temp_file, # Use the wrapper
                    #     elements_list=features_list,
                    #     date_column_name=date_col,
                    #     place_column_name=place_col,
                    #     train_test_ratio=train_test_ratio,
                    #     place_id=station_id_str, # Pass the specific station ID
                    #     date_tag=date_tag,
                    #     model_dir=model_dir # Use hardcoded dir
                    # )
                    model_path_DL, eval_dict_DL, model_type_list_DL = trainWithDLModel(
                        df=df,
                        elements_list=features_list,
                        date_column_name=date_col,
                        place_column_name=place_col,
                        train_test_ratio=train_test_ratio,
                        place_id=station_id_str,
                        date_tag=date_tag,
                        model_dir=model_dir
                    )
                    model_path_ML, eval_dict_ML, model_type_list_ML = trainWithMLModel(
                        df=df,
                        elements_list=features_list,
                        date_column_name=date_col,
                        place_column_name=place_col,
                        train_test_ratio=train_test_ratio,
                        place_id=station_id_str,
                        date_tag=date_tag,
                        model_dir=model_dir
                    )
                    model_paths = {**model_path_DL, **model_path_ML}
                    eval_dict = {**eval_dict_DL, **eval_dict_ML}
                    model_type_list = model_type_list_DL + model_type_list_ML
                    logger.info(f"Training completed for station {station_id_str}. Model paths: {model_paths}, Types: {model_type_list}, Evaluation: {eval_dict}")

                    # --- Create Metadata for each trained model ---
                    if not isinstance(model_paths, dict) or not model_type_list:
                         logger.error(f"Training function did not return expected dict for paths or list for types for station {station_id_str}. Paths: {model_paths}, Types: {model_type_list}. Cannot save metadata.")
                         continue

                    for model_type in model_type_list:
                        model_path = model_paths.get(model_type) # Get path from dict using type as key
                        if not model_path:
                            logger.warning(f"Could not find model path for type \'{model_type}\' in returned paths dict for station {station_id_str}. Skipping metadata save for this type.")
                            continue


                        model_description = f"Mô hình {model_type} huấn luyện cho trạm {station_id_str}.\n Huấn luyện vào: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

                        # Create Pydantic payload
                        model_create_payload = AIModelCreate(
                            name=model_type,
                            version=date_tag, # Use date_tag as version
                            file_path=model_path, # Use the specific path for this model type
                            description=model_description,
                            # trained_at=datetime.now(timezone.utc), # Set training time
                            # now in Vietnam timezone
                            trained_at=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))),
                            station_id=station_uuid, # Use the UUID object
                            # availability defaults to True in Pydantic model
                            parameter_list=features_list # Store the features used for training (filtered ones)
                        )

                        logger.info(f"Saving metadata for model: {model_type} v{date_tag}, path: {model_path}")
                        try:
                            created_db_model = await use_case.create_model(model_create_payload)
                            created_models.append(created_db_model)
                            logger.info(f"Metadata saved successfully for model ID: {created_db_model.id}")
                        except UseCaseError as uc_err:
                            logger.error(f"Error saving metadata for model {model_type}: {uc_err}", exc_info=True)
                            # Decide if one failure should stop processing others
                        except Exception as meta_err:
                            logger.error(f"Unexpected error saving metadata for model {model_type}: {meta_err}", exc_info=True)
                    # --- End Metadata Creation ---

                except grpc.aio.AioRpcError as e:
                    logger.error(f"gRPC Error processing station {station_id_str}: {e.code()} - {e.details()}", exc_info=True)
                except FileNotFoundError as e:
                    logger.error(f"Training Error (FileNotFound - check model_dir '{model_dir}?'): {e} for station {station_id_str}", exc_info=True)
                except KeyError as e: # Catch potential key errors accessing model_paths dict
                    logger.error(f"Training Error: Model type '{e}' not found in model_paths dictionary returned by trainWithMLModel for station {station_id_str}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing station {station_id_str}: {e}", exc_info=True)
                    # Decide whether to continue or raise; for now, log and continue

        logger.info(f"gRPC channel to {GRPC_TARGET_ADDRESS} closed.")

    except grpc.aio.AioRpcError as e:
         logger.error(f"Failed to connect to gRPC service at {GRPC_TARGET_ADDRESS}: {e.code()} - {e.details()}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to data service: {e.details()}")
    except Exception as e:
        logger.error(f"Unexpected error during training API execution: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during model training process.")

    if not created_models:
        logger.warning("Training API completed, but no models were successfully trained or saved.")
        # 500 error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Training API completed, but no models were successfully trained or saved.")

    return created_models

@router.post("/predict-station-models", response_model=List[PredictionResult], tags=["AI Models Training/Prediction"])
async def predict_station_models(
    use_case: ModelUseCase = Depends(get_model_use_case),
    place_ids: Optional[List[str]] = Form(None, description="List of Station IDs (UUIDs) to predict for. If empty, predicts for all stations."),
    num_step: int = Form(7, ge=1, le=14),
    freq_days: int = Form(7, ge=1, le=14),
    model_types: List[str] = Form(..., description="List of model types ('xgb', 'rf', 'ETSformerPar', 'ETSformer') to use for prediction.")
):
    """
    Fetches data for specified stations (or all if none specified), finds newest models, 
    runs prediction using MLCodeForAPI.predictWithMLModel, saves predicted data points to WQ service.
    """
    results: List[PredictionResult] = []
    all_station_ids_to_process: List[UUID] = []
    grpc_options = [
        ('grpc.enable_retries', 1),
        ('grpc.keepalive_time_ms', 10000)
    ]
    GRPC_TARGET_ADDRESS = os.getenv("WATER_QUALITY_GRPC_ADDRESS")
    model_dir = "saved_models" # Hardcoded as per previous steps
    date_column_name = "monitoring_time" # Default
    place_column_name = "station_id" # Default
    if place_ids is not None:
        place_ids = [e.strip() for e in place_ids[0].split(",") if e.strip()]
    prediction_schema_id: Optional[str] = None
    if not model_types:
        model_types = ["rf", "xgb", "ETSformerPar", "ETSformer"]

    # Parse model_types if provided as comma-separated string (Form sometimes does this)
    if isinstance(model_types, list) and len(model_types) == 1 and isinstance(model_types[0], str):
        model_types = [mt.strip() for mt in model_types[0].split(',') if mt.strip()]
    elif not isinstance(model_types, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model_types must be a list of strings.")

    try:
        async with grpc.aio.insecure_channel(GRPC_TARGET_ADDRESS, options=grpc_options) as channel:
            stub = water_quality_pb2_grpc.WaterQualityServiceStub(channel)
            logger.info(f"Prediction: gRPC channel opened to {GRPC_TARGET_ADDRESS}")

            # --- Determine target station IDs (same logic as training_api) --- 
            if not place_ids:
                logger.info("Fetching all stations for prediction...")
                try:
                    list_req = water_quality_pb2.ListStationsRequest(options=common_pb2.FilterOptions(limit=1000))
                    list_res = await stub.ListStations(list_req, timeout=30.0)
                    for station in list_res.stations:
                        try: all_station_ids_to_process.append(UUID(station.id))
                        except ValueError: logger.warning(f"Invalid UUID '{station.id}' from ListStations.")
                    if not all_station_ids_to_process: return []
                except Exception as e:
                     logger.error(f"Prediction: Error fetching station list: {e}", exc_info=True)
                     raise HTTPException(status_code=500, detail="Failed to retrieve station list.")
            else:
                valid_provided_ids = []
                for pid in place_ids:
                     try: valid_provided_ids.append(UUID(pid))
                     except ValueError: logger.warning(f"Invalid UUID in input: '{pid}'.")
                if not valid_provided_ids: raise HTTPException(status_code=400, detail="No valid station IDs provided.")
                all_station_ids_to_process = valid_provided_ids
            
            prediction_schema_id = await get_or_create_prediction_schema(stub)
            if not prediction_schema_id: raise HTTPException(status_code=503, detail="Failed to get prediction schema.")

            # --- Process Each Station --- 
            for station_uuid in all_station_ids_to_process:
                station_id_str = str(station_uuid)
                logger.info(f"Prediction: Processing station ID: {station_id_str}")
                
                # Wrap station processing in try/except to allow continuing on single station failure
                try: 
                    # 1. Fetch and filter ACTUAL data for input
                    data_df: Optional[pd.DataFrame] = None
                    prediction_features_list: Optional[List[str]] = None
                    input_data_available = False
                    try: # Inner try for data fetching/processing
                        # --- Prepare filters --- 
                        filters = make_proto_value_map({
                            "observation_type": "actual" # Filter for actual observations
                        })
                        filter_options = common_pb2.FilterOptions(
                            limit=10000, # Keep large limit
                            filters=filters
                        )
                        request = water_quality_pb2.ListDataPointsByStationRequest(
                            station_id=station_id_str,
                            options=filter_options
                        )
                        logger.info(f"Prediction: Calling ListDataPointsByStationPost for {station_id_str} with filters")
                        # --- Make the POST call --- 
                        grpc_response = await stub.ListDataPointsByStationPost(request, timeout=60.0)
                        # No need to filter locally anymore, gRPC call already did
                        # actual_data_points = [dp for dp in grpc_response.data_points if dp.observation_type == water_quality_pb2.OBSERVATION_TYPE_ACTUAL]
                        if grpc_response.data_points:
                            df, features, date_col, place_col = _convert_data_points_to_dataframe(grpc_response.data_points)
                            if df is not None and not df.empty:
                                data_df = df
                                prediction_features_list = features 
                                date_column_name = date_col or date_column_name
                                place_column_name = place_col or place_column_name
                                input_data_available = True
                            else: logger.warning(f"Prediction: Could not process ACTUAL data for {station_id_str}.")
                        else: logger.warning(f"Prediction: No ACTUAL data points found for {station_id_str}.")
                    except Exception as data_err: # Catch errors during data fetch/process for this station
                         logger.error(f"Prediction: Error getting/processing data for {station_id_str}: {data_err}", exc_info=True)
                         results.append(PredictionResult(station_id=station_uuid, error=f"Data fetching/processing error: {str(data_err)}", schema_id_used=prediction_schema_id))
                         continue # Skip to next station
                    
                    if not input_data_available:
                        logger.warning(f"Prediction: Skipping station {station_id_str} due to lack of valid input data.")
                        results.append(PredictionResult(station_id=station_uuid, error="No valid actual data available for prediction input.", schema_id_used=prediction_schema_id))
                        continue # Skip to next station
                    
                    # --- Predict --- 
                    # 1. Find and run predictions for each requested model type
                    # 2. Find and run predictions for each requested model type
                    for model_type in model_types:
                        model_to_use = None
                        model_id_for_result: Optional[UUID] = None
                        model_name_for_result: Optional[str] = None
                        model_version_for_result: Optional[str] = None
                        error_msg: Optional[str] = None
                        prediction_output_restructured = None
                        created_count = 0
                        raw_prediction_output = None
                        try:
                            logger.info(f"Prediction: Finding newest '{model_type}' model for {station_id_str}")
                            model_to_use = await use_case.find_newest_model_for_station(station_uuid, model_type)

                            if model_to_use:
                                model_id_for_result = model_to_use.id
                                model_name_for_result = model_to_use.name # Store the actual model name (e.g., rf_200425)
                                model_version_for_result = model_to_use.version
                                model_params_required = set(model_to_use.parameter_list or [])
                                # Features available in input data (filtered by PREDICTION purpose during conversion)
                                data_features_available = set(prediction_features_list or []) 

                                logger.info(f"Prediction: Found model {model_name_for_result} v{model_version_for_result}. Required params: {model_params_required}. Available features in input: {data_features_available}")

                                # Check if required params are available in the input data (prediction_features_list)
                                if model_params_required and not model_params_required.issubset(data_features_available):
                                    error_msg = f"Input data missing required features for model: {model_params_required - data_features_available}"
                                    logger.warning(f"Prediction: {error_msg} for station {station_id_str}, model {model_id_for_result}")
                                else:
                                    # Use the parameters the model was trained on for the prediction call
                                    elements_for_prediction_call = model_to_use.parameter_list 
                                    
                                    logger.info(f"Prediction: Calling predict for station {station_id_str} using model {model_id_for_result} ({model_name_for_result}) version {model_version_for_result} with elements: {elements_for_prediction_call}")
                                    
                                    prediction_output_raw_dict = {}
                                    if model_to_use.name == "rf" or model_to_use.name == "xgb":
                                        # Call the wrapper function from MLCodeForAPI
                                        prediction_output_raw_dict_ML = predictWithMLModel(
                                            df=data_df,
                                            num_step=num_step,
                                            freq_days=freq_days,
                                            elements_list=elements_for_prediction_call, # Critical: Use model's params
                                            date_column_name=date_column_name,
                                            place_column_name=place_column_name,
                                            place_id=station_id_str,
                                            date_tag=model_to_use.version, # Use the specific model's version tag
                                            model_dir=model_dir
                                        )
                                        prediction_output_raw_dict = prediction_output_raw_dict_ML
                                    if model_to_use.name == "ETSformerPar" or model_to_use.name == "ETSformer":
                                        prediction_output_raw_dict_DL = predictWithDLModel(
                                            df=data_df,
                                            num_step=num_step,
                                            freq_days=freq_days,
                                            elements_list=elements_for_prediction_call, # Critical: Use model's params
                                            date_column_name=date_column_name,
                                            place_column_name=place_column_name,
                                            place_id=station_id_str,
                                            date_tag=model_to_use.version, # Use the specific model's version tag
                                            model_dir=model_dir
                                        )
                                        prediction_output_raw_dict = prediction_output_raw_dict_DL
                                    raw_prediction_output = convert_np(prediction_output_raw_dict) # Store converted raw output
                                    logger.info(f"Prediction: predictWithMLModel completed for {station_id_str}, type {model_type}")

                                    # --- Restructure the output for the CURRENT model_type --- 
                                    model_specific_predictions = prediction_output_raw_dict.get(model_type) # Extract based on loop's type
                                    if model_specific_predictions and isinstance(model_specific_predictions, dict):
                                        # ... (Restructuring logic: group by date, sort) ...
                                        temp_restructured = {}
                                        for element, steps in model_specific_predictions.items():
                                            if not isinstance(steps, list): continue
                                            for step_data in steps:
                                                if isinstance(step_data, dict) and 'predicted_date' in step_data and 'predicted_value' in step_data:
                                                    pred_date = step_data['predicted_date']
                                                    pred_value = step_data['predicted_value']
                                                    if pred_date not in temp_restructured:
                                                        temp_restructured[pred_date] = {'timestamp': pred_date}
                                                    temp_restructured[pred_date][element] = pred_value
                                        if temp_restructured:
                                            prediction_output_restructured = sorted(list(temp_restructured.values()), key=lambda x: x['timestamp'])
                                    else:
                                         error_msg = (error_msg or "") + f" Invalid/empty output from prediction function for type '{model_type}'."
                                         logger.warning(f"Prediction output for '{model_type}' invalid: {model_specific_predictions}")
                                    # --- End Restructure ---

                                    # 3. Process restructured output and create datapoints
                                    if prediction_output_restructured:
                                        datapoint_inputs = []
                                        for predicted_step in prediction_output_restructured:
                                            # ... (Timestamp parsing) ...
                                            try:
                                                ts_dt = predicted_step['timestamp']
                                                if not isinstance(ts_dt, datetime): raise ValueError("TS not datetime")
                                                if ts_dt.tzinfo is None: ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                                                pb_ts = Timestamp(); pb_ts.FromDatetime(ts_dt)
                                            except Exception as ts_err:
                                                logger.warning(f"Prediction: Skipping step due to timestamp error: {ts_err} for step: {predicted_step}")
                                                continue
                                            
                                            # --- WQI Calculation --- 
                                            calculated_wqi = None
                                            # Define required features and mapping (assuming AH -> Aeromonas)
                                            required_wqi_features = {
                                                "pH": "ph", 
                                                "DO": "DO", 
                                                "EC": "EC", 
                                                "N-NO2": "N_NO2", 
                                                "N-NH4": "N_NH4", 
                                                "P-PO4": "P_PO4", 
                                                "TSS": "TSS", 
                                                "COD": "COD", 
                                                "AH": "Aeromonas" # Map input 'AH' to function parameter 'Aeromonas'
                                            }
                                            wqi_params = {}
                                            all_wqi_features_present = True
                                            for feature_name, param_name in required_wqi_features.items():
                                                if feature_name in predicted_step and isinstance(predicted_step[feature_name], (np.float64, np.float32)):
                                                    wqi_params[param_name] = float(predicted_step[feature_name])
                                                else:
                                                    all_wqi_features_present = False
                                                    logger.info(f"Prediction: Skipping WQI calculation for step {ts_dt}. Missing or invalid feature: {feature_name}")
                                                    logger.info(f"Prediction: Predicted step: {predicted_step}")
                                                    logger.info(f"Prediction: Required features: {required_wqi_features}")
                                                    logger.info(f"Prediction: WQI params: {wqi_params}")
                                                    
                                                    break
                                            
                                            if all_wqi_features_present:
                                                try:
                                                    logger.info(f"Prediction: Calculating WQI for step {ts_dt} with params: {wqi_params}")
                                                    calculated_wqi = calculate_WQI(**wqi_params)
                                                    logger.info(f"Prediction: Calculated WQI = {calculated_wqi} for step {ts_dt}")
                                                except Exception as wqi_calc_err:
                                                    logger.info(f"Prediction: Error calculating WQI for step {ts_dt}: {wqi_calc_err}. Params: {wqi_params}")
                                            # --- End WQI Calculation --- 
                                            
                                            # ... (Feature processing - use model_params_required for filtering output) ...
                                            feature_inputs = []
                                            for f_name, f_val in predicted_step.items():
                                                if f_name == 'timestamp': continue
                                                # Only include features the model was *supposed* to predict
                                                if model_params_required and f_name not in model_params_required: continue
                                                f_input = water_quality_pb2.DataPointFeatureInput(name=f_name, purpose=water_quality_pb2.INDICATOR_PURPOSE_PREDICTION)
                                                if isinstance(f_val, (int, float)): f_input.value = float(f_val)
                                                elif f_val is not None: f_input.textual_value = str(f_val)
                                                feature_inputs.append(f_input)
                                                
                                            if feature_inputs:
                                                dp_input = water_quality_pb2.DataPointInput(
                                                    station_id=station_id_str, monitoring_time=pb_ts,
                                                    observation_type=water_quality_pb2.OBSERVATION_TYPE_PREDICTED,
                                                    source=model_type, # Use actual model name
                                                    data_source_schema_id=prediction_schema_id,
                                                    features=feature_inputs,
                                                    wqi=calculated_wqi # Add calculated WQI (will be null if calculation failed/skipped)
                                                )
                                                datapoint_inputs.append(dp_input)
                                                
                                        if datapoint_inputs:
                                            try:
                                                dp_req = water_quality_pb2.CreateDataPointsRequest(data_points=datapoint_inputs)
                                                # --- UNCOMMENT TO SAVE --- 
                                                dp_res = await stub.CreateDataPoints(dp_req, timeout=60.0)
                                                created_count = len(dp_res.data_points)
                                                logger.info(f"Prediction: Created {created_count} points.") 
                                            except grpc.aio.AioRpcError as dp_err:
                                                error_msg = (error_msg or "") + f" Failed to save predictions: {dp_err.details()}"
                                                logger.error(f"Prediction Save Error (gRPC): {error_msg}", exc_info=True)
                                            except Exception as dp_ex:
                                                error_msg = (error_msg or "") + f" Failed to save predictions: {str(dp_ex)}"
                                                logger.error(f"Prediction Save Error: {error_msg}", exc_info=True)
                                    else:
                                        error_msg = (error_msg or "") + " Failed to restructure prediction output or no valid steps found."
                                        logger.warning(f"No data points created for {station_id_str}/{model_type} after restructuring.")
                        except FileNotFoundError as fnf_err:
                             error_msg = f"Prediction failed: Model file not found for {model_id_for_result}. {str(fnf_err)}"
                             logger.error(f"Prediction: FileNotFoundError for {model_id_for_result} ({model_name_for_result}): {fnf_err}", exc_info=True)
                        except Exception as model_pred_err:
                            error_msg = f"Prediction failed unexpectedly for {model_type}: {str(model_pred_err)}"
                            logger.error(f"Prediction error for {station_id_str}/{model_type}: {model_pred_err}", exc_info=True)
                        
                        # Append result for this model type regardless of success/failure inside the try
                        results.append(PredictionResult(
                            station_id=station_uuid, model_id=model_id_for_result, model_name=model_name_for_result,
                            model_version=model_version_for_result, 
                            prediction_output=raw_prediction_output, 
                            error=error_msg, predicted_datapoints_created=created_count, 
                            schema_id_used=prediction_schema_id
                        ))
                
                # --- ADDED Missing Except Blocks for outer station loop --- 
                except grpc.aio.AioRpcError as grpc_station_err:
                    # This would catch channel errors during calls within the station loop (less likely if initial connect worked)
                    logger.error(f"Prediction: gRPC Error processing station {station_id_str}: {grpc_station_err.details()}", exc_info=True)
                    results.append(PredictionResult(station_id=station_uuid, error=f"gRPC error during processing: {grpc_station_err.details()}", schema_id_used=prediction_schema_id))
                    # Continue to next station
                except Exception as station_err:
                    # Catch any other unexpected error during the entire processing for one station
                    logger.error(f"Prediction: Unexpected error processing station {station_id_str}: {station_err}", exc_info=True)
                    results.append(PredictionResult(station_id=station_uuid, error=f"Unexpected station processing error: {str(station_err)}", schema_id_used=prediction_schema_id))
                    # Continue to next station
                # --- End Added Blocks --- 

        logger.info(f"gRPC channel to {GRPC_TARGET_ADDRESS} closed.")

    except grpc.aio.AioRpcError as e:
         logger.error(f"Prediction: Failed to connect to gRPC service at {GRPC_TARGET_ADDRESS}: {e.code()} - {e.details()}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to data service: {e.details()}")
    except Exception as e:
        logger.error(f"Prediction: Unexpected error during prediction API execution: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during prediction process.")

    return results

# --- DEPRECATED ---
@router.post("/predict", tags=["DEPRECATED"])
async def predict(
    file: UploadFile = File(...),
    elements_list: List[str] = Form(...),
    date_column_name: str = Form(...),
    place_column_name: str = Form(...),
    place_id: str = Form(...),
    date_tag: str = Form(...),
    num_step: int = Form(4),
    freq_days: int = Form(7),
):
    """Runs prediction using a specified model and input data."""
    model_dir = "saved_models" # Hardcoded model directory
    try:
        # Ensure elements_list is parsed correctly if it comes as a single comma-separated string
        if isinstance(elements_list, list) and len(elements_list) == 1 and isinstance(elements_list[0], str):
             parsed_elements_list = [e.strip() for e in elements_list[0].split(',') if e.strip()]
        else:
             parsed_elements_list = elements_list # Assume it's already a list of strings

        result = predictWithMLModel(
            file=file,
            num_step=num_step,
            freq_days=freq_days,
            elements_list=parsed_elements_list,
            date_column_name=date_column_name,
            place_column_name = place_column_name,
            place_id=place_id,
            date_tag=date_tag,
            model_dir=model_dir # Use hardcoded dir
        )
        return {"success": True, "results": convert_np(result)}
    except Exception as e:
        logger.error(f"API Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

@router.post("/train", tags=["DEPRECATED"])
async def train(
    file: UploadFile = File(...),
    elements_list: List[str] = Form(...),
    date_column_name: str = Form(...),
    place_column_name: str = Form(...),
    train_test_ratio: float = Form(0.7),
    place_id: str = Form(...),
    date_tag: str = Form("170425"),
):
    """(DEPRECATED - Use /train-station-models) Trains a model using uploaded CSV data."""
    model_dir = "saved_models" # Hardcoded model directory
    try:
        # Ensure elements_list is parsed correctly
        if isinstance(elements_list, list) and len(elements_list) == 1 and isinstance(elements_list[0], str):
             parsed_elements_list = [e.strip() for e in elements_list[0].split(',') if e.strip()]
        else:
             parsed_elements_list = elements_list

        model_path, eval_dict, model_types = trainWithMLModel( # Assume it returns types now too
            file=file,
            elements_list=parsed_elements_list,
            date_column_name=date_column_name,
            place_column_name=place_column_name,
            train_test_ratio=train_test_ratio,
            place_id=place_id,
            date_tag=date_tag,
            model_dir=model_dir # Use hardcoded dir
        )
        return {"success": True, "model_paths": model_path, "evaluation": convert_np(eval_dict), "model_types": model_types}
    except Exception as e:
        logger.error(f"API Error during training: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Training failed: {str(e)}")

def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (datetime, pd.Timestamp)): 
        if obj.tzinfo is not None and obj.tzinfo.utcoffset(obj) is not None:
            return obj.isoformat() 
        else:
            return obj.isoformat()
    else:
        return obj

# Re-add get_or_create_prediction_schema definition
async def get_or_create_prediction_schema(stub: water_quality_pb2_grpc.WaterQualityServiceStub) -> Optional[str]:
    """Finds or creates the DataSourceSchema for AI predictions."""
    schema_id = None
    try:
        list_req = water_quality_pb2.ListDataSourceSchemasRequest(options=common_pb2.FilterOptions(limit=1000))
        list_res = await stub.ListDataSourceSchemas(list_req, timeout=10.0)
        for schema in list_res.schemas:
            if schema.name == AI_PREDICTION_SCHEMA_NAME:
                schema_id = schema.id
                logger.info(f"Found existing prediction DataSourceSchema: {schema_id}")
                break
        if not schema_id:
            logger.info(f"Prediction DataSourceSchema '{AI_PREDICTION_SCHEMA_NAME}' not found, creating...")
            schema_input = water_quality_pb2.DataSourceSchemaInput(
                name=AI_PREDICTION_SCHEMA_NAME,
                source_identifier=AI_PREDICTION_SCHEMA_IDENTIFIER,
                source_type="ai-service",
                description="Schema for data points generated by AI service predictions.",
                schema_definition=Struct()
            )
            create_req = water_quality_pb2.CreateDataSourceSchemaRequest(schema=schema_input)
            create_res = await stub.CreateDataSourceSchema(create_req, timeout=10.0)
            schema_id = create_res.schema.id
            logger.info(f"Created prediction DataSourceSchema: {schema_id}")
        return schema_id
    except grpc.aio.AioRpcError as e:
        logger.error(f"gRPC Error finding/creating DataSourceSchema: {e.code()} - {e.details()}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error finding/creating DataSourceSchema: {e}", exc_info=True)
        return None

# Helper function to create a dictionary mapping strings to Protobuf Value objects
def make_proto_value_map(d: dict) -> dict:
    value_map = {}
    for key, value in d.items():
        # Create a google.protobuf.Value object for each value
        pb_value = Struct(); # Use Struct temporarily to leverage its value conversion
        pb_value.fields[key].string_value = str(value) # Simple string conversion for now
        # TODO: Enhance this to handle different types (number, bool, etc.) if needed
        # More robust way using Value() directly:
        # from google.protobuf.struct_pb2 import Value
        # pb_value = Value()
        # if isinstance(value, str):
        #     pb_value.string_value = value
        # elif isinstance(value, (int, float)):
        #     pb_value.number_value = value
        # elif isinstance(value, bool):
        #     pb_value.bool_value = value
        # ... etc
        # For this specific case ("observation_type": "actual"), string is sufficient
        value_map[key] = pb_value.fields[key]
    return value_map
