import logging
import os
from datetime import datetime
import asyncio
from model_service import model_controller
from proto.water_quality_service import water_quality_pb2, water_quality_pb2_grpc
from proto.core import common_pb2
import grpc

logger = logging.getLogger(__name__)

def print_ok_job():
    """A simple scheduled job that prints 'Ok'."""
    message = "Ok"
    logger.info(f"Scheduled job triggered: Printing '{message}'")
    print(message) # Also print to stdout for visibility 

# New: Train models for all stations
async def train_all_stations_job():
    logger.info("Scheduled job: Training models for all stations started.")
    GRPC_TARGET_ADDRESS = os.getenv("WATER_QUALITY_GRPC_ADDRESS")
    grpc_options = [
        ('grpc.enable_retries', 1),
        ('grpc.keepalive_time_ms', 10000)
    ]
    model_dir = "saved_models"
    date_tag = datetime.now().strftime("%d%m%y")
    train_test_ratio = 0.7
    try:
        async with grpc.aio.insecure_channel(GRPC_TARGET_ADDRESS, options=grpc_options) as channel:
            stub = water_quality_pb2_grpc.WaterQualityServiceStub(channel)
            logger.info(f"Fetching all stations for training...")
            list_req = water_quality_pb2.ListStationsRequest(options=common_pb2.FilterOptions(limit=1000))
            list_res = await stub.ListStations(list_req, timeout=30.0)
            all_station_ids = [station.id for station in list_res.stations]
            logger.info(f"Found {len(all_station_ids)} stations for training.")
            # Use the same logic as training_api, but as a job
            for station_id in all_station_ids:
                try:
                    await model_controller.training_api(
                        use_case=None, # Will use default dependency
                        train_test_ratio=train_test_ratio,
                        place_ids=[station_id],
                        date_tag=date_tag
                    )
                    logger.info(f"Trained models for station {station_id}")
                except Exception as e:
                    logger.error(f"Error training models for station {station_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in train_all_stations_job: {e}", exc_info=True)
    logger.info("Scheduled job: Training models for all stations completed.")

# New: Predict for all stations
async def predict_all_stations_job():
    logger.info("Scheduled job: Predicting for all stations started.")
    GRPC_TARGET_ADDRESS = os.getenv("WATER_QUALITY_GRPC_ADDRESS")
    grpc_options = [
        ('grpc.enable_retries', 1),
        ('grpc.keepalive_time_ms', 10000)
    ]
    try:
        async with grpc.aio.insecure_channel(GRPC_TARGET_ADDRESS, options=grpc_options) as channel:
            stub = water_quality_pb2_grpc.WaterQualityServiceStub(channel)
            logger.info(f"Fetching all stations for prediction...")
            list_req = water_quality_pb2.ListStationsRequest(options=common_pb2.FilterOptions(limit=1000))
            list_res = await stub.ListStations(list_req, timeout=30.0)
            all_station_ids = [station.id for station in list_res.stations]
            logger.info(f"Found {len(all_station_ids)} stations for prediction.")
            for station_id in all_station_ids:
                try:
                    await model_controller.predict_station_models(
                        use_case=None, # Will use default dependency
                        place_ids=[station_id],
                        num_step=7,
                        freq_days=7,
                        model_types=["rf", "xgb", "ETSformerPar", "ETSformer"]
                    )
                    logger.info(f"Predicted for station {station_id}")
                except Exception as e:
                    logger.error(f"Error predicting for station {station_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in predict_all_stations_job: {e}", exc_info=True)
    logger.info("Scheduled job: Predicting for all stations completed.") 