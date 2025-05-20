import logging
import asyncio
from typing import Optional

from database import init_db
from model_service import model_controller
from schedule.scheduler import init_scheduler
# from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Removed imports specific to the old router/endpoints
from fastapi import FastAPI#, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Models Service",
    description="API for AI model training, prediction, and metadata management.",
    version="1.0.0",
    docs_url="/api/v2/swagger",
    openapi_url="/api/v2/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_controller.router)

# # Global variable to hold the scheduler instance
app_scheduler: Optional[object] = None

@app.on_event("startup")
async def startup_event():
    global app_scheduler
    # logger.info("Initializing database...")

    try:
        await init_db()
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}", exc_info=True)
        # Depending on severity, you might want to raise or exit here

    logger.info("Initializing scheduler...")
    try:
        app_scheduler = init_scheduler()
    except Exception as e:
        logger.error(f"Error initializing scheduler: {e}", exc_info=True)
        # Decide if the app should start without the scheduler

@app.on_event("shutdown")
async def shutdown_event():
    global app_scheduler
    if app_scheduler and app_scheduler.running:
        logger.info("Shutting down scheduler...")
        app_scheduler.shutdown()
        logger.info("Scheduler shut down.")