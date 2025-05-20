import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from .jobs import print_ok_job, train_all_stations_job, predict_all_stations_job

logger = logging.getLogger(__name__)

def init_scheduler() -> AsyncIOScheduler:
    """Initializes and starts the APScheduler."""
    scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh") # Use Vietnam timezone

    # Add the job to run daily at midnight (00:00)
    scheduler.add_job(
        print_ok_job, 
        trigger=CronTrigger(hour=0, minute=0), 
        id="daily_print_ok", # Unique ID for the job
        name="Daily Print Ok Job",
        replace_existing=True
    )
    # Add the async job to train all stations daily at midnight
    scheduler.add_job(
        train_all_stations_job,
        trigger=CronTrigger(hour=0, minute=0),
        id="daily_train_all_stations",
        name="Daily Train All Stations Job",
        replace_existing=True,
        coalesce=True
    )
    # Add the async job to predict for all stations daily at midnight
    scheduler.add_job(
        predict_all_stations_job,
        trigger=CronTrigger(hour=0, minute=0),
        id="daily_predict_all_stations",
        name="Daily Predict All Stations Job",
        replace_existing=True,
        coalesce=True
    )
    
    logger.info("Starting scheduler...")
    scheduler.start()
    logger.info("Scheduler started.")
    scheduler.print_jobs()
    for job in scheduler.get_jobs():
        logger.info(f"Scheduled job: {job}")
    return scheduler 