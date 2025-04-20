import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from .jobs import print_ok_job # Import the job function

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
    
    logger.info("Starting scheduler...")
    scheduler.start()
    logger.info("Scheduler started. Scheduled jobs:")
    scheduler.print_jobs()
    
    return scheduler 