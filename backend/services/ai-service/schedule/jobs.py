import logging

logger = logging.getLogger(__name__)

def print_ok_job():
    """A simple scheduled job that prints 'Ok'."""
    message = "Ok"
    logger.info(f"Scheduled job triggered: Printing '{message}'")
    print(message) # Also print to stdout for visibility 