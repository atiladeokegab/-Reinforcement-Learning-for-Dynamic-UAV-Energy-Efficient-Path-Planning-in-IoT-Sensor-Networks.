import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to file
        logging.StreamHandler()           # Log to console
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting RL...")
    create_diagram()
    logger.info("RL finished!")