import logging
import os

# Create the logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose logs during dev
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/hackrx.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Named logger instance (optional: use different names for modular logging)
logger = logging.getLogger("hackrx")
