import logging
import sys
import os

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures the logger specifically for the RAG system.
    Uses the RAG_LOG_LEVEL from the unified .env file (defaults to INFO).
    Outputs to both the console and a 'log.txt' file.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Fetch the specific log level for the RAG system
        log_level_str = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(numeric_level)
        
        # Create the format
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s - [RAG] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler (Outputs to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        #File Handler
        file_handler = logging.FileHandler("log.txt", mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger