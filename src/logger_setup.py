# src/logger_setup.py
import logging
import sys
from pathlib import Path
from typing import Union
import colorlog
from src.config_loader import settings

def setup_logging(name: str) -> logging.Logger:
    """
    Configures and returns a logger with console and file handlers.

    The logging level is determined by the application environment settings.
    Console logs are colored for better readability in development.

    Args:
        name (str): The name for the logger, typically __name__.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.logging.level)
    logger.propagate = False  
    
    if logger.hasHandlers():
        return logger

    handler = colorlog.StreamHandler(sys.stdout)
    log_format = (
        '%(asctime)s - '
        '%(log_color)s%(levelname)-8s%(reset)s - '
        '%(name)s:%(funcName)s:%(lineno)d - '
        '%(message)s'
    )
    formatter = colorlog.ColoredFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # File Handler
    file_handler = logging.FileHandler(settings.logging.log_file, mode='a')
    file_format = (
        '%(asctime)s - %(levelname)-8s - %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger