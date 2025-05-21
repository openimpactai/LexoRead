#!/usr/bin/env python3
"""
Logging configuration for the LexoRead API.

This module sets up logging for the API, including log formatting and handlers.
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import settings

def setup_logging():
    """
    Set up logging for the API.
    
    This function configures logging for the API, including console and file handlers.
    """
    # Get log level from settings
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    verbose_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter if settings.APP_ENV == "production" else verbose_formatter)
    root_logger.addHandler(console_handler)
    
    # Set up file handler in non-development environments
    if settings.APP_ENV != "development":
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Set up rotating file handler
        file_handler = RotatingFileHandler(
            logs_dir / "lexoread-api.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific loggers
    for name, level in [
        ("uvicorn", log_level),
        ("uvicorn.access", logging.WARNING),
        ("uvicorn.error", logging.ERROR),
        ("sqlalchemy", logging.WARNING),
        ("alembic", logging.WARNING),
        ("api", log_level),
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Don't propagate to root for third-party loggers
        if name not in ["api"]:
            logger.propagate = False
            
            # Add handlers to non-propagating loggers
            for handler in root_logger.handlers:
                logger.addHandler(handler)
    
    # Log startup information
    logging.info(f"Logging initialized (level: {log_level_name})")
    logging.info(f"Environment: {settings.APP_ENV}")
    logging.info(f"Debug mode: {settings.DEBUG}")
