#!/usr/bin/env python3
"""
Configuration settings for the LexoRead API.

This module loads configuration from environment variables or a .env file.
"""

import os
from pydantic import BaseSettings, Field
from typing import List, Optional, Dict, Any
from functools import lru_cache

class Settings(BaseSettings):
    """
    API configuration settings.
    
    Environment variables:
    - APP_ENV: Application environment (development, staging, production)
    - DEBUG: Enable debug mode
    - SECRET_KEY: Secret key for JWT encoding
    - CORS_ORIGINS: Comma-separated list of allowed origins for CORS
    - DB_URL: Database connection string
    - MODEL_DIR: Directory containing model files
    - LOG_LEVEL: Logging level
    - RATE_LIMIT_ANONYMOUS: Rate limit for anonymous users (requests per minute)
    - RATE_LIMIT_AUTHENTICATED: Rate limit for authenticated users (requests per minute)
    """
    
    # Application settings
    APP_ENV: str = Field("development", env="APP_ENV")
    DEBUG: bool = Field(False, env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "https://lexoread.openimpactai.org"],
        env="CORS_ORIGINS"
    )
    
    # Database settings
    DB_URL: str = Field("sqlite:///./lexoread.db", env="DB_URL")
    
    # Model settings
    MODEL_DIR: str = Field("../models", env="MODEL_DIR")
    
    # Logging settings
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # Rate limiting
    RATE_LIMIT_ANONYMOUS: int = Field(100, env="RATE_LIMIT_ANONYMOUS")
    RATE_LIMIT_AUTHENTICATED: int = Field(300, env="RATE_LIMIT_AUTHENTICATED")
    
    # Authentication settings
    TOKEN_EXPIRE_MINUTES: int = Field(60 * 24, env="TOKEN_EXPIRE_MINUTES")  # 24 hours
    
    # Service integrations (for future expansion)
    SERVICES: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    """
    return Settings()

# Export settings instance
settings = get_settings()

# Environment-aware settings
def is_development() -> bool:
    """Check if running in development environment."""
    return settings.APP_ENV.lower() == "development"

def is_staging() -> bool:
    """Check if running in staging environment."""
    return settings.APP_ENV.lower() == "staging"

def is_production() -> bool:
    """Check if running in production environment."""
    return settings.APP_ENV.lower() == "production"
