#!/usr/bin/env python3
"""
Dependency injection for the LexoRead API.

This module provides dependencies that can be injected into API routes.
"""

import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from typing import Optional
import os
from pathlib import Path

from config import settings
from models.requests import TokenData
from services.text_service import TextService
from services.ocr_service import OCRService
from services.tts_service import TTSService
from services.reading_level_service import ReadingLevelService
from services.user_service import UserService

# Set up OAuth2 for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# Logger instance
logger = logging.getLogger("api.dependencies")

# Model path helpers
def get_model_path(model_name: str) -> str:
    """
    Get the path to a model file.
    
    Args:
        model_name (str): Name of the model (e.g., "text_adaptation", "ocr")
        
    Returns:
        str: Full path to the model file
    """
    model_dir = settings.MODEL_DIR
    
    # Default model filenames
    filenames = {
        "text_adaptation": "text_adaptation_model.pt",
        "ocr": "ocr_model.pt",
        "tts": "tts_model.pt",
        "reading_level": "reading_level_model.pt"
    }
    
    if model_name not in filenames:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model_path = os.path.join(model_dir, model_name, filenames[model_name])
    
    # Check if the model file exists
    if not Path(model_path).is_file():
        logger.warning(f"Model file not found: {model_path}")
        # In development, we'll continue without a model file
        if settings.APP_ENV == "development":
            return None
        # In production, we'll raise an error
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Get the current authenticated user from the JWT token.
    
    Args:
        token (str): JWT token from authorization header
        
    Returns:
        TokenData: User information from the token
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"]
        )
        
        # Extract user information
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Create token data
        token_data = TokenData(user_id=user_id)
        
    except (JWTError, ValidationError):
        raise credentials_exception
    
    return token_data

# Optional authentication dependency
async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[TokenData]:
    """
    Get the current authenticated user if available, or None if not authenticated.
    
    Args:
        token (str, optional): JWT token from authorization header
        
    Returns:
        TokenData or None: User information from the token, or None if not authenticated
    """
    if token is None:
        return None
    
    try:
        return await get_current_user(token)
    except HTTPException:
        return None

# Service dependencies
def get_text_service() -> TextService:
    """
    Get the text adaptation service.
    
    Returns:
        TextService: Text adaptation service instance
    """
    model_path = get_model_path("text_adaptation")
    return TextService(model_path=model_path)

def get_ocr_service() -> OCRService:
    """
    Get the OCR service.
    
    Returns:
        OCRService: OCR service instance
    """
    model_path = get_model_path("ocr")
    return OCRService(model_path=model_path)

def get_tts_service() -> TTSService:
    """
    Get the text-to-speech service.
    
    Returns:
        TTSService: Text-to-speech service instance
    """
    model_path = get_model_path("tts")
    return TTSService(model_path=model_path)

def get_reading_level_service() -> ReadingLevelService:
    """
    Get the reading level assessment service.
    
    Returns:
        ReadingLevelService: Reading level assessment service instance
    """
    model_path = get_model_path("reading_level")
    return ReadingLevelService(model_path=model_path)

def get_user_service() -> UserService:
    """
    Get the user profile service.
    
    Returns:
        UserService: User profile service instance
    """
    # User service doesn't have a model file
    return UserService()
