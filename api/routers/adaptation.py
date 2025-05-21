#!/usr/bin/env python3
"""
API routes for text adaptation.

This module provides endpoints for adapting text for dyslexic readers.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional

from dependencies import get_text_service, get_optional_user
from models.requests import TextAdaptRequest, TextSimplifyRequest, TokenData
from models.responses import TextAdaptationResult, TextFormatsResponse, TextFormatOption

# Set up logger
logger = logging.getLogger("api.text_adaptation")

# Create router
router = APIRouter()

@router.post("/adapt", response_model=TextAdaptationResult)
async def adapt_text(
    request: TextAdaptRequest,
    text_service = Depends(get_text_service),
    current_user: Optional[TokenData] = Depends(get_optional_user)
):
    """
    Adapt text for easier reading by individuals with dyslexia.
    
    This endpoint processes text to make it more accessible for dyslexic readers
    by adjusting formatting, highlighting complex words, and applying other
    adaptations based on user preferences if available.
    
    - **text**: The text to adapt
    - **user_id**: Optional user ID for personalized adaptations
    - **reading_level**: Optional target reading level (0-5)
    - **adaptations**: Optional specific adaptations to apply
    
    Returns the adapted text with formatting instructions.
    """
    try:
        # Use authenticated user ID if available
        user_id = current_user.user_id if current_user else request.user_id
        
        # Get adaptations from text service
        result = text_service.adapt_text(
            text=request.text,
            user_id=user_id,
            reading_level=request.reading_level,
            adaptations=request.adaptations
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error adapting text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adapting text: {str(e)}"
        )

@router.post("/simplify", response_model=TextAdaptationResult)
async def simplify_text(
    request: TextSimplifyRequest,
    text_service = Depends(get_text_service)
):
    """
    Simplify text by reducing complexity of vocabulary and grammar.
    
    This endpoint processes text to simplify complex words and sentence 
    structures while preserving the original meaning as much as possible.
    
    - **text**: The text to simplify
    - **target_level**: Optional target reading level (0-5)
    - **preserve_meaning**: Whether to prioritize preserving meaning over simplification
    - **simplify_vocabulary**: Whether to simplify complex vocabulary
    - **simplify_grammar**: Whether to simplify sentence structure and grammar
    
    Returns the simplified text with formatting instructions.
    """
    try:
        # Get simplified text from text service
        result = text_service.simplify_text(
            text=request.text,
            target_level=request.target_level,
            preserve_meaning=request.preserve_meaning,
            simplify_vocabulary=request.simplify_vocabulary,
            simplify_grammar=request.simplify_grammar
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error simplifying text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error simplifying text: {str(e)}"
        )

@router.get("/formats", response_model=TextFormatsResponse)
async def get_text_formats(
    text_service = Depends(get_text_service)
):
    """
    Get available text formatting options.
    
    This endpoint returns the available text formatting options that can be
    applied to adapt text for dyslexic readers.
    
    Returns a list of available formatting options with their descriptions and default values.
    """
    try:
        # Get available formats from text service
        formats = text_service.get_available_formats()
        
        return TextFormatsResponse(formats=formats)
        
    except Exception as e:
        logger.error(f"Error getting text formats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting text formats: {str(e)}"
        )
