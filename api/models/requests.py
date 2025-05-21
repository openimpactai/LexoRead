#!/usr/bin/env python3
"""
Pydantic models for API request validation.

This module defines the request data models for the LexoRead API.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import base64
import re

# Authentication models
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """JWT token data."""
    user_id: Optional[str] = None

class UserLogin(BaseModel):
    """User login credentials."""
    username: str
    password: str

# Text adaptation models
class TextAdaptRequest(BaseModel):
    """Request for text adaptation."""
    text: str = Field(..., min_length=1, max_length=10000)
    user_id: Optional[str] = None
    reading_level: Optional[int] = Field(None, ge=0, le=5)
    adaptations: Optional[Dict[str, Any]] = None

class TextSimplifyRequest(BaseModel):
    """Request for text simplification."""
    text: str = Field(..., min_length=1, max_length=10000)
    target_level: Optional[int] = Field(None, ge=0, le=5)
    preserve_meaning: bool = True
    simplify_vocabulary: bool = True
    simplify_grammar: bool = True

# OCR models
class ImageSource(str, Enum):
    """Source type for image data."""
    BASE64 = "base64"
    URL = "url"
    FILE = "file"

class OCRExtractRequest(BaseModel):
    """Request for OCR text extraction."""
    image_data: str = Field(..., min_length=1)
    source_type: ImageSource = Field(ImageSource.BASE64)
    enhance_image: bool = True
    detect_orientation: bool = True
    
    @validator('image_data')
    def validate_image_data(cls, v, values):
        source_type = values.get('source_type')
        
        if source_type == ImageSource.BASE64:
            # Validate base64 encoding
            try:
                # Check for data URI scheme
                if v.startswith('data:image'):
                    # Extract base64 part
                    v = re.sub(r'^data:image/[a-zA-Z]+;base64,', '', v)
                
                # Validate base64
                base64.b64decode(v)
            except Exception:
                raise ValueError('Invalid base64 image data')
        
        elif source_type == ImageSource.URL:
            # Basic URL validation
            if not v.startswith(('http://', 'https://')):
                raise ValueError('URL must start with http:// or https://')
        
        return v

class OCRExtractRegionsRequest(OCRExtractRequest):
    """Request for OCR text region extraction."""
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    merge_regions: bool = False

# TTS models
class TTSVoice(str, Enum):
    """Available voices for TTS."""
    DEFAULT = "default"
    MALE = "male"
    FEMALE = "female"
    CHILD = "child"

class TTSSynthesizeRequest(BaseModel):
    """Request for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice: TTSVoice = TTSVoice.DEFAULT
    speed: float = Field(1.0, ge=0.5, le=2.0)
    emphasis: float = Field(1.0, ge=0.5, le=2.0)
    output_format: str = "mp3"

class TTSAdaptiveRequest(BaseModel):
    """Request for adaptive text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=5000)
    user_id: Optional[str] = None
    voice: Optional[TTSVoice] = None
    text_difficulty: Optional[float] = None
    highlight_complex_words: bool = True

# Reading level models
class ReadingLevelAssessRequest(BaseModel):
    """Request for reading level assessment."""
    text: str = Field(..., min_length=1, max_length=10000)
    include_features: bool = False

class ReadingLevelAnalyzeRequest(BaseModel):
    """Request for detailed text analysis."""
    text: str = Field(..., min_length=1, max_length=10000)
    user_id: Optional[str] = None

class ReadingLevelSuggestRequest(BaseModel):
    """Request for text simplification suggestions."""
    text: str = Field(..., min_length=1, max_length=10000)
    target_level: Optional[int] = Field(None, ge=0, le=5)

# User profile models
class ReadingSessionRecord(BaseModel):
    """Record of a reading session."""
    text: Optional[str] = None
    word_count: Optional[int] = None
    reading_time: Optional[float] = None
    comprehension_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    text_difficulty: Optional[float] = Field(None, ge=0.0, le=100.0)
    adaptations_used: Optional[Dict[str, Any]] = None
    difficult_words: Optional[List[str]] = None
