#!/usr/bin/env python3
"""
Pydantic models for API response formatting.

This module defines the response data models for the LexoRead API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

# General response models
class ErrorResponse(BaseModel):
    """Error response."""
    message: str
    detail: Optional[str] = None
    code: Optional[str] = None

class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str
    data: Optional[Dict[str, Any]] = None

# Text adaptation response models
class TextAdaptationResult(BaseModel):
    """Result of text adaptation."""
    original_text: str
    adapted_text: str
    meta_info: Dict[str, Any] = Field(default_factory=dict)
    formatting: Dict[str, Any] = Field(default_factory=dict)
    
class TextFormatOption(BaseModel):
    """Text formatting option."""
    id: str
    name: str
    description: str
    type: str
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = None

class TextFormatsResponse(BaseModel):
    """Available text formatting options."""
    formats: List[TextFormatOption]

# OCR response models
class OCRTextRegion(BaseModel):
    """Text region detected by OCR."""
    text: str
    confidence: float
    bounding_box: Dict[str, int]  # x, y, width, height

class OCRExtractResult(BaseModel):
    """Result of OCR text extraction."""
    text: str
    confidence: float
    processing_time: float
    orientation: Optional[float] = None
    regions: Optional[List[OCRTextRegion]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

class OCRExtractRegionsResult(BaseModel):
    """Result of OCR text region extraction."""
    regions: List[OCRTextRegion]
    processing_time: float
    image_width: int
    image_height: int

# TTS response models
class TTSVoiceOption(BaseModel):
    """Voice option for TTS."""
    id: str
    name: str
    gender: str
    description: str
    preview_url: Optional[str] = None

class TTSVoicesResponse(BaseModel):
    """Available TTS voices."""
    voices: List[TTSVoiceOption]

class TTSSynthesisResult(BaseModel):
    """Result of TTS synthesis."""
    audio_url: str
    duration: float
    processing_time: float
    text_length: int
    parameters: Dict[str, Any]

# Reading level response models
class ReadingLevelFeatures(BaseModel):
    """Features extracted from text for reading level assessment."""
    num_sentences: int
    num_words: int
    num_chars: int
    num_syllables: int
    avg_word_length: float
    avg_syllables_per_word: float
    avg_words_per_sentence: float
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    unique_words_ratio: float
    rare_words_ratio: float
    consonant_cluster_ratio: Optional[float] = None
    similar_sounding_ratio: Optional[float] = None
    
class ReadingLevelResult(BaseModel):
    """Result of reading level assessment."""
    level: int
    grade: float
    difficulty: float
    readability_score: float
    dyslexia_difficulty: float
    features: Optional[ReadingLevelFeatures] = None

class TextAnalysisIssue(BaseModel):
    """Issue identified in text analysis."""
    type: str
    description: str
    severity: float  # 0-1
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    text: Optional[str] = None
    suggestions: Optional[List[str]] = None

class TextAnalysisResult(BaseModel):
    """Result of detailed text analysis."""
    reading_level: ReadingLevelResult
    issues: List[TextAnalysisIssue]
    difficult_words: List[str]
    long_sentences: List[Dict[str, Any]]
    overall_score: float

class SimplificationSuggestion(BaseModel):
    """Suggestion for text simplification."""
    original: str
    simplified: str
    explanation: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    confidence: float

class SimplificationSuggestionsResult(BaseModel):
    """Result of text simplification suggestions."""
    original_text: str
    current_level: ReadingLevelResult
    target_level: int
    suggestions: List[SimplificationSuggestion]

# User profile response models
class UserReadingMetrics(BaseModel):
    """User reading metrics."""
    avg_reading_speed: Optional[float] = None
    avg_comprehension: Optional[float] = None
    difficulty_trend: Optional[float] = None
    session_count: int

class UserProfileResponse(BaseModel):
    """User reading profile."""
    user_id: str
    metrics: UserReadingMetrics
    preferred_adaptations: Dict[str, Any]
    difficult_patterns: List[str]
    created_at: str
    updated_at: str

class UserRecommendationResponse(BaseModel):
    """Personalized recommendations for a user."""
    user_id: str
    adaptations: Dict[str, Any]
    text_challenges: Optional[Dict[str, Any]] = None
    reading_level: Optional[ReadingLevelResult] = None
