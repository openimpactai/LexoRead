#!/usr/bin/env python3
"""
Emotional TTS Router for LexoRead API
Provides endpoints for emotion-aware text-to-speech synthesis.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import base64
import io
import numpy as np
import soundfile as sf
from typing import Dict, Optional, List
import logging
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.requests import TTSRequest
from api.models.responses import TTSResponse
from api.dependencies import get_current_user
from models.emotion_tts import EmotionAwareTTS, EmotionAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

# Global TTS instance (initialized on first use)
_emotion_tts_instance = None

def get_emotion_tts():
    """Get or create EmotionAwareTTS instance."""
    global _emotion_tts_instance
    if _emotion_tts_instance is None:
        _emotion_tts_instance = EmotionAwareTTS(enable_emotions=True)
        logger.info("Initialized EmotionAwareTTS instance")
    return _emotion_tts_instance

# Request model for emotional TTS
from pydantic import BaseModel, Field

class EmotionalTTSRequest(BaseModel):
    """Request model for emotional TTS synthesis."""
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice to use")
    enable_emotions: bool = Field(default=True, description="Enable emotion analysis")
    base_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Base speech rate")
    base_pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Base pitch")
    base_volume: float = Field(default=1.0, ge=0.1, le=2.0, description="Base volume")
    emotion_intensity: float = Field(default=0.7, ge=0.0, le=1.0, 
                                   description="How strongly emotions affect speech (0=none, 1=maximum)")
    return_emotion_report: bool = Field(default=True, description="Include emotion analysis in response")
    output_format: str = Field(default="wav", description="Audio format (wav, mp3)")

class EmotionSegment(BaseModel):
    """Emotion analysis for a text segment."""
    text_preview: str
    detected_emotion: str
    confidence: float
    tts_adjustments: Dict[str, float]

class EmotionalTTSResponse(BaseModel):
    """Response model for emotional TTS."""
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    emotion_report: Optional[Dict] = None
    segments: Optional[List[EmotionSegment]] = None

@router.post("/emotional", response_model=EmotionalTTSResponse)
async def synthesize_emotional_speech(
    request: EmotionalTTSRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Synthesize speech with emotion-based voice modulation.
    
    This endpoint analyzes the emotional content of the text and adjusts
    speech parameters (pitch, speed, emphasis) accordingly.
    """
    try:
        # Get TTS instance
        tts = get_emotion_tts()
        
        # Prepare base parameters
        base_params = {
            "speed": request.base_speed,
            "pitch": request.base_pitch,
            "volume": request.base_volume
        }
        
        # Set emotion intensity
        if hasattr(tts.emotion_analyzer, 'intensity'):
            tts.emotion_analyzer.intensity = request.emotion_intensity
        
        # Get emotion report if requested
        emotion_report = None
        segments = None
        
        if request.return_emotion_report and request.enable_emotions:
            report = tts.get_emotion_report(request.text)
            emotion_report = {
                "overall_emotions": report["overall_emotions"],
                "dominant_emotion": report["dominant_emotion"]
            }
            
            # Convert segments to response format
            segments = [
                EmotionSegment(
                    text_preview=seg["text"],
                    detected_emotion=seg["emotion"],
                    confidence=seg["confidence"],
                    tts_adjustments=seg["tts_adjustments"]
                )
                for seg in report["segments"]
            ]
        
        # Synthesize audio
        if request.enable_emotions:
            audio = tts.synthesize_with_emotion(
                request.text,
                voice=request.voice,
                base_params=base_params
            )
        else:
            # Use regular synthesis without emotions
            audio = tts.synthesize(
                request.text,
                voice=request.voice,
                **base_params
            )
        
        # Ensure audio is numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Get sample rate
        sample_rate = getattr(tts.audio_processor, 'sample_rate', 22050)
        
        # Calculate duration
        duration = len(audio) / sample_rate
        
        # Convert to desired format
        audio_buffer = io.BytesIO()
        
        if request.output_format.lower() == "wav":
            sf.write(audio_buffer, audio, sample_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        else:
            # For now, only support WAV
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {request.output_format}"
            )
        
        # Log synthesis info
        logger.info(f"Synthesized {duration:.2f}s of emotional speech for user {current_user['id']}")
        
        # Return response
        return EmotionalTTSResponse(
            audio_base64=audio_base64,
            sample_rate=sample_rate,
            duration_seconds=duration,
            emotion_report=emotion_report,
            segments=segments
        )
        
    except Exception as e:
        logger.error(f"Error in emotional TTS synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-emotions")
async def analyze_text_emotions(
    text: str,
    segment_size: int = 100,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze emotions in text without synthesizing speech.
    
    Useful for previewing how the text will be spoken with emotions.
    """
    try:
        analyzer = EmotionAnalyzer()
        
        # Full text analysis
        overall_emotions = analyzer.analyze_emotion(text)
        
        # Segment analysis
        segments = analyzer.analyze_text_segments(text, segment_size)
        
        # Format response
        response = {
            "overall_analysis": {
                "emotions": overall_emotions,
                "dominant_emotion": max(overall_emotions, key=overall_emotions.get),
                "emotional_diversity": len([e for e in overall_emotions.values() if e > 0.1])
            },
            "segments": [
                {
                    "text": seg["text"][:100] + "..." if len(seg["text"]) > 100 else seg["text"],
                    "word_count": seg["word_count"],
                    "emotion": seg["parameters"]["detected_emotion"],
                    "confidence": seg["parameters"]["emotion_confidence"],
                    "suggested_voice_adjustments": {
                        k: v for k, v in seg["parameters"].items()
                        if k in ["speed", "pitch", "volume", "emphasis", "pause_factor"]
                    }
                }
                for seg in segments
            ],
            "summary": {
                "total_segments": len(segments),
                "emotion_changes": len(set(s["parameters"]["detected_emotion"] for s in segments)) - 1,
                "average_confidence": np.mean([s["parameters"]["emotion_confidence"] for s in segments])
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing emotions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/emotion-voices")
async def get_emotion_voice_mappings(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get available emotion-to-voice parameter mappings.
    
    Returns the current configuration for how each emotion affects voice parameters.
    """
    analyzer = EmotionAnalyzer()
    
    return {
        "available_emotions": list(analyzer.emotion_parameters.keys()),
        "emotion_parameters": analyzer.emotion_parameters,
        "parameter_descriptions": {
            "speed": "Speech rate (1.0 = normal, >1 = faster, <1 = slower)",
            "pitch": "Voice pitch (1.0 = normal, >1 = higher, <1 = lower)",
            "volume": "Audio volume (1.0 = normal)",
            "emphasis": "Word emphasis strength (1.0 = normal)",
            "pause_factor": "Pause duration between words/sentences (1.0 = normal)"
        }
    }

@router.post("/preview-ssml")
async def preview_emotional_ssml(
    request: EmotionalTTSRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate SSML markup preview for emotional speech.
    
    Shows how the text will be marked up with emotion-based prosody tags.
    """
    try:
        tts = get_emotion_tts()
        
        # Get emotion data
        emotion_data = tts.emotion_processor.prepare_emotional_tts(
            request.text,
            {
                "speed": request.base_speed,
                "pitch": request.base_pitch,
                "volume": request.base_volume
            }
        )
        
        # Generate SSML
        ssml = tts.emotion_processor.get_ssml_markup(request.text, emotion_data)
        
        return {
            "ssml": ssml,
            "segments": len(emotion_data["segments"]),
            "description": "SSML markup showing emotion-based prosody adjustments"
        }
        
    except Exception as e:
        logger.error(f"Error generating SSML preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def emotional_tts_health():
    """Check if emotional TTS service is healthy."""
    try:
        # Try to initialize the model
        tts = get_emotion_tts()
        
        # Test emotion analysis
        analyzer = EmotionAnalyzer()
        test_result = analyzer.analyze_emotion("Hello world")
        
        return {
            "status": "healthy",
            "emotion_model_loaded": True,
            "tts_model_loaded": tts is not None,
            "test_emotion": max(test_result, key=test_result.get) if test_result else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Example usage endpoint for documentation
@router.get("/example")
async def get_emotional_tts_example():
    """Get example requests and responses for emotional TTS."""
    return {
        "example_request": {
            "text": "I'm so happy to see you! This is wonderful news. However, I'm a bit worried about the deadline.",
            "voice": "default",
            "enable_emotions": True,
            "base_speed": 0.9,
            "emotion_intensity": 0.8,
            "return_emotion_report": True
        },
        "example_response": {
            "audio_base64": "UklGRi4AAABXQVZFZm10IBA...",
            "sample_rate": 22050,
            "duration_seconds": 5.4,
            "emotion_report": {
                "overall_emotions": {
                    "joy": 0.45,
                    "surprise": 0.25,
                    "fear": 0.20,
                    "neutral": 0.10
                },
                "dominant_emotion": "joy"
            },
            "segments": [
                {
                    "text_preview": "I'm so happy to see you!",
                    "detected_emotion": "joy",
                    "confidence": 0.92,
                    "tts_adjustments": {
                        "speed": 1.08,
                        "pitch": 1.12,
                        "volume": 1.04,
                        "emphasis": 1.16
                    }
                }
            ]
        },
        "description": "Emotional TTS analyzes text sentiment and adjusts voice parameters to convey appropriate emotions."
    }