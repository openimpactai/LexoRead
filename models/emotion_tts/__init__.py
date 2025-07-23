"""
Emotion-aware TTS module for LexoRead.
"""

from .emotion_analyzer import EmotionAnalyzer, EmotionTTSProcessor
from .enhanced_tts_model import EmotionAwareTTS

__all__ = [
    "EmotionAnalyzer",
    "EmotionTTSProcessor", 
    "EmotionAwareTTS"
]