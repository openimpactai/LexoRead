#!/usr/bin/env python3
"""
Emotion Analysis Module for Dynamic TTS
This module analyzes text emotions to dynamically adjust TTS parameters.
"""

import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Analyzes text emotions using pre-trained transformer models.
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize emotion analyzer with a pre-trained model.
        
        Args:
            model_name: HuggingFace model for emotion classification
        """
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=model_name,
                return_all_scores=True
            )
            logger.info(f"Loaded emotion model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            # Fallback to simple sentiment analysis
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Using fallback sentiment analysis model")
            
        # Emotion to TTS parameter mappings
        self.emotion_parameters = {
            "joy": {
                "speed": 1.1,
                "pitch": 1.15,
                "volume": 1.05,
                "emphasis": 1.2,
                "pause_factor": 0.9
            },
            "sadness": {
                "speed": 0.85,
                "pitch": 0.9,
                "volume": 0.95,
                "emphasis": 0.8,
                "pause_factor": 1.2
            },
            "anger": {
                "speed": 1.05,
                "pitch": 0.95,
                "volume": 1.1,
                "emphasis": 1.3,
                "pause_factor": 0.95
            },
            "fear": {
                "speed": 1.15,
                "pitch": 1.1,
                "volume": 0.9,
                "emphasis": 1.1,
                "pause_factor": 0.85
            },
            "surprise": {
                "speed": 1.2,
                "pitch": 1.2,
                "volume": 1.1,
                "emphasis": 1.25,
                "pause_factor": 0.8
            },
            "disgust": {
                "speed": 0.9,
                "pitch": 0.85,
                "volume": 0.95,
                "emphasis": 1.15,
                "pause_factor": 1.1
            },
            "neutral": {
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
                "emphasis": 1.0,
                "pause_factor": 1.0
            },
            # Fallback for sentiment analysis
            "POSITIVE": {
                "speed": 1.05,
                "pitch": 1.1,
                "volume": 1.02,
                "emphasis": 1.1,
                "pause_factor": 0.95
            },
            "NEGATIVE": {
                "speed": 0.9,
                "pitch": 0.95,
                "volume": 0.98,
                "emphasis": 0.9,
                "pause_factor": 1.1
            }
        }
        
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Analyze emotion in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion scores
        """
        try:
            results = self.classifier(text)
            
            # Handle different output formats
            if isinstance(results[0], list):
                # Multi-label emotion classification
                emotion_scores = {
                    result['label']: result['score'] 
                    for result in results[0]
                }
            else:
                # Simple sentiment analysis
                emotion_scores = {
                    results[0]['label']: results[0]['score']
                }
                
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {"neutral": 1.0}
    
    def get_tts_parameters(self, text: str, base_params: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get TTS parameters based on text emotion.
        
        Args:
            text: Input text
            base_params: Base TTS parameters to modify
            
        Returns:
            Modified TTS parameters
        """
        # Analyze emotion
        emotion_scores = self.analyze_emotion(text)
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        # Get emotion parameters
        emotion_params = self.emotion_parameters.get(
            dominant_emotion, 
            self.emotion_parameters["neutral"]
        )
        
        # Initialize result with base parameters
        if base_params:
            result = base_params.copy()
        else:
            result = {
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
                "emphasis": 1.0,
                "pause_factor": 1.0
            }
        
        # Apply emotion parameters with confidence weighting
        for param, value in emotion_params.items():
            if param in result:
                # Weighted average between base and emotion value
                result[param] = result[param] * (1 - confidence * 0.7) + value * confidence * 0.7
            else:
                result[param] = value
                
        # Add emotion metadata
        result["detected_emotion"] = dominant_emotion
        result["emotion_confidence"] = confidence
        
        return result
    
    def analyze_text_segments(self, text: str, segment_size: int = 100) -> List[Dict]:
        """
        Analyze emotion for text segments.
        
        Args:
            text: Full text to analyze
            segment_size: Approximate size of each segment in words
            
        Returns:
            List of segment analyses with emotions and parameters
        """
        # Split text into sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        segments = []
        current_segment = []
        current_word_count = 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            
            if current_word_count + word_count > segment_size and current_segment:
                # Process current segment
                segment_text = '. '.join(current_segment) + '.'
                segments.append({
                    "text": segment_text,
                    "parameters": self.get_tts_parameters(segment_text),
                    "word_count": current_word_count
                })
                
                # Start new segment
                current_segment = [sentence]
                current_word_count = word_count
            else:
                current_segment.append(sentence)
                current_word_count += word_count
        
        # Process last segment
        if current_segment:
            segment_text = '. '.join(current_segment) + '.'
            segments.append({
                "text": segment_text,
                "parameters": self.get_tts_parameters(segment_text),
                "word_count": current_word_count
            })
            
        return segments


class EmotionTTSProcessor:
    """
    Processes text with emotion analysis for dynamic TTS.
    """
    
    def __init__(self, emotion_analyzer: Optional[EmotionAnalyzer] = None):
        """
        Initialize the processor.
        
        Args:
            emotion_analyzer: Pre-initialized emotion analyzer
        """
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        
    def prepare_emotional_tts(self, text: str, base_voice_params: Optional[Dict] = None) -> Dict:
        """
        Prepare text for emotional TTS synthesis.
        
        Args:
            text: Input text
            base_voice_params: Base voice parameters
            
        Returns:
            Dictionary with segments and their emotional parameters
        """
        # Analyze text segments
        segments = self.emotion_analyzer.analyze_text_segments(text)
        
        # Add transitions between segments
        processed_segments = []
        for i, segment in enumerate(segments):
            processed_segment = segment.copy()
            
            # Add transition parameters
            if i > 0:
                prev_params = segments[i-1]["parameters"]
                curr_params = segment["parameters"]
                
                # Calculate smooth transition
                transition = {}
                for param in ["speed", "pitch", "volume"]:
                    if param in prev_params and param in curr_params:
                        transition[param] = (prev_params[param] + curr_params[param]) / 2
                        
                processed_segment["transition"] = transition
                
            processed_segments.append(processed_segment)
            
        return {
            "segments": processed_segments,
            "total_segments": len(processed_segments),
            "base_parameters": base_voice_params or {}
        }
    
    def get_ssml_markup(self, text: str, emotion_data: Dict) -> str:
        """
        Generate SSML markup for emotional speech.
        
        Args:
            text: Original text
            emotion_data: Emotion analysis data
            
        Returns:
            SSML formatted text
        """
        ssml_parts = ['<speak>']
        
        for segment in emotion_data["segments"]:
            params = segment["parameters"]
            
            # Build prosody tag
            prosody_attrs = []
            
            # Speed (rate in SSML)
            rate_percent = int((params["speed"] - 1) * 100)
            if rate_percent != 0:
                prosody_attrs.append(f'rate="{rate_percent:+d}%"')
                
            # Pitch
            pitch_percent = int((params["pitch"] - 1) * 100)
            if pitch_percent != 0:
                prosody_attrs.append(f'pitch="{pitch_percent:+d}%"')
                
            # Volume
            volume_percent = int((params["volume"] - 1) * 100)
            if volume_percent != 0:
                prosody_attrs.append(f'volume="{volume_percent:+d}%"')
                
            # Build SSML for segment
            if prosody_attrs:
                prosody_tag = f'<prosody {" ".join(prosody_attrs)}>'
                ssml_parts.append(f'{prosody_tag}{segment["text"]}</prosody>')
            else:
                ssml_parts.append(segment["text"])
                
            # Add pause between segments
            pause_ms = int(500 * params.get("pause_factor", 1.0))
            if pause_ms != 500:
                ssml_parts.append(f'<break time="{pause_ms}ms"/>')
                
        ssml_parts.append('</speak>')
        
        return ' '.join(ssml_parts)


if __name__ == "__main__":
    # Test the emotion analyzer
    analyzer = EmotionAnalyzer()
    processor = EmotionTTSProcessor(analyzer)
    
    # Test texts
    test_texts = [
        "I'm so happy to see you today! This is wonderful news!",
        "The loss was devastating. Everyone felt the deep sadness.",
        "This is absolutely unacceptable! How could this happen?",
        "The mysterious sound in the darkness made everyone nervous.",
        "Wow! I can't believe you did that! Amazing!",
        "The report shows that productivity increased by 15% this quarter."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        emotion_scores = analyzer.analyze_emotion(text)
        print(f"Emotions: {emotion_scores}")
        
        tts_params = analyzer.get_tts_parameters(text)
        print(f"TTS Parameters: {tts_params}")
        
        # Test full processing
        emotion_data = processor.prepare_emotional_tts(text)
        print(f"Segments: {len(emotion_data['segments'])}")
        
        # Generate SSML
        ssml = processor.get_ssml_markup(text, emotion_data)
        print(f"SSML preview: {ssml[:100]}...")