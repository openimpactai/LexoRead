#!/usr/bin/env python3
"""
Enhanced TTS Model with Emotion Support
Extends the base TTS model with dynamic emotion-based voice modulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from tts.model import DyslexiaTTS, DyslexiaTTSModel, AudioProcessor
from emotion_tts.emotion_analyzer import EmotionAnalyzer, EmotionTTSProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionAwareTTS(DyslexiaTTS):
    """
    Enhanced TTS with emotion-aware speech synthesis.
    """
    
    def __init__(self, model_path=None, device=None, vocab_path=None, 
                 enable_emotions=True):
        """
        Initialize emotion-aware TTS.
        
        Args:
            model_path: Path to pre-trained model
            device: Device to run on
            vocab_path: Path to vocabulary
            enable_emotions: Whether to enable emotion analysis
        """
        super().__init__(model_path, device, vocab_path)
        
        self.enable_emotions = enable_emotions
        if enable_emotions:
            self.emotion_analyzer = EmotionAnalyzer()
            self.emotion_processor = EmotionTTSProcessor(self.emotion_analyzer)
            logger.info("Emotion analysis enabled")
        else:
            self.emotion_analyzer = None
            self.emotion_processor = None
            
    def synthesize_with_emotion(self, text: str, 
                               voice: str = "default",
                               base_params: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize speech with emotion-based modulation.
        
        Args:
            text: Input text
            voice: Voice selection
            base_params: Base synthesis parameters
            
        Returns:
            Audio waveform as numpy array
        """
        if not self.enable_emotions:
            # Fall back to regular synthesis
            return self.synthesize(text, voice=voice, **base_params or {})
            
        # Prepare emotional segments
        emotion_data = self.emotion_processor.prepare_emotional_tts(text, base_params)
        
        # Synthesize each segment
        audio_segments = []
        
        for i, segment in enumerate(emotion_data["segments"]):
            # Get segment text and parameters
            segment_text = segment["text"]
            segment_params = segment["parameters"]
            
            # Log emotion info
            logger.info(f"Segment {i+1}: {segment_params['detected_emotion']} "
                       f"(confidence: {segment_params['emotion_confidence']:.2f})")
            
            # Apply transition if available
            if "transition" in segment:
                # Synthesize transition
                transition_audio = self._synthesize_transition(
                    segment.get("transition", {}),
                    duration_ms=200
                )
                audio_segments.append(transition_audio)
            
            # Synthesize segment with emotion parameters
            segment_audio = self._synthesize_segment(
                segment_text, 
                voice=voice,
                **segment_params
            )
            audio_segments.append(segment_audio)
            
        # Concatenate all segments
        final_audio = np.concatenate(audio_segments)
        
        return final_audio
    
    def _synthesize_segment(self, text: str, voice: str = "default", 
                           speed: float = 1.0, pitch: float = 1.0,
                           volume: float = 1.0, emphasis: float = 1.0,
                           pause_factor: float = 1.0, **kwargs) -> np.ndarray:
        """
        Synthesize a single segment with specific parameters.
        
        Args:
            text: Segment text
            voice: Voice to use
            speed: Speech rate multiplier
            pitch: Pitch multiplier
            volume: Volume multiplier
            emphasis: Emphasis factor
            pause_factor: Pause duration multiplier
            
        Returns:
            Audio waveform
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Encode text
        text_indices = self._text_to_indices(processed_text)
        text_tensor = torch.tensor(text_indices).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate mel spectrogram with emotion parameters
            mel_output, _, alignments = self.model(
                text_tensor,
                speed_factor=speed,
                pitch_factor=pitch
            )
            
            # Apply volume adjustment
            mel_output = mel_output * volume
            
            # Convert to numpy
            mel_output = mel_output.squeeze(0).cpu().numpy()
            
        # Apply time stretching for speed
        if speed != 1.0:
            mel_output = self._time_stretch_mel(mel_output, speed)
            
        # Apply pitch shifting
        if pitch != 1.0:
            mel_output = self._pitch_shift_mel(mel_output, pitch)
            
        # Convert mel to audio
        audio = self.audio_processor.griffin_lim(mel_output)
        
        # Apply emphasis processing
        if emphasis != 1.0:
            audio = self._apply_emphasis(audio, emphasis)
            
        # Add appropriate pauses
        if pause_factor != 1.0:
            pause_duration = int(0.3 * self.audio_processor.sample_rate * pause_factor)
            pause = np.zeros(pause_duration)
            audio = np.concatenate([audio, pause])
            
        return audio
    
    def _synthesize_transition(self, transition_params: Dict, 
                              duration_ms: int = 200) -> np.ndarray:
        """
        Create smooth transition between segments.
        
        Args:
            transition_params: Transition parameters
            duration_ms: Transition duration
            
        Returns:
            Transition audio (mostly silence with parameter interpolation)
        """
        samples = int(self.audio_processor.sample_rate * duration_ms / 1000)
        
        # Create smooth transition (fade or silence)
        transition = np.zeros(samples)
        
        # Add subtle white noise for natural transition
        noise_level = 0.001
        transition += np.random.normal(0, noise_level, samples)
        
        # Apply envelope
        envelope = np.hanning(samples)
        transition *= envelope
        
        return transition
    
    def _time_stretch_mel(self, mel: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply time stretching to mel spectrogram.
        
        Args:
            mel: Mel spectrogram [time, mel_bins]
            factor: Stretch factor (>1 = slower, <1 = faster)
            
        Returns:
            Stretched mel spectrogram
        """
        if factor == 1.0:
            return mel
            
        # Simple linear interpolation for time stretching
        original_len = mel.shape[0]
        new_len = int(original_len / factor)
        
        time_indices = np.linspace(0, original_len - 1, new_len)
        stretched_mel = np.zeros((new_len, mel.shape[1]))
        
        for i in range(mel.shape[1]):
            stretched_mel[:, i] = np.interp(time_indices, 
                                           np.arange(original_len), 
                                           mel[:, i])
            
        return stretched_mel
    
    def _pitch_shift_mel(self, mel: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply pitch shifting to mel spectrogram.
        
        Args:
            mel: Mel spectrogram
            factor: Pitch shift factor
            
        Returns:
            Pitch-shifted mel spectrogram
        """
        if factor == 1.0:
            return mel
            
        # Simplified pitch shifting by shifting mel bins
        shift_bins = int((factor - 1) * 12)  # Approximate semitones
        
        if shift_bins > 0:
            # Shift up
            shifted = np.pad(mel[:, :-shift_bins], 
                           ((0, 0), (shift_bins, 0)), 
                           mode='edge')
        elif shift_bins < 0:
            # Shift down
            shifted = np.pad(mel[:, -shift_bins:], 
                           ((0, 0), (0, -shift_bins)), 
                           mode='edge')
        else:
            shifted = mel
            
        return shifted
    
    def _apply_emphasis(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply emphasis to audio.
        
        Args:
            audio: Audio waveform
            factor: Emphasis factor
            
        Returns:
            Emphasized audio
        """
        if factor == 1.0:
            return audio
            
        # Apply dynamic range compression/expansion
        # Simple implementation using power law
        sign = np.sign(audio)
        magnitude = np.abs(audio)
        
        # Apply emphasis
        emphasized = sign * (magnitude ** (1 / factor))
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(emphasized))
        if max_val > 0:
            emphasized = emphasized / max_val * 0.95
            
        return emphasized
    
    def get_emotion_report(self, text: str) -> Dict:
        """
        Get detailed emotion analysis report for text.
        
        Args:
            text: Input text
            
        Returns:
            Emotion analysis report
        """
        if not self.enable_emotions:
            return {"error": "Emotion analysis not enabled"}
            
        # Analyze full text
        overall_emotions = self.emotion_analyzer.analyze_emotion(text)
        
        # Analyze segments
        segments = self.emotion_analyzer.analyze_text_segments(text)
        
        report = {
            "overall_emotions": overall_emotions,
            "dominant_emotion": max(overall_emotions, key=overall_emotions.get),
            "segments": []
        }
        
        for segment in segments:
            report["segments"].append({
                "text": segment["text"][:50] + "...",
                "emotion": segment["parameters"]["detected_emotion"],
                "confidence": segment["parameters"]["emotion_confidence"],
                "tts_adjustments": {
                    k: v for k, v in segment["parameters"].items()
                    if k in ["speed", "pitch", "volume", "emphasis"]
                }
            })
            
        return report


def demo_emotion_tts():
    """
    Demonstrate emotion-aware TTS capabilities.
    """
    # Initialize model
    tts = EmotionAwareTTS(enable_emotions=True)
    
    # Test texts with different emotions
    test_cases = [
        {
            "text": "I just won the lottery! This is the best day of my life!",
            "expected_emotion": "joy"
        },
        {
            "text": "We regret to inform you that your application has been rejected.",
            "expected_emotion": "sadness"
        },
        {
            "text": "How dare you speak to me like that! This is outrageous!",
            "expected_emotion": "anger"
        },
        {
            "text": "The technical documentation clearly states the requirements.",
            "expected_emotion": "neutral"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Text: {test_case['text']}")
        
        # Get emotion report
        report = tts.get_emotion_report(test_case['text'])
        print(f"Detected emotion: {report['dominant_emotion']}")
        print(f"Expected emotion: {test_case['expected_emotion']}")
        
        # Show TTS adjustments
        if report["segments"]:
            adjustments = report["segments"][0]["tts_adjustments"]
            print("TTS Adjustments:")
            for param, value in adjustments.items():
                print(f"  {param}: {value:.2f}")


if __name__ == "__main__":
    demo_emotion_tts()