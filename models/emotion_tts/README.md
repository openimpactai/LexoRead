# Emotion-Aware Dynamic TTS for LexoRead

This module enhances LexoRead's Text-to-Speech capabilities with emotion-based voice modulation, creating more engaging and comprehensible audio output for dyslexic readers.

## Features

- **Real-time Emotion Analysis**: Analyzes text sentiment using state-of-the-art transformer models
- **Dynamic Voice Modulation**: Adjusts speech parameters based on detected emotions
- **Smooth Transitions**: Creates natural transitions between emotional segments
- **Dyslexia-Optimized**: Maintains clarity while adding emotional expression

## How It Works

1. **Text Segmentation**: Divides text into manageable segments (sentences or paragraphs)
2. **Emotion Detection**: Analyzes each segment for emotional content
3. **Parameter Mapping**: Maps emotions to TTS parameters (speed, pitch, volume, emphasis)
4. **Synthesis**: Generates speech with emotion-appropriate voice characteristics
5. **Blending**: Smoothly transitions between segments with different emotions

## Supported Emotions

- **Joy/Happiness**: Faster, higher pitch, increased emphasis
- **Sadness**: Slower, lower pitch, softer volume
- **Anger**: Moderate speed, lower pitch, increased volume and emphasis
- **Fear**: Faster, higher pitch, variable volume
- **Surprise**: Very fast, high pitch, increased volume
- **Neutral**: Baseline parameters

## Installation

```bash
cd models/emotion_tts
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from emotion_tts import EmotionAwareTTS

# Initialize the model
tts = EmotionAwareTTS(enable_emotions=True)

# Synthesize speech with emotion
text = "I'm so excited about this new feature! It will help many people."
audio = tts.synthesize_with_emotion(text)

# Save audio
import soundfile as sf
sf.write("emotional_speech.wav", audio, 22050)
```

### Get Emotion Analysis Report

```python
# Analyze emotions in text
report = tts.get_emotion_report(text)
print(f"Dominant emotion: {report['dominant_emotion']}")

# View segment-by-segment analysis
for segment in report['segments']:
    print(f"Text: {segment['text']}")
    print(f"Emotion: {segment['emotion']} ({segment['confidence']:.2f})")
    print(f"TTS Adjustments: {segment['tts_adjustments']}")
```

### Advanced Configuration

```python
# Custom base parameters
base_params = {
    "speed": 0.9,  # Slightly slower for dyslexia
    "volume": 1.1,  # Slightly louder
    "pause_factor": 1.2  # Longer pauses
}

audio = tts.synthesize_with_emotion(text, base_params=base_params)
```

## Integration with LexoRead API

### Add to API Router

```python
# In api/routers/tts.py
from models.emotion_tts import EmotionAwareTTS

@router.post("/tts/emotional")
async def emotional_tts(request: EmotionalTTSRequest):
    tts = EmotionAwareTTS(enable_emotions=True)
    
    # Get emotion report
    report = tts.get_emotion_report(request.text)
    
    # Synthesize with emotions
    audio = tts.synthesize_with_emotion(
        request.text,
        voice=request.voice,
        base_params=request.base_params
    )
    
    # Return audio and emotion data
    return {
        "audio": base64.b64encode(audio.tobytes()).decode(),
        "emotion_report": report,
        "sample_rate": 22050
    }
```

### Request Model

```python
# In api/models/requests.py
class EmotionalTTSRequest(BaseModel):
    text: str
    voice: str = "default"
    enable_emotions: bool = True
    base_params: Optional[Dict[str, float]] = None
```

## Architecture

### EmotionAnalyzer
- Uses HuggingFace's emotion classification models
- Falls back to sentiment analysis if needed
- Maps emotions to TTS parameter adjustments

### EmotionTTSProcessor
- Segments text for granular emotion analysis
- Prepares smooth transitions between segments
- Generates SSML markup for compatible TTS engines

### EmotionAwareTTS
- Extends base DyslexiaTTS model
- Applies emotion-based parameter modulation
- Handles audio processing (time stretching, pitch shifting)

## Performance Considerations

- **Latency**: Emotion analysis adds ~100-200ms per segment
- **Memory**: Transformer models require ~500MB RAM
- **GPU**: Optional but recommended for faster processing

## Future Enhancements

1. **Multi-speaker emotions**: Different voices for different characters
2. **Cultural adaptation**: Emotion expression varies by culture
3. **Real-time streaming**: Process and synthesize in chunks
4. **User preferences**: Learn individual emotion preferences
5. **Emotion intensity**: Fine-grained control over expression level

## Contributing

To add new emotion mappings or improve the system:

1. Add emotion categories to `emotion_parameters` dict
2. Test with diverse text samples
3. Validate with dyslexic users
4. Submit PR with test results

## License

Same as LexoRead (MIT)