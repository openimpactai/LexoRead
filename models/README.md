# LexoRead Models

This directory contains machine learning models and related utilities for the LexoRead project. These models power the core functionality of the reading assistant.

## Model Components

The LexoRead system is composed of several model components:

1. **Text Adaptation Model** (`text_adaptation/`): Adapts text to make it more readable for individuals with dyslexia.
2. **OCR Model** (`ocr/`): Extracts text from images and documents.
3. **Text-to-Speech Model** (`tts/`): Converts text to natural-sounding speech.
4. **Reading Level Assessment** (`reading_level/`): Analyzes text complexity and determines reading difficulty.
5. **User Profiling** (`user_profile/`): Learns user-specific reading patterns and preferences.

## Setup

Set up the models by installing dependencies:

```bash
pip install -r requirements.txt
```

For GPU support:

```bash
pip install -r requirements-gpu.txt
```

## Model Usage

Each model component has its own directory with specific instructions for training and inference. Here's a quick overview:

### Text Adaptation

```python
from lexoread.models.text_adaptation import DyslexiaTextAdapter

adapter = DyslexiaTextAdapter()
adapted_text = adapter.adapt("Original text to be adapted for easier reading")
```

### OCR

```python
from lexoread.models.ocr import DyslexiaOCR

ocr = DyslexiaOCR()
text = ocr.extract_text("path/to/image.jpg")
```

### Text-to-Speech

```python
from lexoread.models.tts import DyslexiaTTS

tts = DyslexiaTTS()
audio = tts.synthesize("Text to be read aloud", voice="female", speed=1.0)
audio.save("output.wav")
```

### Reading Level Assessment

```python
from lexoread.models.reading_level import ReadingLevelAssessor

assessor = ReadingLevelAssessor()
difficulty = assessor.assess_text("Text to analyze")
print(f"Reading level: {difficulty.level}, Grade: {difficulty.grade}")
```

### User Profiling

```python
from lexoread.models.user_profile import UserProfiler

profiler = UserProfiler(user_id="user123")
profiler.update(text="Example text", reading_time=120, comprehension_score=0.8)
adaptations = profiler.get_recommended_adaptations()
```

## Training

To train the models, use the training scripts in each model directory:

```bash
# Train Text Adaptation model
python models/text_adaptation/train.py --data_path datasets/text_corpus --epochs 10

# Train OCR model
python models/ocr/train.py --data_path datasets/ocr_data --epochs 20

# Train TTS model
python models/tts/train.py --data_path datasets/speech_samples --epochs 30
```

## Pre-trained Models

We provide pre-trained model weights for quick start. To download them:

```bash
python models/download_pretrained.py --models all
```

Available options for `--models`:
- `all`: All pre-trained models
- `text_adaptation`: Text Adaptation model only
- `ocr`: OCR model only
- `tts`: Text-to-Speech model only
- `reading_level`: Reading Level Assessment model only

## Evaluation

Evaluate the models with test datasets:

```bash
python models/evaluate.py --model text_adaptation --test_data datasets/evaluation/text_adaptation_test.json
```

## Model Architecture Details

### Text Adaptation

- Based on transformer architecture fine-tuned for dyslexia-specific text adaptation
- Uses BERT-based encoder with customized output layer for word-level adjustments
- Supports various adaptation strategies (font changes, spacing, color schemes)

### OCR

- Built on EfficientNet backbone with transformer decoder
- Optimized for dyslexic handwriting and difficult-to-read text
- Includes specialized preprocessing for image enhancement

### Text-to-Speech

- Based on Tacotron2 + WaveGlow architecture
- Optimized for clear, dyslexia-friendly speech synthesis
- Includes customizable speech parameters (speed, emphasis)

### Reading Level Assessment

- Ensemble of feature-based models and transformer-based text encoders
- Outputs multiple metrics (Flesch-Kincaid, syllable complexity, sentence structure)
- Calibrated specifically for dyslexic readers

### User Profiling

- Reinforcement learning system that adapts to individual reading patterns
- Tracks reading speed, comprehension, and user feedback
- Recommends personalized text adaptations

## Extending the Models

To add a new model component:

1. Create a new directory under `models/`
2. Implement the model using the template structure (see `models/template/`)
3. Add training and inference scripts
4. Update the README
5. Add tests to ensure proper functionality

## License

The model code is licensed under the MIT License. However, pre-trained model weights may have different licenses depending on the training data used. Please check the README in each model directory for specific licensing information.
