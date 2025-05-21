# Data Preprocessing Tools

This directory contains preprocessing utilities for the LexoRead project. These tools help prepare and normalize the data for training and evaluation.

## Available Scripts

- `text_preprocessing.py` - Tools for processing text data
- `audio_preprocessing.py` - Tools for processing audio data
- `image_preprocessing.py` - Tools for processing image data for OCR

## Text Preprocessing

The text preprocessing module (`text_preprocessing.py`) provides utilities for:

- Cleaning and normalizing text
- Tokenizing text into sentences and words
- Calculating readability metrics
- Identifying difficult words for dyslexic readers

### Usage

```bash
# Process a single file
python text_preprocessing.py input.txt output.json

# Process all files in a directory
python text_preprocessing.py input_dir/ output_dir/

# Process all files in a directory and its subdirectories
python text_preprocessing.py input_dir/ output_dir/ --recursive
```

### Readability Metrics

The text preprocessing tool calculates the following readability metrics:

- **Flesch Reading Ease**: Higher scores (90-100) indicate text that is very easy to read, while lower scores (0-30) indicate text that is very difficult to read.
- **Flesch-Kincaid Grade Level**: Corresponds to U.S. grade level required to understand the text.

## Audio Preprocessing

The audio preprocessing module provides utilities for:

- Converting audio to a standard format
- Normalizing volume levels
- Extracting audio features for speech models

## Image Preprocessing

The image preprocessing module provides utilities for:

- Normalizing images for OCR
- Enhancing text visibility
- Converting images to appropriate formats for model training

## Configuration

Preprocessing parameters can be configured in `../config/dataset_config.json`. Each preprocessing module reads its configuration from this file.

## Requirements

Preprocessing tools require the dependencies listed in `../requirements.txt`. Install them with:

```bash
pip install -r ../requirements.txt
```

## Adding New Preprocessing Tools

To add a new preprocessing tool:

1. Create a new Python file in this directory
2. Follow the existing pattern of accepting input and output paths
3. Add any new configuration parameters to `../config/dataset_config.json`
4. Update this README to document the new tool
