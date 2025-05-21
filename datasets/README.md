# LexoRead Datasets

This directory contains datasets and utilities for the LexoRead project, an AI-powered reading assistant for dyslexia and reading impairments.

## Directory Structure

```
datasets/
│
├── download_scripts/     # Scripts to download and prepare external datasets
│
├── text_corpus/          # Reading materials in various difficulty levels
│   ├── reading_levels/
│   │   ├── elementary/
│   │   ├── middle_school/
│   │   └── high_school/
│
├── dyslexia_samples/     # Samples of dyslexic handwriting and reading patterns
│   ├── handwriting/
│   ├── reading_errors/
│
├── ocr_data/             # Data for OCR model training
│   ├── annotated_images/
│   ├── test_samples/
│
├── speech_samples/       # Audio data for TTS and STT models
│   ├── tts_training/
│
├── preprocessing/        # Utilities for data preprocessing
│
└── config/               # Configuration files for dataset parameters
```

## Dataset Sources

The datasets used in this project come from the following sources:

1. **Common Voice** - Mozilla's open-source collection of voice recordings
2. **TextOCR** - A dataset for text recognition in natural images
3. **Dyslexia Handwriting Samples** - Collected from various research studies
4. **Project Gutenberg** - Free eBooks for reading samples
5. **OpenDyslexic Font** - Typeface designed for readers with dyslexia

## Getting Started

To download and prepare the datasets:

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the download scripts:
```bash
cd download_scripts
python download_all.py
```

This will download and organize all the necessary datasets in their appropriate directories.

## Data Usage

These datasets are intended for:

1. Training OCR models to recognize text in images
2. Developing text-to-speech models optimized for dyslexic readers
3. Building text simplification and adaptation models
4. Training models to detect and assist with common dyslexic reading patterns

## Contributing

If you have access to additional datasets that might be valuable for the LexoRead project, please follow these steps:

1. Create a new download script in the `download_scripts/` directory
2. Update the README to include information about the new dataset
3. Submit a pull request with your changes

## License

Each dataset has its own license terms. Please refer to the original sources for specific licensing information.

## Contact

For questions about the datasets, please open an issue in the repository or contact the project maintainers.
