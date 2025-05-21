# Download Scripts

This directory contains scripts for downloading and preparing datasets for the LexoRead project.

## Available Scripts

- `download_all.py` - Main script to download all datasets
- `download_common_voice.py` - Downloads Mozilla Common Voice dataset
- `download_textocr.py` - Downloads TextOCR dataset
- `download_dyslexia_handwriting.py` - Downloads dyslexia handwriting samples
- `download_gutenberg_samples.py` - Downloads reading samples from Project Gutenberg
- `download_utils.py` - Utility functions for downloading and processing

## Usage

To download all datasets:

```bash
python download_all.py
```

To download a specific dataset:

```bash
python download_common_voice.py
python download_textocr.py
# etc.
```

## Configuration

Dataset configurations are stored in `../config/dataset_config.json`. You can modify this file to:

- Change download paths
- Select specific language subsets
- Adjust dataset sizes
- Set preprocessing parameters

## Requirements

These scripts require the following dependencies:

- requests
- tqdm
- pandas
- numpy
- librosa (for audio processing)
- Pillow (for image processing)

Install them using:

```bash
pip install -r requirements.txt
```

## Adding New Datasets

To add a new dataset:

1. Create a new download script (e.g., `download_new_dataset.py`)
2. Use the `download_utils.py` functions for common operations
3. Add the dataset to `download_all.py`
4. Update the dataset configuration in `../config/dataset_config.json`

Your script should:
- Download the dataset files
- Process/extract if necessary
- Organize files into the appropriate directory structure
- Generate any metadata files needed

## Troubleshooting

### Common Issues:

1. **Interrupted Downloads**: Add the `--resume` flag to continue interrupted downloads.
2. **Space Issues**: Use the `--sample` flag to download a smaller subset for testing.
3. **Authentication Errors**: Some datasets may require registration - follow the URL provided in the error message.

If you encounter any other issues, please open an issue in the repository.
