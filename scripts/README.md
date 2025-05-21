# LexoRead Scripts

This directory contains utility scripts for the LexoRead project. These scripts help with development, deployment, and maintenance tasks.

## Available Scripts

- `setup_project.sh`: Set up the development environment
- `download_demo_data.py`: Download demo data for testing
- `generate_test_data.py`: Generate synthetic test data for models
- `evaluate_models.py`: Evaluate model performance on test datasets
- `benchmark.py`: Benchmark API performance
- `export_model.py`: Export models to different formats (ONNX, TorchScript)
- `update_docs.py`: Generate updated API documentation

## Setup Scripts

### setup_project.sh

This script sets up the development environment by installing dependencies, downloading models, and configuring the project.

```bash
./scripts/setup_project.sh
```

### download_demo_data.py

This script downloads demo data for testing the LexoRead models and API.

```bash
python scripts/download_demo_data.py --output_dir ./data/demo
```

## Data Generation Scripts

### generate_test_data.py

This script generates synthetic test data for LexoRead models, including text samples with various reading difficulties, images for OCR testing, and more.

```bash
python scripts/generate_test_data.py --num_samples 100 --output_dir ./data/test
```

## Evaluation Scripts

### evaluate_models.py

This script evaluates the performance of LexoRead models on test datasets, producing detailed performance metrics and visualizations.

```bash
python scripts/evaluate_models.py --model text_adaptation --test_data ./data/test/text_adaptation_test.json
```

### benchmark.py

This script benchmarks the performance of the LexoRead API under various load conditions.

```bash
python scripts/benchmark.py --endpoint /api/text/adapt --concurrency 10 --duration 60
```

## Model Export Scripts

### export_model.py

This script exports LexoRead models to different formats, such as ONNX or TorchScript, for deployment in different environments.

```bash
python scripts/export_model.py --model text_adaptation --format onnx --output ./models/exported
```

## Documentation Scripts

### update_docs.py

This script generates updated API documentation based on the current API implementation.

```bash
python scripts/update_docs.py --output ./docs/api
```

## Creating New Scripts

When creating new scripts, please follow these guidelines:

1. Use a clear, descriptive name that indicates the script's purpose
2. Add a docstring at the top of the script explaining its functionality
3. Include command-line argument parsing for configurable parameters
4. Add proper error handling and logging
5. Update this README with information about the new script

## Script Templates

You can use the templates in the `scripts/templates` directory as a starting point for new scripts.
