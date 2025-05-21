#!/usr/bin/env python3
"""
Script to download pre-trained models for LexoRead.

This script downloads pre-trained model weights for the various components
of the LexoRead system, allowing for quick setup without training models from scratch.
"""

import os
import argparse
import logging
import requests
import json
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_downloader")

# Base URL for model downloads
# Note: This is a placeholder. In a real project, you would set up a server to host these files.
BASE_URL = "https://models.lexoread.openimpactai.org"

# Model information
MODELS = {
    "text_adaptation": {
        "filename": "text_adaptation_model.pt",
        "size_bytes": 250000000,  # ~250 MB
        "description": "Model for adapting text to be more readable for dyslexic users",
        "version": "1.0.0"
    },
    "ocr": {
        "filename": "ocr_model.pt",
        "size_bytes": 350000000,  # ~350 MB
        "description": "OCR model optimized for dyslexic handwriting",
        "version": "1.0.0"
    },
    "tts": {
        "filename": "tts_model.pt",
        "size_bytes": 500000000,  # ~500 MB
        "description": "Text-to-Speech model with clear, dyslexia-friendly pronunciation",
        "version": "1.0.0"
    },
    "reading_level": {
        "filename": "reading_level_model.pt",
        "size_bytes": 150000000,  # ~150 MB
        "description": "Model for assessing reading level and text complexity",
        "version": "1.0.0"
    }
}

def download_file(url, output_path, file_size=None):
    """
    Download a file with progress tracking.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file
        file_size (int, optional): Expected file size in bytes
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the file size from the response headers if not provided
        if file_size is None:
            file_size = int(response.headers.get('content-length', 0))
        
        # Show a progress bar during download
        with open(output_path, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {output_path}")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def download_model(model_name, output_dir=None, force=False):
    """
    Download a pre-trained model.
    
    Args:
        model_name (str): Name of the model to download
        output_dir (str, optional): Directory to save the model
        force (bool): Whether to force download even if file exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    if model_name not in MODELS:
        logger.error(f"Unknown model: {model_name}")
        return False
    
    model_info = MODELS[model_name]
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / model_name
    else:
        output_dir = Path(output_dir) / model_name
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output path
    output_path = output_dir / model_info["filename"]
    
    # Check if file already exists
    if output_path.exists() and not force:
        logger.info(f"Model {model_name} already exists at {output_path}. Use --force to redownload.")
        return True
    
    # Download the model
    url = f"{BASE_URL}/{model_name}/{model_info['filename']}"
    success = download_file(url, output_path, model_info["size_bytes"])
    
    if success:
        # Create a metadata file
        metadata = {
            "name": model_name,
            "version": model_info["version"],
            "description": model_info["description"],
            "download_date": Path(output_path).stat().st_mtime,
            "file_size": Path(output_path).stat().st_size
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return success

def download_all_models(output_dir=None, force=False):
    """
    Download all pre-trained models.
    
    Args:
        output_dir (str, optional): Directory to save the models
        force (bool): Whether to force download even if files exist
        
    Returns:
        bool: True if all downloads were successful, False otherwise
    """
    success = True
    
    for model_name in MODELS:
        logger.info(f"Downloading model: {model_name}")
        model_success = download_model(model_name, output_dir, force)
        success = success and model_success
    
    return success

def create_demo_files(output_dir=None):
    """
    Create demo files for testing when actual models are not available.
    This is useful for development and testing without requiring large downloads.
    
    Args:
        output_dir (str, optional): Directory to save the demo files
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Creating demo model files for testing")
    
    for model_name, model_info in MODELS.items():
        # Determine output directory
        if output_dir is None:
            model_dir = Path(__file__).parent / model_name
        else:
            model_dir = Path(output_dir) / model_name
        
        # Create output directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a small dummy model file
        output_path = model_dir / model_info["filename"]
        
        with open(output_path, 'wb') as f:
            # Write a few kilobytes of random data
            f.write(os.urandom(1024 * 10))
        
        logger.info(f"Created demo file for {model_name} at {output_path}")
        
        # Create a metadata file
        metadata = {
            "name": model_name,
            "version": model_info["version"],
            "description": model_info["description"],
            "download_date": Path(output_path).stat().st_mtime,
            "file_size": Path(output_path).stat().st_size,
            "is_demo": True
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return True

def main():
    """Parse arguments and start the download process."""
    parser = argparse.ArgumentParser(description="Download pre-trained models for LexoRead")
    
    parser.add_argument('--models', type=str, nargs='+', default=['all'],
                        choices=['all', 'text_adaptation', 'ocr', 'tts', 'reading_level'],
                        help='Models to download')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the models')
    
    parser.add_argument('--force', action='store_true',
                        help='Force download even if files already exist')
    
    parser.add_argument('--demo', action='store_true',
                        help='Create demo files instead of downloading actual models')
    
    args = parser.parse_args()
    
    if args.demo:
        create_demo_files(args.output_dir)
    elif 'all' in args.models:
        download_all_models(args.output_dir, args.force)
    else:
        for model_name in args.models:
            download_model(model_name, args.output_dir, args.force)

if __name__ == "__main__":
    main()
