#!/usr/bin/env python3
"""
Main script to download all datasets for the LexoRead project.
This script orchestrates the download of all required datasets.
"""

import os
import argparse
import json
import logging
from pathlib import Path

# Import individual download scripts
from download_common_voice import download_common_voice
from download_textocr import download_textocr
from download_dyslexia_handwriting import download_dyslexia_handwriting
from download_gutenberg_samples import download_gutenberg_samples
import download_utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("dataset_downloader")

def load_config():
    """Load the dataset configuration file."""
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'dataset_config.json'
        if not config_path.exists():
            logger.warning(f"Configuration file not found at {config_path}. Using default settings.")
            return {}
        
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def create_directories():
    """Create necessary directories for the datasets."""
    base_dir = Path(__file__).parent.parent
    
    # Main directories
    directories = [
        base_dir / 'text_corpus' / 'reading_levels' / 'elementary',
        base_dir / 'text_corpus' / 'reading_levels' / 'middle_school',
        base_dir / 'text_corpus' / 'reading_levels' / 'high_school',
        base_dir / 'dyslexia_samples' / 'handwriting',
        base_dir / 'dyslexia_samples' / 'reading_errors',
        base_dir / 'ocr_data' / 'annotated_images',
        base_dir / 'ocr_data' / 'test_samples',
        base_dir / 'speech_samples' / 'tts_training',
        base_dir / 'config'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_all_datasets(args):
    """Download all datasets based on configuration and arguments."""
    config = load_config()
    
    # Create necessary directories
    create_directories()
    
    # Download datasets
    logger.info("Starting downloads of all datasets...")
    
    try:
        # Common Voice dataset
        if args.common_voice:
            logger.info("Downloading Common Voice dataset...")
            download_common_voice(
                sample_only=args.sample,
                languages=config.get('common_voice', {}).get('languages', ['en'])
            )
        
        # TextOCR dataset
        if args.textocr:
            logger.info("Downloading TextOCR dataset...")
            download_textocr(
                sample_only=args.sample
            )
        
        # Dyslexia handwriting samples
        if args.dyslexia_handwriting:
            logger.info("Downloading Dyslexia handwriting samples...")
            download_dyslexia_handwriting(
                sample_only=args.sample
            )
        
        # Project Gutenberg samples
        if args.gutenberg:
            logger.info("Downloading Project Gutenberg samples...")
            download_gutenberg_samples(
                sample_only=args.sample,
                num_books=config.get('gutenberg', {}).get('num_books', 10)
            )
        
        logger.info("All selected datasets downloaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during download: {e}", exc_info=True)
        return False
    
    return True

def main():
    """Main function to parse arguments and start downloads."""
    parser = argparse.ArgumentParser(description="Download and prepare datasets for LexoRead")
    
    parser.add_argument('--sample', action='store_true', 
                        help='Download only a small sample of each dataset for testing')
    
    parser.add_argument('--resume', action='store_true',
                        help='Attempt to resume interrupted downloads')
    
    parser.add_argument('--all', action='store_true',
                        help='Download all datasets')
    
    parser.add_argument('--common-voice', action='store_true',
                        help='Download Common Voice dataset')
    
    parser.add_argument('--textocr', action='store_true',
                        help='Download TextOCR dataset')
    
    parser.add_argument('--dyslexia-handwriting', action='store_true',
                        help='Download dyslexia handwriting samples')
    
    parser.add_argument('--gutenberg', action='store_true',
                        help='Download reading samples from Project Gutenberg')
    
    args = parser.parse_args()
    
    # If --all is specified or no specific dataset is selected, download all
    if args.all or not (args.common_voice or args.textocr or 
                        args.dyslexia_handwriting or args.gutenberg):
        args.common_voice = True
        args.textocr = True
        args.dyslexia_handwriting = True
        args.gutenberg = True
    
    download_all_datasets(args)

if __name__ == "__main__":
    main()
