#!/usr/bin/env python3
"""
Script to download and prepare Mozilla Common Voice dataset for the LexoRead project.
Common Voice is a crowdsourced speech dataset that can be used for training TTS models.
"""

import os
import argparse
import logging
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

# Import utility functions
import download_utils

# Set up logging
logger = logging.getLogger("common_voice_downloader")

# URLs for Common Voice downloads
# Note: Common Voice requires registration, so these URLs are for reference
# Users need to download the dataset manually or use an API key
COMMON_VOICE_BASE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-10.0-2022-07-04/"
COMMON_VOICE_LANGUAGES = {
    'en': 'en/en.tar.gz',
    'fr': 'fr/fr.tar.gz',
    'de': 'de/de.tar.gz',
    'es': 'es/es.tar.gz',
    'tr': 'tr/tr.tar.gz',
}

def download_common_voice(sample_only=False, languages=None):
    """
    Download and prepare the Mozilla Common Voice dataset.
    
    Args:
        sample_only (bool): If True, download only a small sample
        languages (list): List of language codes to download
    
    Returns:
        bool: True if successful, False otherwise
    """
    if languages is None:
        languages = ['en']  # Default to English only
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'speech_samples' / 'tts_training' / 'common_voice'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading Common Voice dataset for languages: {languages}")
    logger.info("Note: Common Voice requires registration. You may need to download files manually.")
    
    for lang_code in languages:
        if lang_code not in COMMON_VOICE_LANGUAGES:
            logger.warning(f"Language code '{lang_code}' not found in available languages. Skipping.")
            continue
        
        lang_url = COMMON_VOICE_BASE_URL + COMMON_VOICE_LANGUAGES[lang_code]
        lang_output_dir = output_dir / lang_code
        archive_path = output_dir / f"{lang_code}.tar.gz"
        
        logger.info(f"Processing language: {lang_code}")
        
        # Create a URL notice file with instructions
        notice_path = lang_output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(notice_path, 'w') as f:
            f.write(f"""
Mozilla Common Voice Dataset

Due to licensing restrictions, this dataset cannot be downloaded automatically.
Please visit the Common Voice website, create an account, and download the dataset manually:

1. Visit: https://commonvoice.mozilla.org/en/datasets
2. Create an account and agree to the terms
3. Download the {lang_code} dataset (version 10.0 or later)
4. Extract the downloaded file to this directory

Download URL (requires login): {lang_url}

For LexoRead project use, we need the following files:
- clips/ directory (contains .mp3 files)
- validated.tsv (metadata for validated clips)
""")
        
        logger.info(f"Created download instructions at {notice_path}")
        
        # Create a sample dataset for testing if requested
        if sample_only:
            create_sample_dataset(lang_output_dir, lang_code)
            
    # Create metadata file
    metadata = {
        'name': 'Mozilla Common Voice',
        'version': '10.0',
        'description': 'Crowdsourced multilingual speech dataset',
        'url': 'https://commonvoice.mozilla.org/en/datasets',
        'languages': languages,
        'date_prepared': datetime.now().strftime('%Y-%m-%d'),
        'license': 'CC0',
        'citation': 'Mozilla Common Voice. https://commonvoice.mozilla.org/en/datasets'
    }
    
    download_utils.create_metadata_file(metadata, output_dir / 'metadata.json')
    
    logger.info("Common Voice dataset preparation complete.")
    return True

def create_sample_dataset(output_dir, lang_code):
    """
    Create a small sample dataset for testing purposes.
    This generates synthetic data to mimic the structure of Common Voice.
    
    Args:
        output_dir (Path): Directory to create sample in
        lang_code (str): Language code
    """
    sample_dir = output_dir / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = sample_dir / 'clips'
    clips_dir.mkdir(exist_ok=True)
    
    # Sample sentences for dyslexia-related content
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Reading can be challenging for people with dyslexia.",
        "Special fonts can help dyslexic readers distinguish letters more easily.",
        "Text-to-speech technology makes reading more accessible.",
    ]
    
    # Create a sample TSV file
    data = []
    for i, sentence in enumerate(sentences):
        filename = f"sample_{i}.mp3"
        data.append({
            'client_id': f'sample_user_{i}',
            'path': filename,
            'sentence': sentence,
            'up_votes': 5,
            'down_votes': 0,
            'age': 'twenties',
            'gender': 'other',
            'accent': lang_code,
            'locale': lang_code,
            'segment': ''
        })
    
    df = pd.DataFrame(data)
    df.to_csv(sample_dir / 'validated.tsv', sep='\t', index=False)
    
    # Create empty MP3 files (placeholders)
    for i in range(len(sentences)):
        with open(clips_dir / f"sample_{i}.mp3", 'wb') as f:
            # Write a minimal MP3 header to create a valid file
            f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    
    logger.info(f"Created sample dataset at {sample_dir}")

def main():
    """Parse arguments and start the download process."""
    parser = argparse.ArgumentParser(description="Download Mozilla Common Voice dataset")
    
    parser.add_argument('--sample', action='store_true', 
                        help='Create a small sample dataset for testing')
    
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                        help='Language codes to download (e.g., en fr de)')
    
    args = parser.parse_args()
    
    download_common_voice(sample_only=args.sample, languages=args.languages)

if __name__ == "__main__":
    main()
