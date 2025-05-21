#!/usr/bin/env python3
"""
Script to download dyslexia handwriting samples for the LexoRead project.
This script downloads samples from the dyslexia_detection GitHub repository
and other sources for training handwriting recognition models.
"""

import os
import argparse
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import utility functions
import download_utils

# Set up logging
logger = logging.getLogger("dyslexia_handwriting_downloader")

# Source repositories for dyslexia handwriting data
DYSLEXIA_DETECTION_REPO = "algoasylum/Dyslexia_detection"
DYSLEXIA_APP_REPO = "isha-git/App-for-Dyslexia"

def download_dyslexia_handwriting(sample_only=False):
    """
    Download and prepare dyslexia handwriting samples.
    
    Args:
        sample_only (bool): If True, download only a small sample
        
    Returns:
        bool: True if successful, False otherwise
    """
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'dyslexia_samples' / 'handwriting'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading dyslexia handwriting samples")
    
    if not sample_only:
        # Download from GitHub repositories
        repo_dir = output_dir / "repos"
        repo_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading repository: {DYSLEXIA_DETECTION_REPO}")
        if not download_utils.download_github_repo(DYSLEXIA_DETECTION_REPO, repo_dir / "dyslexia_detection"):
            logger.error(f"Failed to download repository: {DYSLEXIA_DETECTION_REPO}")
            return False
        
        logger.info(f"Downloading repository: {DYSLEXIA_APP_REPO}")
        if not download_utils.download_github_repo(DYSLEXIA_APP_REPO, repo_dir / "dyslexia_app"):
            logger.error(f"Failed to download repository: {DYSLEXIA_APP_REPO}")
            return False
            
        # Process the downloaded repositories to extract relevant files
        process_repositories(repo_dir, output_dir)
    else:
        # Create a small sample dataset for testing
        create_sample_dataset(output_dir)
    
    # Create metadata file
    metadata = {
        'name': 'Dyslexia Handwriting Samples',
        'description': 'Collection of handwriting samples from individuals with dyslexia',
        'sources': [
            {
                'name': 'Dyslexia Detection',
                'url': f'https://github.com/{DYSLEXIA_DETECTION_REPO}'
            },
            {
                'name': 'App for Dyslexia',
                'url': f'https://github.com/{DYSLEXIA_APP_REPO}'
            }
        ],
        'date_prepared': datetime.now().strftime('%Y-%m-%d'),
        'license': 'MIT',
    }
    
    download_utils.create_metadata_file(metadata, output_dir / 'metadata.json')
    
    logger.info("Dyslexia handwriting samples preparation complete")
    return True

def process_repositories(repo_dir, output_dir):
    """
    Process downloaded repositories to extract relevant handwriting samples.
    
    Args:
        repo_dir (Path): Directory containing the downloaded repos
        output_dir (Path): Directory to save processed files
    """
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Process Dyslexia_detection repository
    detection_dir = repo_dir / "dyslexia_detection"
    if detection_dir.exists():
        # Typically, repositories have data in specific directories like 'data', 'samples', etc.
        # This is a placeholder - you would need to adapt to the actual repository structure
        for data_dir in ['data', 'datasets', 'samples', 'example_data']:
            source_data_dir = detection_dir / data_dir
            if source_data_dir.exists():
                logger.info(f"Processing data from {source_data_dir}")
                
                # Copy all image files to the processed directory
                for img_path in source_data_dir.glob('**/*.png'):
                    dst_path = processed_dir / f"detection_{img_path.name}"
                    shutil.copy(img_path, dst_path)
                    logger.debug(f"Copied {img_path} to {dst_path}")
                
                for img_path in source_data_dir.glob('**/*.jpg'):
                    dst_path = processed_dir / f"detection_{img_path.name}"
                    shutil.copy(img_path, dst_path)
                    logger.debug(f"Copied {img_path} to {dst_path}")
    
    # Process App-for-Dyslexia repository
    app_dir = repo_dir / "dyslexia_app"
    if app_dir.exists():
        # Same approach as above
        for data_dir in ['data', 'assets', 'images', 'app/src/main/res']:
            source_data_dir = app_dir / data_dir
            if source_data_dir.exists():
                logger.info(f"Processing data from {source_data_dir}")
                
                # Copy all image files to the processed directory
                for img_path in source_data_dir.glob('**/*.png'):
                    dst_path = processed_dir / f"app_{img_path.name}"
                    shutil.copy(img_path, dst_path)
                    logger.debug(f"Copied {img_path} to {dst_path}")
                
                for img_path in source_data_dir.glob('**/*.jpg'):
                    dst_path = processed_dir / f"app_{img_path.name}"
                    shutil.copy(img_path, dst_path)
                    logger.debug(f"Copied {img_path} to {dst_path}")
    
    # Create an index file listing all processed samples
    create_index_file(processed_dir, output_dir / "index.json")

def create_index_file(samples_dir, index_path):
    """
    Create an index JSON file listing all handwriting samples.
    
    Args:
        samples_dir (Path): Directory containing the samples
        index_path (Path): Path to save the index file
    """
    index = {
        'samples': []
    }
    
    for img_path in samples_dir.glob('*.png'):
        index['samples'].append({
            'filename': img_path.name,
            'type': 'png',
            'path': str(img_path.relative_to(samples_dir.parent))
        })
    
    for img_path in samples_dir.glob('*.jpg'):
        index['samples'].append({
            'filename': img_path.name,
            'type': 'jpg',
            'path': str(img_path.relative_to(samples_dir.parent))
        })
    
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Created index file with {len(index['samples'])} samples")

def create_sample_dataset(output_dir):
    """
    Create a small sample dataset of synthetic dyslexic handwriting.
    
    Args:
        output_dir (Path): Directory to create sample in
    """
    sample_dir = output_dir / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Common letter reversals and errors in dyslexic handwriting
    dyslexia_patterns = [
        {'text': 'b', 'error': 'd'},
        {'text': 'p', 'error': 'q'},
        {'text': 'was', 'error': 'saw'},
        {'text': 'on', 'error': 'no'},
        {'text': 'from', 'error': 'form'},
        {'text': 'there', 'error': 'three'},
        {'text': 'quiet', 'error': 'quite'},
        {'text': 'their', 'error': 'thier'},
        {'text': 'The quick brown fox', 'error': 'The qiuck brwon fox'},
        {'text': 'Reading is fun', 'error': 'Raeding is fun'},
    ]
    
    # Create synthetic handwriting samples
    for i, pattern in enumerate(dyslexia_patterns):
        # Create correct version
        create_handwriting_image(pattern['text'], 
                                 sample_dir / f"sample_correct_{i}.png",
                                 dyslexic=False)
        
        # Create dyslexic version
        create_handwriting_image(pattern['error'], 
                                 sample_dir / f"sample_dyslexic_{i}.png",
                                 dyslexic=True)
    
    # Create an annotations file
    annotations = {
        'samples': []
    }
    
    for i, pattern in enumerate(dyslexia_patterns):
        annotations['samples'].append({
            'id': i,
            'correct_text': pattern['text'],
            'dyslexic_text': pattern['error'],
            'correct_file': f"sample_correct_{i}.png",
            'dyslexic_file': f"sample_dyslexic_{i}.png"
        })
    
    with open(sample_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(f"Created sample dyslexia handwriting dataset at {sample_dir}")

def create_handwriting_image(text, output_path, dyslexic=False):
    """
    Create a synthetic handwriting image.
    
    Args:
        text (str): Text to write
        output_path (Path): Path to save the image
        dyslexic (bool): Whether to apply dyslexic handwriting features
    """
    # Create a blank white image
    width = max(len(text) * 40, 400)
    height = 200
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a handwriting-like font
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Draw the text
    x = 20
    y = height // 2 - 30
    
    if dyslexic:
        # Apply dyslexic features (letter spacing issues, slight rotation, etc.)
        for char in text:
            # Random slight rotation for some letters
            rotation = random.uniform(-10, 10) if random.random() < 0.3 else 0
            
            # Random vertical position for some letters
            y_offset = random.uniform(-5, 5) if random.random() < 0.3 else 0
            
            # Create a separate image for the rotated character
            if rotation != 0:
                char_img = Image.new('RGBA', (50, 50), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((25, 25), char, fill=(0, 0, 0), font=font)
                char_img = char_img.rotate(rotation, resample=Image.BICUBIC, expand=0)
                
                # Paste the rotated character onto the main image
                image.paste(char_img, (x, int(y + y_offset)), char_img)
            else:
                draw.text((x, y + y_offset), char, fill=(0, 0, 0), font=font)
            
            # Adjust x position (more variable spacing for dyslexic writing)
            x += font.getsize(char)[0] + random.randint(0, 10)
    else:
        # Normal writing
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    # Add slight noise for realism
    arr = np.array(image)
    noise = np.random.randint(0, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    
    # Save the image
    image.save(output_path)

def main():
    """Parse arguments and start the download process."""
    parser = argparse.ArgumentParser(description="Download dyslexia handwriting samples")
    
    parser.add_argument('--sample', action='store_true', 
                        help='Create a small sample dataset for testing')
    
    args = parser.parse_args()
    
    download_dyslexia_handwriting(sample_only=args.sample)

if __name__ == "__main__":
    main()
