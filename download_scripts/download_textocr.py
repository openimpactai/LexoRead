#!/usr/bin/env python3
"""
Script to download and prepare TextOCR dataset for the LexoRead project.
TextOCR provides annotations for text detection and recognition in natural images.
"""

import os
import argparse
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
import random

# Import utility functions
import download_utils

# Set up logging
logger = logging.getLogger("textocr_downloader")

# URLs for TextOCR downloads
TEXTOCR_ANNOTATIONS_URL = "https://textvqa.org/textocr/data/TextOCR_0.1_val.json"
TEXTOCR_IMAGES_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"

def download_textocr(sample_only=False):
    """
    Download and prepare the TextOCR dataset.
    
    Args:
        sample_only (bool): If True, download only a small sample
    
    Returns:
        bool: True if successful, False otherwise
    """
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'ocr_data' / 'textocr'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_path = output_dir / "TextOCR_0.1_val.json"
    images_path = output_dir / "train_val_images.zip"
    
    logger.info("Downloading TextOCR dataset")
    
    # Download annotations
    if not sample_only:
        logger.info("Downloading TextOCR annotations")
        if not download_utils.download_file(TEXTOCR_ANNOTATIONS_URL, annotations_path):
            logger.error("Failed to download TextOCR annotations")
            return False
    
        # Download images archive
        logger.info("Downloading TextOCR images (large file, may take time)")
        if not download_utils.download_file(TEXTOCR_IMAGES_URL, images_path):
            logger.error("Failed to download TextOCR images")
            return False
        
        # Extract images (optional, can be very large)
        logger.info("Note: Image extraction is commented out to save space")
        # Uncomment to extract images (warning: requires significant disk space)
        # if not download_utils.extract_archive(images_path, output_dir / "images"):
        #     logger.error("Failed to extract TextOCR images")
        #     return False
    else:
        # Create a small sample dataset for testing
        create_sample_dataset(output_dir)
    
    # Create metadata file
    metadata = {
        'name': 'TextOCR',
        'version': '0.1',
        'description': 'Dataset for text recognition in natural images',
        'url': 'https://textvqa.org/textocr/dataset',
        'date_prepared': datetime.now().strftime('%Y-%m-%d'),
        'license': 'MIT',
        'citation': 'Singh, A., Pang, G., Toh, M., Huang, J., Galuba, W., & Hassner, T. (2021). TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8802-8812).'
    }
    
    download_utils.create_metadata_file(metadata, output_dir / 'metadata.json')
    
    logger.info("TextOCR dataset preparation complete")
    return True

def create_sample_dataset(output_dir):
    """
    Create a small sample dataset for testing purposes.
    
    Args:
        output_dir (Path): Directory to create sample in
    """
    sample_dir = output_dir / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    images_dir = sample_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Create sample annotations
    sample_annotations = {
        "info": {
            "description": "TextOCR Sample Dataset for LexoRead",
            "version": "0.1",
            "year": 2023,
        },
        "images": [],
        "annotations": []
    }
    
    # Sample text for dyslexia-related content
    sample_texts = [
        "Reading Assistant",
        "Dyslexia Support",
        "Text to Speech",
        "Learning Tools",
        "Reading Made Easy"
    ]
    
    # Create sample image entries and annotations
    image_id = 0
    annotation_id = 0
    
    for i in range(5):
        # Add image entry
        image_id = i
        img_filename = f"sample_image_{i}.jpg"
        sample_annotations["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": 640,
            "height": 480
        })
        
        # Create sample image file (1x1 pixel black image)
        with open(images_dir / img_filename, 'wb') as f:
            # Simple JPG header for a valid file
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01\x01\x01\x00\x48\x00\x48\x00\x00\xFF\xDB\x00\x43\x00\xFF\xC0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x1F\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\xFF\xC4\x00\xB5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01\x7D\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x41\x06\x13\x51\x61\x07\x22\x71\x14\x32\x81\x91\xA1\x08\x23\x42\xB1\xC1\x15\x52\xD1\xF0\x24\x33\x62\x72\x82\x09\x0A\x16\x17\x18\x19\x1A\x25\x26\x27\x28\x29\x2A\x34\x35\x36\x37\x38\x39\x3A\x43\x44\x45\x46\x47\x48\x49\x4A\x53\x54\x55\x56\x57\x58\x59\x5A\x63\x64\x65\x66\x67\x68\x69\x6A\x73\x74\x75\x76\x77\x78\x79\x7A\x83\x84\x85\x86\x87\x88\x89\x8A\x92\x93\x94\x95\x96\x97\x98\x99\x9A\xA2\xA3\xA4\xA5\xA6\xA7\xA8\xA9\xAA\xB2\xB3\xB4\xB5\xB6\xB7\xB8\xB9\xBA\xC2\xC3\xC4\xC5\xC6\xC7\xC8\xC9\xCA\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9\xDA\xE1\xE2\xE3\xE4\xE5\xE6\xE7\xE8\xE9\xEA\xF1\xF2\xF3\xF4\xF5\xF6\xF7\xF8\xF9\xFA\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00\xFD\xFC\xA2\x8A\x28\x03\xFF\xD9')
        
        # Add annotation for text in the image
        text = sample_texts[i]
        annotation_id = i
        
        # Create a bounding box for the text (random position within the image)
        x = random.randint(50, 400)
        y = random.randint(50, 350)
        width = random.randint(100, 200)
        height = random.randint(30, 50)
        
        sample_annotations["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # Text category
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0,
            "text": text
        })
    
    # Save the sample annotations
    with open(sample_dir / "TextOCR_sample.json", 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    logger.info(f"Created sample TextOCR dataset at {sample_dir}")

def main():
    """Parse arguments and start the download process."""
    parser = argparse.ArgumentParser(description="Download TextOCR dataset")
    
    parser.add_argument('--sample', action='store_true', 
                        help='Create a small sample dataset for testing')
    
    args = parser.parse_args()
    
    download_textocr(sample_only=args.sample)

if __name__ == "__main__":
    main()
