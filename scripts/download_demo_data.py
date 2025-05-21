#!/usr/bin/env python3
"""
Download demo data for the LexoRead project.

This script downloads demo data for testing the LexoRead models and API,
including text samples, images for OCR, and audio samples.
"""

import os
import argparse
import logging
from pathlib import Path
import json
import requests
from tqdm import tqdm
import zipfile
import shutil
import random
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("download_demo_data")

# Demo data sources
DEMO_DATA_SOURCES = {
    "text_samples": {
        "url": "https://demo.lexoread.openimpactai.org/data/text_samples.zip",
        "size": 2000000,  # Approximately 2MB
        "description": "Text samples with various reading levels and complexities"
    },
    "ocr_images": {
        "url": "https://demo.lexoread.openimpactai.org/data/ocr_images.zip",
        "size": 5000000,  # Approximately 5MB
        "description": "Images for OCR testing, including handwritten text and printed documents"
    },
    "audio_samples": {
        "url": "https://demo.lexoread.openimpactai.org/data/audio_samples.zip",
        "size": 10000000,  # Approximately 10MB
        "description": "Audio samples for TTS testing"
    },
    "evaluation_data": {
        "url": "https://demo.lexoread.openimpactai.org/data/evaluation_data.zip",
        "size": 3000000,  # Approximately 3MB
        "description": "Data for model evaluation, including ground truth labels"
    }
}

def download_file(url, output_path, expected_size=None, timeout=30):
    """
    Download a file with progress tracking.
    
    Args:
        url (str): URL to download
        output_path (Path): Path to save the file
        expected_size (int, optional): Expected file size in bytes
        timeout (int): Connection timeout in seconds
        
    Returns:
        bool: Whether the download was successful
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get the file size from the response headers if not provided
        total_size = int(response.headers.get('content-length', 0)) or expected_size or 0
        
        # Write the file with progress tracking
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def create_demo_file(output_path, size_kb=1000):
    """
    Create a demo file with random content when a real download isn't available.
    This is useful for development and testing.
    
    Args:
        output_path (Path): Path to save the file
        size_kb (int): Size of the file in KB
        
    Returns:
        bool: Whether the file creation was successful
    """
    try:
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a zip file with random content
        with zipfile.ZipFile(output_path, 'w') as zipf:
            # Create random text files
            for i in range(5):
                # Generate random content
                content = f"Demo content for file {i+1}\n\n"
                content += "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz0123456789\n")
                                  for _ in range(size_kb * 100))
                
                # Add file to zip
                zipf.writestr(f"file_{i+1}.txt", content)
        
        logger.info(f"Created demo file: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating demo file {output_path}: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """
    Extract a zip file.
    
    Args:
        zip_path (Path): Path to the zip file
        extract_dir (Path): Directory to extract to
        
    Returns:
        bool: Whether the extraction was successful
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            # Count the number of files for progress tracking
            file_count = len(zipf.infolist())
            
            # Extract files with progress tracking
            with tqdm(total=file_count, desc=f"Extracting {zip_path.name}") as pbar:
                for file in zipf.infolist():
                    zipf.extract(file, extract_dir)
                    pbar.update(1)
        
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def download_and_extract(data_name, data_info, output_dir, use_demo=False):
    """
    Download and extract a demo data file.
    
    Args:
        data_name (str): Name of the data
        data_info (dict): Information about the data
        output_dir (Path): Directory to save the data
        use_demo (bool): Whether to create a demo file instead of downloading
        
    Returns:
        bool: Whether the download and extraction was successful
    """
    # Create download directory if it doesn't exist
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directory for extracted files
    extract_dir = output_dir / data_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Download or create demo file
    zip_path = download_dir / f"{data_name}.zip"
    
    if use_demo:
        success = create_demo_file(zip_path, size_kb=data_info.get("size", 1000000) // 1000)
    else:
        success = download_file(
            data_info["url"],
            zip_path,
            expected_size=data_info.get("size")
        )
    
    if not success:
        logger.error(f"Failed to download or create {data_name}")
        return False
    
    # Extract the zip file
    success = extract_zip(zip_path, extract_dir)
    
    if not success:
        logger.error(f"Failed to extract {data_name}")
        return False
    
    logger.info(f"Successfully downloaded and extracted {data_name}")
    return True

def create_metadata(output_dir):
    """
    Create a metadata file with information about the demo data.
    
    Args:
        output_dir (Path): Directory where the data is stored
    """
    metadata = {
        "datasets": {},
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "demo_data": True
    }
    
    # Add information about each dataset
    for name, info in DEMO_DATA_SOURCES.items():
        dataset_dir = output_dir / name
        
        if dataset_dir.exists():
            # Count files in the dataset
            file_count = sum(1 for _ in dataset_dir.glob("**/*") if _.is_file())
            
            metadata["datasets"][name] = {
                "description": info.get("description", ""),
                "file_count": file_count,
                "path": str(dataset_dir.relative_to(output_dir.parent)),
            }
    
    # Write metadata file
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata file created: {output_dir / 'metadata.json'}")

def download_demo_data(output_dir, datasets=None, use_demo=False):
    """
    Download demo data for the LexoRead project.
    
    Args:
        output_dir (Path): Directory to save the data
        datasets (list, optional): List of datasets to download (default: all)
        use_demo (bool): Whether to create demo files instead of downloading
        
    Returns:
        bool: Whether all downloads were successful
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to download
    if datasets is None:
        datasets = list(DEMO_DATA_SOURCES.keys())
    
    # Download and extract each dataset
    success = True
    for data_name in datasets:
        if data_name not in DEMO_DATA_SOURCES:
            logger.warning(f"Unknown dataset: {data_name}")
            continue
        
        logger.info(f"Downloading {data_name}...")
        data_success = download_and_extract(
            data_name,
            DEMO_DATA_SOURCES[data_name],
            output_dir,
            use_demo
        )
        success = success and data_success
    
    # Create metadata file
    create_metadata(output_dir)
    
    return success

def main():
    """Parse arguments and start the download process."""
    parser = argparse.ArgumentParser(description="Download demo data for the LexoRead project")
    
    parser.add_argument("--output_dir", type=str, default="./data/demo",
                        help="Directory to save the data")
    
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DEMO_DATA_SOURCES.keys()) + ["all"],
                        default=["all"],
                        help="Datasets to download")
    
    parser.add_argument("--demo", action="store_true",
                        help="Create demo files instead of downloading")
    
    args = parser.parse_args()
    
    # Handle "all" dataset
    if "all" in args.datasets:
        datasets = None
    else:
        datasets = args.datasets
    
    # Download demo data
    success = download_demo_data(
        args.output_dir,
        datasets=datasets,
        use_demo=args.demo
    )
    
    if success:
        logger.info("Demo data download complete!")
    else:
        logger.error("Some downloads failed. Check the logs for details.")
        exit(1)

if __name__ == "__main__":
    main()
