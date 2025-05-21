#!/usr/bin/env python3
"""
Utility functions for downloading and processing datasets for the LexoRead project.
"""

import os
import requests
import shutil
import zipfile
import tarfile
import logging
import hashlib
from pathlib import Path
from tqdm import tqdm

# Set up logging
logger = logging.getLogger("download_utils")

def calculate_md5(file_path, chunk_size=8192):
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path (str): Path to the file
        chunk_size (int): Size of chunks to read
        
    Returns:
        str: MD5 hash of the file
    """
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def validate_file(file_path, expected_md5=None, min_size_kb=1):
    """
    Validate a downloaded file by checking MD5 hash and minimum size.
    
    Args:
        file_path (str): Path to the file
        expected_md5 (str, optional): Expected MD5 hash
        min_size_kb (int): Minimum expected file size in KB
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Check minimum size
    file_size_kb = os.path.getsize(file_path) / 1024
    if file_size_kb < min_size_kb:
        logger.error(f"File too small: {file_path} ({file_size_kb:.2f} KB < {min_size_kb} KB)")
        return False
    
    # Check MD5 if provided
    if expected_md5:
        actual_md5 = calculate_md5(file_path)
        if actual_md5 != expected_md5:
            logger.error(f"MD5 mismatch for {file_path}: Expected {expected_md5}, got {actual_md5}")
            return False
        logger.info(f"MD5 verified for {file_path}")
    
    return True

def download_file(url, output_path, resume=True, expected_md5=None):
    """
    Download a file with progress tracking and optional resumption.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file
        resume (bool): Whether to resume a partial download
        expected_md5 (str, optional): Expected MD5 hash for validation
        
    Returns:
        bool: True if download successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_size = 0
    headers = {}
    
    # Check if file exists and we should resume
    if resume and output_path.exists():
        existing_size = output_path.stat().st_size
        headers = {'Range': f'bytes={existing_size}-'}
        logger.info(f"Resuming download of {url} from byte {existing_size}")
    
    try:
        # Make request with stream=True to download in chunks
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            # Get total size if available
            total_size = int(r.headers.get('content-length', 0)) + existing_size
            
            # Create a progress bar
            progress = tqdm(
                total=total_size,
                initial=existing_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {output_path.name}"
            )
            
            # Open file in append mode if resuming, otherwise write mode
            with open(output_path, 'ab' if existing_size > 0 else 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            
            progress.close()
            
            # Validate the downloaded file
            if expected_md5:
                if not validate_file(output_path, expected_md5):
                    logger.error(f"Downloaded file validation failed: {output_path}")
                    return False
            
            logger.info(f"Successfully downloaded {url} to {output_path}")
            return True
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_archive(archive_path, output_dir, remove_after=False):
    """
    Extract a zip or tar archive to an output directory.
    
    Args:
        archive_path (str): Path to the archive file
        output_dir (str): Directory to extract to
        remove_after (bool): Whether to remove the archive after extraction
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Extracting {archive_path} to {output_dir}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Get total size for progress
                total_size = sum(info.file_size for info in zip_ref.infolist())
                extracted_size = 0
                
                for member in tqdm(zip_ref.infolist(), desc=f"Extracting {archive_path.name}"):
                    zip_ref.extract(member, output_dir)
                    extracted_size += member.file_size
                    
        elif archive_path.suffix in ['.tar', '.gz', '.bz2', '.xz']:
            with tarfile.open(archive_path) as tar_ref:
                members = tar_ref.getmembers()
                for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                    tar_ref.extract(member, output_dir)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
        
        if remove_after:
            logger.info(f"Removing archive after extraction: {archive_path}")
            archive_path.unlink()
        
        logger.info(f"Successfully extracted {archive_path} to {output_dir}")
        return True
        
    except (zipfile.BadZipFile, tarfile.ReadError) as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error extracting {archive_path}: {e}")
        return False

def download_github_repo(repo_url, output_dir, branch='main'):
    """
    Download a GitHub repository (without using git).
    
    Args:
        repo_url (str): URL of the GitHub repository (e.g., 'username/repo')
        output_dir (str): Directory to save the repository
        branch (str): Branch to download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    # Remove https://github.com/ if present
    repo_url = repo_url.replace('https://github.com/', '')
    
    # Construct the archive URL
    archive_url = f"https://github.com/{repo_url}/archive/refs/heads/{branch}.zip"
    archive_path = Path(output_dir) / f"{repo_url.replace('/', '_')}_{branch}.zip"
    
    # Download the archive
    if not download_file(archive_url, archive_path):
        return False
    
    # Extract the archive
    return extract_archive(archive_path, output_dir, remove_after=True)

def create_metadata_file(data, output_path):
    """
    Create a metadata JSON file.
    
    Args:
        data (dict): Metadata dictionary
        output_path (str): Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    import json
    
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Created metadata file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")
        return False
