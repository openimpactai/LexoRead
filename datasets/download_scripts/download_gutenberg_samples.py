#!/usr/bin/env python3
"""
Script to download reading samples from Project Gutenberg for the LexoRead project.
This script downloads text from open-domain books to create a corpus for testing
and training reading comprehension models.
"""

import os
import argparse
import logging
import requests
import json
import random
import re
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# Import utility functions
import download_utils

# Set up logging
logger = logging.getLogger("gutenberg_downloader")

# Project Gutenberg base URL
GUTENBERG_BASE_URL = "https://www.gutenberg.org"
GUTENBERG_MIRROR = "https://www.gutenberg.org/files"

# Dyslexia-friendly reading materials (books with simpler language)
DYSLEXIA_FRIENDLY_BOOKS = [
    # Children's literature (often simpler language)
    {"id": 11, "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "level": "middle_school"},
    {"id": 16, "title": "Peter Pan", "author": "J. M. Barrie", "level": "elementary"},
    {"id": 844, "title": "The Importance of Being Earnest", "author": "Oscar Wilde", "level": "high_school"},
    {"id": 2591, "title": "Grimms' Fairy Tales", "author": "Jacob Grimm and Wilhelm Grimm", "level": "elementary"},
    {"id": 23, "title": "Narrative of the Life of Frederick Douglass", "author": "Frederick Douglass", "level": "high_school"},
    {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen", "level": "high_school"},
    {"id": 174, "title": "The Picture of Dorian Gray", "author": "Oscar Wilde", "level": "high_school"},
    {"id": 1661, "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "level": "middle_school"},
    {"id": 1497, "title": "Republic", "author": "Plato", "level": "high_school"},
    {"id": 345, "title": "Dracula", "author": "Bram Stoker", "level": "high_school"},
    {"id": 76, "title": "Adventures of Huckleberry Finn", "author": "Mark Twain", "level": "middle_school"},
    {"id": 33, "title": "The Scarlet Letter", "author": "Nathaniel Hawthorne", "level": "high_school"},
    {"id": 98, "title": "A Tale of Two Cities", "author": "Charles Dickens", "level": "high_school"},
    {"id": 1080, "title": "A Modest Proposal", "author": "Jonathan Swift", "level": "high_school"},
    {"id": 84, "title": "Frankenstein", "author": "Mary Wollstonecraft Shelley", "level": "high_school"},
]

def download_gutenberg_samples(sample_only=False, num_books=None):
    """
    Download and prepare reading samples from Project Gutenberg.
    
    Args:
        sample_only (bool): If True, download only a small sample
        num_books (int): Number of books to download, None for all
    
    Returns:
        bool: True if successful, False otherwise
    """
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'text_corpus' / 'reading_levels'
    
    # Create directories for each reading level
    for level in ['elementary', 'middle_school', 'high_school']:
        level_dir = output_dir / level
        level_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading Project Gutenberg reading samples")
    
    if num_books is None:
        num_books = len(DYSLEXIA_FRIENDLY_BOOKS)
    
    books_to_download = DYSLEXIA_FRIENDLY_BOOKS[:num_books]
    
    if sample_only:
        # Just process a small subset for testing
        books_to_download = books_to_download[:3]
    
    for book in books_to_download:
        book_id = book["id"]
        level = book["level"]
        title = book["title"]
        author = book["author"]
        
        level_dir = output_dir / level
        book_dir = level_dir / f"book_{book_id}"
        book_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing book: {title} by {author} (ID: {book_id})")
        
        # Determine URL for the text file
        # Project Gutenberg URLs can vary, but most follow a pattern
        text_url = f"{GUTENBERG_MIRROR}/{book_id}/{book_id}-0.txt"
        alt_text_url = f"{GUTENBERG_MIRROR}/{book_id}/{book_id}.txt"
        
        # Try to download the text file
        text_path = book_dir / f"{book_id}.txt"
        if not text_path.exists():
            try:
                download_utils.download_file(text_url, text_path)
            except:
                try:
                    download_utils.download_file(alt_text_url, text_path)
                except:
                    logger.warning(f"Could not download text for book {book_id}. Using sample text.")
                    create_sample_text(book, text_path)
        
        # Process the text file to create reading samples
        create_reading_samples(text_path, book_dir, book)
    
    # Create metadata file
    metadata = {
        'name': 'Project Gutenberg Reading Samples',
        'description': 'Reading samples from public domain books, organized by reading level',
        'source': 'Project Gutenberg (https://www.gutenberg.org)',
        'books': [
            {
                'id': book['id'],
                'title': book['title'],
                'author': book['author'],
                'level': book['level']
            }
            for book in books_to_download
        ],
        'date_prepared': datetime.now().strftime('%Y-%m-%d'),
        'license': 'Public Domain',
    }
    
    download_utils.create_metadata_file(metadata, output_dir / 'metadata.json')
    
    logger.info("Project Gutenberg reading samples preparation complete")
    return True

def create_sample_text(book, output_path):
    """
    Create a sample text file when download fails.
    
    Args:
        book (dict): Book information
        output_path (Path): Path to save the sample text
    """
    title = book["title"]
    author = book["author"]
    
    sample_text = f"""
{title}
by {author}

This is a sample text file created for the LexoRead project.
The actual text from Project Gutenberg could not be downloaded.

Chapter 1

It was a bright cold day in April, and the clocks were striking thirteen.
Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, 
slipped quickly through the glass doors of Victory Mansions, though not quickly enough to 
prevent a swirl of gritty dust from entering along with him.

The hallway smelt of boiled cabbage and old rag mats. At one end of it a colored poster, 
too large for indoor display, had been tacked to the wall. It depicted simply an enormous face, 
more than a meter wide: the face of a man of about forty-five, with a heavy black mustache and 
ruggedly handsome features.

(This sample text includes an excerpt from George Orwell's 1984, which is not yet in the public domain 
in the United States. This is only a placeholder and should be replaced with the actual text from 
Project Gutenberg when available.)
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

def create_reading_samples(text_path, output_dir, book_info):
    """
    Process a book text file to create reading samples at different difficulty levels.
    
    Args:
        text_path (Path): Path to the book text file
        output_dir (Path): Directory to save the samples
        book_info (dict): Book information
    """
    if not text_path.exists():
        logger.error(f"Text file not found: {text_path}")
        return
    
    try:
        with open(text_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Error reading text file {text_path}: {e}")
        return
    
    # Extract metadata
    title = book_info["title"]
    author = book_info["author"]
    level = book_info["level"]
    
    # Clean up the text
    # Remove Project Gutenberg header and footer
    text = remove_gutenberg_header_footer(text)
    
    # Split into chapters or sections (simple approach)
    sections = split_into_sections(text)
    
    # Create a JSON file with metadata and samples
    samples = []
    
    for i, section in enumerate(sections):
        # Skip empty sections
        if len(section.strip()) < 100:
            continue
        
        # Create a sample with the section
        sample = {
            'id': f"{book_info['id']}_{i}",
            'title': f"{title} - Section {i+1}",
            'author': author,
            'level': level,
            'text': section[:1000],  # First 1000 chars as preview
            'full_text_file': f"section_{i+1}.txt"
        }
        
        samples.append(sample)
        
        # Save the full section text
        section_path = output_dir / f"section_{i+1}.txt"
        with open(section_path, 'w', encoding='utf-8') as f:
            f.write(section)
    
    # Save the samples metadata
    samples_metadata = {
        'book_id': book_info['id'],
        'title': title,
        'author': author,
        'level': level,
        'samples': samples
    }
    
    with open(output_dir / "samples.json", 'w', encoding='utf-8') as f:
        json.dump(samples_metadata, f, indent=2)
    
    logger.info(f"Created {len(samples)} reading samples for {title}")

def remove_gutenberg_header_footer(text):
    """
    Remove Project Gutenberg header and footer from the text.
    
    Args:
        text (str): The text to process
        
    Returns:
        str: Text with header and footer removed
    """
    # Common start/end markers for Project Gutenberg texts
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
        "The Project Gutenberg Etext of",
        "The Project Gutenberg EBook of"
    ]
    
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg's",
        "End of the Project Gutenberg EBook"
    ]
    
    # Find the start of the actual text
    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Move to the end of the line
            pos = text.find('\n', pos)
            if pos != -1:
                start_pos = pos + 1
                break
    
    # Find the end of the actual text
    end_pos = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break
    
    # Extract the text between markers
    if start_pos < end_pos:
        return text[start_pos:end_pos].strip()
    else:
        # If markers not found, return the original text
        return text