#!/usr/bin/env python3
"""
Text preprocessing utilities for the LexoRead project.
This module contains functions for preprocessing text data for dyslexia-related models.
"""

import re
import json
import logging
import unicodedata
import nltk
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("text_preprocessing")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer models")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords")
    nltk.download('stopwords')

def load_config():
    """
    Load preprocessing configuration.
    
    Returns:
        dict: Preprocessing configuration
    """
    config_path = Path(__file__).parent.parent / 'config' / 'dataset_config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('preprocessing', {}).get('text', {})
    except Exception as e:
        logger.warning(f"Failed to load text preprocessing config: {e}")
        return {}

def clean_text(text, config=None):
    """
    Clean and normalize text.
    
    Args:
        text (str): Input text
        config (dict): Preprocessing configuration
        
    Returns:
        str: Cleaned text
    """
    if config is None:
        config = load_config()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags if specified
    if config.get('strip_html', True):
        text = re.sub(r'<.*?>', '', text)
    
    # Convert to lowercase if specified
    if config.get('lowercase', True):
        text = text.lower()
    
    # Remove punctuation if specified
    if config.get('remove_punctuation', False):
        text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip()

def tokenize_sentences(text):
    """
    Split text into sentences.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
    """
    return sent_tokenize(text)

def tokenize_words(text):
    """
    Split text into words.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of words
    """
    return word_tokenize(text)

def remove_stopwords(words, language='english'):
    """
    Remove stopwords from a list of words.
    
    Args:
        words (list): List of words
        language (str): Language for stopwords
        
    Returns:
        list: List of words with stopwords removed
    """
    stop_words = set(stopwords.words(language))
    return [word for word in words if word.lower() not in stop_words]

def calculate_readability_metrics(text):
    """
    Calculate readability metrics for the text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of readability metrics
    """
    # Tokenize the text
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    
    # Basic metrics
    num_sentences = len(sentences)
    num_words = len(words)
    num_chars = len(text)
    num_syllables = estimate_syllables(text)
    
    # Average metrics
    if num_sentences > 0:
        avg_words_per_sentence = num_words / num_sentences
    else:
        avg_words_per_sentence = 0
    
    if num_words > 0:
        avg_chars_per_word = num_chars / num_words
        avg_syllables_per_word = num_syllables / num_words
    else:
        avg_chars_per_word = 0
        avg_syllables_per_word = 0
    
    # Flesch Reading Ease score
    if num_words > 0 and num_sentences > 0:
        flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    else:
        flesch_reading_ease = 0
    
    # Flesch-Kincaid Grade Level
    if num_words > 0 and num_sentences > 0:
        flesch_kincaid_grade = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
    else:
        flesch_kincaid_grade = 0
    
    return {
        'num_sentences': num_sentences,
        'num_words': num_words,
        'num_chars': num_chars,
        'num_syllables': num_syllables,
        'avg_words_per_sentence': avg_words_per_sentence,
        'avg_chars_per_word': avg_chars_per_word,
        'avg_syllables_per_word': avg_syllables_per_word,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade
    }

def estimate_syllables(text):
    """
    Estimate the number of syllables in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Estimated syllable count
    """
    # English syllable estimation - this is a simplified approach
    vowels = "aeiouy"
    words = tokenize_words(text.lower())
    count = 0
    
    for word in words:
        word_count = 0
        if len(word) <= 3:
            word_count = 1
        else:
            # Count vowel groups
            prev_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    word_count += 1
                prev_is_vowel = is_vowel
            
            # Adjust for common patterns
            if word.endswith('e'):
                word_count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                word_count += 1
            if word_count == 0:
                word_count = 1
        
        count += word_count
    
    return count

def find_difficult_words(text, difficulty_threshold=8):
    """
    Find words that might be difficult for dyslexic readers.
    
    Args:
        text (str): Input text
        difficulty_threshold (int): Character length threshold for difficult words
        
    Returns:
        list: List of difficult words
    """
    words = tokenize_words(text)
    difficult_words = []
    
    for word in words:
        # Skip short words
        if len(word) < difficulty_threshold:
            continue
        
        # Skip non-alphabetic words
        if not word.isalpha():
            continue
        
        difficult_words.append(word)
    
    return difficult_words

def preprocess_text_file(input_path, output_path, config=None):
    """
    Preprocess a text file and save the result.
    
    Args:
        input_path (str): Path to input file
        output_path (str): Path to output file
        config (dict): Preprocessing configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Clean the text
        cleaned_text = clean_text(text, config)
        
        # Calculate readability metrics
        metrics = calculate_readability_metrics(cleaned_text)
        
        # Find difficult words
        difficult_words = find_difficult_words(cleaned_text)
        
        # Create output with metadata
        output = {
            'original_path': str(input_path),
            'cleaned_text': cleaned_text,
            'metrics': metrics,
            'difficult_words': difficult_words
        }
        
        # Save to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Preprocessed {input_path} -> {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error preprocessing {input_path}: {e}")
        return False

def preprocess_directory(input_dir, output_dir, recursive=True, config=None):
    """
    Preprocess all text files in a directory.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        recursive (bool): Whether to process subdirectories
        config (dict): Preprocessing configuration
        
    Returns:
        int: Number of files processed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files
    if recursive:
        files = list(input_dir.glob('**/*.txt'))
    else:
        files = list(input_dir.glob('*.txt'))
    
    processed_count = 0
    
    for file_path in files:
        # Determine relative path to maintain directory structure
        rel_path = file_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.json')
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the file
        if preprocess_text_file(file_path, output_path, config):
            processed_count += 1
    
    logger.info(f"Processed {processed_count} text files")
    return processed_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess text files for LexoRead")
    parser.add_argument('input', help="Input file or directory path")
    parser.add_argument('output', help="Output file or directory path")
    parser.add_argument('--recursive', action='store_true', help="Process subdirectories")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        preprocess_directory(input_path, output_path, args.recursive)
    else:
        preprocess_text_file(input_path, output_path)
