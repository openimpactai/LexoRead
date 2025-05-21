#!/usr/bin/env python3
"""
Reading Level Assessment Model for LexoRead

This module provides models to assess the reading level and complexity
of text, with specific focus on analyzing text for dyslexic readers.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from pathlib import Path
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reading_level_model")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer models")
    nltk.download('punkt')

# Load CMU pronunciation dictionary if available
try:
    nltk.data.find('corpora/cmudict')
    cmu = cmudict.dict()
except LookupError:
    logger.info("Downloading CMU Pronunciation Dictionary")
    nltk.download('cmudict')
    cmu = cmudict.dict()
except:
    logger.warning("Failed to load CMU Dictionary. Syllable counting will use fallback method.")
    cmu = None

class ReadingLevelFeatureExtractor:
    """
    Extract features from text for reading level assessment.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_features(self, text):
        """
        Extract linguistic features from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of features
        """
        # Basic cleaning
        text = text.strip()
        
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Filter out punctuation
        words = [word for word in words if word.isalpha()]
        
        # Basic counts
        num_sentences = len(sentences)
        num_words = len(words)
        num_chars = sum(len(word) for word in words)
        num_syllables = self.count_syllables(words)
        
        # Calculate averages
        if num_words > 0:
            avg_word_length = num_chars / num_words
            avg_syllables_per_word = num_syllables / num_words
        else:
            avg_word_length = 0
            avg_syllables_per_word = 0
            
        if num_sentences > 0:
            avg_words_per_sentence = num_words / num_sentences
        else:
            avg_words_per_sentence = 0
        
        # Word frequency
        frequency_features = self.calculate_word_frequency(words)
        
        # Readability indices
        flesch_reading_ease = self.calculate_flesch_reading_ease(num_syllables, num_words, num_sentences)
        flesch_kincaid_grade = self.calculate_flesch_kincaid_grade(num_syllables, num_words, num_sentences)
        smog_index = self.calculate_smog_index(text)
        gunning_fog = self.calculate_gunning_fog(words, sentences)
        
        # Dyslexia-specific features
        complex_words = self.identify_complex_words(words)
        phonetic_features = self.extract_phonetic_features(words)
        
        # Combine all features
        features = {
            "num_sentences": num_sentences,
            "num_words": num_words,
            "num_chars": num_chars,
            "num_syllables": num_syllables,
            "avg_word_length": avg_word_length,
            "avg_syllables_per_word": avg_syllables_per_word,
            "avg_words_per_sentence": avg_words_per_sentence,
            "num_complex_words": len(complex_words),
            "ratio_complex_words": len(complex_words) / num_words if num_words > 0 else 0,
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "smog_index": smog_index,
            "gunning_fog": gunning_fog,
            **frequency_features,
            **phonetic_features
        }
        
        return features
    
    def count_syllables(self, words):
        """
        Count the number of syllables in a list of words.
        
        Args:
            words (list): List of words
            
        Returns:
            int: Total syllable count
        """
        total_syllables = 0
        
        for word in words:
            if cmu is not None and word.lower() in cmu:
                # Use CMU dictionary for precise syllable counting
                syllables = max([len([phoneme for phoneme in phones if phoneme[-1].isdigit()]) 
                               for phones in cmu[word.lower()]])
                total_syllables += syllables
            else:
                # Fallback method for unknown words
                total_syllables += self.count_syllables_fallback(word)
                
        return total_syllables
    
    def count_syllables_fallback(self, word):
        """
        Count syllables using a fallback method.
        
        Args:
            word (str): A word
            
        Returns:
            int: Syllable count
        """
        word = word.lower()
        
        # Remove non-alphanumeric characters
        word = re.sub(r'[^a-zA-Z]', '', word)
        
        # Handle special cases
        if len(word) <= 3:
            return 1
            
        # Remove trailing e
        if word.endswith('e'):
            word = word[:-1]
            
        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Handle edge cases
        if count == 0:
            count = 1
            
        return count
    
    def calculate_word_frequency(self, words):
        """
        Calculate word frequency statistics.
        
        Args:
            words (list): List of words
            
        Returns:
            dict: Word frequency features
        """
        if not words:
            return {
                "unique_words_ratio": 0,
                "rare_words_ratio": 0
            }
            
        # Count word frequencies
        word_count = Counter(words)
        unique_words = len(word_count)
        
        # Calculate unique words ratio
        unique_words_ratio = unique_words / len(words)
        
        # Calculate rare words (words that appear only once)
        rare_words = sum(1 for word, count in word_count.items() if count == 1)
        rare_words_ratio = rare_words / len(words)
        
        return {
            "unique_words_ratio": unique_words_ratio,
            "rare_words_ratio": rare_words_ratio
        }
    
    def calculate_flesch_reading_ease(self, num_syllables, num_words, num_sentences):
        """
        Calculate Flesch Reading Ease score.
        
        Args:
            num_syllables (int): Number of syllables
            num_words (int): Number of words
            num_sentences (int): Number of sentences
            
        Returns:
            float: Flesch Reading Ease score
        """
        if num_words == 0 or num_sentences == 0:
            return 0
            
        # Calculate statistics
        avg_sentence_length = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return score
    
    def calculate_flesch_kincaid_grade(self, num_syllables, num_words, num_sentences):
        """
        Calculate Flesch-Kincaid Grade Level.
        
        Args:
            num_syllables (int): Number of syllables
            num_words (int): Number of words
            num_sentences (int): Number of sentences
            
        Returns:
            float: Flesch-Kincaid Grade Level
        """
        if num_words == 0 or num_sentences == 0:
            return 0
            
        # Calculate statistics
        avg_sentence_length = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words
        
        # Flesch-Kincaid Grade Level formula
        grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        
        return grade
    
    def calculate_smog_index(self, text):
        """
        Calculate SMOG index.
        
        Args:
            text (str): Input text
            
        Returns:
            float: SMOG index
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 30:
            return 0  # SMOG requires at least 30 sentences
            
        # Count polysyllabic words (words with 3+ syllables)
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        
        polysyllable_count = 0
        for word in words:
            syllables = 0
            if cmu is not None and word.lower() in cmu:
                syllables = max([len([phoneme for phoneme in phones if phoneme[-1].isdigit()]) 
                              for phones in cmu[word.lower()]])
            else:
                syllables = self.count_syllables_fallback(word)
                
            if syllables >= 3:
                polysyllable_count += 1
                
        # SMOG formula
        smog = 1.043 * np.sqrt(polysyllable_count * (30 / len(sentences))) + 3.1291
        
        return smog
    
    def calculate_gunning_fog(self, words, sentences):
        """
        Calculate Gunning Fog index.
        
        Args:
            words (list): List of words
            sentences (list): List of sentences
            
        Returns:
            float: Gunning Fog index
        """
        if not words or not sentences:
            return 0
            
        # Count complex words (words with 3+ syllables)
        complex_word_count = 0
        for word in words:
            syllables = 0
            if cmu is not None and word.lower() in cmu:
                syllables = max([len([phoneme for phoneme in phones if phoneme[-1].isdigit()]) 
                              for phones in cmu[word.lower()]])
            else:
                syllables = self.count_syllables_fallback(word)
                
            if syllables >= 3:
                complex_word_count += 1
                
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Gunning Fog formula
        fog = 0.4 * (avg_sentence_length + 100 * (complex_word_count / len(words)))
        
        return fog
    
    def identify_complex_words(self, words):
        """
        Identify words that might be difficult for dyslexic readers.
        
        Args:
            words (list): List of words
            
        Returns:
            list: List of complex words
        """
        complex_words = []
        
        for word in words:
            word = word.lower()
            
            # Check word length (longer words tend to be more difficult)
            if len(word) >= 7:
                complex_words.append(word)
                continue
                
            # Check for difficult letter combinations
            difficult_patterns = [
                'ough', 'augh', 'tion', 'sion', 'cial', 'tial', 'ious', 'gn', 'mn',
                'sci', 'xc', 'gh', 'ph', 'rh', 'dge'
            ]
            
            for pattern in difficult_patterns:
                if pattern in word:
                    complex_words.append(word)
                    break
        
        return complex_words
    
    def extract_phonetic_features(self, words):
        """
        Extract phonetic features relevant to dyslexia.
        
        Args:
            words (list): List of words
            
        Returns:
            dict: Phonetic features
        """
        if not words:
            return {
                "consonant_cluster_ratio": 0,
                "similar_sounding_ratio": 0
            }
        
        # Count words with consonant clusters
        consonant_clusters = ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr',
                             'pl', 'pr', 'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st',
                             'str', 'sw', 'tr', 'tw', 'spl', 'spr', 'chr', 'sch']
        
        consonant_cluster_count = 0
        for word in words:
            for cluster in consonant_clusters:
                if cluster in word.lower():
                    consonant_cluster_count += 1
                    break
        
        # Count words with commonly confused sounds
        similar_sound_pairs = [
            ('b', 'd'), ('p', 'q'), ('m', 'n'), ('u', 'v'), ('f', 'v'),
            ('ch', 'sh'), ('c', 'k'), ('s', 'z'), ('th', 'f')
        ]
        
        similar_sounding_count = 0
        for word in words:
            word = word.lower()
            for pair in similar_sound_pairs:
                if pair[0] in word and pair[1] in word:
                    similar_sounding_count += 1
                    break
        
        return {
            "consonant_cluster_ratio": consonant_cluster_count / len(words),
            "similar_sounding_ratio": similar_sounding_count / len(words)
        }

class BERTReadingLevelModel(nn.Module):
    """
    BERT-based model for reading level prediction.
    """
    
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=6):
        """
        Initialize the model.
        
        Args:
            bert_model_name (str): Name of the pre-trained BERT model
            num_classes (int): Number of reading level classes to predict
        """
        super(BERTReadingLevelModel, self).__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            
        Returns:
            torch.Tensor: Logits for each reading level class
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classify
        logits = self.classifier(pooled_output)
        
        return logits

class ReadingLevelAssessor:
    """
    Main class for assessing text reading level.
    
    This class combines the feature-based and BERT-based approaches
    to provide comprehensive reading level assessment.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the reading level assessor.
        
        Args:
            model_path (str, optional): Path to the pre-trained model
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERTReadingLevelModel()
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model path provided or model not found. Using untrained model.")
            
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature extractor
        self.feature_extractor = ReadingLevelFeatureExtractor()
        
        # Load configurations
        self.config = self._load_config()
        
        # Reading level labels
        self.reading_levels = [
            "Early Elementary (Grades K-2)",
            "Elementary (Grades 3-5)",
            "Middle School (Grades 6-8)",
            "High School (Grades 9-12)",
            "College",
            "Advanced"
        ]
    
    def _load_config(self):
        """
        Load configuration for the model.
        
        Returns:
            dict: Configuration parameters
        """
        config_path = Path(__file__).parent / 'config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "readability_thresholds": {
                    "flesch_reading_ease": {
                        "very_easy": 90,
                        "easy": 80,
                        "fairly_easy": 70,
                        "standard": 60,
                        "fairly_difficult": 50,
                        "difficult": 30,
                        "very_difficult": 0
                    },
                    "flesch_kincaid_grade": {
                        "grade_1": 1,
                        "grade_2": 2,
                        "grade_3": 3,
                        "grade_4": 4,
                        "grade_5": 5,
                        "grade_6": 6,
                        "grade_7": 7,
                        "grade_8": 8,
                        "grade_9": 9,
                        "grade_10": 10,
                        "grade_11": 11,
                        "grade_12": 12,
                        "college": 13
                    }
                },
                "dyslexia_difficulty_factors": {
                    "complex_word_weight": 2.0,
                    "consonant_cluster_weight": 1.5,
                    "similar_sounding_weight": 1.5,
                    "sentence_length_weight": 1.0,
                    "unique_words_weight": 0.5
                }
            }
    
    def assess_text(self, text):
        """
        Assess the reading level of text.
        
        Args:
            text (str): Text to assess
            
        Returns:
            ReadingLevel: Reading level assessment result
        """
        # Check for empty text
        if not text or len(text.strip()) == 0:
            return ReadingLevel(
                level=0,
                grade=0,
                difficulty=0,
                readability_score=100,
                dyslexia_difficulty=0,
                features={}
            )
        
        # Extract features
        features = self.feature_extractor.extract_features(text)
        
        # Feature-based assessment
        feature_level = self.assess_using_features(features)
        
        # BERT-based assessment
        bert_level = self.assess_using_bert(text)
        
        # Combine assessments (weighted average)
        level = 0.7 * feature_level["level"] + 0.3 * bert_level["level"]
        
        # Ensure level is an integer
        level = int(round(level))
        
        # Calculate grade level
        grade = self.convert_level_to_grade(level)
        
        # Calculate difficulty rating (0-100, where 100 is most difficult)
        base_difficulty = min(100, max(0, 100 * level / 5))
        
        # Adjust difficulty based on dyslexia-specific factors
        dyslexia_difficulty = self.calculate_dyslexia_difficulty(features)
        
        # Overall difficulty (weighted average of base difficulty and dyslexia difficulty)
        difficulty = 0.6 * base_difficulty + 0.4 * dyslexia_difficulty
        
        # Calculate readability score (inverse of difficulty)
        readability_score = 100 - difficulty
        
        return ReadingLevel(
            level=level,
            grade=grade,
            difficulty=difficulty,
            readability_score=readability_score,
            dyslexia_difficulty=dyslexia_difficulty,
            features=features
        )
    
    def assess_using_features(self, features):
        """
        Assess reading level using extracted features.
        
        Args:
            features (dict): Extracted text features
            
        Returns:
            dict: Assessment results
        """
        # Get readability scores
        flesch_reading_ease = features["flesch_reading_ease"]
        flesch_kincaid_grade = features["flesch_kincaid_grade"]
        
        # Determine level based on Flesch-Kincaid Grade Level
        if flesch_kincaid_grade < 3:
            level = 0  # Early Elementary
        elif flesch_kincaid_grade < 6:
            level = 1  # Elementary
        elif flesch_kincaid_grade < 9:
            level = 2  # Middle School
        elif flesch_kincaid_grade < 13:
            level = 3  # High School
        elif flesch_kincaid_grade < 16:
            level = 4  # College
        else:
            level = 5  # Advanced
        
        # Adjust based on Flesch Reading Ease score
        if flesch_reading_ease > 90:
            level = max(0, level - 1)  # Very easy, decrease level
        elif flesch_reading_ease < 30:
            level = min(5, level + 1)  # Very difficult, increase level
        
        # Adjust based on other features
        if features["avg_words_per_sentence"] > 25:
            level = min(5, level + 1)  # Long sentences, increase level
        
        if features["ratio_complex_words"] > 0.2:
            level = min(5, level + 1)  # Many complex words, increase level
        
        return {
            "level": level
        }
    
    def assess_using_bert(self, text):
        """
        Assess reading level using BERT model.
        
        Args:
            text (str): Text to assess
            
        Returns:
            dict: Assessment results
        """
        # Tokenize text
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(**tokens)
        
        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Get probabilities for each level
        level_probabilities = probabilities[0].cpu().numpy()
        
        return {
            "level": predicted_class,
            "probabilities": level_probabilities
        }
    
    def convert_level_to_grade(self, level):
        """
        Convert reading level to grade level.
        
        Args:
            level (int): Reading level (0-5)
            
        Returns:
            float: Approximate US grade level
        """
        # Map reading levels to approximate grade levels
        grade_map = {
            0: 1.5,    # Early Elementary (K-2) -> Grade 1.5 (average of K-2)
            1: 4,      # Elementary (3-5) -> Grade 4
            2: 7,      # Middle School (6-8) -> Grade 7
            3: 10.5,   # High School (9-12) -> Grade 10.5
            4: 14,     # College -> Grade 14
            5: 16      # Advanced -> Grade 16+
        }
        
        return grade_map.get(level, 0)
    
    def calculate_dyslexia_difficulty(self, features):
        """
        Calculate difficulty rating specifically for dyslexic readers.
        
        Args:
            features (dict): Extracted text features
            
        Returns:
            float: Dyslexia difficulty score (0-100)
        """
        # Get weights from config
        weights = self.config["dyslexia_difficulty_factors"]
        
        # Calculate component scores (0-100 scale)
        complex_word_score = min(100, features["ratio_complex_words"] * 500)
        consonant_cluster_score = min(100, features["consonant_cluster_ratio"] * 300)
        similar_sounding_score = min(100, features["similar_sounding_ratio"] * 300)
        sentence_length_score = min(100, features["avg_words_per_sentence"] * 4)
        unique_words_score = min(100, features["unique_words_ratio"] * 100)
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        
        weighted_score = (
            complex_word_score * weights["complex_word_weight"] +
            consonant_cluster_score * weights["consonant_cluster_weight"] +
            similar_sounding_score * weights["similar_sounding_weight"] +
            sentence_length_score * weights["sentence_length_weight"] +
            unique_words_score * weights["unique_words_weight"]
        ) / total_weight
        
        return weighted_score
    
    def get_reading_level_label(self, level):
        """
        Get human-readable label for a reading level.
        
        Args:
            level (int): Reading level (0-5)
            
        Returns:
            str: Reading level label
        """
        if 0 <= level < len(self.reading_levels):
            return self.reading_levels[level]
        else:
            return "Unknown"
    
    def get_difficulty_explanation(self, assessment):
        """
        Get explanation of difficulty factors.
        
        Args:
            assessment (ReadingLevel): Reading level assessment
            
        Returns:
            dict: Explanation of difficulty factors
        """
        features = assessment.features
        
        explanation = {
            "complex_words": {
                "score": min(100, features["ratio_complex_words"] * 500),
                "description": "Text contains many complex words that may be challenging for dyslexic readers."
                if features["ratio_complex_words"] > 0.1 else
                "Text contains few complex words."
            },
            "sentence_structure": {
                "score": min(100, features["avg_words_per_sentence"] * 4),
                "description": "Sentences are long and may be difficult to follow."
                if features["avg_words_per_sentence"] > 20 else
                "Sentences are concise and easier to read."
            },
            "phonetic_difficulty": {
                "score": min(100, (features["consonant_cluster_ratio"] + features["similar_sounding_ratio"]) * 200),
                "description": "Text contains many words with challenging letter combinations."
                if (features["consonant_cluster_ratio"] + features["similar_sounding_ratio"]) > 0.2 else
                "Text has few challenging letter combinations."
            },
            "overall_readability": {
                "score": max(0, 100 - features["flesch_reading_ease"]),
                "description": "Overall text readability is challenging."
                if features["flesch_reading_ease"] < 60 else
                "Overall text readability is good."
            }
        }
        
        return explanation

class ReadingLevel:
    """
    Reading level assessment result.
    """
    
    def __init__(self, level, grade, difficulty, readability_score, dyslexia_difficulty, features):
        """
        Initialize reading level result.
        
        Args:
            level (int): Reading level (0-5)
            grade (float): Approximate US grade level
            difficulty (float): Overall difficulty rating (0-100)
            readability_score (float): Readability score (0-100)
            dyslexia_difficulty (float): Dyslexia-specific difficulty score (0-100)
            features (dict): Extracted text features
        """
        self.level = level
        self.grade = grade
        self.difficulty = difficulty
        self.readability_score = readability_score
        self.dyslexia_difficulty = dyslexia_difficulty
        self.features = features
    
    def get_summary(self):
        """
        Get a summary of the reading level assessment.
        
        Returns:
            dict: Summary of reading level assessment
        """
        return {
            "level": self.level,
            "grade": self.grade,
            "difficulty": self.difficulty,
            "readability_score": self.readability_score,
            "dyslexia_difficulty": self.dyslexia_difficulty,
            "flesch_reading_ease": self.features.get("flesch_reading_ease", 0),
            "flesch_kincaid_grade": self.features.get("flesch_kincaid_grade", 0)
        }
    
    def __str__(self):
        """String representation of reading level."""
        return f"Level: {self.level}, Grade: {self.grade:.1f}, Difficulty: {self.difficulty:.1f}/100"

if __name__ == "__main__":
    # Simple demo
    assessor = ReadingLevelAssessor()
    
    # Example texts of varying difficulty
    texts = [
        "The cat sat on the mat. The dog ran fast. I like to play.",  # Very simple
        "The quick brown fox jumps over the lazy dog. The weather is nice today. I enjoy reading books about animals.",  # Simple
        "The European economy has been experiencing significant challenges in recent years, including inflation, supply chain disruptions, and geopolitical tensions that affect trade relationships.",  # Moderate
        "The intricate interplay between quantum mechanics and general relativity presents a formidable challenge in theoretical physics, necessitating a comprehensive paradigm shift in our understanding of fundamental forces."  # Complex
    ]
    
    for i, text in enumerate(texts):
        assessment = assessor.assess_text(text)
        
        print(f"\nText {i+1}:")
        print(f"  {text}")
        print(f"  Assessment: {assessment}")
        print(f"  Reading Level: {assessor.get_reading_level_label(assessment.level)}")
        print(f"  Dyslexia Difficulty: {assessment.dyslexia_difficulty:.1f}/100")
        print(f"  Flesch Reading Ease: {assessment.features['flesch_reading_ease']:.1f}")
        print(f"  Flesch-Kincaid Grade Level: {assessment.features['flesch_kincaid_grade']:.1f}")
