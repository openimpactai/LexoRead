#!/usr/bin/env python3
"""
User Profiling Model for LexoRead

This module provides models to track and learn user-specific reading patterns,
preferences, and difficulties to personalize the reading experience.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from pathlib import Path
import logging
import datetime
import uuid
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("user_profile_model")

class UserReadingData:
    """
    Class to store and process user reading data.
    """
    
    def __init__(self):
        """Initialize user reading data storage."""
        self.sessions = []
        self.reading_speeds = []
        self.comprehension_scores = []
        self.text_difficulties = []
        self.adaptation_preferences = {}
        self.difficult_words = set()
        
    def add_session(self, session_data):
        """
        Add a reading session data point.
        
        Args:
            session_data (dict): Reading session data
        """
        self.sessions.append(session_data)
        
        # Extract metrics
        if "reading_speed" in session_data:
            self.reading_speeds.append(session_data["reading_speed"])
            
        if "comprehension_score" in session_data:
            self.comprehension_scores.append(session_data["comprehension_score"])
            
        if "text_difficulty" in session_data:
            self.text_difficulties.append(session_data["text_difficulty"])
            
        # Update adaptation preferences if provided
        if "adaptations_used" in session_data:
            for adaptation, value in session_data["adaptations_used"].items():
                if adaptation not in self.adaptation_preferences:
                    self.adaptation_preferences[adaptation] = []
                self.adaptation_preferences[adaptation].append(value)
        
        # Update difficult words
        if "difficult_words" in session_data:
            self.difficult_words.update(session_data["difficult_words"])
    
    def get_average_reading_speed(self):
        """
        Get average reading speed (words per minute).
        
        Returns:
            float: Average reading speed
        """
        if not self.reading_speeds:
            return None
        return sum(self.reading_speeds) / len(self.reading_speeds)
    
    def get_average_comprehension(self):
        """
        Get average comprehension score (0-1).
        
        Returns:
            float: Average comprehension score
        """
        if not self.comprehension_scores:
            return None
        return sum(self.comprehension_scores) / len(self.comprehension_scores)
    
    def get_preferred_adaptations(self):
        """
        Get user's preferred adaptations.
        
        Returns:
            dict: Preferred adaptations
        """
        preferred = {}
        
        for adaptation, values in self.adaptation_preferences.items():
            if len(values) > 0:
                if isinstance(values[0], bool):
                    # Boolean preference (enabled/disabled)
                    preferred[adaptation] = sum(values) / len(values) > 0.5
                else:
                    # Numeric preference (average value)
                    preferred[adaptation] = sum(values) / len(values)
        
        return preferred
    
    def get_reading_difficulty_trend(self):
        """
        Get trend in reading difficulty over time.
        
        Returns:
            float: Trend coefficient
        """
        if len(self.text_difficulties) < 2:
            return 0.0
            
        # Simple linear regression
        x = np.arange(len(self.text_difficulties))
        y = np.array(self.text_difficulties)
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        return slope
    
    def to_dict(self):
        """
        Convert user reading data to dictionary.
        
        Returns:
            dict: User reading data
        """
        return {
            "sessions": self.sessions,
            "reading_speeds": self.reading_speeds,
            "comprehension_scores": self.comprehension_scores,
            "text_difficulties": self.text_difficulties,
            "adaptation_preferences": self.adaptation_preferences,
            "difficult_words": list(self.difficult_words)
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create user reading data from dictionary.
        
        Args:
            data (dict): User reading data dictionary
            
        Returns:
            UserReadingData: User reading data object
        """
        user_data = cls()
        
        user_data.sessions = data.get("sessions", [])
        user_data.reading_speeds = data.get("reading_speeds", [])
        user_data.comprehension_scores = data.get("comprehension_scores", [])
        user_data.text_difficulties = data.get("text_difficulties", [])
        user_data.adaptation_preferences = data.get("adaptation_preferences", {})
        user_data.difficult_words = set(data.get("difficult_words", []))
        
        return user_data

class UserReadingProfile:
    """
    User reading profile model.
    
    This model analyzes user reading patterns to generate a profile
    that can be used to personalize the reading experience.
    """
    
    def __init__(self, reading_data=None):
        """
        Initialize user reading profile.
        
        Args:
            reading_data (UserReadingData, optional): User reading data
        """
        self.reading_data = reading_data or UserReadingData()
        
        # Reading metrics
        self.avg_reading_speed = None
        self.avg_comprehension = None
        self.reading_level = None
        self.difficulty_trend = None
        
        # Adaptation preferences
        self.preferred_adaptations = {}
        
        # Difficulty patterns
        self.difficult_word_patterns = []
        self.difficult_phonetic_patterns = []
        
        # Update profile if data is provided
        if reading_data:
            self.update_profile()
    
    def update_profile(self):
        """Update profile based on reading data."""
        # Update reading metrics
        self.avg_reading_speed = self.reading_data.get_average_reading_speed()
        self.avg_comprehension = self.reading_data.get_average_comprehension()
        self.difficulty_trend = self.reading_data.get_reading_difficulty_trend()
        
        # Update adaptation preferences
        self.preferred_adaptations = self.reading_data.get_preferred_adaptations()
        
        # Analyze difficult words
        self._analyze_difficult_words()
    
    def _analyze_difficult_words(self):
        """Analyze difficult words to identify patterns."""
        if len(self.reading_data.difficult_words) < 5:
            return
            
        # Convert to list for analysis
        difficult_words = list(self.reading_data.difficult_words)
        
        # Extract character n-grams
        ngrams = []
        for word in difficult_words:
            # Add character bigrams and trigrams
            word = word.lower()
            for i in range(len(word) - 1):
                ngrams.append(word[i:i+2])
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])
        
        # Count frequency of n-grams
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Sort by frequency
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Keep top patterns
        self.difficult_word_patterns = [ngram for ngram, count in sorted_ngrams[:10]]
        
        # Identify phonetic patterns (simplified)
        phonetic_patterns = [
            ('b', 'd'),  # Commonly confused
            ('p', 'q'),  # Commonly confused
            ('ou', 'uo'),  # Reversals
            ('ie', 'ei'),  # Reversals
            ('th', 'ht'),  # Digraphs
            ('ch', 'hc'),  # Digraphs
            ('sh', 'hs')   # Digraphs
        ]
        
        pattern_counts = {}
        for pattern, reversed_pattern in phonetic_patterns:
            count = 0
            for word in difficult_words:
                if pattern in word.lower():
                    count += 1
            
            if count > 0:
                pattern_counts[pattern] = count
        
        # Sort by frequency
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Keep top patterns
        self.difficult_phonetic_patterns = [pattern for pattern, count in sorted_patterns]
    
    def get_reading_level(self):
        """
        Estimate user's comfortable reading level.
        
        Returns:
            float: Estimated reading level
        """
        if not self.reading_data.sessions:
            return None
            
        # Find sessions with good comprehension
        good_sessions = [s for s in self.reading_data.sessions 
                        if s.get("comprehension_score", 0) > 0.7]
        
        if not good_sessions:
            return None
        
        # Average difficulty of well-comprehended texts
        difficulties = [s.get("text_difficulty", 0) for s in good_sessions]
        return sum(difficulties) / len(difficulties)
    
    def recommend_adaptations(self, text_difficulty=None):
        """
        Recommend text adaptations based on user profile.
        
        Args:
            text_difficulty (float, optional): Difficulty of text to adapt
            
        Returns:
            dict: Recommended adaptations
        """
        # Start with user's preferred adaptations
        adaptations = self.preferred_adaptations.copy()
        
        # If text difficulty is provided, adjust adaptations accordingly
        if text_difficulty is not None:
            reading_level = self.get_reading_level()
            
            if reading_level is not None:
                difficulty_difference = text_difficulty - reading_level
                
                # Increase adaptations if text is more difficult than user's level
                if difficulty_difference > 0:
                    # Adjust font size
                    if "font_size" in adaptations:
                        adaptations["font_size"] = min(2.0, adaptations["font_size"] * (1 + 0.1 * difficulty_difference))
                    
                    # Adjust line spacing
                    if "line_spacing" in adaptations:
                        adaptations["line_spacing"] = min(3.0, adaptations["line_spacing"] * (1 + 0.1 * difficulty_difference))
                    
                    # Enable additional aids for difficult text
                    adaptations["highlight_complex_words"] = True
                    adaptations["show_syllable_breaks"] = True
        
        return adaptations
    
    def identify_challenging_words(self, text):
        """
        Identify potentially challenging words in text based on user profile.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: Potentially challenging words
        """
        if not self.difficult_word_patterns:
            return []
        
        challenging_words = []
        
        # Split text into words
        words = text.lower().split()
        
        for word in words:
            # Check if word contains any difficult patterns
            for pattern in self.difficult_word_patterns:
                if pattern in word:
                    challenging_words.append(word)
                    break
            
            # Check if word contains any difficult phonetic patterns
            if word not in challenging_words:
                for pattern in self.difficult_phonetic_patterns:
                    if pattern in word:
                        challenging_words.append(word)
                        break
        
        return challenging_words
    
    def to_dict(self):
        """
        Convert profile to dictionary.
        
        Returns:
            dict: Profile dictionary
        """
        return {
            "avg_reading_speed": self.avg_reading_speed,
            "avg_comprehension": self.avg_comprehension,
            "reading_level": self.reading_level,
            "difficulty_trend": self.difficulty_trend,
            "preferred_adaptations": self.preferred_adaptations,
            "difficult_word_patterns": self.difficult_word_patterns,
            "difficult_phonetic_patterns": self.difficult_phonetic_patterns
        }
    
    @classmethod
    def from_dict(cls, data, reading_data=None):
        """
        Create profile from dictionary.
        
        Args:
            data (dict): Profile dictionary
            reading_data (UserReadingData, optional): User reading data
            
        Returns:
            UserReadingProfile: User reading profile
        """
        profile = cls(reading_data)
        
        profile.avg_reading_speed = data.get("avg_reading_speed")
        profile.avg_comprehension = data.get("avg_comprehension")
        profile.reading_level = data.get("reading_level")
        profile.difficulty_trend = data.get("difficulty_trend")
        profile.preferred_adaptations = data.get("preferred_adaptations", {})
        profile.difficult_word_patterns = data.get("difficult_word_patterns", [])
        profile.difficult_phonetic_patterns = data.get("difficult_phonetic_patterns", [])
        
        return profile

class UserModelNN(nn.Module):
    """
    Neural network model for predicting optimal text adaptations.
    """
    
    def __init__(self, input_dim=20, hidden_dim=50, output_dim=10):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
        """
        super(UserModelNN, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

class UserProfiler:
    """
    Main class for managing user profiles.
    
    This class provides methods to load, update, and utilize user profiles
    for personalizing the reading experience.
    """
    
    def __init__(self, user_id, storage_dir=None, device=None):
        """
        Initialize the user profiler.
        
        Args:
            user_id (str): User ID
            storage_dir (str, optional): Directory to store user data
            device (str, optional): Device to run models on ('cuda' or 'cpu')
        """
        self.user_id = user_id
        
        # Set storage directory
        if storage_dir is None:
            self.storage_dir = Path(__file__).parent / "user_data"
        else:
            self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize neural network
        self.model = UserModelNN()
        self.model.to(self.device)
        
        # Load user data and profile
        self.reading_data = self._load_user_data()
        self.profile = UserReadingProfile(self.reading_data)
        
        # Load configurations
        self.config = self._load_config()
    
    def _load_config(self):
        """
        Load configuration.
        
        Returns:
            dict: Configuration
        """
        config_path = Path(__file__).parent / 'config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "update_threshold": 5,
                "min_sessions_for_prediction": 10,
                "adaptation_ranges": {
                    "font_size": [0.8, 2.0],
                    "line_spacing": [1.0, 3.0],
                    "word_spacing": [1.0, 2.0],
                    "highlight_complex_words": [0, 1],
                    "show_syllable_breaks": [0, 1],
                    "use_dyslexic_font": [0, 1],
                    "text_to_speech": [0, 1],
                    "reading_guide": [0, 1]
                }
            }
    
    def _load_user_data(self):
        """
        Load user reading data from storage.
        
        Returns:
            UserReadingData: User reading data
        """
        data_path = self.storage_dir / f"{self.user_id}_data.json"
        
        if data_path.exists():
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                return UserReadingData.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading user data: {e}")
                return UserReadingData()
        else:
            logger.info(f"No existing data found for user {self.user_id}. Creating new profile.")
            return UserReadingData()
    
    def _save_user_data(self):
        """Save user reading data to storage."""
        data_path = self.storage_dir / f"{self.user_id}_data.json"
        
        try:
            with open(data_path, 'w') as f:
                json.dump(self.reading_data.to_dict(), f, indent=2)
            logger.info(f"User data saved for {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
    
    def update(self, text=None, reading_time=None, comprehension_score=None, 
               text_difficulty=None, adaptations_used=None, difficult_words=None):
        """
        Update user profile with new reading session data.
        
        Args:
            text (str, optional): Text that was read
            reading_time (float, optional): Time spent reading (seconds)
            comprehension_score (float, optional): Comprehension score (0-1)
            text_difficulty (float, optional): Text difficulty rating (0-100)
            adaptations_used (dict, optional): Adaptations used during reading
            difficult_words (list, optional): Words the user found difficult
            
        Returns:
            bool: True if profile was updated
        """
        # Create session data
        session_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": str(uuid.uuid4())
        }
        
        # Add provided data
        if text is not None:
            # Store word count instead of full text to save space
            word_count = len(text.split())
            session_data["word_count"] = word_count
            
            if reading_time is not None:
                # Calculate reading speed (words per minute)
                reading_speed = (word_count / reading_time) * 60
                session_data["reading_time"] = reading_time
                session_data["reading_speed"] = reading_speed
        
        if comprehension_score is not None:
            session_data["comprehension_score"] = comprehension_score
            
        if text_difficulty is not None:
            session_data["text_difficulty"] = text_difficulty
            
        if adaptations_used is not None:
            session_data["adaptations_used"] = adaptations_used
            
        if difficult_words is not None:
            session_data["difficult_words"] = difficult_words
        
        # Add session data
        self.reading_data.add_session(session_data)
        
        # Update profile
        self.profile.update_profile()
        
        # Save user data
        self._save_user_data()
        
        return True
    
    def get_recommended_adaptations(self, text=None, text_difficulty=None):
        """
        Get recommended adaptations for a text.
        
        Args:
            text (str, optional): Text to get adaptations for
            text_difficulty (float, optional): Text difficulty rating (0-100)
            
        Returns:
            dict: Recommended adaptations
        """
        # If we don't have enough sessions, use rule-based recommendations
        if len(self.reading_data.sessions) < self.config["min_sessions_for_prediction"]:
            return self.profile.recommend_adaptations(text_difficulty)
        
        # Else, use neural network predictions if we have text
        if text is not None:
            return self._predict_adaptations_nn(text, text_difficulty)
        else:
            return self.profile.recommend_adaptations(text_difficulty)
    
    def _predict_adaptations_nn(self, text, text_difficulty=None):
        """
        Predict optimal adaptations using neural network.
        
        Args:
            text (str): Text to get adaptations for
            text_difficulty (float, optional): Text difficulty rating
            
        Returns:
            dict: Predicted adaptations
        """
        # Extract features from text
        features = self._extract_features(text, text_difficulty)
        
        # Convert features to tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict adaptations
        with torch.no_grad():
            predictions = self.model(x).squeeze(0).cpu().numpy()
        
        # Convert predictions to adaptations dictionary
        adaptations = {}
        
        # Map prediction outputs to adaptation parameters
        adaptation_keys = list(self.config["adaptation_ranges"].keys())
        
        for i, key in enumerate(adaptation_keys):
            if i < len(predictions):
                # Scale prediction to adaptation range
                value_range = self.config["adaptation_ranges"][key]
                scaled_value = value_range[0] + predictions[i] * (value_range[1] - value_range[0])
                
                # Round boolean adaptations
                if value_range[1] - value_range[0] == 1:
                    adaptations[key] = round(scaled_value)
                else:
                    adaptations[key] = scaled_value
        
        return adaptations
    
    def _extract_features(self, text, text_difficulty=None):
        """
        Extract features for neural network prediction.
        
        Args:
            text (str): Text to extract features from
            text_difficulty (float, optional): Text difficulty rating
            
        Returns:
            list: Features
        """
        # Text features
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Count complex words
        complex_word_count = sum(1 for word in words if len(word) >= 7)
        complex_word_ratio = complex_word_count / len(words) if words else 0
        
        # Count words matching difficult patterns
        difficult_pattern_count = 0
        for word in words:
            for pattern in self.profile.difficult_word_patterns:
                if pattern in word.lower():
                    difficult_pattern_count += 1
                    break
        
        difficult_pattern_ratio = difficult_pattern_count / len(words) if words else 0
        
        # User profile features
        avg_reading_speed = self.profile.avg_reading_speed or 0
        avg_comprehension = self.profile.avg_comprehension or 0
        
        # Create feature vector
        features = [
            avg_word_length / 10,  # Normalize to ~0-1 range
            complex_word_ratio,
            difficult_pattern_ratio,
            text_difficulty / 100 if text_difficulty is not None else 0.5,  # Normalize to 0-1
            avg_reading_speed / 300 if avg_reading_speed else 0.5,  # Normalize to ~0-1
            avg_comprehension if avg_comprehension else 0.5
        ]
        
        # Pad to expected input dimension
        while len(features) < 20:
            features.append(0.0)
        
        return features
    
    def get_profile_summary(self):
        """
        Get summary of user profile.
        
        Returns:
            dict: Profile summary
        """
        return {
            "user_id": self.user_id,
            "sessions_count": len(self.reading_data.sessions),
            "avg_reading_speed": self.profile.avg_reading_speed,
            "avg_comprehension": self.profile.avg_comprehension,
            "difficulty_trend": self.profile.difficulty_trend,
            "preferred_adaptations": self.profile.preferred_adaptations,
            "difficult_patterns": self.profile.difficult_word_patterns[:5]
        }
    
    def identify_text_challenges(self, text):
        """
        Identify aspects of text that might be challenging for the user.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Text challenges
        """
        # Use the profile to identify challenging words
        challenging_words = self.profile.identify_challenging_words(text)
        
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Identify long sentences
        long_sentences = []
        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count > 20:  # Threshold for "long" sentences
                long_sentences.append(sentence)
        
        # Calculate overall challenge rating
        challenge_rating = min(100, (len(challenging_words) / len(text.split()) * 100) + 
                              (len(long_sentences) / len(sentences) * 50 if sentences else 0))
        
        return {
            "challenging_words": challenging_words,
            "long_sentences": long_sentences,
            "challenge_rating": challenge_rating
        }
    
    def train_model(self, epochs=100):
        """
        Train the neural network model on user data.
        
        Args:
            epochs (int): Number of training epochs
            
        Returns:
            float: Final loss
        """
        # Check if we have enough sessions
        if len(self.reading_data.sessions) < self.config["min_sessions_for_prediction"]:
            logger.warning("Not enough sessions to train model")
            return None
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X is None or y is None:
            logger.warning("Could not prepare training data")
            return None
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Train
        self.model.train()
        final_loss = None
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        return final_loss
    
    def _prepare_training_data(self):
        """
        Prepare training data from user reading sessions.
        
        Returns:
            tuple: (X, y) training data
        """
        X = []
        y = []
        
        for session in self.reading_data.sessions:
            # Skip sessions without necessary data
            if not all(k in session for k in ["word_count", "reading_time", "comprehension_score", "adaptations_used"]):
                continue
            
            # Extract features
            word_count = session["word_count"]
            reading_time = session["reading_time"]
            reading_speed = (word_count / reading_time) * 60
            comprehension_score = session["comprehension_score"]
            text_difficulty = session.get("text_difficulty", 50)  # Default to middle difficulty
            
            # Create input features (simplified for example)
            features = [
                word_count / 1000,  # Normalize
                reading_speed / 300,  # Normalize
                comprehension_score,
                text_difficulty / 100  # Normalize
            ]
            
            # Pad to expected input dimension
            while len(features) < 20:
                features.append(0.0)
            
            # Extract adaptation values
            adaptation_keys = list(self.config["adaptation_ranges"].keys())
            adaptation_values = []
            
            for key in adaptation_keys:
                if key in session["adaptations_used"]:
                    value = session["adaptations_used"][key]
                    value_range = self.config["adaptation_ranges"][key]
                    
                    # Normalize to 0-1 range
                    normalized_value = (value - value_range[0]) / (value_range[1] - value_range[0])
                    adaptation_values.append(normalized_value)
                else:
                    # Default value
                    adaptation_values.append(0.5)
            
            # Add to training data
            X.append(features)
            y.append(adaptation_values)
        
        if not X or not y:
            return None, None
        
        return X, y

if __name__ == "__main__":
    # Simple demo
    profiler = UserProfiler(user_id="demo_user")
    
    # Add some sample reading sessions
    sample_texts = [
        "The cat sat on the mat. The dog ran fast. I like to play.",
        "The quick brown fox jumps over the lazy dog. The weather is nice today.",
        "The European economy has been experiencing significant challenges in recent years.",
        "The intricate interplay between quantum mechanics and general relativity presents a formidable challenge."
    ]
    
    sample_difficulties = [20, 40, 70, 90]
    
    # Simulate reading sessions
    for i, (text, difficulty) in enumerate(zip(sample_texts, sample_difficulties)):
        # Simulate reading time based on text length and difficulty
        word_count = len(text.split())
        reading_time = word_count * (0.5 + difficulty / 100)  # Harder texts take longer to read
        
        # Simulate comprehension score based on difficulty
        comprehension = max(0.4, 1.0 - (difficulty / 100) * 0.5)  # Harder texts have lower comprehension
        
        # Simulate adaptations used
        adaptations = {
            "font_size": 1.0 + (difficulty / 100) * 0.5,  # Increase font size for harder texts
            "line_spacing": 1.5,
            "word_spacing": 1.0 + (difficulty / 100) * 0.3,
            "highlight_complex_words": difficulty > 50,
            "use_dyslexic_font": True
        }
        
        # Update profile
        profiler.update(
            text=text,
            reading_time=reading_time,
            comprehension_score=comprehension,
            text_difficulty=difficulty,
            adaptations_used=adaptations,
            difficult_words=["intricate", "formidable", "quantum", "relativity"] if i == 3 else None
        )
    
    # Print profile summary
    print("User Profile Summary:")
    summary = profiler.get_profile_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get recommended adaptations for a new text
    new_text = "The complexity of neural networks has increased dramatically in recent years."
    adaptations = profiler.get_recommended_adaptations(new_text, text_difficulty=65)
    
    print("\nRecommended Adaptations:")
    for key, value in adaptations.items():
        print(f"  {key}: {value}")
    
    # Identify challenges in text
    challenges = profiler.identify_text_challenges(new_text)
    
    print("\nText Challenges:")
    print(f"  Challenge Rating: {challenges['challenge_rating']:.1f}/100")
    print(f"  Challenging Words: {challenges['challenging_words']}")
    print(f"  Long Sentences: {len(challenges['long_sentences'])}")
