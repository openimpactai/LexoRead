#!/usr/bin/env python3
"""
Text adaptation service for LexoRead API.

This service provides methods for adapting text for dyslexic readers.
It serves as a bridge between the API layer and the ML models.
"""

import logging
import sys
import os
from typing import Dict, List, Any, Optional
import uuid
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import text adaptation model
from models.text_adaptation.model import DyslexiaTextAdapter

# Import user profile model for personalization
from models.user_profile.model import UserProfiler

# Set up logging
logger = logging.getLogger("api.services.text")

class TextService:
    """
    Service for text adaptation operations.
    
    This service provides methods for adapting text for dyslexic readers
    using the DyslexiaTextAdapter model and personalizing adaptations
    using the UserProfiler model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the text service.
        
        Args:
            model_path (str, optional): Path to the text adaptation model weights
        """
        self.model_path = model_path
        
        try:
            # Initialize text adaptation model
            self.adapter = DyslexiaTextAdapter(model_path=model_path)
            logger.info(f"Initialized text adaptation model from {model_path}")
        except Exception as e:
            logger.error(f"Error initializing text adaptation model: {e}", exc_info=True)
            # In development mode, we'll use a mock adapter
            self.adapter = MockTextAdapter()
            logger.warning("Using mock text adapter")
    
    def adapt_text(self, text: str, user_id: Optional[str] = None, 
                  reading_level: Optional[int] = None, 
                  adaptations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapt text for easier reading by individuals with dyslexia.
        
        Args:
            text (str): Text to adapt
            user_id (str, optional): User ID for personalized adaptations
            reading_level (int, optional): Target reading level (0-5)
            adaptations (dict, optional): Specific adaptations to apply
            
        Returns:
            dict: Adapted text with formatting instructions
        """
        # Log request
        logger.info(f"Adapting text for user {user_id}, length: {len(text)}")
        
        # Get user profile if user_id is provided
        user_profile = None
        if user_id:
            try:
                profiler = UserProfiler(user_id=user_id)
                user_profile = profiler.get_profile_summary()
                logger.info(f"Found user profile for {user_id}")
            except Exception as e:
                logger.warning(f"Error loading user profile for {user_id}: {e}")
        
        # Apply adaptations
        try:
            # Use the model to adapt the text
            result = self.adapter.adapt(
                text=text,
                user_profile=user_profile,
                strategy=None  # We'll handle strategy based on adaptations
            )
            
            # Override with provided adaptations if any
            if adaptations:
                if "formatting" in result:
                    result["formatting"].update(adaptations)
                else:
                    result["formatting"] = adaptations
            
            # Update user profile if user_id is provided
            if user_id:
                try:
                    # Create session data
                    session_data = {
                        "text": text,
                        "adaptations_used": result.get("formatting", {}),
                        "text_difficulty": result.get("meta_info", {}).get("adaptation_score", 50)
                    }
                    
                    # Update user profile
                    profiler.update(**session_data)
                    logger.info(f"Updated user profile for {user_id}")
                except Exception as e:
                    logger.warning(f"Error updating user profile for {user_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adapting text: {e}", exc_info=True)
            raise
    
    def simplify_text(self, text: str, target_level: Optional[int] = None,
                     preserve_meaning: bool = True, simplify_vocabulary: bool = True,
                     simplify_grammar: bool = True) -> Dict[str, Any]:
        """
        Simplify text by reducing complexity of vocabulary and grammar.
        
        Args:
            text (str): Text to simplify
            target_level (int, optional): Target reading level (0-5)
            preserve_meaning (bool): Whether to prioritize preserving meaning
            simplify_vocabulary (bool): Whether to simplify complex vocabulary
            simplify_grammar (bool): Whether to simplify sentence structure
            
        Returns:
            dict: Simplified text with formatting instructions
        """
        # Log request
        logger.info(f"Simplifying text, length: {len(text)}")
        
        # Apply simplification
        try:
            # Create a strategy based on simplification parameters
            strategy = "simplify_vocabulary" if simplify_vocabulary else None
            if simplify_grammar and strategy:
                strategy += "_grammar"
            elif simplify_grammar:
                strategy = "simplify_grammar"
            
            # Use the model to simplify the text
            result = self.adapter.adapt(
                text=text,
                strategy=strategy
            )
            
            # Check if target level is provided
            if target_level is not None:
                # Get current readability score
                current_score = self.adapter.get_readability_score(text)
                
                # Get simplification level
                simplification_level = max(0, min(5, 5 - target_level))
                
                # Apply more aggressive simplification if needed
                if simplification_level > 0:
                    # Apply additional simplifications based on level
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error simplifying text: {e}", exc_info=True)
            raise
    
    def get_available_formats(self) -> List[Dict[str, Any]]:
        """
        Get available text formatting options.
        
        Returns:
            list: Available formatting options
        """
        # Return available formatting options
        return [
            {
                "id": "font_size",
                "name": "Font Size",
                "description": "Adjust the size of the text",
                "type": "float",
                "default_value": 1.0,
                "min_value": 0.8,
                "max_value": 2.0
            },
            {
                "id": "line_spacing",
                "name": "Line Spacing",
                "description": "Adjust the spacing between lines",
                "type": "float",
                "default_value": 1.5,
                "min_value": 1.0,
                "max_value": 3.0
            },
            {
                "id": "word_spacing",
                "name": "Word Spacing",
                "description": "Adjust the spacing between words",
                "type": "float",
                "default_value": 1.0,
                "min_value": 1.0,
                "max_value": 2.0
            },
            {
                "id": "use_dyslexic_font",
                "name": "Dyslexic Font",
                "description": "Use a font designed for dyslexic readers",
                "type": "boolean",
                "default_value": True
            },
            {
                "id": "highlight_complex_words",
                "name": "Highlight Complex Words",
                "description": "Highlight words that may be difficult to read",
                "type": "boolean",
                "default_value": True
            },
            {
                "id": "show_syllable_breaks",
                "name": "Show Syllable Breaks",
                "description": "Show syllable breaks in complex words",
                "type": "boolean",
                "default_value": False
            }
        ]

class MockTextAdapter:
    """
    Mock text adapter for development and testing.
    
    This class provides a simple implementation of the DyslexiaTextAdapter
    interface for use when the actual model is not available.
    """
    
    def adapt(self, text: str, user_profile=None, strategy=None) -> Dict[str, Any]:
        """
        Mock implementation of text adaptation.
        
        Args:
            text (str): Text to adapt
            user_profile (dict, optional): User profile for personalization
            strategy (str, optional): Adaptation strategy
            
        Returns:
            dict: Adapted text with formatting instructions
        """
        # Simple mock implementation
        return {
            "original_text": text,
            "adapted_text": text,  # No actual adaptation in the mock
            "meta_info": {
                "adaptation_score": 0.5,
                "sentence_complexity": 1,
                "simplification_confidence": 0.8
            },
            "formatting": {
                "font_size": 1.2,
                "line_spacing": 1.5,
                "word_spacing": 1.2,
                "use_dyslexic_font": True,
                "highlight_complex_words": True,
                "complex_words": ["complex", "vocabulary", "significantly"]
            }
        }
    
    def get_readability_score(self, text: str) -> float:
        """
        Mock implementation of readability scoring.
        
        Args:
            text (str): Text to score
            
        Returns:
            float: Readability score
        """
        # Simple mock implementation
        return 70.0  # Fixed score for mock
