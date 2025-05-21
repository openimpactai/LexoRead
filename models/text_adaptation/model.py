#!/usr/bin/env python3
"""
Text Adaptation Model for LexoRead

This module provides a model to adapt text for easier reading by individuals with dyslexia.
The model can adjust text formatting, simplify complex words, and modify sentence structures.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("text_adaptation_model")

class DyslexiaTextAdapterModel(nn.Module):
    """
    Neural model for adapting text to be more dyslexia-friendly.
    
    This model uses a pre-trained BERT encoder followed by task-specific
    layers for text adaptation tasks.
    """
    
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=3):
        """
        Initialize the model.
        
        Args:
            bert_model_name (str): Name of the pre-trained BERT model to use
            num_labels (int): Number of adaptation task labels
        """
        super(DyslexiaTextAdapterModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Task-specific layers
        self.word_simplification = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.sentence_complexity = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.adaptation_score = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            
        Returns:
            dict: Model outputs including word simplifications, sentence complexity, and adaptation score
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the hidden states
        sequence_output = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # Shape: [batch_size, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Task 1: Word simplification (vocabulary replacement)
        word_logits = self.word_simplification(sequence_output)  # Shape: [batch_size, seq_len, vocab_size]
        
        # Task 2: Sentence complexity classification
        sentence_logits = self.sentence_complexity(pooled_output)  # Shape: [batch_size, num_labels]
        
        # Task 3: Overall adaptation score
        adaptation_score = self.adaptation_score(pooled_output).squeeze(-1)  # Shape: [batch_size]
        
        return {
            "word_logits": word_logits,
            "sentence_logits": sentence_logits,
            "adaptation_score": adaptation_score
        }

class DyslexiaTextAdapter:
    """
    Main class for adapting text for dyslexic readers.
    
    This class provides methods to load the model, adapt text,
    and apply various text transformation strategies.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the text adapter.
        
        Args:
            model_path (str, optional): Path to the pre-trained model
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize model
        self.model = DyslexiaTextAdapterModel()
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model path provided or model not found. Using untrained model.")
            
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Load configurations
        self.config = self._load_config()
        
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
                "simplification_threshold": 0.7,
                "complex_word_length": 8,
                "max_sentence_length": 20,
                "adaptation_strategies": {
                    "font_size": 1.2,
                    "line_spacing": 1.5,
                    "word_spacing": 1.2,
                    "use_dyslexic_font": True,
                    "highlight_complex_words": True
                }
            }
    
    def adapt(self, text, user_profile=None, strategy=None):
        """
        Adapt text for dyslexic readers.
        
        Args:
            text (str): Text to adapt
            user_profile (dict, optional): User-specific profile for customization
            strategy (str, optional): Specific adaptation strategy to apply
            
        Returns:
            dict: Adapted text with formatting instructions
        """
        # Apply preprocessing
        text = self._preprocess_text(text)
        
        # Apply model-based adaptations
        adapted_text, meta_info = self._apply_model_adaptations(text)
        
        # Apply rule-based adaptations
        adapted_text = self._apply_rule_based_adaptations(adapted_text)
        
        # Apply user-specific adaptations if profile is provided
        if user_profile:
            adapted_text = self._apply_user_adaptations(adapted_text, user_profile)
        
        # Apply specific strategy if provided
        if strategy:
            adapted_text = self._apply_strategy(adapted_text, strategy)
        
        # Prepare the output
        output = {
            "original_text": text,
            "adapted_text": adapted_text,
            "meta_info": meta_info,
            "formatting": self._get_formatting_instructions(text, adapted_text, user_profile)
        }
        
        return output
    
    def _preprocess_text(self, text):
        """
        Preprocess text before adaptation.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Simple preprocessing for now
        text = text.strip()
        return text
    
    def _apply_model_adaptations(self, text):
        """
        Apply model-based adaptations to the text.
        
        Args:
            text (str): Text to adapt
            
        Returns:
            tuple: (adapted_text, meta_info)
        """
        # Tokenize the text
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Process word simplifications
        word_logits = outputs["word_logits"][0].cpu().numpy()
        input_ids = tokens["input_ids"][0].cpu().numpy()
        
        # Generate replacement tokens
        simplifications = []
        for i, logits in enumerate(word_logits):
            original_token_id = input_ids[i]
            if i == 0 or i >= len(input_ids) - 1:  # Skip [CLS] and [SEP]
                simplifications.append(original_token_id)
                continue
                
            # Get top predictions
            top_indices = np.argsort(logits)[-5:]  # Get top 5 alternatives
            if top_indices[0] != original_token_id and logits[top_indices[0]] > self.config["simplification_threshold"]:
                simplifications.append(top_indices[0])
            else:
                simplifications.append(original_token_id)
        
        # Convert back to text
        adapted_tokens = self.tokenizer.convert_ids_to_tokens(simplifications)
        adapted_text = self.tokenizer.convert_tokens_to_string(adapted_tokens)
        
        # Extract meta information
        sentence_logits = outputs["sentence_logits"][0].cpu().numpy()
        adaptation_score = outputs["adaptation_score"].item()
        
        # Determine sentence complexity
        sentence_complexity = np.argmax(sentence_logits)  # 0: simple, 1: medium, 2: complex
        
        meta_info = {
            "adaptation_score": adaptation_score,
            "sentence_complexity": int(sentence_complexity),
            "simplification_confidence": float(np.mean(np.max(word_logits, axis=1)))
        }
        
        return adapted_text, meta_info
    
    def _apply_rule_based_adaptations(self, text):
        """
        Apply rule-based adaptations to the text.
        
        Args:
            text (str): Text to adapt
            
        Returns:
            str: Adapted text
        """
        # Split into sentences
        sentences = text.split('.')
        adapted_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Simplify long sentences
            words = sentence.split()
            if len(words) > self.config["max_sentence_length"]:
                # Split long sentences
                midpoint = len(words) // 2
                # Find a good breaking point around the midpoint
                for i in range(midpoint, midpoint - 5, -1):
                    if i > 0 and words[i].lower() in ['and', 'but', 'or', 'so', 'then']:
                        adapted_sentences.append(' '.join(words[:i]) + '.')
                        adapted_sentences.append(' '.join(words[i:]) + '.')
                        break
                else:
                    adapted_sentences.append(' '.join(words) + '.')
            else:
                adapted_sentences.append(sentence + '.')
        
        return ' '.join(adapted_sentences)
    
    def _apply_user_adaptations(self, text, user_profile):
        """
        Apply user-specific adaptations.
        
        Args:
            text (str): Text to adapt
            user_profile (dict): User profile
            
        Returns:
            str: Adapted text
        """
        # This would be implemented based on user-specific needs
        # For now, just return the text
        return text
    
    def _apply_strategy(self, text, strategy):
        """
        Apply a specific adaptation strategy.
        
        Args:
            text (str): Text to adapt
            strategy (str): Strategy to apply
            
        Returns:
            str: Adapted text
        """
        if strategy == "simplify_vocabulary":
            # Implement vocabulary simplification
            pass
        elif strategy == "shorten_sentences":
            # Implement sentence shortening
            pass
        elif strategy == "add_explanations":
            # Implement adding explanations
            pass
        
        # For now, just return the text
        return text
    
    def _get_formatting_instructions(self, original_text, adapted_text, user_profile=None):
        """
        Get formatting instructions for the adapted text.
        
        Args:
            original_text (str): Original text
            adapted_text (str): Adapted text
            user_profile (dict, optional): User profile
            
        Returns:
            dict: Formatting instructions
        """
        # Default formatting based on configuration
        formatting = self.config["adaptation_strategies"].copy()
        
        # Adjust based on user profile if available
        if user_profile and "preferences" in user_profile:
            for key, value in user_profile["preferences"].items():
                if key in formatting:
                    formatting[key] = value
        
        # Identify complex words to highlight
        if formatting["highlight_complex_words"]:
            complex_words = []
            for word in adapted_text.split():
                if len(word) >= self.config["complex_word_length"]:
                    complex_words.append(word)
            formatting["complex_words"] = complex_words
        
        return formatting
    
    def get_readability_score(self, text):
        """
        Calculate readability score for a text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Readability score
        """
        # Tokenize the text
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Get adaptation score
        adaptation_score = outputs["adaptation_score"].item()
        
        # Scale to 0-100 range
        readability_score = max(0, min(100, (1 - adaptation_score) * 100))
        
        return readability_score

if __name__ == "__main__":
    # Simple demo
    adapter = DyslexiaTextAdapter()
    
    # Example text
    example_text = "The quick brown fox jumps over the lazy dog. This is a complex sentence with multisyllabic words that might be difficult for individuals with dyslexia to read efficiently."
    
    # Adapt the text
    result = adapter.adapt(example_text)
    
    # Print the result
    print("Original text:", result["original_text"])
    print("Adapted text:", result["adapted_text"])
    print("Meta info:", result["meta_info"])
    print("Formatting:", result["formatting"])
    
    # Calculate readability score
    score = adapter.get_readability_score(example_text)
    print(f"Readability score: {score:.2f}/100")
