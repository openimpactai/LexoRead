"""
Smart Text Summarization Model
Implements LLM-based text summarization with reading level adaptation
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for summarization model"""
    model_name: str = "facebook/bart-large-cnn"
    max_input_length: int = 1024
    min_summary_length: int = 50
    max_summary_length: int = 150
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reading_levels: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.reading_levels is None:
            self.reading_levels = {
                "beginner": {"min_length": 30, "max_length": 80, "complexity": 0.3},
                "intermediate": {"min_length": 50, "max_length": 120, "complexity": 0.6},
                "advanced": {"min_length": 80, "max_length": 200, "complexity": 0.9}
            }


class SmartSummarizationModel:
    """Smart text summarization with reading level adaptation"""
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self._load_models()
        self.keyword_extractor = KeywordExtractor()
        
    def _load_models(self):
        """Load summarization models"""
        try:
            # Primary summarization model (BART)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name
            ).to(self.config.device)
            
            # T5 for reading level adaptation
            self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                "t5-base"
            ).to(self.config.device)
            
            # Create pipeline for easier use
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.device == "cuda" else -1
            )
            
            logger.info("Summarization models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        reading_level: str = "intermediate",
        extract_keywords: bool = True
    ) -> Dict[str, any]:
        """
        Generate smart summary with reading level adaptation
        
        Args:
            text: Input text to summarize
            reading_level: Target reading level (beginner/intermediate/advanced)
            extract_keywords: Whether to extract key concepts
            
        Returns:
            Dictionary with summary, keywords, and metadata
        """
        # Validate reading level
        if reading_level not in self.config.reading_levels:
            reading_level = "intermediate"
            
        level_config = self.config.reading_levels[reading_level]
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Generate initial summary
        summary = self._generate_summary(
            processed_text,
            min_length=level_config["min_length"],
            max_length=level_config["max_length"]
        )
        
        # Adapt summary to reading level
        adapted_summary = self._adapt_to_reading_level(
            summary,
            reading_level,
            level_config["complexity"]
        )
        
        # Extract keywords if requested
        keywords = []
        if extract_keywords:
            keywords = self.keyword_extractor.extract(text, adapted_summary)
        
        # Calculate summary metrics
        metrics = self._calculate_metrics(text, adapted_summary)
        
        return {
            "summary": adapted_summary,
            "keywords": keywords,
            "reading_level": reading_level,
            "metrics": metrics,
            "original_length": len(text.split()),
            "summary_length": len(adapted_summary.split())
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace(''', "'").replace('"', '"').replace('"', '"')
        
        # Ensure text ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text.strip()
    
    def _generate_summary(
        self,
        text: str,
        min_length: int,
        max_length: int
    ) -> str:
        """Generate base summary using BART"""
        try:
            # Chunk text if too long
            chunks = self._chunk_text(text, self.config.max_input_length)
            
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined = ' '.join(summaries)
                # Re-summarize combined text
                final_summary = self.summarizer(
                    combined,
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=False
                )[0]['summary_text']
                return final_summary
            
            return summaries[0]
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:max_length] + "..."
    
    def _adapt_to_reading_level(
        self,
        summary: str,
        reading_level: str,
        complexity: float
    ) -> str:
        """Adapt summary to target reading level using T5"""
        try:
            # Create prompt for T5
            if reading_level == "beginner":
                prompt = f"simplify: {summary}"
            elif reading_level == "advanced":
                prompt = f"elaborate: {summary}"
            else:
                return summary  # No adaptation for intermediate
            
            # Tokenize and generate
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.config.device)
            
            outputs = self.t5_model.generate(
                **inputs,
                max_length=self.config.reading_levels[reading_level]["max_length"],
                min_length=self.config.reading_levels[reading_level]["min_length"],
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            adapted = self.t5_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return adapted
            
        except Exception as e:
            logger.error(f"Error adapting summary: {e}")
            return summary
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += 1
            
            if current_length >= max_length * 0.8:  # Leave some margin
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def _calculate_metrics(self, original: str, summary: str) -> Dict[str, float]:
        """Calculate summary quality metrics"""
        original_words = len(original.split())
        summary_words = len(summary.split())
        
        return {
            "compression_ratio": summary_words / original_words if original_words > 0 else 0,
            "avg_sentence_length": self._avg_sentence_length(summary),
            "readability_score": self._calculate_readability(summary)
        }
    
    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0
            
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (0-1)"""
        avg_sent_len = self._avg_sentence_length(text)
        
        # Simple heuristic: shorter sentences = more readable
        if avg_sent_len <= 10:
            return 1.0
        elif avg_sent_len <= 15:
            return 0.8
        elif avg_sent_len <= 20:
            return 0.6
        elif avg_sent_len <= 25:
            return 0.4
        else:
            return 0.2


class KeywordExtractor:
    """Extract key concepts from text and summary"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> set:
        """Load common stop words"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'is', 'was', 'are', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their'
        }
    
    def extract(self, original_text: str, summary: str, max_keywords: int = 10) -> List[Dict[str, any]]:
        """Extract keywords with importance scores"""
        # Combine texts for analysis
        combined = f"{original_text} {summary}"
        
        # Simple keyword extraction (can be enhanced with RAKE, YAKE, etc.)
        words = re.findall(r'\b[a-z]+\b', combined.lower())
        word_freq = {}
        
        for word in words:
            if word not in self.stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_keywords]
        
        # Format keywords with scores
        keywords = []
        for word, freq in sorted_words:
            keywords.append({
                "keyword": word,
                "score": freq / len(words),  # Normalized frequency
                "highlighted": word in summary.lower()
            })
        
        return keywords