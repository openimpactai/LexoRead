"""
Text Summarization API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
from ..models.summarization_model import SmartSummarizationModel, SummarizationConfig
from ..api.auth import get_current_user
from ..api.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/summarize", tags=["summarization"])

# Initialize model (singleton pattern)
_summarization_model = None


def get_summarization_model():
    """Get or create summarization model instance"""
    global _summarization_model
    if _summarization_model is None:
        _summarization_model = SmartSummarizationModel()
    return _summarization_model


class SummarizationRequest(BaseModel):
    """Request model for text summarization"""
    text: str = Field(..., min_length=50, description="Text to summarize")
    reading_level: str = Field(
        default="intermediate",
        description="Target reading level",
        regex="^(beginner|intermediate|advanced)$"
    )
    extract_keywords: bool = Field(
        default=True,
        description="Extract key concepts"
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Override maximum summary length"
    )


class KeywordInfo(BaseModel):
    """Keyword information"""
    keyword: str
    score: float
    highlighted: bool


class SummarizationResponse(BaseModel):
    """Response model for text summarization"""
    summary: str
    keywords: List[KeywordInfo]
    reading_level: str
    metrics: Dict[str, float]
    original_length: int
    summary_length: int


class BatchSummarizationRequest(BaseModel):
    """Request for batch summarization"""
    texts: List[str] = Field(..., min_items=1, max_items=10)
    reading_level: str = Field(default="intermediate")
    extract_keywords: bool = Field(default=True)


@router.post("/", response_model=SummarizationResponse)
async def summarize_text(
    request: SummarizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate smart summary with reading level adaptation
    
    - Automatically adapts to user's reading level
    - Extracts key concepts
    - Provides readability metrics
    """
    try:
        model = get_summarization_model()
        
        # Override reading level if user has preference
        if hasattr(current_user, 'reading_level') and current_user.reading_level:
            request.reading_level = current_user.reading_level
        
        # Generate summary
        result = model.summarize(
            text=request.text,
            reading_level=request.reading_level,
            extract_keywords=request.extract_keywords
        )
        
        # Convert keywords to response format
        keywords = [
            KeywordInfo(**kw) for kw in result["keywords"]
        ]
        
        return SummarizationResponse(
            summary=result["summary"],
            keywords=keywords,
            reading_level=result["reading_level"],
            metrics=result["metrics"],
            original_length=result["original_length"],
            summary_length=result["summary_length"]
        )
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[SummarizationResponse])
async def batch_summarize(
    request: BatchSummarizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Summarize multiple texts in batch
    
    Useful for processing multiple documents or chapters
    """
    try:
        model = get_summarization_model()
        responses = []
        
        for text in request.texts:
            result = model.summarize(
                text=text,
                reading_level=request.reading_level,
                extract_keywords=request.extract_keywords
            )
            
            keywords = [
                KeywordInfo(**kw) for kw in result["keywords"]
            ]
            
            responses.append(SummarizationResponse(
                summary=result["summary"],
                keywords=keywords,
                reading_level=result["reading_level"],
                metrics=result["metrics"],
                original_length=result["original_length"],
                summary_length=result["summary_length"]
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reading-levels")
async def get_reading_levels():
    """Get available reading levels and their configurations"""
    model = get_summarization_model()
    return {
        "levels": list(model.config.reading_levels.keys()),
        "configurations": model.config.reading_levels
    }


@router.post("/highlight-keywords")
async def highlight_keywords_in_text(
    text: str,
    keywords: List[str],
    current_user: User = Depends(get_current_user)
):
    """
    Highlight keywords in original text
    
    Returns text with HTML markup for highlighting
    """
    try:
        highlighted_text = text
        
        # Sort keywords by length (longest first) to avoid partial replacements
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Case-insensitive replacement with highlighting
            import re
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark class="keyword-highlight">{keyword}</mark>',
                highlighted_text
            )
        
        return {
            "highlighted_text": highlighted_text,
            "keyword_count": len(keywords)
        }
        
    except Exception as e:
        logger.error(f"Keyword highlighting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive-summary")
async def generate_adaptive_summary(
    text: str,
    target_length: int = 100,
    current_user: User = Depends(get_current_user)
):
    """
    Generate summary with specific target length
    
    Adapts summary to meet exact length requirements
    """
    try:
        model = get_summarization_model()
        
        # Determine reading level from target length
        if target_length < 80:
            reading_level = "beginner"
        elif target_length < 150:
            reading_level = "intermediate"
        else:
            reading_level = "advanced"
        
        # Generate summary with custom length
        result = model.summarize(
            text=text,
            reading_level=reading_level,
            extract_keywords=True
        )
        
        # Adjust summary if needed
        summary_words = result["summary"].split()
        if len(summary_words) > target_length:
            result["summary"] = ' '.join(summary_words[:target_length]) + "..."
        
        return {
            "summary": result["summary"],
            "actual_length": len(result["summary"].split()),
            "target_length": target_length,
            "keywords": result["keywords"]
        }
        
    except Exception as e:
        logger.error(f"Adaptive summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))