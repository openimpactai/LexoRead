# Smart Text Summarization Feature

## Overview
This feature adds intelligent text summarization capabilities to LexoRead, allowing users to quickly understand long texts through AI-generated summaries adapted to their reading level.

## Features

### 1. **Adaptive Summarization**
- Generates summaries based on user's reading level (beginner/intermediate/advanced)
- Adjusts summary complexity and length automatically
- Uses BART for base summarization and T5 for adaptation

### 2. **Key Concept Extraction**
- Identifies and highlights important keywords
- Provides importance scores for each keyword
- Shows which keywords appear in the summary

### 3. **Multiple Endpoints**

#### `/api/summarize/` - Main summarization endpoint
```json
POST /api/summarize/
{
    "text": "Your long text here...",
    "reading_level": "intermediate",
    "extract_keywords": true
}
```

#### `/api/summarize/batch` - Batch processing
Process multiple texts at once:
```json
POST /api/summarize/batch
{
    "texts": ["Text 1...", "Text 2..."],
    "reading_level": "beginner"
}
```

#### `/api/summarize/adaptive-summary` - Length-specific summaries
Generate summaries with exact word count:
```json
POST /api/summarize/adaptive-summary
{
    "text": "Your text...",
    "target_length": 100
}
```

#### `/api/summarize/highlight-keywords` - Keyword highlighting
Highlight keywords in original text:
```json
POST /api/summarize/highlight-keywords
{
    "text": "Original text...",
    "keywords": ["keyword1", "keyword2"]
}
```

## Installation

1. Install additional dependencies:
```bash
pip install -r models/summarization_requirements.txt
```

2. The models will be downloaded automatically on first use:
   - BART: `facebook/bart-large-cnn` (~1.6GB)
   - T5: `t5-base` (~900MB)

## Usage Example

```python
import requests

# Summarize text
response = requests.post(
    "http://localhost:8000/api/summarize/",
    json={
        "text": "Your long article text here...",
        "reading_level": "beginner",
        "extract_keywords": True
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

result = response.json()
print(f"Summary: {result['summary']}")
print(f"Keywords: {[k['keyword'] for k in result['keywords']]}")
print(f"Compression ratio: {result['metrics']['compression_ratio']}")
```

## Response Format

```json
{
    "summary": "Simplified summary text...",
    "keywords": [
        {
            "keyword": "important",
            "score": 0.85,
            "highlighted": true
        }
    ],
    "reading_level": "beginner",
    "metrics": {
        "compression_ratio": 0.15,
        "avg_sentence_length": 12.5,
        "readability_score": 0.9
    },
    "original_length": 500,
    "summary_length": 75
}
```

## Configuration

The summarization model can be configured in `models/summarization_model.py`:

```python
SummarizationConfig(
    model_name="facebook/bart-large-cnn",  # Base model
    max_input_length=1024,                  # Max input tokens
    reading_levels={
        "beginner": {"min_length": 30, "max_length": 80},
        "intermediate": {"min_length": 50, "max_length": 120},
        "advanced": {"min_length": 80, "max_length": 200}
    }
)
```

## Performance Considerations

- First request will be slower due to model loading
- Models are cached after first load
- GPU recommended for faster inference
- Batch processing available for multiple texts

## Future Enhancements

1. **Multi-language support** - Add models for other languages
2. **Domain-specific summarization** - Fine-tune for specific topics
3. **Extractive + Abstractive hybrid** - Combine approaches
4. **User feedback loop** - Learn from user preferences
5. **Streaming summaries** - Real-time summary generation