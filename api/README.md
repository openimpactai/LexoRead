# LexoRead API

This directory contains the API layer for the LexoRead project, providing HTTP endpoints for interacting with the system's core functionality.

## Architecture

The LexoRead API follows a modular microservices architecture, allowing for flexibility and scalability. Each component of the system is exposed through dedicated endpoints, enabling frontend applications to access the functionality they need.

```
api/
│
├── app.py                     # Main FastAPI application entry point
├── config.py                  # API configuration settings
├── dependencies.py            # Dependency injection for API routes
│
├── routers/                   # API route modules
│   ├── text_adaptation.py     # Text adaptation endpoints
│   ├── ocr.py                 # OCR processing endpoints
│   ├── tts.py                 # Text-to-speech endpoints
│   ├── reading_level.py       # Reading level assessment endpoints
│   └── user_profile.py        # User profile management endpoints
│
├── models/                    # API data models (Pydantic)
│   ├── requests.py            # Request models
│   └── responses.py           # Response models
│
├── services/                  # Business logic services
│   ├── text_service.py        # Text processing service
│   ├── ocr_service.py         # OCR service
│   ├── tts_service.py         # Text-to-speech service
│   ├── reading_level_service.py  # Reading level assessment service
│   └── user_service.py        # User profile service
│
├── utils/                     # Utility functions and helpers
│   ├── error_handling.py      # Error handling utilities
│   └── logging.py             # Logging configuration
│
├── middleware/                # API middleware components
│   ├── auth.py                # Authentication middleware
│   └── rate_limiter.py        # Rate limiting middleware
│
└── tests/                     # API tests
    ├── test_text_adaptation.py
    ├── test_ocr.py
    ├── test_tts.py
    ├── test_reading_level.py
    └── test_user_profile.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) (for dependency management)

### Installation

1. Install dependencies:
   ```bash
   cd api
   poetry install
   ```

2. Configure the environment:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. Run the development server:
   ```bash
   poetry run uvicorn app:app --reload
   ```

4. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

The API provides the following main endpoints:

### Text Adaptation

- `POST /api/text/adapt`: Adapt text for easier reading
- `POST /api/text/simplify`: Simplify complex words and sentences
- `GET /api/text/formats`: Get available text formatting options

### OCR

- `POST /api/ocr/extract`: Extract text from an image
- `POST /api/ocr/extract-regions`: Extract text regions from an image
- `POST /api/ocr/scan-document`: Process a multi-page document

### Text-to-Speech

- `POST /api/tts/synthesize`: Convert text to speech
- `GET /api/tts/voices`: Get available voice options
- `POST /api/tts/adaptive`: Generate speech with dyslexia-friendly parameters

### Reading Level Assessment

- `POST /api/reading-level/assess`: Assess the reading level of text
- `POST /api/reading-level/analyze`: Get detailed analysis of text complexity
- `POST /api/reading-level/suggest`: Get suggestions for text simplification

### User Profile

- `GET /api/users/{user_id}/profile`: Get user reading profile
- `POST /api/users/{user_id}/sessions`: Record a reading session
- `GET /api/users/{user_id}/recommendations`: Get personalized recommendations

## Authentication

The API uses JWT-based authentication for protected endpoints. To authenticate:

1. Obtain a token using the `/api/auth/token` endpoint
2. Include the token in the `Authorization` header of subsequent requests:
   ```
   Authorization: Bearer {your_token}
   ```

## Rate Limiting

To prevent abuse, the API implements rate limiting based on client IP address. The default limits are:

- 100 requests per minute for anonymous users
- 300 requests per minute for authenticated users

## Development and Contributing

1. Create a feature branch from `main`
2. Implement your changes
3. Write tests for your new features
4. Run the test suite:
   ```bash
   poetry run pytest
   ```
5. Submit a pull request

## Next Steps and Future Development

The current API implementation provides the core functionality for the LexoRead project, but there are several areas for future expansion and improvement:

1. **Enhanced Authentication**: Implement OAuth2 support for third-party authentication
2. **Caching Layer**: Add Redis-based caching for frequently accessed resources
3. **Batch Processing**: Support batch operations for processing multiple texts/images
4. **Websocket Support**: Real-time feedback for interactive frontend applications
5. **Internationalization**: Expanding language support beyond English
6. **Analytics Integration**: Add endpoints for usage analytics and reporting
7. **Plugin System**: Allow for dynamic extension of API functionality

Contributors are encouraged to work on these areas or suggest new features through GitHub issues.
