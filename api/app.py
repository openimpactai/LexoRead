#!/usr/bin/env python3
"""
Main API application for LexoRead.

This module sets up the FastAPI application with all routes and dependencies.
"""

import time
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Import routers
from routers import text_adaptation, ocr, tts, reading_level, user_profile, auth
from api.summarization import router as summarization_router

# Import configuration
from config import settings

# Set up logging
from utils.logging import setup_logging
logger = logging.getLogger("api")

# Create FastAPI application
app = FastAPI(
    title="LexoRead API",
    description="API for LexoRead: AI-powered reading assistant for individuals with dyslexia and reading impairments",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(text_adaptation.router, prefix="/api/text", tags=["Text Adaptation"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["OCR"])
app.include_router(tts.router, prefix="/api/tts", tags=["Text-to-Speech"])
app.include_router(reading_level.router, prefix="/api/reading-level", tags=["Reading Level"])
app.include_router(user_profile.router, prefix="/api/users", tags=["User Profile"])
app.include_router(summarization_router, prefix="/api", tags=["Text Summarization"])

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "healthy", "api_version": app.version}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    setup_logging()
    logger.info("Starting LexoRead API")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LexoRead API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
