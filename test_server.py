#!/usr/bin/env python3
"""
Simple test server for API v2 validation
Testing Infrastructure Specialist Agent Mission

Bypasses complex initialization to test API v2 endpoints directly.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="LeanVibe Agent Hive 2.0 - Test Server",
    description="Simplified server for testing API v2 endpoints",
    version="2.0.0-test"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include API v2 routes
from app.api.v2 import api_router as api_v2_router

app.include_router(api_v2_router, prefix="/api/v2")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0-test",
        "message": "Testing Infrastructure API v2 validation server"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "LeanVibe Agent Hive 2.0 - Test Server for API v2 validation"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)