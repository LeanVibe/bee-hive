#!/usr/bin/env python3
"""
Minimal FastAPI test server for frontend-backend connectivity testing.
This server provides basic endpoints to test API connectivity and CORS.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
import json

# Create FastAPI app
app = FastAPI(
    title="LeanVibe Test Server",
    description="Minimal test server for frontend-backend connectivity",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],  # Allow frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "LeanVibe Test Server",
        "version": "1.0.0"
    }

# Status endpoint
@app.get("/status")
def get_status():
    """System status endpoint"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "uptime": "active",
        "database": "mocked",
        "redis": "mocked",
        "services": {
            "api": "online",
            "websocket": "mocked",
            "database": "mocked"
        },
        "message": "Test server is running successfully"
    }

# API info endpoint
@app.get("/api/info")
def api_info():
    """API information endpoint"""
    return {
        "name": "LeanVibe Agent Hive 2.0 Test API",
        "version": "1.0.0",
        "description": "Test server for frontend-backend connectivity validation",
        "endpoints": [
            "/health",
            "/status", 
            "/api/info",
            "/api/test"
        ],
        "cors_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

# Test endpoint for POST requests
@app.post("/api/test")
def test_endpoint(data: dict = None):
    """Test endpoint for POST requests"""
    return {
        "success": True,
        "message": "POST request received successfully",
        "received_data": data,
        "timestamp": datetime.now().isoformat(),
        "echo": data if data else {}
    }

# Test data endpoint
@app.get("/api/test-data")
def get_test_data():
    """Test data endpoint"""
    return {
        "agents": [
            {"id": "agent-1", "name": "Test Agent 1", "status": "active"},
            {"id": "agent-2", "name": "Test Agent 2", "status": "idle"},
            {"id": "agent-3", "name": "Test Agent 3", "status": "active"},
        ],
        "metrics": {
            "total_agents": 3,
            "active_agents": 2,
            "idle_agents": 1,
            "uptime": "100%"
        },
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "LeanVibe Agent Hive 2.0 Test Server",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "api_info": "/api/info",
            "test": "/api/test",
            "test_data": "/api/test-data"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting LeanVibe Test Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 Available endpoints:")
    print("   - GET /health (health check)")
    print("   - GET /status (system status)")
    print("   - GET /api/info (API information)")
    print("   - POST /api/test (test endpoint)")
    print("   - GET /api/test-data (test data)")
    print("🔧 CORS enabled for frontend development")
    
    uvicorn.run(
        "test_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
