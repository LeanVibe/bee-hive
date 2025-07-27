"""System management API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def get_system_status():
    """Get system status."""
    return {
        "status": "healthy",
        "message": "LeanVibe Agent Hive 2.0 is running",
        "version": "2.0.0"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}