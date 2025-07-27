"""Session management API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_sessions():
    """List all sessions."""
    return {"message": "Sessions endpoint - coming soon"}


@router.post("/")
async def create_session():
    """Create a new session."""
    return {"message": "Create session - coming soon"}