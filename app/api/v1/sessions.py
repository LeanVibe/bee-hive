"""Session management API endpoints."""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

router = APIRouter()


class CompactRequest(BaseModel):
    """Request model for context compression."""
    compression_level: str = Field(
        default="standard",
        description="Compression level: light, standard, or aggressive",
        pattern="^(light|standard|aggressive)$"
    )
    target_tokens: Optional[int] = Field(
        default=None,
        description="Target token count for adaptive compression",
        gt=0
    )
    preserve_decisions: bool = Field(
        default=True,
        description="Whether to preserve decision information"
    )
    preserve_patterns: bool = Field(
        default=True,
        description="Whether to preserve pattern information"
    )


class CompactResponse(BaseModel):
    """Response model for context compression."""
    success: bool
    session_id: str
    compression_level: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int
    compression_time_seconds: float
    summary: str
    key_insights: list[str]
    decisions_made: list[str]
    patterns_identified: list[str]
    importance_score: float
    message: str
    performance_met: bool
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/")
async def list_sessions():
    """List all sessions."""
    return {"message": "Sessions endpoint - coming soon"}


@router.post("/")
async def create_session():
    """Create a new session."""
    return {"message": "Create session - coming soon"}


@router.post("/{session_id}/compact", response_model=CompactResponse)
async def compact_session_context(
    session_id: str,
    request: CompactRequest
) -> CompactResponse:
    """
    Compress conversation context for a specific session.
    
    This endpoint extracts the conversation history, shared context, and coordination
    events from a session and compresses them using Claude's intelligent summarization
    while preserving key insights, decisions, and patterns.
    
    Args:
        session_id: UUID of the session to compress
        request: Compression parameters and options
        
    Returns:
        CompactResponse with compression results and metrics
        
    Raises:
        HTTPException: If session not found or compression fails
    """
    try:
        # Import here to avoid circular imports
        from ...core.hive_slash_commands import get_hive_command_registry
        
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id format: {session_id}"
            )
        
        # Get the hive command registry
        registry = get_hive_command_registry()
        compact_command = registry.get_command("compact")
        
        if not compact_command:
            raise HTTPException(
                status_code=500,
                detail="Context compression service not available"
            )
        
        # Build command arguments
        args = [session_id]
        args.append(f"--level={request.compression_level}")
        
        if request.target_tokens:
            args.append(f"--target-tokens={request.target_tokens}")
        
        if request.preserve_decisions:
            args.append("--preserve-decisions")
        else:
            args.append("--no-preserve-decisions")
            
        if request.preserve_patterns:
            args.append("--preserve-patterns")
        else:
            args.append("--no-preserve-patterns")
        
        # Execute compression command
        start_time = datetime.utcnow()
        result = await compact_command.execute(args=args, context={
            "api_request": True,
            "request_time": start_time.isoformat(),
            "session_id": session_id
        })
        
        # Handle command execution failure
        if not result.get("success", False):
            error_message = result.get("error", "Context compression failed")
            
            # Specific error handling
            if "not found" in error_message.lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found"
                )
            elif "no context" in error_message.lower():
                raise HTTPException(
                    status_code=400,
                    detail="No conversation context found in session"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=error_message
                )
        
        # Return successful response
        return CompactResponse(
            success=True,
            session_id=session_id,
            compression_level=result.get("compression_level", request.compression_level),
            original_tokens=result.get("original_tokens", 0),
            compressed_tokens=result.get("compressed_tokens", 0),
            compression_ratio=result.get("compression_ratio", 0.0),
            tokens_saved=result.get("tokens_saved", 0),
            compression_time_seconds=result.get("compression_time_seconds", 0.0),
            summary=result.get("summary", ""),
            key_insights=result.get("key_insights", []),
            decisions_made=result.get("decisions_made", []),
            patterns_identified=result.get("patterns_identified", []),
            importance_score=result.get("importance_score", 0.0),
            message=result.get("message", "Context compression completed"),
            performance_met=result.get("performance_met", False),
            timestamp=result.get("timestamp", start_time.isoformat()),
            metadata=result.get("metadata")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during context compression: {str(e)}"
        )


@router.get("/{session_id}/compact/status")
async def get_compression_status(session_id: str):
    """
    Get compression status and history for a session.
    
    Args:
        session_id: UUID of the session
        
    Returns:
        Compression status and metadata
    """
    try:
        # Import here to avoid circular imports
        from ...models.session import Session
        from ...core.database import get_db_session
        
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id format: {session_id}"
            )
        
        async with get_db_session() as db:
            session = await db.get(Session, session_id)
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found"
                )
            
            # Get compression data from shared context
            compressed_context = session.get_shared_context("compressed_context")
            compression_history = session.get_shared_context("compression_history")
            
            if not compressed_context and not compression_history:
                return {
                    "session_id": session_id,
                    "has_compressed_context": False,
                    "message": "No compression data available for this session"
                }
            
            return {
                "session_id": session_id,
                "has_compressed_context": bool(compressed_context),
                "compression_history": compression_history,
                "compressed_data": {
                    "summary_length": len(compressed_context.get("summary", "")) if compressed_context else 0,
                    "key_insights_count": len(compressed_context.get("key_insights", [])) if compressed_context else 0,
                    "decisions_count": len(compressed_context.get("decisions_made", [])) if compressed_context else 0,
                    "patterns_count": len(compressed_context.get("patterns_identified", [])) if compressed_context else 0,
                    "importance_score": compressed_context.get("importance_score", 0.0) if compressed_context else 0.0,
                    "compressed_at": compressed_context.get("compressed_at") if compressed_context else None
                },
                "session_info": {
                    "name": session.name,
                    "type": session.session_type.value,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving compression status: {str(e)}"
        )