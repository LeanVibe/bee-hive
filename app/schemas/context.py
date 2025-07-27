"""Pydantic schemas for Context API endpoints."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from ..models.context import ContextType


class ContextCreate(BaseModel):
    """Schema for creating new context entries."""
    title: str = Field(..., min_length=1, max_length=255, description="Context title")
    content: str = Field(..., min_length=1, description="Context content")
    context_type: ContextType = Field(..., description="Type of context")
    agent_id: Optional[uuid.UUID] = Field(None, description="Associated agent ID")
    session_id: Optional[uuid.UUID] = Field(None, description="Associated session ID")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    tags: Optional[List[str]] = Field(default_factory=list, description="Context tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ContextUpdate(BaseModel):
    """Schema for updating existing context entries."""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Context title")
    content: Optional[str] = Field(None, min_length=1, description="Context content")
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Importance score")
    tags: Optional[List[str]] = Field(None, description="Context tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ContextResponse(BaseModel):
    """Schema for context API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    title: str
    content: str
    context_type: ContextType
    agent_id: Optional[str]
    session_id: Optional[str]
    importance_score: float
    access_count: int
    relevance_decay: float
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    is_consolidated: bool
    consolidation_summary: Optional[str]
    created_at: Optional[datetime]
    accessed_at: Optional[datetime]


class ContextListResponse(BaseModel):
    """Schema for paginated context list responses."""
    contexts: List[ContextResponse]
    total: int
    offset: int
    limit: int


class ContextSearchRequest(BaseModel):
    """Schema for context search requests."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    context_type: Optional[ContextType] = Field(None, description="Filter by context type")
    agent_id: Optional[uuid.UUID] = Field(None, description="Filter by agent ID")
    session_id: Optional[uuid.UUID] = Field(None, description="Filter by session ID")
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum relevance score")