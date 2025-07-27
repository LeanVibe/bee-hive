"""Pydantic schemas for Session API endpoints."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from ..models.session import SessionStatus, SessionType


class SessionCreate(BaseModel):
    """Schema for creating new sessions."""
    name: str = Field(..., min_length=1, max_length=255, description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    session_type: SessionType = Field(..., description="Type of development session")
    objectives: Optional[List[str]] = Field(default_factory=list, description="Session objectives")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session configuration")


class SessionUpdate(BaseModel):
    """Schema for updating existing sessions."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    status: Optional[SessionStatus] = Field(None, description="Session status")
    objectives: Optional[List[str]] = Field(None, description="Session objectives")
    config: Optional[Dict[str, Any]] = Field(None, description="Session configuration")


class SessionResponse(BaseModel):
    """Schema for session API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    name: str
    description: Optional[str]
    session_type: SessionType
    status: SessionStatus
    participant_agents: Optional[List[str]]
    lead_agent_id: Optional[str]
    objectives: Optional[List[str]]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class SessionListResponse(BaseModel):
    """Schema for paginated session list responses."""
    sessions: List[SessionResponse]
    total: int
    offset: int
    limit: int