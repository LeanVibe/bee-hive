"""
Pydantic schemas for Agent API endpoints.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from ..models.agent import AgentStatus, AgentType


class AgentCapabilityCreate(BaseModel):
    """Schema for creating agent capabilities."""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0.0-1.0)")
    specialization_areas: List[str] = Field(default_factory=list, description="Areas of specialization")


class AgentCreate(BaseModel):
    """Schema for creating new agents."""
    name: str = Field(..., min_length=1, max_length=255, description="Agent name")
    type: AgentType = Field(default=AgentType.CLAUDE, description="Agent implementation type")
    role: Optional[str] = Field(None, max_length=100, description="Agent role")
    capabilities: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Agent capabilities")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent configuration")


class AgentUpdate(BaseModel):
    """Schema for updating existing agents."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Agent name")
    role: Optional[str] = Field(None, max_length=100, description="Agent role")
    capabilities: Optional[List[Dict[str, Any]]] = Field(None, description="Agent capabilities")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    status: Optional[AgentStatus] = Field(None, description="Agent status")


class AgentResponse(BaseModel):
    """Schema for agent API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    name: str
    type: AgentType
    role: Optional[str]
    capabilities: Optional[List[Dict[str, Any]]]
    status: AgentStatus
    config: Optional[Dict[str, Any]]
    tmux_session: Optional[str]
    total_tasks_completed: int
    total_tasks_failed: int
    average_response_time: float
    context_window_usage: float
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    last_heartbeat: Optional[datetime]
    last_active: Optional[datetime]


class AgentListResponse(BaseModel):
    """Schema for paginated agent list responses."""
    agents: List[AgentResponse]
    total: int
    offset: int
    limit: int


class AgentStatsResponse(BaseModel):
    """Schema for detailed agent statistics."""
    agent_id: uuid.UUID
    total_tasks_completed: int
    total_tasks_failed: int
    success_rate: float
    average_response_time: float
    context_window_usage: float
    uptime_hours: float
    last_active: Optional[datetime]
    capabilities_count: int


class AgentActivationRequest(BaseModel):
    """Schema for agent activation requests."""
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Activation configuration")