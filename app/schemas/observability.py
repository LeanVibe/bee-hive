"""
Observability schemas for LeanVibe Agent Hive 2.0.

Pydantic schemas for event processing, hook management, and monitoring.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class HookEventCreate(BaseModel):
    """Schema for creating hook events."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    session_id: uuid.UUID = Field(..., description="Session identifier")
    agent_id: uuid.UUID = Field(..., description="Agent identifier")
    event_type: str = Field(..., description="Type of event")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    correlation_id: Optional[str] = Field(None, description="Optional correlation ID")
    severity: Optional[str] = Field("info", description="Event severity level")
    tags: Optional[Dict[str, str]] = Field(None, description="Optional metadata tags")


class HookEventResponse(BaseModel):
    """Schema for hook event response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    event_id: str = Field(..., description="Generated event ID")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(..., description="Processing timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class EventAnalyticsRequest(BaseModel):
    """Schema for event analytics request."""
    
    session_id: Optional[uuid.UUID] = None
    agent_id: Optional[uuid.UUID] = None
    time_range_hours: int = Field(default=1, ge=1, le=168)
    event_types: Optional[List[str]] = None
    include_trends: bool = True
    include_patterns: bool = True


class EventAnalyticsResponse(BaseModel):
    """Schema for event analytics response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    summary: Dict[str, Any]
    event_distribution: Dict[str, Any]
    performance_trends: Dict[str, Any]
    error_patterns: Dict[str, Any]
    agent_activity: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    generated_at: datetime


class WebSocketSubscription(BaseModel):
    """Schema for WebSocket subscription preferences."""
    
    event_types: List[str] = Field(default_factory=list)
    agent_filters: List[str] = Field(default_factory=list)
    session_filters: List[str] = Field(default_factory=list)
    severity_filters: List[str] = Field(
        default=["info", "warning", "error", "critical"]
    )


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    websocket_connections: Dict[str, Any] = Field(..., description="WebSocket statistics")
    system_info: Dict[str, Any] = Field(..., description="System information")
    health_checks: Dict[str, str] = Field(..., description="Individual health checks")


class TestEventRequest(BaseModel):
    """Schema for test event triggering."""
    
    event_type: str = Field(..., description="Event type to trigger")
    agent_id: Optional[str] = Field(None, description="Optional agent ID")
    session_id: Optional[str] = Field(None, description="Optional session ID")
    payload: Optional[Dict[str, Any]] = Field(None, description="Optional event payload")


class TestEventResponse(BaseModel):
    """Schema for test event response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    success: bool = Field(..., description="Whether test event was triggered")
    event_id: str = Field(..., description="Generated event ID")
    event_type: str = Field(..., description="Event type that was triggered")
    agent_id: str = Field(..., description="Agent ID used")
    session_id: str = Field(..., description="Session ID used")
    timestamp: datetime = Field(..., description="Trigger timestamp")