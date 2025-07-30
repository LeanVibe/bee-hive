"""
Observability models for LeanVibe Agent Hive 2.0

Database models for event tracking, performance monitoring, and system observability.
Optimized for high-throughput event capture and efficient querying.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray

logger = structlog.get_logger()


class EventType(str, enum.Enum):
    """
    Event types for agent lifecycle hooks as defined in the observability PRD.
    
    These correspond to Claude Code Hooks and system events that need monitoring.
    """
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    NOTIFICATION = "Notification"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"


class AgentEvent(Base):
    """
    Agent event model for comprehensive observability and monitoring.
    
    Stores all agent lifecycle events with JSONB payloads for flexible event data.
    Optimized for high-throughput writes and efficient time-series queries.
    """
    
    __tablename__ = "agent_events"
    
    # Primary key using BigInteger for high-volume inserts
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Core event identifiers
    session_id: Mapped[uuid.UUID] = mapped_column(DatabaseAgnosticUUID(), nullable=False)
    agent_id: Mapped[uuid.UUID] = mapped_column(DatabaseAgnosticUUID(), nullable=False)
    
    # Event type from enum
    event_type: Mapped[EventType] = mapped_column(nullable=False)
    
    # Flexible JSONB payload for event-specific data
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Performance metric - tool execution latency
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamp with timezone support
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<AgentEvent(id={self.id}, "
            f"event_type={self.event_type}, "
            f"agent_id={self.agent_id}, "
            f"session_id={self.session_id}, "
            f"created_at={self.created_at})>"
        )
    
    @property
    def tool_name(self) -> Optional[str]:
        """Extract tool name from payload if available."""
        return self.payload.get("tool_name")
    
    @property
    def success(self) -> Optional[bool]:
        """Extract success status from payload if available."""
        return self.payload.get("success")
    
    @property
    def error_message(self) -> Optional[str]:
        """Extract error message from payload if available."""
        return self.payload.get("error")
    
    @property
    def correlation_id(self) -> Optional[str]:
        """Extract correlation ID from payload for tracing."""
        return self.payload.get("correlation_id")
    
    @property
    def execution_time_ms(self) -> Optional[int]:
        """Extract execution time from payload if available."""
        return self.payload.get("execution_time_ms")
    
    @classmethod
    def create_pre_tool_use(
        cls,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        tool_name: str,
        parameters: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> "AgentEvent":
        """
        Create a PreToolUse event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            tool_name: Name of the tool being executed
            parameters: Tool parameters
            correlation_id: Optional correlation ID for request tracing
            
        Returns:
            AgentEvent instance
        """
        payload = {
            "tool_name": tool_name,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if correlation_id:
            payload["correlation_id"] = correlation_id
            
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.PRE_TOOL_USE,
            payload=payload
        )
    
    @classmethod
    def create_post_tool_use(
        cls,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        correlation_id: Optional[str] = None,
        latency_ms: Optional[int] = None
    ) -> "AgentEvent":
        """
        Create a PostToolUse event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            tool_name: Name of the tool that was executed
            success: Whether the tool execution succeeded
            result: Tool execution result (if successful)
            error: Error message (if failed)
            execution_time_ms: Tool execution time in milliseconds
            correlation_id: Optional correlation ID for request tracing
            latency_ms: Overall latency including overhead
            
        Returns:
            AgentEvent instance
        """
        payload = {
            "tool_name": tool_name,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result is not None:
            # Truncate large results to avoid payload bloat
            if isinstance(result, str) and len(result) > 10000:
                payload["result"] = result[:10000] + "... (truncated)"
                payload["result_truncated"] = True
                payload["full_result_size"] = len(result)
            else:
                payload["result"] = result
        
        if error:
            payload["error"] = error
            
        if execution_time_ms is not None:
            payload["execution_time_ms"] = execution_time_ms
            
        if correlation_id:
            payload["correlation_id"] = correlation_id
            
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.POST_TOOL_USE,
            payload=payload,
            latency_ms=latency_ms
        )
    
    @classmethod
    def create_notification(
        cls,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "AgentEvent":
        """
        Create a Notification event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            level: Notification level (info, warning, error)
            message: Notification message
            details: Optional additional details
            
        Returns:
            AgentEvent instance
        """
        payload = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
            
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.NOTIFICATION,
            payload=payload
        )
    
    @classmethod
    def create_stop(
        cls,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "AgentEvent":
        """
        Create a Stop event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            reason: Reason for stopping
            details: Optional additional details
            
        Returns:
            AgentEvent instance
        """
        payload = {
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
            
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.STOP,
            payload=payload
        )
    
    @classmethod
    def create_subagent_stop(
        cls,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        subagent_id: uuid.UUID,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "AgentEvent":
        """
        Create a SubagentStop event.
        
        Args:
            session_id: Session UUID
            agent_id: Parent agent UUID
            subagent_id: Subagent UUID that stopped
            reason: Reason for stopping
            details: Optional additional details
            
        Returns:
            AgentEvent instance
        """
        payload = {
            "subagent_id": str(subagent_id),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
            
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.SUBAGENT_STOP,
            payload=payload
        )


class ChatTranscript(Base):
    """
    Chat transcript model for optional S3/MinIO storage.
    
    Stores references to chat logs stored in object storage with metadata
    for efficient retrieval and analysis.
    """
    
    __tablename__ = "chat_transcripts"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        DatabaseAgnosticUUID(), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Session and agent references
    session_id: Mapped[uuid.UUID] = mapped_column(
        DatabaseAgnosticUUID(), 
        ForeignKey("sessions.id"),
        nullable=False
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        DatabaseAgnosticUUID(), 
        ForeignKey("agents.id"),
        nullable=False
    )
    
    # S3/MinIO storage key
    s3_key: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # File size for storage management
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<ChatTranscript(id={self.id}, "
            f"session_id={self.session_id}, "
            f"agent_id={self.agent_id}, "
            f"s3_key={self.s3_key}, "
            f"size_bytes={self.size_bytes})>"
        )