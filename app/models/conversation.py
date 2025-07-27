"""
Conversation model for inter-agent communication tracking.
"""

import uuid
from datetime import datetime
from typing import Dict, Any
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from ..core.database import Base


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    COMPLETION = "completion"
    ERROR = "error"
    COLLABORATION = "collaboration"
    COORDINATION = "coordination"


class Conversation(Base):
    """Inter-agent conversation tracking."""
    
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True, index=True)
    from_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    to_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True, index=True)
    
    message_type = Column(SQLEnum(MessageType), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    
    context_refs = Column(JSON, nullable=True, default=list)
    conversation_metadata = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id) if self.session_id else None,
            "from_agent_id": str(self.from_agent_id),
            "to_agent_id": str(self.to_agent_id) if self.to_agent_id else None,
            "message_type": self.message_type.value,
            "content": self.content,
            "context_refs": self.context_refs,
            "metadata": self.conversation_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }