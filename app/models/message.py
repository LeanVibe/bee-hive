"""
Message models for Redis Streams communication system.
Provides reliable, durable message passing between agents.
"""

import uuid
import time
import hmac
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from ..core.database import Base


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result" 
    EVENT = "event"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class StreamMessage(BaseModel):
    """
    Pydantic model for Redis Streams messages.
    Provides validation and serialization for inter-agent communication.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = Field(..., min_length=1, max_length=255)
    to_agent: Optional[str] = Field(None, max_length=255)  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = Field(default_factory=time.time)
    ttl: Optional[int] = Field(None, gt=0)  # TTL in seconds
    correlation_id: Optional[str] = Field(None)
    signature: Optional[str] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            MessageType: lambda v: v.value,
            MessagePriority: lambda v: v.value
        }
    
    def to_redis_dict(self) -> Dict[str, str]:
        """Convert message to Redis-compatible dictionary."""
        data = {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent or "broadcast",
            "message_type": self.message_type.value,
            "payload": json.dumps(self.payload),
            "priority": self.priority.value,
            "timestamp": str(self.timestamp),
            "correlation_id": self.correlation_id or "",
            "signature": self.signature or ""
        }
        
        if self.ttl:
            data["ttl"] = str(self.ttl)
            
        return data
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, str]) -> "StreamMessage":
        """Create message from Redis dictionary."""
        return cls(
            id=data["id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"] if data["to_agent"] != "broadcast" else None,
            message_type=MessageType(data["message_type"]),
            payload=json.loads(data["payload"]),
            priority=MessagePriority(data["priority"]),
            timestamp=float(data["timestamp"]),
            ttl=int(data["ttl"]) if data.get("ttl") else None,
            correlation_id=data.get("correlation_id") or None,
            signature=data.get("signature") or None
        )
    
    def sign(self, secret_key: str) -> None:
        """Sign message with HMAC for authenticity."""
        payload_str = json.dumps(self.payload, sort_keys=True)
        message_data = f"{self.id}{self.from_agent}{self.to_agent or 'broadcast'}{self.message_type.value}{payload_str}{self.timestamp}"
        
        self.signature = hmac.new(
            secret_key.encode(),
            message_data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, secret_key: str) -> bool:
        """Verify message signature."""
        if not self.signature:
            return False
            
        # Create signature without existing signature
        original_signature = self.signature
        self.signature = None
        
        try:
            self.sign(secret_key)
            is_valid = hmac.compare_digest(original_signature, self.signature)
            return is_valid
        finally:
            self.signature = original_signature
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if not self.ttl:
            return False
        
        return time.time() > (self.timestamp + self.ttl)
    
    def get_stream_name(self) -> str:
        """Get Redis stream name for this message."""
        if self.to_agent:
            return f"agent_messages:{self.to_agent}"
        else:
            return "agent_messages:broadcast"
    
    def get_consumer_group(self) -> str:
        """Get consumer group name based on message type."""
        return f"consumers:{self.message_type.value}"


class MessageAudit(Base):
    """
    Database model for message auditing and persistence.
    Provides durability and compliance tracking.
    """
    
    __tablename__ = "message_audit"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(String(255), nullable=False, index=True)
    stream_name = Column(String(255), nullable=False, index=True)
    from_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    to_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True, index=True)
    
    message_type = Column(SQLEnum(MessageType), nullable=False, index=True)
    priority = Column(SQLEnum(MessagePriority), nullable=False, default=MessagePriority.NORMAL)
    status = Column(SQLEnum(MessageStatus), nullable=False, default=MessageStatus.PENDING, index=True)
    
    payload = Column(JSON, nullable=False)
    correlation_id = Column(String(255), nullable=True, index=True)
    
    # Timing and delivery tracking
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(String, nullable=False, default="0")  # Int as string for consistency
    
    # Performance metrics
    delivery_latency_ms = Column(String, nullable=True)  # Float as string
    processing_latency_ms = Column(String, nullable=True)  # Float as string
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary."""
        return {
            "id": str(self.id),
            "message_id": self.message_id,
            "stream_name": self.stream_name,
            "from_agent_id": str(self.from_agent_id),
            "to_agent_id": str(self.to_agent_id) if self.to_agent_id else None,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "error_message": self.error_message,
            "retry_count": int(self.retry_count or 0),
            "delivery_latency_ms": float(self.delivery_latency_ms or 0),
            "processing_latency_ms": float(self.processing_latency_ms or 0)
        }
    
    def mark_acknowledged(self) -> None:
        """Mark message as acknowledged."""
        self.status = MessageStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        
        if self.sent_at:
            delta = self.acknowledged_at - self.sent_at
            self.delivery_latency_ms = str(delta.total_seconds() * 1000)
    
    def mark_processed(self) -> None:
        """Mark message as processed."""
        self.status = MessageStatus.ACKNOWLEDGED
        self.processed_at = datetime.utcnow()
        
        if self.acknowledged_at:
            delta = self.processed_at - self.acknowledged_at
            self.processing_latency_ms = str(delta.total_seconds() * 1000)
    
    def mark_failed(self, error_message: str) -> None:
        """Mark message as failed."""
        self.status = MessageStatus.FAILED
        self.failed_at = datetime.utcnow()
        self.error_message = error_message
        self.retry_count = str(int(self.retry_count or 0) + 1)
    
    def mark_dead_letter(self) -> None:
        """Move message to dead letter status."""
        self.status = MessageStatus.DEAD_LETTER
        
    def is_retryable(self, max_retries: int = 3) -> bool:
        """Check if message can be retried."""
        return (
            self.status == MessageStatus.FAILED and 
            int(self.retry_count or 0) < max_retries
        )


class ConsumerGroupInfo(BaseModel):
    """Information about Redis consumer group."""
    
    name: str
    consumers: int
    pending: int
    last_delivered_id: str
    lag: int
    
    
class StreamInfo(BaseModel):
    """Information about Redis stream."""
    
    name: str
    length: int
    groups: List[ConsumerGroupInfo]
    first_entry_id: Optional[str]
    last_entry_id: Optional[str]
    max_deleted_entry_id: Optional[str]


class MessageDeliveryReport(BaseModel):
    """Report on message delivery performance."""
    
    total_sent: int
    total_acknowledged: int
    total_failed: int
    success_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_msg_per_sec: float
    error_rate: float