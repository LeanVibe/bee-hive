"""
Context model for LeanVibe Agent Hive 2.0

Represents stored context and memory with semantic vector search
capabilities for intelligent context retrieval and consolidation.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Float, ForeignKey
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class ContextType(Enum):
    """Types of context information."""
    CONVERSATION = "conversation"
    CODE_SNIPPET = "code_snippet"
    DOCUMENTATION = "documentation"
    DECISION = "decision"
    LEARNING = "learning"
    ERROR_RESOLUTION = "error_resolution"
    ARCHITECTURE = "architecture"
    TASK_RESULT = "task_result"
    SYSTEM_STATE = "system_state"


class Context(Base):
    """
    Represents stored context with semantic vector embeddings.
    
    Enables intelligent context retrieval, memory consolidation,
    and semantic search across agent conversations and decisions.
    """
    
    __tablename__ = "contexts"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    
    # Classification and relationships
    context_type = Column(SQLEnum(ContextType), nullable=False, index=True)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True, index=True)
    session_id = Column(DatabaseAgnosticUUID(), ForeignKey("sessions.id"), nullable=True, index=True)
    
    # Semantic search
    embedding = Column(Vector(1536), nullable=True)  # OpenAI embedding dimension
    
    # Context hierarchy and relationships
    parent_context_id = Column(DatabaseAgnosticUUID(), ForeignKey("contexts.id"), nullable=True)
    related_context_ids = Column(JSON, nullable=True, default=list)
    
    # Importance and relevance
    importance_score = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    access_count = Column(String, nullable=False, default="0")
    relevance_decay = Column(Float, nullable=False, default=1.0)   # Decreases over time
    
    # Metadata and tags
    context_metadata = Column(JSON, nullable=True, default=dict)
    tags = Column(JSON, nullable=True, default=list)
    
    # Consolidation state
    is_consolidated = Column(String, nullable=False, default="false")  # Bool as string
    consolidation_summary = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    consolidated_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<Context(id={self.id}, title='{self.title}', type='{self.context_type}', importance={self.importance_score})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "id": str(self.id),
            "title": self.title,
            "content": self.content,
            "context_type": self.context_type.value,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "parent_context_id": str(self.parent_context_id) if self.parent_context_id else None,
            "related_context_ids": self.related_context_ids,
            "importance_score": self.importance_score,
            "access_count": int(self.access_count or 0),
            "relevance_decay": self.relevance_decay,
            "metadata": self.context_metadata,
            "tags": self.tags,
            "is_consolidated": self.is_consolidated == "true",
            "consolidation_summary": self.consolidation_summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "consolidated_at": self.consolidated_at.isoformat() if self.consolidated_at else None
        }
    
    def mark_accessed(self) -> None:
        """Mark context as accessed and update relevance."""
        self.accessed_at = datetime.utcnow()
        current_count = int(self.access_count or 0)
        self.access_count = str(current_count + 1)
        
        # Boost relevance when accessed
        self.relevance_decay = min(1.0, self.relevance_decay + 0.1)
    
    def calculate_current_relevance(self) -> float:
        """Calculate current relevance considering time decay."""
        if not self.created_at:
            return self.importance_score
        
        # Time-based decay
        age_days = (datetime.utcnow() - self.created_at).days
        time_decay = max(0.1, 1.0 - (age_days * 0.01))  # 1% decay per day, min 10%
        
        # Access-based boost
        access_count = int(self.access_count or 0)
        access_boost = min(0.5, access_count * 0.1)  # Up to 50% boost
        
        return min(1.0, (self.importance_score * time_decay * self.relevance_decay) + access_boost)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the context."""
        if self.tags is None:
            self.tags = []
        
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the context."""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if context has a specific tag."""
        return self.tags is not None and tag in self.tags
    
    def add_related_context(self, context_id: uuid.UUID) -> None:
        """Add a related context reference."""
        if self.related_context_ids is None:
            self.related_context_ids = []
        
        context_str = str(context_id)
        if context_str not in self.related_context_ids:
            self.related_context_ids.append(context_str)
    
    def remove_related_context(self, context_id: uuid.UUID) -> None:
        """Remove a related context reference."""
        if self.related_context_ids:
            context_str = str(context_id)
            if context_str in self.related_context_ids:
                self.related_context_ids.remove(context_str)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata information."""
        if self.context_metadata is None:
            self.context_metadata = {}
        
        self.context_metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        if not self.context_metadata:
            return default
        
        return self.context_metadata.get(key, default)
    
    def consolidate(self, summary: str) -> None:
        """Mark context as consolidated with a summary."""
        self.is_consolidated = "true"
        self.consolidation_summary = summary
        self.consolidated_at = datetime.utcnow()
        
        # Boost importance for consolidated contexts
        self.importance_score = min(1.0, self.importance_score + 0.2)
    
    def is_stale(self, days_threshold: int = 30) -> bool:
        """Check if context is stale and should be archived."""
        if not self.accessed_at:
            return True
        
        age_days = (datetime.utcnow() - self.accessed_at).days
        return age_days > days_threshold and self.calculate_current_relevance() < 0.3
    
    def should_be_consolidated(self) -> bool:
        """Check if context should be consolidated."""
        if self.is_consolidated == "true":
            return False
        
        # Consolidate if highly accessed or important
        access_count = int(self.access_count or 0)
        return (access_count >= 5 or 
                self.importance_score >= 0.8 or
                self.context_type in [ContextType.DECISION, ContextType.ARCHITECTURE])
    
    def get_age_days(self) -> int:
        """Get age of context in days."""
        if not self.created_at:
            return 0
        
        return (datetime.utcnow() - self.created_at).days
    
    def create_child_context(
        self,
        title: str,
        content: str,
        context_type: ContextType,
        importance_score: float = 0.5
    ) -> "Context":
        """Create a child context linked to this one."""
        child = Context(
            title=title,
            content=content,
            context_type=context_type,
            parent_context_id=self.id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            importance_score=importance_score
        )
        
        # Inherit some metadata
        if self.context_metadata:
            child.context_metadata = {
                "parent_context": str(self.id),
                "inherited_from": self.title
            }
        
        return child