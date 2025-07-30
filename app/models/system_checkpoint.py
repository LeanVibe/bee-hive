"""
System checkpoint model for state management.
"""

import uuid
from datetime import datetime
from typing import Dict, Any
from enum import Enum

from sqlalchemy import Column, String, DateTime, JSON, Enum as SQLEnum, Text
from sqlalchemy.sql import func

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class CheckpointType(Enum):
    """Types of system checkpoints."""
    SCHEDULED = "scheduled"
    PRE_SLEEP = "pre_sleep"
    ERROR_RECOVERY = "error_recovery"
    MANUAL = "manual"


class SystemCheckpoint(Base):
    """System state checkpoints."""
    
    __tablename__ = "system_checkpoints"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    checkpoint_type = Column(SQLEnum(CheckpointType), nullable=False, index=True)
    state = Column(JSON, nullable=False)
    git_commit_hash = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "checkpoint_type": self.checkpoint_type.value,
            "state": self.state,
            "git_commit_hash": self.git_commit_hash,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }