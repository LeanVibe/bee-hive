"""
Performance metrics model for system monitoring.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class PerformanceMetric(Base):
    """Performance metrics tracking."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    tags = Column(JSON, nullable=True, default=dict)
    
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True, index=True)
    session_id = Column(DatabaseAgnosticUUID(), ForeignKey("sessions.id"), nullable=True, index=True)
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "tags": self.tags,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }