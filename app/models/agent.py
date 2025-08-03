"""
Agent model for LeanVibe Agent Hive 2.0

Represents individual AI agents in the multi-agent system with their
capabilities, status, and configuration for coordinated development.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class AgentStatus(Enum):
    """Agent lifecycle status."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentType(Enum):
    """Type of agent implementation."""
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    CUSTOM = "custom"


class Agent(Base):
    """
    Represents an AI agent in the multi-agent development system.
    
    Each agent has specific capabilities, roles, and can participate
    in coordinated development workflows through the orchestrator.
    """
    
    __tablename__ = "agents"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    type = Column(SQLEnum(AgentType, native_enum=False), nullable=False, default=AgentType.CLAUDE)
    
    # Role and capabilities
    role = Column(String(100), nullable=True, index=True)
    capabilities = Column(JSON, nullable=True, default=list)
    system_prompt = Column(Text, nullable=True)
    
    # Current status and configuration
    status = Column(SQLEnum(AgentStatus, native_enum=False), nullable=False, default=AgentStatus.INACTIVE, index=True)
    config = Column(JSON, nullable=True, default=dict)
    
    # tmux integration
    tmux_session = Column(String(255), nullable=True)
    
    # Performance tracking
    total_tasks_completed = Column(String, nullable=True, default="0")
    total_tasks_failed = Column(String, nullable=True, default="0") 
    average_response_time = Column(String, nullable=True, default="0.0")
    context_window_usage = Column(String, nullable=True, default="0.0")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    last_active = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    sleep_windows = relationship("SleepWindow", back_populates="agent", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="agent", cascade="all, delete-orphan")
    sleep_wake_cycles = relationship("SleepWakeCycle", back_populates="agent", cascade="all, delete-orphan")
    sleep_wake_analytics = relationship("SleepWakeAnalytics", back_populates="agent", cascade="all, delete-orphan")
    # Note: performance_history, persona_assignments and persona_performance relationships removed during stabilization
    # TODO: Fix import issues with AgentPerformanceHistory, PersonaAssignmentModel and PersonaPerformanceModel
    
    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name='{self.name}', role='{self.role}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.type.value,
            "role": self.role,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "config": self.config,
            "tmux_session": self.tmux_session,
            "total_tasks_completed": int(self.total_tasks_completed or 0),
            "total_tasks_failed": int(self.total_tasks_failed or 0),
            "average_response_time": float(self.average_response_time or 0.0),
            "context_window_usage": float(self.context_window_usage or 0.0),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_active": self.last_active.isoformat() if self.last_active else None
        }
    
    def update_heartbeat(self) -> None:
        """Update the agent's heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
        if self.status == AgentStatus.ACTIVE:
            self.last_active = datetime.utcnow()
    
    def add_capability(self, name: str, description: str, confidence: float, areas: list) -> None:
        """Add a new capability to the agent."""
        if self.capabilities is None:
            self.capabilities = []
        
        capability = {
            "name": name,
            "description": description, 
            "confidence_level": confidence,
            "specialization_areas": areas
        }
        
        self.capabilities.append(capability)
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        if not self.capabilities:
            return False
        
        return any(cap.get("name") == capability_name for cap in self.capabilities)
    
    def get_capability_confidence(self, capability_name: str) -> float:
        """Get confidence level for a specific capability."""
        if not self.capabilities:
            return 0.0
        
        for cap in self.capabilities:
            if cap.get("name") == capability_name:
                return cap.get("confidence_level", 0.0)
        
        return 0.0
    
    def is_available_for_task(self) -> bool:
        """Check if agent is available to take on new tasks."""
        return self.status in [AgentStatus.ACTIVE] and self.context_window_usage and float(self.context_window_usage) < 0.8
    
    def calculate_task_suitability(self, task_type: str, required_capabilities: list) -> float:
        """Calculate how suitable this agent is for a specific task."""
        if not self.is_available_for_task():
            return 0.0
        
        if not self.capabilities or not required_capabilities:
            return 0.5  # Neutral score if no capability information
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for req_cap in required_capabilities:
            max_possible_score += 1.0
            
            # Find matching capability
            for agent_cap in self.capabilities:
                if req_cap.lower() in agent_cap.get("name", "").lower():
                    total_score += agent_cap.get("confidence_level", 0.0)
                    break
                elif any(req_cap.lower() in area.lower() for area in agent_cap.get("specialization_areas", [])):
                    total_score += agent_cap.get("confidence_level", 0.0) * 0.8
                    break
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0