"""
Agent Performance Tracking Models for LeanVibe Agent Hive 2.0

Enhanced performance tracking models for intelligent task routing
and capability-based agent selection optimization.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Integer, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class PerformanceCategory(Enum):
    """Categories of performance metrics."""
    TASK_COMPLETION = "task_completion"
    RESPONSE_TIME = "response_time"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    SPECIALIZATION = "specialization"
    WORKLOAD = "workload"


class AgentPerformanceHistory(Base):
    """
    Historical performance data for agents to enable intelligent routing.
    
    Tracks detailed performance metrics over time for capability-based
    task routing and performance optimization.
    """
    
    __tablename__ = "agent_performance_history"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=False, index=True)
    task_id = Column(DatabaseAgnosticUUID(), ForeignKey("tasks.id"), nullable=True, index=True)
    
    # Performance metrics
    task_type = Column(String(100), nullable=True, index=True)
    success = Column(Boolean, nullable=False, default=False)
    completion_time_minutes = Column(Float, nullable=True)
    estimated_time_minutes = Column(Float, nullable=True)
    time_variance_ratio = Column(Float, nullable=True)  # actual/estimated
    
    # Quality metrics
    retry_count = Column(Integer, nullable=False, default=0)
    error_rate = Column(Float, nullable=False, default=0.0)
    confidence_score = Column(Float, nullable=True)
    
    # Context and metadata
    context_window_usage = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    # Task characteristics
    priority_level = Column(Integer, nullable=True)
    complexity_score = Column(Float, nullable=True)
    required_capabilities = Column(JSON, nullable=True, default=list)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="performance_history")
    task = relationship("Task")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "task_id": str(self.task_id) if self.task_id else None,
            "task_type": self.task_type,
            "success": self.success,
            "completion_time_minutes": self.completion_time_minutes,
            "estimated_time_minutes": self.estimated_time_minutes,
            "time_variance_ratio": self.time_variance_ratio,
            "retry_count": self.retry_count,
            "error_rate": self.error_rate,
            "confidence_score": self.confidence_score,
            "context_window_usage": self.context_window_usage,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "priority_level": self.priority_level,
            "complexity_score": self.complexity_score,
            "required_capabilities": self.required_capabilities,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None
        }
    
    def calculate_efficiency_score(self) -> float:
        """Calculate efficiency score based on time variance."""
        if not self.time_variance_ratio:
            return 0.5  # Neutral score
        
        # Efficiency is better when actual time is less than estimated
        if self.time_variance_ratio <= 1.0:
            return min(1.0, 1.0 + (1.0 - self.time_variance_ratio) * 0.5)
        else:
            return max(0.0, 1.0 - (self.time_variance_ratio - 1.0) * 0.5)
    
    def calculate_reliability_score(self) -> float:
        """Calculate reliability score based on success and error rate."""
        if not self.success:
            return 0.0
        
        return max(0.0, 1.0 - self.error_rate)


class TaskRoutingDecision(Base):
    """
    Records of task routing decisions for analysis and optimization.
    
    Tracks routing decisions to enable learning and improvement of
    the intelligent task routing algorithms.
    """
    
    __tablename__ = "task_routing_decisions"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    task_id = Column(DatabaseAgnosticUUID(), ForeignKey("tasks.id"), nullable=False, index=True)
    selected_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Routing context
    routing_strategy = Column(String(100), nullable=True)
    candidate_agents = Column(JSON, nullable=True, default=list)
    selection_criteria = Column(JSON, nullable=True, default=dict)
    
    # Scoring information
    agent_scores = Column(JSON, nullable=True, default=dict)
    final_score = Column(Float, nullable=True)
    confidence_level = Column(Float, nullable=True)
    
    # Performance tracking
    routing_time_ms = Column(Float, nullable=True)
    decision_factors = Column(JSON, nullable=True, default=dict)
    
    # Outcome tracking
    task_completed = Column(Boolean, nullable=True)
    task_success = Column(Boolean, nullable=True)
    actual_completion_time = Column(Float, nullable=True)
    outcome_score = Column(Float, nullable=True)
    
    # Timestamps
    decided_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    task = relationship("Task")
    selected_agent = relationship("Agent")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "selected_agent_id": str(self.selected_agent_id),
            "routing_strategy": self.routing_strategy,
            "candidate_agents": self.candidate_agents,
            "selection_criteria": self.selection_criteria,
            "agent_scores": self.agent_scores,
            "final_score": self.final_score,
            "confidence_level": self.confidence_level,
            "routing_time_ms": self.routing_time_ms,
            "decision_factors": self.decision_factors,
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "actual_completion_time": self.actual_completion_time,
            "outcome_score": self.outcome_score,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "outcome_recorded_at": self.outcome_recorded_at.isoformat() if self.outcome_recorded_at else None
        }
    
    def calculate_routing_accuracy(self) -> Optional[float]:
        """Calculate accuracy of the routing decision."""
        if self.task_success is None:
            return None
        
        # Simple accuracy based on task success
        base_accuracy = 1.0 if self.task_success else 0.0
        
        # Adjust based on confidence level
        if self.confidence_level:
            if self.task_success and self.confidence_level > 0.8:
                return min(1.0, base_accuracy + 0.1)  # Bonus for high confidence success
            elif not self.task_success and self.confidence_level < 0.5:
                return base_accuracy + 0.1  # Less penalty for low confidence failure
        
        return base_accuracy


class AgentCapabilityScore(Base):
    """
    Dynamic capability scores for agents based on performance history.
    
    Tracks evolving capability scores that improve over time based on
    actual performance in different task types and capability areas.
    """
    
    __tablename__ = "agent_capability_scores"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Capability identification
    capability_name = Column(String(255), nullable=False, index=True)
    task_type = Column(String(100), nullable=True, index=True)
    
    # Scoring metrics
    base_score = Column(Float, nullable=False, default=0.5)
    experience_factor = Column(Float, nullable=False, default=0.0)
    recent_performance = Column(Float, nullable=False, default=0.5)
    confidence_level = Column(Float, nullable=False, default=0.5)
    
    # Statistical data
    total_tasks = Column(Integer, nullable=False, default=0)
    successful_tasks = Column(Integer, nullable=False, default=0)
    average_completion_time = Column(Float, nullable=True)
    
    # Trending information
    trend_direction = Column(String(20), nullable=True)  # 'improving', 'declining', 'stable'
    trend_strength = Column(Float, nullable=True)
    
    # Timestamps
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "capability_name": self.capability_name,
            "task_type": self.task_type,
            "base_score": self.base_score,
            "experience_factor": self.experience_factor,
            "recent_performance": self.recent_performance,
            "confidence_level": self.confidence_level,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "average_completion_time": self.average_completion_time,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def calculate_composite_score(self) -> float:
        """Calculate composite capability score."""
        # Weighted combination of different factors
        weights = {
            "base": 0.4,
            "experience": 0.3,
            "recent": 0.2,
            "confidence": 0.1
        }
        
        composite = (
            self.base_score * weights["base"] +
            self.experience_factor * weights["experience"] +
            self.recent_performance * weights["recent"] +
            self.confidence_level * weights["confidence"]
        )
        
        return min(1.0, max(0.0, composite))
    
    def update_with_task_result(self, success: bool, completion_time: Optional[float] = None) -> None:
        """Update capability score with new task result."""
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        
        # Update recent performance with exponential moving average
        new_performance = 1.0 if success else 0.0
        alpha = 0.3  # Learning rate
        self.recent_performance = (1 - alpha) * self.recent_performance + alpha * new_performance
        
        # Update experience factor based on total tasks
        self.experience_factor = min(1.0, self.total_tasks / 50.0)  # Max experience at 50 tasks
        
        # Update base score
        if self.total_tasks > 0:
            self.base_score = self.successful_tasks / self.total_tasks
        
        # Update average completion time
        if completion_time and self.average_completion_time:
            self.average_completion_time = (
                (self.average_completion_time * (self.total_tasks - 1) + completion_time) 
                / self.total_tasks
            )
        elif completion_time:
            self.average_completion_time = completion_time
        
        # Update confidence based on consistency
        if self.total_tasks >= 5:
            # Higher confidence with more consistent results
            success_rate = self.successful_tasks / self.total_tasks
            variance = abs(success_rate - 0.5) * 2  # 0 to 1 scale
            self.confidence_level = min(1.0, 0.5 + variance * 0.5)
        
        self.last_updated = datetime.utcnow()


class WorkloadSnapshot(Base):
    """
    Point-in-time snapshots of agent workloads for load balancing analysis.
    
    Tracks workload distribution over time to optimize load balancing
    algorithms and identify performance bottlenecks.
    """
    
    __tablename__ = "workload_snapshots"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Workload metrics
    active_tasks = Column(Integer, nullable=False, default=0)
    pending_tasks = Column(Integer, nullable=False, default=0)
    context_usage_percent = Column(Float, nullable=False, default=0.0)
    
    # Resource utilization
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    estimated_capacity = Column(Float, nullable=False, default=1.0)
    utilization_ratio = Column(Float, nullable=False, default=0.0)
    
    # Task distribution
    priority_distribution = Column(JSON, nullable=True, default=dict)
    task_type_distribution = Column(JSON, nullable=True, default=dict)
    
    # Performance indicators
    average_response_time_ms = Column(Float, nullable=True)
    throughput_tasks_per_hour = Column(Float, nullable=True)
    error_rate_percent = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    snapshot_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    agent = relationship("Agent")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "active_tasks": self.active_tasks,
            "pending_tasks": self.pending_tasks,
            "context_usage_percent": self.context_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "estimated_capacity": self.estimated_capacity,
            "utilization_ratio": self.utilization_ratio,
            "priority_distribution": self.priority_distribution,
            "task_type_distribution": self.task_type_distribution,
            "average_response_time_ms": self.average_response_time_ms,
            "throughput_tasks_per_hour": self.throughput_tasks_per_hour,
            "error_rate_percent": self.error_rate_percent,
            "snapshot_time": self.snapshot_time.isoformat() if self.snapshot_time else None
        }
    
    def calculate_load_factor(self) -> float:
        """Calculate overall load factor for this agent."""
        # Weighted combination of different load indicators
        task_load = min(1.0, (self.active_tasks + self.pending_tasks) / 5.0)  # Assume max 5 concurrent tasks
        context_load = self.context_usage_percent / 100.0
        resource_load = max(
            (self.memory_usage_mb or 0) / 1000.0,  # Assume 1GB as full load
            (self.cpu_usage_percent or 0) / 100.0
        )
        
        # Weighted average
        return (task_load * 0.4 + context_load * 0.3 + resource_load * 0.3)
    
    def is_overloaded(self, threshold: float = 0.85) -> bool:
        """Check if agent is overloaded based on threshold."""
        return self.calculate_load_factor() > threshold
    
    def is_underloaded(self, threshold: float = 0.3) -> bool:
        """Check if agent is underloaded based on threshold."""
        return self.calculate_load_factor() < threshold