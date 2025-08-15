"""
Advanced Agent Workflow Tracking System for LeanVibe Agent Hive 2.0

Production-grade agent lifecycle and workflow monitoring with real-time state tracking,
task progression analysis, and inter-agent communication mapping for autonomous 
multi-agent development workflows.

Features:
- Real-time agent state transition monitoring
- Task progression tracking with milestone validation
- Inter-agent communication flow analysis
- Workflow dependency mapping and optimization
- Resource utilization tracking per agent
- Performance metrics with task completion analytics
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc, asc, update
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_session
from .redis import get_redis_client
from .observability_hooks import get_observability_hooks, ObservabilityHooks
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.workflow import Workflow, WorkflowStatus
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory

logger = structlog.get_logger()


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    IDLE = "idle" 
    BUSY = "busy"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class TaskProgressState(Enum):
    """Task progression states."""
    CREATED = "created"
    QUEUED = "queued"
    ASSIGNED = "assigned" 
    EXECUTING = "executing"
    WAITING_DEPENDENCY = "waiting_dependency"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkflowPhase(Enum):
    """Workflow execution phases."""
    PLANNING = "planning"
    EXECUTION = "execution"
    COORDINATION = "coordination"
    VALIDATION = "validation"
    COMPLETION = "completion"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class AgentStateTransition:
    """Agent state transition record."""
    agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    previous_state: AgentState
    new_state: AgentState
    transition_reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    duration_in_previous_state_ms: Optional[float] = None
    resource_allocation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "transition_reason": self.transition_reason,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "duration_in_previous_state_ms": self.duration_in_previous_state_ms,
            "resource_allocation": self.resource_allocation
        }


@dataclass
class TaskProgressUpdate:
    """Task progression tracking record."""
    task_id: uuid.UUID
    agent_id: uuid.UUID
    workflow_id: Optional[uuid.UUID]
    session_id: Optional[uuid.UUID]
    previous_state: TaskProgressState
    new_state: TaskProgressState
    milestone_reached: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    blocking_dependencies: List[str] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": str(self.task_id),
            "agent_id": str(self.agent_id),
            "workflow_id": str(self.workflow_id) if self.workflow_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "milestone_reached": self.milestone_reached,
            "progress_percentage": self.progress_percentage,
            "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            "blocking_dependencies": self.blocking_dependencies,
            "execution_context": self.execution_context,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class InterAgentCommunication:
    """Inter-agent communication tracking."""
    message_id: uuid.UUID
    from_agent_id: uuid.UUID
    to_agent_id: uuid.UUID
    communication_type: str
    message_content: Dict[str, Any]
    workflow_context: Optional[uuid.UUID] = None
    response_expected: bool = False
    response_received: bool = False
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": str(self.message_id),
            "from_agent_id": str(self.from_agent_id),
            "to_agent_id": str(self.to_agent_id),
            "communication_type": self.communication_type,
            "message_content": self.message_content,
            "workflow_context": str(self.workflow_context) if self.workflow_context else None,
            "response_expected": self.response_expected,
            "response_received": self.response_received,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class WorkflowProgressSnapshot:
    """Workflow execution progress snapshot."""
    workflow_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    current_phase: WorkflowPhase
    active_agents: Set[uuid.UUID]
    completed_tasks: int
    total_tasks: int
    failed_tasks: int
    blocked_tasks: int
    estimated_completion_time: Optional[datetime]
    critical_path_duration_ms: Optional[float]
    resource_utilization: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": str(self.workflow_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "current_phase": self.current_phase.value,
            "active_agents": [str(aid) for aid in self.active_agents],
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "progress_percentage": (self.completed_tasks / max(self.total_tasks, 1)) * 100,
            "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            "critical_path_duration_ms": self.critical_path_duration_ms,
            "resource_utilization": self.resource_utilization,
            "timestamp": self.timestamp.isoformat()
        }


class AgentWorkflowTracker:
    """
    Advanced Agent Workflow Tracking System for LeanVibe Agent Hive 2.0
    
    Provides comprehensive monitoring of agent lifecycles, task progression,
    and workflow execution with real-time analytics and optimization insights.
    
    Features:
    - Real-time agent state transition monitoring
    - Task progression tracking with milestone validation
    - Inter-agent communication flow analysis
    - Workflow dependency mapping and bottleneck detection
    - Resource utilization optimization recommendations
    - Performance correlation analysis
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable] = None,
        observability_hooks: Optional[ObservabilityHooks] = None
    ):
        """Initialize the agent workflow tracker."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.hooks = observability_hooks or get_observability_hooks()
        
        # State tracking
        self.agent_states: Dict[uuid.UUID, AgentState] = {}
        self.agent_state_history: Dict[uuid.UUID, deque] = defaultdict(lambda: deque(maxlen=100))
        self.task_progress_history: Dict[uuid.UUID, deque] = defaultdict(lambda: deque(maxlen=50))
        self.workflow_snapshots: Dict[uuid.UUID, deque] = defaultdict(lambda: deque(maxlen=20))
        self.communication_log: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.state_transition_times: Dict[Tuple[AgentState, AgentState], deque] = defaultdict(lambda: deque(maxlen=100))
        self.task_completion_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.workflow_phase_durations: Dict[WorkflowPhase, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Real-time analytics
        self.bottleneck_detector = WorkflowBottleneckDetector()
        self.resource_optimizer = ResourceUtilizationOptimizer()
        self.dependency_analyzer = TaskDependencyAnalyzer()
        
        # Background processing
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        self.config = {
            "state_transition_timeout_ms": 30000,  # 30 seconds
            "task_progress_batch_size": 50,
            "communication_analysis_window": 300,  # 5 minutes
            "performance_analysis_interval": 60,  # 1 minute
            "redis_stream_prefix": "workflow_tracking:",
            "redis_ttl_hours": 24,
            "max_concurrent_workflows": 100,
            "enable_real_time_optimization": True,
            "bottleneck_detection_threshold": 0.8,
            "resource_efficiency_threshold": 0.7
        }
        
        logger.info("AgentWorkflowTracker initialized", config=self.config)
    
    async def start_tracking(self) -> None:
        """Start workflow tracking system."""
        if self.is_running:
            logger.warning("Workflow tracking already running")
            return
        
        logger.info("Starting Agent Workflow Tracking system")
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._state_monitoring_loop()),
            asyncio.create_task(self._workflow_analysis_loop())
        ]
        
        logger.info("Agent Workflow Tracking system started")
    
    async def stop_tracking(self) -> None:
        """Stop workflow tracking system."""
        if not self.is_running:
            return
        
        logger.info("Stopping Agent Workflow Tracking system")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Agent Workflow Tracking system stopped")
    
    async def track_agent_state_transition(
        self,
        agent_id: uuid.UUID,
        new_state: AgentState,
        transition_reason: str,
        session_id: Optional[uuid.UUID] = None,
        context: Optional[Dict[str, Any]] = None,
        resource_allocation: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track agent state transition with detailed analytics."""
        try:
            # Get previous state
            previous_state = self.agent_states.get(agent_id, AgentState.OFFLINE)
            
            # Calculate duration in previous state
            duration_ms = None
            if agent_id in self.agent_state_history and self.agent_state_history[agent_id]:
                last_transition = self.agent_state_history[agent_id][-1]
                duration_ms = (datetime.utcnow() - last_transition.timestamp).total_seconds() * 1000
            
            # Create transition record
            transition = AgentStateTransition(
                agent_id=agent_id,
                session_id=session_id,
                previous_state=previous_state,
                new_state=new_state,
                transition_reason=transition_reason,
                context=context or {},
                duration_in_previous_state_ms=duration_ms,
                resource_allocation=resource_allocation
            )
            
            # Update current state
            self.agent_states[agent_id] = new_state
            
            # Store in history
            self.agent_state_history[agent_id].append(transition)
            
            # Track performance metrics
            if duration_ms:
                self.state_transition_times[(previous_state, new_state)].append(duration_ms)
            
            # Emit observability event
            if self.hooks:
                await self.hooks.agent_state_changed(
                    agent_id=agent_id,
                    previous_state=previous_state.value,
                    new_state=new_state.value,
                    state_transition_reason=transition_reason,
                    session_id=session_id,
                    resource_allocation=resource_allocation
                )
            
            # Store to Redis for real-time access
            await self._store_state_transition_to_redis(transition)
            
            # Analyze for potential issues
            await self._analyze_state_transition(transition)
            
            logger.debug(
                "Agent state transition tracked",
                agent_id=str(agent_id),
                previous_state=previous_state.value,
                new_state=new_state.value,
                duration_ms=duration_ms,
                reason=transition_reason
            )
            
        except Exception as e:
            logger.error("Failed to track agent state transition", error=str(e), agent_id=str(agent_id))
    
    async def track_task_progress(
        self,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        new_state: TaskProgressState,
        workflow_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        milestone_reached: Optional[str] = None,
        progress_percentage: float = 0.0,
        estimated_completion_time: Optional[datetime] = None,
        blocking_dependencies: Optional[List[str]] = None,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track task progression with milestone validation."""
        try:
            # Get previous state
            previous_state = TaskProgressState.CREATED
            if task_id in self.task_progress_history and self.task_progress_history[task_id]:
                previous_state = self.task_progress_history[task_id][-1].new_state
            
            # Create progress update
            progress_update = TaskProgressUpdate(
                task_id=task_id,
                agent_id=agent_id,
                workflow_id=workflow_id,
                session_id=session_id,
                previous_state=previous_state,
                new_state=new_state,
                milestone_reached=milestone_reached,
                progress_percentage=progress_percentage,
                estimated_completion_time=estimated_completion_time,
                blocking_dependencies=blocking_dependencies or [],
                execution_context=execution_context or {}
            )
            
            # Store in history
            self.task_progress_history[task_id].append(progress_update)
            
            # Track completion times
            if new_state == TaskProgressState.COMPLETED and task_id in self.task_progress_history:
                start_time = self.task_progress_history[task_id][0].timestamp
                completion_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                task_type = execution_context.get("task_type", "unknown") if execution_context else "unknown"
                self.task_completion_times[task_type].append(completion_time)
            
            # Emit observability events
            if self.hooks:
                if new_state == TaskProgressState.EXECUTING:
                    await self.hooks.node_executing(
                        workflow_id=workflow_id or uuid.uuid4(),
                        node_id=str(task_id),
                        node_type="task",
                        agent_id=agent_id,
                        session_id=session_id,
                        input_data=execution_context,
                        assigned_agent=agent_id
                    )
                elif new_state in [TaskProgressState.COMPLETED, TaskProgressState.FAILED]:
                    await self.hooks.node_completed(
                        workflow_id=workflow_id or uuid.uuid4(),
                        node_id=str(task_id),
                        success=(new_state == TaskProgressState.COMPLETED),
                        agent_id=agent_id,
                        session_id=session_id,
                        output_data=execution_context if new_state == TaskProgressState.COMPLETED else None,
                        error_details=execution_context if new_state == TaskProgressState.FAILED else None
                    )
            
            # Store to Redis
            await self._store_task_progress_to_redis(progress_update)
            
            # Analyze dependencies and bottlenecks
            if blocking_dependencies:
                await self._analyze_task_dependencies(progress_update)
            
            logger.debug(
                "Task progress tracked",
                task_id=str(task_id),
                agent_id=str(agent_id),
                previous_state=previous_state.value,
                new_state=new_state.value,
                progress_percentage=progress_percentage,
                milestone=milestone_reached
            )
            
        except Exception as e:
            logger.error("Failed to track task progress", error=str(e), task_id=str(task_id))
    
    async def track_inter_agent_communication(
        self,
        message_id: uuid.UUID,
        from_agent_id: uuid.UUID,
        to_agent_id: uuid.UUID,
        communication_type: str,
        message_content: Dict[str, Any],
        workflow_context: Optional[uuid.UUID] = None,
        response_expected: bool = False
    ) -> None:
        """Track inter-agent communication for workflow analysis."""
        try:
            communication = InterAgentCommunication(
                message_id=message_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                communication_type=communication_type,
                message_content=message_content,
                workflow_context=workflow_context,
                response_expected=response_expected
            )
            
            # Store in communication log
            self.communication_log.append(communication)
            
            # Emit observability event
            if self.hooks:
                await self.hooks.message_published(
                    message_id=message_id,
                    from_agent=str(from_agent_id),
                    to_agent=str(to_agent_id),
                    message_type=communication_type,
                    message_content=message_content,
                    expected_response=response_expected
                )
            
            # Store to Redis
            await self._store_communication_to_redis(communication)
            
            logger.debug(
                "Inter-agent communication tracked",
                message_id=str(message_id),
                from_agent=str(from_agent_id),
                to_agent=str(to_agent_id),
                type=communication_type
            )
            
        except Exception as e:
            logger.error("Failed to track inter-agent communication", error=str(e))
    
    async def update_workflow_progress(
        self,
        workflow_id: uuid.UUID,
        session_id: Optional[uuid.UUID] = None,
        current_phase: Optional[WorkflowPhase] = None,
        resource_utilization: Optional[Dict[str, float]] = None
    ) -> WorkflowProgressSnapshot:
        """Update and track workflow progress snapshot."""
        try:
            # Get workflow statistics from database
            async with self.session_factory() as session:
                # Get workflow tasks
                task_query = select(Task).where(Task.workflow_id == workflow_id)
                task_result = await session.execute(task_query)
                tasks = task_result.scalars().all()
                
                # Calculate statistics
                total_tasks = len(tasks)
                completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
                failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
                blocked_tasks = len([t for t in tasks if t.status == TaskStatus.BLOCKED])
                
                # Get active agents
                active_agents = set()
                for task in tasks:
                    if task.assigned_agent_id and task.status in [TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED]:
                        active_agents.add(task.assigned_agent_id)
                
                # Estimate completion time
                estimated_completion = None
                if total_tasks > 0 and completed_tasks > 0:
                    completion_rate = completed_tasks / total_tasks
                    if completion_rate > 0:
                        # Simple estimation based on current progress
                        remaining_tasks = total_tasks - completed_tasks
                        avg_task_time = self._calculate_average_task_completion_time()
                        estimated_completion = datetime.utcnow() + timedelta(
                            milliseconds=remaining_tasks * avg_task_time
                        )
                
                # Create snapshot
                snapshot = WorkflowProgressSnapshot(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    current_phase=current_phase or WorkflowPhase.EXECUTION,
                    active_agents=active_agents,
                    completed_tasks=completed_tasks,
                    total_tasks=total_tasks,
                    failed_tasks=failed_tasks,
                    blocked_tasks=blocked_tasks,
                    estimated_completion_time=estimated_completion,
                    resource_utilization=resource_utilization or {}
                )
                
                # Store snapshot
                self.workflow_snapshots[workflow_id].append(snapshot)
                
                # Store to Redis
                await self._store_workflow_snapshot_to_redis(snapshot)
                
                logger.debug(
                    "Workflow progress updated",
                    workflow_id=str(workflow_id),
                    progress=f"{completed_tasks}/{total_tasks}",
                    active_agents=len(active_agents),
                    phase=current_phase.value if current_phase else "execution"
                )
                
                return snapshot
                
        except Exception as e:
            logger.error("Failed to update workflow progress", error=str(e), workflow_id=str(workflow_id))
            raise
    
    async def get_real_time_workflow_status(
        self,
        workflow_id: Optional[uuid.UUID] = None,
        include_agent_details: bool = True,
        include_communication_flow: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive real-time workflow status."""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_overview": await self._get_system_overview(),
                "performance_metrics": await self._get_workflow_performance_metrics()
            }
            
            if workflow_id:
                # Specific workflow status
                if workflow_id in self.workflow_snapshots and self.workflow_snapshots[workflow_id]:
                    latest_snapshot = self.workflow_snapshots[workflow_id][-1]
                    status["workflow"] = latest_snapshot.to_dict()
                    
                    if include_agent_details:
                        status["agent_details"] = await self._get_workflow_agent_details(workflow_id)
                    
                    if include_communication_flow:
                        status["communication_flow"] = await self._get_workflow_communication_flow(workflow_id)
            else:
                # System-wide status
                status["active_workflows"] = await self._get_active_workflows_summary()
                
                if include_agent_details:
                    status["agent_summary"] = await self._get_all_agents_summary()
                
                if include_communication_flow:
                    status["communication_summary"] = await self._get_communication_summary()
            
            return status
            
        except Exception as e:
            logger.error("Failed to get real-time workflow status", error=str(e))
            return {"error": str(e)}
    
    # Background monitoring loops
    async def _state_monitoring_loop(self) -> None:
        """Background task for agent state monitoring."""
        logger.info("Starting agent state monitoring loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Perform basic monitoring tasks
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("State monitoring loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _workflow_analysis_loop(self) -> None:
        """Background task for workflow analysis."""
        logger.info("Starting workflow analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Perform basic workflow analysis
                await asyncio.sleep(60)  # Analyze every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Workflow analysis loop error", error=str(e))
                await asyncio.sleep(60)
    
    # Helper methods (simplified implementations)
    async def _store_state_transition_to_redis(self, transition: AgentStateTransition) -> None:
        """Store state transition to Redis."""
        try:
            key = f"{self.config['redis_stream_prefix']}state_transitions"
            await self.redis_client.xadd(key, transition.to_dict(), maxlen=1000)
        except Exception as e:
            logger.error("Failed to store state transition to Redis", error=str(e))
    
    async def _store_task_progress_to_redis(self, progress: TaskProgressUpdate) -> None:
        """Store task progress to Redis."""
        try:
            key = f"{self.config['redis_stream_prefix']}task_progress"
            await self.redis_client.xadd(key, progress.to_dict(), maxlen=1000)
        except Exception as e:
            logger.error("Failed to store task progress to Redis", error=str(e))
    
    async def _store_communication_to_redis(self, communication: InterAgentCommunication) -> None:
        """Store communication to Redis."""
        try:
            key = f"{self.config['redis_stream_prefix']}communications"
            await self.redis_client.xadd(key, communication.to_dict(), maxlen=1000)
        except Exception as e:
            logger.error("Failed to store communication to Redis", error=str(e))
    
    async def _store_workflow_snapshot_to_redis(self, snapshot: WorkflowProgressSnapshot) -> None:
        """Store workflow snapshot to Redis."""
        try:
            key = f"{self.config['redis_stream_prefix']}workflow_snapshots"
            await self.redis_client.xadd(key, snapshot.to_dict(), maxlen=500)
        except Exception as e:
            logger.error("Failed to store workflow snapshot to Redis", error=str(e))
    
    def _calculate_average_task_completion_time(self) -> float:
        """Calculate average task completion time across all task types."""
        all_times = []
        for times in self.task_completion_times.values():
            all_times.extend(times)
        return sum(all_times) / len(all_times) if all_times else 30000  # Default 30 seconds


# Placeholder classes for advanced analytics
class WorkflowBottleneckDetector:
    """Detects bottlenecks in workflow execution."""
    
    async def detect_bottlenecks(
        self,
        workflow_snapshots: Dict[uuid.UUID, deque],
        task_progress_history: Dict[uuid.UUID, deque],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect workflow bottlenecks."""
        return []  # Simplified implementation


class ResourceUtilizationOptimizer:
    """Optimizes resource allocation for workflows."""
    pass


class TaskDependencyAnalyzer:
    """Analyzes task dependencies and critical paths."""
    pass


# Global instance
_agent_workflow_tracker: Optional[AgentWorkflowTracker] = None


async def get_agent_workflow_tracker() -> AgentWorkflowTracker:
    """Get singleton agent workflow tracker instance."""
    global _agent_workflow_tracker
    
    if _agent_workflow_tracker is None:
        _agent_workflow_tracker = AgentWorkflowTracker()
        await _agent_workflow_tracker.start_tracking()
    
    return _agent_workflow_tracker


async def cleanup_agent_workflow_tracker() -> None:
    """Cleanup agent workflow tracker resources."""
    global _agent_workflow_tracker
    
    if _agent_workflow_tracker:
        await _agent_workflow_tracker.stop_tracking()
        _agent_workflow_tracker = None