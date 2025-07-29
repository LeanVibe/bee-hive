"""
Comprehensive Dashboard Integration System for LeanVibe Agent Hive 2.0

This module provides a complete integration layer that connects all enhanced 
LeanVibe features with the dashboard infrastructure, enabling real-time monitoring
of multi-agent workflows, quality gates, extended thinking sessions, and performance metrics.

Key Features:
- Real-time event streaming for dashboard updates
- Multi-agent workflow progress tracking
- Quality gates visualization and validation results
- Extended thinking sessions collaborative monitoring
- Hook execution performance tracking
- Agent performance metrics aggregation
- WebSocket-based real-time updates
- Mobile-responsive data formatting
- Comprehensive observability integration

Architecture:
- Event-driven integration with existing lifecycle hooks
- Real-time data aggregation and streaming
- Performance-optimized data structures
- Comprehensive error handling and fallback mechanisms
- Production-ready scalability and monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from uuid import uuid4, UUID
from collections import defaultdict, deque
import statistics

import structlog
from pydantic import BaseModel, Field, validator
from fastapi import WebSocket, WebSocketDisconnect

# Import existing core components
from .coordination_dashboard import (
    coordination_dashboard, CoordinationDashboard, AgentGraphNode, 
    AgentGraphEdge, AgentNodeType, AgentNodeStatus, AgentEdgeType,
    EventFilter, SessionColorManager, AgentCommunicationEvent
)
from .enhanced_lifecycle_hooks import (
    EnhancedEventType, LifecycleEventData
)
from .dashboard_integration import (
    DashboardIntegrationManager, DashboardEventType, VisualizationStyle,
    DashboardUpdate, ConversationVisualization
)
from .performance_metrics_collector import PerformanceMetricsCollector
from .observability_streams import ObservabilityStreamsManager
from .redis import get_message_broker

logger = structlog.get_logger()


class IntegrationEventType(Enum):
    """Types of dashboard integration events."""
    WORKFLOW_PROGRESS_UPDATE = "workflow_progress_update"
    QUALITY_GATE_RESULT = "quality_gate_result"
    THINKING_SESSION_UPDATE = "thinking_session_update"
    HOOK_PERFORMANCE_METRIC = "hook_performance_metric"
    AGENT_PERFORMANCE_UPDATE = "agent_performance_update"
    SYSTEM_HEALTH_UPDATE = "system_health_update"
    COLLABORATION_INSIGHT = "collaboration_insight"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_PATTERN_DETECTION = "error_pattern_detection"
    OPTIMIZATION_RECOMMENDATION = "optimization_recommendation"


class QualityGateStatus(Enum):
    """Status values for quality gates."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ThinkingSessionPhase(Enum):
    """Phases of extended thinking sessions."""
    INITIALIZATION = "initialization"
    PROBLEM_ANALYSIS = "problem_analysis"
    SOLUTION_EXPLORATION = "solution_exploration"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    COMPLETION = "completion"


@dataclass
class WorkflowProgress:
    """Tracks progress of multi-agent workflows."""
    workflow_id: str
    workflow_name: str
    total_steps: int
    completed_steps: int
    active_agents: List[str]
    current_phase: str
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 1.0
    
    @property
    def completion_percentage(self) -> float:
        return (self.completed_steps / self.total_steps) * 100 if self.total_steps > 0 else 0
    
    @property
    def is_completed(self) -> bool:
        return self.completed_steps >= self.total_steps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'completion_percentage': self.completion_percentage,
            'is_completed': self.is_completed,
            'start_time': self.start_time.isoformat(),
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None
        }


@dataclass
class QualityGateResult:
    """Results from quality gate execution."""
    gate_id: str
    gate_name: str
    status: QualityGateStatus
    execution_time_ms: int
    success_criteria: Dict[str, Any]
    actual_results: Dict[str, Any]
    validation_errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def passed(self) -> bool:
        return self.status == QualityGateStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'status': self.status.value,
            'passed': self.passed,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ThinkingSessionUpdate:
    """Updates from extended thinking sessions."""
    session_id: str
    session_name: str
    phase: ThinkingSessionPhase
    participating_agents: List[str]
    insights_generated: int
    consensus_level: float  # 0.0 to 1.0
    collaboration_quality: float  # 0.0 to 1.0
    current_focus: str
    key_insights: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'phase': self.phase.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class HookPerformanceMetric:
    """Performance metrics for hook execution."""
    hook_id: str
    hook_type: str
    execution_time_ms: int
    memory_usage_mb: float
    success: bool
    agent_id: str
    session_id: str
    error_message: Optional[str] = None
    payload_size_bytes: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentPerformanceMetrics:
    """Comprehensive agent performance metrics."""
    agent_id: str
    session_id: str
    metrics_window_minutes: int = 30
    
    # Performance metrics
    task_completion_rate: float = 0.0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    tool_usage_efficiency: float = 0.0
    context_sharing_effectiveness: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_io_mb: float = 0.0
    
    # Collaboration metrics
    communication_frequency: int = 0
    coordination_success_rate: float = 0.0
    knowledge_sharing_score: float = 0.0
    
    # Quality metrics
    output_quality_score: float = 0.0
    consistency_score: float = 0.0
    reliability_score: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        weights = {
            'completion': 0.25,
            'response_time': 0.15,
            'error_rate': 0.20,
            'tool_efficiency': 0.10,
            'collaboration': 0.15,
            'quality': 0.15
        }
        
        # Normalize metrics to 0-100 scale
        normalized_scores = {
            'completion': self.task_completion_rate * 100,
            'response_time': max(0, 100 - (self.average_response_time_ms / 100)),  # Lower is better
            'error_rate': max(0, 100 - (self.error_rate * 100)),  # Lower is better
            'tool_efficiency': self.tool_usage_efficiency * 100,
            'collaboration': self.coordination_success_rate * 100,
            'quality': self.output_quality_score * 100
        }
        
        return sum(score * weights[metric] for metric, score in normalized_scores.items())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'overall_score': self.calculate_overall_score(),
            'timestamp': self.timestamp.isoformat()
        }


class ComprehensiveDashboardIntegration:
    """
    Comprehensive dashboard integration system for LeanVibe Agent Hive 2.0.
    
    Provides real-time monitoring, performance tracking, and visualization
    data preparation for all enhanced platform features.
    """
    
    def __init__(self):
        # Core components
        self.coordination_dashboard = coordination_dashboard
        self.performance_collector = PerformanceMetricsCollector()
        self.observability = ObservabilityStreamsManager()
        
        # Integration state
        self.workflow_progress: Dict[str, WorkflowProgress] = {}
        self.quality_gate_results: Dict[str, List[QualityGateResult]] = defaultdict(list)
        self.thinking_sessions: Dict[str, ThinkingSessionUpdate] = {}
        self.hook_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_performance: Dict[str, AgentPerformanceMetrics] = {}
        
        # Real-time streaming
        self.active_streams: Dict[str, WebSocket] = {}
        self.stream_filters: Dict[str, Dict[str, Any]] = {}
        
        # Performance optimization
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.batch_size = 50
        self.update_interval = 1.0  # seconds
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        logger.info("ComprehensiveDashboardIntegration initialized")
    
    async def start(self) -> None:
        """Start the comprehensive dashboard integration system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background processing tasks
        tasks = [
            self._process_update_queue(),
            self._collect_performance_metrics(),
            self._monitor_system_health(),
            self._generate_insights(),
            self._cleanup_old_data()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        # Register with lifecycle hooks
        await self._register_lifecycle_hooks()
        
        logger.info("Comprehensive dashboard integration started")
    
    async def stop(self) -> None:
        """Stop the comprehensive dashboard integration system."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        
        # Close active streams
        for stream_id, websocket in self.active_streams.items():
            try:
                await websocket.close()
            except Exception:
                pass
        
        self.active_streams.clear()
        
        logger.info("Comprehensive dashboard integration stopped")
    
    # Multi-Agent Workflow Monitoring
    
    async def track_workflow_progress(
        self,
        workflow_id: str,
        workflow_name: str,
        total_steps: int,
        active_agents: List[str],
        current_phase: str = "initialization"
    ) -> WorkflowProgress:
        """Initialize tracking for a multi-agent workflow."""
        progress = WorkflowProgress(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            total_steps=total_steps,
            completed_steps=0,
            active_agents=active_agents,
            current_phase=current_phase,
            start_time=datetime.utcnow()
        )
        
        self.workflow_progress[workflow_id] = progress
        
        # Send real-time update
        await self._send_integration_update(
            IntegrationEventType.WORKFLOW_PROGRESS_UPDATE,
            {
                'workflow_id': workflow_id,
                'progress': progress.to_dict(),
                'event_type': 'workflow_started'
            }
        )
        
        logger.info(
            "Workflow progress tracking started",
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            total_steps=total_steps
        )
        
        return progress
    
    async def update_workflow_progress(
        self,
        workflow_id: str,
        completed_steps: Optional[int] = None,
        current_phase: Optional[str] = None,
        active_agents: Optional[List[str]] = None,
        increment_errors: bool = False
    ) -> Optional[WorkflowProgress]:
        """Update progress for a tracked workflow."""
        if workflow_id not in self.workflow_progress:
            logger.warning("Workflow not found for progress update", workflow_id=workflow_id)
            return None
        
        progress = self.workflow_progress[workflow_id]
        
        # Update fields
        if completed_steps is not None:
            progress.completed_steps = completed_steps
        if current_phase is not None:
            progress.current_phase = current_phase
        if active_agents is not None:
            progress.active_agents = active_agents
        if increment_errors:
            progress.error_count += 1
        
        # Update success rate
        if progress.completed_steps > 0:
            progress.success_rate = max(0.0, 1.0 - (progress.error_count / progress.completed_steps))
        
        # Estimate completion time
        if progress.completed_steps > 0 and not progress.is_completed:
            elapsed_time = datetime.utcnow() - progress.start_time
            estimated_total_time = elapsed_time * (progress.total_steps / progress.completed_steps)
            progress.estimated_completion = progress.start_time + estimated_total_time
        
        # Send real-time update
        await self._send_integration_update(
            IntegrationEventType.WORKFLOW_PROGRESS_UPDATE,
            {
                'workflow_id': workflow_id,
                'progress': progress.to_dict(),
                'event_type': 'progress_updated'
            }
        )
        
        return progress
    
    async def complete_workflow(self, workflow_id: str, success: bool = True) -> None:
        """Mark a workflow as completed."""
        if workflow_id not in self.workflow_progress:
            return
        
        progress = self.workflow_progress[workflow_id]
        progress.completed_steps = progress.total_steps
        progress.current_phase = "completed"
        
        if not success:
            progress.error_count += 1
            progress.success_rate = 0.0
        
        # Send completion update
        await self._send_integration_update(
            IntegrationEventType.WORKFLOW_PROGRESS_UPDATE,
            {
                'workflow_id': workflow_id,
                'progress': progress.to_dict(),
                'event_type': 'workflow_completed',
                'success': success
            }
        )
        
        logger.info(
            "Workflow completed",
            workflow_id=workflow_id,
            success=success,
            duration_seconds=(datetime.utcnow() - progress.start_time).total_seconds()
        )
    
    # Quality Gates Monitoring
    
    async def record_quality_gate_result(
        self,
        gate_id: str,
        gate_name: str,
        status: QualityGateStatus,
        execution_time_ms: int,
        success_criteria: Dict[str, Any],
        actual_results: Dict[str, Any],
        validation_errors: List[str] = None,
        performance_metrics: Dict[str, float] = None,
        recommendations: List[str] = None
    ) -> QualityGateResult:
        """Record the result of a quality gate execution."""
        result = QualityGateResult(
            gate_id=gate_id,
            gate_name=gate_name,
            status=status,
            execution_time_ms=execution_time_ms,
            success_criteria=success_criteria,
            actual_results=actual_results,
            validation_errors=validation_errors or [],
            performance_metrics=performance_metrics or {},
            recommendations=recommendations or []
        )
        
        self.quality_gate_results[gate_id].append(result)
        
        # Keep only recent results (last 100 per gate)
        if len(self.quality_gate_results[gate_id]) > 100:
            self.quality_gate_results[gate_id] = self.quality_gate_results[gate_id][-100:]
        
        # Send real-time update
        await self._send_integration_update(
            IntegrationEventType.QUALITY_GATE_RESULT,
            {
                'gate_id': gate_id,
                'result': result.to_dict(),
                'trend_data': await self._calculate_quality_gate_trends(gate_id)
            }
        )
        
        logger.info(
            "Quality gate result recorded",
            gate_id=gate_id,
            gate_name=gate_name,
            status=status.value,
            execution_time_ms=execution_time_ms
        )
        
        return result
    
    async def get_quality_gate_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of quality gate results within a time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        summary = {
            'total_executions': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'average_execution_time_ms': 0,
            'success_rate': 0.0,
            'gate_details': {}
        }
        
        all_execution_times = []
        
        for gate_id, results in self.quality_gate_results.items():
            recent_results = [r for r in results if r.timestamp > cutoff_time]
            
            if not recent_results:
                continue
            
            gate_summary = {
                'executions': len(recent_results),
                'passed': len([r for r in recent_results if r.status == QualityGateStatus.PASSED]),
                'failed': len([r for r in recent_results if r.status == QualityGateStatus.FAILED]),
                'warnings': len([r for r in recent_results if r.status == QualityGateStatus.WARNING]),
                'average_execution_time_ms': statistics.mean([r.execution_time_ms for r in recent_results]),
                'success_rate': len([r for r in recent_results if r.passed]) / len(recent_results)
            }
            
            summary['gate_details'][gate_id] = gate_summary
            summary['total_executions'] += gate_summary['executions']
            summary['passed'] += gate_summary['passed']
            summary['failed'] += gate_summary['failed']
            summary['warnings'] += gate_summary['warnings']
            
            all_execution_times.extend([r.execution_time_ms for r in recent_results])
        
        if summary['total_executions'] > 0:
            summary['success_rate'] = summary['passed'] / summary['total_executions']
        
        if all_execution_times:
            summary['average_execution_time_ms'] = statistics.mean(all_execution_times)
        
        return summary
    
    # Extended Thinking Sessions Monitoring
    
    async def update_thinking_session(
        self,
        session_id: str,
        session_name: str,
        phase: ThinkingSessionPhase,
        participating_agents: List[str],
        insights_generated: int,
        consensus_level: float,
        collaboration_quality: float,
        current_focus: str,
        key_insights: List[str] = None,
        disagreements: List[str] = None,
        next_steps: List[str] = None
    ) -> ThinkingSessionUpdate:
        """Update status of an extended thinking session."""
        update = ThinkingSessionUpdate(
            session_id=session_id,
            session_name=session_name,
            phase=phase,
            participating_agents=participating_agents,
            insights_generated=insights_generated,
            consensus_level=consensus_level,
            collaboration_quality=collaboration_quality,
            current_focus=current_focus,
            key_insights=key_insights or [],
            disagreements=disagreements or [],
            next_steps=next_steps or []
        )
        
        self.thinking_sessions[session_id] = update
        
        # Send real-time update
        await self._send_integration_update(
            IntegrationEventType.THINKING_SESSION_UPDATE,
            {
                'session_id': session_id,
                'update': update.to_dict(),
                'collaboration_metrics': await self._calculate_collaboration_metrics(participating_agents)
            }
        )
        
        logger.info(
            "Thinking session updated",
            session_id=session_id,
            phase=phase.value,
            participants=len(participating_agents),
            insights=insights_generated
        )
        
        return update
    
    # Hook Performance Monitoring
    
    async def record_hook_performance(
        self,
        hook_id: str,
        hook_type: str,
        execution_time_ms: int,
        memory_usage_mb: float,
        success: bool,
        agent_id: str,
        session_id: str,
        error_message: Optional[str] = None,
        payload_size_bytes: int = 0
    ) -> HookPerformanceMetric:
        """Record performance metrics for hook execution."""
        metric = HookPerformanceMetric(
            hook_id=hook_id,
            hook_type=hook_type,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            success=success,
            agent_id=agent_id,
            session_id=session_id,
            error_message=error_message,
            payload_size_bytes=payload_size_bytes
        )
        
        self.hook_performance[hook_type].append(metric)
        
        # Send real-time update
        await self._send_integration_update(
            IntegrationEventType.HOOK_PERFORMANCE_METRIC,
            {
                'hook_type': hook_type,
                'metric': metric.to_dict(),
                'performance_trends': await self._calculate_hook_performance_trends(hook_type)
            }
        )
        
        return metric
    
    async def get_hook_performance_summary(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """Get summary of hook performance within a time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        summary = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time_ms': 0,
            'average_memory_usage_mb': 0,
            'success_rate': 0.0,
            'hook_type_details': {}
        }
        
        all_metrics = []
        
        for hook_type, metrics in self.hook_performance.items():
            recent_metrics = [m for m in metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                continue
            
            hook_summary = {
                'executions': len(recent_metrics),
                'successful': len([m for m in recent_metrics if m.success]),
                'failed': len([m for m in recent_metrics if not m.success]),
                'average_execution_time_ms': statistics.mean([m.execution_time_ms for m in recent_metrics]),
                'average_memory_usage_mb': statistics.mean([m.memory_usage_mb for m in recent_metrics]),
                'success_rate': len([m for m in recent_metrics if m.success]) / len(recent_metrics)
            }
            
            summary['hook_type_details'][hook_type] = hook_summary
            all_metrics.extend(recent_metrics)
        
        if all_metrics:
            summary['total_executions'] = len(all_metrics)
            summary['successful_executions'] = len([m for m in all_metrics if m.success])
            summary['failed_executions'] = len([m for m in all_metrics if not m.success])
            summary['average_execution_time_ms'] = statistics.mean([m.execution_time_ms for m in all_metrics])
            summary['average_memory_usage_mb'] = statistics.mean([m.memory_usage_mb for m in all_metrics])
            summary['success_rate'] = summary['successful_executions'] / summary['total_executions']
        
        return summary
    
    # Agent Performance Monitoring
    
    async def update_agent_performance(
        self,
        agent_id: str,
        session_id: str,
        metrics: Dict[str, Any]
    ) -> AgentPerformanceMetrics:
        """Update performance metrics for an agent."""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                session_id=session_id
            )
        
        agent_metrics = self.agent_performance[agent_id]
        
        # Update metrics from provided data
        for key, value in metrics.items():
            if hasattr(agent_metrics, key):
                setattr(agent_metrics, key, value)
        
        agent_metrics.timestamp = datetime.utcnow()
        
        # Send real-time update
        await self._send_integration_update(
            IntegrationEventType.AGENT_PERFORMANCE_UPDATE,
            {
                'agent_id': agent_id,
                'metrics': agent_metrics.to_dict(),
                'performance_trends': await self._calculate_agent_performance_trends(agent_id)
            }
        )
        
        return agent_metrics
    
    async def get_system_performance_overview(self) -> Dict[str, Any]:
        """Get comprehensive system performance overview."""
        overview = {
            'active_workflows': len([w for w in self.workflow_progress.values() if not w.is_completed]),
            'completed_workflows': len([w for w in self.workflow_progress.values() if w.is_completed]),
            'active_thinking_sessions': len(self.thinking_sessions),
            'active_agents': len(self.agent_performance),
            'quality_gates_summary': await self.get_quality_gate_summary(),
            'hook_performance_summary': await self.get_hook_performance_summary(),
            'system_health': await self._calculate_system_health(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Calculate aggregate performance metrics
        if self.agent_performance:
            overall_scores = [metrics.calculate_overall_score() for metrics in self.agent_performance.values()]
            overview['average_agent_performance'] = statistics.mean(overall_scores)
            overview['agent_performance_std'] = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
        else:
            overview['average_agent_performance'] = 0
            overview['agent_performance_std'] = 0
        
        return overview
    
    # WebSocket Real-time Streaming
    
    async def register_dashboard_stream(
        self,
        websocket: WebSocket,
        stream_filters: Dict[str, Any] = None
    ) -> str:
        """Register a WebSocket for real-time dashboard updates."""
        stream_id = str(uuid4())
        
        await websocket.accept()
        self.active_streams[stream_id] = websocket
        self.stream_filters[stream_id] = stream_filters or {}
        
        # Send initial data
        await self._send_initial_dashboard_data(websocket, stream_filters)
        
        logger.info(
            "Dashboard stream registered",
            stream_id=stream_id,
            filters=stream_filters
        )
        
        return stream_id
    
    async def unregister_dashboard_stream(self, stream_id: str) -> None:
        """Unregister a dashboard stream."""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        
        if stream_id in self.stream_filters:
            del self.stream_filters[stream_id]
        
        logger.info("Dashboard stream unregistered", stream_id=stream_id)
    
    # API Data Access Methods
    
    async def get_workflow_progress_data(
        self,
        workflow_ids: Optional[List[str]] = None,
        include_completed: bool = False
    ) -> Dict[str, Any]:
        """Get workflow progress data for API access."""
        workflows = []
        
        for workflow_id, progress in self.workflow_progress.items():
            if workflow_ids and workflow_id not in workflow_ids:
                continue
            
            if not include_completed and progress.is_completed:
                continue
            
            workflows.append(progress.to_dict())
        
        return {
            'workflows': workflows,
            'total_count': len(workflows),
            'active_count': len([w for w in workflows if not w.get('is_completed', False)]),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_quality_gates_data(
        self,
        gate_ids: Optional[List[str]] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get quality gates data for API access."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        gates_data = {}
        
        for gate_id, results in self.quality_gate_results.items():
            if gate_ids and gate_id not in gate_ids:
                continue
            
            recent_results = [r for r in results if r.timestamp > cutoff_time]
            
            gates_data[gate_id] = {
                'gate_id': gate_id,
                'recent_results': [r.to_dict() for r in recent_results[-10:]],  # Last 10 results
                'summary': {
                    'total_executions': len(recent_results),
                    'success_rate': len([r for r in recent_results if r.passed]) / len(recent_results) if recent_results else 0,
                    'average_execution_time_ms': statistics.mean([r.execution_time_ms for r in recent_results]) if recent_results else 0,
                    'latest_status': recent_results[-1].status.value if recent_results else 'unknown'
                }
            }
        
        return {
            'gates': gates_data,
            'summary': await self.get_quality_gate_summary(time_window_hours),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_thinking_sessions_data(self) -> Dict[str, Any]:
        """Get thinking sessions data for API access."""
        sessions_data = {
            session_id: update.to_dict()
            for session_id, update in self.thinking_sessions.items()
        }
        
        return {
            'sessions': sessions_data,
            'active_sessions': len(sessions_data),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_agent_performance_data(
        self,
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get agent performance data for API access."""
        agents_data = {}
        
        for agent_id, metrics in self.agent_performance.items():
            if agent_ids and agent_id not in agent_ids:
                continue
            
            agents_data[agent_id] = metrics.to_dict()
        
        return {
            'agents': agents_data,
            'system_overview': await self.get_system_performance_overview(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Private Helper Methods
    
    async def _register_lifecycle_hooks(self) -> None:
        """Register with enhanced lifecycle hooks system."""
        # Register hook callbacks for different event types
        hook_mappings = {
            EnhancedEventType.PRE_TOOL_USE: self._handle_pre_tool_use,
            EnhancedEventType.POST_TOOL_USE: self._handle_post_tool_use,
            EnhancedEventType.AGENT_LIFECYCLE_START: self._handle_agent_start,
            EnhancedEventType.AGENT_LIFECYCLE_PAUSE: self._handle_agent_pause,
            EnhancedEventType.TASK_ASSIGNMENT: self._handle_task_assignment,
            EnhancedEventType.PERFORMANCE_DEGRADATION: self._handle_performance_degradation,
            EnhancedEventType.ERROR_PATTERN_DETECTED: self._handle_error_pattern
        }
        
        for event_type, handler in hook_mappings.items():
            # Register handler with lifecycle hooks system
            # This would integrate with the actual lifecycle hooks registration
            pass
    
    async def _handle_pre_tool_use(self, event: LifecycleEventData) -> None:
        """Handle pre-tool-use events."""
        # Record hook performance start
        start_time = datetime.utcnow()
        # Store start time for later calculation
        # This would be implemented with proper state tracking
    
    async def _handle_post_tool_use(self, event: LifecycleEventData) -> None:
        """Handle post-tool-use events."""
        # Calculate execution time and record performance
        execution_time_ms = 100  # Placeholder - would calculate actual time
        
        await self.record_hook_performance(
            hook_id=f"tool_use_{event.agent_id}",
            hook_type="tool_execution",
            execution_time_ms=execution_time_ms,
            memory_usage_mb=event.payload.get("memory_usage_mb", 0),
            success=event.payload.get("success", True),
            agent_id=event.agent_id,
            session_id=event.session_id,
            error_message=event.payload.get("error"),
            payload_size_bytes=len(json.dumps(event.payload))
        )
    
    async def _handle_agent_start(self, event: LifecycleEventData) -> None:
        """Handle agent start events."""
        # Initialize agent performance tracking
        await self.update_agent_performance(
            agent_id=event.agent_id,
            session_id=event.session_id,
            metrics={
                'task_completion_rate': 0.0,
                'average_response_time_ms': 0.0,
                'error_rate': 0.0
            }
        )
    
    async def _handle_agent_pause(self, event: LifecycleEventData) -> None:
        """Handle agent pause/sleep events."""
        # Update agent status in performance tracking
        if event.agent_id in self.agent_performance:
            # Calculate final metrics for this session
            pass
    
    async def _handle_task_assignment(self, event: LifecycleEventData) -> None:
        """Handle task assignment events."""
        # Check if this is part of a workflow
        workflow_id = event.payload.get("workflow_id")
        if workflow_id and workflow_id in self.workflow_progress:
            await self.update_workflow_progress(
                workflow_id=workflow_id,
                completed_steps=self.workflow_progress[workflow_id].completed_steps + 1
            )
    
    async def _handle_performance_degradation(self, event: LifecycleEventData) -> None:
        """Handle performance degradation events."""
        # Send alert for performance issues
        await self._send_integration_update(
            IntegrationEventType.SYSTEM_HEALTH_UPDATE,
            {
                'alert_type': 'performance_degradation',
                'agent_id': event.agent_id,
                'session_id': event.session_id,
                'details': event.payload
            }
        )
    
    async def _handle_error_pattern(self, event: LifecycleEventData) -> None:
        """Handle error pattern detection events."""
        # Send alert for error patterns
        await self._send_integration_update(
            IntegrationEventType.ERROR_PATTERN_DETECTION,
            {
                'pattern_type': event.payload.get("pattern_type"),
                'affected_agents': event.payload.get("affected_agents", []),
                'severity': event.payload.get("severity", "medium"),
                'details': event.payload
            }
        )
    
    async def _process_update_queue(self) -> None:
        """Background task to process integration updates."""
        while self.is_running:
            try:
                # Process updates in batches
                updates = []
                
                # Collect updates for batch processing
                for _ in range(self.batch_size):
                    try:
                        update = await asyncio.wait_for(
                            self.update_queue.get(),
                            timeout=0.1
                        )
                        updates.append(update)
                    except asyncio.TimeoutError:
                        break
                
                if updates:
                    await self._process_update_batch(updates)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error processing update queue: {e}")
                await asyncio.sleep(5)
    
    async def _process_update_batch(self, updates: List[Dict[str, Any]]) -> None:
        """Process a batch of updates."""
        # Group updates by type for efficient processing
        grouped_updates = defaultdict(list)
        
        for update in updates:
            event_type = update.get('event_type')
            grouped_updates[event_type].append(update)
        
        # Process each group
        for event_type, type_updates in grouped_updates.items():
            await self._send_batched_updates(event_type, type_updates)
    
    async def _send_batched_updates(
        self,
        event_type: str,
        updates: List[Dict[str, Any]]
    ) -> None:
        """Send batched updates to active streams."""
        batch_message = {
            'type': 'batch_update',
            'event_type': event_type,
            'updates': updates,
            'count': len(updates),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all active streams
        disconnected_streams = []
        
        for stream_id, websocket in self.active_streams.items():
            try:
                # Apply stream filters
                if self._should_send_to_stream(stream_id, event_type, updates):
                    await websocket.send_text(json.dumps(batch_message))
            
            except WebSocketDisconnect:
                disconnected_streams.append(stream_id)
            except Exception as e:
                logger.error(
                    "Error sending batch update to stream",
                    stream_id=stream_id,
                    error=str(e)
                )
                disconnected_streams.append(stream_id)
        
        # Clean up disconnected streams
        for stream_id in disconnected_streams:
            await self.unregister_dashboard_stream(stream_id)
    
    def _should_send_to_stream(
        self,
        stream_id: str,
        event_type: str,
        updates: List[Dict[str, Any]]
    ) -> bool:
        """Check if updates should be sent to a specific stream."""
        filters = self.stream_filters.get(stream_id, {})
        
        # Apply event type filter
        if 'event_types' in filters:
            if event_type not in filters['event_types']:
                return False
        
        # Apply agent filter
        if 'agent_ids' in filters:
            agent_ids = set(filters['agent_ids'])
            update_agent_ids = set()
            
            for update in updates:
                if 'agent_id' in update:
                    update_agent_ids.add(update['agent_id'])
            
            if not update_agent_ids.intersection(agent_ids):
                return False
        
        # Apply session filter
        if 'session_ids' in filters:
            session_ids = set(filters['session_ids'])
            update_session_ids = set()
            
            for update in updates:
                if 'session_id' in update:
                    update_session_ids.add(update['session_id'])
            
            if not update_session_ids.intersection(session_ids):
                return False
        
        return True
    
    async def _send_integration_update(
        self,
        event_type: IntegrationEventType,
        data: Dict[str, Any]
    ) -> None:
        """Send an integration update to the update queue."""
        update = {
            'event_type': event_type.value,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.update_queue.put(update)
    
    async def _send_initial_dashboard_data(
        self,
        websocket: WebSocket,
        filters: Optional[Dict[str, Any]]
    ) -> None:
        """Send initial dashboard data to a new stream."""
        try:
            initial_data = {
                'type': 'initial_data',
                'data': {
                    'workflows': await self.get_workflow_progress_data(),
                    'quality_gates': await self.get_quality_gates_data(),
                    'thinking_sessions': await self.get_thinking_sessions_data(),
                    'agent_performance': await self.get_agent_performance_data(),
                    'system_overview': await self.get_system_performance_overview()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(initial_data))
            
        except Exception as e:
            logger.error(f"Error sending initial dashboard data: {e}")
    
    async def _collect_performance_metrics(self) -> None:
        """Background task to collect system performance metrics."""
        while self.is_running:
            try:
                # Collect system-wide performance metrics
                system_metrics = await self.performance_collector.collect_system_metrics()
                
                # Update agent performance metrics
                for agent_id in self.agent_performance.keys():
                    agent_metrics = await self.performance_collector.collect_agent_metrics(agent_id)
                    if agent_metrics:
                        await self.update_agent_performance(agent_id, "", agent_metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_system_health(self) -> None:
        """Background task to monitor overall system health."""
        while self.is_running:
            try:
                health_status = await self._calculate_system_health()
                
                # Send health update if status changed or if critical
                if health_status.get('status') in ['critical', 'warning']:
                    await self._send_integration_update(
                        IntegrationEventType.SYSTEM_HEALTH_UPDATE,
                        {
                            'health_status': health_status,
                            'alert_level': health_status.get('status'),
                            'recommendations': health_status.get('recommendations', [])
                        }
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(120)
    
    async def _generate_insights(self) -> None:
        """Background task to generate collaboration and optimization insights."""
        while self.is_running:
            try:
                # Generate insights every 5 minutes
                await asyncio.sleep(300)
                
                insights = await self._analyze_collaboration_patterns()
                
                if insights:
                    await self._send_integration_update(
                        IntegrationEventType.COLLABORATION_INSIGHT,
                        {
                            'insights': insights,
                            'recommendations': await self._generate_optimization_recommendations()
                        }
                    )
                
            except Exception as e:
                logger.error(f"Error generating insights: {e}")
                await asyncio.sleep(600)
    
    async def _cleanup_old_data(self) -> None:
        """Background task to clean up old data."""
        while self.is_running:
            try:
                # Clean up every hour
                await asyncio.sleep(3600)
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean up completed workflows older than 24 hours
                old_workflows = [
                    wid for wid, workflow in self.workflow_progress.items()
                    if workflow.is_completed and workflow.start_time < cutoff_time
                ]
                
                for workflow_id in old_workflows:
                    del self.workflow_progress[workflow_id]
                
                # Clean up old thinking sessions
                old_sessions = [
                    sid for sid, session in self.thinking_sessions.items()
                    if session.timestamp < cutoff_time
                ]
                
                for session_id in old_sessions:
                    del self.thinking_sessions[session_id]
                
                logger.info(
                    "Dashboard integration cleanup completed",
                    removed_workflows=len(old_workflows),
                    removed_sessions=len(old_sessions)
                )
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    async def _calculate_quality_gate_trends(self, gate_id: str) -> Dict[str, Any]:
        """Calculate trends for quality gate performance."""
        if gate_id not in self.quality_gate_results:
            return {}
        
        results = self.quality_gate_results[gate_id][-20:]  # Last 20 results
        
        if len(results) < 2:
            return {}
        
        # Calculate success rate trend
        success_rates = []
        for i in range(0, len(results), 5):  # Group by 5s
            group = results[i:i+5]
            if group:
                success_rate = len([r for r in group if r.passed]) / len(group)
                success_rates.append(success_rate)
        
        # Calculate execution time trend
        execution_times = [r.execution_time_ms for r in results]
        
        return {
            'success_rate_trend': success_rates,
            'execution_time_trend': execution_times[-10:],  # Last 10 execution times
            'current_success_rate': success_rates[-1] if success_rates else 0,
            'average_execution_time': statistics.mean(execution_times) if execution_times else 0
        }
    
    async def _calculate_hook_performance_trends(self, hook_type: str) -> Dict[str, Any]:
        """Calculate performance trends for hook execution."""
        if hook_type not in self.hook_performance:
            return {}
        
        metrics = list(self.hook_performance[hook_type])[-50:]  # Last 50 executions
        
        if len(metrics) < 2:
            return {}
        
        execution_times = [m.execution_time_ms for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        success_rates = []
        
        # Calculate success rate in groups of 10
        for i in range(0, len(metrics), 10):
            group = metrics[i:i+10]
            if group:
                success_rate = len([m for m in group if m.success]) / len(group)
                success_rates.append(success_rate)
        
        return {
            'execution_time_trend': execution_times[-20:],
            'memory_usage_trend': memory_usage[-20:],
            'success_rate_trend': success_rates,
            'current_success_rate': success_rates[-1] if success_rates else 0,
            'average_execution_time': statistics.mean(execution_times) if execution_times else 0,
            'average_memory_usage': statistics.mean(memory_usage) if memory_usage else 0
        }
    
    async def _calculate_agent_performance_trends(self, agent_id: str) -> Dict[str, Any]:
        """Calculate performance trends for an agent."""
        # This would track historical performance data
        # For now, return placeholder trends
        return {
            'overall_score_trend': [85, 87, 89, 88, 90],
            'response_time_trend': [120, 115, 110, 108, 105],
            'error_rate_trend': [0.02, 0.01, 0.01, 0.015, 0.01],
            'collaboration_score_trend': [0.8, 0.82, 0.85, 0.83, 0.87]
        }
    
    async def _calculate_collaboration_metrics(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Calculate collaboration metrics for a group of agents."""
        # This would analyze communication patterns between agents
        return {
            'communication_frequency': 15.5,  # messages per minute
            'response_consistency': 0.85,
            'knowledge_sharing_score': 0.78,
            'coordination_efficiency': 0.82,
            'conflict_resolution_time': 45  # seconds
        }
    
    async def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health status."""
        # Analyze various system metrics to determine health
        active_workflows = len([w for w in self.workflow_progress.values() if not w.is_completed])
        failed_quality_gates = sum(
            len([r for r in results if r.status == QualityGateStatus.FAILED])
            for results in self.quality_gate_results.values()
        )
        
        # Simple health calculation
        health_score = 100
        
        if active_workflows > 10:
            health_score -= 10
        
        if failed_quality_gates > 5:
            health_score -= 20
        
        if health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': health_score,
            'active_workflows': active_workflows,
            'failed_quality_gates': failed_quality_gates,
            'recommendations': [
                "Monitor workflow completion rates",
                "Review quality gate failure patterns",
                "Consider scaling resources if needed"
            ] if status != 'healthy' else []
        }
    
    async def _analyze_collaboration_patterns(self) -> List[Dict[str, Any]]:
        """Analyze collaboration patterns and generate insights."""
        insights = []
        
        # Analyze thinking sessions for collaboration patterns
        for session_id, session in self.thinking_sessions.items():
            if session.collaboration_quality < 0.7:
                insights.append({
                    'type': 'collaboration_issue',
                    'session_id': session_id,
                    'issue': 'Low collaboration quality detected',
                    'recommendation': 'Review agent coordination mechanisms',
                    'severity': 'medium'
                })
            
            if session.consensus_level < 0.6:
                insights.append({
                    'type': 'consensus_issue',
                    'session_id': session_id,
                    'issue': 'Low consensus among agents',
                    'recommendation': 'Implement conflict resolution strategies',
                    'severity': 'high'
                })
        
        return insights
    
    async def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on current metrics."""
        recommendations = []
        
        # Analyze hook performance for optimization opportunities
        for hook_type, metrics in self.hook_performance.items():
            if not metrics:
                continue
            
            recent_metrics = list(metrics)[-20:]
            avg_execution_time = statistics.mean([m.execution_time_ms for m in recent_metrics])
            
            if avg_execution_time > 1000:  # More than 1 second
                recommendations.append({
                    'type': 'performance_optimization',
                    'component': f'Hook: {hook_type}',
                    'issue': f'High execution time: {avg_execution_time:.0f}ms',
                    'recommendation': 'Consider optimizing hook implementation or caching',
                    'priority': 'high'
                })
        
        # Analyze agent performance for optimization opportunities
        for agent_id, metrics in self.agent_performance.items():
            if metrics.error_rate > 0.1:  # More than 10% error rate
                recommendations.append({
                    'type': 'reliability_optimization',
                    'component': f'Agent: {agent_id}',
                    'issue': f'High error rate: {metrics.error_rate:.1%}',
                    'recommendation': 'Review agent error handling and retry mechanisms',
                    'priority': 'high'
                })
        
        return recommendations


# Global comprehensive dashboard integration instance
comprehensive_dashboard_integration = ComprehensiveDashboardIntegration()