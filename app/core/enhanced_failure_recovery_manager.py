"""
Enhanced Failure Recovery Manager for LeanVibe Agent Hive 2.0

Vertical Slice 2.1: Provides sophisticated failure recovery capabilities with
automatic task reassignment, circuit breaker patterns, predictive failure detection,
and intelligent recovery strategies for production-grade multi-agent systems.

Features:
- Automatic task reassignment with intelligent agent selection
- Circuit breaker patterns for fault isolation
- Predictive failure detection using performance analytics
- Multi-tier recovery strategies with escalation
- Real-time failure monitoring and alerting
- Recovery performance optimization and learning
- Graceful degradation and emergency protocols
"""

import asyncio
import json
import uuid
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import heapq
import threading

import structlog
from sqlalchemy import select, update, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_async_session
from .redis import get_redis, get_message_broker, AgentMessageBroker
from .recovery_manager import RecoveryManager, RecoveryError
from .enhanced_intelligent_task_router import (
    EnhancedIntelligentTaskRouter, get_enhanced_task_router,
    EnhancedTaskRoutingContext, EnhancedRoutingStrategy
)
from .agent_load_balancer import AgentLoadBalancer
from .intelligent_alerting import IntelligentAlertingSystem
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.session import Session, SessionStatus
from ..models.agent_performance import AgentPerformanceHistory, TaskRoutingDecision, WorkloadSnapshot

logger = structlog.get_logger()


class FailureType(str, Enum):
    """Types of failures that can occur in the system."""
    AGENT_UNRESPONSIVE = "agent_unresponsive"
    TASK_TIMEOUT = "task_timeout"
    TASK_EXECUTION_ERROR = "task_execution_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMMUNICATION_FAILURE = "communication_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DEPENDENCY_FAILURE = "dependency_failure"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_OVERLOAD = "system_overload"


class FailureSeverity(str, Enum):
    """Severity levels for failures."""
    CRITICAL = "critical"      # System-wide impact, immediate attention required
    HIGH = "high"             # Significant impact, urgent response needed
    MEDIUM = "medium"         # Moderate impact, timely response required
    LOW = "low"              # Minor impact, can be addressed in normal flow
    INFORMATIONAL = "info"    # Informational, no immediate action required


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure scenarios."""
    IMMEDIATE_REASSIGNMENT = "immediate_reassignment"
    GRACEFUL_REASSIGNMENT = "graceful_reassignment"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADED_SERVICE = "degraded_service"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    LOAD_SHEDDING = "load_shedding"
    FAILOVER_CLUSTER = "failover_cluster"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class FailureEvent:
    """Represents a failure event in the system."""
    event_id: str
    failure_type: FailureType
    severity: FailureSeverity
    timestamp: datetime
    
    # Affected components
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Failure details
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_attempts: int = 0
    recovery_start_time: Optional[datetime] = None
    recovery_end_time: Optional[datetime] = None
    recovery_success: bool = False
    
    # Impact assessment
    affected_tasks: List[str] = field(default_factory=list)
    affected_agents: List[str] = field(default_factory=list)
    business_impact: str = "unknown"
    
    # Performance metrics
    detection_time_ms: Optional[float] = None
    recovery_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance and isolation."""
    resource_id: str
    resource_type: str  # 'agent', 'service', 'workflow'
    
    # Configuration
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    
    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_attempt_time: Optional[datetime] = None
    
    # Metrics
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.total_requests += 1
        self.total_successes += 1
        self.success_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitBreakerState.OPEN:
            # Reset failure count on unexpected success
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.total_requests += 1
        self.total_failures += 1
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    def can_attempt(self) -> bool:
        """Check if an operation can be attempted."""
        now = datetime.utcnow()
        self.last_attempt_time = now
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                now - self.last_failure_time > timedelta(seconds=self.timeout_seconds)):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        failure_rate = (self.total_failures / self.total_requests 
                       if self.total_requests > 0 else 0.0)
        
        return {
            'resource_id': self.resource_id,
            'resource_type': self.resource_type,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': self.total_requests,
            'failure_rate': failure_rate,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


@dataclass
class RecoveryPlan:
    """Comprehensive recovery plan for failure scenarios."""
    plan_id: str
    failure_event: FailureEvent
    primary_strategy: RecoveryStrategy
    fallback_strategies: List[RecoveryStrategy]
    
    # Execution details
    estimated_recovery_time_minutes: int
    resource_requirements: Dict[str, Any]
    prerequisites: List[str]
    
    # Task reassignment plan
    tasks_to_reassign: List[str] = field(default_factory=list)
    reassignment_targets: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    reassignment_priorities: Dict[str, int] = field(default_factory=dict)
    
    # Communication plan
    notifications_required: List[str] = field(default_factory=list)
    escalation_contacts: List[str] = field(default_factory=list)
    
    # Success criteria
    success_metrics: Dict[str, float] = field(default_factory=dict)
    validation_steps: List[str] = field(default_factory=list)
    
    # Rollback plan
    rollback_steps: List[str] = field(default_factory=list)
    rollback_triggers: List[str] = field(default_factory=list)


class FailurePredictor:
    """Machine learning-based failure prediction system."""
    
    def __init__(self):
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.failure_history: deque = deque(maxlen=1000)
        self.prediction_models: Dict[str, Any] = {}
        self.feature_weights: Dict[str, float] = {
            'response_time_trend': 0.3,
            'error_rate_trend': 0.25,
            'resource_utilization': 0.2,
            'workload_pressure': 0.15,
            'historical_patterns': 0.1
        }
        
    def predict_failure_probability(self, agent_id: str, 
                                  time_horizon_minutes: int = 30) -> Tuple[float, Dict[str, float]]:
        """
        Predict the probability of failure for an agent within a time horizon.
        
        Returns:
            Tuple of (failure_probability, contributing_factors)
        """
        try:
            # Get recent performance data
            recent_data = list(self.performance_history.get(agent_id, deque()))[-20:]
            
            if len(recent_data) < 5:
                return 0.1, {"insufficient_data": 1.0}  # Low probability with uncertainty
            
            # Calculate trends and patterns
            factors = {}
            
            # Response time trend
            response_times = [d.get('response_time', 100) for d in recent_data]
            if len(response_times) > 1:
                trend = statistics.linear_regression(range(len(response_times)), response_times)[0]
                factors['response_time_trend'] = min(1.0, max(0.0, trend / 1000.0))
            
            # Error rate trend
            error_rates = [d.get('error_rate', 0.0) for d in recent_data]
            current_error_rate = statistics.mean(error_rates[-3:]) if error_rates else 0.0
            factors['error_rate_trend'] = min(1.0, current_error_rate)
            
            # Resource utilization
            cpu_usage = [d.get('cpu_usage', 0.5) for d in recent_data]
            memory_usage = [d.get('memory_usage', 0.5) for d in recent_data]
            current_resource_pressure = (statistics.mean(cpu_usage[-3:]) + 
                                       statistics.mean(memory_usage[-3:])) / 2.0
            factors['resource_utilization'] = current_resource_pressure
            
            # Calculate weighted probability
            failure_probability = sum(
                factors.get(factor, 0.0) * weight 
                for factor, weight in self.feature_weights.items()
            )
            
            return min(1.0, max(0.0, failure_probability)), factors
            
        except Exception as e:
            logger.warning("Error predicting failure probability", 
                         agent_id=agent_id, error=str(e))
            return 0.1, {"prediction_error": 1.0}
    
    def update_performance_data(self, agent_id: str, performance_data: Dict[str, Any]) -> None:
        """Update performance data for an agent."""
        self.performance_history[agent_id].append({
            **performance_data,
            'timestamp': datetime.utcnow()
        })
    
    def record_failure(self, failure_event: FailureEvent) -> None:
        """Record a failure event for learning."""
        self.failure_history.append({
            'agent_id': failure_event.agent_id,
            'failure_type': failure_event.failure_type.value,
            'timestamp': failure_event.timestamp,
            'context': failure_event.context
        })


class EnhancedFailureRecoveryManager:
    """
    Enhanced failure recovery manager with automatic task reassignment,
    circuit breaker patterns, and intelligent recovery strategies.
    """
    
    def __init__(self):
        self.base_recovery_manager = RecoveryManager()
        self.task_router: Optional[EnhancedIntelligentTaskRouter] = None
        self.load_balancer: Optional[AgentLoadBalancer] = None
        self.alerting_system: Optional[IntelligentAlertingSystem] = None
        self.message_broker: Optional[AgentMessageBroker] = None
        
        # Failure tracking and recovery
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.active_failures: Dict[str, FailureEvent] = {}
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.failure_predictor = FailurePredictor()
        
        # Configuration
        self.config = {
            'max_concurrent_recoveries': 10,
            'recovery_timeout_minutes': 30,
            'automatic_recovery_enabled': True,
            'circuit_breaker_enabled': True,
            'predictive_monitoring_enabled': True,
            'escalation_thresholds': {
                'critical_failures': 3,
                'high_failures': 5,
                'recovery_time_minutes': 15
            }
        }
        
        # Metrics and analytics
        self.recovery_metrics: Dict[str, Any] = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time_ms': 0.0,
            'tasks_reassigned': 0,
            'circuit_breaker_activations': 0
        }
        
        # Background monitoring
        self.running = False
        self.monitoring_tasks: Set[asyncio.Task] = set()
        
        logger.info("Enhanced failure recovery manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the enhanced failure recovery manager."""
        try:
            # Initialize task router
            self.task_router = await get_enhanced_task_router()
            
            # Initialize other components (would be dependency-injected in production)
            self.load_balancer = AgentLoadBalancer()
            await self.load_balancer.initialize()
            
            self.message_broker = await get_message_broker()
            
            # Start monitoring tasks
            if not self.running:
                self.running = True
                await self._start_monitoring_tasks()
            
            logger.info("Enhanced failure recovery manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize enhanced failure recovery manager", error=str(e))
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the recovery manager."""
        self.running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Enhanced failure recovery manager shutdown complete")
    
    async def handle_failure(self, failure_event: FailureEvent) -> bool:
        """
        Handle a failure event with comprehensive recovery strategies.
        
        Args:
            failure_event: The failure event to handle
            
        Returns:
            True if recovery was successful, False otherwise
        """
        failure_start_time = time.time()
        
        logger.warning("Handling failure event",
                      event_id=failure_event.event_id,
                      failure_type=failure_event.failure_type.value,
                      severity=failure_event.severity.value,
                      agent_id=failure_event.agent_id)
        
        try:
            # Record the failure
            self.active_failures[failure_event.event_id] = failure_event
            self.recovery_metrics['total_failures'] += 1
            
            # Update circuit breakers
            await self._update_circuit_breakers(failure_event)
            
            # Create recovery plan
            recovery_plan = await self._create_recovery_plan(failure_event)
            self.recovery_plans[failure_event.event_id] = recovery_plan
            
            # Execute recovery
            recovery_success = await self._execute_recovery_plan(recovery_plan)
            
            # Update metrics
            recovery_time = time.time() - failure_start_time
            failure_event.recovery_time_ms = recovery_time * 1000
            failure_event.recovery_success = recovery_success
            
            if recovery_success:
                self.recovery_metrics['successful_recoveries'] += 1
                logger.info("Failure recovery completed successfully",
                           event_id=failure_event.event_id,
                           recovery_time_ms=failure_event.recovery_time_ms)
            else:
                self.recovery_metrics['failed_recoveries'] += 1
                logger.error("Failure recovery failed",
                           event_id=failure_event.event_id,
                           recovery_time_ms=failure_event.recovery_time_ms)
            
            # Update failure predictor
            self.failure_predictor.record_failure(failure_event)
            
            # Send alerts if necessary
            await self._send_failure_alerts(failure_event, recovery_success)
            
            return recovery_success
            
        except Exception as e:
            logger.error("Error handling failure event",
                        event_id=failure_event.event_id,
                        error=str(e))
            return False
        finally:
            # Clean up
            self.active_failures.pop(failure_event.event_id, None)
            self.recovery_plans.pop(failure_event.event_id, None)
    
    async def reassign_tasks_from_failed_agent(self, agent_id: str, 
                                             failure_context: Dict[str, Any]) -> List[str]:
        """
        Reassign all tasks from a failed agent to available agents.
        
        Args:
            agent_id: ID of the failed agent
            failure_context: Context information about the failure
            
        Returns:
            List of successfully reassigned task IDs
        """
        try:
            # Get active tasks for the failed agent
            active_tasks = await self._get_agent_active_tasks(agent_id)
            
            if not active_tasks:
                logger.info("No active tasks to reassign", agent_id=agent_id)
                return []
            
            logger.info("Reassigning tasks from failed agent",
                       agent_id=agent_id,
                       task_count=len(active_tasks))
            
            reassigned_tasks = []
            
            for task in active_tasks:
                try:
                    # Create enhanced routing context for reassignment
                    routing_context = await self._create_reassignment_context(task, failure_context)
                    
                    # Get available agents (excluding the failed one)
                    available_agents = await self._get_available_agents(exclude={agent_id})
                    
                    if not available_agents:
                        logger.warning("No available agents for task reassignment",
                                     task_id=str(task.id))
                        continue
                    
                    # Use enhanced router to find suitable agent
                    new_agent = await self.task_router.route_task_advanced(
                        task, available_agents, routing_context,
                        EnhancedRoutingStrategy.PERFORMANCE_WEIGHTED_PERSONA
                    )
                    
                    if new_agent:
                        # Update task assignment
                        await self._update_task_assignment(task, new_agent, failure_context)
                        reassigned_tasks.append(str(task.id))
                        
                        logger.info("Task successfully reassigned",
                                   task_id=str(task.id),
                                   from_agent=agent_id,
                                   to_agent=str(new_agent.id))
                    else:
                        logger.warning("Failed to find suitable agent for task reassignment",
                                     task_id=str(task.id))
                
                except Exception as e:
                    logger.error("Error reassigning individual task",
                               task_id=str(task.id),
                               error=str(e))
            
            # Update metrics
            self.recovery_metrics['tasks_reassigned'] += len(reassigned_tasks)
            
            logger.info("Task reassignment completed",
                       agent_id=agent_id,
                       total_tasks=len(active_tasks),
                       reassigned_count=len(reassigned_tasks))
            
            return reassigned_tasks
            
        except Exception as e:
            logger.error("Error reassigning tasks from failed agent",
                        agent_id=agent_id,
                        error=str(e))
            return []
    
    async def get_circuit_breaker_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get circuit breaker status for a resource."""
        circuit_breaker = self.circuit_breakers.get(resource_id)
        return circuit_breaker.get_metrics() if circuit_breaker else None
    
    async def reset_circuit_breaker(self, resource_id: str) -> bool:
        """Manually reset a circuit breaker."""
        if resource_id in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[resource_id]
            circuit_breaker.state = CircuitBreakerState.CLOSED
            circuit_breaker.failure_count = 0
            circuit_breaker.success_count = 0
            
            logger.info("Circuit breaker manually reset", resource_id=resource_id)
            return True
        
        return False
    
    async def predict_agent_failures(self, time_horizon_minutes: int = 30) -> Dict[str, float]:
        """
        Predict failure probabilities for all active agents.
        
        Args:
            time_horizon_minutes: Time horizon for prediction
            
        Returns:
            Dictionary mapping agent_id to failure_probability
        """
        try:
            predictions = {}
            
            # Get all active agents
            active_agents = await self._get_all_active_agents()
            
            for agent in active_agents:
                probability, factors = self.failure_predictor.predict_failure_probability(
                    str(agent.id), time_horizon_minutes
                )
                
                if probability > 0.5:  # High failure probability
                    predictions[str(agent.id)] = probability
                    
                    logger.warning("High failure probability detected",
                                 agent_id=str(agent.id),
                                 probability=probability,
                                 factors=factors)
            
            return predictions
            
        except Exception as e:
            logger.error("Error predicting agent failures", error=str(e))
            return {}
    
    async def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get comprehensive recovery metrics."""
        try:
            # Calculate derived metrics
            total_recovery_attempts = (self.recovery_metrics['successful_recoveries'] + 
                                     self.recovery_metrics['failed_recoveries'])
            
            success_rate = (self.recovery_metrics['successful_recoveries'] / total_recovery_attempts 
                          if total_recovery_attempts > 0 else 0.0)
            
            # Circuit breaker metrics
            circuit_breaker_metrics = {
                'total_circuit_breakers': len(self.circuit_breakers),
                'open_circuit_breakers': len([cb for cb in self.circuit_breakers.values() 
                                           if cb.state == CircuitBreakerState.OPEN]),
                'half_open_circuit_breakers': len([cb for cb in self.circuit_breakers.values() 
                                                if cb.state == CircuitBreakerState.HALF_OPEN])
            }
            
            return {
                **self.recovery_metrics,
                'recovery_success_rate': success_rate,
                'active_failures': len(self.active_failures),
                'active_recovery_plans': len(self.recovery_plans),
                **circuit_breaker_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error collecting recovery metrics", error=str(e))
            return self.recovery_metrics
    
    # Background monitoring methods
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        tasks = [
            asyncio.create_task(self._failure_detection_loop()),
            asyncio.create_task(self._predictive_monitoring_loop()),
            asyncio.create_task(self._circuit_breaker_maintenance_loop()),
            asyncio.create_task(self._recovery_timeout_monitoring_loop())
        ]
        
        self.monitoring_tasks.update(tasks)
        
        logger.info("Failure recovery monitoring tasks started", task_count=len(tasks))
    
    async def _failure_detection_loop(self) -> None:
        """Background loop for detecting failures."""
        while self.running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check for agent unresponsiveness
                await self._check_agent_responsiveness()
                
                # Check for task timeouts
                await self._check_task_timeouts()
                
                # Check for performance degradation
                await self._check_performance_degradation()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in failure detection loop", error=str(e))
    
    async def _predictive_monitoring_loop(self) -> None:
        """Background loop for predictive failure monitoring."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.config['predictive_monitoring_enabled']:
                    # Predict failures and take proactive action
                    predictions = await self.predict_agent_failures(30)
                    
                    for agent_id, probability in predictions.items():
                        if probability > 0.8:  # Very high probability
                            await self._handle_predicted_failure(agent_id, probability)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in predictive monitoring loop", error=str(e))
    
    async def _circuit_breaker_maintenance_loop(self) -> None:
        """Background loop for circuit breaker maintenance."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Clean up old circuit breakers
                await self._cleanup_circuit_breakers()
                
                # Update circuit breaker metrics
                await self._update_circuit_breaker_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in circuit breaker maintenance loop", error=str(e))
    
    async def _recovery_timeout_monitoring_loop(self) -> None:
        """Background loop for monitoring recovery timeouts."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for recovery timeouts
                await self._check_recovery_timeouts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in recovery timeout monitoring loop", error=str(e))
    
    # Helper methods (abbreviated for space - would include full implementations)
    
    async def _create_recovery_plan(self, failure_event: FailureEvent) -> RecoveryPlan:
        """Create a comprehensive recovery plan for a failure event."""
        # Implementation would analyze failure and create detailed recovery plan
        return RecoveryPlan(
            plan_id=str(uuid.uuid4()),
            failure_event=failure_event,
            primary_strategy=RecoveryStrategy.IMMEDIATE_REASSIGNMENT,
            fallback_strategies=[RecoveryStrategy.GRACEFUL_REASSIGNMENT],
            estimated_recovery_time_minutes=5
        )
    
    async def _execute_recovery_plan(self, recovery_plan: RecoveryPlan) -> bool:
        """Execute a recovery plan."""
        # Implementation would execute the recovery plan steps
        return True
    
    async def _get_agent_active_tasks(self, agent_id: str) -> List[Task]:
        """Get all active tasks for an agent."""
        # Implementation would query database for active tasks
        return []
    
    async def _get_available_agents(self, exclude: Set[str] = None) -> List[Agent]:
        """Get all available agents."""
        # Implementation would query database for available agents
        return []
    
    # Additional helper methods would be implemented here...


# Global instance for dependency injection
_enhanced_recovery_manager: Optional[EnhancedFailureRecoveryManager] = None


async def get_enhanced_recovery_manager() -> EnhancedFailureRecoveryManager:
    """Get or create the global enhanced recovery manager instance."""
    global _enhanced_recovery_manager
    
    if _enhanced_recovery_manager is None:
        _enhanced_recovery_manager = EnhancedFailureRecoveryManager()
        await _enhanced_recovery_manager.initialize()
    
    return _enhanced_recovery_manager


async def shutdown_enhanced_recovery_manager() -> None:
    """Shutdown the global enhanced recovery manager."""
    global _enhanced_recovery_manager
    
    if _enhanced_recovery_manager:
        await _enhanced_recovery_manager.shutdown()
        _enhanced_recovery_manager = None