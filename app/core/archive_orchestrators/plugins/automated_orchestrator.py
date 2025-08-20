"""
Automated Sleep/Wake Orchestrator with Advanced Recovery Mechanisms.

Provides intelligent automation for sleep-wake cycle management with:
- Proactive sleep/wake scheduling based on activity patterns
- Intelligent recovery with multi-tier fallback strategies
- Circuit breaker patterns for fault tolerance
- Automated health monitoring and self-healing
- Performance-driven optimization and adaptation
- Event-driven orchestration with real-time responsiveness
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, update
from sqlalchemy.orm import selectinload

from ..models.sleep_wake import (
    SleepWakeCycle, SleepState, CheckpointType, Checkpoint, 
    SleepWindow, ConsolidationJob, ConsolidationStatus
)
from ..models.agent import Agent, AgentStatus
from ..models.performance_metric import PerformanceMetric
from ..core.database import get_async_session
from ..core.sleep_wake_manager import get_sleep_wake_manager
from ..core.recovery_manager import get_recovery_manager
from ..core.sleep_scheduler import get_sleep_scheduler
from ..core.sleep_analytics import get_sleep_analytics_engine
from ..core.intelligent_sleep_manager import get_intelligent_sleep_manager
from ..core.consolidation_engine import get_consolidation_engine
from ..core.redis import get_redis
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Orchestration strategies for different scenarios."""
    PROACTIVE = "proactive"           # Anticipate and schedule operations
    REACTIVE = "reactive"             # Respond to events and conditions
    HYBRID = "hybrid"                 # Combine proactive and reactive approaches
    MAINTENANCE = "maintenance"       # Focus on system health and optimization
    EMERGENCY = "emergency"           # Emergency response mode


class RecoveryTier(Enum):
    """Recovery tiers with increasing intervention levels."""
    AUTOMATIC = "automatic"           # Automated recovery without human intervention
    ASSISTED = "assisted"             # Automated with human notification
    MANUAL = "manual"                 # Requires human intervention
    EMERGENCY = "emergency"           # Critical system failure requiring immediate attention


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"                 # Normal operation
    OPEN = "open"                     # Failing, reject requests
    HALF_OPEN = "half_open"          # Testing if service is recovered


@dataclass
class OrchestrationEvent:
    """Event for orchestration system."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    agent_id: Optional[UUID] = None
    priority: int = 500  # Higher number = higher priority
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryPlan:
    """Recovery plan with multi-tier strategies."""
    recovery_id: UUID = field(default_factory=uuid4)
    agent_id: Optional[UUID] = None
    failure_type: str = ""
    failure_severity: int = 1  # 1-5 scale
    recovery_tier: RecoveryTier = RecoveryTier.AUTOMATIC
    strategies: List[str] = field(default_factory=list)
    estimated_recovery_time: float = 0.0
    success_probability: float = 0.0
    fallback_plans: List['RecoveryPlan'] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    test_request_timeout: int = 30
    failure_window: int = 300  # 5 minutes


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_test_time = None
        
        # Failure tracking
        self.failure_history = deque(maxlen=100)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if not self.last_failure_time:
            return True
        
        return (time.time() - self.last_failure_time) > self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        self.failure_history.append({
            "timestamp": datetime.utcnow(),
            "failure_count": self.failure_count
        })
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class AutomatedOrchestrator:
    """
    Automated orchestrator for sleep-wake operations with advanced recovery.
    
    Features:
    - Intelligent proactive scheduling based on patterns
    - Event-driven reactive orchestration
    - Multi-tier recovery strategies with automatic escalation
    - Circuit breaker patterns for fault tolerance
    - Performance-driven optimization and adaptation
    - Health monitoring with self-healing capabilities
    - Real-time responsiveness to system conditions
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Core components (initialized on first use)
        self._sleep_wake_manager = None
        self._recovery_manager = None
        self._sleep_scheduler = None
        self._analytics_engine = None
        self._intelligent_sleep_manager = None
        self._consolidation_engine = None
        
        # Orchestration configuration
        self.strategy = OrchestrationStrategy.HYBRID
        self.orchestration_interval_seconds = 30
        self.health_check_interval_seconds = 60
        self.optimization_interval_seconds = 300  # 5 minutes
        
        # Recovery configuration
        self.recovery_thresholds = {
            "error_rate": 0.1,           # 10% error rate triggers recovery
            "response_time": 30000,      # 30s response time triggers optimization
            "success_rate": 0.85,        # Below 85% success rate triggers intervention
            "uptime": 0.95              # Below 95% uptime triggers escalation
        }
        
        # Event processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_handlers: Dict[str, Callable] = {}
        
        # Recovery management
        self._active_recovery_plans: Dict[UUID, RecoveryPlan] = {}
        self._recovery_history: List[RecoveryPlan] = []
        
        # Circuit breakers for critical operations
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()
        
        # Orchestration state
        self._orchestration_enabled = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self._operation_metrics = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "recovery_operations": 0,
            "successful_recoveries": 0,
            "circuit_breaker_activations": 0,
            "average_response_time_ms": 0.0,
            "last_optimization": None
        }
        
        # Health monitoring
        self._system_health_score = 100.0
        self._health_history = deque(maxlen=100)
        
        # Register default event handlers
        self._register_default_handlers()
    
    async def initialize(self) -> None:
        """Initialize the automated orchestrator."""
        try:
            logger.info("Initializing Automated Orchestrator")
            
            # Initialize core components
            self._sleep_wake_manager = await get_sleep_wake_manager()
            self._recovery_manager = get_recovery_manager()
            self._sleep_scheduler = await get_sleep_scheduler()
            self._analytics_engine = get_sleep_analytics_engine()
            self._intelligent_sleep_manager = await get_intelligent_sleep_manager()
            self._consolidation_engine = get_consolidation_engine()
            
            logger.info("Automated Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Automated Orchestrator: {e}")
            raise
    
    async def start_orchestration(self) -> None:
        """Start automated orchestration."""
        if self._orchestration_enabled:
            logger.warning("Orchestration is already running")
            return
        
        try:
            logger.info("Starting automated orchestration")
            self._orchestration_enabled = True
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._orchestration_loop()),
                asyncio.create_task(self._event_processing_loop()),
                asyncio.create_task(self._health_monitoring_loop()),
                asyncio.create_task(self._optimization_loop()),
                asyncio.create_task(self._recovery_monitoring_loop())
            ]
            
            self._background_tasks.extend(tasks)
            
            # Emit startup event
            await self.emit_event("orchestration_started", payload={"strategy": self.strategy.value})
            
            logger.info("Automated orchestration started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start orchestration: {e}")
            await self.stop_orchestration()
            raise
    
    async def stop_orchestration(self) -> None:
        """Stop automated orchestration."""
        if not self._orchestration_enabled:
            return
        
        logger.info("Stopping automated orchestration")
        self._orchestration_enabled = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        
        # Emit shutdown event
        await self.emit_event("orchestration_stopped")
        
        logger.info("Automated orchestration stopped")
    
    async def emit_event(
        self, 
        event_type: str, 
        agent_id: Optional[UUID] = None,
        priority: int = 500,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit an orchestration event."""
        event = OrchestrationEvent(
            event_type=event_type,
            agent_id=agent_id,
            priority=priority,
            payload=payload or {}
        )
        
        await self._event_queue.put(event)
        logger.debug(f"Emitted event: {event_type} for agent {agent_id}")
    
    async def schedule_proactive_operation(
        self, 
        operation_type: str,
        agent_id: UUID,
        scheduled_time: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Schedule a proactive operation."""
        try:
            await self.emit_event(
                "proactive_operation_scheduled",
                agent_id=agent_id,
                priority=300,
                payload={
                    "operation_type": operation_type,
                    "scheduled_time": scheduled_time.isoformat(),
                    "metadata": metadata or {}
                }
            )
            
            logger.info(f"Scheduled proactive {operation_type} for agent {agent_id} at {scheduled_time}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule proactive operation: {e}")
            return False
    
    async def trigger_recovery(
        self, 
        agent_id: Optional[UUID],
        failure_type: str,
        failure_severity: int = 1
    ) -> bool:
        """Trigger recovery operation."""
        try:
            # Create recovery plan
            recovery_plan = await self._create_recovery_plan(agent_id, failure_type, failure_severity)
            
            if not recovery_plan:
                logger.error(f"Failed to create recovery plan for {failure_type}")
                return False
            
            # Execute recovery plan
            success = await self._execute_recovery_plan(recovery_plan)
            
            if success:
                logger.info(f"Recovery successful for agent {agent_id}, failure type: {failure_type}")
            else:
                logger.error(f"Recovery failed for agent {agent_id}, failure type: {failure_type}")
                
                # Try fallback plans
                for fallback_plan in recovery_plan.fallback_plans:
                    logger.info(f"Attempting fallback recovery plan")
                    fallback_success = await self._execute_recovery_plan(fallback_plan)
                    if fallback_success:
                        logger.info("Fallback recovery successful")
                        success = True
                        break
            
            # Record recovery attempt
            recovery_plan.success_probability = 1.0 if success else 0.0
            self._recovery_history.append(recovery_plan)
            
            return success
            
        except Exception as e:
            logger.error(f"Error triggering recovery: {e}")
            return False
    
    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status."""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestration_enabled": self._orchestration_enabled,
                "strategy": self.strategy.value,
                "system_health_score": self._system_health_score,
                "metrics": self._operation_metrics.copy(),
                "circuit_breakers": {},
                "active_recovery_plans": len(self._active_recovery_plans),
                "event_queue_size": self._event_queue.qsize(),
                "background_tasks": len([t for t in self._background_tasks if not t.done()])
            }
            
            # Circuit breaker status
            for name, breaker in self._circuit_breakers.items():
                status["circuit_breakers"][name] = {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "last_failure": breaker.last_failure_time
                }
            
            # Recent health scores
            if self._health_history:
                status["recent_health_scores"] = list(self._health_history)[-10:]
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting orchestration status: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for critical operations."""
        config = CircuitBreakerConfig()
        
        critical_operations = [
            "sleep_cycle_initiation",
            "wake_cycle_initiation", 
            "checkpoint_creation",
            "recovery_operation",
            "consolidation_process"
        ]
        
        for operation in critical_operations:
            self._circuit_breakers[operation] = CircuitBreaker(operation, config)
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self._event_handlers.update({
            "agent_error": self._handle_agent_error,
            "consolidation_failed": self._handle_consolidation_failure,
            "checkpoint_failed": self._handle_checkpoint_failure,
            "recovery_required": self._handle_recovery_required,
            "proactive_operation_scheduled": self._handle_proactive_operation,
            "health_degraded": self._handle_health_degradation,
            "performance_threshold_exceeded": self._handle_performance_issues,
            "system_overload": self._handle_system_overload
        })
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self._orchestration_enabled:
            try:
                start_time = time.time()
                
                # Execute orchestration strategy
                await self._execute_orchestration_strategy()
                
                # Update metrics
                execution_time = (time.time() - start_time) * 1000
                self._operation_metrics["total_orchestrations"] += 1
                self._update_average_metric("average_response_time_ms", execution_time)
                
                await asyncio.sleep(self.orchestration_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _event_processing_loop(self) -> None:
        """Event processing loop."""
        while self._orchestration_enabled:
            try:
                # Get next event (with timeout to check orchestration state)
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process event
                await self._process_event(event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _health_monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while self._orchestration_enabled:
            try:
                # Calculate system health score
                health_score = await self._calculate_system_health()
                self._system_health_score = health_score
                self._health_history.append(health_score)
                
                # Check for health degradation
                if health_score < 80.0:
                    await self.emit_event(
                        "health_degraded",
                        priority=800,
                        payload={"health_score": health_score}
                    )
                
                await asyncio.sleep(self.health_check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _optimization_loop(self) -> None:
        """Optimization loop."""
        while self._orchestration_enabled:
            try:
                # Perform system optimization
                await self._perform_system_optimization()
                
                self._operation_metrics["last_optimization"] = datetime.utcnow().isoformat()
                
                await asyncio.sleep(self.optimization_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)
    
    async def _recovery_monitoring_loop(self) -> None:
        """Recovery monitoring loop."""
        while self._orchestration_enabled:
            try:
                # Monitor active recovery plans
                completed_recoveries = []
                
                for recovery_id, plan in self._active_recovery_plans.items():
                    # Check if recovery plan should be escalated or completed
                    if await self._should_escalate_recovery(plan):
                        await self._escalate_recovery(plan)
                    elif await self._is_recovery_complete(plan):
                        completed_recoveries.append(recovery_id)
                
                # Remove completed recoveries
                for recovery_id in completed_recoveries:
                    del self._active_recovery_plans[recovery_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _execute_orchestration_strategy(self) -> None:
        """Execute the current orchestration strategy."""
        try:
            if self.strategy == OrchestrationStrategy.PROACTIVE:
                await self._execute_proactive_orchestration()
            elif self.strategy == OrchestrationStrategy.REACTIVE:
                await self._execute_reactive_orchestration()
            elif self.strategy == OrchestrationStrategy.HYBRID:
                await self._execute_hybrid_orchestration()
            elif self.strategy == OrchestrationStrategy.MAINTENANCE:
                await self._execute_maintenance_orchestration()
            elif self.strategy == OrchestrationStrategy.EMERGENCY:
                await self._execute_emergency_orchestration()
            
            self._operation_metrics["successful_orchestrations"] += 1
            
        except Exception as e:
            logger.error(f"Error executing orchestration strategy: {e}")
    
    async def _execute_proactive_orchestration(self) -> None:
        """Execute proactive orchestration strategy."""
        # Get agents that need proactive sleep scheduling
        agents_for_sleep = await self._get_agents_needing_proactive_sleep()
        
        for agent_id in agents_for_sleep:
            await self._circuit_breakers["sleep_cycle_initiation"].call(
                self._schedule_proactive_sleep, agent_id
            )
        
        # Get agents that should be woken up
        agents_for_wake = await self._get_agents_needing_wake()
        
        for agent_id in agents_for_wake:
            await self._circuit_breakers["wake_cycle_initiation"].call(
                self._schedule_proactive_wake, agent_id
            )
    
    async def _execute_reactive_orchestration(self) -> None:
        """Execute reactive orchestration strategy."""
        # Check for immediate response needs
        system_status = await self._sleep_wake_manager.get_system_status()
        
        # React to error states
        for agent_id, agent_info in system_status.get("agents", {}).items():
            if agent_info.get("sleep_state") == "ERROR":
                await self.emit_event(
                    "agent_error",
                    agent_id=UUID(agent_id),
                    priority=900,
                    payload={"error_type": "sleep_state_error"}
                )
    
    async def _execute_hybrid_orchestration(self) -> None:
        """Execute hybrid orchestration strategy."""
        # Combine proactive and reactive approaches
        await self._execute_proactive_orchestration()
        await self._execute_reactive_orchestration()
    
    async def _execute_maintenance_orchestration(self) -> None:
        """Execute maintenance orchestration strategy."""
        # Focus on system health and optimization
        await self._perform_system_optimization()
        
        # Schedule maintenance sleep cycles for idle agents
        idle_agents = await self._get_idle_agents()
        
        for agent_id in idle_agents:
            await self.schedule_proactive_operation(
                "maintenance_sleep",
                agent_id,
                datetime.utcnow() + timedelta(minutes=5)
            )
    
    async def _execute_emergency_orchestration(self) -> None:
        """Execute emergency orchestration strategy."""
        # Focus on critical recovery operations
        critical_agents = await self._get_agents_in_critical_state()
        
        for agent_id in critical_agents:
            await self.trigger_recovery(
                agent_id,
                "emergency_state",
                failure_severity=5
            )
    
    async def _process_event(self, event: OrchestrationEvent) -> None:
        """Process an orchestration event."""
        try:
            logger.debug(f"Processing event: {event.event_type} for agent {event.agent_id}")
            
            # Get handler for event type
            handler = self._event_handlers.get(event.event_type)
            
            if handler:
                await handler(event)
            else:
                logger.warning(f"No handler found for event type: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_type}: {e}")
            
            # Retry logic
            if event.retry_count < event.max_retries:
                event.retry_count += 1
                await asyncio.sleep(2 ** event.retry_count)  # Exponential backoff
                await self._event_queue.put(event)
    
    # Event handlers
    
    async def _handle_agent_error(self, event: OrchestrationEvent) -> None:
        """Handle agent error event."""
        agent_id = event.agent_id
        error_type = event.payload.get("error_type", "unknown")
        
        await self.trigger_recovery(agent_id, error_type, failure_severity=3)
    
    async def _handle_consolidation_failure(self, event: OrchestrationEvent) -> None:
        """Handle consolidation failure event."""
        agent_id = event.agent_id
        
        # Try recovery with consolidation focus
        await self.trigger_recovery(agent_id, "consolidation_failure", failure_severity=2)
    
    async def _handle_checkpoint_failure(self, event: OrchestrationEvent) -> None:
        """Handle checkpoint failure event."""
        agent_id = event.agent_id
        
        # Try recovery with checkpoint recreation
        await self.trigger_recovery(agent_id, "checkpoint_failure", failure_severity=2)
    
    async def _handle_recovery_required(self, event: OrchestrationEvent) -> None:
        """Handle recovery required event."""
        agent_id = event.agent_id
        severity = event.payload.get("severity", 1)
        
        await self.trigger_recovery(agent_id, "manual_recovery", failure_severity=severity)
    
    async def _handle_proactive_operation(self, event: OrchestrationEvent) -> None:
        """Handle proactive operation event."""
        operation_type = event.payload.get("operation_type")
        agent_id = event.agent_id
        
        if operation_type == "sleep":
            await self._sleep_wake_manager.initiate_sleep_cycle(agent_id, "proactive")
        elif operation_type == "wake":
            await self._sleep_wake_manager.initiate_wake_cycle(agent_id)
        elif operation_type == "maintenance_sleep":
            await self._sleep_wake_manager.initiate_sleep_cycle(agent_id, "maintenance")
    
    async def _handle_health_degradation(self, event: OrchestrationEvent) -> None:
        """Handle health degradation event."""
        health_score = event.payload.get("health_score", 0)
        
        if health_score < 50:
            # Critical health degradation - switch to emergency mode
            self.strategy = OrchestrationStrategy.EMERGENCY
            logger.warning(f"Health critically degraded ({health_score}), switching to emergency mode")
        elif health_score < 70:
            # Moderate degradation - switch to maintenance mode
            self.strategy = OrchestrationStrategy.MAINTENANCE
            logger.warning(f"Health moderately degraded ({health_score}), switching to maintenance mode")
    
    async def _handle_performance_issues(self, event: OrchestrationEvent) -> None:
        """Handle performance threshold exceeded event."""
        await self._perform_system_optimization()
    
    async def _handle_system_overload(self, event: OrchestrationEvent) -> None:
        """Handle system overload event."""
        # Temporarily reduce orchestration frequency
        self.orchestration_interval_seconds = min(120, self.orchestration_interval_seconds * 2)
        
        # Schedule sleep for active agents to reduce load
        active_agents = await self._get_active_agents()
        
        for agent_id in active_agents[:3]:  # Sleep up to 3 agents
            await self.schedule_proactive_operation(
                "sleep",
                agent_id,
                datetime.utcnow() + timedelta(seconds=30)
            )
    
    # Helper methods
    
    async def _create_recovery_plan(
        self, 
        agent_id: Optional[UUID], 
        failure_type: str, 
        failure_severity: int
    ) -> Optional[RecoveryPlan]:
        """Create a recovery plan based on failure type and severity."""
        try:
            recovery_plan = RecoveryPlan(
                agent_id=agent_id,
                failure_type=failure_type,
                failure_severity=failure_severity
            )
            
            # Determine recovery tier based on severity
            if failure_severity >= 4:
                recovery_plan.recovery_tier = RecoveryTier.EMERGENCY
            elif failure_severity >= 3:
                recovery_plan.recovery_tier = RecoveryTier.MANUAL
            elif failure_severity >= 2:
                recovery_plan.recovery_tier = RecoveryTier.ASSISTED
            else:
                recovery_plan.recovery_tier = RecoveryTier.AUTOMATIC
            
            # Create strategies based on failure type
            if failure_type == "consolidation_failure":
                recovery_plan.strategies = [
                    "restart_consolidation_engine",
                    "recreate_checkpoint",
                    "fallback_to_previous_checkpoint"
                ]
            elif failure_type == "checkpoint_failure":
                recovery_plan.strategies = [
                    "validate_checkpoint_integrity", 
                    "recreate_checkpoint",
                    "use_fallback_checkpoint"
                ]
            elif failure_type == "agent_error":
                recovery_plan.strategies = [
                    "reset_agent_state",
                    "restart_agent_process",
                    "restore_from_checkpoint"
                ]
            else:
                recovery_plan.strategies = [
                    "standard_recovery",
                    "restore_from_checkpoint",
                    "emergency_restart"
                ]
            
            # Estimate recovery time and success probability
            recovery_plan.estimated_recovery_time = len(recovery_plan.strategies) * 30.0  # 30s per strategy
            recovery_plan.success_probability = max(0.9 - failure_severity * 0.1, 0.3)
            
            # Create fallback plans for higher severity failures
            if failure_severity >= 3:
                fallback_plan = RecoveryPlan(
                    agent_id=agent_id,
                    failure_type=f"{failure_type}_fallback",
                    failure_severity=failure_severity - 1,
                    recovery_tier=RecoveryTier.EMERGENCY,
                    strategies=["emergency_recovery", "manual_intervention_required"]
                )
                recovery_plan.fallback_plans.append(fallback_plan)
            
            return recovery_plan
            
        except Exception as e:
            logger.error(f"Error creating recovery plan: {e}")
            return None
    
    async def _execute_recovery_plan(self, recovery_plan: RecoveryPlan) -> bool:
        """Execute a recovery plan."""
        try:
            logger.info(f"Executing recovery plan {recovery_plan.recovery_id} for agent {recovery_plan.agent_id}")
            
            # Track active recovery
            self._active_recovery_plans[recovery_plan.recovery_id] = recovery_plan
            
            # Execute strategies in order
            for strategy in recovery_plan.strategies:
                try:
                    success = await self._execute_recovery_strategy(strategy, recovery_plan.agent_id)
                    
                    if success:
                        logger.info(f"Recovery strategy '{strategy}' succeeded")
                        self._operation_metrics["successful_recoveries"] += 1
                        return True
                    else:
                        logger.warning(f"Recovery strategy '{strategy}' failed, trying next")
                        
                except Exception as e:
                    logger.error(f"Error executing recovery strategy '{strategy}': {e}")
                    continue
            
            logger.error(f"All recovery strategies failed for plan {recovery_plan.recovery_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error executing recovery plan: {e}")
            return False
        finally:
            self._operation_metrics["recovery_operations"] += 1
    
    async def _execute_recovery_strategy(self, strategy: str, agent_id: Optional[UUID]) -> bool:
        """Execute a specific recovery strategy."""
        try:
            if strategy == "restart_consolidation_engine":
                # Restart consolidation for the agent
                if agent_id:
                    return await self._consolidation_engine.schedule_background_consolidation(agent_id)
                return False
            
            elif strategy == "recreate_checkpoint":
                # Create a new checkpoint
                if agent_id:
                    checkpoint = await self._recovery_manager.checkpoint_manager.create_checkpoint(
                        agent_id=agent_id,
                        checkpoint_type=CheckpointType.ERROR_RECOVERY
                    )
                    return checkpoint is not None
                return False
            
            elif strategy == "fallback_to_previous_checkpoint":
                # Use recovery manager to restore from previous checkpoint
                success, _ = await self._recovery_manager.initiate_recovery(agent_id, recovery_type="automatic")
                return success
            
            elif strategy == "validate_checkpoint_integrity":
                # Validate the latest checkpoint
                if agent_id:
                    checkpoints = await self._recovery_manager.checkpoint_manager.get_checkpoint_fallbacks(agent_id, 1)
                    if checkpoints:
                        is_valid, _ = await self._recovery_manager.checkpoint_manager.validate_checkpoint(checkpoints[0].id)
                        return is_valid
                return False
            
            elif strategy == "reset_agent_state":
                # Reset agent to awake state
                if agent_id:
                    async with get_async_session() as session:
                        agent = await session.get(Agent, agent_id)
                        if agent:
                            agent.current_sleep_state = SleepState.AWAKE
                            agent.current_cycle_id = None
                            await session.commit()
                            return True
                return False
            
            elif strategy == "emergency_recovery":
                # Use emergency recovery
                return await self._recovery_manager.emergency_recovery(agent_id)
            
            elif strategy == "standard_recovery":
                # Use standard recovery
                success, _ = await self._recovery_manager.initiate_recovery(agent_id, recovery_type="automatic")
                return success
            
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing recovery strategy '{strategy}': {e}")
            return False
    
    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        try:
            health_components = []
            
            # Get system status
            system_status = await self._sleep_wake_manager.get_system_status()
            
            # Calculate component health scores
            if system_status.get("system_healthy", False):
                health_components.append(100.0)
            else:
                health_components.append(50.0)
            
            # Success rate health
            metrics = system_status.get("metrics", {})
            if metrics.get("total_sleep_cycles", 0) > 0:
                sleep_success_rate = metrics.get("successful_sleep_cycles", 0) / metrics["total_sleep_cycles"]
                health_components.append(sleep_success_rate * 100)
            
            if metrics.get("total_wake_cycles", 0) > 0:
                wake_success_rate = metrics.get("successful_wake_cycles", 0) / metrics["total_wake_cycles"]
                health_components.append(wake_success_rate * 100)
            
            # Circuit breaker health
            circuit_health = []
            for breaker in self._circuit_breakers.values():
                if breaker.state == CircuitBreakerState.CLOSED:
                    circuit_health.append(100.0)
                elif breaker.state == CircuitBreakerState.HALF_OPEN:
                    circuit_health.append(50.0)
                else:  # OPEN
                    circuit_health.append(0.0)
            
            if circuit_health:
                health_components.append(sum(circuit_health) / len(circuit_health))
            
            # Calculate overall health
            if health_components:
                return sum(health_components) / len(health_components)
            else:
                return 50.0  # Default moderate health
                
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return 0.0  # Critical error state
    
    async def _perform_system_optimization(self) -> None:
        """Perform system optimization."""
        try:
            # Use sleep-wake manager optimization
            await self._sleep_wake_manager.optimize_performance()
            
            # Optimize consolidation engine
            if hasattr(self._consolidation_engine, 'enable_background_optimization'):
                await self._consolidation_engine.enable_background_optimization()
            
            # Analytics optimization
            await self._analytics_engine.update_daily_analytics()
            
            logger.info("System optimization completed")
            
        except Exception as e:
            logger.error(f"Error during system optimization: {e}")
    
    async def _get_agents_needing_proactive_sleep(self) -> List[UUID]:
        """Get agents that need proactive sleep scheduling."""
        try:
            # Use intelligent sleep manager to identify candidates
            if hasattr(self._intelligent_sleep_manager, 'get_sleep_candidates'):
                return await self._intelligent_sleep_manager.get_sleep_candidates()
            else:
                # Fallback: get agents that have been awake for a long time
                async with get_async_session() as session:
                    cutoff_time = datetime.utcnow() - timedelta(hours=4)
                    
                    result = await session.execute(
                        select(Agent.id).where(
                            and_(
                                Agent.current_sleep_state == SleepState.AWAKE,
                                or_(
                                    Agent.last_wake_time < cutoff_time,
                                    Agent.last_wake_time.is_(None)
                                )
                            )
                        )
                    )
                    
                    return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error getting agents needing proactive sleep: {e}")
            return []
    
    async def _get_agents_needing_wake(self) -> List[UUID]:
        """Get agents that should be woken up."""
        try:
            async with get_async_session() as session:
                # Get agents that have been sleeping for more than expected
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                result = await session.execute(
                    select(Agent.id).where(
                        and_(
                            Agent.current_sleep_state == SleepState.SLEEPING,
                            Agent.last_sleep_time < cutoff_time
                        )
                    )
                )
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting agents needing wake: {e}")
            return []
    
    async def _get_idle_agents(self) -> List[UUID]:
        """Get agents that are idle and suitable for maintenance."""
        try:
            async with get_async_session() as session:
                # Get agents that have been awake but inactive
                cutoff_time = datetime.utcnow() - timedelta(hours=2)
                
                result = await session.execute(
                    select(Agent.id).where(
                        and_(
                            Agent.current_sleep_state == SleepState.AWAKE,
                            Agent.status == AgentStatus.active,
                            or_(
                                Agent.last_wake_time < cutoff_time,
                                Agent.last_wake_time.is_(None)
                            )
                        )
                    ).limit(2)  # Limit to avoid too many maintenance operations
                )
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting idle agents: {e}")
            return []
    
    async def _get_active_agents(self) -> List[UUID]:
        """Get currently active agents."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(Agent.id).where(
                        and_(
                            Agent.current_sleep_state == SleepState.AWAKE,
                            Agent.status == AgentStatus.active
                        )
                    )
                )
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting active agents: {e}")
            return []
    
    async def _get_agents_in_critical_state(self) -> List[UUID]:
        """Get agents in critical error state."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(Agent.id).where(
                        Agent.current_sleep_state == SleepState.ERROR
                    )
                )
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting agents in critical state: {e}")
            return []
    
    async def _schedule_proactive_sleep(self, agent_id: UUID) -> None:
        """Schedule proactive sleep for an agent."""
        await self._sleep_wake_manager.initiate_sleep_cycle(agent_id, "proactive")
    
    async def _schedule_proactive_wake(self, agent_id: UUID) -> None:
        """Schedule proactive wake for an agent."""
        await self._sleep_wake_manager.initiate_wake_cycle(agent_id)
    
    async def _should_escalate_recovery(self, recovery_plan: RecoveryPlan) -> bool:
        """Check if recovery plan should be escalated."""
        # Simple time-based escalation
        elapsed_time = time.time() - recovery_plan.estimated_recovery_time
        return elapsed_time > 300  # 5 minutes
    
    async def _escalate_recovery(self, recovery_plan: RecoveryPlan) -> None:
        """Escalate recovery plan to higher tier."""
        if recovery_plan.recovery_tier == RecoveryTier.AUTOMATIC:
            recovery_plan.recovery_tier = RecoveryTier.ASSISTED
        elif recovery_plan.recovery_tier == RecoveryTier.ASSISTED:
            recovery_plan.recovery_tier = RecoveryTier.MANUAL
        elif recovery_plan.recovery_tier == RecoveryTier.MANUAL:
            recovery_plan.recovery_tier = RecoveryTier.EMERGENCY
        
        logger.warning(f"Escalated recovery plan {recovery_plan.recovery_id} to {recovery_plan.recovery_tier.value}")
    
    async def _is_recovery_complete(self, recovery_plan: RecoveryPlan) -> bool:
        """Check if recovery plan is complete."""
        if not recovery_plan.agent_id:
            return True  # System-wide recovery assumed complete
        
        try:
            async with get_async_session() as session:
                agent = await session.get(Agent, recovery_plan.agent_id)
                if agent and agent.current_sleep_state != SleepState.ERROR:
                    return True
                    
        except Exception:
            pass
        
        return False
    
    def _update_average_metric(self, metric_name: str, new_value: float) -> None:
        """Update a rolling average metric."""
        current_avg = self._operation_metrics.get(metric_name, 0)
        # Simple exponential moving average with alpha = 0.1
        self._operation_metrics[metric_name] = current_avg * 0.9 + new_value * 0.1


# Global automated orchestrator instance
_automated_orchestrator_instance: Optional[AutomatedOrchestrator] = None


async def get_automated_orchestrator() -> AutomatedOrchestrator:
    """Get the global automated orchestrator instance."""
    global _automated_orchestrator_instance
    if _automated_orchestrator_instance is None:
        _automated_orchestrator_instance = AutomatedOrchestrator()
        await _automated_orchestrator_instance.initialize()
    return _automated_orchestrator_instance


async def start_automated_orchestration() -> bool:
    """Start automated orchestration."""
    try:
        orchestrator = await get_automated_orchestrator()
        await orchestrator.start_orchestration()
        return True
    except Exception as e:
        logger.error(f"Failed to start automated orchestration: {e}")
        return False


async def stop_automated_orchestration() -> bool:
    """Stop automated orchestration."""
    try:
        global _automated_orchestrator_instance
        if _automated_orchestrator_instance:
            await _automated_orchestrator_instance.stop_orchestration()
        return True
    except Exception as e:
        logger.error(f"Failed to stop automated orchestration: {e}")
        return False