"""
Enhanced Error Handling for Workflow Engine - LeanVibe Agent Hive 2.0 - VS 3.3

Comprehensive error handling and recovery mechanisms for workflow execution:
- Circuit breaker integration for task execution
- Intelligent retry policies for transient failures
- Graceful degradation for workflow dependencies
- Comprehensive error classification and recovery strategies
- Performance monitoring with <30s recovery time targets
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

from .circuit_breaker import CircuitBreaker, CircuitBreakerState, get_circuit_breaker
from .retry_policies import RetryPolicyFactory, RetryConfig, RetryStrategy, JitterType, RetryExecutor
from .graceful_degradation import GracefulDegradationManager, DegradationLevel, get_degradation_manager
from .error_handling_integration import get_error_handling_integration

logger = structlog.get_logger()


class WorkflowErrorType(Enum):
    """Types of workflow execution errors."""
    TASK_EXECUTION_FAILURE = "task_execution_failure"
    DEPENDENCY_RESOLUTION_ERROR = "dependency_resolution_error"
    AGENT_COMMUNICATION_ERROR = "agent_communication_error"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    TIMEOUT_ERROR = "timeout_error"
    DATABASE_ERROR = "database_error"
    SEMANTIC_MEMORY_ERROR = "semantic_memory_error"
    WORKFLOW_STATE_ERROR = "workflow_state_error"
    BATCH_EXECUTION_ERROR = "batch_execution_error"
    RECOVERY_FAILURE = "recovery_failure"


class WorkflowRecoveryStrategy(Enum):
    """Recovery strategies for workflow errors."""
    RETRY_TASK = "retry_task"
    SKIP_TASK = "skip_task"
    RETRY_BATCH = "retry_batch"
    WORKFLOW_RESTART = "workflow_restart"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    FAILOVER_AGENT = "failover_agent"
    CHECKPOINT_ROLLBACK = "checkpoint_rollback"


@dataclass
class WorkflowErrorContext:
    """Context information for workflow errors."""
    workflow_id: str
    task_id: Optional[str] = None
    batch_id: Optional[str] = None
    agent_id: Optional[str] = None
    error_type: WorkflowErrorType = WorkflowErrorType.TASK_EXECUTION_FAILURE
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    execution_time_ms: float = 0.0
    affected_dependencies: List[str] = field(default_factory=list)
    recovery_suggestions: List[str] = field(default_factory=list)
    
    # Performance context
    processing_start_time: datetime = field(default_factory=datetime.utcnow)
    timeout_threshold_ms: float = 30000.0  # 30 seconds
    
    # State context
    workflow_status: str = "unknown"
    checkpoint_available: bool = False
    semantic_memory_affected: bool = False


@dataclass
class WorkflowErrorHandlingConfig:
    """Configuration for workflow error handling."""
    enabled: bool = True
    max_task_retries: int = 3
    max_batch_retries: int = 2
    max_workflow_retries: int = 1
    
    # Timeout configurations
    task_timeout_ms: int = 300000      # 5 minutes
    batch_timeout_ms: int = 1800000    # 30 minutes
    workflow_timeout_ms: int = 3600000 # 1 hour
    
    # Circuit breaker settings
    enable_circuit_breakers: bool = True
    task_failure_threshold: int = 5
    agent_failure_threshold: int = 10
    
    # Retry settings
    base_retry_delay_ms: int = 1000
    max_retry_delay_ms: int = 30000
    retry_jitter_enabled: bool = True
    
    # Recovery settings
    enable_graceful_degradation: bool = True
    enable_checkpoint_recovery: bool = True
    recovery_timeout_ms: int = 30000  # 30 seconds target
    
    # Observability settings
    enable_detailed_logging: bool = True
    emit_recovery_events: bool = True


class WorkflowErrorAnalyzer:
    """Analyzes workflow errors to determine optimal recovery strategies."""
    
    def __init__(self, config: WorkflowErrorHandlingConfig):
        self.config = config
        self.error_patterns: Dict[str, List[Tuple[datetime, str]]] = {}
        
    def analyze_error(
        self,
        error: Exception,
        context: WorkflowErrorContext
    ) -> Tuple[WorkflowRecoveryStrategy, Dict[str, Any]]:
        """
        Analyze workflow error and determine recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Workflow error context
            
        Returns:
            Tuple of (recovery_strategy, recovery_parameters)
        """
        # Record error pattern
        self._record_error_pattern(context.workflow_id, str(error))
        
        # Classify error type
        error_type = self._classify_error(error, context)
        context.error_type = error_type
        
        # Determine recovery strategy based on error type and context
        strategy, params = self._determine_recovery_strategy(error_type, context)
        
        # Add recovery suggestions
        context.recovery_suggestions = self._generate_recovery_suggestions(
            error_type, context, strategy
        )
        
        logger.info(
            "🔍 Workflow error analyzed",
            workflow_id=context.workflow_id,
            error_type=error_type.value,
            recovery_strategy=strategy.value,
            retry_count=context.retry_count,
            suggestions=len(context.recovery_suggestions)
        )
        
        return strategy, params
    
    def _classify_error(self, error: Exception, context: WorkflowErrorContext) -> WorkflowErrorType:
        """Classify error into workflow error type."""
        
        error_type_name = type(error).__name__
        error_message = str(error).lower()
        
        # Database-related errors
        if "database" in error_message or "sql" in error_message:
            return WorkflowErrorType.DATABASE_ERROR
        
        # Timeout errors
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)) or "timeout" in error_message:
            return WorkflowErrorType.TIMEOUT_ERROR
        
        # Communication errors
        if "connection" in error_message or "network" in error_message:
            return WorkflowErrorType.AGENT_COMMUNICATION_ERROR
        
        # Resource errors
        if "resource" in error_message or "unavailable" in error_message:
            return WorkflowErrorType.RESOURCE_UNAVAILABLE
        
        # Semantic memory errors
        if context.semantic_memory_affected or "semantic" in error_message:
            return WorkflowErrorType.SEMANTIC_MEMORY_ERROR
        
        # Dependency errors
        if "dependency" in error_message or context.affected_dependencies:
            return WorkflowErrorType.DEPENDENCY_RESOLUTION_ERROR
        
        # Batch execution errors
        if context.batch_id and "batch" in error_message:
            return WorkflowErrorType.BATCH_EXECUTION_ERROR
        
        # Workflow state errors
        if "workflow" in error_message and "state" in error_message:
            return WorkflowErrorType.WORKFLOW_STATE_ERROR
        
        # Default to task execution failure
        return WorkflowErrorType.TASK_EXECUTION_FAILURE
    
    def _determine_recovery_strategy(
        self,
        error_type: WorkflowErrorType,
        context: WorkflowErrorContext
    ) -> Tuple[WorkflowRecoveryStrategy, Dict[str, Any]]:
        """Determine optimal recovery strategy based on error type and context."""
        
        params = {}
        
        # Check retry limits first
        if self._should_retry(error_type, context):
            if context.task_id and context.retry_count < self.config.max_task_retries:
                return WorkflowRecoveryStrategy.RETRY_TASK, {
                    "delay_ms": self._calculate_retry_delay(context.retry_count),
                    "max_retries": self.config.max_task_retries
                }
            elif context.batch_id and context.retry_count < self.config.max_batch_retries:
                return WorkflowRecoveryStrategy.RETRY_BATCH, {
                    "delay_ms": self._calculate_retry_delay(context.retry_count),
                    "max_retries": self.config.max_batch_retries
                }
        
        # Strategy based on error type
        if error_type == WorkflowErrorType.TASK_EXECUTION_FAILURE:
            if context.retry_count >= self.config.max_task_retries:
                return WorkflowRecoveryStrategy.SKIP_TASK, {
                    "mark_as_failed": True,
                    "continue_workflow": True
                }
            
        elif error_type == WorkflowErrorType.AGENT_COMMUNICATION_ERROR:
            return WorkflowRecoveryStrategy.FAILOVER_AGENT, {
                "exclude_agent": context.agent_id,
                "require_capabilities": context.error_details.get("required_capabilities", [])
            }
            
        elif error_type == WorkflowErrorType.TIMEOUT_ERROR:
            if context.checkpoint_available:
                return WorkflowRecoveryStrategy.CHECKPOINT_ROLLBACK, {
                    "rollback_to_last_checkpoint": True,
                    "increase_timeout": True
                }
            else:
                return WorkflowRecoveryStrategy.GRACEFUL_DEGRADATION, {
                    "degradation_level": DegradationLevel.PARTIAL.value
                }
                
        elif error_type == WorkflowErrorType.DATABASE_ERROR:
            return WorkflowRecoveryStrategy.RETRY_TASK, {
                "delay_ms": 5000,  # Longer delay for DB errors
                "use_circuit_breaker": True
            }
            
        elif error_type == WorkflowErrorType.SEMANTIC_MEMORY_ERROR:
            return WorkflowRecoveryStrategy.GRACEFUL_DEGRADATION, {
                "degradation_level": DegradationLevel.MINIMAL.value,
                "disable_semantic_features": True
            }
            
        elif error_type == WorkflowErrorType.RESOURCE_UNAVAILABLE:
            return WorkflowRecoveryStrategy.GRACEFUL_DEGRADATION, {
                "degradation_level": DegradationLevel.PARTIAL.value,
                "reduce_parallelism": True
            }
            
        elif error_type in [WorkflowErrorType.WORKFLOW_STATE_ERROR, WorkflowErrorType.RECOVERY_FAILURE]:
            return WorkflowRecoveryStrategy.MANUAL_INTERVENTION, {
                "escalate_to_human": True,
                "preserve_state": True
            }
        
        # Default strategy
        return WorkflowRecoveryStrategy.GRACEFUL_DEGRADATION, {
            "degradation_level": DegradationLevel.MINIMAL.value
        }
    
    def _should_retry(self, error_type: WorkflowErrorType, context: WorkflowErrorContext) -> bool:
        """Determine if error is retryable."""
        
        # Non-retryable error types
        non_retryable = {
            WorkflowErrorType.WORKFLOW_STATE_ERROR,
            WorkflowErrorType.RECOVERY_FAILURE
        }
        
        if error_type in non_retryable:
            return False
        
        # Check timeout
        elapsed_time = (datetime.utcnow() - context.processing_start_time).total_seconds() * 1000
        if elapsed_time > context.timeout_threshold_ms:
            return False
        
        # Check error frequency for this workflow
        if self._is_error_frequent(context.workflow_id):
            return False
        
        return True
    
    def _calculate_retry_delay(self, retry_count: int) -> int:
        """Calculate retry delay with exponential backoff."""
        delay = min(
            self.config.base_retry_delay_ms * (2 ** retry_count),
            self.config.max_retry_delay_ms
        )
        
        if self.config.retry_jitter_enabled:
            import random
            jitter = random.uniform(0.5, 1.5)
            delay = int(delay * jitter)
        
        return delay
    
    def _record_error_pattern(self, workflow_id: str, error_message: str) -> None:
        """Record error pattern for analysis."""
        if workflow_id not in self.error_patterns:
            self.error_patterns[workflow_id] = []
        
        current_time = datetime.utcnow()
        self.error_patterns[workflow_id].append((current_time, error_message))
        
        # Keep only recent errors (last hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.error_patterns[workflow_id] = [
            (time, msg) for time, msg in self.error_patterns[workflow_id]
            if time > cutoff_time
        ]
    
    def _is_error_frequent(self, workflow_id: str) -> bool:
        """Check if errors are occurring frequently for a workflow."""
        if workflow_id not in self.error_patterns:
            return False
        
        recent_errors = self.error_patterns[workflow_id]
        if len(recent_errors) < 5:
            return False
        
        # Check if 5+ errors in last 10 minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        recent_frequent_errors = [
            (time, msg) for time, msg in recent_errors
            if time > cutoff_time
        ]
        
        return len(recent_frequent_errors) >= 5
    
    def _generate_recovery_suggestions(
        self,
        error_type: WorkflowErrorType,
        context: WorkflowErrorContext,
        strategy: WorkflowRecoveryStrategy
    ) -> List[str]:
        """Generate recovery suggestions for the error."""
        
        suggestions = []
        
        if error_type == WorkflowErrorType.TIMEOUT_ERROR:
            suggestions.extend([
                "Consider increasing task timeout configuration",
                "Check for resource contention or blocking operations",
                "Review workflow dependency chain for bottlenecks"
            ])
            
        elif error_type == WorkflowErrorType.AGENT_COMMUNICATION_ERROR:
            suggestions.extend([
                "Verify agent connectivity and health status",
                "Check network configuration and firewall rules",
                "Consider increasing communication timeout"
            ])
            
        elif error_type == WorkflowErrorType.DATABASE_ERROR:
            suggestions.extend([
                "Check database connection pool and limits",
                "Verify database health and performance",
                "Review transaction isolation and locking"
            ])
            
        elif error_type == WorkflowErrorType.RESOURCE_UNAVAILABLE:
            suggestions.extend([
                "Scale up available resources if possible",
                "Implement resource queuing or throttling",
                "Consider workflow scheduling optimization"
            ])
        
        if context.retry_count > 2:
            suggestions.append("High retry count detected - investigate root cause")
        
        if strategy == WorkflowRecoveryStrategy.MANUAL_INTERVENTION:
            suggestions.extend([
                "Human intervention required for safe recovery",
                "Preserve workflow state for debugging",
                "Contact system administrator for assistance"
            ])
        
        return suggestions


class WorkflowErrorRecoveryManager:
    """Manages error recovery for workflow execution."""
    
    def __init__(self, config: WorkflowErrorHandlingConfig):
        self.config = config
        self.error_analyzer = WorkflowErrorAnalyzer(config)
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Degradation manager
        self.degradation_manager = get_degradation_manager()
        
        # Integration with observability
        self.integration = get_error_handling_integration()
        
        # Recovery metrics
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_times: List[float] = []
        
        logger.info(
            "🛡️ Workflow error recovery manager initialized",
            circuit_breakers_enabled=config.enable_circuit_breakers,
            graceful_degradation_enabled=config.enable_graceful_degradation,
            checkpoint_recovery_enabled=config.enable_checkpoint_recovery
        )
    
    async def handle_workflow_error(
        self,
        error: Exception,
        context: WorkflowErrorContext,
        recovery_callback: Optional[Callable] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle workflow error with comprehensive recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Workflow error context
            recovery_callback: Optional callback for custom recovery
            
        Returns:
            Tuple of (recovery_success, recovery_details)
        """
        start_time = time.time()
        self.recovery_attempts += 1
        
        try:
            # Analyze error and determine recovery strategy
            strategy, params = self.error_analyzer.analyze_error(error, context)
            
            logger.info(
                "🚨 Handling workflow error",
                workflow_id=context.workflow_id,
                error_type=context.error_type.value,
                recovery_strategy=strategy.value,
                retry_count=context.retry_count
            )
            
            # Execute recovery strategy
            recovery_success, recovery_details = await self._execute_recovery_strategy(
                strategy, params, context, error
            )
            
            # Record recovery metrics
            recovery_time = (time.time() - start_time) * 1000
            self.recovery_times.append(recovery_time)
            
            if recovery_success:
                self.successful_recoveries += 1
                
                # Emit recovery event
                if self.config.emit_recovery_events:
                    await self.integration.emit_error_handling_failure(
                        error_type=context.error_type.value,
                        error_message=f"Workflow error recovered using {strategy.value}",
                        component="workflow_recovery_manager",
                        context={
                            "workflow_id": context.workflow_id,
                            "recovery_strategy": strategy.value,
                            "recovery_time_ms": recovery_time,
                            "retry_count": context.retry_count,
                            "recovery_details": recovery_details
                        },
                        agent_id=uuid.UUID(context.agent_id) if context.agent_id else None
                    )
                
                logger.info(
                    "✅ Workflow error recovery successful",
                    workflow_id=context.workflow_id,
                    strategy=strategy.value,
                    recovery_time_ms=round(recovery_time, 2)
                )
            else:
                self.failed_recoveries += 1
                
                # Emit failure event
                await self.integration.emit_error_handling_failure(
                    error_type="workflow_recovery_failure",
                    error_message=f"Failed to recover workflow error: {str(error)}",
                    component="workflow_recovery_manager",
                    context={
                        "workflow_id": context.workflow_id,
                        "original_error_type": context.error_type.value,
                        "attempted_strategy": strategy.value,
                        "recovery_time_ms": recovery_time,
                        "retry_count": context.retry_count,
                        "recovery_details": recovery_details
                    },
                    agent_id=uuid.UUID(context.agent_id) if context.agent_id else None
                )
                
                logger.error(
                    "❌ Workflow error recovery failed",
                    workflow_id=context.workflow_id,
                    strategy=strategy.value,
                    recovery_time_ms=round(recovery_time, 2),
                    error=str(error)
                )
            
            return recovery_success, recovery_details
            
        except Exception as recovery_error:
            recovery_time = (time.time() - start_time) * 1000
            self.failed_recoveries += 1
            
            logger.error(
                "💥 Critical error in workflow recovery",
                workflow_id=context.workflow_id,
                original_error=str(error),
                recovery_error=str(recovery_error),
                recovery_time_ms=round(recovery_time, 2),
                exc_info=True
            )
            
            return False, {
                "error": "Critical recovery failure",
                "original_error": str(error),
                "recovery_error": str(recovery_error),
                "recovery_time_ms": recovery_time
            }
    
    async def _execute_recovery_strategy(
        self,
        strategy: WorkflowRecoveryStrategy,
        params: Dict[str, Any],
        context: WorkflowErrorContext,
        original_error: Exception
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute specific recovery strategy."""
        
        try:
            if strategy == WorkflowRecoveryStrategy.RETRY_TASK:
                return await self._retry_task_recovery(params, context)
                
            elif strategy == WorkflowRecoveryStrategy.RETRY_BATCH:
                return await self._retry_batch_recovery(params, context)
                
            elif strategy == WorkflowRecoveryStrategy.SKIP_TASK:
                return await self._skip_task_recovery(params, context)
                
            elif strategy == WorkflowRecoveryStrategy.FAILOVER_AGENT:
                return await self._failover_agent_recovery(params, context)
                
            elif strategy == WorkflowRecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation_recovery(params, context, original_error)
                
            elif strategy == WorkflowRecoveryStrategy.CHECKPOINT_ROLLBACK:
                return await self._checkpoint_rollback_recovery(params, context)
                
            elif strategy == WorkflowRecoveryStrategy.MANUAL_INTERVENTION:
                return await self._manual_intervention_recovery(params, context)
                
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False, {"error": f"Unknown strategy: {strategy}"}
                
        except Exception as strategy_error:
            logger.error(
                f"Error executing recovery strategy {strategy}",
                error=str(strategy_error),
                workflow_id=context.workflow_id
            )
            return False, {"error": str(strategy_error)}
    
    async def _retry_task_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute task retry recovery."""
        
        delay_ms = params.get("delay_ms", 1000)
        
        # Wait before retry
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        
        # Use circuit breaker if specified
        if params.get("use_circuit_breaker", False) and context.task_id:
            circuit_breaker = self._get_circuit_breaker(f"task_{context.task_id}")
            
            try:
                # Check if circuit breaker allows the request
                state = await circuit_breaker.get_state()
                if state == CircuitBreakerState.OPEN:
                    return False, {"error": "Circuit breaker is open for task"}
            except Exception as cb_error:
                logger.warning(f"Circuit breaker check failed: {cb_error}")
        
        return True, {
            "strategy": "retry_task",
            "delay_ms": delay_ms,
            "retry_count": context.retry_count + 1
        }
    
    async def _retry_batch_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute batch retry recovery."""
        
        delay_ms = params.get("delay_ms", 2000)
        
        # Wait before retry
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        
        return True, {
            "strategy": "retry_batch",
            "delay_ms": delay_ms,
            "retry_count": context.retry_count + 1
        }
    
    async def _skip_task_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute skip task recovery."""
        
        mark_as_failed = params.get("mark_as_failed", True)
        continue_workflow = params.get("continue_workflow", True)
        
        return True, {
            "strategy": "skip_task",
            "task_skipped": context.task_id,
            "marked_as_failed": mark_as_failed,
            "continue_workflow": continue_workflow
        }
    
    async def _failover_agent_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute agent failover recovery."""
        
        excluded_agent = params.get("exclude_agent")
        required_capabilities = params.get("require_capabilities", [])
        
        # Record agent failure in circuit breaker
        if excluded_agent and self.config.enable_circuit_breakers:
            agent_circuit_breaker = self._get_circuit_breaker(f"agent_{excluded_agent}")
            await agent_circuit_breaker.record_failure()
        
        return True, {
            "strategy": "failover_agent",
            "excluded_agent": excluded_agent,
            "required_capabilities": required_capabilities,
            "find_alternative_agent": True
        }
    
    async def _graceful_degradation_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext,
        original_error: Exception
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute graceful degradation recovery."""
        
        degradation_level_str = params.get("degradation_level", DegradationLevel.MINIMAL.value)
        
        # Convert string to enum
        degradation_level = DegradationLevel(degradation_level_str)
        
        # Apply degradation
        service_path = f"/workflows/{context.workflow_id}"
        
        degraded_response = await self.degradation_manager.apply_degradation(
            service_path=service_path,
            degradation_level=degradation_level,
            error_context={
                "error_type": context.error_type.value,
                "error_message": context.error_message,
                "workflow_id": context.workflow_id,
                "task_id": context.task_id
            }
        )
        
        # Emit degradation event
        await self.integration.emit_graceful_degradation_event(
            service_path=service_path,
            degradation_level=degradation_level,
            fallback_strategy="workflow_degradation",
            success=degraded_response is not None,
            error_context=context.error_details,
            agent_id=uuid.UUID(context.agent_id) if context.agent_id else None
        )
        
        return degraded_response is not None, {
            "strategy": "graceful_degradation",
            "degradation_level": degradation_level.value,
            "degraded_response_available": degraded_response is not None,
            "disable_semantic_features": params.get("disable_semantic_features", False),
            "reduce_parallelism": params.get("reduce_parallelism", False)
        }
    
    async def _checkpoint_rollback_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute checkpoint rollback recovery."""
        
        rollback_to_last = params.get("rollback_to_last_checkpoint", True)
        increase_timeout = params.get("increase_timeout", False)
        
        if not context.checkpoint_available:
            return False, {"error": "No checkpoint available for rollback"}
        
        return True, {
            "strategy": "checkpoint_rollback",
            "rollback_to_last_checkpoint": rollback_to_last,
            "increase_timeout": increase_timeout,
            "checkpoint_available": True
        }
    
    async def _manual_intervention_recovery(
        self,
        params: Dict[str, Any],
        context: WorkflowErrorContext
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute manual intervention recovery."""
        
        escalate_to_human = params.get("escalate_to_human", True)
        preserve_state = params.get("preserve_state", True)
        
        # Create intervention details
        intervention_details = {
            "workflow_id": context.workflow_id,
            "error_type": context.error_type.value,
            "error_message": context.error_message,
            "suggestions": context.recovery_suggestions,
            "preserve_state": preserve_state,
            "requires_human_decision": True
        }
        
        return False, {  # Manual intervention is not automatic success
            "strategy": "manual_intervention",
            "escalated_to_human": escalate_to_human,
            "intervention_details": intervention_details,
            "workflow_paused": True
        }
    
    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = get_circuit_breaker(
                name=name,
                failure_threshold=self.config.task_failure_threshold,
                success_threshold=3,
                timeout_seconds=60
            )
        
        return self.circuit_breakers[name]
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get comprehensive recovery metrics."""
        
        avg_recovery_time = 0.0
        if self.recovery_times:
            avg_recovery_time = sum(self.recovery_times) / len(self.recovery_times)
        
        recovery_success_rate = 0.0
        if self.recovery_attempts > 0:
            recovery_success_rate = self.successful_recoveries / self.recovery_attempts
        
        # Performance target check (30s recovery time)
        target_met = avg_recovery_time <= self.config.recovery_timeout_ms
        
        return {
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_success_rate": recovery_success_rate,
            "average_recovery_time_ms": avg_recovery_time,
            "max_recovery_time_ms": max(self.recovery_times) if self.recovery_times else 0,
            "recovery_target_met": target_met,
            "target_recovery_time_ms": self.config.recovery_timeout_ms,
            "circuit_breakers_active": len(self.circuit_breakers),
            "configuration": {
                "enabled": self.config.enabled,
                "max_task_retries": self.config.max_task_retries,
                "max_batch_retries": self.config.max_batch_retries,
                "circuit_breakers_enabled": self.config.enable_circuit_breakers,
                "graceful_degradation_enabled": self.config.enable_graceful_degradation
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of recovery manager."""
        
        metrics = self.get_recovery_metrics()
        
        # Determine health status
        issues = []
        if not metrics["recovery_target_met"]:
            issues.append("Recovery time exceeds 30s target")
        
        if metrics["recovery_success_rate"] < 0.8:  # <80% success rate
            issues.append("Low recovery success rate")
        
        if metrics["failed_recoveries"] > metrics["successful_recoveries"]:
            issues.append("More failed recoveries than successful ones")
        
        # Check circuit breaker health
        circuit_breaker_issues = []
        for name, cb in self.circuit_breakers.items():
            cb_health = await cb.health_check()
            if cb_health["status"] != "healthy":
                circuit_breaker_issues.append(f"Circuit breaker {name} is {cb_health['status']}")
        
        if circuit_breaker_issues:
            issues.extend(circuit_breaker_issues)
        
        status = "healthy" if not issues else "degraded"
        
        return {
            "status": status,
            "issues": issues,
            "metrics": metrics,
            "circuit_breakers": {
                name: await cb.health_check()
                for name, cb in self.circuit_breakers.items()
            },
            "recommendations": self._get_health_recommendations(metrics, issues)
        }
    
    def _get_health_recommendations(
        self,
        metrics: Dict[str, Any],
        issues: List[str]
    ) -> List[str]:
        """Get health recommendations based on metrics and issues."""
        
        recommendations = []
        
        if not metrics["recovery_target_met"]:
            recommendations.append("Consider optimizing recovery strategies for faster resolution")
        
        if metrics["recovery_success_rate"] < 0.8:
            recommendations.append("Review and improve error analysis and recovery strategies")
        
        if metrics["failed_recoveries"] > 5:
            recommendations.append("Investigate common failure patterns and add preventive measures")
        
        if len(self.circuit_breakers) > 10:
            recommendations.append("Consider consolidating circuit breakers for better management")
        
        return recommendations


# Global recovery manager instance
_workflow_recovery_manager: Optional[WorkflowErrorRecoveryManager] = None


def get_workflow_recovery_manager(
    config: Optional[WorkflowErrorHandlingConfig] = None
) -> WorkflowErrorRecoveryManager:
    """Get or create global workflow recovery manager."""
    global _workflow_recovery_manager
    
    if _workflow_recovery_manager is None:
        _workflow_recovery_manager = WorkflowErrorRecoveryManager(
            config or WorkflowErrorHandlingConfig()
        )
    
    return _workflow_recovery_manager


def initialize_workflow_error_handling(
    config: Optional[WorkflowErrorHandlingConfig] = None
) -> WorkflowErrorRecoveryManager:
    """Initialize workflow error handling system."""
    global _workflow_recovery_manager
    
    _workflow_recovery_manager = WorkflowErrorRecoveryManager(
        config or WorkflowErrorHandlingConfig()
    )
    
    logger.info("✅ Workflow error handling system initialized")
    return _workflow_recovery_manager