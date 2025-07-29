"""
Error Handling Integration with Observability System for LeanVibe Agent Hive 2.0 - VS 3.3

Integration layer connecting error handling components with observability hooks:
- Enhanced error event emission with detailed context
- Circuit breaker state change notifications
- Retry attempt tracking and metrics
- Graceful degradation event monitoring
- Performance metrics integration
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import structlog

from .observability_hooks import get_observability_hooks
from .error_handling_middleware import ErrorHandlingConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig
from .retry_policies import RetryPolicy, RetryResult
from .graceful_degradation import GracefulDegradationManager, DegradationLevel

logger = structlog.get_logger()


class ErrorHandlingObservabilityIntegration:
    """
    Integration layer for error handling observability.
    
    Features:
    - Comprehensive error event emission
    - Circuit breaker state monitoring
    - Retry metrics tracking
    - Degradation event logging
    - Performance impact analysis
    """
    
    def __init__(self, enable_detailed_logging: bool = True):
        """Initialize error handling observability integration."""
        self.enable_detailed_logging = enable_detailed_logging
        self.observability_hooks = get_observability_hooks()
        
        # Event tracking
        self.error_events_emitted = 0
        self.recovery_events_emitted = 0
        self.degradation_events_emitted = 0
        
        logger.info(
            "🔗 Error handling observability integration initialized",
            detailed_logging=enable_detailed_logging,
            hooks_available=bool(self.observability_hooks)
        )
    
    async def emit_error_handling_failure(
        self,
        error_type: str,
        error_message: str,
        component: str,
        context: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        workflow_id: Optional[uuid.UUID] = None
    ) -> None:
        """
        Emit comprehensive error handling failure event.
        
        Args:
            error_type: Type of error that occurred
            error_message: Detailed error message
            component: Component where error occurred
            context: Error context information
            agent_id: Associated agent ID
            session_id: Associated session ID
            workflow_id: Associated workflow ID
        """
        if not self.observability_hooks:
            return
        
        try:
            # Determine severity based on error type and context
            severity = self._determine_error_severity(error_type, context)
            
            # Create comprehensive error details
            error_details = {
                "error_type": error_type,
                "error_message": error_message,
                "component": component,
                "context": context,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "error_handling_metadata": {
                    "middleware_version": "3.3",
                    "integration_enabled": True,
                    "detailed_logging": self.enable_detailed_logging
                }
            }
            
            # Add performance impact if available
            if "processing_time_ms" in context:
                error_details["performance_impact"] = {
                    "processing_time_ms": context["processing_time_ms"],
                    "performance_degraded": context["processing_time_ms"] > 5.0
                }
            
            # Add retry information if available
            if "retry_count" in context:
                error_details["retry_information"] = {
                    "retry_count": context["retry_count"],
                    "max_retries": context.get("max_retries", "unknown"),
                    "retry_strategy": context.get("retry_strategy", "unknown")
                }
            
            # Add circuit breaker information if available
            if "circuit_breaker_state" in context:
                error_details["circuit_breaker"] = {
                    "state": context["circuit_breaker_state"],
                    "failure_count": context.get("failure_count", 0),
                    "state_changed": context.get("state_changed", False)
                }
            
            # Add degradation information if available
            if "degradation_level" in context:
                error_details["graceful_degradation"] = {
                    "level": context["degradation_level"],
                    "fallback_used": context.get("fallback_used", False),
                    "fallback_strategy": context.get("fallback_strategy", "unknown")
                }
            
            # Emit failure detected event
            await self.observability_hooks.failure_detected(
                failure_type=error_type,
                failure_description=error_message,
                affected_component=f"error_handling.{component}",
                severity=severity,
                error_details=error_details,
                agent_id=agent_id,
                session_id=session_id,
                workflow_id=workflow_id,
                detection_method="error_handling_middleware",
                impact_assessment=self._assess_error_impact(error_type, context)
            )
            
            self.error_events_emitted += 1
            
            if self.enable_detailed_logging:
                logger.error(
                    "📡 Error handling failure event emitted",
                    error_type=error_type,
                    component=component,
                    severity=severity,
                    agent_id=str(agent_id) if agent_id else None,
                    session_id=str(session_id) if session_id else None
                )
                
        except Exception as emission_error:
            logger.warning(
                "⚠️ Failed to emit error handling failure event",
                original_error=error_message,
                emission_error=str(emission_error)
            )
    
    async def emit_circuit_breaker_state_change(
        self,
        circuit_breaker_name: str,
        old_state: CircuitBreakerState,
        new_state: CircuitBreakerState,
        reason: str,
        metrics: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None
    ) -> None:
        """
        Emit circuit breaker state change event.
        
        Args:
            circuit_breaker_name: Name of the circuit breaker
            old_state: Previous state
            new_state: New state
            reason: Reason for state change
            metrics: Circuit breaker metrics
            agent_id: Associated agent ID
            session_id: Associated session ID
        """
        if not self.observability_hooks:
            return
        
        try:
            # Determine if this is a recovery or failure
            is_recovery = (old_state != CircuitBreakerState.CLOSED and 
                          new_state == CircuitBreakerState.CLOSED)
            
            if is_recovery:
                # Emit recovery event
                await self.observability_hooks.recovery_initiated(
                    recovery_strategy="circuit_breaker_recovery",
                    trigger_failure=f"circuit_breaker_open:{circuit_breaker_name}",
                    recovery_steps=[
                        "circuit_breaker_timeout_elapsed",
                        "half_open_success_threshold_met",
                        "circuit_breaker_closed"
                    ],
                    agent_id=agent_id,
                    session_id=session_id,
                    estimated_recovery_time_ms=metrics.get("recovery_time_ms"),
                    backup_systems_activated=["graceful_degradation"],
                    rollback_checkpoint=f"circuit_breaker_{circuit_breaker_name}_closed"
                )
                
                self.recovery_events_emitted += 1
                
                if self.enable_detailed_logging:
                    logger.info(
                        "🔄 Circuit breaker recovery event emitted",
                        circuit_breaker=circuit_breaker_name,
                        old_state=old_state.value,
                        new_state=new_state.value,
                        reason=reason
                    )
            else:
                # Emit failure event
                severity = "high" if new_state == CircuitBreakerState.OPEN else "medium"
                
                await self.observability_hooks.failure_detected(
                    failure_type="circuit_breaker_state_change",
                    failure_description=f"Circuit breaker {circuit_breaker_name} changed from {old_state.value} to {new_state.value}",
                    affected_component=f"circuit_breaker.{circuit_breaker_name}",
                    severity=severity,
                    error_details={
                        "circuit_breaker_name": circuit_breaker_name,
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "reason": reason,
                        "metrics": metrics,
                        "state_change_timestamp": datetime.utcnow().isoformat()
                    },
                    agent_id=agent_id,
                    session_id=session_id,
                    detection_method="circuit_breaker_monitoring",
                    impact_assessment={
                        "service_availability": new_state != CircuitBreakerState.CLOSED,
                        "fallback_activated": new_state == CircuitBreakerState.OPEN,
                        "estimated_impact_duration_seconds": metrics.get("timeout_seconds", 60)
                    }
                )
                
                if self.enable_detailed_logging:
                    logger.warning(
                        "⚡ Circuit breaker failure event emitted",
                        circuit_breaker=circuit_breaker_name,
                        old_state=old_state.value,
                        new_state=new_state.value,
                        reason=reason,
                        severity=severity
                    )
                
        except Exception as emission_error:
            logger.warning(
                "⚠️ Failed to emit circuit breaker state change event",
                circuit_breaker=circuit_breaker_name,
                emission_error=str(emission_error)
            )
    
    async def emit_retry_attempt_event(
        self,
        operation_name: str,
        attempt_number: int,
        retry_result: RetryResult,
        error_context: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None
    ) -> None:
        """
        Emit retry attempt event for monitoring.
        
        Args:
            operation_name: Name of operation being retried
            attempt_number: Current attempt number
            retry_result: Result of retry calculation
            error_context: Context about the error triggering retry
            agent_id: Associated agent ID
            session_id: Associated session ID
        """
        if not self.observability_hooks:
            return
        
        try:
            # Only emit for significant retry attempts (attempt 2 and higher)
            if attempt_number < 2:
                return
            
            # Create retry event details
            retry_details = {
                "operation_name": operation_name,
                "attempt_number": attempt_number,
                "should_retry": retry_result.should_retry,
                "delay_ms": retry_result.delay_ms,
                "total_elapsed_ms": retry_result.total_elapsed_ms,
                "reason": retry_result.reason,
                "error_context": error_context,
                "retry_metadata": {
                    "next_delay_ms": retry_result.next_delay_ms,
                    "strategy": error_context.get("retry_strategy", "unknown")
                }
            }
            
            # Determine severity based on attempt number
            if attempt_number >= 3:
                severity = "high"
            elif attempt_number >= 2:
                severity = "medium"
            else:
                severity = "low"
            
            # Emit as a failure event with retry context
            await self.observability_hooks.failure_detected(
                failure_type="retry_attempt",
                failure_description=f"Retry attempt {attempt_number} for operation {operation_name}",
                affected_component=f"retry_policy.{operation_name}",
                severity=severity,
                error_details=retry_details,
                agent_id=agent_id,
                session_id=session_id,
                detection_method="retry_policy_monitoring",
                impact_assessment={
                    "operation_delayed": True,
                    "delay_ms": retry_result.delay_ms,
                    "retry_exhaustion_risk": not retry_result.should_retry
                }
            )
            
            if self.enable_detailed_logging:
                logger.warning(
                    "🔄 Retry attempt event emitted",
                    operation=operation_name,
                    attempt=attempt_number,
                    delay_ms=retry_result.delay_ms,
                    should_retry=retry_result.should_retry
                )
                
        except Exception as emission_error:
            logger.warning(
                "⚠️ Failed to emit retry attempt event",
                operation=operation_name,
                emission_error=str(emission_error)
            )
    
    async def emit_graceful_degradation_event(
        self,
        service_path: str,
        degradation_level: DegradationLevel,
        fallback_strategy: str,
        success: bool,
        error_context: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None
    ) -> None:
        """
        Emit graceful degradation event.
        
        Args:
            service_path: Path of degraded service
            degradation_level: Level of degradation applied
            fallback_strategy: Strategy used for fallback
            success: Whether degradation was successful
            error_context: Context about the original error
            agent_id: Associated agent ID
            session_id: Associated session ID
        """
        if not self.observability_hooks:
            return
        
        try:
            degradation_details = {
                "service_path": service_path,
                "degradation_level": degradation_level.value,
                "fallback_strategy": fallback_strategy,
                "success": success,
                "error_context": error_context,
                "degradation_timestamp": datetime.utcnow().isoformat()
            }
            
            if success:
                # Emit as recovery event
                await self.observability_hooks.recovery_initiated(
                    recovery_strategy="graceful_degradation",
                    trigger_failure=f"service_failure:{service_path}",
                    recovery_steps=[
                        "degradation_level_determined",
                        f"fallback_strategy_applied:{fallback_strategy}",
                        "degraded_response_generated"
                    ],
                    agent_id=agent_id,
                    session_id=session_id,
                    backup_systems_activated=[f"fallback_strategy:{fallback_strategy}"],
                    rollback_checkpoint=f"service_degraded:{service_path}"
                )
                
                self.degradation_events_emitted += 1
                
                if self.enable_detailed_logging:
                    logger.info(
                        "🛡️ Graceful degradation success event emitted",
                        service_path=service_path,
                        degradation_level=degradation_level.value,
                        fallback_strategy=fallback_strategy
                    )
            else:
                # Emit as failure event
                await self.observability_hooks.failure_detected(
                    failure_type="graceful_degradation_failure",
                    failure_description=f"Failed to apply graceful degradation for {service_path}",
                    affected_component=f"graceful_degradation.{service_path}",
                    severity="high",
                    error_details=degradation_details,
                    agent_id=agent_id,
                    session_id=session_id,
                    detection_method="graceful_degradation_monitoring",
                    impact_assessment={
                        "service_unavailable": True,
                        "fallback_failed": True,
                        "user_impact": "high"
                    }
                )
                
                if self.enable_detailed_logging:
                    logger.error(
                        "❌ Graceful degradation failure event emitted",
                        service_path=service_path,
                        degradation_level=degradation_level.value,
                        fallback_strategy=fallback_strategy
                    )
                
        except Exception as emission_error:
            logger.warning(
                "⚠️ Failed to emit graceful degradation event",
                service_path=service_path,
                emission_error=str(emission_error)
            )
    
    def _determine_error_severity(self, error_type: str, context: Dict[str, Any]) -> str:
        """Determine error severity based on type and context."""
        
        # High severity errors
        high_severity_types = {
            "ConnectionError", "TimeoutError", "DatabaseError", 
            "CircuitBreakerError", "ServiceUnavailable"
        }
        
        if error_type in high_severity_types:
            return "high"
        
        # Check HTTP status codes
        if "status_code" in context:
            status_code = context["status_code"]
            if status_code >= 500:
                return "high"
            elif status_code == 429:  # Rate limiting
                return "medium"
            elif status_code >= 400:
                return "low"
        
        # Check retry count for escalation
        if "retry_count" in context:
            retry_count = context["retry_count"]
            if retry_count >= 3:
                return "high"
            elif retry_count >= 2:
                return "medium"
        
        # Check processing time impact
        if "processing_time_ms" in context:
            processing_time = context["processing_time_ms"]
            if processing_time > 10000:  # >10 seconds
                return "high"
            elif processing_time > 5000:  # >5 seconds
                return "medium"
        
        return "medium"  # Default
    
    def _assess_error_impact(self, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of an error for observability."""
        
        impact = {
            "availability_affected": False,
            "performance_degraded": False,
            "user_experience_impacted": False,
            "data_consistency_risk": False
        }
        
        # Availability impact
        availability_affecting_errors = {
            "ConnectionError", "TimeoutError", "ServiceUnavailable", "CircuitBreakerError"
        }
        if error_type in availability_affecting_errors:
            impact["availability_affected"] = True
        
        # Performance impact
        if "processing_time_ms" in context:
            processing_time = context["processing_time_ms"]
            if processing_time > 1000:  # >1 second
                impact["performance_degraded"] = True
        
        # User experience impact
        if context.get("retry_count", 0) > 0:
            impact["user_experience_impacted"] = True
        
        if "status_code" in context and context["status_code"] >= 400:
            impact["user_experience_impacted"] = True
        
        # Data consistency risk
        data_affecting_errors = {"DatabaseError", "TransactionError", "ValidationError"}
        if error_type in data_affecting_errors:
            impact["data_consistency_risk"] = True
        
        return impact
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics."""
        return {
            "error_events_emitted": self.error_events_emitted,
            "recovery_events_emitted": self.recovery_events_emitted,
            "degradation_events_emitted": self.degradation_events_emitted,
            "total_events_emitted": (
                self.error_events_emitted + 
                self.recovery_events_emitted + 
                self.degradation_events_emitted
            ),
            "observability_hooks_available": bool(self.observability_hooks),
            "detailed_logging_enabled": self.enable_detailed_logging
        }


# Global integration instance
_error_handling_integration: Optional[ErrorHandlingObservabilityIntegration] = None


def get_error_handling_integration(
    enable_detailed_logging: bool = True
) -> ErrorHandlingObservabilityIntegration:
    """Get or create global error handling observability integration."""
    global _error_handling_integration
    
    if _error_handling_integration is None:
        _error_handling_integration = ErrorHandlingObservabilityIntegration(
            enable_detailed_logging=enable_detailed_logging
        )
    
    return _error_handling_integration


def initialize_error_handling_integration(
    enable_detailed_logging: bool = True
) -> ErrorHandlingObservabilityIntegration:
    """Initialize and set global error handling observability integration."""
    global _error_handling_integration
    
    _error_handling_integration = ErrorHandlingObservabilityIntegration(
        enable_detailed_logging=enable_detailed_logging
    )
    
    logger.info("✅ Global error handling observability integration initialized")
    return _error_handling_integration