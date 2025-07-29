"""
Comprehensive Observability Hooks System for LeanVibe Agent Hive 2.0 - VS 6.1

Production-ready hook system with <5ms overhead performance targets:
- Asynchronous non-blocking hook execution
- Critical event categories: workflow, agent, tool, memory, communication, recovery
- Dynamic sampling and verbosity control
- Redis Streams integration for event persistence
- MessagePack serialization for minimal network overhead
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from enum import Enum
import msgpack
import structlog

from ..schemas.observability import (
    BaseObservabilityEvent,
    EventCategory,
    WorkflowStartedEvent, WorkflowEndedEvent, NodeExecutingEvent, NodeCompletedEvent,
    AgentStateChangedEvent, AgentCapabilityUtilizedEvent,
    PreToolUseEvent, PostToolUseEvent,
    SemanticQueryEvent, SemanticUpdateEvent,
    MessagePublishedEvent, MessageReceivedEvent,
    FailureDetectedEvent, RecoveryInitiatedEvent,
    SystemHealthCheckEvent,
    PerformanceMetrics, EventMetadata
)
from .redis import get_redis

logger = structlog.get_logger()


class HookVerbosity(Enum):
    """Hook verbosity levels for dynamic sampling."""
    MINIMAL = "minimal"      # Only critical events (failures, stops)
    STANDARD = "standard"    # Standard operational events
    VERBOSE = "verbose"      # All events including debug
    DEBUG = "debug"          # Everything + performance metrics


class SamplingStrategy(Enum):
    """Sampling strategies for high-throughput scenarios."""
    NONE = "none"           # No sampling, capture all events
    RANDOM = "random"       # Random sampling based on probability
    RATE_LIMITED = "rate_limited"   # Rate-based sampling
    ADAPTIVE = "adaptive"   # Adaptive sampling based on system load


@dataclass
class HookConfiguration:
    """Configuration for hook system behavior."""
    enabled: bool = True
    verbosity: HookVerbosity = HookVerbosity.STANDARD
    sampling_strategy: SamplingStrategy = SamplingStrategy.NONE
    sampling_rate: float = 1.0  # 0.0 to 1.0
    max_events_per_second: int = 1000
    max_payload_size: int = 50000
    enable_performance_tracking: bool = True
    redis_stream_name: str = "observability_events"
    redis_max_len: int = 100000


class PerformanceTracker:
    """Tracks hook performance to ensure <5ms overhead targets."""
    
    def __init__(self):
        self.execution_times: List[float] = []
        self.max_samples = 1000
        self._lock = asyncio.Lock()
    
    async def record_execution_time(self, duration_ms: float) -> None:
        """Record hook execution time."""
        async with self._lock:
            self.execution_times.append(duration_ms)
            if len(self.execution_times) > self.max_samples:
                self.execution_times.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring."""
        if not self.execution_times:
            return {"avg_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0}
        
        sorted_times = sorted(self.execution_times)
        length = len(sorted_times)
        
        return {
            "avg_ms": sum(sorted_times) / length,
            "p95_ms": sorted_times[int(0.95 * length)] if length > 0 else 0,
            "p99_ms": sorted_times[int(0.99 * length)] if length > 0 else 0,
            "max_ms": max(sorted_times),
            "sample_count": length
        }
    
    def is_within_target(self, target_ms: float = 5.0) -> bool:
        """Check if performance is within target."""
        metrics = self.get_performance_metrics()
        return metrics["p95_ms"] <= target_ms


class EventSampler:
    """Handles event sampling based on configured strategy."""
    
    def __init__(self, config: HookConfiguration):
        self.config = config
        self._event_count = 0
        self._last_reset = time.time()
        self._rate_limiter_tokens = config.max_events_per_second
    
    def should_capture_event(self, event_category: EventCategory, event_type: str) -> bool:
        """Determine if event should be captured based on sampling strategy."""
        # Always capture critical events regardless of sampling
        if self._is_critical_event(event_category, event_type):
            return True
        
        # Check verbosity level
        if not self._passes_verbosity_filter(event_category, event_type):
            return False
        
        # Apply sampling strategy
        if self.config.sampling_strategy == SamplingStrategy.NONE:
            return True
        elif self.config.sampling_strategy == SamplingStrategy.RANDOM:
            import random
            return random.random() < self.config.sampling_rate
        elif self.config.sampling_strategy == SamplingStrategy.RATE_LIMITED:
            return self._rate_limit_check()
        elif self.config.sampling_strategy == SamplingStrategy.ADAPTIVE:
            return self._adaptive_sampling()
        
        return True
    
    def _is_critical_event(self, event_category: EventCategory, event_type: str) -> bool:
        """Check if event is critical and should always be captured."""
        critical_events = {
            "WorkflowEnded", "NodeFailed", "FailureDetected", "RecoveryInitiated",
            "AgentStopped", "SystemHealthCheck"
        }
        return event_type in critical_events
    
    def _passes_verbosity_filter(self, event_category: EventCategory, event_type: str) -> bool:
        """Check if event passes verbosity filter."""
        if self.config.verbosity == HookVerbosity.DEBUG:
            return True
        elif self.config.verbosity == HookVerbosity.VERBOSE:
            return event_category != EventCategory.SYSTEM or event_type != "DebugEvent"
        elif self.config.verbosity == HookVerbosity.STANDARD:
            excluded_events = {"SemanticQuery", "MessageReceived", "DebugEvent"}
            return event_type not in excluded_events
        elif self.config.verbosity == HookVerbosity.MINIMAL:
            minimal_events = {
                "WorkflowStarted", "WorkflowEnded", "NodeFailed",
                "PreToolUse", "PostToolUse", "FailureDetected"
            }
            return event_type in minimal_events
        
        return True
    
    def _rate_limit_check(self) -> bool:
        """Check rate limiting."""
        current_time = time.time()
        
        # Reset tokens every second
        if current_time - self._last_reset >= 1.0:
            self._rate_limiter_tokens = self.config.max_events_per_second
            self._last_reset = current_time
        
        if self._rate_limiter_tokens > 0:
            self._rate_limiter_tokens -= 1
            return True
        
        return False
    
    def _adaptive_sampling(self) -> bool:
        """Adaptive sampling based on system performance."""
        # Simplified adaptive logic - could be enhanced with system metrics
        metrics = self.performance_tracker.get_performance_metrics() if hasattr(self, 'performance_tracker') else {"avg_ms": 0}
        avg_time = metrics.get("avg_ms", 0)
        
        if avg_time > 3.0:  # If approaching 5ms limit
            return self.config.sampling_rate * 0.5 > 0.1  # Reduce sampling
        
        return True


class ObservabilityHooks:
    """
    Comprehensive observability hooks system for multi-agent coordination.
    
    Provides asynchronous, high-performance event capture with <5ms overhead targets.
    """
    
    def __init__(self, config: Optional[HookConfiguration] = None):
        """Initialize observability hooks system."""
        self.config = config or HookConfiguration()
        self.performance_tracker = PerformanceTracker()
        self.sampler = EventSampler(self.config)
        self.sampler.performance_tracker = self.performance_tracker
        
        # Redis client for event streaming
        self._redis_client = None
        self._initialization_lock = asyncio.Lock()
        
        # Event emission statistics
        self._events_emitted = 0
        self._events_dropped = 0
        self._last_emission_time = None
        
        logger.info(
            "🔌 ObservabilityHooks initialized",
            verbosity=self.config.verbosity.value,
            sampling_strategy=self.config.sampling_strategy.value,
            sampling_rate=self.config.sampling_rate
        )
    
    async def _ensure_redis_client(self) -> None:
        """Ensure Redis client is initialized."""
        if self._redis_client is None:
            async with self._initialization_lock:
                if self._redis_client is None:
                    self._redis_client = await get_redis()
    
    async def _emit_event_async(self, event: BaseObservabilityEvent) -> Optional[str]:
        """
        Emit event asynchronously with performance tracking.
        
        Returns:
            Stream ID if successful, None if dropped or failed
        """
        if not self.config.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # Check sampling
            if not self.sampler.should_capture_event(event.event_category, event.event_type):
                self._events_dropped += 1
                return None
            
            # Ensure Redis client
            await self._ensure_redis_client()
            
            # Serialize event with MessagePack for performance
            event_data = await self._serialize_event(event)
            
            # Emit to Redis stream
            stream_id = await self._redis_client.xadd(
                self.config.redis_stream_name,
                event_data,
                maxlen=self.config.redis_max_len
            )
            
            self._events_emitted += 1
            self._last_emission_time = datetime.utcnow()
            
            # Track performance
            execution_time_ms = (time.time() - start_time) * 1000
            await self.performance_tracker.record_execution_time(execution_time_ms)
            
            logger.debug(
                "📡 Event emitted",
                stream_id=stream_id,
                event_type=event.event_type,
                execution_time_ms=round(execution_time_ms, 2)
            )
            
            return stream_id
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            await self.performance_tracker.record_execution_time(execution_time_ms)
            
            logger.error(
                "❌ Failed to emit event",
                error=str(e),
                event_type=event.event_type,
                execution_time_ms=round(execution_time_ms, 2),
                exc_info=True
            )
            return None
    
    async def _serialize_event(self, event: BaseObservabilityEvent) -> Dict[str, bytes]:
        """Serialize event with MessagePack for optimal performance."""
        try:
            # Convert to dict and handle UUID/datetime serialization
            event_dict = event.model_dump(mode='json')
            
            # Use MessagePack for binary serialization
            serialized_event = msgpack.packb(event_dict, use_bin_type=True)
            
            # Ensure payload size limits
            if len(serialized_event) > self.config.max_payload_size:
                # Truncate payload if too large
                event_dict["payload"] = {"truncated": True, "original_size": len(serialized_event)}
                serialized_event = msgpack.packb(event_dict, use_bin_type=True)
            
            return {
                "event_data": serialized_event,
                "event_type": event.event_type.encode(),
                "event_category": event.event_category.value.encode(),
                "timestamp": event.timestamp.isoformat().encode(),
                "agent_id": str(event.agent_id).encode() if event.agent_id else b"",
                "session_id": str(event.session_id).encode() if event.session_id else b"",
                "workflow_id": str(event.workflow_id).encode() if event.workflow_id else b""
            }
            
        except Exception as e:
            logger.error("❌ Event serialization failed", error=str(e), exc_info=True)
            raise
    
    # ==========================================
    # WORKFLOW LIFECYCLE HOOKS
    # ==========================================
    
    async def workflow_started(
        self,
        workflow_id: uuid.UUID,
        workflow_name: str,
        workflow_definition: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        estimated_duration_ms: Optional[float] = None,
        priority: Optional[str] = None
    ) -> Optional[str]:
        """Hook for workflow started events."""
        event = WorkflowStartedEvent(
            workflow_id=workflow_id,
            agent_id=agent_id,
            session_id=session_id,
            workflow_name=workflow_name,
            workflow_definition=workflow_definition,
            initial_context=initial_context,
            estimated_duration_ms=estimated_duration_ms,
            priority=priority,
            initiating_agent=agent_id
        )
        
        return await self._emit_event_async(event)
    
    async def workflow_ended(
        self,
        workflow_id: uuid.UUID,
        status: str,
        completion_reason: str,
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        final_result: Optional[Dict[str, Any]] = None,
        total_tasks_executed: Optional[int] = None,
        failed_tasks: Optional[int] = None,
        actual_duration_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for workflow ended events."""
        event = WorkflowEndedEvent(
            workflow_id=workflow_id,
            agent_id=agent_id,
            session_id=session_id,
            status=status,
            completion_reason=completion_reason,
            final_result=final_result,
            total_tasks_executed=total_tasks_executed,
            failed_tasks=failed_tasks,
            actual_duration_ms=actual_duration_ms
        )
        
        return await self._emit_event_async(event)
    
    async def node_executing(
        self,
        workflow_id: uuid.UUID,
        node_id: str,
        node_type: str,
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        node_name: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        dependencies_satisfied: Optional[List[str]] = None,
        assigned_agent: Optional[uuid.UUID] = None,
        estimated_execution_time_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for task node executing events."""
        event = NodeExecutingEvent(
            workflow_id=workflow_id,
            agent_id=agent_id,
            session_id=session_id,
            node_id=node_id,
            node_type=node_type,
            node_name=node_name,
            input_data=input_data,
            dependencies_satisfied=dependencies_satisfied,
            assigned_agent=assigned_agent,
            estimated_execution_time_ms=estimated_execution_time_ms
        )
        
        return await self._emit_event_async(event)
    
    async def node_completed(
        self,
        workflow_id: uuid.UUID,
        node_id: str,
        success: bool,
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
        retry_count: Optional[int] = None,
        downstream_nodes: Optional[List[str]] = None,
        execution_time_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for task node completed events."""
        event = NodeCompletedEvent(
            workflow_id=workflow_id,
            agent_id=agent_id,
            session_id=session_id,
            node_id=node_id,
            success=success,
            output_data=output_data,
            error_details=error_details,
            retry_count=retry_count,
            downstream_nodes=downstream_nodes,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=execution_time_ms
            ) if execution_time_ms else None
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # AGENT STATE HOOKS
    # ==========================================
    
    async def agent_state_changed(
        self,
        agent_id: uuid.UUID,
        previous_state: str,
        new_state: str,
        state_transition_reason: str,
        session_id: Optional[uuid.UUID] = None,
        capabilities: Optional[List[str]] = None,
        resource_allocation: Optional[Dict[str, Any]] = None,
        persona_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Hook for agent state changed events."""
        event = AgentStateChangedEvent(
            agent_id=agent_id,
            session_id=session_id,
            previous_state=previous_state,
            new_state=new_state,
            state_transition_reason=state_transition_reason,
            capabilities=capabilities,
            resource_allocation=resource_allocation,
            persona_data=persona_data
        )
        
        return await self._emit_event_async(event)
    
    async def agent_capability_utilized(
        self,
        agent_id: uuid.UUID,
        capability_name: str,
        utilization_context: str,
        session_id: Optional[uuid.UUID] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
        capability_result: Optional[Dict[str, Any]] = None,
        efficiency_score: Optional[float] = None,
        execution_time_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for agent capability utilized events."""
        event = AgentCapabilityUtilizedEvent(
            agent_id=agent_id,
            session_id=session_id,
            capability_name=capability_name,
            utilization_context=utilization_context,
            input_parameters=input_parameters,
            capability_result=capability_result,
            efficiency_score=efficiency_score,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=execution_time_ms
            ) if execution_time_ms else None
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # TOOL EXECUTION HOOKS (MOST CRITICAL)
    # ==========================================
    
    async def pre_tool_use(
        self,
        agent_id: uuid.UUID,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[uuid.UUID] = None,
        tool_version: Optional[str] = None,
        expected_output_type: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        retry_policy: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Hook for pre-tool use events - most critical for debugging."""
        event = PreToolUseEvent(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            parameters=parameters,
            tool_version=tool_version,
            expected_output_type=expected_output_type,
            timeout_ms=timeout_ms,
            retry_policy=retry_policy
        )
        
        return await self._emit_event_async(event)
    
    async def post_tool_use(
        self,
        agent_id: uuid.UUID,
        tool_name: str,
        success: bool,
        session_id: Optional[uuid.UUID] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        retry_count: Optional[int] = None,
        execution_time_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None
    ) -> Optional[str]:
        """Hook for post-tool use events - critical for performance monitoring."""
        event = PostToolUseEvent(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            error_type=error_type,
            retry_count=retry_count,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent
            ) if any([execution_time_ms, memory_usage_mb, cpu_usage_percent]) else None
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # SEMANTIC MEMORY HOOKS
    # ==========================================
    
    async def semantic_query(
        self,
        query_text: str,
        query_embedding: List[float],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        similarity_threshold: Optional[float] = None,
        max_results: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        results_count: Optional[int] = None,
        search_strategy: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for semantic query events."""
        event = SemanticQueryEvent(
            agent_id=agent_id,
            session_id=session_id,
            query_text=query_text,
            query_embedding=query_embedding,
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            filter_criteria=filter_criteria,
            results_count=results_count,
            search_strategy=search_strategy,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=execution_time_ms
            ) if execution_time_ms else None
        )
        
        return await self._emit_event_async(event)
    
    async def semantic_update(
        self,
        operation_type: str,
        content: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        content_embedding: Optional[List[float]] = None,
        content_id: Optional[str] = None,
        content_type: Optional[str] = None,
        content_metadata: Optional[Dict[str, Any]] = None,
        affected_records: Optional[int] = None,
        execution_time_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for semantic update events."""
        event = SemanticUpdateEvent(
            agent_id=agent_id,
            session_id=session_id,
            operation_type=operation_type,
            content=content,
            content_embedding=content_embedding,
            content_id=content_id,
            content_type=content_type,
            content_metadata=content_metadata,
            affected_records=affected_records,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=execution_time_ms
            ) if execution_time_ms else None
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # COMMUNICATION HOOKS
    # ==========================================
    
    async def message_published(
        self,
        message_id: uuid.UUID,
        from_agent: str,
        to_agent: str,
        message_type: str,
        message_content: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        priority: Optional[str] = None,
        delivery_method: Optional[str] = None,
        expected_response: Optional[bool] = False
    ) -> Optional[str]:
        """Hook for message published events."""
        event = MessagePublishedEvent(
            agent_id=agent_id,
            session_id=session_id,
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            message_content=message_content,
            priority=priority,
            delivery_method=delivery_method,
            expected_response=expected_response
        )
        
        return await self._emit_event_async(event)
    
    async def message_received(
        self,
        message_id: uuid.UUID,
        from_agent: str,
        processing_status: str,
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        processing_reason: Optional[str] = None,
        response_generated: Optional[bool] = False,
        delivery_latency_ms: Optional[float] = None
    ) -> Optional[str]:
        """Hook for message received events."""
        event = MessageReceivedEvent(
            agent_id=agent_id,
            session_id=session_id,
            message_id=message_id,
            from_agent=from_agent,
            processing_status=processing_status,
            processing_reason=processing_reason,
            response_generated=response_generated,
            delivery_latency_ms=delivery_latency_ms
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # RECOVERY HOOKS
    # ==========================================
    
    async def failure_detected(
        self,
        failure_type: str,
        failure_description: str,
        affected_component: str,
        severity: str,
        error_details: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        workflow_id: Optional[uuid.UUID] = None,
        detection_method: Optional[str] = None,
        impact_assessment: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Hook for failure detected events."""
        event = FailureDetectedEvent(
            agent_id=agent_id,
            session_id=session_id,
            workflow_id=workflow_id,
            failure_type=failure_type,
            failure_description=failure_description,
            affected_component=affected_component,
            severity=severity,
            error_details=error_details,
            detection_method=detection_method,
            impact_assessment=impact_assessment
        )
        
        return await self._emit_event_async(event)
    
    async def recovery_initiated(
        self,
        recovery_strategy: str,
        trigger_failure: str,
        recovery_steps: List[str],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        workflow_id: Optional[uuid.UUID] = None,
        estimated_recovery_time_ms: Optional[float] = None,
        backup_systems_activated: Optional[List[str]] = None,
        rollback_checkpoint: Optional[str] = None
    ) -> Optional[str]:
        """Hook for recovery initiated events."""
        event = RecoveryInitiatedEvent(
            agent_id=agent_id,
            session_id=session_id,
            workflow_id=workflow_id,
            recovery_strategy=recovery_strategy,
            trigger_failure=trigger_failure,
            recovery_steps=recovery_steps,
            estimated_recovery_time_ms=estimated_recovery_time_ms,
            backup_systems_activated=backup_systems_activated,
            rollback_checkpoint=rollback_checkpoint
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # SYSTEM MONITORING HOOKS
    # ==========================================
    
    async def system_health_check(
        self,
        health_status: str,
        check_type: str,
        component_statuses: Dict[str, str],
        performance_indicators: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        alerts_triggered: Optional[List[str]] = None,
        recommended_actions: Optional[List[str]] = None
    ) -> Optional[str]:
        """Hook for system health check events."""
        event = SystemHealthCheckEvent(
            agent_id=agent_id,
            session_id=session_id,
            health_status=health_status,
            check_type=check_type,
            component_statuses=component_statuses,
            performance_indicators=performance_indicators,
            alerts_triggered=alerts_triggered,
            recommended_actions=recommended_actions
        )
        
        return await self._emit_event_async(event)
    
    # ==========================================
    # SYSTEM MANAGEMENT
    # ==========================================
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        hook_metrics = self.performance_tracker.get_performance_metrics()
        
        return {
            "hook_performance": hook_metrics,
            "events_emitted": self._events_emitted,
            "events_dropped": self._events_dropped,
            "drop_rate": self._events_dropped / max(1, self._events_emitted + self._events_dropped),
            "last_emission_time": self._last_emission_time.isoformat() if self._last_emission_time else None,
            "within_target": self.performance_tracker.is_within_target(),
            "configuration": {
                "verbosity": self.config.verbosity.value,
                "sampling_strategy": self.config.sampling_strategy.value,
                "sampling_rate": self.config.sampling_rate,
                "enabled": self.config.enabled
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the hooks system."""
        try:
            # Test Redis connectivity
            await self._ensure_redis_client()
            await self._redis_client.ping()
            redis_healthy = True
        except:
            redis_healthy = False
        
        performance_metrics = self.get_performance_metrics()
        within_target = performance_metrics["within_target"]
        
        status = "healthy" if redis_healthy and within_target else "degraded"
        if not redis_healthy:
            status = "unhealthy"
        
        return {
            "status": status,
            "redis_healthy": redis_healthy,
            "performance_within_target": within_target,
            "metrics": performance_metrics
        }
    
    def update_configuration(self, config: HookConfiguration) -> None:
        """Update hook configuration at runtime."""
        self.config = config
        self.sampler = EventSampler(config)
        
        logger.info(
            "🔧 Hook configuration updated",
            verbosity=config.verbosity.value,
            sampling_strategy=config.sampling_strategy.value,
            enabled=config.enabled
        )


# Global hooks instance
_observability_hooks: Optional[ObservabilityHooks] = None


def get_observability_hooks() -> Optional[ObservabilityHooks]:
    """Get the global observability hooks instance."""
    return _observability_hooks


def initialize_observability_hooks(config: Optional[HookConfiguration] = None) -> ObservabilityHooks:
    """Initialize and set the global observability hooks."""
    global _observability_hooks
    
    _observability_hooks = ObservabilityHooks(config)
    
    logger.info("✅ Global observability hooks initialized")
    return _observability_hooks


def get_hooks() -> ObservabilityHooks:
    """Convenience function to get hooks or create default instance."""
    global _observability_hooks
    
    if _observability_hooks is None:
        _observability_hooks = ObservabilityHooks()
    
    return _observability_hooks