"""
VS 7.1 Observability Hooks for Sleep/Wake API with Checkpointing

Integrates with existing Phase 4 observability system to provide:
- Performance metrics collection for checkpointing operations
- Recovery operation monitoring and alerting
- State management performance tracking
- Circuit breaker status monitoring
- Real-time dashboards and notifications
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID
from contextlib import asynccontextmanager

from ..core.checkpoint_manager import get_checkpoint_manager
from ..core.recovery_manager import get_recovery_manager
from ..core.enhanced_state_manager import get_enhanced_state_manager
from ..observability.hooks import ObservabilityHook, HookEvent
from ..observability.prometheus_exporter import PrometheusExporter
from ..observability.alerting import AlertManager
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class VS71ObservabilityHooks:
    """
    VS 7.1 specific observability hooks for enhanced monitoring.
    
    Integrates with existing Phase 4 observability to provide:
    - Checkpoint creation/restoration metrics
    - Recovery operation performance tracking
    - State consistency monitoring
    - API response time tracking
    - Circuit breaker status monitoring
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.prometheus_exporter = PrometheusExporter()
        self.alert_manager = AlertManager()
        
        # Performance thresholds for alerting
        self.checkpoint_creation_threshold_ms = 5000  # 5s
        self.recovery_time_threshold_ms = 10000  # 10s
        self.api_response_threshold_ms = 2000  # 2s
        self.state_consistency_failure_threshold = 0.05  # 5%
        
        # Metrics collectors
        self._checkpoint_metrics = {}
        self._recovery_metrics = {}
        self._api_metrics = {}
        self._state_metrics = {}
        
        # Hook registrations
        self._registered_hooks = set()
    
    async def initialize(self) -> None:
        """Initialize VS 7.1 observability hooks."""
        try:
            logger.info("Initializing VS 7.1 Observability Hooks")
            
            # Register all hooks
            await self._register_checkpoint_hooks()
            await self._register_recovery_hooks()
            await self._register_state_management_hooks()
            await self._register_api_hooks()
            await self._register_circuit_breaker_hooks()
            
            # Start background monitoring tasks
            asyncio.create_task(self._background_metrics_collection())
            asyncio.create_task(self._background_alerting())
            
            logger.info("VS 7.1 Observability Hooks initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VS 7.1 observability hooks: {e}")
            raise
    
    async def _register_checkpoint_hooks(self) -> None:
        """Register checkpoint operation monitoring hooks."""
        
        @ObservabilityHook("checkpoint.creation.started")
        async def on_checkpoint_creation_started(event: HookEvent):
            """Monitor checkpoint creation start."""
            checkpoint_id = event.data.get("checkpoint_id")
            agent_id = event.data.get("agent_id")
            
            self._checkpoint_metrics[checkpoint_id] = {
                "start_time": time.time(),
                "agent_id": agent_id,
                "checkpoint_type": event.data.get("checkpoint_type"),
                "status": "in_progress"
            }
            
            # Prometheus metrics
            self.prometheus_exporter.increment_counter(
                "checkpoint_creation_started_total",
                labels={"agent_id": str(agent_id), "type": event.data.get("checkpoint_type", "unknown")}
            )
        
        @ObservabilityHook("checkpoint.creation.completed")
        async def on_checkpoint_creation_completed(event: HookEvent):
            """Monitor checkpoint creation completion."""
            checkpoint_id = event.data.get("checkpoint_id")
            success = event.data.get("success", False)
            
            if checkpoint_id in self._checkpoint_metrics:
                start_time = self._checkpoint_metrics[checkpoint_id]["start_time"]
                duration_ms = (time.time() - start_time) * 1000
                
                self._checkpoint_metrics[checkpoint_id].update({
                    "end_time": time.time(),
                    "duration_ms": duration_ms,
                    "success": success,
                    "status": "completed" if success else "failed"
                })
                
                # Prometheus metrics
                self.prometheus_exporter.observe_histogram(
                    "checkpoint_creation_duration_ms",
                    duration_ms,
                    labels={"success": str(success)}
                )
                
                # Alert if threshold exceeded
                if duration_ms > self.checkpoint_creation_threshold_ms:
                    await self.alert_manager.send_alert(
                        "checkpoint_creation_slow",
                        f"Checkpoint creation took {duration_ms:.0f}ms (threshold: {self.checkpoint_creation_threshold_ms}ms)",
                        severity="warning",
                        metadata={
                            "checkpoint_id": checkpoint_id,
                            "duration_ms": duration_ms,
                            "agent_id": str(self._checkpoint_metrics[checkpoint_id].get("agent_id"))
                        }
                    )
                
                # Alert if failed
                if not success:
                    await self.alert_manager.send_alert(
                        "checkpoint_creation_failed",
                        f"Checkpoint creation failed for checkpoint {checkpoint_id}",
                        severity="error",
                        metadata={
                            "checkpoint_id": checkpoint_id,
                            "agent_id": str(self._checkpoint_metrics[checkpoint_id].get("agent_id"))
                        }
                    )
        
        @ObservabilityHook("checkpoint.validation.failed")
        async def on_checkpoint_validation_failed(event: HookEvent):
            """Monitor checkpoint validation failures."""
            checkpoint_id = event.data.get("checkpoint_id")
            errors = event.data.get("errors", [])
            
            # Prometheus metrics
            self.prometheus_exporter.increment_counter(
                "checkpoint_validation_failures_total",
                labels={"checkpoint_id": str(checkpoint_id)}
            )
            
            # Alert on validation failure
            await self.alert_manager.send_alert(
                "checkpoint_validation_failed",
                f"Checkpoint {checkpoint_id} failed validation: {', '.join(errors)}",
                severity="error",
                metadata={
                    "checkpoint_id": checkpoint_id,
                    "validation_errors": errors
                }
            )
        
        self._registered_hooks.update([
            "checkpoint.creation.started",
            "checkpoint.creation.completed", 
            "checkpoint.validation.failed"
        ])
    
    async def _register_recovery_hooks(self) -> None:
        """Register recovery operation monitoring hooks."""
        
        @ObservabilityHook("recovery.operation.started")
        async def on_recovery_started(event: HookEvent):
            """Monitor recovery operation start."""
            recovery_id = event.data.get("recovery_id")
            agent_id = event.data.get("agent_id")
            
            self._recovery_metrics[recovery_id] = {
                "start_time": time.time(),
                "agent_id": agent_id,
                "recovery_type": event.data.get("recovery_type"),
                "status": "in_progress"
            }
            
            # Prometheus metrics
            self.prometheus_exporter.increment_counter(
                "recovery_operations_started_total",
                labels={"agent_id": str(agent_id), "type": event.data.get("recovery_type", "unknown")}
            )
        
        @ObservabilityHook("recovery.operation.completed")
        async def on_recovery_completed(event: HookEvent):
            """Monitor recovery operation completion."""
            recovery_id = event.data.get("recovery_id")
            success = event.data.get("success", False)
            
            if recovery_id in self._recovery_metrics:
                start_time = self._recovery_metrics[recovery_id]["start_time"]
                duration_ms = (time.time() - start_time) * 1000
                
                self._recovery_metrics[recovery_id].update({
                    "end_time": time.time(),
                    "duration_ms": duration_ms,
                    "success": success,
                    "status": "completed" if success else "failed"
                })
                
                # Prometheus metrics
                self.prometheus_exporter.observe_histogram(
                    "recovery_operation_duration_ms",
                    duration_ms,
                    labels={"success": str(success)}
                )
                
                # Alert if threshold exceeded
                if duration_ms > self.recovery_time_threshold_ms:
                    await self.alert_manager.send_alert(
                        "recovery_operation_slow",
                        f"Recovery operation took {duration_ms:.0f}ms (threshold: {self.recovery_time_threshold_ms}ms)",
                        severity="warning",
                        metadata={
                            "recovery_id": recovery_id,
                            "duration_ms": duration_ms,
                            "agent_id": str(self._recovery_metrics[recovery_id].get("agent_id"))
                        }
                    )
                
                # Alert if failed
                if not success:
                    await self.alert_manager.send_alert(
                        "recovery_operation_failed",
                        f"Recovery operation failed for recovery {recovery_id}",
                        severity="critical",
                        metadata={
                            "recovery_id": recovery_id,
                            "agent_id": str(self._recovery_metrics[recovery_id].get("agent_id"))
                        }
                    )
        
        @ObservabilityHook("recovery.health_check.failed")
        async def on_recovery_health_check_failed(event: HookEvent):
            """Monitor recovery health check failures."""
            agent_id = event.data.get("agent_id")
            check_type = event.data.get("check_type")
            
            # Prometheus metrics
            self.prometheus_exporter.increment_counter(
                "recovery_health_check_failures_total",
                labels={"agent_id": str(agent_id), "check_type": check_type}
            )
            
            # Alert on health check failure
            await self.alert_manager.send_alert(
                "recovery_health_check_failed",
                f"Recovery health check failed for agent {agent_id}: {check_type}",
                severity="warning",
                metadata={
                    "agent_id": str(agent_id),
                    "check_type": check_type
                }
            )
        
        self._registered_hooks.update([
            "recovery.operation.started",
            "recovery.operation.completed",
            "recovery.health_check.failed"
        ])
    
    async def _register_state_management_hooks(self) -> None:
        """Register state management monitoring hooks."""
        
        @ObservabilityHook("state.consistency.check.failed")
        async def on_state_consistency_failed(event: HookEvent):
            """Monitor state consistency failures."""
            agent_id = event.data.get("agent_id")
            inconsistencies = event.data.get("inconsistencies", [])
            
            # Prometheus metrics
            self.prometheus_exporter.increment_counter(
                "state_consistency_failures_total",
                labels={"agent_id": str(agent_id)}
            )
            
            # Alert on consistency failure
            await self.alert_manager.send_alert(
                "state_consistency_failure",
                f"State consistency check failed for agent {agent_id}: {len(inconsistencies)} inconsistencies",
                severity="error",
                metadata={
                    "agent_id": str(agent_id),
                    "inconsistencies": inconsistencies
                }
            )
        
        @ObservabilityHook("state.cache.performance")
        async def on_state_cache_performance(event: HookEvent):
            """Monitor state cache performance."""
            cache_hit_rate = event.data.get("cache_hit_rate", 0)
            redis_read_ratio = event.data.get("redis_read_ratio", 0)
            
            # Prometheus metrics
            self.prometheus_exporter.set_gauge(
                "state_cache_hit_rate",
                cache_hit_rate
            )
            
            self.prometheus_exporter.set_gauge(
                "state_redis_read_ratio",
                redis_read_ratio
            )
            
            # Alert on low cache hit rate
            if cache_hit_rate < 0.8:  # Less than 80%
                await self.alert_manager.send_alert(
                    "state_cache_hit_rate_low",
                    f"State cache hit rate is low: {cache_hit_rate:.2%}",
                    severity="warning",
                    metadata={"cache_hit_rate": cache_hit_rate}
                )
        
        self._registered_hooks.update([
            "state.consistency.check.failed",
            "state.cache.performance"
        ])
    
    async def _register_api_hooks(self) -> None:
        """Register API performance monitoring hooks."""
        
        @ObservabilityHook("api.request.started")
        async def on_api_request_started(event: HookEvent):
            """Monitor API request start."""
            request_id = event.data.get("request_id")
            endpoint = event.data.get("endpoint")
            
            self._api_metrics[request_id] = {
                "start_time": time.time(),
                "endpoint": endpoint,
                "status": "in_progress"
            }
        
        @ObservabilityHook("api.request.completed")
        async def on_api_request_completed(event: HookEvent):
            """Monitor API request completion."""
            request_id = event.data.get("request_id")
            status_code = event.data.get("status_code", 200)
            endpoint = event.data.get("endpoint")
            
            if request_id in self._api_metrics:
                start_time = self._api_metrics[request_id]["start_time"]
                duration_ms = (time.time() - start_time) * 1000
                
                self._api_metrics[request_id].update({
                    "end_time": time.time(),
                    "duration_ms": duration_ms,
                    "status_code": status_code,
                    "status": "completed"
                })
                
                # Prometheus metrics
                self.prometheus_exporter.observe_histogram(
                    "api_request_duration_ms",
                    duration_ms,
                    labels={"endpoint": endpoint, "status_code": str(status_code)}
                )
                
                # Alert if response time threshold exceeded
                if duration_ms > self.api_response_threshold_ms:
                    await self.alert_manager.send_alert(
                        "api_response_slow",
                        f"API endpoint {endpoint} responded in {duration_ms:.0f}ms (threshold: {self.api_response_threshold_ms}ms)",
                        severity="warning",
                        metadata={
                            "endpoint": endpoint,
                            "duration_ms": duration_ms,
                            "request_id": request_id
                        }
                    )
                
                # Alert on error status codes
                if status_code >= 500:
                    await self.alert_manager.send_alert(
                        "api_error_response",
                        f"API endpoint {endpoint} returned error status {status_code}",
                        severity="error",
                        metadata={
                            "endpoint": endpoint,
                            "status_code": status_code,
                            "request_id": request_id
                        }
                    )
        
        self._registered_hooks.update([
            "api.request.started",
            "api.request.completed"
        ])
    
    async def _register_circuit_breaker_hooks(self) -> None:
        """Register circuit breaker monitoring hooks."""
        
        @ObservabilityHook("circuit_breaker.state_changed")
        async def on_circuit_breaker_state_changed(event: HookEvent):
            """Monitor circuit breaker state changes."""
            circuit_name = event.data.get("circuit_name")
            old_state = event.data.get("old_state")
            new_state = event.data.get("new_state")
            
            # Prometheus metrics
            self.prometheus_exporter.set_gauge(
                "circuit_breaker_state",
                1 if new_state == "open" else 0,
                labels={"circuit_name": circuit_name, "state": new_state}
            )
            
            # Alert on circuit breaker opening
            if new_state == "open":
                await self.alert_manager.send_alert(
                    "circuit_breaker_opened",
                    f"Circuit breaker {circuit_name} opened (was {old_state})",
                    severity="critical",
                    metadata={
                        "circuit_name": circuit_name,
                        "old_state": old_state,
                        "new_state": new_state
                    }
                )
            elif old_state == "open" and new_state == "closed":
                await self.alert_manager.send_alert(
                    "circuit_breaker_recovered",
                    f"Circuit breaker {circuit_name} recovered (closed)",
                    severity="info",
                    metadata={
                        "circuit_name": circuit_name,
                        "old_state": old_state,
                        "new_state": new_state
                    }
                )
        
        self._registered_hooks.add("circuit_breaker.state_changed")
    
    async def _background_metrics_collection(self) -> None:
        """Background task for collecting performance metrics."""
        while True:
            try:
                # Collect checkpoint manager metrics
                checkpoint_manager = get_checkpoint_manager()
                checkpoint_metrics = await checkpoint_manager.get_checkpoint_performance_metrics()
                
                if checkpoint_metrics:
                    self.prometheus_exporter.set_gauge(
                        "checkpoint_success_rate",
                        checkpoint_metrics.get("success_rate", 0)
                    )
                    
                    self.prometheus_exporter.set_gauge(
                        "checkpoint_average_creation_time_ms",
                        checkpoint_metrics.get("average_creation_time_ms", 0)
                    )
                
                # Collect recovery manager metrics
                recovery_manager = get_recovery_manager()
                recovery_metrics = await recovery_manager.get_recovery_performance_metrics()
                
                if recovery_metrics:
                    self.prometheus_exporter.set_gauge(
                        "recovery_success_rate",
                        recovery_metrics.get("success_rate", 0)
                    )
                    
                    self.prometheus_exporter.set_gauge(
                        "recovery_average_time_ms",
                        recovery_metrics.get("average_recovery_time_ms", 0)
                    )
                
                # Collect state manager metrics
                state_manager = await get_enhanced_state_manager()
                state_metrics = await state_manager.get_performance_metrics()
                
                if state_metrics:
                    self.prometheus_exporter.set_gauge(
                        "state_cache_hit_rate",
                        state_metrics.get("cache_hit_rate", 0)
                    )
                    
                    self.prometheus_exporter.set_gauge(
                        "state_consistency_failure_rate",
                        state_metrics.get("consistency_failure_rate", 0)
                    )
                
                # Sleep for 30 seconds before next collection
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in background metrics collection: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _background_alerting(self) -> None:
        """Background task for monitoring and alerting."""
        while True:
            try:
                # Check state consistency failure rate
                state_manager = await get_enhanced_state_manager()
                state_metrics = await state_manager.get_performance_metrics()
                
                consistency_failure_rate = state_metrics.get("consistency_failure_rate", 0)
                if consistency_failure_rate > self.state_consistency_failure_threshold:
                    await self.alert_manager.send_alert(
                        "state_consistency_failure_rate_high",
                        f"State consistency failure rate is high: {consistency_failure_rate:.2%}",
                        severity="error",
                        metadata={"failure_rate": consistency_failure_rate}
                    )
                
                # Check checkpoint performance trends
                checkpoint_manager = get_checkpoint_manager()
                checkpoint_metrics = await checkpoint_manager.get_checkpoint_performance_metrics()
                
                avg_creation_time = checkpoint_metrics.get("average_creation_time_ms", 0)
                if avg_creation_time > self.checkpoint_creation_threshold_ms:
                    await self.alert_manager.send_alert(
                        "checkpoint_creation_time_degraded",
                        f"Average checkpoint creation time degraded: {avg_creation_time:.0f}ms",
                        severity="warning",
                        metadata={"average_time_ms": avg_creation_time}
                    )
                
                # Check recovery performance trends
                recovery_manager = get_recovery_manager()
                recovery_metrics = await recovery_manager.get_recovery_performance_metrics()
                
                avg_recovery_time = recovery_metrics.get("average_recovery_time_ms", 0)
                if avg_recovery_time > self.recovery_time_threshold_ms:
                    await self.alert_manager.send_alert(
                        "recovery_time_degraded",
                        f"Average recovery time degraded: {avg_recovery_time:.0f}ms",
                        severity="warning",
                        metadata={"average_time_ms": avg_recovery_time}
                    )
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in background alerting: {e}")
                await asyncio.sleep(300)  # Wait same time on error
    
    async def get_vs71_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for VS 7.1 monitoring."""
        try:
            # Collect all performance metrics
            checkpoint_manager = get_checkpoint_manager()
            recovery_manager = get_recovery_manager()
            state_manager = await get_enhanced_state_manager()
            
            checkpoint_metrics = await checkpoint_manager.get_checkpoint_performance_metrics()
            recovery_metrics = await recovery_manager.get_recovery_performance_metrics()
            state_metrics = await state_manager.get_performance_metrics()
            
            # Recent operation summaries
            recent_checkpoints = len([
                m for m in self._checkpoint_metrics.values()
                if m.get("end_time", 0) > time.time() - 3600  # Last hour
            ])
            
            recent_recoveries = len([
                m for m in self._recovery_metrics.values()
                if m.get("end_time", 0) > time.time() - 3600  # Last hour
            ])
            
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "recent_checkpoints_1h": recent_checkpoints,
                    "recent_recoveries_1h": recent_recoveries,
                    "active_operations": len([
                        m for m in {**self._checkpoint_metrics, **self._recovery_metrics}.values()
                        if m.get("status") == "in_progress"
                    ])
                },
                "performance": {
                    "checkpoint": {
                        "average_creation_time_ms": checkpoint_metrics.get("average_creation_time_ms", 0),
                        "success_rate": checkpoint_metrics.get("success_rate", 0),
                        "meets_target": checkpoint_metrics.get("meets_performance_target", False)
                    },
                    "recovery": {
                        "average_time_ms": recovery_metrics.get("average_recovery_time_ms", 0),
                        "success_rate": recovery_metrics.get("success_rate", 0),
                        "cache_hit_rate": recovery_metrics.get("cache_hit_rate", 0)
                    },
                    "state": {
                        "cache_hit_rate": state_metrics.get("cache_hit_rate", 0),
                        "consistency_failure_rate": state_metrics.get("consistency_failure_rate", 0),
                        "redis_read_ratio": state_metrics.get("redis_read_ratio", 0)
                    }
                },
                "thresholds": {
                    "checkpoint_creation_ms": self.checkpoint_creation_threshold_ms,
                    "recovery_time_ms": self.recovery_time_threshold_ms,
                    "api_response_ms": self.api_response_threshold_ms,
                    "consistency_failure_rate": self.state_consistency_failure_threshold
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting VS 7.1 dashboard data: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


# Global VS 7.1 observability hooks instance
_vs71_hooks_instance: Optional[VS71ObservabilityHooks] = None


async def get_vs71_observability_hooks() -> VS71ObservabilityHooks:
    """Get the global VS 7.1 observability hooks instance."""
    global _vs71_hooks_instance
    if _vs71_hooks_instance is None:
        _vs71_hooks_instance = VS71ObservabilityHooks()
        await _vs71_hooks_instance.initialize()
    return _vs71_hooks_instance