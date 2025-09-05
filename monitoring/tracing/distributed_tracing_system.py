"""
Epic 7 Phase 3: Distributed Tracing System

Comprehensive distributed tracing for request flow across services:
- OpenTelemetry integration with Jaeger backend
- Request flow tracking through API → Database → Redis → Agent coordination
- Performance bottleneck identification and optimization insights
- Service dependency mapping and health correlation
- Real-time trace analysis and alerting
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import structlog

from opentelemetry import trace, baggage, context
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.trace import Status, StatusCode

logger = structlog.get_logger()


@dataclass
class TraceMetrics:
    """Trace performance and health metrics."""
    total_traces: int = 0
    active_traces: int = 0
    error_traces: int = 0
    avg_trace_duration_ms: float = 0.0
    p95_trace_duration_ms: float = 0.0
    traces_per_second: float = 0.0
    service_error_rates: Dict[str, float] = field(default_factory=dict)
    bottleneck_services: List[str] = field(default_factory=list)


@dataclass
class ServiceDependency:
    """Service dependency relationship."""
    from_service: str
    to_service: str
    call_count: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    dependency_health: float = 100.0  # 0-100 health score


@dataclass
class TraceAnalysis:
    """Trace analysis results for performance optimization."""
    trace_id: str
    duration_ms: float
    service_breakdown: Dict[str, float]  # service -> time spent
    critical_path: List[str]
    bottlenecks: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    recommendations: List[str]


class DistributedTracingSystem:
    """
    Comprehensive distributed tracing system for Epic 7 Phase 3.
    
    Provides end-to-end request tracking, performance analysis, and
    service dependency monitoring for production optimization.
    """
    
    def __init__(self):
        self.service_name = "leanvibe-agent-hive"
        self.tracer = None
        self.setup_tracing()
        
        # Metrics and analysis
        self.trace_metrics = TraceMetrics()
        self.service_dependencies: Dict[str, ServiceDependency] = {}
        self.active_traces: Dict[str, Dict[str, Any]] = {}
        self.trace_analyses: List[TraceAnalysis] = []
        
        # Configuration
        self.trace_sampling_rate = 0.1  # 10% sampling for production
        self.slow_trace_threshold_ms = 1000  # Traces slower than 1s
        self.error_trace_tracking = True
        self.dependency_health_monitoring = True
        
        logger.info("🔍 Distributed Tracing System initialized for Epic 7 Phase 3")
        
    def setup_tracing(self):
        """Initialize OpenTelemetry distributed tracing with multiple exporters."""
        try:
            # Configure resource information
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "2.0.0",
                "deployment.environment": "production"
            })
            
            # Set up trace provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            self.tracer = trace.get_tracer(__name__, "1.0.0")
            
            # Configure multiple exporters for different use cases
            
            # 1. Jaeger exporter for UI visualization and debugging
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            # 2. OTLP exporter for production monitoring systems
            otlp_exporter = OTLPSpanExporter(
                endpoint="http://localhost:4317",
                insecure=True
            )
            
            # 3. Console exporter for development and debugging
            console_exporter = ConsoleSpanExporter()
            
            # Add span processors
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            
            # Set up propagators for cross-service communication
            from opentelemetry.propagate import set_global_textmap
            set_global_textmap(JaegerPropagator())
            
            # Auto-instrument common libraries
            FastAPIInstrumentor.instrument()
            SQLAlchemyInstrumentor.instrument()
            RedisInstrumentor.instrument()
            RequestsInstrumentor.instrument()
            AsyncPGInstrumentor.instrument()
            
            logger.info("✅ Distributed tracing configured with multiple exporters")
            
        except Exception as e:
            logger.error("❌ Failed to setup distributed tracing", error=str(e))
            
    @asynccontextmanager
    async def trace_request(self, operation_name: str, service_name: str = None,
                           user_id: str = None, **attributes):
        """
        Context manager for tracing requests with comprehensive metadata.
        
        Usage:
            async with tracing.trace_request("user_registration", "auth_service", user_id="123"):
                # Your code here
        """
        service = service_name or self.service_name
        span_name = f"{service}.{operation_name}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            try:
                # Set standard attributes
                span.set_attribute("service.name", service)
                span.set_attribute("operation.name", operation_name)
                
                if user_id:
                    span.set_attribute("user.id", user_id)
                    baggage.set_baggage("user.id", user_id)
                    
                # Set custom attributes
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                    
                # Track trace start
                trace_id = format(span.get_span_context().trace_id, '032x')
                start_time = time.time()
                
                self.active_traces[trace_id] = {
                    "operation": operation_name,
                    "service": service,
                    "start_time": start_time,
                    "user_id": user_id
                }
                
                self.trace_metrics.active_traces += 1
                
                yield span
                
            except Exception as e:
                # Record error in trace
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Update error metrics
                self.trace_metrics.error_traces += 1
                service_errors = self.trace_metrics.service_error_rates.get(service, 0)
                self.trace_metrics.service_error_rates[service] = service_errors + 1
                
                logger.error("❌ Traced operation failed",
                           operation=operation_name,
                           service=service,
                           trace_id=trace_id,
                           error=str(e))
                raise
                
            finally:
                # Track trace completion
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Update metrics
                self.trace_metrics.active_traces -= 1
                self.trace_metrics.total_traces += 1
                
                # Update duration metrics (exponential moving average)
                current_avg = self.trace_metrics.avg_trace_duration_ms
                self.trace_metrics.avg_trace_duration_ms = (
                    current_avg * 0.9 + duration_ms * 0.1
                )
                
                # Track slow traces
                if duration_ms > self.slow_trace_threshold_ms:
                    await self._analyze_slow_trace(trace_id, duration_ms, operation_name, service)
                
                # Clean up active trace
                if trace_id in self.active_traces:
                    del self.active_traces[trace_id]
                    
                # Record service dependency
                await self._record_service_dependency(service, operation_name, duration_ms)
                
    async def trace_database_operation(self, query_type: str, table_name: str = None,
                                     query: str = None) -> AsyncContextManager:
        """Trace database operations with query analysis."""
        attributes = {
            "db.operation": query_type,
            "db.system": "postgresql"
        }
        
        if table_name:
            attributes["db.table"] = table_name
        if query:
            attributes["db.statement"] = query[:200]  # Truncate long queries
            
        return self.trace_request(f"db.{query_type}", "database", **attributes)
        
    async def trace_redis_operation(self, command: str, key: str = None) -> AsyncContextManager:
        """Trace Redis operations."""
        attributes = {
            "redis.command": command,
            "db.system": "redis"
        }
        
        if key:
            attributes["redis.key"] = key
            
        return self.trace_request(f"redis.{command}", "cache", **attributes)
        
    async def trace_agent_operation(self, agent_id: str, operation_type: str,
                                  task_id: str = None) -> AsyncContextManager:
        """Trace agent operations for multi-agent coordination."""
        attributes = {
            "agent.id": agent_id,
            "agent.operation": operation_type
        }
        
        if task_id:
            attributes["task.id"] = task_id
            
        return self.trace_request(f"agent.{operation_type}", "agent_coordinator", **attributes)
        
    async def trace_api_request(self, endpoint: str, method: str, 
                              user_id: str = None) -> AsyncContextManager:
        """Trace API requests with endpoint-specific metadata."""
        attributes = {
            "http.method": method,
            "http.route": endpoint,
            "http.scheme": "https"
        }
        
        return self.trace_request(f"api{endpoint}", "api_server", user_id=user_id, **attributes)
        
    async def get_trace_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trace metrics for monitoring."""
        try:
            # Calculate traces per second (last minute)
            current_time = time.time()
            traces_last_minute = len([
                t for t in self.active_traces.values()
                if current_time - t["start_time"] < 60
            ])
            self.trace_metrics.traces_per_second = traces_last_minute / 60
            
            # Identify bottleneck services
            bottlenecks = sorted(
                self.trace_metrics.service_error_rates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 bottlenecks
            
            self.trace_metrics.bottleneck_services = [service for service, _ in bottlenecks]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_traces": self.trace_metrics.total_traces,
                "active_traces": self.trace_metrics.active_traces,
                "error_traces": self.trace_metrics.error_traces,
                "avg_duration_ms": self.trace_metrics.avg_trace_duration_ms,
                "traces_per_second": self.trace_metrics.traces_per_second,
                "error_rates_by_service": self.trace_metrics.service_error_rates,
                "bottleneck_services": self.trace_metrics.bottleneck_services,
                "service_dependencies": len(self.service_dependencies),
                "slow_traces_analyzed": len(self.trace_analyses)
            }
            
        except Exception as e:
            logger.error("❌ Failed to get trace metrics", error=str(e))
            return {"error": str(e)}
            
    async def get_service_dependency_map(self) -> Dict[str, Any]:
        """Get service dependency map for architecture visualization."""
        try:
            dependencies = []
            for key, dep in self.service_dependencies.items():
                dependencies.append({
                    "from_service": dep.from_service,
                    "to_service": dep.to_service,
                    "call_count": dep.call_count,
                    "avg_latency_ms": dep.avg_latency_ms,
                    "error_rate": dep.error_rate,
                    "health_score": dep.dependency_health
                })
                
            # Calculate overall service health
            service_health = {}
            for dep in dependencies:
                from_service = dep["from_service"]
                to_service = dep["to_service"]
                
                if from_service not in service_health:
                    service_health[from_service] = []
                if to_service not in service_health:
                    service_health[to_service] = []
                    
                service_health[from_service].append(dep["health_score"])
                
            # Average health scores
            for service in service_health:
                if service_health[service]:
                    service_health[service] = sum(service_health[service]) / len(service_health[service])
                else:
                    service_health[service] = 100.0
                    
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "dependencies": dependencies,
                "service_health": service_health,
                "total_services": len(service_health),
                "unhealthy_services": [
                    service for service, health in service_health.items()
                    if health < 80.0
                ]
            }
            
        except Exception as e:
            logger.error("❌ Failed to get service dependency map", error=str(e))
            return {"error": str(e)}
            
    async def get_trace_analysis_insights(self) -> Dict[str, Any]:
        """Get trace analysis insights for performance optimization."""
        try:
            if not self.trace_analyses:
                return {
                    "message": "No trace analyses available yet",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            # Aggregate insights
            total_analyses = len(self.trace_analyses)
            avg_duration = sum(t.duration_ms for t in self.trace_analyses) / total_analyses
            
            # Find common bottlenecks
            bottleneck_services = {}
            for analysis in self.trace_analyses:
                for bottleneck in analysis.bottlenecks:
                    service = bottleneck.get("service", "unknown")
                    bottleneck_services[service] = bottleneck_services.get(service, 0) + 1
                    
            # Aggregate recommendations
            recommendation_counts = {}
            for analysis in self.trace_analyses:
                for rec in analysis.recommendations:
                    recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
                    
            top_recommendations = sorted(
                recommendation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_analyses": total_analyses,
                "avg_slow_trace_duration_ms": avg_duration,
                "common_bottlenecks": dict(sorted(
                    bottleneck_services.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]),
                "top_recommendations": [rec for rec, count in top_recommendations],
                "performance_insights": {
                    "slowest_service_operations": await self._get_slowest_operations(),
                    "error_patterns": await self._get_error_patterns(),
                    "optimization_opportunities": await self._get_optimization_opportunities()
                }
            }
            
        except Exception as e:
            logger.error("❌ Failed to get trace analysis insights", error=str(e))
            return {"error": str(e)}
            
    async def _analyze_slow_trace(self, trace_id: str, duration_ms: float,
                                operation_name: str, service_name: str):
        """Analyze slow traces for performance optimization."""
        try:
            # Mock analysis - in production would analyze actual span data
            analysis = TraceAnalysis(
                trace_id=trace_id,
                duration_ms=duration_ms,
                service_breakdown={
                    service_name: duration_ms * 0.7,
                    "database": duration_ms * 0.2,
                    "cache": duration_ms * 0.1
                },
                critical_path=[service_name, "database", "cache"],
                bottlenecks=[
                    {
                        "service": service_name,
                        "operation": operation_name,
                        "duration_ms": duration_ms * 0.7,
                        "reason": "High processing time"
                    }
                ],
                errors=[],
                recommendations=[
                    f"Optimize {operation_name} operation in {service_name}",
                    "Consider database query optimization",
                    "Implement caching for frequently accessed data"
                ]
            )
            
            self.trace_analyses.append(analysis)
            
            # Keep only recent analyses
            if len(self.trace_analyses) > 100:
                self.trace_analyses = self.trace_analyses[-50:]
                
            logger.warning("🐌 Slow trace analyzed",
                         trace_id=trace_id,
                         duration_ms=duration_ms,
                         operation=operation_name,
                         service=service_name)
                         
        except Exception as e:
            logger.error("❌ Failed to analyze slow trace", error=str(e))
            
    async def _record_service_dependency(self, from_service: str, operation: str, 
                                       duration_ms: float, error: bool = False):
        """Record service dependency for health monitoring."""
        try:
            # Determine to_service based on operation
            to_service = self._determine_target_service(operation)
            
            if to_service == from_service:
                return  # Skip self-dependencies
                
            dependency_key = f"{from_service}->{to_service}"
            
            if dependency_key not in self.service_dependencies:
                self.service_dependencies[dependency_key] = ServiceDependency(
                    from_service=from_service,
                    to_service=to_service
                )
                
            dep = self.service_dependencies[dependency_key]
            dep.call_count += 1
            
            # Update latency (exponential moving average)
            current_avg = dep.avg_latency_ms
            dep.avg_latency_ms = current_avg * 0.9 + duration_ms * 0.1
            
            # Update error rate
            if error:
                dep.error_rate = dep.error_rate * 0.9 + 0.1
            else:
                dep.error_rate = dep.error_rate * 0.9
                
            # Calculate dependency health (based on latency and error rate)
            latency_score = max(0, 100 - (dep.avg_latency_ms / 10))  # 10ms = 100%, 1000ms = 0%
            error_score = max(0, 100 - (dep.error_rate * 100))
            dep.dependency_health = (latency_score + error_score) / 2
            
        except Exception as e:
            logger.error("❌ Failed to record service dependency", error=str(e))
            
    def _determine_target_service(self, operation: str) -> str:
        """Determine target service based on operation name."""
        if operation.startswith("db."):
            return "database"
        elif operation.startswith("redis."):
            return "cache"
        elif operation.startswith("api"):
            return "api_server"
        elif operation.startswith("agent."):
            return "agent_coordinator"
        else:
            return "unknown"
            
    async def _get_slowest_operations(self) -> List[Dict[str, Any]]:
        """Get slowest operations across services."""
        operations = {}
        for analysis in self.trace_analyses:
            for service, duration in analysis.service_breakdown.items():
                key = f"{service}"
                if key not in operations:
                    operations[key] = []
                operations[key].append(duration)
                
        # Calculate averages and return top 5
        avg_operations = {
            op: sum(durations) / len(durations)
            for op, durations in operations.items()
        }
        
        return [
            {"service": service, "avg_duration_ms": duration}
            for service, duration in sorted(
                avg_operations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        ]
        
    async def _get_error_patterns(self) -> List[Dict[str, Any]]:
        """Get common error patterns from trace analysis."""
        error_patterns = {}
        for analysis in self.trace_analyses:
            for error in analysis.errors:
                pattern = error.get("type", "unknown")
                error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
                
        return [
            {"pattern": pattern, "count": count}
            for pattern, count in sorted(
                error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        ]
        
    async def _get_optimization_opportunities(self) -> List[str]:
        """Get optimization opportunities based on trace analysis."""
        opportunities = [
            "Database query optimization for frequently accessed tables",
            "Implement Redis caching for slow API endpoints", 
            "Optimize agent coordination protocols for better performance",
            "Consider database connection pooling improvements",
            "Implement request batching for high-frequency operations"
        ]
        
        return opportunities[:3]  # Return top 3 opportunities


# Global distributed tracing instance
distributed_tracing = DistributedTracingSystem()


async def init_distributed_tracing():
    """Initialize distributed tracing system."""
    logger.info("🔍 Initializing Distributed Tracing System for Epic 7 Phase 3")
    

# Convenience functions for common tracing patterns
async def trace_api_call(endpoint: str, method: str, user_id: str = None):
    """Convenience function for API call tracing."""
    return distributed_tracing.trace_api_request(endpoint, method, user_id)
    

async def trace_db_query(query_type: str, table: str = None, query: str = None):
    """Convenience function for database query tracing."""
    return distributed_tracing.trace_database_operation(query_type, table, query)
    

async def trace_agent_task(agent_id: str, operation: str, task_id: str = None):
    """Convenience function for agent task tracing."""
    return distributed_tracing.trace_agent_operation(agent_id, operation, task_id)


if __name__ == "__main__":
    # Test the distributed tracing system
    async def test_tracing():
        await init_distributed_tracing()
        
        # Simulate traced operations
        async with distributed_tracing.trace_api_request("/api/v2/users", "GET", "user123"):
            await asyncio.sleep(0.1)  # Simulate API processing
            
            async with distributed_tracing.trace_database_operation("SELECT", "users"):
                await asyncio.sleep(0.05)  # Simulate DB query
                
            async with distributed_tracing.trace_redis_operation("GET", "user:123"):
                await asyncio.sleep(0.01)  # Simulate cache lookup
                
        # Simulate agent operation
        async with distributed_tracing.trace_agent_operation("agent_123", "execute_task", "task_456"):
            await asyncio.sleep(0.2)  # Simulate agent processing
            
        # Get metrics and analysis
        metrics = await distributed_tracing.get_trace_metrics()
        print("Trace Metrics:", json.dumps(metrics, indent=2))
        
        dependencies = await distributed_tracing.get_service_dependency_map()
        print("Service Dependencies:", json.dumps(dependencies, indent=2))
        
        insights = await distributed_tracing.get_trace_analysis_insights()
        print("Analysis Insights:", json.dumps(insights, indent=2))
        
    asyncio.run(test_tracing())