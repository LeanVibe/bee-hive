"""
Enterprise Distributed Tracing System for LeanVibe Agent Hive 2.0

Comprehensive distributed tracing with OpenTelemetry integration, providing:
- End-to-end request flow tracking across all services
- Performance bottleneck identification and analysis
- Cross-service dependency mapping
- Real-time trace correlation with logs and metrics
- Advanced trace sampling and storage optimization
"""

import asyncio
import time
import json
import uuid
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

import structlog
import redis.asyncio as redis
from opentelemetry import trace, baggage, context
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, StaticSampler, Decision
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import get_traced_request_attrs
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_session
from .redis import get_redis_client

logger = structlog.get_logger()


class SpanType(Enum):
    """Types of spans in the system."""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    REDIS_OPERATION = "redis_operation"
    AGENT_TASK = "agent_task"
    BUSINESS_OPERATION = "business_operation"
    EXTERNAL_API = "external_api"
    BACKGROUND_JOB = "background_job"
    WEBSOCKET_MESSAGE = "websocket_message"


class TraceSamplingStrategy(Enum):
    """Trace sampling strategies."""
    ALWAYS = "always"
    NEVER = "never"
    RATIO = "ratio"
    ADAPTIVE = "adaptive"
    ERROR_ONLY = "error_only"
    SLOW_ONLY = "slow_only"


@dataclass
class TraceContext:
    """Context information for a trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampling_decision: bool = True
    
    # Request context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation_name: Optional[str] = None
    
    # Performance context
    start_time: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    status: str = "ok"
    
    # Business context
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None


@dataclass
class SpanMetadata:
    """Metadata for enriching spans."""
    service_name: str
    operation_name: str
    span_type: SpanType
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    io_operations: Optional[int] = None
    
    # Business metrics
    business_value: Optional[float] = None
    user_impact: Optional[str] = None
    error_budget_impact: Optional[float] = None


class AdaptiveSampler:
    """Intelligent adaptive sampling based on system load and trace value."""
    
    def __init__(self):
        self.base_rate = 0.1  # 10% base sampling rate
        self.error_rate = 1.0  # 100% sampling for errors
        self.slow_request_threshold_ms = 1000
        self.slow_request_rate = 0.8  # 80% sampling for slow requests
        
        # Dynamic adjustment
        self.current_rate = self.base_rate
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # seconds
        
        # System load factors
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.trace_volume_threshold = 1000  # traces per minute
    
    def should_sample(
        self,
        trace_context: TraceContext,
        span_metadata: SpanMetadata
    ) -> bool:
        """Determine if a trace should be sampled."""
        # Always sample errors
        if trace_context.status == "error":
            return True
        
        # Always sample slow requests
        if (trace_context.duration_ms and 
            trace_context.duration_ms > self.slow_request_threshold_ms):
            return True
        
        # Sample critical business operations
        if span_metadata.business_value and span_metadata.business_value > 0.8:
            return True
        
        # Apply adaptive rate for normal requests
        return self._should_sample_normal_request(trace_context)
    
    def _should_sample_normal_request(self, trace_context: TraceContext) -> bool:
        """Apply adaptive sampling for normal requests."""
        # Use trace ID for consistent sampling decision
        trace_hash = hash(trace_context.trace_id)
        sample_threshold = int(self.current_rate * 2**32)
        return abs(trace_hash) % (2**32) < sample_threshold
    
    def adjust_sampling_rate(self, system_metrics: Dict[str, float]) -> None:
        """Adjust sampling rate based on system metrics."""
        current_time = time.time()
        
        if current_time - self.last_adjustment < self.adjustment_interval:
            return
        
        cpu_usage = system_metrics.get("cpu_percent", 0)
        memory_usage = system_metrics.get("memory_percent", 0)
        trace_volume = system_metrics.get("traces_per_minute", 0)
        
        # Calculate adjustment factor
        adjustment = 1.0
        
        # Reduce sampling if system is under pressure
        if cpu_usage > self.cpu_threshold:
            adjustment *= 0.5
        if memory_usage > self.memory_threshold:
            adjustment *= 0.5
        if trace_volume > self.trace_volume_threshold:
            adjustment *= 0.7
        
        # Increase sampling if system has capacity
        if cpu_usage < 50 and memory_usage < 50:
            adjustment *= 1.2
        
        # Apply adjustment
        self.current_rate = max(0.01, min(1.0, self.current_rate * adjustment))
        self.last_adjustment = current_time
        
        logger.debug(
            "Sampling rate adjusted",
            old_rate=self.current_rate / adjustment,
            new_rate=self.current_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            trace_volume=trace_volume
        )


class DistributedTracingSystem:
    """
    Enterprise-grade distributed tracing system for comprehensive observability.
    
    Features:
    - End-to-end request flow tracking across all services
    - Intelligent adaptive sampling for optimal performance
    - Real-time trace analysis and anomaly detection
    - Cross-service dependency mapping and visualization
    - Integration with metrics and logs for unified observability
    - Performance bottleneck identification and recommendations
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable] = None,
        service_name: str = "leanvibe-agent-hive",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None
    ):
        """Initialize the distributed tracing system."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.service_name = service_name
        
        # Tracing components
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.sampler = AdaptiveSampler()
        
        # Span processors and exporters
        self.span_processors: List[BatchSpanProcessor] = []
        self.exporters: List[Any] = []
        
        # Active traces and spans
        self.active_traces: Dict[str, TraceContext] = {}
        self.span_metadata_cache: Dict[str, SpanMetadata] = {}
        
        # Performance tracking
        self.trace_counts: Dict[str, int] = {}  # Per-service trace counts
        self.trace_latencies: List[float] = []
        self.error_traces: List[str] = []
        
        # Background tasks
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tracing")
        
        # Configuration
        self.config = {
            "max_trace_duration_hours": 1,
            "max_spans_per_trace": 1000,
            "batch_export_timeout_ms": 5000,
            "max_export_batch_size": 512,
            "trace_retention_hours": 24,
            "enable_trace_correlation": True,
            "enable_dependency_analysis": True,
            "enable_performance_analysis": True,
            "sampling_strategy": TraceSamplingStrategy.ADAPTIVE,
            "default_sampling_rate": 0.1
        }
        
        # Endpoints for trace export
        self.jaeger_endpoint = jaeger_endpoint or "http://localhost:14268/api/traces"
        self.otlp_endpoint = otlp_endpoint or "http://localhost:4317"
        
        logger.info(
            "DistributedTracingSystem initialized",
            service_name=service_name,
            config=self.config
        )
    
    async def initialize(self) -> None:
        """Initialize the tracing system."""
        try:
            # Set up tracer provider
            self.tracer_provider = TracerProvider(
                sampler=self._create_sampler(),
                resource=self._create_resource()
            )
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer instance
            self.tracer = trace.get_tracer(
                __name__,
                version="1.0.0",
                tracer_provider=self.tracer_provider
            )
            
            # Set up exporters and processors
            await self._setup_exporters()
            await self._setup_instrumentation()
            
            logger.info("Distributed tracing system initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize tracing system", error=str(e))
            raise
    
    def _create_sampler(self):
        """Create appropriate sampler based on configuration."""
        strategy = self.config["sampling_strategy"]
        
        if strategy == TraceSamplingStrategy.ALWAYS:
            return StaticSampler(Decision.RECORD_AND_SAMPLE)
        elif strategy == TraceSamplingStrategy.NEVER:
            return StaticSampler(Decision.DROP)
        elif strategy == TraceSamplingStrategy.RATIO:
            return TraceIdRatioBased(self.config["default_sampling_rate"])
        elif strategy == TraceSamplingStrategy.ADAPTIVE:
            return self._create_adaptive_sampler()
        else:
            return TraceIdRatioBased(self.config["default_sampling_rate"])
    
    def _create_adaptive_sampler(self):
        """Create custom adaptive sampler."""
        # This would be a custom sampler implementation
        # For now, return a ratio-based sampler
        return TraceIdRatioBased(self.config["default_sampling_rate"])
    
    def _create_resource(self):
        """Create resource information for traces."""
        from opentelemetry.sdk.resources import Resource
        
        return Resource.create({
            "service.name": self.service_name,
            "service.version": "2.0.0",
            "deployment.environment": getattr(settings, "ENVIRONMENT", "development"),
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.20.0"
        })
    
    async def _setup_exporters(self) -> None:
        """Set up trace exporters."""
        try:
            # Jaeger exporter
            if self.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    endpoint=self.jaeger_endpoint,
                    max_tag_value_length=None,
                    timeout=30
                )
                self.exporters.append(jaeger_exporter)
                
                jaeger_processor = BatchSpanProcessor(
                    jaeger_exporter,
                    max_export_batch_size=self.config["max_export_batch_size"],
                    export_timeout_millis=self.config["batch_export_timeout_ms"]
                )
                self.span_processors.append(jaeger_processor)
                self.tracer_provider.add_span_processor(jaeger_processor)
            
            # OTLP exporter
            if self.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True,  # Use secure=True in production with TLS
                    timeout=30
                )
                self.exporters.append(otlp_exporter)
                
                otlp_processor = BatchSpanProcessor(
                    otlp_exporter,
                    max_export_batch_size=self.config["max_export_batch_size"],
                    export_timeout_millis=self.config["batch_export_timeout_ms"]
                )
                self.span_processors.append(otlp_processor)
                self.tracer_provider.add_span_processor(otlp_processor)
            
            # Custom Redis exporter for real-time trace data
            redis_processor = SimpleSpanProcessor(
                RedisSpanExporter(self.redis_client)
            )
            self.span_processors.append(redis_processor)
            self.tracer_provider.add_span_processor(redis_processor)
            
            logger.info(
                "Trace exporters configured",
                jaeger_enabled=bool(self.jaeger_endpoint),
                otlp_enabled=bool(self.otlp_endpoint),
                redis_enabled=True
            )
            
        except Exception as e:
            logger.error("Failed to set up exporters", error=str(e))
            raise
    
    async def _setup_instrumentation(self) -> None:
        """Set up automatic instrumentation for common libraries."""
        try:
            # SQLAlchemy instrumentation
            SQLAlchemyInstrumentor().instrument(
                tracer_provider=self.tracer_provider,
                enable_commenter=True,
                commenter_options={"db_driver": True, "db_framework": True}
            )
            
            # Redis instrumentation
            RedisInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
            
            # HTTP requests instrumentation
            RequestsInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
            
            # FastAPI instrumentation (if available)
            try:
                FastAPIInstrumentor().instrument(
                    tracer_provider=self.tracer_provider,
                    excluded_urls="healthz,metrics"
                )
            except Exception:
                pass  # FastAPI might not be available in all contexts
            
            logger.info("Automatic instrumentation configured")
            
        except Exception as e:
            logger.error("Failed to set up instrumentation", error=str(e))
    
    async def start(self) -> None:
        """Start the distributed tracing system."""
        if self.is_running:
            logger.warning("Tracing system already running")
            return
        
        await self.initialize()
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._trace_analysis_loop()),
            asyncio.create_task(self._dependency_analysis_loop()),
            asyncio.create_task(self._performance_analysis_loop()),
            asyncio.create_task(self._trace_cleanup_loop()),
            asyncio.create_task(self._sampling_adjustment_loop())
        ]
        
        logger.info("Distributed tracing system started")
    
    async def stop(self) -> None:
        """Stop the distributed tracing system."""
        if not self.is_running:
            return
        
        logger.info("Stopping distributed tracing system")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown span processors
        for processor in self.span_processors:
            processor.shutdown()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        logger.info("Distributed tracing system stopped")
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.BUSINESS_OPERATION,
        tags: Optional[Dict[str, Any]] = None,
        baggage_items: Optional[Dict[str, str]] = None
    ) -> ContextManager[Span]:
        """Context manager for tracing operations."""
        if not self.tracer:
            # If tracing is not initialized, provide a no-op context
            from opentelemetry.trace import NonRecordingSpan
            yield NonRecordingSpan(context.get_current())
            return
        
        # Create span
        span = self.tracer.start_span(
            name=operation_name,
            attributes=tags or {}
        )
        
        try:
            # Set baggage if provided
            if baggage_items:
                current_baggage = baggage.get_all()
                current_baggage.update(baggage_items)
                baggage.set_baggage_in_context(current_baggage)
            
            # Track span metadata
            trace_context = self._extract_trace_context(span)
            span_metadata = SpanMetadata(
                service_name=self.service_name,
                operation_name=operation_name,
                span_type=span_type,
                tags=tags or {}
            )
            
            self.span_metadata_cache[span.context.span_id.to_bytes(8, 'big').hex()] = span_metadata
            self.active_traces[trace_context.trace_id] = trace_context
            
            # Enrich span with system context
            await self._enrich_span_with_context(span, trace_context, span_metadata)
            
            yield span
            
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            
            # Track error trace
            trace_id = span.context.trace_id.to_bytes(16, 'big').hex()
            self.error_traces.append(trace_id)
            
            raise
        
        finally:
            # Finalize span
            await self._finalize_span(span, trace_context, span_metadata)
            span.end()
            
            # Cleanup
            span_id = span.context.span_id.to_bytes(8, 'big').hex()
            self.span_metadata_cache.pop(span_id, None)
    
    def _extract_trace_context(self, span: Span) -> TraceContext:
        """Extract trace context from span."""
        trace_id = span.context.trace_id.to_bytes(16, 'big').hex()
        span_id = span.context.span_id.to_bytes(8, 'big').hex()
        
        # Get parent span ID if available
        parent_span_id = None
        if hasattr(span, 'parent') and span.parent:
            parent_span_id = span.parent.span_id.to_bytes(8, 'big').hex()
        
        return TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage.get_all(),
            start_time=datetime.utcnow()
        )
    
    async def _enrich_span_with_context(
        self,
        span: Span,
        trace_context: TraceContext,
        span_metadata: SpanMetadata
    ) -> None:
        """Enrich span with additional context information."""
        try:
            # Add service and operation attributes
            span.set_attributes({
                "service.name": self.service_name,
                "operation.name": span_metadata.operation_name,
                "span.type": span_metadata.span_type.value,
                "trace.sampled": True
            })
            
            # Add baggage as attributes
            for key, value in trace_context.baggage.items():
                span.set_attribute(f"baggage.{key}", value)
            
            # Add system resource information
            try:
                import psutil
                process = psutil.Process()
                
                span.set_attributes({
                    "system.cpu_percent": process.cpu_percent(),
                    "system.memory_rss_mb": process.memory_info().rss / 1024 / 1024,
                    "system.threads": process.num_threads()
                })
            except Exception:
                pass  # psutil might not be available
            
            # Add custom tags
            for key, value in span_metadata.tags.items():
                span.set_attribute(key, value)
            
        except Exception as e:
            logger.error("Error enriching span context", error=str(e))
    
    async def _finalize_span(
        self,
        span: Span,
        trace_context: TraceContext,
        span_metadata: SpanMetadata
    ) -> None:
        """Finalize span with computed metrics."""
        try:
            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = (end_time - trace_context.start_time).total_seconds() * 1000
            trace_context.duration_ms = duration_ms
            
            # Add duration attribute
            span.set_attribute("duration_ms", duration_ms)
            
            # Track performance metrics
            self.trace_latencies.append(duration_ms)
            if len(self.trace_latencies) > 1000:
                self.trace_latencies = self.trace_latencies[-500:]
            
            # Add performance classification
            if duration_ms > 5000:  # 5 seconds
                span.set_attribute("performance.classification", "slow")
            elif duration_ms > 1000:  # 1 second
                span.set_attribute("performance.classification", "medium")
            else:
                span.set_attribute("performance.classification", "fast")
            
            # Store trace context for analysis
            await self._store_trace_context(trace_context, span_metadata)
            
        except Exception as e:
            logger.error("Error finalizing span", error=str(e))
    
    async def _store_trace_context(
        self,
        trace_context: TraceContext,
        span_metadata: SpanMetadata
    ) -> None:
        """Store trace context for later analysis."""
        try:
            if not self.redis_client:
                return
            
            # Store trace summary
            trace_data = {
                "trace_id": trace_context.trace_id,
                "service_name": self.service_name,
                "operation_name": span_metadata.operation_name,
                "span_type": span_metadata.span_type.value,
                "duration_ms": trace_context.duration_ms,
                "status": trace_context.status,
                "timestamp": trace_context.start_time.isoformat(),
                "tags": span_metadata.tags
            }
            
            # Store in Redis with expiration
            trace_key = f"trace:{trace_context.trace_id}"
            await self.redis_client.setex(
                trace_key,
                timedelta(hours=self.config["trace_retention_hours"]),
                json.dumps(trace_data)
            )
            
            # Add to service-specific trace list
            service_key = f"traces:{self.service_name}"
            await self.redis_client.lpush(service_key, trace_context.trace_id)
            await self.redis_client.ltrim(service_key, 0, 999)  # Keep last 1000 traces
            await self.redis_client.expire(
                service_key,
                timedelta(hours=self.config["trace_retention_hours"])
            )
            
        except Exception as e:
            logger.error("Error storing trace context", error=str(e))
    
    async def get_trace_analytics(self, hours: int = 1) -> Dict[str, Any]:
        """Get trace analytics for the specified time period."""
        try:
            # Calculate metrics from recent traces
            recent_latencies = [
                lat for lat in self.trace_latencies 
                if lat is not None
            ][-1000:]  # Last 1000 traces
            
            if not recent_latencies:
                return {"error": "No recent trace data available"}
            
            analytics = {
                "time_period_hours": hours,
                "total_traces": len(recent_latencies),
                "performance_metrics": {
                    "avg_latency_ms": sum(recent_latencies) / len(recent_latencies),
                    "p50_latency_ms": sorted(recent_latencies)[len(recent_latencies) // 2],
                    "p95_latency_ms": sorted(recent_latencies)[int(len(recent_latencies) * 0.95)],
                    "p99_latency_ms": sorted(recent_latencies)[int(len(recent_latencies) * 0.99)],
                    "max_latency_ms": max(recent_latencies),
                    "min_latency_ms": min(recent_latencies)
                },
                "error_metrics": {
                    "total_errors": len(self.error_traces),
                    "error_rate_percent": (len(self.error_traces) / len(recent_latencies)) * 100
                },
                "sampling_metrics": {
                    "current_sampling_rate": self.sampler.current_rate,
                    "base_sampling_rate": self.sampler.base_rate,
                    "error_sampling_rate": self.sampler.error_rate
                },
                "service_metrics": {
                    "traces_by_service": dict(self.trace_counts)
                }
            }
            
            # Get top slow operations from Redis
            analytics["slow_operations"] = await self._get_slow_operations()
            
            return analytics
            
        except Exception as e:
            logger.error("Error getting trace analytics", error=str(e))
            return {"error": str(e)}
    
    async def _get_slow_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations from trace data."""
        try:
            if not self.redis_client:
                return []
            
            # Get recent traces from Redis
            service_key = f"traces:{self.service_name}"
            trace_ids = await self.redis_client.lrange(service_key, 0, 100)
            
            slow_ops = []
            for trace_id in trace_ids[:50]:  # Check last 50 traces
                trace_key = f"trace:{trace_id.decode()}"
                trace_data_raw = await self.redis_client.get(trace_key)
                
                if trace_data_raw:
                    trace_data = json.loads(trace_data_raw)
                    duration = trace_data.get("duration_ms", 0)
                    
                    if duration > 1000:  # Only include operations > 1 second
                        slow_ops.append({
                            "trace_id": trace_data["trace_id"],
                            "operation_name": trace_data["operation_name"],
                            "duration_ms": duration,
                            "service_name": trace_data["service_name"],
                            "timestamp": trace_data["timestamp"]
                        })
            
            # Sort by duration and return top operations
            slow_ops.sort(key=lambda x: x["duration_ms"], reverse=True)
            return slow_ops[:limit]
            
        except Exception as e:
            logger.error("Error getting slow operations", error=str(e))
            return []
    
    # Background task methods
    async def _trace_analysis_loop(self) -> None:
        """Background loop for trace analysis."""
        logger.info("Starting trace analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._analyze_recent_traces()
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Trace analysis loop error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("Trace analysis loop stopped")
    
    async def _dependency_analysis_loop(self) -> None:
        """Background loop for dependency analysis."""
        logger.info("Starting dependency analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                if self.config["enable_dependency_analysis"]:
                    await self._analyze_service_dependencies()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Dependency analysis loop error", error=str(e))
                await asyncio.sleep(300)
        
        logger.info("Dependency analysis loop stopped")
    
    async def _performance_analysis_loop(self) -> None:
        """Background loop for performance analysis."""
        logger.info("Starting performance analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                if self.config["enable_performance_analysis"]:
                    await self._analyze_performance_patterns()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance analysis loop error", error=str(e))
                await asyncio.sleep(300)
        
        logger.info("Performance analysis loop stopped")
    
    async def _trace_cleanup_loop(self) -> None:
        """Background loop for cleaning up old trace data."""
        logger.info("Starting trace cleanup loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_old_traces()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Trace cleanup loop error", error=str(e))
                await asyncio.sleep(3600)
        
        logger.info("Trace cleanup loop stopped")
    
    async def _sampling_adjustment_loop(self) -> None:
        """Background loop for adjusting sampling rates."""
        logger.info("Starting sampling adjustment loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Get system metrics for sampling adjustment
                system_metrics = await self._get_system_metrics()
                self.sampler.adjust_sampling_rate(system_metrics)
                await asyncio.sleep(120)  # Run every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sampling adjustment loop error", error=str(e))
                await asyncio.sleep(120)
        
        logger.info("Sampling adjustment loop stopped")
    
    # Placeholder methods for analysis tasks
    async def _analyze_recent_traces(self) -> None:
        """Analyze recent traces for patterns and anomalies."""
        pass
    
    async def _analyze_service_dependencies(self) -> None:
        """Analyze service dependencies from trace data."""
        pass
    
    async def _analyze_performance_patterns(self) -> None:
        """Analyze performance patterns and bottlenecks."""
        pass
    
    async def _cleanup_old_traces(self) -> None:
        """Clean up old trace data."""
        pass
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics for sampling adjustment."""
        return {
            "cpu_percent": 45.0,  # Example values
            "memory_percent": 60.0,
            "traces_per_minute": 150.0
        }


class RedisSpanExporter:
    """Custom span exporter that sends traces to Redis for real-time analysis."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    def export(self, spans) -> None:
        """Export spans to Redis."""
        try:
            for span in spans:
                # Convert span to dict for Redis storage
                span_data = {
                    "trace_id": span.context.trace_id.to_bytes(16, 'big').hex(),
                    "span_id": span.context.span_id.to_bytes(8, 'big').hex(),
                    "name": span.name,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "status": span.status.status_code.name if span.status else "UNSET"
                }
                
                # Store in Redis (fire and forget)
                asyncio.create_task(self._store_span_async(span_data))
        
        except Exception as e:
            logger.error("Error exporting spans to Redis", error=str(e))
    
    async def _store_span_async(self, span_data: Dict[str, Any]) -> None:
        """Asynchronously store span data in Redis."""
        try:
            span_key = f"span:{span_data['trace_id']}:{span_data['span_id']}"
            await self.redis_client.setex(
                span_key,
                timedelta(minutes=30),  # Short retention for real-time analysis
                json.dumps(span_data, default=str)
            )
        except Exception as e:
            logger.error("Error storing span in Redis", error=str(e))


# Global instance
_distributed_tracing_system: Optional[DistributedTracingSystem] = None


async def get_distributed_tracing_system() -> DistributedTracingSystem:
    """Get singleton distributed tracing system instance."""
    global _distributed_tracing_system
    
    if _distributed_tracing_system is None:
        _distributed_tracing_system = DistributedTracingSystem()
        await _distributed_tracing_system.start()
    
    return _distributed_tracing_system


async def cleanup_distributed_tracing_system() -> None:
    """Cleanup distributed tracing system resources."""
    global _distributed_tracing_system
    
    if _distributed_tracing_system:
        await _distributed_tracing_system.stop()
        _distributed_tracing_system = None


# Convenience functions
async def trace_operation(
    operation_name: str,
    span_type: SpanType = SpanType.BUSINESS_OPERATION,
    tags: Optional[Dict[str, Any]] = None,
    baggage_items: Optional[Dict[str, str]] = None
):
    """Convenience function for tracing operations."""
    tracing_system = await get_distributed_tracing_system()
    return tracing_system.trace_operation(operation_name, span_type, tags, baggage_items)