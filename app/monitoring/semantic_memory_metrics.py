"""
Prometheus Metrics Integration for Semantic Memory Performance Monitoring.

This module provides comprehensive metrics collection for semantic memory
operations with Prometheus integration for real-time monitoring and alerting.

Features:
- Real-time performance metrics collection
- Prometheus metrics exposition
- Custom performance dashboards
- Alert rule definitions
- Historical performance tracking
- SLA monitoring and reporting
- Resource utilization tracking
- Error rate and reliability metrics
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server, multiprocess, values
)
from prometheus_client.core import REGISTRY

logger = logging.getLogger(__name__)


# Custom metric collectors
class SemanticMemoryMetrics:
    """Prometheus metrics for semantic memory system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        
        # Operation counters
        self.operations_total = Counter(
            'semantic_memory_operations_total',
            'Total number of semantic memory operations',
            ['operation_type', 'status', 'agent_id'],
            registry=self.registry
        )
        
        # Latency histograms
        self.operation_duration = Histogram(
            'semantic_memory_operation_duration_seconds',
            'Duration of semantic memory operations',
            ['operation_type', 'agent_id'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Search-specific metrics
        self.search_latency = Histogram(
            'semantic_memory_search_latency_seconds',
            'Semantic search operation latency',
            ['search_type', 'agent_id'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.search_results_count = Histogram(
            'semantic_memory_search_results_count',
            'Number of results returned by search operations',
            ['search_type', 'agent_id'],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
            registry=self.registry
        )
        
        # Ingestion metrics
        self.ingestion_throughput = Gauge(
            'semantic_memory_ingestion_throughput_docs_per_second',
            'Document ingestion throughput',
            ['batch_size_range', 'agent_id'],
            registry=self.registry
        )
        
        self.ingestion_batch_size = Histogram(
            'semantic_memory_ingestion_batch_size',
            'Size of document ingestion batches',
            ['agent_id'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry
        )
        
        # Context compression metrics
        self.compression_ratio = Histogram(
            'semantic_memory_compression_ratio',
            'Context compression ratio achieved',
            ['compression_method', 'agent_id'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            registry=self.registry
        )
        
        self.compression_duration = Histogram(
            'semantic_memory_compression_duration_seconds',
            'Context compression operation duration',
            ['compression_method', 'context_size_range'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.semantic_preservation_score = Histogram(
            'semantic_memory_semantic_preservation_score',
            'Semantic preservation score after compression',
            ['compression_method'],
            buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
            registry=self.registry
        )
        
        # Knowledge sharing metrics
        self.knowledge_transfer_latency = Histogram(
            'semantic_memory_knowledge_transfer_latency_seconds',
            'Cross-agent knowledge transfer latency',
            ['knowledge_type', 'source_agent', 'target_agent'],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.knowledge_cache_hits = Counter(
            'semantic_memory_knowledge_cache_hits_total',
            'Knowledge cache hit operations',
            ['cache_type', 'agent_id'],
            registry=self.registry
        )
        
        self.knowledge_cache_misses = Counter(
            'semantic_memory_knowledge_cache_misses_total',
            'Knowledge cache miss operations',
            ['cache_type', 'agent_id'],
            registry=self.registry
        )
        
        # Resource utilization metrics
        self.memory_usage_bytes = Gauge(
            'semantic_memory_memory_usage_bytes',
            'Memory usage by semantic memory components',
            ['component'],
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'semantic_memory_cpu_usage_percent',
            'CPU usage by semantic memory components',
            ['component'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'semantic_memory_active_connections',
            'Number of active database connections',
            ['connection_pool'],
            registry=self.registry
        )
        
        # Vector index metrics
        self.vector_index_size = Gauge(
            'semantic_memory_vector_index_size_mb',
            'Size of vector indexes in megabytes',
            ['index_type', 'agent_id'],
            registry=self.registry
        )
        
        self.vector_index_documents = Gauge(
            'semantic_memory_vector_index_documents_total',
            'Total number of documents in vector indexes',
            ['index_type', 'agent_id'],
            registry=self.registry
        )
        
        # Error tracking
        self.errors_total = Counter(
            'semantic_memory_errors_total',
            'Total number of errors in semantic memory operations',
            ['operation_type', 'error_type', 'agent_id'],
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'semantic_memory_error_rate',
            'Error rate for semantic memory operations',
            ['operation_type', 'time_window'],
            registry=self.registry
        )
        
        # SLA and performance targets
        self.sla_compliance = Gauge(
            'semantic_memory_sla_compliance_percent',
            'SLA compliance percentage',
            ['sla_type', 'time_window'],
            registry=self.registry
        )
        
        self.performance_target_achievement = Gauge(
            'semantic_memory_performance_target_achievement',
            'Achievement of performance targets (1=met, 0=not met)',
            ['target_type', 'target_name'],
            registry=self.registry
        )
        
        # Workflow integration metrics
        self.workflow_overhead = Histogram(
            'semantic_memory_workflow_overhead_seconds',
            'Additional overhead when used in workflow context',
            ['workflow_type', 'semantic_operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )
        
        # Business metrics
        self.agent_productivity = Gauge(
            'semantic_memory_agent_productivity_score',
            'Agent productivity score based on semantic memory usage',
            ['agent_id', 'productivity_metric'],
            registry=self.registry
        )
        
        # System health indicators
        self.system_health_score = Gauge(
            'semantic_memory_system_health_score',
            'Overall system health score (0-1)',
            ['component'],
            registry=self.registry
        )
        
        # Performance regression indicators
        self.regression_detected = Counter(
            'semantic_memory_regression_detected_total',
            'Number of performance regressions detected',
            ['metric_name', 'severity', 'detection_method'],
            registry=self.registry
        )
        
        # Info metrics
        self.build_info = Info(
            'semantic_memory_build_info',
            'Build information for semantic memory system',
            registry=self.registry
        )
        
        # Initialize build info
        self.build_info.info({
            'version': '1.0.0',
            'build_time': datetime.utcnow().isoformat(),
            'component': 'semantic_memory'
        })


class MetricsCollector:
    """Collects and manages semantic memory metrics."""
    
    def __init__(self, metrics: SemanticMemoryMetrics):
        self.metrics = metrics
        self.start_time = time.time()
        
        # Sliding window for rate calculations
        self.operation_windows = defaultdict(lambda: deque(maxlen=100))
        self.error_windows = defaultdict(lambda: deque(maxlen=100))
        
        # Performance target tracking
        self.performance_targets = {
            'search_latency_p95_ms': 200.0,
            'ingestion_throughput_docs_per_sec': 500.0,
            'compression_time_ms': 500.0,
            'knowledge_transfer_latency_ms': 200.0,
            'workflow_overhead_ms': 10.0
        }
        
        # SLA definitions
        self.sla_targets = {
            'availability_percent': 99.9,
            'response_time_p95_ms': 200.0,
            'error_rate_percent': 1.0
        }
        
        # Background tasks
        self._monitoring_tasks = []
        self._running = False
    
    def record_operation(
        self,
        operation_type: str,
        duration_seconds: float,
        status: str = 'success',
        agent_id: str = 'unknown',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a semantic memory operation."""
        # Update counters and histograms
        self.metrics.operations_total.labels(
            operation_type=operation_type,
            status=status,
            agent_id=agent_id
        ).inc()
        
        self.metrics.operation_duration.labels(
            operation_type=operation_type,
            agent_id=agent_id
        ).observe(duration_seconds)
        
        # Track in sliding window for rate calculations
        timestamp = time.time()
        self.operation_windows[operation_type].append({
            'timestamp': timestamp,
            'duration': duration_seconds,
            'status': status,
            'agent_id': agent_id
        })
        
        # Record error if failed
        if status != 'success':
            self.record_error(operation_type, status, agent_id)
    
    def record_search_operation(
        self,
        search_type: str,
        duration_seconds: float,
        results_count: int,
        agent_id: str = 'unknown',
        status: str = 'success'
    ):
        """Record a search operation with specific metrics."""
        self.record_operation('search', duration_seconds, status, agent_id)
        
        self.metrics.search_latency.labels(
            search_type=search_type,
            agent_id=agent_id
        ).observe(duration_seconds)
        
        self.metrics.search_results_count.labels(
            search_type=search_type,
            agent_id=agent_id
        ).observe(results_count)
        
        # Check performance targets
        latency_ms = duration_seconds * 1000
        target_met = latency_ms <= self.performance_targets['search_latency_p95_ms']
        self.metrics.performance_target_achievement.labels(
            target_type='latency',
            target_name='search_p95'
        ).set(1.0 if target_met else 0.0)
    
    def record_ingestion_operation(
        self,
        duration_seconds: float,
        documents_count: int,
        batch_size: int,
        agent_id: str = 'unknown',
        status: str = 'success'
    ):
        """Record a document ingestion operation."""
        self.record_operation('ingestion', duration_seconds, status, agent_id)
        
        # Calculate throughput
        throughput = documents_count / duration_seconds if duration_seconds > 0 else 0
        
        # Categorize batch size
        if batch_size <= 10:
            batch_range = 'small'
        elif batch_size <= 50:
            batch_range = 'medium'
        else:
            batch_range = 'large'
        
        self.metrics.ingestion_throughput.labels(
            batch_size_range=batch_range,
            agent_id=agent_id
        ).set(throughput)
        
        self.metrics.ingestion_batch_size.labels(
            agent_id=agent_id
        ).observe(batch_size)
        
        # Check performance targets
        target_met = throughput >= self.performance_targets['ingestion_throughput_docs_per_sec']
        self.metrics.performance_target_achievement.labels(
            target_type='throughput',
            target_name='ingestion_docs_per_sec'
        ).set(1.0 if target_met else 0.0)
    
    def record_compression_operation(
        self,
        compression_method: str,
        duration_seconds: float,
        compression_ratio: float,
        semantic_preservation_score: float,
        context_size: int,
        agent_id: str = 'unknown',
        status: str = 'success'
    ):
        """Record a context compression operation."""
        self.record_operation('compression', duration_seconds, status, agent_id)
        
        # Categorize context size
        if context_size <= 100:
            size_range = 'small'
        elif context_size <= 1000:
            size_range = 'medium'
        else:
            size_range = 'large'
        
        self.metrics.compression_duration.labels(
            compression_method=compression_method,
            context_size_range=size_range
        ).observe(duration_seconds)
        
        self.metrics.compression_ratio.labels(
            compression_method=compression_method,
            agent_id=agent_id
        ).observe(compression_ratio)
        
        self.metrics.semantic_preservation_score.labels(
            compression_method=compression_method
        ).observe(semantic_preservation_score)
        
        # Check performance targets
        duration_ms = duration_seconds * 1000
        target_met = duration_ms <= self.performance_targets['compression_time_ms']
        self.metrics.performance_target_achievement.labels(
            target_type='latency',
            target_name='compression_time'
        ).set(1.0 if target_met else 0.0)
    
    def record_knowledge_transfer(
        self,
        knowledge_type: str,
        duration_seconds: float,
        source_agent: str,
        target_agent: str,
        cache_hit: bool = False,
        status: str = 'success'
    ):
        """Record a knowledge transfer operation."""
        self.record_operation('knowledge_transfer', duration_seconds, status, target_agent)
        
        self.metrics.knowledge_transfer_latency.labels(
            knowledge_type=knowledge_type,
            source_agent=source_agent,
            target_agent=target_agent
        ).observe(duration_seconds)
        
        # Track cache performance
        if cache_hit:
            self.metrics.knowledge_cache_hits.labels(
                cache_type=knowledge_type,
                agent_id=target_agent
            ).inc()
        else:
            self.metrics.knowledge_cache_misses.labels(
                cache_type=knowledge_type,
                agent_id=target_agent
            ).inc()
        
        # Check performance targets
        latency_ms = duration_seconds * 1000
        target_met = latency_ms <= self.performance_targets['knowledge_transfer_latency_ms']
        self.metrics.performance_target_achievement.labels(
            target_type='latency',
            target_name='knowledge_transfer'
        ).set(1.0 if target_met else 0.0)
    
    def record_workflow_overhead(
        self,
        workflow_type: str,
        semantic_operation: str,
        overhead_seconds: float
    ):
        """Record workflow integration overhead."""
        self.metrics.workflow_overhead.labels(
            workflow_type=workflow_type,
            semantic_operation=semantic_operation
        ).observe(overhead_seconds)
        
        # Check performance targets
        overhead_ms = overhead_seconds * 1000
        target_met = overhead_ms <= self.performance_targets['workflow_overhead_ms']
        self.metrics.performance_target_achievement.labels(
            target_type='overhead',
            target_name='workflow_integration'
        ).set(1.0 if target_met else 0.0)
    
    def record_error(
        self,
        operation_type: str,
        error_type: str,
        agent_id: str = 'unknown'
    ):
        """Record an error occurrence."""
        self.metrics.errors_total.labels(
            operation_type=operation_type,
            error_type=error_type,
            agent_id=agent_id
        ).inc()
        
        # Track in sliding window
        timestamp = time.time()
        self.error_windows[operation_type].append({
            'timestamp': timestamp,
            'error_type': error_type,
            'agent_id': agent_id
        })
    
    def record_regression(
        self,
        metric_name: str,
        severity: str,
        detection_method: str
    ):
        """Record a performance regression detection."""
        self.metrics.regression_detected.labels(
            metric_name=metric_name,
            severity=severity,
            detection_method=detection_method
        ).inc()
    
    def update_resource_metrics(
        self,
        component: str,
        memory_usage_bytes: float,
        cpu_usage_percent: float
    ):
        """Update resource utilization metrics."""
        self.metrics.memory_usage_bytes.labels(component=component).set(memory_usage_bytes)
        self.metrics.cpu_usage_percent.labels(component=component).set(cpu_usage_percent)
    
    def update_vector_index_metrics(
        self,
        index_type: str,
        agent_id: str,
        size_mb: float,
        documents_count: int
    ):
        """Update vector index metrics."""
        self.metrics.vector_index_size.labels(
            index_type=index_type,
            agent_id=agent_id
        ).set(size_mb)
        
        self.metrics.vector_index_documents.labels(
            index_type=index_type,
            agent_id=agent_id
        ).set(documents_count)
    
    def update_agent_productivity(
        self,
        agent_id: str,
        productivity_metric: str,
        score: float
    ):
        """Update agent productivity metrics."""
        self.metrics.agent_productivity.labels(
            agent_id=agent_id,
            productivity_metric=productivity_metric
        ).set(score)
    
    def update_system_health(self, component: str, health_score: float):
        """Update system health score."""
        self.metrics.system_health_score.labels(component=component).set(health_score)
    
    def calculate_error_rates(self):
        """Calculate and update error rates."""
        current_time = time.time()
        
        for operation_type, error_window in self.error_windows.items():
            operation_window = self.operation_windows.get(operation_type, deque())
            
            # Count recent operations and errors
            recent_operations = sum(
                1 for op in operation_window
                if current_time - op['timestamp'] <= 300  # 5 minutes
            )
            
            recent_errors = sum(
                1 for error in error_window
                if current_time - error['timestamp'] <= 300  # 5 minutes
            )
            
            # Calculate error rate
            error_rate = (recent_errors / recent_operations * 100) if recent_operations > 0 else 0
            
            self.metrics.error_rate.labels(
                operation_type=operation_type,
                time_window='5m'
            ).set(error_rate)
    
    def calculate_sla_compliance(self):
        """Calculate and update SLA compliance metrics."""
        current_time = time.time()
        
        # Calculate availability (simplified - based on error rates)
        total_operations = 0
        total_errors = 0
        
        for operation_type, operation_window in self.operation_windows.items():
            recent_ops = [
                op for op in operation_window
                if current_time - op['timestamp'] <= 3600  # 1 hour
            ]
            
            total_operations += len(recent_ops)
            total_errors += sum(1 for op in recent_ops if op['status'] != 'success')
        
        availability = ((total_operations - total_errors) / total_operations * 100) if total_operations > 0 else 100
        
        self.metrics.sla_compliance.labels(
            sla_type='availability',
            time_window='1h'
        ).set(availability)
        
        # Calculate response time SLA (based on search operations)
        search_ops = [
            op for op in self.operation_windows.get('search', [])
            if current_time - op['timestamp'] <= 3600
        ]
        
        if search_ops:
            latencies = [op['duration'] * 1000 for op in search_ops]  # Convert to ms
            latencies.sort()
            p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0
            
            response_time_sla_met = p95_latency <= self.sla_targets['response_time_p95_ms']
            
            self.metrics.sla_compliance.labels(
                sla_type='response_time_p95',
                time_window='1h'
            ).set(100.0 if response_time_sla_met else 0.0)
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        self._running = True
        
        # Start periodic metric calculations
        self._monitoring_tasks = [
            asyncio.create_task(self._error_rate_calculator()),
            asyncio.create_task(self._sla_calculator()),
            asyncio.create_task(self._health_calculator())
        ]
        
        logger.info("📊 Semantic memory metrics monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._running = False
        
        # Cancel tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        logger.info("📊 Semantic memory metrics monitoring stopped")
    
    async def _error_rate_calculator(self):
        """Background task to calculate error rates."""
        while self._running:
            try:
                self.calculate_error_rates()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error rate calculation failed: {e}")
                await asyncio.sleep(30)
    
    async def _sla_calculator(self):
        """Background task to calculate SLA compliance."""
        while self._running:
            try:
                self.calculate_sla_compliance()
                await asyncio.sleep(60)  # Update every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SLA calculation failed: {e}")
                await asyncio.sleep(60)
    
    async def _health_calculator(self):
        """Background task to calculate system health."""
        while self._running:
            try:
                # Calculate overall system health based on multiple factors
                health_factors = {}
                
                # Error rate health (0-1 scale)
                total_error_rate = 0
                error_count = 0
                
                for operation_type in self.operation_windows.keys():
                    try:
                        error_rate = self.metrics.error_rate.labels(
                            operation_type=operation_type,
                            time_window='5m'
                        )._value._value  # Get current value
                        
                        total_error_rate += error_rate
                        error_count += 1
                    except:
                        pass
                
                avg_error_rate = total_error_rate / error_count if error_count > 0 else 0
                error_health = max(0, 1 - (avg_error_rate / 10))  # 10% error rate = 0 health
                health_factors['error_rate'] = error_health
                
                # Performance target health
                target_achievements = []
                for target_type in ['latency', 'throughput', 'overhead']:
                    try:
                        # This is a simplified approach - in practice, you'd aggregate across targets
                        achievement = 0.9  # Placeholder
                        target_achievements.append(achievement)
                    except:
                        pass
                
                performance_health = sum(target_achievements) / len(target_achievements) if target_achievements else 1.0
                health_factors['performance'] = performance_health
                
                # Calculate overall health
                overall_health = sum(health_factors.values()) / len(health_factors) if health_factors else 1.0
                
                self.update_system_health('overall', overall_health)
                self.update_system_health('error_rate', error_health)
                self.update_system_health('performance', performance_health)
                
                await asyncio.sleep(120)  # Update every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health calculation failed: {e}")
                await asyncio.sleep(120)


class PrometheusExporter:
    """Prometheus metrics exporter for semantic memory system."""
    
    def __init__(self, port: int = 8090, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.metrics = SemanticMemoryMetrics()
        self.collector = MetricsCollector(self.metrics)
        self._server = None
    
    def start_server(self):
        """Start Prometheus metrics server."""
        try:
            self._server = start_http_server(self.port, self.host)
            logger.info(f"📊 Prometheus metrics server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            return False
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        return generate_latest(self.metrics.registry)
    
    async def start_monitoring(self):
        """Start metrics collection and monitoring."""
        await self.collector.start_monitoring()
    
    async def stop_monitoring(self):
        """Stop metrics collection and monitoring."""
        await self.collector.stop_monitoring()
    
    def get_collector(self) -> MetricsCollector:
        """Get metrics collector for recording metrics."""
        return self.collector


# Global metrics instance
_prometheus_exporter: Optional[PrometheusExporter] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _prometheus_exporter
    
    if _prometheus_exporter is None:
        _prometheus_exporter = PrometheusExporter()
        # Start server in background if not already started
        _prometheus_exporter.start_server()
    
    return _prometheus_exporter.get_collector()


async def initialize_monitoring(port: int = 8090, host: str = '0.0.0.0') -> PrometheusExporter:
    """Initialize semantic memory monitoring system."""
    global _prometheus_exporter
    
    if _prometheus_exporter is None:
        _prometheus_exporter = PrometheusExporter(port, host)
        _prometheus_exporter.start_server()
        await _prometheus_exporter.start_monitoring()
    
    return _prometheus_exporter


async def shutdown_monitoring():
    """Shutdown monitoring system."""
    global _prometheus_exporter
    
    if _prometheus_exporter:
        await _prometheus_exporter.stop_monitoring()
        _prometheus_exporter = None


# Context manager for operation timing
class OperationTimer:
    """Context manager for timing semantic memory operations."""
    
    def __init__(
        self,
        operation_type: str,
        agent_id: str = 'unknown',
        collector: Optional[MetricsCollector] = None
    ):
        self.operation_type = operation_type
        self.agent_id = agent_id
        self.collector = collector or get_metrics_collector()
        self.start_time = None
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status = 'success' if exc_type is None else 'error'
        
        self.collector.record_operation(
            operation_type=self.operation_type,
            duration_seconds=duration,
            status=status,
            agent_id=self.agent_id,
            metadata=self.metadata
        )
        
        if exc_type is not None:
            self.collector.record_error(
                operation_type=self.operation_type,
                error_type=exc_type.__name__,
                agent_id=self.agent_id
            )
    
    def add_metadata(self, **kwargs):
        """Add metadata to the operation."""
        self.metadata.update(kwargs)


# Decorator for automatic operation timing
def monitor_operation(operation_type: str, agent_id_field: str = None):
    """Decorator to automatically monitor semantic memory operations."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Extract agent_id if specified
            agent_id = 'unknown'
            if agent_id_field and agent_id_field in kwargs:
                agent_id = kwargs[agent_id_field]
            
            with OperationTimer(operation_type, agent_id) as timer:
                result = await func(*args, **kwargs)
                
                # Add result metadata if available
                if hasattr(result, 'metadata'):
                    timer.add_metadata(**result.metadata)
                
                return result
        
        def sync_wrapper(*args, **kwargs):
            # Extract agent_id if specified
            agent_id = 'unknown'
            if agent_id_field and agent_id_field in kwargs:
                agent_id = kwargs[agent_id_field]
            
            with OperationTimer(operation_type, agent_id) as timer:
                result = func(*args, **kwargs)
                
                # Add result metadata if available
                if hasattr(result, 'metadata'):
                    timer.add_metadata(**result.metadata)
                
                return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator