"""
Enterprise Prometheus Metrics Exporter for LeanVibe Agent Hive 2.0

Comprehensive Prometheus metrics collection and export system providing:
- Real-time business KPIs and operational metrics
- Custom metric registration and collection
- High-performance metrics aggregation
- Prometheus-native format export
- Integration with existing performance monitoring system
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

import structlog
import redis.asyncio as redis
from prometheus_client import (
    Counter as PrometheusCounter, 
    Gauge, 
    Histogram, 
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Info,
    Enum as PrometheusEnum
)
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_session
from .redis import get_redis_client
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory

logger = structlog.get_logger()


class MetricCategory(Enum):
    """Categories for organizing Prometheus metrics."""
    BUSINESS = "business"
    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class PrometheusMetricDefinition:
    """Definition for a Prometheus metric."""
    name: str
    metric_type: str  # counter, gauge, histogram, summary
    help_text: str
    labels: List[str] = field(default_factory=list)
    category: MetricCategory = MetricCategory.CUSTOM
    buckets: Optional[List[float]] = None  # For histograms
    
    def create_metric(self, registry: CollectorRegistry):
        """Create the actual Prometheus metric object."""
        if self.metric_type == "counter":
            return PrometheusCounter(
                name=self.name,
                documentation=self.help_text,
                labelnames=self.labels,
                registry=registry
            )
        elif self.metric_type == "gauge":
            return Gauge(
                name=self.name,
                documentation=self.help_text,
                labelnames=self.labels,
                registry=registry
            )
        elif self.metric_type == "histogram":
            buckets = self.buckets or [0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
            return Histogram(
                name=self.name,
                documentation=self.help_text,
                labelnames=self.labels,
                buckets=buckets,
                registry=registry
            )
        elif self.metric_type == "summary":
            return Summary(
                name=self.name,
                documentation=self.help_text,
                labelnames=self.labels,
                registry=registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")


class PrometheusMetricsExporter:
    """
    Enterprise-grade Prometheus metrics exporter for LeanVibe Agent Hive 2.0.
    
    Features:
    - Comprehensive system and business metrics
    - Real-time metric collection and aggregation
    - Custom metric registration and management
    - High-performance metrics export
    - Integration with existing monitoring infrastructure
    - Automated metric lifecycle management
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable] = None,
        metrics_collector: Optional[PerformanceMetricsCollector] = None,
        collection_interval: float = 15.0  # seconds
    ):
        """Initialize the Prometheus metrics exporter."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.metrics_collector = metrics_collector
        self.collection_interval = collection_interval
        
        # Prometheus registry and metrics
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, PrometheusMetricDefinition] = {}
        
        # Collection state
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None
        self.last_collection = datetime.utcnow()
        
        # Performance tracking
        self.collection_latencies: List[float] = []
        self.export_latencies: List[float] = []
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prometheus")
        
        # Configuration
        self.config = {
            "max_metric_age_hours": 24,
            "metric_collection_timeout": 30.0,
            "high_cardinality_limit": 10000,
            "batch_size": 100,
            "enable_detailed_agent_metrics": True,
            "enable_business_metrics": True,
            "enable_security_metrics": True,
            "metric_prefix": "leanvibe_",
            "cache_metrics_seconds": 30
        }
        
        # Metric cache for performance
        self.metric_cache: Dict[str, Tuple[float, Any]] = {}
        self.cache_lock = threading.Lock()
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        logger.info(
            "PrometheusMetricsExporter initialized",
            collection_interval=collection_interval,
            registry_metrics=len(self.metrics)
        )
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core Prometheus metrics."""
        
        # System metrics
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}system_cpu_percent",
            metric_type="gauge",
            help_text="System CPU usage percentage",
            category=MetricCategory.SYSTEM
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}system_memory_usage_bytes",
            metric_type="gauge",
            help_text="System memory usage in bytes",
            category=MetricCategory.SYSTEM
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}system_memory_percent",
            metric_type="gauge",
            help_text="System memory usage percentage",
            category=MetricCategory.SYSTEM
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}system_disk_io_bytes_total",
            metric_type="counter",
            help_text="Total disk I/O bytes",
            labels=["direction"],  # read/write
            category=MetricCategory.SYSTEM
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}system_network_bytes_total",
            metric_type="counter",
            help_text="Total network bytes",
            labels=["direction"],  # sent/received
            category=MetricCategory.SYSTEM
        ))
        
        # Agent metrics
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agents_total",
            metric_type="gauge",
            help_text="Total number of agents",
            labels=["status", "type"],
            category=MetricCategory.AGENT
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agent_health_score",
            metric_type="gauge",
            help_text="Agent health score (0-1)",
            labels=["agent_id", "agent_type"],
            category=MetricCategory.AGENT
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agent_tasks_active",
            metric_type="gauge",
            help_text="Number of active tasks per agent",
            labels=["agent_id"],
            category=MetricCategory.AGENT
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agent_tasks_pending",
            metric_type="gauge",
            help_text="Number of pending tasks per agent",
            labels=["agent_id"],
            category=MetricCategory.AGENT
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agent_response_time_seconds",
            metric_type="histogram",
            help_text="Agent response time in seconds",
            labels=["agent_id"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            category=MetricCategory.AGENT
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agent_throughput_tasks_per_hour",
            metric_type="gauge",
            help_text="Agent throughput in tasks per hour",
            labels=["agent_id"],
            category=MetricCategory.AGENT
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}agent_error_rate_percent",
            metric_type="gauge",
            help_text="Agent error rate percentage",
            labels=["agent_id"],
            category=MetricCategory.AGENT
        ))
        
        # Task metrics
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}tasks_total",
            metric_type="counter",
            help_text="Total number of tasks",
            labels=["status", "priority", "agent_type"],
            category=MetricCategory.TASK
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}task_execution_time_seconds",
            metric_type="histogram",
            help_text="Task execution time in seconds",
            labels=["task_type", "priority"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            category=MetricCategory.TASK
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}task_queue_size",
            metric_type="gauge",
            help_text="Current task queue size",
            labels=["priority"],
            category=MetricCategory.TASK
        ))
        
        # Business metrics
        if self.config["enable_business_metrics"]:
            self.register_metric(PrometheusMetricDefinition(
                name=f"{self.config['metric_prefix']}business_active_sessions",
                metric_type="gauge",
                help_text="Number of active user sessions",
                category=MetricCategory.BUSINESS
            ))
            
            self.register_metric(PrometheusMetricDefinition(
                name=f"{self.config['metric_prefix']}business_requests_per_minute",
                metric_type="gauge",
                help_text="Business requests per minute",
                labels=["endpoint", "method"],
                category=MetricCategory.BUSINESS
            ))
            
            self.register_metric(PrometheusMetricDefinition(
                name=f"{self.config['metric_prefix']}business_success_rate_percent",
                metric_type="gauge",
                help_text="Business operation success rate",
                labels=["operation"],
                category=MetricCategory.BUSINESS
            ))
        
        # Performance metrics
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}performance_collection_duration_seconds",
            metric_type="histogram",
            help_text="Time spent collecting metrics",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            category=MetricCategory.PERFORMANCE
        ))
        
        self.register_metric(PrometheusMetricDefinition(
            name=f"{self.config['metric_prefix']}performance_export_duration_seconds",
            metric_type="histogram",
            help_text="Time spent exporting metrics",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            category=MetricCategory.PERFORMANCE
        ))
        
        # Security metrics (if enabled)
        if self.config["enable_security_metrics"]:
            self.register_metric(PrometheusMetricDefinition(
                name=f"{self.config['metric_prefix']}security_authentication_attempts_total",
                metric_type="counter",
                help_text="Total authentication attempts",
                labels=["result", "method"],
                category=MetricCategory.SECURITY
            ))
            
            self.register_metric(PrometheusMetricDefinition(
                name=f"{self.config['metric_prefix']}security_rate_limit_hits_total",
                metric_type="counter",
                help_text="Total rate limit hits",
                labels=["endpoint"],
                category=MetricCategory.SECURITY
            ))
    
    def register_metric(self, definition: PrometheusMetricDefinition) -> None:
        """Register a new Prometheus metric."""
        if definition.name in self.metric_definitions:
            logger.warning("Metric already registered", metric_name=definition.name)
            return
        
        try:
            # Create the Prometheus metric
            metric_obj = definition.create_metric(self.registry)
            
            # Store definition and metric
            self.metric_definitions[definition.name] = definition
            self.metrics[definition.name] = metric_obj
            
            logger.debug(
                "Metric registered successfully",
                metric_name=definition.name,
                metric_type=definition.metric_type,
                category=definition.category.value
            )
            
        except Exception as e:
            logger.error("Failed to register metric", metric_name=definition.name, error=str(e))
    
    async def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self.collection_active:
            logger.warning("Metrics collection already active")
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        # Start metrics collector if not already running
        if self.metrics_collector:
            await self.metrics_collector.start_collection()
        
        logger.info("Prometheus metrics collection started")
    
    async def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        logger.info("Prometheus metrics collection stopped")
    
    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.collection_active:
            try:
                start_time = time.time()
                
                # Collect all categories of metrics
                await self._collect_system_metrics()
                await self._collect_agent_metrics()
                await self._collect_task_metrics()
                
                if self.config["enable_business_metrics"]:
                    await self._collect_business_metrics()
                
                if self.config["enable_security_metrics"]:
                    await self._collect_security_metrics()
                
                await self._collect_performance_metrics()
                
                # Record collection performance
                collection_time = time.time() - start_time
                self.collection_latencies.append(collection_time)
                
                # Update performance histogram
                if f"{self.config['metric_prefix']}performance_collection_duration_seconds" in self.metrics:
                    self.metrics[f"{self.config['metric_prefix']}performance_collection_duration_seconds"].observe(collection_time)
                
                self.last_collection = datetime.utcnow()
                
                # Cleanup old data
                if len(self.collection_latencies) > 100:
                    self.collection_latencies = self.collection_latencies[-50:]
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            if not self.metrics_collector:
                return
            
            # Get system performance summary
            performance_data = await self.metrics_collector.get_performance_summary()
            system_metrics = performance_data.get("system_metrics", {})
            
            # Update Prometheus metrics
            if "system.cpu.percent" in system_metrics:
                self._update_metric(f"{self.config['metric_prefix']}system_cpu_percent", 
                                  system_metrics["system.cpu.percent"])
            
            if "system.memory.rss_mb" in system_metrics:
                memory_bytes = system_metrics["system.memory.rss_mb"] * 1024 * 1024
                self._update_metric(f"{self.config['metric_prefix']}system_memory_usage_bytes", memory_bytes)
            
            if "system.total.memory_percent" in system_metrics:
                self._update_metric(f"{self.config['metric_prefix']}system_memory_percent", 
                                  system_metrics["system.total.memory_percent"])
            
            # Disk I/O metrics (counters - increment with delta)
            if "system.disk.read_bytes" in system_metrics:
                self._increment_counter(f"{self.config['metric_prefix']}system_disk_io_bytes_total",
                                      system_metrics["system.disk.read_bytes"], {"direction": "read"})
            
            if "system.disk.write_bytes" in system_metrics:
                self._increment_counter(f"{self.config['metric_prefix']}system_disk_io_bytes_total",
                                      system_metrics["system.disk.write_bytes"], {"direction": "write"})
            
            # Network I/O metrics
            if "system.network.bytes_sent" in system_metrics:
                self._increment_counter(f"{self.config['metric_prefix']}system_network_bytes_total",
                                      system_metrics["system.network.bytes_sent"], {"direction": "sent"})
            
            if "system.network.bytes_recv" in system_metrics:
                self._increment_counter(f"{self.config['metric_prefix']}system_network_bytes_total",
                                      system_metrics["system.network.bytes_recv"], {"direction": "received"})
            
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
    
    async def _collect_agent_metrics(self) -> None:
        """Collect agent performance metrics."""
        try:
            async with self.session_factory() as session:
                # Get agent counts by status and type
                agent_query = select(
                    Agent.status, 
                    Agent.agent_type, 
                    func.count(Agent.id).label('count')
                ).group_by(Agent.status, Agent.agent_type)
                
                result = await session.execute(agent_query)
                agent_counts = result.all()
                
                # Update agent total metrics
                for row in agent_counts:
                    status = row.status.value if hasattr(row.status, 'value') else str(row.status)
                    agent_type = row.agent_type.value if hasattr(row.agent_type, 'value') else str(row.agent_type)
                    
                    self._update_metric(
                        f"{self.config['metric_prefix']}agents_total",
                        row.count,
                        {"status": status, "type": agent_type}
                    )
                
                # Get detailed agent metrics from performance collector
                if self.config["enable_detailed_agent_metrics"] and self.metrics_collector:
                    performance_data = await self.metrics_collector.get_performance_summary()
                    
                    # Get recent workload snapshots
                    snapshot_query = select(WorkloadSnapshot).where(
                        WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=15)
                    ).order_by(WorkloadSnapshot.snapshot_time.desc())
                    
                    snapshot_result = await session.execute(snapshot_query)
                    snapshots = snapshot_result.scalars().all()
                    
                    # Process each agent's metrics
                    agent_metrics = defaultdict(lambda: {"snapshots": [], "latest": None})
                    for snapshot in snapshots:
                        agent_id = str(snapshot.agent_id)
                        agent_metrics[agent_id]["snapshots"].append(snapshot)
                        if agent_metrics[agent_id]["latest"] is None or snapshot.snapshot_time > agent_metrics[agent_id]["latest"].snapshot_time:
                            agent_metrics[agent_id]["latest"] = snapshot
                    
                    # Update per-agent metrics
                    for agent_id, data in agent_metrics.items():
                        latest = data["latest"]
                        if not latest:
                            continue
                        
                        # Get agent type for labels
                        agent_type = "unknown"  # Default, could be enhanced with actual lookup
                        
                        # Health score
                        health_score = self._calculate_health_score(latest)
                        self._update_metric(
                            f"{self.config['metric_prefix']}agent_health_score",
                            health_score,
                            {"agent_id": agent_id, "agent_type": agent_type}
                        )
                        
                        # Task counts
                        self._update_metric(
                            f"{self.config['metric_prefix']}agent_tasks_active",
                            latest.active_tasks,
                            {"agent_id": agent_id}
                        )
                        
                        self._update_metric(
                            f"{self.config['metric_prefix']}agent_tasks_pending",
                            latest.pending_tasks,
                            {"agent_id": agent_id}
                        )
                        
                        # Response time
                        if latest.average_response_time_ms:
                            response_time_seconds = latest.average_response_time_ms / 1000.0
                            self._observe_histogram(
                                f"{self.config['metric_prefix']}agent_response_time_seconds",
                                response_time_seconds,
                                {"agent_id": agent_id}
                            )
                        
                        # Throughput
                        if latest.throughput_tasks_per_hour:
                            self._update_metric(
                                f"{self.config['metric_prefix']}agent_throughput_tasks_per_hour",
                                latest.throughput_tasks_per_hour,
                                {"agent_id": agent_id}
                            )
                        
                        # Error rate
                        self._update_metric(
                            f"{self.config['metric_prefix']}agent_error_rate_percent",
                            latest.error_rate_percent,
                            {"agent_id": agent_id}
                        )
            
        except Exception as e:
            logger.error("Error collecting agent metrics", error=str(e))
    
    def _calculate_health_score(self, snapshot: WorkloadSnapshot) -> float:
        """Calculate agent health score from snapshot data."""
        health = 1.0
        
        # Factor in error rate
        if snapshot.error_rate_percent > 0:
            health -= min(0.5, snapshot.error_rate_percent / 100.0)
        
        # Factor in load
        load_factor = snapshot.calculate_load_factor()
        if load_factor > 0.85:
            health -= (load_factor - 0.85) * 0.5
        
        # Factor in response time
        if snapshot.average_response_time_ms and snapshot.average_response_time_ms > 5000:
            health -= min(0.3, (snapshot.average_response_time_ms - 5000) / 10000)
        
        return max(0.0, min(1.0, health))
    
    async def _collect_task_metrics(self) -> None:
        """Collect task-related metrics."""
        try:
            async with self.session_factory() as session:
                # Task counts by status and priority
                task_query = select(
                    Task.status,
                    Task.priority,
                    Task.agent_type,
                    func.count(Task.id).label('count')
                ).group_by(Task.status, Task.priority, Task.agent_type)
                
                result = await session.execute(task_query)
                task_counts = result.all()
                
                # Update task total counters (increment since last collection)
                for row in task_counts:
                    status = row.status.value if hasattr(row.status, 'value') else str(row.status)
                    priority = row.priority.value if hasattr(row.priority, 'value') else str(row.priority)
                    agent_type = row.agent_type.value if hasattr(row.agent_type, 'value') else str(row.agent_type)
                    
                    # For total counters, we use the current count as the value
                    # Note: In a real implementation, you'd track increments
                    labels = {"status": status, "priority": priority, "agent_type": agent_type}
                    self._set_counter_value(
                        f"{self.config['metric_prefix']}tasks_total",
                        row.count,
                        labels
                    )
                
                # Task queue sizes by priority
                queue_query = select(
                    Task.priority,
                    func.count(Task.id).label('count')
                ).where(
                    Task.status.in_([TaskStatus.PENDING, TaskStatus.QUEUED])
                ).group_by(Task.priority)
                
                queue_result = await session.execute(queue_query)
                queue_counts = queue_result.all()
                
                for row in queue_counts:
                    priority = row.priority.value if hasattr(row.priority, 'value') else str(row.priority)
                    self._update_metric(
                        f"{self.config['metric_prefix']}task_queue_size",
                        row.count,
                        {"priority": priority}
                    )
                
                # Task execution times (from recent completed tasks)
                execution_query = select(Task).where(
                    and_(
                        Task.status == TaskStatus.COMPLETED,
                        Task.completed_at.isnot(None),
                        Task.created_at.isnot(None),
                        Task.completed_at >= datetime.utcnow() - timedelta(minutes=15)
                    )
                ).order_by(Task.completed_at.desc()).limit(100)
                
                execution_result = await session.execute(execution_query)
                recent_tasks = execution_result.scalars().all()
                
                for task in recent_tasks:
                    if task.completed_at and task.created_at:
                        execution_time = (task.completed_at - task.created_at).total_seconds()
                        priority = task.priority.value if hasattr(task.priority, 'value') else str(task.priority)
                        
                        self._observe_histogram(
                            f"{self.config['metric_prefix']}task_execution_time_seconds",
                            execution_time,
                            {"task_type": "generic", "priority": priority}
                        )
            
        except Exception as e:
            logger.error("Error collecting task metrics", error=str(e))
    
    async def _collect_business_metrics(self) -> None:
        """Collect business-related metrics."""
        try:
            # Get business metrics from Redis cache or calculate
            business_data = await self._get_cached_business_metrics()
            
            if "active_sessions" in business_data:
                self._update_metric(
                    f"{self.config['metric_prefix']}business_active_sessions",
                    business_data["active_sessions"]
                )
            
            if "requests_per_minute" in business_data:
                for endpoint_data in business_data["requests_per_minute"]:
                    self._update_metric(
                        f"{self.config['metric_prefix']}business_requests_per_minute",
                        endpoint_data["rate"],
                        {"endpoint": endpoint_data["endpoint"], "method": endpoint_data["method"]}
                    )
            
            if "success_rates" in business_data:
                for operation_data in business_data["success_rates"]:
                    self._update_metric(
                        f"{self.config['metric_prefix']}business_success_rate_percent",
                        operation_data["rate"],
                        {"operation": operation_data["operation"]}
                    )
            
        except Exception as e:
            logger.error("Error collecting business metrics", error=str(e))
    
    async def _collect_security_metrics(self) -> None:
        """Collect security-related metrics."""
        try:
            # Get security metrics from Redis or calculate
            security_data = await self._get_cached_security_metrics()
            
            if "auth_attempts" in security_data:
                for auth_data in security_data["auth_attempts"]:
                    self._increment_counter(
                        f"{self.config['metric_prefix']}security_authentication_attempts_total",
                        auth_data["count"],
                        {"result": auth_data["result"], "method": auth_data["method"]}
                    )
            
            if "rate_limit_hits" in security_data:
                for rate_data in security_data["rate_limit_hits"]:
                    self._increment_counter(
                        f"{self.config['metric_prefix']}security_rate_limit_hits_total",
                        rate_data["count"],
                        {"endpoint": rate_data["endpoint"]}
                    )
            
        except Exception as e:
            logger.error("Error collecting security metrics", error=str(e))
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance monitoring metrics."""
        try:
            # Collection latency statistics
            if self.collection_latencies:
                avg_latency = sum(self.collection_latencies) / len(self.collection_latencies)
                # This is recorded in the main collection loop
                
            # Export latency statistics
            if self.export_latencies:
                for latency in self.export_latencies[-10:]:  # Last 10 exports
                    self._observe_histogram(
                        f"{self.config['metric_prefix']}performance_export_duration_seconds",
                        latency
                    )
                
                # Clear processed latencies
                self.export_latencies = self.export_latencies[-10:]
            
        except Exception as e:
            logger.error("Error collecting performance metrics", error=str(e))
    
    async def _get_cached_business_metrics(self) -> Dict[str, Any]:
        """Get cached business metrics or calculate them."""
        cache_key = "business_metrics"
        
        with self.cache_lock:
            if cache_key in self.metric_cache:
                cache_time, data = self.metric_cache[cache_key]
                if time.time() - cache_time < self.config["cache_metrics_seconds"]:
                    return data
        
        # Calculate business metrics
        try:
            business_data = {}
            
            # Simulate business metrics (replace with actual business logic)
            business_data["active_sessions"] = 150  # Example value
            business_data["requests_per_minute"] = [
                {"endpoint": "/api/agents", "method": "GET", "rate": 45.2},
                {"endpoint": "/api/tasks", "method": "POST", "rate": 23.1},
                {"endpoint": "/api/health", "method": "GET", "rate": 120.0}
            ]
            business_data["success_rates"] = [
                {"operation": "task_execution", "rate": 98.5},
                {"operation": "agent_coordination", "rate": 99.2},
                {"operation": "user_authentication", "rate": 97.8}
            ]
            
            # Cache the results
            with self.cache_lock:
                self.metric_cache[cache_key] = (time.time(), business_data)
            
            return business_data
            
        except Exception as e:
            logger.error("Error calculating business metrics", error=str(e))
            return {}
    
    async def _get_cached_security_metrics(self) -> Dict[str, Any]:
        """Get cached security metrics or calculate them."""
        cache_key = "security_metrics"
        
        with self.cache_lock:
            if cache_key in self.metric_cache:
                cache_time, data = self.metric_cache[cache_key]
                if time.time() - cache_time < self.config["cache_metrics_seconds"]:
                    return data
        
        # Calculate security metrics
        try:
            security_data = {}
            
            # Simulate security metrics (replace with actual security monitoring)
            security_data["auth_attempts"] = [
                {"result": "success", "method": "jwt", "count": 1250},
                {"result": "failure", "method": "jwt", "count": 15},
                {"result": "success", "method": "oauth", "count": 340}
            ]
            security_data["rate_limit_hits"] = [
                {"endpoint": "/api/tasks", "count": 8},
                {"endpoint": "/api/agents", "count": 3}
            ]
            
            # Cache the results
            with self.cache_lock:
                self.metric_cache[cache_key] = (time.time(), security_data)
            
            return security_data
            
        except Exception as e:
            logger.error("Error calculating security metrics", error=str(e))
            return {}
    
    def _update_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Update a Prometheus metric value."""
        try:
            if metric_name not in self.metrics:
                logger.warning("Metric not found", metric_name=metric_name)
                return
            
            metric = self.metrics[metric_name]
            
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
                
        except Exception as e:
            logger.error("Error updating metric", metric_name=metric_name, error=str(e))
    
    def _increment_counter(self, metric_name: str, increment: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a Prometheus counter."""
        try:
            if metric_name not in self.metrics:
                logger.warning("Counter not found", metric_name=metric_name)
                return
            
            counter = self.metrics[metric_name]
            
            if labels:
                counter.labels(**labels).inc(increment)
            else:
                counter.inc(increment)
                
        except Exception as e:
            logger.error("Error incrementing counter", metric_name=metric_name, error=str(e))
    
    def _set_counter_value(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set counter to specific value (for gauges used as counters)."""
        # Note: This is a workaround for total counters where we have absolute values
        self._update_metric(metric_name, value, labels)
    
    def _observe_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a Prometheus histogram."""
        try:
            if metric_name not in self.metrics:
                logger.warning("Histogram not found", metric_name=metric_name)
                return
            
            histogram = self.metrics[metric_name]
            
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)
                
        except Exception as e:
            logger.error("Error observing histogram", metric_name=metric_name, error=str(e))
    
    async def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        try:
            start_time = time.time()
            
            # Generate Prometheus metrics output
            metrics_output = generate_latest(self.registry).decode('utf-8')
            
            # Record export performance
            export_time = time.time() - start_time
            self.export_latencies.append(export_time)
            
            return metrics_output
            
        except Exception as e:
            logger.error("Error exporting metrics", error=str(e))
            return ""
    
    def get_content_type(self) -> str:
        """Get the content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics state."""
        try:
            return {
                "total_metrics": len(self.metrics),
                "metrics_by_category": {
                    category.value: len([
                        def_name for def_name, definition in self.metric_definitions.items()
                        if definition.category == category
                    ])
                    for category in MetricCategory
                },
                "collection_active": self.collection_active,
                "last_collection": self.last_collection.isoformat() if self.last_collection else None,
                "collection_interval": self.collection_interval,
                "average_collection_latency_ms": (
                    sum(self.collection_latencies) / len(self.collection_latencies) * 1000
                    if self.collection_latencies else 0
                ),
                "cache_size": len(self.metric_cache),
                "config": self.config
            }
            
        except Exception as e:
            logger.error("Error getting metrics summary", error=str(e))
            return {"error": str(e)}


# Global instance
_prometheus_exporter: Optional[PrometheusMetricsExporter] = None


async def get_prometheus_exporter() -> PrometheusMetricsExporter:
    """Get singleton Prometheus metrics exporter instance."""
    global _prometheus_exporter
    
    if _prometheus_exporter is None:
        from .performance_metrics_collector import PerformanceMetricsCollector
        
        # Create metrics collector if needed
        metrics_collector = PerformanceMetricsCollector()
        
        _prometheus_exporter = PrometheusMetricsExporter(
            metrics_collector=metrics_collector
        )
        await _prometheus_exporter.start_collection()
    
    return _prometheus_exporter


async def cleanup_prometheus_exporter() -> None:
    """Cleanup Prometheus exporter resources."""
    global _prometheus_exporter
    
    if _prometheus_exporter:
        await _prometheus_exporter.stop_collection()
        _prometheus_exporter = None