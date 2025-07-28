"""
Advanced Performance Metrics Collector for LeanVibe Agent Hive 2.0

Provides comprehensive performance monitoring, metrics collection, and real-time
analytics for agent orchestration system with sub-second latency tracking.
"""

import asyncio
import time
import psutil
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, update, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.agent_performance import (
    AgentPerformanceHistory, WorkloadSnapshot, TaskRoutingDecision,
    AgentCapabilityScore, PerformanceCategory
)
from ..models.task import Task, TaskStatus, TaskPriority

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class MetricAggregation(Enum):
    """Metric aggregation methods."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MetricSeries:
    """Time series of metric values."""
    name: str
    metric_type: MetricType
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_value(self, value: Union[int, float], timestamp: Optional[datetime] = None) -> None:
        """Add value to the series."""
        timestamp = timestamp or datetime.utcnow()
        self.values.append((timestamp, value))
        self.last_updated = timestamp
    
    def get_recent_values(self, duration_seconds: int = 300) -> List[Tuple[datetime, Union[int, float]]]:
        """Get values from the last N seconds."""
        cutoff = datetime.utcnow() - timedelta(seconds=duration_seconds)
        return [(ts, val) for ts, val in self.values if ts >= cutoff]
    
    def calculate_aggregation(self, aggregation: MetricAggregation, duration_seconds: int = 300) -> Optional[float]:
        """Calculate aggregated value over time period."""
        recent_values = [val for _, val in self.get_recent_values(duration_seconds)]
        
        if not recent_values:
            return None
        
        if aggregation == MetricAggregation.SUM:
            return sum(recent_values)
        elif aggregation == MetricAggregation.AVERAGE:
            return statistics.mean(recent_values)
        elif aggregation == MetricAggregation.MIN:
            return min(recent_values)
        elif aggregation == MetricAggregation.MAX:
            return max(recent_values)
        elif aggregation == MetricAggregation.COUNT:
            return len(recent_values)
        elif aggregation == MetricAggregation.P50:
            return statistics.median(recent_values)
        elif aggregation == MetricAggregation.P95:
            return statistics.quantiles(recent_values, n=20)[18] if len(recent_values) >= 20 else max(recent_values)
        elif aggregation == MetricAggregation.P99:
            return statistics.quantiles(recent_values, n=100)[98] if len(recent_values) >= 100 else max(recent_values)
        else:
            return statistics.mean(recent_values)


@dataclass
class PerformanceProfile:
    """Performance profile for an agent or system component."""
    entity_id: str
    entity_type: str  # "agent", "system", "task"
    metrics: Dict[str, MetricSeries] = field(default_factory=dict)
    health_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_metric(self, name: str, value: Union[int, float], metric_type: MetricType, tags: Optional[Dict[str, str]] = None) -> None:
        """Update a metric in the profile."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(
                name=name,
                metric_type=metric_type,
                tags=tags or {}
            )
        
        self.metrics[name].add_value(value)
        self.last_updated = datetime.utcnow()
    
    def get_metric_summary(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for metric_name, series in self.metrics.items():
            recent_values = series.get_recent_values(duration_seconds)
            if recent_values:
                values = [val for _, val in recent_values]
                summary[metric_name] = {
                    "count": len(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else None,
                    "type": series.metric_type.value
                }
        
        return summary


class PerformanceMetricsCollector:
    """
    Advanced performance metrics collector for comprehensive system monitoring.
    
    Features:
    - Real-time metrics collection with sub-second precision
    - Automatic system resource monitoring
    - Agent performance profiling
    - Custom metric registration and tracking
    - Redis-based distributed metrics storage
    - Performance anomaly detection
    """
    
    def __init__(
        self,
        redis_client=None,
        session_factory: Optional[Callable] = None,
        collection_interval: float = 5.0  # seconds
    ):
        self.redis_client = redis_client
        self.session_factory = session_factory or get_session
        self.collection_interval = collection_interval
        
        # Metrics storage
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.global_metrics: Dict[str, MetricSeries] = {}
        
        # Collection state
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None
        self.last_collection = datetime.utcnow()
        
        # Performance tracking
        self.collection_latencies: deque = deque(maxlen=100)
        self.collection_errors: deque = deque(maxlen=50)
        
        # System monitoring
        self.system_process = psutil.Process()
        self.cpu_percent_history: deque = deque(maxlen=60)  # 5 minutes at 5s intervals
        self.memory_history: deque = deque(maxlen=60)
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")
        
        # Configuration
        self.config = {
            "max_profiles": 1000,
            "metric_retention_hours": 24,
            "redis_key_prefix": "metrics:",
            "batch_size": 50,
            "anomaly_detection_threshold": 2.0,  # Standard deviations
            "collection_timeout": 30.0,  # seconds
            "system_metrics_enabled": True,
            "agent_metrics_enabled": True,
            "custom_metrics_enabled": True
        }
        
        logger.info("PerformanceMetricsCollector initialized",
                   collection_interval=collection_interval,
                   config=self.config)
    
    async def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self.collection_active:
            logger.warning("Metrics collection already active")
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collection started")
    
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
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.collection_active:
            try:
                start_time = time.time()
                
                # Collect system metrics
                if self.config["system_metrics_enabled"]:
                    await self._collect_system_metrics()
                
                # Collect agent metrics
                if self.config["agent_metrics_enabled"]:
                    await self._collect_agent_metrics()
                
                # Store metrics to Redis
                await self._store_metrics_to_redis()
                
                # Persist critical metrics to database
                await self._persist_metrics_to_database()
                
                # Detect anomalies
                await self._detect_performance_anomalies()
                
                # Record collection performance
                collection_time = time.time() - start_time
                self.collection_latencies.append(collection_time)
                self.last_collection = datetime.utcnow()
                
                if collection_time > 1.0:  # Log slow collections
                    logger.warning("Slow metrics collection",
                                 collection_time_ms=collection_time * 1000)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                self.collection_errors.append({
                    "timestamp": datetime.utcnow(),
                    "error": str(e)
                })
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide performance metrics."""
        try:
            # CPU metrics
            cpu_percent = self.system_process.cpu_percent()
            self.cpu_percent_history.append(cpu_percent)
            self._update_global_metric("system.cpu.percent", cpu_percent, MetricType.GAUGE)
            
            # Memory metrics
            memory_info = self.system_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_history.append(memory_mb)
            self._update_global_metric("system.memory.rss_mb", memory_mb, MetricType.GAUGE)
            self._update_global_metric("system.memory.vms_mb", memory_info.vms / 1024 / 1024, MetricType.GAUGE)
            
            # System-wide CPU and memory
            system_cpu = await asyncio.get_event_loop().run_in_executor(
                self.executor, psutil.cpu_percent, 0.1
            )
            system_memory = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: psutil.virtual_memory().percent
            )
            
            self._update_global_metric("system.total.cpu_percent", system_cpu, MetricType.GAUGE)
            self._update_global_metric("system.total.memory_percent", system_memory, MetricType.GAUGE)
            
            # Disk I/O
            disk_io = await asyncio.get_event_loop().run_in_executor(
                self.executor, psutil.disk_io_counters
            )
            if disk_io:
                self._update_global_metric("system.disk.read_bytes", disk_io.read_bytes, MetricType.COUNTER)
                self._update_global_metric("system.disk.write_bytes", disk_io.write_bytes, MetricType.COUNTER)
            
            # Network I/O
            net_io = await asyncio.get_event_loop().run_in_executor(
                self.executor, psutil.net_io_counters
            )
            if net_io:
                self._update_global_metric("system.network.bytes_sent", net_io.bytes_sent, MetricType.COUNTER)
                self._update_global_metric("system.network.bytes_recv", net_io.bytes_recv, MetricType.COUNTER)
            
            # Process counts
            process_count = len(psutil.pids())
            self._update_global_metric("system.processes.count", process_count, MetricType.GAUGE)
            
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
    
    async def _collect_agent_metrics(self) -> None:
        """Collect agent-specific performance metrics."""
        try:
            async with self.session_factory() as session:
                # Get recent workload snapshots
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=10)
                ).order_by(WorkloadSnapshot.snapshot_time.desc())
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                # Process each snapshot
                for snapshot in snapshots:
                    agent_id = str(snapshot.agent_id)
                    
                    # Ensure profile exists
                    if agent_id not in self.profiles:
                        self.profiles[agent_id] = PerformanceProfile(
                            entity_id=agent_id,
                            entity_type="agent"
                        )
                    
                    profile = self.profiles[agent_id]
                    
                    # Update metrics
                    profile.update_metric("tasks.active", snapshot.active_tasks, MetricType.GAUGE)
                    profile.update_metric("tasks.pending", snapshot.pending_tasks, MetricType.GAUGE)
                    profile.update_metric("context.usage_percent", snapshot.context_usage_percent, MetricType.GAUGE)
                    
                    if snapshot.memory_usage_mb:
                        profile.update_metric("memory.usage_mb", snapshot.memory_usage_mb, MetricType.GAUGE)
                    
                    if snapshot.cpu_usage_percent:
                        profile.update_metric("cpu.usage_percent", snapshot.cpu_usage_percent, MetricType.GAUGE)
                    
                    if snapshot.average_response_time_ms:
                        profile.update_metric("response_time.avg_ms", snapshot.average_response_time_ms, MetricType.GAUGE)
                    
                    if snapshot.throughput_tasks_per_hour:
                        profile.update_metric("throughput.tasks_per_hour", snapshot.throughput_tasks_per_hour, MetricType.GAUGE)
                    
                    profile.update_metric("error_rate.percent", snapshot.error_rate_percent, MetricType.GAUGE)
                    profile.update_metric("utilization.ratio", snapshot.utilization_ratio, MetricType.GAUGE)
                    
                    # Calculate and update health score
                    health_score = self._calculate_agent_health_score(snapshot)
                    profile.health_score = health_score
                    profile.update_metric("health.score", health_score, MetricType.GAUGE)
                
                # Update global agent metrics
                total_agents = len(self.profiles)
                active_agents = len([p for p in self.profiles.values() 
                                   if (datetime.utcnow() - p.last_updated).seconds < 300])
                
                self._update_global_metric("agents.total", total_agents, MetricType.GAUGE)
                self._update_global_metric("agents.active", active_agents, MetricType.GAUGE)
                
                # Average load factor across all agents
                if snapshots:
                    avg_load = statistics.mean([s.calculate_load_factor() for s in snapshots])
                    self._update_global_metric("agents.avg_load_factor", avg_load, MetricType.GAUGE)
                
        except Exception as e:
            logger.error("Error collecting agent metrics", error=str(e))
    
    def _calculate_agent_health_score(self, snapshot: WorkloadSnapshot) -> float:
        """Calculate health score for agent based on metrics."""
        # Base health score
        health = 1.0
        
        # Penalize high error rates
        if snapshot.error_rate_percent > 0:
            health -= min(0.5, snapshot.error_rate_percent / 100.0)
        
        # Penalize overutilization
        load_factor = snapshot.calculate_load_factor()
        if load_factor > 0.85:
            health -= (load_factor - 0.85) * 0.5
        
        # Penalize slow response times
        if snapshot.average_response_time_ms and snapshot.average_response_time_ms > 5000:  # 5 seconds
            health -= min(0.3, (snapshot.average_response_time_ms - 5000) / 10000)
        
        # Reward high throughput
        if snapshot.throughput_tasks_per_hour and snapshot.throughput_tasks_per_hour > 10:
            health += min(0.1, (snapshot.throughput_tasks_per_hour - 10) / 100)
        
        return max(0.0, min(1.0, health))
    
    def _update_global_metric(self, name: str, value: Union[int, float], metric_type: MetricType) -> None:
        """Update a global system metric."""
        if name not in self.global_metrics:
            self.global_metrics[name] = MetricSeries(name=name, metric_type=metric_type)
        
        self.global_metrics[name].add_value(value)
    
    async def _store_metrics_to_redis(self) -> None:
        """Store current metrics to Redis for real-time access."""
        if not self.redis_client:
            return
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Store global metrics
            global_key = f"{self.config['redis_key_prefix']}global"
            global_data = {
                name: {
                    "latest": series.values[-1][1] if series.values else 0,
                    "timestamp": series.last_updated.isoformat(),
                    "type": series.metric_type.value
                }
                for name, series in self.global_metrics.items()
            }
            
            pipe.hset(global_key, mapping={k: json.dumps(v) for k, v in global_data.items()})
            pipe.expire(global_key, 3600)  # 1 hour TTL
            
            # Store agent profiles
            for agent_id, profile in self.profiles.items():
                profile_key = f"{self.config['redis_key_prefix']}agent:{agent_id}"
                profile_data = {
                    "health_score": profile.health_score,
                    "last_updated": profile.last_updated.isoformat(),
                    "metrics": profile.get_metric_summary(300)  # 5-minute summary
                }
                
                pipe.hset(profile_key, mapping={
                    "data": json.dumps(profile_data)
                })
                pipe.expire(profile_key, 1800)  # 30 minutes TTL
            
            await pipe.execute()
            
        except Exception as e:
            logger.error("Error storing metrics to Redis", error=str(e))
    
    async def _persist_metrics_to_database(self) -> None:
        """Persist critical metrics to database for long-term storage."""
        try:
            # Only persist every 5th collection cycle to reduce DB load
            if len(self.collection_latencies) % 5 != 0:
                return
            
            async with self.session_factory() as session:
                # Create workload snapshots for current agent states
                for agent_id, profile in self.profiles.items():
                    if (datetime.utcnow() - profile.last_updated).seconds > 300:
                        continue  # Skip stale profiles
                    
                    # Extract current metric values
                    metrics = profile.get_metric_summary(60)  # 1-minute summary
                    
                    snapshot = WorkloadSnapshot(
                        agent_id=uuid.UUID(agent_id) if self._is_valid_uuid(agent_id) else uuid.uuid4(),
                        active_tasks=int(metrics.get("tasks.active", {}).get("latest", 0)),
                        pending_tasks=int(metrics.get("tasks.pending", {}).get("latest", 0)),
                        context_usage_percent=float(metrics.get("context.usage_percent", {}).get("latest", 0)),
                        memory_usage_mb=metrics.get("memory.usage_mb", {}).get("latest"),
                        cpu_usage_percent=metrics.get("cpu.usage_percent", {}).get("latest"),
                        estimated_capacity=1.0,
                        utilization_ratio=float(metrics.get("utilization.ratio", {}).get("latest", 0)),
                        average_response_time_ms=metrics.get("response_time.avg_ms", {}).get("latest"),
                        throughput_tasks_per_hour=metrics.get("throughput.tasks_per_hour", {}).get("latest"),
                        error_rate_percent=float(metrics.get("error_rate.percent", {}).get("latest", 0)),
                        snapshot_time=datetime.utcnow()
                    )
                    
                    session.add(snapshot)
                
                await session.commit()
                
        except Exception as e:
            logger.error("Error persisting metrics to database", error=str(e))
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    async def _detect_performance_anomalies(self) -> None:
        """Detect performance anomalies and alert."""
        try:
            threshold = self.config["anomaly_detection_threshold"]
            
            # Check global metrics for anomalies
            for metric_name, series in self.global_metrics.items():
                recent_values = [val for _, val in series.get_recent_values(300)]
                if len(recent_values) < 10:  # Need sufficient data
                    continue
                
                mean_val = statistics.mean(recent_values)
                std_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
                latest_val = recent_values[-1]
                
                # Check for anomaly
                if std_val > 0 and abs(latest_val - mean_val) > threshold * std_val:
                    await self._record_anomaly({
                        "type": "global_metric_anomaly",
                        "metric": metric_name,
                        "latest_value": latest_val,
                        "mean_value": mean_val,
                        "std_deviation": std_val,
                        "z_score": (latest_val - mean_val) / std_val,
                        "severity": "high" if abs(latest_val - mean_val) > 3 * std_val else "medium"
                    })
            
            # Check agent health scores
            unhealthy_agents = [
                agent_id for agent_id, profile in self.profiles.items()
                if profile.health_score < 0.5
            ]
            
            if unhealthy_agents:
                await self._record_anomaly({
                    "type": "unhealthy_agents",
                    "agent_count": len(unhealthy_agents),
                    "agent_ids": unhealthy_agents[:5],  # Limit to first 5
                    "severity": "high" if len(unhealthy_agents) > 3 else "medium"
                })
        
        except Exception as e:
            logger.error("Error detecting performance anomalies", error=str(e))
    
    async def _record_anomaly(self, anomaly_data: Dict[str, Any]) -> None:
        """Record a detected performance anomaly."""
        anomaly_data["timestamp"] = datetime.utcnow().isoformat()
        
        logger.warning("Performance anomaly detected",
                      anomaly_type=anomaly_data["type"],
                      severity=anomaly_data.get("severity", "medium"),
                      details=anomaly_data)
        
        # Store to Redis for real-time alerts
        if self.redis_client:
            try:
                anomaly_key = f"{self.config['redis_key_prefix']}anomalies"
                await self.redis_client.lpush(anomaly_key, json.dumps(anomaly_data))
                await self.redis_client.ltrim(anomaly_key, 0, 99)  # Keep last 100
                await self.redis_client.expire(anomaly_key, 86400)  # 24 hours TTL
            except Exception as e:
                logger.error("Error storing anomaly to Redis", error=str(e))
    
    async def record_custom_metric(
        self,
        entity_id: str,
        metric_name: str,
        value: Union[int, float],
        metric_type: MetricType,
        entity_type: str = "custom",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a custom metric for any entity."""
        if not self.config["custom_metrics_enabled"]:
            return
        
        # Ensure profile exists
        if entity_id not in self.profiles:
            self.profiles[entity_id] = PerformanceProfile(
                entity_id=entity_id,
                entity_type=entity_type
            )
        
        profile = self.profiles[entity_id]
        profile.update_metric(metric_name, value, metric_type, tags)
        
        logger.debug("Custom metric recorded",
                    entity_id=entity_id,
                    metric_name=metric_name,
                    value=value,
                    type=metric_type.value)
    
    async def get_performance_summary(
        self,
        entity_id: Optional[str] = None,
        duration_seconds: int = 300
    ) -> Dict[str, Any]:
        """Get performance summary for entity or entire system."""
        try:
            if entity_id:
                # Return specific entity metrics
                if entity_id in self.profiles:
                    profile = self.profiles[entity_id]
                    return {
                        "entity_id": entity_id,
                        "entity_type": profile.entity_type,
                        "health_score": profile.health_score,
                        "last_updated": profile.last_updated.isoformat(),
                        "metrics": profile.get_metric_summary(duration_seconds)
                    }
                else:
                    return {"error": f"Entity {entity_id} not found"}
            
            else:
                # Return system-wide summary
                global_summary = {}
                for name, series in self.global_metrics.items():
                    agg_value = series.calculate_aggregation(MetricAggregation.AVERAGE, duration_seconds)
                    if agg_value is not None:
                        global_summary[name] = agg_value
                
                agent_health_scores = [p.health_score for p in self.profiles.values() 
                                     if p.entity_type == "agent"]
                
                return {
                    "system_metrics": global_summary,
                    "agent_summary": {
                        "total_agents": len([p for p in self.profiles.values() if p.entity_type == "agent"]),
                        "avg_health_score": statistics.mean(agent_health_scores) if agent_health_scores else 0,
                        "unhealthy_agents": len([h for h in agent_health_scores if h < 0.7])
                    },
                    "collection_performance": {
                        "avg_latency_ms": statistics.mean(self.collection_latencies) * 1000 if self.collection_latencies else 0,
                        "error_count_last_hour": len([
                            e for e in self.collection_errors
                            if (datetime.utcnow() - e["timestamp"]).seconds < 3600
                        ]),
                        "last_collection": self.last_collection.isoformat()
                    }
                }
        
        except Exception as e:
            logger.error("Error getting performance summary", error=str(e))
            return {"error": str(e)}
    
    async def get_agent_performance_trends(
        self,
        agent_id: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get performance trends for specific agent."""
        try:
            async with self.session_factory() as session:
                # Get historical data
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id == uuid.UUID(agent_id) if self._is_valid_uuid(agent_id) else None,
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(hours=hours)
                ).order_by(WorkloadSnapshot.snapshot_time.asc())
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                if not snapshots:
                    return {"error": "No data found for agent"}
                
                # Calculate trends
                timestamps = [s.snapshot_time for s in snapshots]
                load_factors = [s.calculate_load_factor() for s in snapshots]
                memory_usage = [s.memory_usage_mb or 0 for s in snapshots]
                response_times = [s.average_response_time_ms or 0 for s in snapshots if s.average_response_time_ms]
                error_rates = [s.error_rate_percent for s in snapshots]
                
                return {
                    "agent_id": agent_id,
                    "time_range_hours": hours,
                    "data_points": len(snapshots),
                    "trends": {
                        "load_factor": {
                            "current": load_factors[-1] if load_factors else 0,
                            "average": statistics.mean(load_factors) if load_factors else 0,
                            "min": min(load_factors) if load_factors else 0,
                            "max": max(load_factors) if load_factors else 0,
                            "trend": "increasing" if len(load_factors) > 1 and load_factors[-1] > load_factors[0] else "stable"
                        },
                        "memory_usage_mb": {
                            "current": memory_usage[-1] if memory_usage else 0,
                            "average": statistics.mean(memory_usage) if memory_usage else 0,
                            "peak": max(memory_usage) if memory_usage else 0
                        },
                        "response_time_ms": {
                            "current": response_times[-1] if response_times else 0,
                            "average": statistics.mean(response_times) if response_times else 0,
                            "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else (max(response_times) if response_times else 0)
                        },
                        "error_rate_percent": {
                            "current": error_rates[-1] if error_rates else 0,
                            "average": statistics.mean(error_rates) if error_rates else 0,
                            "max": max(error_rates) if error_rates else 0
                        }
                    },
                    "health_assessment": "healthy" if (
                        (load_factors[-1] if load_factors else 0) < 0.8 and
                        (error_rates[-1] if error_rates else 0) < 5.0
                    ) else "needs_attention"
                }
        
        except Exception as e:
            logger.error("Error getting agent performance trends", error=str(e))
            return {"error": str(e)}
    
    async def cleanup_old_metrics(self) -> Dict[str, int]:
        """Cleanup old metrics to prevent memory leaks."""
        try:
            cleaned_profiles = 0
            cleaned_global_metrics = 0
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config["metric_retention_hours"])
            
            # Clean old profiles
            profiles_to_remove = [
                entity_id for entity_id, profile in self.profiles.items()
                if profile.last_updated < cutoff_time
            ]
            
            for entity_id in profiles_to_remove:
                del self.profiles[entity_id]
                cleaned_profiles += 1
            
            # Clean old values from global metrics
            for series in self.global_metrics.values():
                old_count = len(series.values)
                series.values = deque(
                    [(ts, val) for ts, val in series.values if ts >= cutoff_time],
                    maxlen=series.values.maxlen
                )
                cleaned_global_metrics += old_count - len(series.values)
            
            logger.info("Metrics cleanup completed",
                       cleaned_profiles=cleaned_profiles,
                       cleaned_global_metrics=cleaned_global_metrics)
            
            return {
                "cleaned_profiles": cleaned_profiles,
                "cleaned_global_metrics": cleaned_global_metrics
            }
        
        except Exception as e:
            logger.error("Error during metrics cleanup", error=str(e))
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup resources when collector is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)