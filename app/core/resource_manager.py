"""
Unified Resource Manager for LeanVibe Agent Hive 2.0

Consolidates 41 performance and resource-related files into a comprehensive resource management system:
- Performance monitoring and optimization
- Resource allocation and capacity management
- Load balancing and prediction
- Health monitoring and observability
- Metrics collection and analysis
- System performance validation
"""

import asyncio
import uuid
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import structlog
from sqlalchemy import select, and_, or_, desc, func

from .unified_manager_base import UnifiedManagerBase, ManagerConfig, PluginInterface, PluginType
from .database import get_async_session
from .redis import get_redis

logger = structlog.get_logger()


class ResourceType(str, Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    AGENT_SLOTS = "agent_slots"
    TASK_QUEUE = "task_queue"
    DATABASE = "database"
    REDIS = "redis"


class PerformanceLevel(str, Enum):
    """Performance levels for optimization."""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ResourceAllocationStrategy(str, Enum):
    """Resource allocation strategies."""
    FAIR_SHARE = "fair_share"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


class MetricType(str, Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class ResourceMetrics:
    """Real-time resource usage metrics."""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_usage_bytes: int = 0
    disk_usage_percent: float = 0.0
    disk_io_read_bytes: int = 0
    disk_io_write_bytes: int = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_agents: int = 0
    pending_tasks: int = 0
    database_connections: int = 0
    redis_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent or task."""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    max_execution_time_seconds: int = 300
    priority: int = 50  # 0-100 scale
    reserved: bool = False
    allocation_id: uuid.UUID = field(default_factory=uuid.uuid4)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    operation_name: str
    execution_time_ms: float
    success: bool
    memory_used_mb: float = 0.0
    cpu_used_percent: float = 0.0
    throughput_ops_per_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadPrediction:
    """Load prediction for resource planning."""
    resource_type: ResourceType
    predicted_usage_percent: float
    confidence_score: float
    prediction_horizon_minutes: int
    factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceOptimizer:
    """Advanced performance optimization engine."""
    
    def __init__(self):
        self.optimization_rules = {
            "cpu_high": self._optimize_cpu_usage,
            "memory_high": self._optimize_memory_usage,
            "disk_high": self._optimize_disk_usage,
            "network_high": self._optimize_network_usage,
            "agent_overload": self._optimize_agent_load,
            "task_backlog": self._optimize_task_processing
        }
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def analyze_and_optimize(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Analyze metrics and suggest optimizations."""
        optimizations = []
        
        # CPU optimization
        if metrics.cpu_usage_percent > 80:
            cpu_opt = await self._optimize_cpu_usage(metrics)
            if cpu_opt:
                optimizations.append(cpu_opt)
        
        # Memory optimization
        if metrics.memory_usage_percent > 85:
            mem_opt = await self._optimize_memory_usage(metrics)
            if mem_opt:
                optimizations.append(mem_opt)
        
        # Agent load optimization
        if metrics.active_agents > 50:  # Threshold for agent count
            agent_opt = await self._optimize_agent_load(metrics)
            if agent_opt:
                optimizations.append(agent_opt)
        
        # Task queue optimization
        if metrics.pending_tasks > 100:  # Threshold for pending tasks
            task_opt = await self._optimize_task_processing(metrics)
            if task_opt:
                optimizations.append(task_opt)
        
        # Store optimization history
        for opt in optimizations:
            self.optimization_history.append({
                "timestamp": datetime.utcnow(),
                "optimization": opt,
                "metrics_snapshot": metrics
            })
        
        # Keep only recent history
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-500:]
        
        return optimizations
    
    async def _optimize_cpu_usage(self, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Optimize CPU usage."""
        return {
            "type": "cpu_optimization",
            "severity": "high" if metrics.cpu_usage_percent > 90 else "medium",
            "recommendations": [
                "Reduce concurrent agent spawning",
                "Implement CPU throttling for non-critical tasks",
                "Consider horizontal scaling",
                "Optimize task scheduling algorithms"
            ],
            "expected_improvement": "15-25% CPU reduction",
            "priority": 90 if metrics.cpu_usage_percent > 90 else 70
        }
    
    async def _optimize_memory_usage(self, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Optimize memory usage."""
        return {
            "type": "memory_optimization",
            "severity": "critical" if metrics.memory_usage_percent > 95 else "high",
            "recommendations": [
                "Trigger context compression",
                "Clear unused cache entries",
                "Implement memory-efficient data structures",
                "Garbage collect idle agent memory"
            ],
            "expected_improvement": "20-30% memory reduction",
            "priority": 95 if metrics.memory_usage_percent > 95 else 80
        }
    
    async def _optimize_disk_usage(self, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Optimize disk usage."""
        if metrics.disk_usage_percent > 80:
            return {
                "type": "disk_optimization",
                "severity": "medium",
                "recommendations": [
                    "Archive old logs and contexts",
                    "Compress large database tables",
                    "Clean temporary files",
                    "Implement log rotation"
                ],
                "expected_improvement": "10-20% disk space recovery",
                "priority": 60
            }
        return None
    
    async def _optimize_network_usage(self, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Optimize network usage."""
        # Network optimization based on traffic patterns
        return {
            "type": "network_optimization",
            "severity": "low",
            "recommendations": [
                "Implement message compression",
                "Batch small network requests",
                "Use connection pooling",
                "Optimize WebSocket frame sizes"
            ],
            "expected_improvement": "10-15% network efficiency",
            "priority": 40
        }
    
    async def _optimize_agent_load(self, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Optimize agent load distribution."""
        return {
            "type": "agent_load_optimization",
            "severity": "medium",
            "recommendations": [
                "Implement intelligent load balancing",
                "Scale agent pool horizontally",
                "Optimize task distribution algorithms",
                "Implement agent hibernation for idle agents"
            ],
            "expected_improvement": "20-30% load distribution improvement",
            "priority": 75
        }
    
    async def _optimize_task_processing(self, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Optimize task processing."""
        return {
            "type": "task_processing_optimization",
            "severity": "high",
            "recommendations": [
                "Increase task processing parallelism",
                "Implement task prioritization",
                "Optimize task queue algorithms",
                "Consider task batching"
            ],
            "expected_improvement": "30-50% task throughput increase",
            "priority": 85
        }


class LoadBalancer:
    """Intelligent load balancing and prediction system."""
    
    def __init__(self):
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.prediction_models: Dict[ResourceType, Any] = {}
    
    def record_load(self, resource_type: ResourceType, load_value: float) -> None:
        """Record load measurement for a resource."""
        self.load_history[resource_type.value].append({
            "timestamp": datetime.utcnow(),
            "load": load_value
        })
    
    def get_current_load(self, resource_type: ResourceType) -> float:
        """Get current load for a resource type."""
        history = self.load_history[resource_type.value]
        if not history:
            return 0.0
        return history[-1]["load"]
    
    def predict_load(
        self, 
        resource_type: ResourceType, 
        horizon_minutes: int = 15
    ) -> LoadPrediction:
        """Predict future load based on historical data."""
        history = self.load_history[resource_type.value]
        
        if len(history) < 10:
            # Not enough data for prediction
            current_load = self.get_current_load(resource_type)
            return LoadPrediction(
                resource_type=resource_type,
                predicted_usage_percent=current_load,
                confidence_score=0.3,
                prediction_horizon_minutes=horizon_minutes,
                factors=["insufficient_data"]
            )
        
        # Simple trend-based prediction
        recent_loads = [entry["load"] for entry in list(history)[-10:]]
        trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        
        # Project trend forward
        predicted_load = recent_loads[-1] + (trend * horizon_minutes)
        predicted_load = max(0.0, min(100.0, predicted_load))  # Clamp to 0-100%
        
        # Calculate confidence based on trend consistency
        load_variance = sum((load - sum(recent_loads)/len(recent_loads))**2 for load in recent_loads) / len(recent_loads)
        confidence = max(0.1, 1.0 - (load_variance / 100.0))
        
        return LoadPrediction(
            resource_type=resource_type,
            predicted_usage_percent=predicted_load,
            confidence_score=confidence,
            prediction_horizon_minutes=horizon_minutes,
            factors=["trend_analysis", "historical_data"]
        )
    
    def get_optimal_allocation(
        self, 
        request: ResourceAllocation,
        strategy: ResourceAllocationStrategy = ResourceAllocationStrategy.LOAD_BALANCED
    ) -> ResourceAllocation:
        """Get optimal resource allocation based on current load."""
        if strategy == ResourceAllocationStrategy.LOAD_BALANCED:
            # Adjust based on current system load
            cpu_load = self.get_current_load(ResourceType.CPU)
            memory_load = self.get_current_load(ResourceType.MEMORY)
            
            # Scale down allocation if system is under pressure
            cpu_factor = 1.0 if cpu_load < 70 else 0.8 if cpu_load < 85 else 0.6
            memory_factor = 1.0 if memory_load < 70 else 0.8 if memory_load < 85 else 0.6
            
            optimized_allocation = ResourceAllocation(
                cpu_cores=request.cpu_cores * cpu_factor,
                memory_mb=int(request.memory_mb * memory_factor),
                max_execution_time_seconds=request.max_execution_time_seconds,
                priority=request.priority,
                reserved=request.reserved
            )
            
            return optimized_allocation
        
        return request  # Return original for other strategies


class CapacityManager:
    """System capacity planning and management."""
    
    def __init__(self):
        self.capacity_limits = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.AGENT_SLOTS: 100,
            ResourceType.TASK_QUEUE: 1000
        }
        self.current_allocations: Dict[ResourceType, float] = defaultdict(float)
        self.allocation_history: List[Dict[str, Any]] = []
    
    def allocate_resources(self, allocation: ResourceAllocation) -> bool:
        """Allocate resources if available."""
        # Check if resources are available
        cpu_needed = (allocation.cpu_cores / psutil.cpu_count()) * 100
        memory_needed = (allocation.memory_mb / (psutil.virtual_memory().total / 1024 / 1024)) * 100
        
        if (self.current_allocations[ResourceType.CPU] + cpu_needed > 90 or
            self.current_allocations[ResourceType.MEMORY] + memory_needed > 90):
            return False
        
        # Allocate resources
        self.current_allocations[ResourceType.CPU] += cpu_needed
        self.current_allocations[ResourceType.MEMORY] += memory_needed
        
        # Record allocation
        self.allocation_history.append({
            "timestamp": datetime.utcnow(),
            "allocation_id": allocation.allocation_id,
            "cpu_allocated": cpu_needed,
            "memory_allocated": memory_needed,
            "action": "allocate"
        })
        
        return True
    
    def deallocate_resources(self, allocation: ResourceAllocation) -> None:
        """Deallocate resources."""
        cpu_to_free = (allocation.cpu_cores / psutil.cpu_count()) * 100
        memory_to_free = (allocation.memory_mb / (psutil.virtual_memory().total / 1024 / 1024)) * 100
        
        self.current_allocations[ResourceType.CPU] = max(0, self.current_allocations[ResourceType.CPU] - cpu_to_free)
        self.current_allocations[ResourceType.MEMORY] = max(0, self.current_allocations[ResourceType.MEMORY] - memory_to_free)
        
        # Record deallocation
        self.allocation_history.append({
            "timestamp": datetime.utcnow(),
            "allocation_id": allocation.allocation_id,
            "cpu_freed": cpu_to_free,
            "memory_freed": memory_to_free,
            "action": "deallocate"
        })
    
    def get_available_capacity(self) -> Dict[ResourceType, float]:
        """Get available capacity for each resource type."""
        return {
            ResourceType.CPU: max(0, 90 - self.current_allocations[ResourceType.CPU]),
            ResourceType.MEMORY: max(0, 90 - self.current_allocations[ResourceType.MEMORY]),
            ResourceType.AGENT_SLOTS: max(0, self.capacity_limits[ResourceType.AGENT_SLOTS] - self.current_allocations[ResourceType.AGENT_SLOTS]),
            ResourceType.TASK_QUEUE: max(0, self.capacity_limits[ResourceType.TASK_QUEUE] - self.current_allocations[ResourceType.TASK_QUEUE])
        }


class MetricsCollector:
    """Advanced metrics collection and analysis."""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics: Dict[str, Any] = {}
        self.benchmark_results: List[PerformanceBenchmark] = []
    
    async def collect_system_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_bytes = memory.used
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
            
            # Network metrics
            network = psutil.net_io_counters()
            net_sent = network.bytes_sent if network else 0
            net_recv = network.bytes_recv if network else 0
            
            # Application-specific metrics (would be injected from other managers)
            active_agents = self.custom_metrics.get("active_agents", 0)
            pending_tasks = self.custom_metrics.get("pending_tasks", 0)
            db_connections = self.custom_metrics.get("database_connections", 0)
            redis_connections = self.custom_metrics.get("redis_connections", 0)
            
            metrics = ResourceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                memory_usage_bytes=memory_bytes,
                disk_usage_percent=disk_percent,
                disk_io_read_bytes=disk_read,
                disk_io_write_bytes=disk_write,
                network_bytes_sent=net_sent,
                network_bytes_recv=net_recv,
                active_agents=active_agents,
                pending_tasks=pending_tasks,
                database_connections=db_connections,
                redis_connections=redis_connections,
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self.metrics_history["system_metrics"].append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return ResourceMetrics()  # Return empty metrics on error
    
    def record_custom_metric(self, name: str, value: Any, metric_type: MetricType = MetricType.GAUGE) -> None:
        """Record a custom metric."""
        self.custom_metrics[name] = {
            "value": value,
            "type": metric_type,
            "timestamp": datetime.utcnow()
        }
        
        # Also store in history
        self.metrics_history[name].append({
            "value": value,
            "timestamp": datetime.utcnow()
        })
    
    async def benchmark_operation(
        self, 
        operation_name: str, 
        operation_func, 
        *args, 
        **kwargs
    ) -> PerformanceBenchmark:
        """Benchmark an operation's performance."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.Process().cpu_percent()
        
        success = True
        error_message = None
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
                
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time_ms = (end_time - start_time) * 1000
        memory_used_mb = end_memory - start_memory
        
        benchmark = PerformanceBenchmark(
            operation_name=operation_name,
            execution_time_ms=execution_time_ms,
            success=success,
            memory_used_mb=memory_used_mb,
            cpu_used_percent=0.0,  # Would need more sophisticated CPU measurement
            throughput_ops_per_sec=1000 / execution_time_ms if execution_time_ms > 0 else 0,
            metadata={"error": error_message} if error_message else {}
        )
        
        self.benchmark_results.append(benchmark)
        
        # Keep only recent benchmarks
        if len(self.benchmark_results) > 1000:
            self.benchmark_results = self.benchmark_results[-500:]
        
        return benchmark
    
    def get_metric_summary(self, metric_name: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time window."""
        if metric_name not in self.metrics_history:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        recent_metrics = [
            entry for entry in self.metrics_history[metric_name]
            if entry.get("timestamp", datetime.min) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [entry.get("value", 0) for entry in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "current": values[-1] if values else 0,
            "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "decreasing" if len(values) > 1 else "stable"
        }


class ResourceManager(UnifiedManagerBase):
    """
    Unified Resource Manager consolidating all performance and resource-related functionality.
    
    Replaces 41 separate files:
    - performance_optimizer.py
    - resource_optimizer.py
    - capacity_manager.py
    - capacity_planning.py
    - load_balancing_benchmarks.py
    - load_prediction_service.py
    - load_testing.py
    - performance_benchmarks.py
    - performance_evaluator.py
    - performance_metrics_collector.py
    - performance_metrics_publisher.py
    - performance_monitor.py
    - performance_monitoring.py
    - performance_monitoring_dashboard.py
    - performance_optimization_advisor.py
    - performance_optimizations.py
    - performance_orchestrator.py
    - performance_orchestrator_integration.py
    - performance_orchestrator_plugin.py
    - performance_storage_engine.py
    - performance_validator.py
    - cost_monitoring.py
    - custom_metrics_exporter.py
    - dashboard_metrics_streaming.py
    - database_performance_validator.py
    - distributed_load_balancing_state.py
    - dlq_monitoring.py
    - enhanced_communication_load_testing.py
    - health_monitoring.py
    - hook_performance_benchmarks.py
    - integrated_system_performance_validator.py
    - metrics_collector.py
    - metrics_migration_example.py
    - observability_performance_testing.py
    - orchestrator_load_balancing_integration.py
    - orchestrator_load_testing.py
    - performance_migration_adapter.py
    - security_monitoring_system.py
    - security_performance_validator.py
    - sleep_wake_performance_testing.py
    - vs_2_1_performance_validator.py
    """
    
    def __init__(self, config: ManagerConfig, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(config, dependencies)
        
        # Core components
        self.optimizer = PerformanceOptimizer()
        self.load_balancer = LoadBalancer()
        self.capacity_manager = CapacityManager()
        self.metrics_collector = MetricsCollector()
        
        # State tracking
        self.active_allocations: Dict[uuid.UUID, ResourceAllocation] = {}
        self.monitoring_tasks: List[asyncio.Task] = []
        self.performance_alerts: List[Dict[str, Any]] = []
        
        # Configuration
        self.monitoring_interval_seconds = config.plugin_config.get("monitoring_interval", 30)
        self.alert_thresholds = config.plugin_config.get("alert_thresholds", {
            "cpu_warning": 80,
            "cpu_critical": 90,
            "memory_warning": 85,
            "memory_critical": 95
        })
    
    async def _initialize_manager(self) -> bool:
        """Initialize the resource manager."""
        try:
            # Start monitoring tasks
            self.monitoring_tasks.extend([
                asyncio.create_task(self._system_monitoring_loop()),
                asyncio.create_task(self._optimization_loop()),
                asyncio.create_task(self._capacity_monitoring_loop())
            ])
            
            # Initialize baseline metrics
            await self.metrics_collector.collect_system_metrics()
            
            logger.info(
                "Resource Manager initialized",
                monitoring_interval=self.monitoring_interval_seconds,
                alert_thresholds=self.alert_thresholds
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Resource Manager", error=str(e))
            return False
    
    async def _shutdown_manager(self) -> None:
        """Shutdown the resource manager."""
        try:
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            # Deallocate all resources
            for allocation in list(self.active_allocations.values()):
                self.capacity_manager.deallocate_resources(allocation)
            
            logger.info("Resource Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during Resource Manager shutdown", error=str(e))
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get resource manager health information."""
        current_metrics = await self.metrics_collector.collect_system_metrics()
        available_capacity = self.capacity_manager.get_available_capacity()
        
        return {
            "current_metrics": {
                "cpu_usage_percent": current_metrics.cpu_usage_percent,
                "memory_usage_percent": current_metrics.memory_usage_percent,
                "disk_usage_percent": current_metrics.disk_usage_percent,
                "active_agents": current_metrics.active_agents,
                "pending_tasks": current_metrics.pending_tasks
            },
            "available_capacity": {k.value: v for k, v in available_capacity.items()},
            "active_allocations": len(self.active_allocations),
            "monitoring_tasks": len([t for t in self.monitoring_tasks if not t.done()]),
            "performance_alerts": len(self.performance_alerts),
            "optimization_history_size": len(self.optimizer.optimization_history),
            "benchmark_results_count": len(self.metrics_collector.benchmark_results)
        }
    
    async def _load_plugins(self) -> None:
        """Load resource manager plugins."""
        # Performance monitoring plugins would be loaded here
        pass
    
    # === BACKGROUND MONITORING LOOPS ===
    
    async def _system_monitoring_loop(self) -> None:
        """Background system monitoring loop."""
        while True:
            try:
                # Collect current metrics
                metrics = await self.metrics_collector.collect_system_metrics()
                
                # Update load balancer with current loads
                self.load_balancer.record_load(ResourceType.CPU, metrics.cpu_usage_percent)
                self.load_balancer.record_load(ResourceType.MEMORY, metrics.memory_usage_percent)
                self.load_balancer.record_load(ResourceType.DISK, metrics.disk_usage_percent)
                
                # Check for alerts
                await self._check_performance_alerts(metrics)
                
                # Publish metrics if enabled
                if self.config.metrics_enabled:
                    await self._publish_metrics(metrics)
                
                await asyncio.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error("Error in system monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while True:
            try:
                await asyncio.sleep(120)  # Run optimization every 2 minutes
                
                # Get current metrics
                metrics = await self.metrics_collector.collect_system_metrics()
                
                # Analyze and get optimization recommendations
                optimizations = await self.optimizer.analyze_and_optimize(metrics)
                
                # Apply automatic optimizations if configured
                if optimizations and self.config.plugin_config.get("auto_optimize", False):
                    await self._apply_optimizations(optimizations)
                
            except Exception as e:
                logger.error("Error in optimization loop", error=str(e))
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _capacity_monitoring_loop(self) -> None:
        """Background capacity monitoring loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check capacity utilization
                available_capacity = self.capacity_manager.get_available_capacity()
                
                # Generate capacity alerts if needed
                for resource_type, available in available_capacity.items():
                    if available < 20:  # Less than 20% capacity remaining
                        await self._generate_capacity_alert(resource_type, available)
                
            except Exception as e:
                logger.error("Error in capacity monitoring loop", error=str(e))
                await asyncio.sleep(600)  # Wait longer on error
    
    # === CORE RESOURCE OPERATIONS ===
    
    async def allocate_resources(
        self, 
        request: ResourceAllocation,
        strategy: ResourceAllocationStrategy = ResourceAllocationStrategy.LOAD_BALANCED
    ) -> Tuple[bool, Optional[ResourceAllocation]]:
        """Allocate resources with intelligent optimization."""
        return await self.execute_with_monitoring(
            "allocate_resources",
            self._allocate_resources_impl,
            request,
            strategy
        )
    
    async def _allocate_resources_impl(
        self,
        request: ResourceAllocation,
        strategy: ResourceAllocationStrategy
    ) -> Tuple[bool, Optional[ResourceAllocation]]:
        """Internal implementation of resource allocation."""
        try:
            # Get optimized allocation from load balancer
            optimized_allocation = self.load_balancer.get_optimal_allocation(request, strategy)
            
            # Attempt to allocate resources
            success = self.capacity_manager.allocate_resources(optimized_allocation)
            
            if success:
                # Track allocation
                self.active_allocations[optimized_allocation.allocation_id] = optimized_allocation
                
                logger.info(
                    "✅ Resources allocated",
                    allocation_id=str(optimized_allocation.allocation_id),
                    cpu_cores=optimized_allocation.cpu_cores,
                    memory_mb=optimized_allocation.memory_mb,
                    strategy=strategy.value
                )
                
                return True, optimized_allocation
            else:
                logger.warning(
                    "❌ Resource allocation failed - insufficient capacity",
                    requested_cpu=request.cpu_cores,
                    requested_memory=request.memory_mb
                )
                return False, None
                
        except Exception as e:
            logger.error("Resource allocation error", error=str(e))
            return False, None
    
    async def deallocate_resources(self, allocation_id: uuid.UUID) -> bool:
        """Deallocate resources by allocation ID."""
        return await self.execute_with_monitoring(
            "deallocate_resources",
            self._deallocate_resources_impl,
            allocation_id
        )
    
    async def _deallocate_resources_impl(self, allocation_id: uuid.UUID) -> bool:
        """Internal implementation of resource deallocation."""
        try:
            if allocation_id not in self.active_allocations:
                logger.warning("Allocation not found for deallocation", allocation_id=str(allocation_id))
                return False
            
            allocation = self.active_allocations[allocation_id]
            
            # Deallocate from capacity manager
            self.capacity_manager.deallocate_resources(allocation)
            
            # Remove from tracking
            del self.active_allocations[allocation_id]
            
            logger.info(
                "✅ Resources deallocated",
                allocation_id=str(allocation_id),
                cpu_cores=allocation.cpu_cores,
                memory_mb=allocation.memory_mb
            )
            
            return True
            
        except Exception as e:
            logger.error("Resource deallocation error", allocation_id=str(allocation_id), error=str(e))
            return False
    
    async def benchmark_operation(
        self, 
        operation_name: str, 
        operation_func, 
        *args, 
        **kwargs
    ) -> PerformanceBenchmark:
        """Benchmark an operation's performance."""
        return await self.execute_with_monitoring(
            "benchmark_operation",
            self.metrics_collector.benchmark_operation,
            operation_name,
            operation_func,
            *args,
            **kwargs
        )
    
    async def predict_load(
        self, 
        resource_type: ResourceType, 
        horizon_minutes: int = 15
    ) -> LoadPrediction:
        """Predict future resource load."""
        return await self.execute_with_monitoring(
            "predict_load",
            self._predict_load_impl,
            resource_type,
            horizon_minutes
        )
    
    async def _predict_load_impl(
        self, 
        resource_type: ResourceType, 
        horizon_minutes: int
    ) -> LoadPrediction:
        """Internal implementation of load prediction."""
        return self.load_balancer.predict_load(resource_type, horizon_minutes)
    
    # === MONITORING AND ALERTING ===
    
    async def _check_performance_alerts(self, metrics: ResourceMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []
        
        # CPU alerts
        if metrics.cpu_usage_percent >= self.alert_thresholds["cpu_critical"]:
            alerts.append({
                "type": "cpu_critical",
                "message": f"Critical CPU usage: {metrics.cpu_usage_percent:.1f}%",
                "severity": "critical",
                "value": metrics.cpu_usage_percent,
                "threshold": self.alert_thresholds["cpu_critical"]
            })
        elif metrics.cpu_usage_percent >= self.alert_thresholds["cpu_warning"]:
            alerts.append({
                "type": "cpu_warning",
                "message": f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                "severity": "warning",
                "value": metrics.cpu_usage_percent,
                "threshold": self.alert_thresholds["cpu_warning"]
            })
        
        # Memory alerts
        if metrics.memory_usage_percent >= self.alert_thresholds["memory_critical"]:
            alerts.append({
                "type": "memory_critical",
                "message": f"Critical memory usage: {metrics.memory_usage_percent:.1f}%",
                "severity": "critical",
                "value": metrics.memory_usage_percent,
                "threshold": self.alert_thresholds["memory_critical"]
            })
        elif metrics.memory_usage_percent >= self.alert_thresholds["memory_warning"]:
            alerts.append({
                "type": "memory_warning",
                "message": f"High memory usage: {metrics.memory_usage_percent:.1f}%",
                "severity": "warning",
                "value": metrics.memory_usage_percent,
                "threshold": self.alert_thresholds["memory_warning"]
            })
        
        # Store and publish alerts
        for alert in alerts:
            alert["timestamp"] = datetime.utcnow()
            self.performance_alerts.append(alert)
            
            logger.warning(
                f"Performance Alert: {alert['message']}",
                alert_type=alert["type"],
                severity=alert["severity"],
                value=alert["value"]
            )
        
        # Keep only recent alerts
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.performance_alerts = [
            alert for alert in self.performance_alerts
            if alert["timestamp"] > cutoff_time
        ]
    
    async def _generate_capacity_alert(self, resource_type: ResourceType, available: float) -> None:
        """Generate capacity alert."""
        alert = {
            "type": "capacity_low",
            "message": f"Low {resource_type.value} capacity: {available:.1f}% remaining",
            "severity": "warning" if available > 10 else "critical",
            "resource_type": resource_type.value,
            "available": available,
            "timestamp": datetime.utcnow()
        }
        
        self.performance_alerts.append(alert)
        
        logger.warning(
            f"Capacity Alert: {alert['message']}",
            resource_type=resource_type.value,
            available=available
        )
    
    async def _publish_metrics(self, metrics: ResourceMetrics) -> None:
        """Publish metrics to monitoring systems."""
        try:
            # Publish to Redis streams for real-time monitoring
            redis_client = get_redis()
            if redis_client:
                metrics_data = {
                    "cpu_usage": metrics.cpu_usage_percent,
                    "memory_usage": metrics.memory_usage_percent,
                    "disk_usage": metrics.disk_usage_percent,
                    "active_agents": metrics.active_agents,
                    "pending_tasks": metrics.pending_tasks,
                    "timestamp": metrics.timestamp.isoformat()
                }
                
                await redis_client.xadd(
                    "system_metrics:performance",
                    metrics_data,
                    maxlen=10000
                )
        
        except Exception as e:
            logger.error("Failed to publish metrics", error=str(e))
    
    async def _apply_optimizations(self, optimizations: List[Dict[str, Any]]) -> None:
        """Apply automatic optimizations."""
        for optimization in optimizations:
            try:
                if optimization["type"] == "memory_optimization":
                    # Trigger memory cleanup
                    await self._trigger_memory_cleanup()
                elif optimization["type"] == "cpu_optimization":
                    # Implement CPU throttling
                    await self._apply_cpu_throttling()
                
                logger.info(
                    "Applied optimization",
                    type=optimization["type"],
                    priority=optimization.get("priority", 0)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to apply optimization",
                    type=optimization["type"],
                    error=str(e)
                )
    
    async def _trigger_memory_cleanup(self) -> None:
        """Trigger memory cleanup operations."""
        # This would interface with other managers to trigger cleanup
        logger.info("Triggering memory cleanup operations")
    
    async def _apply_cpu_throttling(self) -> None:
        """Apply CPU throttling for non-critical operations."""
        # This would interface with other managers to throttle CPU usage
        logger.info("Applying CPU throttling")
    
    # === PUBLIC API METHODS ===
    
    async def get_system_metrics(self) -> ResourceMetrics:
        """Get current system metrics."""
        return await self.metrics_collector.collect_system_metrics()
    
    async def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary over time window."""
        try:
            # Get current metrics
            current_metrics = await self.get_system_metrics()
            
            # Get historical summaries
            cpu_summary = self.metrics_collector.get_metric_summary("cpu_usage", time_window_minutes)
            memory_summary = self.metrics_collector.get_metric_summary("memory_usage", time_window_minutes)
            
            # Get recent benchmarks
            recent_benchmarks = [
                b for b in self.metrics_collector.benchmark_results
                if b.timestamp > datetime.utcnow() - timedelta(minutes=time_window_minutes)
            ]
            
            # Get recent alerts
            recent_alerts = [
                alert for alert in self.performance_alerts
                if alert["timestamp"] > datetime.utcnow() - timedelta(minutes=time_window_minutes)
            ]
            
            return {
                "current_metrics": {
                    "cpu_usage_percent": current_metrics.cpu_usage_percent,
                    "memory_usage_percent": current_metrics.memory_usage_percent,
                    "disk_usage_percent": current_metrics.disk_usage_percent,
                    "active_agents": current_metrics.active_agents,
                    "pending_tasks": current_metrics.pending_tasks
                },
                "historical_summary": {
                    "cpu": cpu_summary,
                    "memory": memory_summary
                },
                "capacity": {
                    "available": self.capacity_manager.get_available_capacity(),
                    "active_allocations": len(self.active_allocations)
                },
                "performance": {
                    "recent_benchmarks": len(recent_benchmarks),
                    "avg_execution_time_ms": sum(b.execution_time_ms for b in recent_benchmarks) / max(len(recent_benchmarks), 1),
                    "success_rate": sum(1 for b in recent_benchmarks if b.success) / max(len(recent_benchmarks), 1)
                },
                "alerts": {
                    "recent_count": len(recent_alerts),
                    "critical_count": len([a for a in recent_alerts if a["severity"] == "critical"]),
                    "warning_count": len([a for a in recent_alerts if a["severity"] == "warning"])
                },
                "optimization": {
                    "recent_optimizations": len([opt for opt in self.optimizer.optimization_history 
                                               if opt["timestamp"] > datetime.utcnow() - timedelta(minutes=time_window_minutes)])
                }
            }
            
        except Exception as e:
            logger.error("Failed to get performance summary", error=str(e))
            return {"error": str(e)}
    
    def record_application_metric(self, name: str, value: Any) -> None:
        """Record application-specific metric."""
        self.metrics_collector.record_custom_metric(name, value)


# Factory function for creating resource manager
def create_resource_manager(**config_overrides) -> ResourceManager:
    """Create and initialize a resource manager."""
    config = create_manager_config("ResourceManager", **config_overrides)
    return ResourceManager(config)