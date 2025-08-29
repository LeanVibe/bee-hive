"""
Epic E Phase 2: Intelligent Resource Management System.

Implements intelligent memory and CPU optimization, resource pooling,
and adaptive resource allocation for system-wide performance excellence.

Features:
- Intelligent memory management with leak detection
- CPU utilization optimization and load balancing
- Resource pooling with adaptive sizing
- Performance-aware resource allocation
- Real-time resource monitoring and optimization
- Predictive resource scaling based on load patterns
"""

import asyncio
import logging
import time
import json
import statistics
import psutil
import threading
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import weakref
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources to manage."""
    MEMORY = "memory"
    CPU = "cpu"
    DATABASE_CONNECTIONS = "database_connections"
    REDIS_CONNECTIONS = "redis_connections"
    WEBSOCKET_CONNECTIONS = "websocket_connections"
    FILE_HANDLES = "file_handles"
    THREAD_POOL = "thread_pool"


class OptimizationStrategy(Enum):
    """Resource optimization strategies."""
    CONSERVATIVE = "conservative"      # Minimize resource usage
    BALANCED = "balanced"             # Balance performance and resource usage
    AGGRESSIVE = "aggressive"         # Maximize performance, higher resource usage
    ADAPTIVE = "adaptive"            # Dynamically adjust based on load


@dataclass
class ResourceMetrics:
    """Metrics for resource utilization."""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    utilization_percentage: float
    allocation_efficiency: float
    waste_percentage: float
    optimization_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationRecommendation:
    """Resource optimization recommendation."""
    resource_type: ResourceType
    current_allocation: float
    recommended_allocation: float
    expected_improvement: float
    confidence_score: float
    implementation_priority: str  # HIGH, MEDIUM, LOW
    rationale: str


class ResourcePool(ABC):
    """Abstract base class for resource pools."""
    
    def __init__(self, name: str, initial_size: int = 10, max_size: int = 100):
        self.name = name
        self.initial_size = initial_size
        self.max_size = max_size
        self.current_size = 0
        self.active_resources = set()
        self.idle_resources = deque()
        self.creation_count = 0
        self.destruction_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.Lock()
    
    @abstractmethod
    async def create_resource(self) -> Any:
        """Create a new resource."""
        pass
    
    @abstractmethod
    async def destroy_resource(self, resource: Any) -> None:
        """Destroy a resource."""
        pass
    
    @abstractmethod
    async def validate_resource(self, resource: Any) -> bool:
        """Validate that a resource is still usable."""
        pass
    
    async def acquire_resource(self) -> Any:
        """Acquire a resource from the pool."""
        with self._lock:
            # Try to get from idle pool first
            if self.idle_resources:
                resource = self.idle_resources.popleft()
                if await self.validate_resource(resource):
                    self.active_resources.add(resource)
                    self.hit_count += 1
                    return resource
                else:
                    # Resource is invalid, destroy it
                    await self.destroy_resource(resource)
                    self.destruction_count += 1
            
            # Create new resource if under max size
            if self.current_size < self.max_size:
                resource = await self.create_resource()
                self.active_resources.add(resource)
                self.current_size += 1
                self.creation_count += 1
                self.miss_count += 1
                return resource
            
            # Pool is at capacity, wait for a resource to become available
            raise ResourceExhaustedException(f"Resource pool {self.name} is exhausted")
    
    async def release_resource(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self._lock:
            if resource in self.active_resources:
                self.active_resources.remove(resource)
                if await self.validate_resource(resource):
                    self.idle_resources.append(resource)
                else:
                    await self.destroy_resource(resource)
                    self.current_size -= 1
                    self.destruction_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        with self._lock:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
            return {
                'name': self.name,
                'current_size': self.current_size,
                'max_size': self.max_size,
                'active_count': len(self.active_resources),
                'idle_count': len(self.idle_resources),
                'hit_rate': hit_rate,
                'creation_count': self.creation_count,
                'destruction_count': self.destruction_count,
                'utilization': len(self.active_resources) / self.current_size if self.current_size > 0 else 0
            }


class DatabaseConnectionPool(ResourcePool):
    """Database connection pool implementation."""
    
    async def create_resource(self) -> Any:
        """Create a database connection."""
        # Simulate database connection creation
        await asyncio.sleep(0.01)  # Connection establishment delay
        return {"connection_id": f"db_conn_{self.creation_count}", "created_at": time.time()}
    
    async def destroy_resource(self, resource: Any) -> None:
        """Destroy a database connection."""
        # Simulate connection cleanup
        await asyncio.sleep(0.005)
    
    async def validate_resource(self, resource: Any) -> bool:
        """Validate database connection is still active."""
        # Simulate connection validation
        if resource and "connection_id" in resource:
            # Simulate occasional connection timeouts
            return time.time() - resource["created_at"] < 300  # 5 minute timeout
        return False


class RedisConnectionPool(ResourcePool):
    """Redis connection pool implementation."""
    
    async def create_resource(self) -> Any:
        """Create a Redis connection."""
        await asyncio.sleep(0.005)  # Faster than DB connection
        return {"connection_id": f"redis_conn_{self.creation_count}", "created_at": time.time()}
    
    async def destroy_resource(self, resource: Any) -> None:
        """Destroy a Redis connection."""
        await asyncio.sleep(0.002)
    
    async def validate_resource(self, resource: Any) -> bool:
        """Validate Redis connection is still active."""
        if resource and "connection_id" in resource:
            return time.time() - resource["created_at"] < 600  # 10 minute timeout
        return False


class MemoryManager:
    """Intelligent memory management system."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = deque(maxlen=1000)
        self.gc_stats = deque(maxlen=100)
        self.object_trackers = {}
        self.memory_pools = {}
        self.leak_detectors = []
        
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        self.monitoring_active = True
        asyncio.create_task(self._memory_monitoring_loop())
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
    
    async def _memory_monitoring_loop(self):
        """Continuous memory monitoring loop."""
        while self.monitoring_active:
            try:
                memory_info = self.process.memory_info()
                memory_sample = {
                    'timestamp': time.time(),
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': self.process.memory_percent(),
                    'gc_counts': gc.get_count()
                }
                
                self.memory_samples.append(memory_sample)
                
                # Trigger memory optimization if needed
                if memory_sample['percent'] > 80.0:  # High memory usage threshold
                    await self.optimize_memory_usage()
                
                await asyncio.sleep(5.0)  # Sample every 5 seconds
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def optimize_memory_usage(self):
        """Optimize memory usage through various strategies."""
        logger.info("Optimizing memory usage...")
        
        # Force garbage collection
        before_gc = self.process.memory_info().rss
        for generation in range(3):
            collected = gc.collect(generation)
            if collected > 0:
                logger.debug(f"GC generation {generation}: collected {collected} objects")
        
        after_gc = self.process.memory_info().rss
        memory_freed = (before_gc - after_gc) / 1024 / 1024  # MB
        
        self.gc_stats.append({
            'timestamp': time.time(),
            'memory_freed_mb': memory_freed,
            'before_gc_mb': before_gc / 1024 / 1024,
            'after_gc_mb': after_gc / 1024 / 1024
        })
        
        # Clear caches if memory usage is still high
        if self.process.memory_percent() > 75.0:
            await self._clear_non_essential_caches()
        
        logger.info(f"Memory optimization complete: freed {memory_freed:.1f}MB")
    
    async def _clear_non_essential_caches(self):
        """Clear non-essential caches to free memory."""
        # Clear memory pools that haven't been used recently
        current_time = time.time()
        for pool_name, pool_data in list(self.memory_pools.items()):
            if current_time - pool_data.get('last_used', 0) > 300:  # 5 minutes
                del self.memory_pools[pool_name]
                logger.debug(f"Cleared memory pool: {pool_name}")
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        if len(self.memory_samples) < 50:
            return []  # Need sufficient samples
        
        # Analyze memory growth trend
        recent_samples = list(self.memory_samples)[-50:]
        timestamps = [s['timestamp'] for s in recent_samples]
        memory_usage = [s['rss_mb'] for s in recent_samples]
        
        # Calculate growth rate
        if len(memory_usage) > 1:
            time_span = timestamps[-1] - timestamps[0]
            memory_growth = memory_usage[-1] - memory_usage[0]
            growth_rate = memory_growth / time_span * 60  # MB per minute
            
            # Detect concerning growth patterns
            leaks = []
            if growth_rate > 10.0:  # Growing >10MB per minute
                leaks.append({
                    'type': 'high_growth_rate',
                    'growth_rate_mb_per_min': growth_rate,
                    'severity': 'HIGH',
                    'recommendation': 'Immediate investigation required'
                })
            elif growth_rate > 5.0:  # Growing >5MB per minute
                leaks.append({
                    'type': 'moderate_growth_rate',
                    'growth_rate_mb_per_min': growth_rate,
                    'severity': 'MEDIUM',
                    'recommendation': 'Monitor closely and consider optimization'
                })
            
            # Check for consistent growth without GC relief
            memory_variance = statistics.variance(memory_usage)
            if memory_variance < 10.0 and growth_rate > 1.0:  # Low variance but growing
                leaks.append({
                    'type': 'consistent_growth',
                    'variance': memory_variance,
                    'growth_rate_mb_per_min': growth_rate,
                    'severity': 'MEDIUM',
                    'recommendation': 'Possible memory leak - investigate object retention'
                })
            
            return leaks
        
        return []
    
    def get_memory_metrics(self) -> ResourceMetrics:
        """Get current memory metrics."""
        if not self.memory_samples:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            return ResourceMetrics(
                resource_type=ResourceType.MEMORY,
                current_usage=current_memory,
                peak_usage=current_memory,
                average_usage=current_memory,
                utilization_percentage=self.process.memory_percent(),
                allocation_efficiency=1.0,
                waste_percentage=0.0,
                optimization_score=100.0
            )
        
        memory_values = [s['rss_mb'] for s in self.memory_samples]
        current_memory = memory_values[-1]
        peak_memory = max(memory_values)
        avg_memory = statistics.mean(memory_values)
        
        # Calculate efficiency metrics
        utilization = self.process.memory_percent()
        efficiency = min(1.0, avg_memory / peak_memory) if peak_memory > 0 else 1.0
        waste = max(0.0, (peak_memory - avg_memory) / peak_memory * 100) if peak_memory > 0 else 0.0
        
        # Calculate optimization score
        optimization_score = (efficiency * 50) + (max(0, 100 - waste) * 0.3) + (max(0, 100 - utilization) * 0.2)
        
        return ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_usage=current_memory,
            peak_usage=peak_memory,
            average_usage=avg_memory,
            utilization_percentage=utilization,
            allocation_efficiency=efficiency,
            waste_percentage=waste,
            optimization_score=optimization_score
        )


class CPUManager:
    """Intelligent CPU utilization management."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_samples = deque(maxlen=500)
        self.load_patterns = deque(maxlen=100)
        self.thread_pools = {}
        
    def start_monitoring(self):
        """Start CPU monitoring."""
        self.monitoring_active = True
        asyncio.create_task(self._cpu_monitoring_loop())
    
    def stop_monitoring(self):
        """Stop CPU monitoring."""
        self.monitoring_active = False
    
    async def _cpu_monitoring_loop(self):
        """Continuous CPU monitoring loop."""
        while self.monitoring_active:
            try:
                cpu_percent = self.process.cpu_percent()
                system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                
                cpu_sample = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'system_load': system_load,
                    'thread_count': self.process.num_threads()
                }
                
                self.cpu_samples.append(cpu_sample)
                
                # Detect load patterns
                await self._analyze_load_patterns()
                
                # Optimize CPU usage if needed
                if cpu_percent > 90.0:
                    await self.optimize_cpu_usage()
                
                await asyncio.sleep(2.0)  # Sample every 2 seconds
                
            except Exception as e:
                logger.warning(f"CPU monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def _analyze_load_patterns(self):
        """Analyze CPU load patterns for optimization opportunities."""
        if len(self.cpu_samples) < 20:
            return
        
        recent_samples = list(self.cpu_samples)[-20:]
        cpu_values = [s['cpu_percent'] for s in recent_samples]
        
        # Detect load pattern
        avg_cpu = statistics.mean(cpu_values)
        cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 0
        
        if cpu_variance < 5.0 and avg_cpu > 70.0:
            pattern_type = "sustained_high"
        elif cpu_variance > 50.0:
            pattern_type = "spiky"
        elif avg_cpu < 20.0:
            pattern_type = "idle"
        else:
            pattern_type = "normal"
        
        self.load_patterns.append({
            'timestamp': time.time(),
            'pattern_type': pattern_type,
            'avg_cpu': avg_cpu,
            'variance': cpu_variance
        })
    
    async def optimize_cpu_usage(self):
        """Optimize CPU utilization."""
        logger.info("Optimizing CPU usage...")
        
        # Analyze current load patterns
        if self.load_patterns:
            latest_pattern = self.load_patterns[-1]
            
            if latest_pattern['pattern_type'] == 'sustained_high':
                # Scale up thread pools for sustained high load
                await self._scale_thread_pools(scale_factor=1.2)
                
            elif latest_pattern['pattern_type'] == 'spiky':
                # Add more threads to handle spikes better
                await self._scale_thread_pools(scale_factor=1.1)
                
            elif latest_pattern['pattern_type'] == 'idle':
                # Scale down to conserve resources
                await self._scale_thread_pools(scale_factor=0.8)
        
        logger.info("CPU optimization complete")
    
    async def _scale_thread_pools(self, scale_factor: float):
        """Scale thread pools based on load patterns."""
        for pool_name, pool in self.thread_pools.items():
            current_size = pool.get('size', 4)
            new_size = max(2, min(20, int(current_size * scale_factor)))
            
            if new_size != current_size:
                pool['size'] = new_size
                logger.debug(f"Scaled thread pool {pool_name}: {current_size} -> {new_size}")
    
    def get_cpu_metrics(self) -> ResourceMetrics:
        """Get current CPU metrics."""
        if not self.cpu_samples:
            current_cpu = self.process.cpu_percent()
            return ResourceMetrics(
                resource_type=ResourceType.CPU,
                current_usage=current_cpu,
                peak_usage=current_cpu,
                average_usage=current_cpu,
                utilization_percentage=current_cpu,
                allocation_efficiency=1.0,
                waste_percentage=0.0,
                optimization_score=100.0
            )
        
        cpu_values = [s['cpu_percent'] for s in self.cpu_samples]
        current_cpu = cpu_values[-1]
        peak_cpu = max(cpu_values)
        avg_cpu = statistics.mean(cpu_values)
        
        # Calculate efficiency and optimization score
        efficiency = 1.0 - (peak_cpu - avg_cpu) / 100.0 if peak_cpu > avg_cpu else 1.0
        waste = (peak_cpu - avg_cpu) / peak_cpu * 100 if peak_cpu > 0 else 0.0
        optimization_score = (efficiency * 60) + (max(0, 100 - avg_cpu) * 0.4)
        
        return ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=current_cpu,
            peak_usage=peak_cpu,
            average_usage=avg_cpu,
            utilization_percentage=current_cpu,
            allocation_efficiency=efficiency,
            waste_percentage=waste,
            optimization_score=optimization_score
        )


class IntelligentResourceManager:
    """Central intelligent resource management system."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cpu_manager = CPUManager()
        self.resource_pools = {}
        self.optimization_history = deque(maxlen=100)
        self.strategy = OptimizationStrategy.ADAPTIVE
        
        # Initialize resource pools
        self._initialize_resource_pools()
    
    def _initialize_resource_pools(self):
        """Initialize various resource pools."""
        self.resource_pools = {
            'database': DatabaseConnectionPool('database', initial_size=5, max_size=50),
            'redis': RedisConnectionPool('redis', initial_size=3, max_size=20),
        }
    
    async def start_management(self):
        """Start intelligent resource management."""
        logger.info("Starting intelligent resource management...")
        
        # Start component managers
        self.memory_manager.start_monitoring()
        self.cpu_manager.start_monitoring()
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
    
    async def stop_management(self):
        """Stop resource management."""
        logger.info("Stopping intelligent resource management...")
        
        self.memory_manager.stop_monitoring()
        self.cpu_manager.stop_monitoring()
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while True:
            try:
                await asyncio.sleep(30.0)  # Run optimization every 30 seconds
                
                # Get current metrics
                memory_metrics = self.memory_manager.get_memory_metrics()
                cpu_metrics = self.cpu_manager.get_cpu_metrics()
                
                # Generate optimization recommendations
                recommendations = await self.generate_optimization_recommendations(
                    memory_metrics, cpu_metrics
                )
                
                # Apply recommendations based on strategy
                await self._apply_optimization_recommendations(recommendations)
                
                # Store optimization results
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'memory_metrics': memory_metrics,
                    'cpu_metrics': cpu_metrics,
                    'recommendations': recommendations
                })
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    async def generate_optimization_recommendations(
        self, 
        memory_metrics: ResourceMetrics, 
        cpu_metrics: ResourceMetrics
    ) -> List[OptimizationRecommendation]:
        """Generate intelligent optimization recommendations."""
        recommendations = []
        
        # Memory optimization recommendations
        if memory_metrics.utilization_percentage > 85.0:
            recommendations.append(OptimizationRecommendation(
                resource_type=ResourceType.MEMORY,
                current_allocation=memory_metrics.current_usage,
                recommended_allocation=memory_metrics.current_usage * 0.9,
                expected_improvement=15.0,
                confidence_score=0.8,
                implementation_priority="HIGH",
                rationale=f"Memory usage at {memory_metrics.utilization_percentage:.1f}% - trigger aggressive cleanup"
            ))
        elif memory_metrics.waste_percentage > 30.0:
            recommendations.append(OptimizationRecommendation(
                resource_type=ResourceType.MEMORY,
                current_allocation=memory_metrics.peak_usage,
                recommended_allocation=memory_metrics.average_usage * 1.2,
                expected_improvement=10.0,
                confidence_score=0.7,
                implementation_priority="MEDIUM",
                rationale=f"Memory waste at {memory_metrics.waste_percentage:.1f}% - optimize allocation patterns"
            ))
        
        # CPU optimization recommendations
        if cpu_metrics.utilization_percentage > 80.0:
            recommendations.append(OptimizationRecommendation(
                resource_type=ResourceType.CPU,
                current_allocation=cpu_metrics.current_usage,
                recommended_allocation=cpu_metrics.current_usage * 0.85,
                expected_improvement=20.0,
                confidence_score=0.75,
                implementation_priority="HIGH",
                rationale=f"CPU usage at {cpu_metrics.utilization_percentage:.1f}% - optimize thread allocation"
            ))
        elif cpu_metrics.average_usage < 30.0:
            recommendations.append(OptimizationRecommendation(
                resource_type=ResourceType.CPU,
                current_allocation=cpu_metrics.current_usage,
                recommended_allocation=cpu_metrics.current_usage * 1.1,
                expected_improvement=5.0,
                confidence_score=0.6,
                implementation_priority="LOW",
                rationale=f"CPU underutilized at {cpu_metrics.average_usage:.1f}% - can handle more load"
            ))
        
        # Resource pool optimization
        for pool_name, pool in self.resource_pools.items():
            pool_metrics = pool.get_metrics()
            if pool_metrics['utilization'] > 0.9:
                recommendations.append(OptimizationRecommendation(
                    resource_type=ResourceType.DATABASE_CONNECTIONS if pool_name == 'database' else ResourceType.REDIS_CONNECTIONS,
                    current_allocation=pool_metrics['current_size'],
                    recommended_allocation=min(pool_metrics['max_size'], pool_metrics['current_size'] * 1.3),
                    expected_improvement=25.0,
                    confidence_score=0.85,
                    implementation_priority="MEDIUM",
                    rationale=f"{pool_name} pool at {pool_metrics['utilization']:.1%} utilization - scale up"
                ))
        
        return recommendations
    
    async def _apply_optimization_recommendations(
        self, 
        recommendations: List[OptimizationRecommendation]
    ):
        """Apply optimization recommendations based on current strategy."""
        for rec in recommendations:
            if rec.implementation_priority == "HIGH" or (
                rec.implementation_priority == "MEDIUM" and rec.confidence_score >= 0.7
            ):
                await self._implement_recommendation(rec)
    
    async def _implement_recommendation(self, recommendation: OptimizationRecommendation):
        """Implement a specific optimization recommendation."""
        try:
            if recommendation.resource_type == ResourceType.MEMORY:
                if recommendation.current_allocation > recommendation.recommended_allocation:
                    await self.memory_manager.optimize_memory_usage()
                    
            elif recommendation.resource_type == ResourceType.CPU:
                if recommendation.current_allocation > recommendation.recommended_allocation:
                    await self.cpu_manager.optimize_cpu_usage()
            
            elif recommendation.resource_type in [ResourceType.DATABASE_CONNECTIONS, ResourceType.REDIS_CONNECTIONS]:
                # Adjust pool sizes
                pool_name = 'database' if recommendation.resource_type == ResourceType.DATABASE_CONNECTIONS else 'redis'
                if pool_name in self.resource_pools:
                    pool = self.resource_pools[pool_name]
                    new_size = int(recommendation.recommended_allocation)
                    pool.max_size = min(pool.max_size * 2, new_size)  # Don't exceed reasonable limits
            
            logger.info(f"Applied optimization: {recommendation.rationale}")
            
        except Exception as e:
            logger.error(f"Failed to implement recommendation: {e}")
    
    def get_system_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive system resource report."""
        memory_metrics = self.memory_manager.get_memory_metrics()
        cpu_metrics = self.cpu_manager.get_cpu_metrics()
        
        # Get resource pool metrics
        pool_metrics = {}
        for pool_name, pool in self.resource_pools.items():
            pool_metrics[pool_name] = pool.get_metrics()
        
        # Detect memory leaks
        memory_leaks = self.memory_manager.detect_memory_leaks()
        
        # Calculate overall system efficiency
        overall_efficiency = (
            (memory_metrics.optimization_score * 0.4) +
            (cpu_metrics.optimization_score * 0.4) +
            (sum(p['hit_rate'] for p in pool_metrics.values()) / len(pool_metrics) * 100 * 0.2)
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_efficiency_score': overall_efficiency,
            'memory_metrics': {
                'current_usage_mb': memory_metrics.current_usage,
                'peak_usage_mb': memory_metrics.peak_usage,
                'utilization_percent': memory_metrics.utilization_percentage,
                'efficiency_score': memory_metrics.optimization_score,
                'detected_leaks': memory_leaks
            },
            'cpu_metrics': {
                'current_usage_percent': cpu_metrics.current_usage,
                'average_usage_percent': cpu_metrics.average_usage,
                'efficiency_score': cpu_metrics.optimization_score
            },
            'resource_pools': pool_metrics,
            'optimization_history_count': len(self.optimization_history),
            'strategy': self.strategy.value
        }


class ResourceExhaustedException(Exception):
    """Exception raised when resource pool is exhausted."""
    pass


# Global resource manager instance
_resource_manager = None

def get_resource_manager() -> IntelligentResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = IntelligentResourceManager()
    return _resource_manager


if __name__ == "__main__":
    async def test_resource_management():
        """Test the intelligent resource management system."""
        manager = get_resource_manager()
        
        try:
            await manager.start_management()
            
            # Let it run for a while to collect data
            await asyncio.sleep(60)
            
            # Generate report
            report = manager.get_system_resource_report()
            print(json.dumps(report, indent=2))
            
        finally:
            await manager.stop_management()
    
    asyncio.run(test_resource_management())