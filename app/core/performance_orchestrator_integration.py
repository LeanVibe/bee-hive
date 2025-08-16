"""
Performance Monitor Integration with Orchestrator and Task Engine

Integrates the unified performance monitoring system with:
- Production orchestrator for agent performance tracking
- Task execution engine for task performance monitoring
- Real-time performance optimization and scaling decisions
- Intelligent load balancing based on performance metrics
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import structlog

from .performance_monitor import (
    get_performance_monitor,
    record_orchestrator_metrics,
    record_task_engine_metrics,
    record_agent_spawn_time,
    monitor_performance
)

logger = structlog.get_logger()


@dataclass
class OrchestrationPerformanceMetrics:
    """Performance metrics for orchestration operations"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    active_agents: int = 0
    pending_tasks: int = 0
    completed_tasks_last_hour: int = 0
    failed_tasks_last_hour: int = 0
    average_task_execution_time: float = 0.0
    average_agent_spawn_time: float = 0.0
    orchestration_load: float = 0.0
    resource_utilization: float = 0.0
    scaling_events: int = 0


@dataclass
class TaskEnginePerformanceMetrics:
    """Performance metrics for task engine operations"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tasks_in_queue: int = 0
    tasks_executing: int = 0
    tasks_completed_per_minute: float = 0.0
    average_queue_wait_time: float = 0.0
    average_execution_time: float = 0.0
    task_success_rate: float = 0.0
    engine_throughput: float = 0.0
    error_rate: float = 0.0


class PerformanceOrchestrator:
    """
    Integrates performance monitoring with orchestrator operations
    Provides real-time performance tracking and optimization
    """
    
    def __init__(self):
        self.performance_monitor = get_performance_monitor()
        self.orchestration_metrics = OrchestrationPerformanceMetrics()
        self.performance_callbacks: List[Callable] = []
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance optimization thresholds
        self.performance_thresholds = {
            "high_load_threshold": 0.8,
            "scale_up_threshold": 0.85,
            "scale_down_threshold": 0.3,
            "max_response_time_ms": 5000,
            "max_queue_size": 100,
            "min_success_rate": 0.95
        }
        
        logger.info("Performance orchestrator integration initialized")
    
    async def start_monitoring(self):
        """Start orchestration performance monitoring"""
        if not self._monitoring_active:
            self._monitoring_task = asyncio.create_task(self._orchestration_monitoring_loop())
            self._monitoring_active = True
            logger.info("Orchestration performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop orchestration performance monitoring"""
        self._monitoring_active = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Orchestration performance monitoring stopped")
    
    async def _orchestration_monitoring_loop(self):
        """Main orchestration monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect orchestration metrics
                await self._collect_orchestration_metrics()
                
                # Analyze performance and make recommendations
                await self._analyze_orchestration_performance()
                
                # Check for scaling opportunities
                await self._check_scaling_conditions()
                
                # Record metrics to unified monitor
                self._record_orchestration_metrics()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error("Orchestration monitoring error", error=str(e))
                await asyncio.sleep(10)
    
    async def _collect_orchestration_metrics(self):
        """Collect current orchestration performance metrics"""
        try:
            # Simulate collecting metrics from orchestrator
            # In real implementation, would interface with actual orchestrator
            
            current_time = datetime.utcnow()
            
            # Simulate realistic orchestration metrics
            import random
            
            self.orchestration_metrics = OrchestrationPerformanceMetrics(
                timestamp=current_time,
                active_agents=random.randint(5, 15),
                pending_tasks=random.randint(0, 25),
                completed_tasks_last_hour=random.randint(50, 200),
                failed_tasks_last_hour=random.randint(0, 10),
                average_task_execution_time=random.uniform(30.0, 300.0),
                average_agent_spawn_time=random.uniform(3.0, 12.0),
                orchestration_load=random.uniform(0.2, 0.9),
                resource_utilization=random.uniform(0.4, 0.8),
                scaling_events=random.randint(0, 3)
            )
            
        except Exception as e:
            logger.error("Failed to collect orchestration metrics", error=str(e))
    
    async def _analyze_orchestration_performance(self):
        """Analyze orchestration performance and generate insights"""
        metrics = self.orchestration_metrics
        
        # Calculate derived metrics
        success_rate = 1.0
        if metrics.completed_tasks_last_hour + metrics.failed_tasks_last_hour > 0:
            success_rate = metrics.completed_tasks_last_hour / (
                metrics.completed_tasks_last_hour + metrics.failed_tasks_last_hour
            )
        
        # Analyze performance trends
        performance_issues = []
        
        if metrics.average_agent_spawn_time > 10.0:
            performance_issues.append("Slow agent spawn times detected")
        
        if metrics.average_task_execution_time > 180.0:
            performance_issues.append("High task execution times detected")
        
        if success_rate < self.performance_thresholds["min_success_rate"]:
            performance_issues.append("Low task success rate detected")
        
        if metrics.orchestration_load > self.performance_thresholds["high_load_threshold"]:
            performance_issues.append("High orchestration load detected")
        
        if performance_issues:
            logger.warning("Orchestration performance issues detected", issues=performance_issues)
            
            # Trigger performance callbacks
            for callback in self.performance_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(metrics, performance_issues)
                    else:
                        callback(metrics, performance_issues)
                except Exception as e:
                    logger.error("Performance callback failed", error=str(e))
    
    async def _check_scaling_conditions(self):
        """Check if scaling actions are needed"""
        metrics = self.orchestration_metrics
        
        scaling_recommendations = []
        
        # Scale up conditions
        if (metrics.orchestration_load > self.performance_thresholds["scale_up_threshold"] and
            metrics.pending_tasks > 20):
            scaling_recommendations.append({
                "action": "scale_up",
                "reason": "High load and pending tasks",
                "current_agents": metrics.active_agents,
                "recommended_agents": min(metrics.active_agents + 3, 20)
            })
        
        # Scale down conditions  
        if (metrics.orchestration_load < self.performance_thresholds["scale_down_threshold"] and
            metrics.pending_tasks < 5 and
            metrics.active_agents > 3):
            scaling_recommendations.append({
                "action": "scale_down",
                "reason": "Low load and few pending tasks",
                "current_agents": metrics.active_agents,
                "recommended_agents": max(metrics.active_agents - 2, 3)
            })
        
        if scaling_recommendations:
            logger.info("Scaling recommendations generated", recommendations=scaling_recommendations)
            
            # Record scaling recommendations as metrics
            for rec in scaling_recommendations:
                self.performance_monitor.record_metric(
                    f"scaling_recommendation_{rec['action']}", 
                    rec["recommended_agents"]
                )
    
    def _record_orchestration_metrics(self):
        """Record orchestration metrics to unified performance monitor"""
        metrics = self.orchestration_metrics
        
        # Record to unified monitor
        record_orchestrator_metrics(
            agent_count=metrics.active_agents,
            task_queue_size=metrics.pending_tasks,
            load_factor=metrics.orchestration_load
        )
        
        # Record additional metrics
        self.performance_monitor.record_metric("orchestrator_resource_utilization", metrics.resource_utilization)
        self.performance_monitor.record_metric("orchestrator_completed_tasks_hour", metrics.completed_tasks_last_hour)
        self.performance_monitor.record_metric("orchestrator_failed_tasks_hour", metrics.failed_tasks_last_hour)
        
        # Record timing metrics
        record_agent_spawn_time(metrics.average_agent_spawn_time)
        record_task_execution_time("orchestrated_task", metrics.average_task_execution_time)
    
    def add_performance_callback(self, callback: Callable):
        """Add callback for performance events"""
        self.performance_callbacks.append(callback)
    
    def remove_performance_callback(self, callback: Callable):
        """Remove performance callback"""
        if callback in self.performance_callbacks:
            self.performance_callbacks.remove(callback)
    
    def get_orchestration_health(self) -> Dict[str, Any]:
        """Get current orchestration health summary"""
        metrics = self.orchestration_metrics
        
        # Calculate health score
        health_factors = []
        
        # Load factor health
        if metrics.orchestration_load < 0.7:
            health_factors.append(1.0)
        elif metrics.orchestration_load < 0.85:
            health_factors.append(0.7)
        else:
            health_factors.append(0.3)
        
        # Response time health
        if metrics.average_task_execution_time < 120:
            health_factors.append(1.0)
        elif metrics.average_task_execution_time < 300:
            health_factors.append(0.7)
        else:
            health_factors.append(0.3)
        
        # Agent spawn time health
        if metrics.average_agent_spawn_time < 8:
            health_factors.append(1.0)
        elif metrics.average_agent_spawn_time < 15:
            health_factors.append(0.7)
        else:
            health_factors.append(0.3)
        
        overall_health = sum(health_factors) / len(health_factors)
        
        if overall_health >= 0.8:
            status = "healthy"
        elif overall_health >= 0.6:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": overall_health,
            "metrics": {
                "active_agents": metrics.active_agents,
                "pending_tasks": metrics.pending_tasks,
                "orchestration_load": metrics.orchestration_load,
                "average_task_execution_time": metrics.average_task_execution_time,
                "average_agent_spawn_time": metrics.average_agent_spawn_time
            },
            "last_updated": metrics.timestamp.isoformat()
        }


class PerformanceTaskEngine:
    """
    Integrates performance monitoring with task engine operations
    Provides task execution performance tracking and optimization
    """
    
    def __init__(self):
        self.performance_monitor = get_performance_monitor()
        self.task_metrics = TaskEnginePerformanceMetrics()
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Task performance tracking
        self.task_execution_times: Dict[str, List[float]] = {}
        self.task_queue_history: List[int] = []
        
        logger.info("Task engine performance integration initialized")
    
    async def start_monitoring(self):
        """Start task engine performance monitoring"""
        if not self._monitoring_active:
            self._monitoring_task = asyncio.create_task(self._task_monitoring_loop())
            self._monitoring_active = True
            logger.info("Task engine performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop task engine performance monitoring"""
        self._monitoring_active = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Task engine performance monitoring stopped")
    
    async def _task_monitoring_loop(self):
        """Main task engine monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect task engine metrics
                await self._collect_task_metrics()
                
                # Analyze task performance
                await self._analyze_task_performance()
                
                # Record metrics to unified monitor
                self._record_task_metrics()
                
                await asyncio.sleep(15)  # Monitor every 15 seconds
                
            except Exception as e:
                logger.error("Task engine monitoring error", error=str(e))
                await asyncio.sleep(5)
    
    async def _collect_task_metrics(self):
        """Collect current task engine performance metrics"""
        try:
            # Simulate collecting metrics from task engine
            # In real implementation, would interface with actual task engine
            
            current_time = datetime.utcnow()
            
            # Simulate realistic task engine metrics
            import random
            
            tasks_in_queue = random.randint(0, 15)
            tasks_executing = random.randint(1, 8)
            
            self.task_metrics = TaskEnginePerformanceMetrics(
                timestamp=current_time,
                tasks_in_queue=tasks_in_queue,
                tasks_executing=tasks_executing,
                tasks_completed_per_minute=random.uniform(5.0, 25.0),
                average_queue_wait_time=random.uniform(1.0, 30.0),
                average_execution_time=random.uniform(15.0, 180.0),
                task_success_rate=random.uniform(0.85, 0.99),
                engine_throughput=random.uniform(10.0, 50.0),
                error_rate=random.uniform(0.01, 0.15)
            )
            
            # Track queue history
            self.task_queue_history.append(tasks_in_queue)
            if len(self.task_queue_history) > 60:  # Keep last hour
                self.task_queue_history.pop(0)
            
        except Exception as e:
            logger.error("Failed to collect task engine metrics", error=str(e))
    
    async def _analyze_task_performance(self):
        """Analyze task engine performance"""
        metrics = self.task_metrics
        
        # Analyze performance issues
        issues = []
        
        if metrics.average_queue_wait_time > 60.0:
            issues.append("High queue wait times detected")
        
        if metrics.task_success_rate < 0.9:
            issues.append("Low task success rate detected")
        
        if metrics.error_rate > 0.1:
            issues.append("High task error rate detected")
        
        if metrics.tasks_in_queue > 20:
            issues.append("Task queue backlog detected")
        
        if issues:
            logger.warning("Task engine performance issues detected", issues=issues)
        
        # Analyze queue trends
        if len(self.task_queue_history) > 10:
            recent_avg = sum(self.task_queue_history[-10:]) / 10
            older_avg = sum(self.task_queue_history[-20:-10]) / 10 if len(self.task_queue_history) > 20 else recent_avg
            
            if recent_avg > older_avg * 1.5:
                logger.warning("Task queue growing rapidly", 
                             recent_avg=recent_avg, 
                             older_avg=older_avg)
    
    def _record_task_metrics(self):
        """Record task engine metrics to unified performance monitor"""
        metrics = self.task_metrics
        
        # Record to unified monitor
        record_task_engine_metrics(
            tasks_executed=int(metrics.tasks_completed_per_minute),
            avg_execution_time=metrics.average_execution_time,
            active_executions=metrics.tasks_executing
        )
        
        # Record additional metrics
        self.performance_monitor.record_metric("task_engine_queue_size", metrics.tasks_in_queue)
        self.performance_monitor.record_metric("task_engine_wait_time", metrics.average_queue_wait_time)
        self.performance_monitor.record_metric("task_engine_success_rate", metrics.task_success_rate)
        self.performance_monitor.record_metric("task_engine_error_rate", metrics.error_rate)
        self.performance_monitor.record_metric("task_engine_throughput", metrics.engine_throughput)
    
    @monitor_performance("task_execution")
    async def track_task_execution(self, task_type: str, task_function: Callable, *args, **kwargs):
        """Track individual task execution performance"""
        start_time = time.time()
        
        try:
            # Execute task
            if asyncio.iscoroutinefunction(task_function):
                result = await task_function(*args, **kwargs)
            else:
                result = task_function(*args, **kwargs)
            
            # Record successful execution
            execution_time = time.time() - start_time
            
            if task_type not in self.task_execution_times:
                self.task_execution_times[task_type] = []
            
            self.task_execution_times[task_type].append(execution_time)
            
            # Keep only recent executions
            if len(self.task_execution_times[task_type]) > 100:
                self.task_execution_times[task_type] = self.task_execution_times[task_type][-100:]
            
            # Record timing
            record_task_execution_time(task_type, execution_time)
            
            logger.debug("Task execution tracked", 
                        task_type=task_type, 
                        execution_time=execution_time)
            
            return result
            
        except Exception as e:
            # Record failed execution
            execution_time = time.time() - start_time
            
            self.performance_monitor.record_counter(f"task_execution_errors_{task_type}")
            self.performance_monitor.record_timing(f"task_execution_failed_{task_type}", execution_time * 1000)
            
            logger.error("Task execution failed", 
                        task_type=task_type, 
                        execution_time=execution_time,
                        error=str(e))
            
            raise
    
    def get_task_performance_summary(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get task performance summary"""
        if task_type and task_type in self.task_execution_times:
            # Specific task type summary
            execution_times = self.task_execution_times[task_type]
            
            if execution_times:
                import statistics
                return {
                    "task_type": task_type,
                    "executions": len(execution_times),
                    "average_time": statistics.mean(execution_times),
                    "min_time": min(execution_times),
                    "max_time": max(execution_times),
                    "median_time": statistics.median(execution_times),
                    "recent_average": statistics.mean(execution_times[-10:]) if len(execution_times) >= 10 else statistics.mean(execution_times)
                }
            else:
                return {"task_type": task_type, "executions": 0}
        
        else:
            # Overall task engine summary
            return {
                "timestamp": self.task_metrics.timestamp.isoformat(),
                "tasks_in_queue": self.task_metrics.tasks_in_queue,
                "tasks_executing": self.task_metrics.tasks_executing,
                "tasks_completed_per_minute": self.task_metrics.tasks_completed_per_minute,
                "average_execution_time": self.task_metrics.average_execution_time,
                "task_success_rate": self.task_metrics.task_success_rate,
                "error_rate": self.task_metrics.error_rate,
                "tracked_task_types": list(self.task_execution_times.keys())
            }


class PerformanceIntegrationManager:
    """
    Manages integration between performance monitoring, orchestrator, and task engine
    Provides unified interface for performance tracking across all systems
    """
    
    def __init__(self):
        self.performance_monitor = get_performance_monitor()
        self.orchestrator_integration = PerformanceOrchestrator()
        self.task_engine_integration = PerformanceTaskEngine()
        
        # Set up cross-system callbacks
        self.orchestrator_integration.add_performance_callback(self._handle_orchestrator_performance)
        
        logger.info("Performance integration manager initialized")
    
    async def start_all_monitoring(self):
        """Start all performance monitoring integrations"""
        await self.performance_monitor.start_monitoring()
        await self.orchestrator_integration.start_monitoring()
        await self.task_engine_integration.start_monitoring()
        
        logger.info("All performance monitoring integrations started")
    
    async def stop_all_monitoring(self):
        """Stop all performance monitoring integrations"""
        await self.performance_monitor.stop_monitoring()
        await self.orchestrator_integration.stop_monitoring()
        await self.task_engine_integration.stop_monitoring()
        
        logger.info("All performance monitoring integrations stopped")
    
    async def _handle_orchestrator_performance(self, metrics: OrchestrationPerformanceMetrics, issues: List[str]):
        """Handle orchestrator performance events"""
        # Log performance issues
        if issues:
            logger.warning("Orchestrator performance issues require attention", 
                          issues=issues,
                          active_agents=metrics.active_agents,
                          load=metrics.orchestration_load)
        
        # Trigger additional monitoring if needed
        if metrics.orchestration_load > 0.9:
            logger.critical("Critical orchestration load detected - immediate attention required")
    
    def get_comprehensive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary across all systems"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "unified_monitor": self.performance_monitor.get_system_health_summary(),
            "orchestrator": self.orchestrator_integration.get_orchestration_health(),
            "task_engine": self.task_engine_integration.get_task_performance_summary(),
            "integration_status": {
                "orchestrator_monitoring": self.orchestrator_integration._monitoring_active,
                "task_engine_monitoring": self.task_engine_integration._monitoring_active,
                "unified_monitoring": self.performance_monitor._monitoring_active
            }
        }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all integrated systems"""
        start_time = time.time()
        
        # Run unified monitor benchmark
        monitor_benchmark = await self.performance_monitor.run_performance_benchmark()
        
        # Get current performance state
        comprehensive_summary = self.get_comprehensive_performance_summary()
        
        benchmark_time = time.time() - start_time
        
        return {
            "benchmark_duration": benchmark_time,
            "unified_monitor_benchmark": monitor_benchmark,
            "comprehensive_summary": comprehensive_summary,
            "integration_health": {
                "all_systems_healthy": (
                    comprehensive_summary["unified_monitor"]["status"] == "healthy" and
                    comprehensive_summary["orchestrator"]["status"] == "healthy"
                ),
                "performance_score": monitor_benchmark.get("overall_score", 0)
            }
        }


# Global integration manager instance
_integration_manager: Optional[PerformanceIntegrationManager] = None


def get_performance_integration_manager() -> PerformanceIntegrationManager:
    """Get global performance integration manager instance"""
    global _integration_manager
    
    if _integration_manager is None:
        _integration_manager = PerformanceIntegrationManager()
    
    return _integration_manager


async def start_integrated_performance_monitoring():
    """Start all integrated performance monitoring"""
    manager = get_performance_integration_manager()
    await manager.start_all_monitoring()


async def stop_integrated_performance_monitoring():
    """Stop all integrated performance monitoring"""
    manager = get_performance_integration_manager()
    await manager.stop_all_monitoring()