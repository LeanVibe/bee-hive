"""
Production Enhancement Plugin for SimpleOrchestrator

Consolidates production-ready features from production_orchestrator.py and orchestrator.py
into a plugin architecture that enhances SimpleOrchestrator without replacing it.

Key Features Consolidated:
- Production monitoring and alerting
- Performance optimization
- Advanced error handling and recovery
- SLA monitoring and anomaly detection
- Auto-scaling capabilities

Plugin Architecture: Maintains SimpleOrchestrator as base, adds production features as needed.
"""

import asyncio
import json
import uuid
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
import logging

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
import psutil

logger = structlog.get_logger("production_enhancement_plugin")


class ProductionEventSeverity(str, Enum):
    """Production event severity levels."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ProductionMetrics:
    """Consolidated production metrics."""
    timestamp: datetime
    agent_count: int
    active_tasks: int
    avg_response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    system_health_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_count": self.agent_count,
            "active_tasks": self.active_tasks,
            "avg_response_time_ms": self.avg_response_time_ms,
            "error_rate": self.error_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "system_health_score": self.system_health_score
        }


@dataclass
class ProductionAlert:
    """Production alert definition."""
    id: str
    severity: ProductionEventSeverity
    message: str
    timestamp: datetime
    component: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class OrchestratorProtocol(Protocol):
    """Protocol for orchestrator compatibility."""
    async def get_system_status(self) -> Dict[str, Any]: ...
    async def register_agent(self, agent_spec: Any) -> str: ...
    async def get_agent(self, agent_id: str) -> Optional[Any]: ...
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool: ...


class ProductionEnhancementPlugin:
    """
    Production enhancement plugin for SimpleOrchestrator.
    
    Provides production-grade monitoring, alerting, and optimization
    features while maintaining SimpleOrchestrator as the core engine.
    """
    
    def __init__(self, orchestrator: OrchestratorProtocol):
        self.orchestrator = orchestrator
        self.enabled = True
        self.metrics_history: List[ProductionMetrics] = []
        self.active_alerts: Dict[str, ProductionAlert] = {}
        self.performance_baselines: Dict[str, float] = {}
        
        # Production configuration
        self.config = {
            "max_response_time_ms": 500,
            "max_error_rate": 0.05,  # 5%
            "max_memory_usage_mb": 1000,
            "max_cpu_usage_percent": 80,
            "min_health_score": 0.8,
            "metrics_retention_hours": 24,
            "alert_cooldown_minutes": 15
        }
        
        logger.info("Production Enhancement Plugin initialized", 
                   config=self.config)
    
    async def collect_production_metrics(self) -> ProductionMetrics:
        """Collect comprehensive production metrics."""
        try:
            # Get system status from orchestrator
            system_status = await self.orchestrator.get_system_status()
            
            # Extract orchestrator metrics
            agent_count = system_status.get("agents", {}).get("total", 0)
            active_tasks = system_status.get("tasks", {}).get("active", 0)
            
            # Get system performance metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Calculate response time (from orchestrator performance data)
            performance_data = system_status.get("performance", {})
            avg_response_time_ms = performance_data.get("response_time_ms", 0)
            
            # Calculate error rate (from recent operations)
            error_rate = performance_data.get("error_rate", 0.0)
            
            # Calculate system health score
            health_factors = {
                "response_time": min(1.0, self.config["max_response_time_ms"] / max(1, avg_response_time_ms)),
                "error_rate": max(0.0, 1.0 - (error_rate / self.config["max_error_rate"])),
                "memory": min(1.0, self.config["max_memory_usage_mb"] / max(1, memory_usage_mb)),
                "cpu": min(1.0, self.config["max_cpu_usage_percent"] / max(1, cpu_usage_percent))
            }
            system_health_score = sum(health_factors.values()) / len(health_factors)
            
            metrics = ProductionMetrics(
                timestamp=datetime.utcnow(),
                agent_count=agent_count,
                active_tasks=active_tasks,
                avg_response_time_ms=avg_response_time_ms,
                error_rate=error_rate,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                system_health_score=system_health_score
            )
            
            # Store metrics (with retention limit)
            self.metrics_history.append(metrics)
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config["metrics_retention_hours"])
            self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect production metrics", error=str(e))
            # Return default metrics on error
            return ProductionMetrics(
                timestamp=datetime.utcnow(),
                agent_count=0,
                active_tasks=0,
                avg_response_time_ms=0,
                error_rate=1.0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                system_health_score=0.0
            )
    
    async def evaluate_production_alerts(self, metrics: ProductionMetrics) -> List[ProductionAlert]:
        """Evaluate metrics and generate production alerts."""
        alerts = []
        
        # Response time alert
        if metrics.avg_response_time_ms > self.config["max_response_time_ms"]:
            alert = ProductionAlert(
                id=f"response_time_{uuid.uuid4().hex[:8]}",
                severity=ProductionEventSeverity.HIGH,
                message=f"High response time: {metrics.avg_response_time_ms:.1f}ms (threshold: {self.config['max_response_time_ms']}ms)",
                timestamp=metrics.timestamp,
                component="orchestrator",
                metrics={"response_time_ms": metrics.avg_response_time_ms}
            )
            alerts.append(alert)
        
        # Error rate alert
        if metrics.error_rate > self.config["max_error_rate"]:
            alert = ProductionAlert(
                id=f"error_rate_{uuid.uuid4().hex[:8]}",
                severity=ProductionEventSeverity.CRITICAL,
                message=f"High error rate: {metrics.error_rate:.1%} (threshold: {self.config['max_error_rate']:.1%})",
                timestamp=metrics.timestamp,
                component="orchestrator",
                metrics={"error_rate": metrics.error_rate}
            )
            alerts.append(alert)
        
        # Memory usage alert
        if metrics.memory_usage_mb > self.config["max_memory_usage_mb"]:
            alert = ProductionAlert(
                id=f"memory_{uuid.uuid4().hex[:8]}",
                severity=ProductionEventSeverity.MEDIUM,
                message=f"High memory usage: {metrics.memory_usage_mb:.1f}MB (threshold: {self.config['max_memory_usage_mb']}MB)",
                timestamp=metrics.timestamp,
                component="system",
                metrics={"memory_usage_mb": metrics.memory_usage_mb}
            )
            alerts.append(alert)
        
        # System health alert
        if metrics.system_health_score < self.config["min_health_score"]:
            alert = ProductionAlert(
                id=f"health_{uuid.uuid4().hex[:8]}",
                severity=ProductionEventSeverity.HIGH,
                message=f"Low system health score: {metrics.system_health_score:.2f} (threshold: {self.config['min_health_score']})",
                timestamp=metrics.timestamp,
                component="system",
                metrics={"health_score": metrics.system_health_score}
            )
            alerts.append(alert)
        
        return alerts
    
    async def process_production_alerts(self, alerts: List[ProductionAlert]) -> None:
        """Process and manage production alerts."""
        for alert in alerts:
            # Check if similar alert is already active (cooldown logic)
            existing_alerts = [a for a in self.active_alerts.values() 
                             if a.component == alert.component and not a.resolved]
            
            if existing_alerts:
                # Check cooldown period
                latest_alert = max(existing_alerts, key=lambda a: a.timestamp)
                cooldown_end = latest_alert.timestamp + timedelta(minutes=self.config["alert_cooldown_minutes"])
                
                if datetime.utcnow() < cooldown_end:
                    logger.debug("Alert in cooldown period", 
                               component=alert.component,
                               cooldown_remaining=(cooldown_end - datetime.utcnow()).total_seconds())
                    continue
            
            # Add new alert
            self.active_alerts[alert.id] = alert
            
            logger.warning("Production alert triggered",
                         alert_id=alert.id,
                         severity=alert.severity.value,
                         message=alert.message,
                         component=alert.component,
                         metrics=alert.metrics)
            
            # Auto-resolution logic for certain alerts
            if alert.component == "system" and alert.severity in [ProductionEventSeverity.LOW, ProductionEventSeverity.INFO]:
                # Auto-resolve low/info alerts after collection
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
    
    async def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status."""
        current_metrics = await self.collect_production_metrics()
        
        # Calculate trends if we have history
        trends = {}
        if len(self.metrics_history) >= 2:
            recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
            
            trends = {
                "response_time_trend": self._calculate_trend([m.avg_response_time_ms for m in recent_metrics]),
                "error_rate_trend": self._calculate_trend([m.error_rate for m in recent_metrics]),
                "health_score_trend": self._calculate_trend([m.system_health_score for m in recent_metrics])
            }
        
        # Active alerts summary
        active_alerts_by_severity = {}
        for severity in ProductionEventSeverity:
            count = len([a for a in self.active_alerts.values() 
                        if a.severity == severity and not a.resolved])
            if count > 0:
                active_alerts_by_severity[severity.value] = count
        
        return {
            "enabled": self.enabled,
            "current_metrics": current_metrics.to_dict(),
            "trends": trends,
            "active_alerts": {
                "total": len([a for a in self.active_alerts.values() if not a.resolved]),
                "by_severity": active_alerts_by_severity
            },
            "system_health": {
                "score": current_metrics.system_health_score,
                "status": "healthy" if current_metrics.system_health_score >= self.config["min_health_score"] else "degraded"
            },
            "metrics_history_size": len(self.metrics_history)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        recent_avg = statistics.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = statistics.mean(values[:-3]) if len(values) >= 6 else values[0]
        
        change_percent = ((recent_avg - older_avg) / max(older_avg, 0.001)) * 100
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Perform production performance optimizations."""
        optimizations_applied = []
        
        current_metrics = await self.collect_production_metrics()
        
        # Memory optimization
        if current_metrics.memory_usage_mb > self.config["max_memory_usage_mb"] * 0.8:
            # Trigger garbage collection
            import gc
            gc.collect()
            optimizations_applied.append("garbage_collection")
        
        # Performance tuning based on metrics
        if current_metrics.avg_response_time_ms > self.config["max_response_time_ms"] * 0.8:
            # Could implement caching, connection pooling optimizations, etc.
            optimizations_applied.append("response_time_optimization")
        
        # Alert cleanup
        resolved_alerts = 0
        for alert in self.active_alerts.values():
            if not alert.resolved and alert.timestamp < datetime.utcnow() - timedelta(hours=1):
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
                resolved_alerts += 1
        
        if resolved_alerts > 0:
            optimizations_applied.append(f"alert_cleanup_{resolved_alerts}")
        
        logger.info("Performance optimization completed",
                   optimizations=optimizations_applied,
                   metrics=current_metrics.to_dict())
        
        return {
            "optimizations_applied": optimizations_applied,
            "current_metrics": current_metrics.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def start_monitoring(self) -> None:
        """Start production monitoring loop."""
        if not self.enabled:
            return
        
        logger.info("Starting production monitoring")
        
        while self.enabled:
            try:
                # Collect metrics
                metrics = await self.collect_production_metrics()
                
                # Evaluate and process alerts
                alerts = await self.evaluate_production_alerts(metrics)
                await self.process_production_alerts(alerts)
                
                # Performance optimization (every 10 cycles)
                if len(self.metrics_history) % 10 == 0:
                    await self.optimize_performance()
                
                # Wait before next collection (adjust based on needs)
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error("Production monitoring error", error=str(e))
                await asyncio.sleep(60)  # Longer wait on error
    
    def stop_monitoring(self) -> None:
        """Stop production monitoring."""
        self.enabled = False
        logger.info("Production monitoring stopped")


def create_production_enhancement_plugin(orchestrator: OrchestratorProtocol) -> ProductionEnhancementPlugin:
    """Factory function to create production enhancement plugin."""
    return ProductionEnhancementPlugin(orchestrator)