"""
Observability System Orchestrator
=================================

Central orchestrator that coordinates all observability components for LeanVibe
Agent Hive 2.0. Provides unified management, performance optimization insights,
and enterprise-grade observability capabilities.

Features:
- Unified observability system management
- Performance optimization insights and pattern analysis
- Automatic system health monitoring and recovery
- Enterprise KPI tracking and reporting
- Integration with all observability components
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import structlog

# Import all observability components
from app.observability.real_time_hooks import (
    get_real_time_processor, 
    shutdown_real_time_processor,
    get_processor_health
)
from app.observability.enhanced_websocket_streaming import (
    get_enhanced_websocket_streaming,
    shutdown_enhanced_websocket_streaming
)
from app.observability.enhanced_prometheus_integration import (
    get_enhanced_prometheus_metrics,
    start_enhanced_metrics_collection
)
from app.observability.intelligent_alerting_system import (
    get_intelligent_alerting_system,
    shutdown_intelligent_alerting_system
)
from app.observability.predictive_analytics_engine import (
    get_predictive_analytics_engine,
    shutdown_predictive_analytics_engine
)

logger = structlog.get_logger()


class SystemHealthStatus(str, Enum):
    """Overall system health status levels."""
    HEALTHY = "healthy"           # All systems operating within targets
    DEGRADED = "degraded"         # Some performance issues but functional
    CRITICAL = "critical"         # Major issues requiring immediate attention
    UNKNOWN = "unknown"           # Health status cannot be determined


class OptimizationCategory(str, Enum):
    """Categories of performance optimization insights."""
    LATENCY = "latency"              # Response time optimizations
    THROUGHPUT = "throughput"        # Processing capacity optimizations  
    RESOURCE = "resource"            # CPU/memory efficiency optimizations
    RELIABILITY = "reliability"     # Error rate and availability improvements
    SCALABILITY = "scalability"     # System scaling recommendations
    COST = "cost"                    # Cost optimization opportunities


@dataclass
class PerformanceInsight:
    """Performance optimization insight with actionable recommendations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: OptimizationCategory
    title: str
    description: str
    
    # Metrics and analysis
    current_performance: Dict[str, float] = field(default_factory=dict)
    target_performance: Dict[str, float] = field(default_factory=dict)
    improvement_potential: float = 0.0  # 0-100% potential improvement
    
    # Implementation guidance
    optimization_actions: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    expected_impact: str = ""
    implementation_priority: int = 5  # 1-10, higher is more urgent
    
    # Supporting data
    analysis_period: timedelta = field(default_factory=lambda: timedelta(hours=1))
    confidence_score: float = 0.0  # 0-1.0
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    implemented: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "current_performance": self.current_performance,
            "target_performance": self.target_performance,
            "improvement_potential": self.improvement_potential,
            "optimization_actions": self.optimization_actions,
            "estimated_effort": self.estimated_effort,
            "expected_impact": self.expected_impact,
            "implementation_priority": self.implementation_priority,
            "confidence_score": self.confidence_score,
            "analysis_period_hours": self.analysis_period.total_seconds() / 3600,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "implemented": self.implemented,
            "supporting_metrics": self.supporting_metrics
        }


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_status: SystemHealthStatus = SystemHealthStatus.UNKNOWN
    overall_score: float = 0.0  # 0-100
    
    # Component health scores
    component_health: Dict[str, float] = field(default_factory=dict)
    
    # Performance target compliance
    target_compliance: Dict[str, bool] = field(default_factory=dict)
    sli_scores: Dict[str, float] = field(default_factory=dict)
    
    # System metrics
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Issues and recommendations
    active_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Performance insights
    insights: List[PerformanceInsight] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "component_health": self.component_health,
            "target_compliance": self.target_compliance,
            "sli_scores": self.sli_scores,
            "system_metrics": self.system_metrics,
            "active_issues": self.active_issues,
            "recommendations": self.recommendations,
            "insights": [insight.to_dict() for insight in self.insights]
        }


class PerformanceOptimizationAnalyzer:
    """Analyzes system performance and generates optimization insights."""
    
    def __init__(self):
        self.historical_insights = {}
        self.optimization_patterns = {}
    
    async def analyze_system_performance(
        self,
        component_metrics: Dict[str, Any],
        system_health: Dict[str, Any]
    ) -> List[PerformanceInsight]:
        """
        Analyze system performance and generate optimization insights.
        
        Args:
            component_metrics: Metrics from all observability components
            system_health: Overall system health information
            
        Returns:
            List of performance optimization insights
        """
        insights = []
        
        try:
            # Analyze latency performance
            latency_insights = await self._analyze_latency_performance(component_metrics)
            insights.extend(latency_insights)
            
            # Analyze throughput performance
            throughput_insights = await self._analyze_throughput_performance(component_metrics)
            insights.extend(throughput_insights)
            
            # Analyze resource utilization
            resource_insights = await self._analyze_resource_utilization(component_metrics)
            insights.extend(resource_insights)
            
            # Analyze reliability metrics
            reliability_insights = await self._analyze_reliability_metrics(component_metrics)
            insights.extend(reliability_insights)
            
            # Analyze scalability opportunities
            scalability_insights = await self._analyze_scalability_opportunities(component_metrics, system_health)
            insights.extend(scalability_insights)
            
            # Prioritize insights by impact and urgency
            insights = await self._prioritize_insights(insights)
            
            logger.info(f"Generated {len(insights)} performance optimization insights")
            
            return insights
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return []
    
    async def _analyze_latency_performance(self, metrics: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze latency performance and generate optimization insights."""
        insights = []
        
        try:
            # Event processing latency analysis
            event_latency_p95 = metrics.get("event_processor", {}).get("p95_processing_latency_ms", 0)
            websocket_latency_p95 = metrics.get("websocket_streaming", {}).get("average_stream_latency_ms", 0)
            
            # Check event processing latency
            if event_latency_p95 > 150:  # Above target
                severity = "high" if event_latency_p95 > 200 else "medium"
                improvement_potential = min(50.0, ((event_latency_p95 - 100) / event_latency_p95) * 100)
                
                insight = PerformanceInsight(
                    category=OptimizationCategory.LATENCY,
                    title="Event Processing Latency Optimization",
                    description=f"P95 event processing latency is {event_latency_p95:.1f}ms, "
                               f"exceeding the 150ms target by {event_latency_p95 - 150:.1f}ms.",
                    current_performance={"p95_latency_ms": event_latency_p95},
                    target_performance={"p95_latency_ms": 100.0},
                    improvement_potential=improvement_potential,
                    optimization_actions=[
                        "Optimize event serialization/deserialization",
                        "Increase event processing worker threads",
                        "Implement event batching optimizations",
                        "Review database query performance",
                        "Consider async I/O improvements"
                    ],
                    estimated_effort="medium",
                    expected_impact=f"Reduce latency by {improvement_potential:.1f}%",
                    implementation_priority=8 if severity == "high" else 6,
                    confidence_score=0.85,
                    supporting_metrics={
                        "current_p95": event_latency_p95,
                        "target_p95": 150,
                        "samples_analyzed": metrics.get("event_processor", {}).get("total_events_processed", 0)
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=6)
                )
                insights.append(insight)
            
            # Check WebSocket streaming latency
            if websocket_latency_p95 > 1000:  # Above 1s target
                improvement_potential = min(40.0, ((websocket_latency_p95 - 500) / websocket_latency_p95) * 100)
                
                insight = PerformanceInsight(
                    category=OptimizationCategory.LATENCY,
                    title="WebSocket Streaming Latency Optimization",
                    description=f"WebSocket streaming latency is {websocket_latency_p95:.1f}ms, "
                               f"exceeding 1s target. Impacts real-time dashboard updates.",
                    current_performance={"avg_stream_latency_ms": websocket_latency_p95},
                    target_performance={"avg_stream_latency_ms": 500.0},
                    improvement_potential=improvement_potential,
                    optimization_actions=[
                        "Optimize WebSocket message serialization",
                        "Implement connection pooling",
                        "Reduce event payload sizes",
                        "Optimize event filtering algorithms",
                        "Consider compression for large payloads"
                    ],
                    estimated_effort="low",
                    expected_impact=f"Reduce streaming latency by {improvement_potential:.1f}%",
                    implementation_priority=7,
                    confidence_score=0.80,
                    supporting_metrics={
                        "current_latency": websocket_latency_p95,
                        "active_connections": metrics.get("websocket_streaming", {}).get("active_connections", 0),
                        "events_per_second": metrics.get("websocket_streaming", {}).get("events_per_second", 0)
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=4)
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Latency analysis failed: {e}")
        
        return insights
    
    async def _analyze_throughput_performance(self, metrics: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze throughput performance and generate optimization insights."""
        insights = []
        
        try:
            events_per_second = metrics.get("event_processor", {}).get("events_per_second", 0)
            target_events_per_second = 1000  # Target throughput
            
            if events_per_second < target_events_per_second * 0.7:  # Below 70% of target
                improvement_potential = ((target_events_per_second - events_per_second) / target_events_per_second) * 100
                
                insight = PerformanceInsight(
                    category=OptimizationCategory.THROUGHPUT,
                    title="Event Processing Throughput Optimization",
                    description=f"Event processing throughput is {events_per_second:.1f} events/s, "
                               f"below optimal capacity of {target_events_per_second} events/s.",
                    current_performance={"events_per_second": events_per_second},
                    target_performance={"events_per_second": target_events_per_second},
                    improvement_potential=improvement_potential,
                    optimization_actions=[
                        "Increase event buffer size for batch processing",
                        "Optimize database connection pooling",
                        "Implement parallel event processing",
                        "Review Redis pipeline optimizations",
                        "Consider horizontal scaling of processors"
                    ],
                    estimated_effort="medium",
                    expected_impact=f"Increase throughput by {improvement_potential:.1f}%",
                    implementation_priority=7,
                    confidence_score=0.75,
                    supporting_metrics={
                        "current_throughput": events_per_second,
                        "buffer_utilization": metrics.get("event_processor", {}).get("buffer_metrics", {}).get("events_buffered", 0),
                        "processing_workers": 1  # Would get actual worker count
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=8)
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Throughput analysis failed: {e}")
        
        return insights
    
    async def _analyze_resource_utilization(self, metrics: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze resource utilization and generate optimization insights."""
        insights = []
        
        try:
            # CPU utilization analysis
            total_cpu_overhead = sum(
                metrics.get("enhanced_prometheus", {}).get("cpu_overhead", {}).values()
            ) if metrics.get("enhanced_prometheus", {}).get("cpu_overhead") else 0
            
            if total_cpu_overhead > 3.0:  # Above 3% target
                improvement_potential = ((total_cpu_overhead - 2.0) / total_cpu_overhead) * 100
                
                insight = PerformanceInsight(
                    category=OptimizationCategory.RESOURCE,
                    title="CPU Overhead Optimization",
                    description=f"Observability system CPU overhead is {total_cpu_overhead:.1f}%, "
                               f"exceeding 3% target. Optimization needed.",
                    current_performance={"cpu_overhead_percent": total_cpu_overhead},
                    target_performance={"cpu_overhead_percent": 2.0},
                    improvement_potential=improvement_potential,
                    optimization_actions=[
                        "Optimize event serialization algorithms",
                        "Reduce metrics collection frequency for non-critical metrics",
                        "Implement more efficient data structures",
                        "Review and optimize hot code paths",
                        "Consider CPU profiling to identify bottlenecks"
                    ],
                    estimated_effort="high",
                    expected_impact=f"Reduce CPU overhead by {improvement_potential:.1f}%",
                    implementation_priority=6,
                    confidence_score=0.70,
                    supporting_metrics={
                        "total_cpu_overhead": total_cpu_overhead,
                        "component_breakdown": metrics.get("enhanced_prometheus", {}).get("cpu_overhead", {})
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=12)
                )
                insights.append(insight)
            
            # Memory utilization analysis (if available)
            buffer_size = metrics.get("event_processor", {}).get("buffer_metrics", {}).get("events_buffered", 0)
            buffer_overflows = metrics.get("event_processor", {}).get("buffer_metrics", {}).get("buffer_overflows", 0)
            
            if buffer_overflows > 0:
                insight = PerformanceInsight(
                    category=OptimizationCategory.RESOURCE,
                    title="Memory Buffer Optimization",
                    description=f"Event buffer overflows detected ({buffer_overflows} overflows). "
                               f"Memory allocation needs optimization.",
                    current_performance={"buffer_overflows": buffer_overflows, "buffer_size": buffer_size},
                    target_performance={"buffer_overflows": 0},
                    improvement_potential=100.0,  # Should be able to eliminate overflows
                    optimization_actions=[
                        "Increase event buffer size",
                        "Implement adaptive buffer sizing",
                        "Optimize memory allocation patterns",
                        "Add back-pressure mechanisms",
                        "Monitor memory usage patterns"
                    ],
                    estimated_effort="low",
                    expected_impact="Eliminate event loss, ensure 100% coverage",
                    implementation_priority=9,
                    confidence_score=1.0,
                    supporting_metrics={"buffer_overflows": buffer_overflows, "buffer_size": buffer_size},
                    valid_until=datetime.utcnow() + timedelta(hours=2)
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Resource analysis failed: {e}")
        
        return insights
    
    async def _analyze_reliability_metrics(self, metrics: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze reliability metrics and generate optimization insights.""" 
        insights = []
        
        try:
            # Event coverage analysis
            event_coverage = metrics.get("event_processor", {}).get("performance_targets", {}).get("current_success_rate", 1.0) * 100
            
            if event_coverage < 99.9:  # Below target
                improvement_potential = (99.9 - event_coverage) / 99.9 * 100
                
                insight = PerformanceInsight(
                    category=OptimizationCategory.RELIABILITY,
                    title="Event Coverage Reliability Improvement",
                    description=f"Event coverage is {event_coverage:.2f}%, below 99.9% target. "
                               f"Some events may be lost.",
                    current_performance={"coverage_percent": event_coverage},
                    target_performance={"coverage_percent": 99.9},
                    improvement_potential=improvement_potential,
                    optimization_actions=[
                        "Implement guaranteed event delivery mechanisms",
                        "Add event retry logic with exponential backoff",
                        "Monitor and resolve event processing bottlenecks",
                        "Implement dead letter queue for failed events",
                        "Add comprehensive error handling"
                    ],
                    estimated_effort="medium",
                    expected_impact=f"Improve coverage to {99.9}%",
                    implementation_priority=9,
                    confidence_score=0.90,
                    supporting_metrics={
                        "current_coverage": event_coverage,
                        "failed_events": metrics.get("event_processor", {}).get("failed_events", 0),
                        "successful_events": metrics.get("event_processor", {}).get("successful_events", 0)
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=4)
                )
                insights.append(insight)
            
            # WebSocket connection reliability
            websocket_errors = metrics.get("websocket_streaming", {}).get("websocket_errors", 0)
            total_connections = metrics.get("websocket_streaming", {}).get("total_connections", 1)
            error_rate = (websocket_errors / max(total_connections, 1)) * 100
            
            if error_rate > 1.0:  # Above 1% error rate
                insight = PerformanceInsight(
                    category=OptimizationCategory.RELIABILITY,
                    title="WebSocket Connection Reliability Improvement",
                    description=f"WebSocket connection error rate is {error_rate:.2f}%, "
                               f"indicating connection stability issues.",
                    current_performance={"error_rate_percent": error_rate},
                    target_performance={"error_rate_percent": 0.5},
                    improvement_potential=50.0,
                    optimization_actions=[
                        "Implement connection health monitoring",
                        "Add automatic connection retry mechanisms",
                        "Optimize connection lifecycle management",
                        "Implement graceful connection degradation",
                        "Monitor network quality metrics"
                    ],
                    estimated_effort="medium",
                    expected_impact="Improve connection reliability by 50%",
                    implementation_priority=6,
                    confidence_score=0.75,
                    supporting_metrics={
                        "websocket_errors": websocket_errors,
                        "total_connections": total_connections,
                        "active_connections": metrics.get("websocket_streaming", {}).get("active_connections", 0)
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=6)
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Reliability analysis failed: {e}")
        
        return insights
    
    async def _analyze_scalability_opportunities(
        self, 
        metrics: Dict[str, Any], 
        system_health: Dict[str, Any]
    ) -> List[PerformanceInsight]:
        """Analyze scalability opportunities and generate insights."""
        insights = []
        
        try:
            # Connection capacity analysis
            active_connections = metrics.get("websocket_streaming", {}).get("active_connections", 0)
            max_connections = 1000  # From config
            utilization = (active_connections / max_connections) * 100
            
            if utilization > 75:  # Above 75% capacity
                insight = PerformanceInsight(
                    category=OptimizationCategory.SCALABILITY,
                    title="WebSocket Connection Scaling Opportunity",
                    description=f"WebSocket connections at {utilization:.1f}% capacity "
                               f"({active_connections}/{max_connections}). Scaling recommended.",
                    current_performance={"connection_utilization_percent": utilization},
                    target_performance={"connection_utilization_percent": 60.0},
                    improvement_potential=25.0,
                    optimization_actions=[
                        "Increase maximum connection limit",
                        "Implement connection load balancing",
                        "Add horizontal scaling for WebSocket servers",
                        "Optimize connection resource usage",
                        "Implement connection pooling strategies"
                    ],
                    estimated_effort="high",
                    expected_impact="Support 2x more concurrent connections",
                    implementation_priority=7,
                    confidence_score=0.85,
                    supporting_metrics={
                        "active_connections": active_connections,
                        "max_connections": max_connections,
                        "utilization": utilization
                    },
                    valid_until=datetime.utcnow() + timedelta(hours=12)
                )
                insights.append(insight)
            
            # Processing capacity analysis
            events_per_second = metrics.get("event_processor", {}).get("events_per_second", 0)
            if events_per_second > 800:  # Approaching limits
                insight = PerformanceInsight(
                    category=OptimizationCategory.SCALABILITY,
                    title="Event Processing Scaling Preparation",
                    description=f"Event processing at {events_per_second:.1f} events/s. "
                               f"Prepare for scaling before reaching limits.",
                    current_performance={"events_per_second": events_per_second},
                    target_performance={"max_sustainable_events_per_second": 1500.0},
                    improvement_potential=40.0,
                    optimization_actions=[
                        "Plan horizontal scaling of event processors",
                        "Implement event sharding strategies",
                        "Optimize database connection pooling",
                        "Add auto-scaling policies",
                        "Monitor resource utilization trends"
                    ],
                    estimated_effort="high",
                    expected_impact="Enable 2x event processing capacity",
                    implementation_priority=5,
                    confidence_score=0.70,
                    supporting_metrics={"current_throughput": events_per_second},
                    valid_until=datetime.utcnow() + timedelta(days=1)
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Scalability analysis failed: {e}")
        
        return insights
    
    async def _prioritize_insights(self, insights: List[PerformanceInsight]) -> List[PerformanceInsight]:
        """Prioritize insights by implementation priority and confidence."""
        try:
            # Sort by priority (higher first), then by confidence (higher first)
            insights.sort(key=lambda i: (-i.implementation_priority, -i.confidence_score))
            
            # Limit to top 10 insights to avoid overwhelming users
            return insights[:10]
            
        except Exception as e:
            logger.error(f"Insight prioritization failed: {e}")
            return insights


class ObservabilitySystemOrchestrator:
    """
    Central orchestrator for the complete observability system.
    
    Coordinates all observability components and provides:
    - Unified system management
    - Performance optimization insights
    - Health monitoring and recovery
    - Enterprise KPI tracking
    """
    
    def __init__(self):
        # Component references
        self.real_time_processor = None
        self.websocket_streaming = None
        self.prometheus_metrics = None
        self.alerting_system = None
        self.analytics_engine = None
        
        # Analysis components
        self.performance_analyzer = PerformanceOptimizationAnalyzer()
        
        # System state
        self.running = False
        self.last_health_check = None
        self.last_insights_generated = None
        
        # Background tasks
        self.orchestration_task = None
        self.health_monitoring_task = None
        
        # Configuration
        self.config = {
            "health_check_interval": 60,        # seconds
            "insights_generation_interval": 300, # seconds
            "performance_analysis_window": 3600,  # seconds
            "auto_recovery_enabled": True,
            "enterprise_reporting_enabled": True
        }
        
        # System metrics
        self.system_metrics = {
            "orchestrator_start_time": datetime.utcnow(),
            "health_checks_completed": 0,
            "insights_generated": 0,
            "auto_recoveries_attempted": 0,
            "system_restarts": 0
        }
        
        logger.info("Observability System Orchestrator initialized")
    
    async def start_system(self) -> bool:
        """
        Start the complete observability system.
        
        Returns:
            bool: True if system started successfully, False otherwise
        """
        if self.running:
            logger.warning("Observability system already running")
            return True
        
        try:
            logger.info("Starting LeanVibe Agent Hive 2.0 Observability System...")
            
            # Start core components
            logger.info("🔧 Starting real-time event processor...")
            self.real_time_processor = await get_real_time_processor()
            
            logger.info("🔄 Starting WebSocket streaming...")
            self.websocket_streaming = await get_enhanced_websocket_streaming()
            
            logger.info("📊 Starting Prometheus metrics collection...")
            self.prometheus_metrics = get_enhanced_prometheus_metrics()
            await start_enhanced_metrics_collection()
            
            logger.info("🚨 Starting intelligent alerting system...")
            self.alerting_system = await get_intelligent_alerting_system()
            
            logger.info("🧠 Starting predictive analytics engine...")
            self.analytics_engine = await get_predictive_analytics_engine()
            
            self.running = True
            
            # Start orchestration tasks
            self.orchestration_task = asyncio.create_task(self._orchestration_loop())
            self.health_monitoring_task = asyncio.create_task(self._health_monitoring_loop())
            
            # Wait a moment for components to initialize
            await asyncio.sleep(2)
            
            # Perform initial health check
            health_report = await self.generate_health_report()
            
            logger.info(
                "✅ LeanVibe Agent Hive 2.0 Observability System STARTED",
                overall_status=health_report.overall_status.value,
                overall_score=f"{health_report.overall_score:.1f}%",
                active_components=len(health_report.component_health)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start observability system: {e}")
            await self.stop_system()
            return False
    
    async def stop_system(self) -> None:
        """Stop the complete observability system."""
        logger.info("Stopping LeanVibe Agent Hive 2.0 Observability System...")
        
        self.running = False
        
        # Cancel orchestration tasks
        if self.orchestration_task:
            self.orchestration_task.cancel()
        if self.health_monitoring_task:
            self.health_monitoring_task.cancel()
        
        # Stop components
        try:
            if self.analytics_engine:
                await shutdown_predictive_analytics_engine()
            if self.alerting_system:
                await shutdown_intelligent_alerting_system()
            if self.websocket_streaming:
                await shutdown_enhanced_websocket_streaming()
            if self.real_time_processor:
                await shutdown_real_time_processor()
        except Exception as e:
            logger.error(f"Error during component shutdown: {e}")
        
        logger.info("✅ Observability system stopped")
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop for system coordination."""
        logger.info("Starting observability orchestration loop")
        
        while self.running:
            try:
                # Generate performance insights periodically
                if (not self.last_insights_generated or 
                    (datetime.utcnow() - self.last_insights_generated).total_seconds() > 
                    self.config["insights_generation_interval"]):
                    
                    await self._generate_performance_insights()
                    self.last_insights_generated = datetime.utcnow()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring and auto-recovery."""
        while self.running:
            try:
                # Perform health check
                health_report = await self.generate_health_report()
                self.system_metrics["health_checks_completed"] += 1
                
                # Auto-recovery if enabled and needed
                if (self.config["auto_recovery_enabled"] and 
                    health_report.overall_status == SystemHealthStatus.CRITICAL):
                    
                    await self._attempt_auto_recovery(health_report)
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _generate_performance_insights(self) -> None:
        """Generate performance optimization insights."""
        try:
            logger.debug("Generating performance optimization insights...")
            
            # Collect metrics from all components
            component_metrics = await self._collect_all_metrics()
            system_health = await self._get_system_health_data()
            
            # Generate insights
            insights = await self.performance_analyzer.analyze_system_performance(
                component_metrics, system_health
            )
            
            self.system_metrics["insights_generated"] += len(insights)
            
            # Broadcast high-priority insights
            for insight in insights:
                if insight.implementation_priority >= 8:  # High priority
                    from app.observability.enhanced_websocket_streaming import broadcast_system_alert
                    await broadcast_system_alert(
                        level="info",
                        message=f"Performance Insight: {insight.title}",
                        source="performance_optimizer",
                        details={
                            "insight_id": insight.id,
                            "category": insight.category.value,
                            "improvement_potential": insight.improvement_potential,
                            "implementation_priority": insight.implementation_priority,
                            "optimization_actions": insight.optimization_actions,
                            "expected_impact": insight.expected_impact
                        }
                    )
            
            if insights:
                logger.info(f"Generated {len(insights)} performance insights")
            
        except Exception as e:
            logger.error(f"Performance insights generation failed: {e}")
    
    async def _collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all observability components."""
        metrics = {}
        
        try:
            # Real-time processor metrics
            if self.real_time_processor:
                processor_health = await get_processor_health()
                metrics["event_processor"] = processor_health.get("metrics", {})
            
            # WebSocket streaming metrics
            if self.websocket_streaming:
                streaming_metrics = self.websocket_streaming.get_metrics()
                metrics["websocket_streaming"] = streaming_metrics
            
            # Prometheus metrics
            if self.prometheus_metrics:
                prometheus_summary = self.prometheus_metrics.get_performance_summary()
                metrics["enhanced_prometheus"] = prometheus_summary
            
            # Alerting system metrics
            if self.alerting_system:
                alerting_summary = self.alerting_system.get_alert_summary()
                metrics["alerting_system"] = alerting_summary
            
            # Analytics engine metrics
            if self.analytics_engine:
                analytics_summary = self.analytics_engine.get_analytics_summary()
                metrics["analytics_engine"] = analytics_summary
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
        
        return metrics
    
    async def _get_system_health_data(self) -> Dict[str, Any]:
        """Get comprehensive system health data."""
        try:
            import psutil
            
            return {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "system_uptime": time.time() - psutil.boot_time(),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"System health data collection failed: {e}")
            return {}
    
    async def _attempt_auto_recovery(self, health_report: SystemHealthReport) -> None:
        """Attempt automatic recovery for critical issues."""
        try:
            logger.warning("Attempting auto-recovery for critical system issues...")
            self.system_metrics["auto_recoveries_attempted"] += 1
            
            recovery_actions = []
            
            # Analyze component health for recovery actions
            for component, health_score in health_report.component_health.items():
                if health_score < 0.5:  # Critical health
                    if component == "real_time_processor":
                        recovery_actions.append("restart_event_processor")
                    elif component == "websocket_streaming":
                        recovery_actions.append("restart_websocket_streaming")
            
            # Execute recovery actions
            for action in recovery_actions:
                await self._execute_recovery_action(action)
            
            if recovery_actions:
                logger.info(f"Auto-recovery attempted: {recovery_actions}")
            
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
    
    async def _execute_recovery_action(self, action: str) -> bool:
        """Execute a specific recovery action."""
        try:
            if action == "restart_event_processor":
                # Restart real-time processor
                await shutdown_real_time_processor()
                await asyncio.sleep(2)
                self.real_time_processor = await get_real_time_processor()
                return True
            
            elif action == "restart_websocket_streaming":
                # Restart WebSocket streaming
                await shutdown_enhanced_websocket_streaming()
                await asyncio.sleep(2)
                self.websocket_streaming = await get_enhanced_websocket_streaming()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Recovery action {action} failed: {e}")
            return False
    
    async def generate_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report."""
        try:
            report = SystemHealthReport()
            
            # Collect component health scores
            component_health = {}
            
            # Real-time processor health
            if self.real_time_processor:
                processor_health = await get_processor_health()
                component_health["real_time_processor"] = (
                    processor_health.get("overall_health_score", 0.5) * 100
                )
            else:
                component_health["real_time_processor"] = 0.0
            
            # WebSocket streaming health
            if self.websocket_streaming:
                streaming_health = await self.websocket_streaming.get_health_status()
                component_health["websocket_streaming"] = (
                    streaming_health.get("overall_health_score", 0.5) * 100
                )
            else:
                component_health["websocket_streaming"] = 0.0
            
            # Prometheus metrics health (assume healthy if available)
            component_health["prometheus_metrics"] = 100.0 if self.prometheus_metrics else 0.0
            
            # Alerting system health (assume healthy if running)
            component_health["alerting_system"] = 100.0 if self.alerting_system else 0.0
            
            # Analytics engine health (assume healthy if running)  
            component_health["analytics_engine"] = 100.0 if self.analytics_engine else 0.0
            
            report.component_health = component_health
            
            # Calculate overall health score
            if component_health:
                report.overall_score = sum(component_health.values()) / len(component_health)
            else:
                report.overall_score = 0.0
            
            # Determine overall status
            if report.overall_score >= 90:
                report.overall_status = SystemHealthStatus.HEALTHY
            elif report.overall_score >= 70:
                report.overall_status = SystemHealthStatus.DEGRADED
            else:
                report.overall_status = SystemHealthStatus.CRITICAL
            
            # Get performance target compliance
            if self.prometheus_metrics:
                prometheus_summary = self.prometheus_metrics.get_performance_summary()
                report.target_compliance = prometheus_summary.get("targets", {})
                report.sli_scores = prometheus_summary.get("sli_scores", {})
            
            # Get system metrics
            report.system_metrics = await self._get_system_health_data()
            
            # Identify active issues
            active_issues = []
            for component, health in component_health.items():
                if health < 70:
                    active_issues.append(f"{component}: Health score {health:.1f}%")
            report.active_issues = active_issues
            
            # Generate recommendations
            recommendations = []
            if report.overall_score < 90:
                recommendations.append("Review component health and performance metrics")
            if any(score < 70 for score in component_health.values()):
                recommendations.append("Investigate and resolve failing components")
            if not report.target_compliance.get("p95_latency_150ms", True):
                recommendations.append("Optimize event processing latency")
            report.recommendations = recommendations
            
            self.last_health_check = datetime.utcnow()
            return report
            
        except Exception as e:
            logger.error(f"Health report generation failed: {e}")
            return SystemHealthReport(overall_status=SystemHealthStatus.UNKNOWN)
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        return {
            "running": self.running,
            "system_metrics": self.system_metrics,
            "config": self.config,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_insights_generated": self.last_insights_generated.isoformat() if self.last_insights_generated else None,
            "components_active": {
                "real_time_processor": self.real_time_processor is not None,
                "websocket_streaming": self.websocket_streaming is not None,
                "prometheus_metrics": self.prometheus_metrics is not None,
                "alerting_system": self.alerting_system is not None,
                "analytics_engine": self.analytics_engine is not None
            }
        }


# Global orchestrator instance
_observability_orchestrator: Optional[ObservabilitySystemOrchestrator] = None


async def get_observability_orchestrator() -> ObservabilitySystemOrchestrator:
    """Get global observability orchestrator instance."""
    global _observability_orchestrator
    
    if _observability_orchestrator is None:
        _observability_orchestrator = ObservabilitySystemOrchestrator()
    
    return _observability_orchestrator


async def start_complete_observability_system() -> bool:
    """Start the complete observability system."""
    orchestrator = await get_observability_orchestrator()
    return await orchestrator.start_system()


async def stop_complete_observability_system() -> None:
    """Stop the complete observability system."""
    global _observability_orchestrator
    
    if _observability_orchestrator:
        await _observability_orchestrator.stop_system()
        _observability_orchestrator = None


async def get_system_health_report() -> SystemHealthReport:
    """Get current system health report."""
    orchestrator = await get_observability_orchestrator()
    return await orchestrator.generate_health_report()


async def get_performance_insights() -> List[Dict[str, Any]]:
    """Get current performance optimization insights."""
    try:
        orchestrator = await get_observability_orchestrator()
        component_metrics = await orchestrator._collect_all_metrics()
        system_health = await orchestrator._get_system_health_data()
        
        insights = await orchestrator.performance_analyzer.analyze_system_performance(
            component_metrics, system_health
        )
        
        return [insight.to_dict() for insight in insights]
        
    except Exception as e:
        logger.error(f"Failed to get performance insights: {e}")
        return []