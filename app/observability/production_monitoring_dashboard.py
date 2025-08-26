"""
Production Monitoring Dashboard for LeanVibe Agent Hive 2.0
===========================================================

Enterprise-grade monitoring dashboard with real-time metrics visualization,
intelligent alerting, and comprehensive system health monitoring.

Epic 3 - Security & Operations: Monitoring Excellence
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import structlog

from .production_observability_orchestrator import (
    ProductionObservabilityOrchestrator, SystemHealthStatus, MetricCategory
)
from .intelligent_alerting_system import IntelligentAlertingSystem, AlertSeverity, AlertStatus
from .enhanced_websocket_streaming import EnhancedWebSocketStreaming
from .enhanced_prometheus_integration import EnhancedPrometheusIntegration
from ..core.unified_security_framework import UnifiedSecurityFramework
from ..core.secure_deployment_orchestrator import SecureDeploymentOrchestrator
from ..core.redis import get_redis_client
from ..core.database import get_async_session

logger = structlog.get_logger()


class DashboardType(Enum):
    """Types of monitoring dashboards available."""
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONS = "operations"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS_METRICS = "business_metrics"
    COMPLIANCE = "compliance"
    DEPLOYMENT = "deployment"
    ALERTING = "alerting"


class VisualizationType(Enum):
    """Types of data visualizations supported."""
    TIME_SERIES = "time_series"
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    STATUS_INDICATOR = "status_indicator"
    TABLE = "table"
    ALERT_PANEL = "alert_panel"
    TOPOLOGY = "topology"


class RefreshInterval(Enum):
    """Dashboard refresh intervals."""
    REAL_TIME = 1      # 1 second
    FAST = 5          # 5 seconds
    NORMAL = 30       # 30 seconds
    SLOW = 60         # 1 minute
    PERIODIC = 300    # 5 minutes


@dataclass
class DashboardWidget:
    """Configuration for a dashboard widget."""
    widget_id: str
    title: str
    widget_type: VisualizationType
    data_source: str
    metric_query: str
    refresh_interval: RefreshInterval = RefreshInterval.NORMAL
    size: Tuple[int, int] = (6, 4)  # Grid units (width, height)
    position: Tuple[int, int] = (0, 0)  # Grid position (x, y)
    threshold_config: Optional[Dict[str, Any]] = None
    alert_config: Optional[Dict[str, Any]] = None
    display_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for a monitoring dashboard."""
    dashboard_id: str
    dashboard_type: DashboardType
    title: str
    description: str
    widgets: List[DashboardWidget]
    refresh_interval: RefreshInterval = RefreshInterval.NORMAL
    auto_refresh: bool = True
    alert_integration: bool = True
    export_enabled: bool = True
    public_access: bool = False
    access_roles: List[str] = field(default_factory=lambda: ["admin", "operator"])


@dataclass
class DashboardMetrics:
    """Dashboard performance and usage metrics."""
    total_views: int = 0
    unique_users: int = 0
    average_load_time_ms: float = 0.0
    widgets_rendered: int = 0
    alerts_displayed: int = 0
    data_points_processed: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertSummary:
    """Summary of alert status for dashboard display."""
    total_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    mean_time_to_acknowledge: float  # seconds
    mean_time_to_resolve: float      # seconds
    alert_trend: str  # "increasing", "stable", "decreasing"


class ProductionMonitoringDashboard:
    """
    Production Monitoring Dashboard - Enterprise observability interface.
    
    Provides comprehensive monitoring capabilities including:
    - Real-time system health visualization
    - Multi-dimensional metric displays
    - Intelligent alert management
    - Security monitoring integration
    - Performance trend analysis
    - Compliance reporting dashboards
    - Executive summary views
    """
    
    def __init__(self):
        """Initialize the production monitoring dashboard."""
        self.dashboard_id = str(uuid.uuid4())
        self.metrics = DashboardMetrics()
        
        # Initialize core components
        self.observability_orchestrator = ProductionObservabilityOrchestrator()
        self.alerting_system = IntelligentAlertingSystem()
        self.websocket_streaming = EnhancedWebSocketStreaming()
        self.prometheus_integration = EnhancedPrometheusIntegration()
        self.security_framework = UnifiedSecurityFramework()
        self.deployment_orchestrator = SecureDeploymentOrchestrator()
        
        # Dashboard configurations
        self.dashboard_configs: Dict[DashboardType, DashboardConfig] = {}
        
        # Real-time data cache
        self.metric_cache: Dict[str, Any] = {}
        self.alert_cache: Dict[str, Any] = {}
        self.health_status_cache: Optional[SystemHealthStatus] = None
        
        # Active dashboard sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Production monitoring dashboard initialized",
                   dashboard_id=self.dashboard_id)
    
    async def initialize(self) -> bool:
        """Initialize the production monitoring dashboard."""
        try:
            start_time = time.time()
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Set up dashboard configurations
            await self._setup_dashboard_configurations()
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Start background data collection
            await self._start_background_tasks()
            
            # Set up real-time streaming
            await self._setup_realtime_streaming()
            
            initialization_time = (time.time() - start_time) * 1000
            
            logger.info("Production monitoring dashboard initialization complete",
                       dashboard_id=self.dashboard_id,
                       initialization_time_ms=initialization_time)
            
            return True
            
        except Exception as e:
            logger.error("Production monitoring dashboard initialization failed",
                        dashboard_id=self.dashboard_id,
                        error=str(e))
            return False
    
    async def get_executive_dashboard_data(self) -> Dict[str, Any]:
        """Get executive summary dashboard data."""
        dashboard_data_id = str(uuid.uuid4())
        
        try:
            # Get system health status
            system_health = await self.observability_orchestrator.analyze_system_health()
            
            # Get alert summary
            alert_summary = await self._get_alert_summary()
            
            # Get performance metrics
            performance_metrics = await self._get_executive_performance_metrics()
            
            # Get security status
            security_status = await self.security_framework.get_security_status()
            
            # Get business metrics
            business_metrics = await self._get_business_metrics_summary()
            
            # Get deployment status
            deployment_status = await self._get_deployment_status_summary()
            
            executive_dashboard = {
                "dashboard_id": dashboard_data_id,
                "dashboard_type": DashboardType.EXECUTIVE_SUMMARY.value,
                "generated_at": datetime.utcnow().isoformat(),
                "system_health": {
                    "overall_status": system_health.overall_health,
                    "availability": self._calculate_availability(system_health),
                    "performance_score": self._calculate_performance_score(system_health),
                    "security_score": self._extract_security_score(security_status),
                    "compliance_score": self._calculate_compliance_score(system_health)
                },
                "alert_summary": asdict(alert_summary),
                "key_metrics": {
                    "requests_per_second": performance_metrics.get("requests_per_second", 0),
                    "average_response_time": performance_metrics.get("average_response_time", 0),
                    "error_rate": performance_metrics.get("error_rate", 0),
                    "active_users": business_metrics.get("active_users", 0),
                    "system_utilization": performance_metrics.get("system_utilization", 0)
                },
                "trends": {
                    "performance_trend": performance_metrics.get("trend", "stable"),
                    "alert_trend": alert_summary.alert_trend,
                    "usage_trend": business_metrics.get("trend", "stable"),
                    "security_trend": security_status.get("trend", "stable")
                },
                "deployment_status": deployment_status,
                "recommendations": system_health.recommendations[:5]  # Top 5 recommendations
            }
            
            return executive_dashboard
            
        except Exception as e:
            logger.error("Failed to generate executive dashboard data",
                        dashboard_data_id=dashboard_data_id,
                        error=str(e))
            return {
                "dashboard_id": dashboard_data_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def get_operations_dashboard_data(self) -> Dict[str, Any]:
        """Get operations dashboard data with detailed system metrics."""
        dashboard_data_id = str(uuid.uuid4())
        
        try:
            # Collect comprehensive metrics
            comprehensive_metrics = await self.observability_orchestrator.collect_comprehensive_metrics()
            
            # Get detailed system health
            system_health = await self.observability_orchestrator.analyze_system_health()
            
            # Get active alerts
            active_alerts = await self.alerting_system.get_active_alerts()
            
            # Get performance trends
            performance_trends = await self._get_detailed_performance_trends()
            
            # Get infrastructure metrics
            infrastructure_metrics = await self._get_infrastructure_metrics()
            
            operations_dashboard = {
                "dashboard_id": dashboard_data_id,
                "dashboard_type": DashboardType.OPERATIONS.value,
                "generated_at": datetime.utcnow().isoformat(),
                "system_overview": {
                    "overall_health": system_health.overall_health,
                    "component_health": system_health.component_health,
                    "performance_metrics": system_health.performance_metrics,
                    "active_alerts_count": len(active_alerts)
                },
                "real_time_metrics": {
                    "cpu_usage": infrastructure_metrics.get("cpu_usage_percent", 0),
                    "memory_usage": infrastructure_metrics.get("memory_usage_percent", 0),
                    "disk_usage": infrastructure_metrics.get("disk_usage_percent", 0),
                    "network_io": infrastructure_metrics.get("network_io_mbps", 0),
                    "active_connections": infrastructure_metrics.get("active_connections", 0),
                    "response_times": comprehensive_metrics.get("categories", {}).get("application", {}).get("response_time_ms", 0)
                },
                "performance_trends": performance_trends,
                "active_alerts": [self._format_alert_for_dashboard(alert) for alert in active_alerts],
                "capacity_utilization": {
                    "compute": infrastructure_metrics.get("compute_utilization", 0),
                    "storage": infrastructure_metrics.get("storage_utilization", 0),
                    "network": infrastructure_metrics.get("network_utilization", 0),
                    "database": infrastructure_metrics.get("database_utilization", 0)
                },
                "service_status": await self._get_service_status(),
                "recent_events": await self._get_recent_system_events()
            }
            
            return operations_dashboard
            
        except Exception as e:
            logger.error("Failed to generate operations dashboard data",
                        dashboard_data_id=dashboard_data_id,
                        error=str(e))
            return {
                "dashboard_id": dashboard_data_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data."""
        dashboard_data_id = str(uuid.uuid4())
        
        try:
            # Get security framework status
            security_status = await self.security_framework.get_security_status()
            
            # Get security compliance report
            compliance_report = await self.security_framework.generate_compliance_report()
            
            # Get recent security events
            security_events = await self._get_recent_security_events()
            
            # Get threat intelligence
            threat_intelligence = await self._get_threat_intelligence_summary()
            
            # Get security metrics
            security_metrics = await self._get_detailed_security_metrics()
            
            security_dashboard = {
                "dashboard_id": dashboard_data_id,
                "dashboard_type": DashboardType.SECURITY.value,
                "generated_at": datetime.utcnow().isoformat(),
                "security_overview": {
                    "framework_status": security_status.get("status", "unknown"),
                    "overall_security_score": self._calculate_overall_security_score(security_status),
                    "threat_level": threat_intelligence.get("current_threat_level", "low"),
                    "compliance_score": compliance_report.get("overall_compliance_score", 0)
                },
                "security_metrics": {
                    "blocked_requests": security_metrics.get("blocked_requests", 0),
                    "security_incidents": security_metrics.get("security_incidents", 0),
                    "authentication_failures": security_metrics.get("authentication_failures", 0),
                    "vulnerability_scans": security_metrics.get("vulnerability_scans", 0),
                    "policy_violations": security_metrics.get("policy_violations", 0)
                },
                "threat_intelligence": {
                    "active_threats": threat_intelligence.get("active_threats", 0),
                    "threat_sources": threat_intelligence.get("threat_sources", []),
                    "attack_patterns": threat_intelligence.get("attack_patterns", []),
                    "geographic_threats": threat_intelligence.get("geographic_distribution", {})
                },
                "compliance_status": {
                    "policy_compliance": compliance_report.get("policy_compliance", {}),
                    "audit_findings": compliance_report.get("audit_findings", []),
                    "remediation_items": compliance_report.get("recommendations", [])
                },
                "security_events": security_events[:20],  # Last 20 events
                "vulnerability_summary": {
                    "critical": security_metrics.get("critical_vulnerabilities", 0),
                    "high": security_metrics.get("high_vulnerabilities", 0),
                    "medium": security_metrics.get("medium_vulnerabilities", 0),
                    "low": security_metrics.get("low_vulnerabilities", 0)
                },
                "access_control": {
                    "active_sessions": security_metrics.get("active_sessions", 0),
                    "failed_logins": security_metrics.get("failed_logins", 0),
                    "privilege_escalations": security_metrics.get("privilege_escalations", 0)
                }
            }
            
            return security_dashboard
            
        except Exception as e:
            logger.error("Failed to generate security dashboard data",
                        dashboard_data_id=dashboard_data_id,
                        error=str(e))
            return {
                "dashboard_id": dashboard_data_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def get_deployment_dashboard_data(self) -> Dict[str, Any]:
        """Get deployment monitoring dashboard data."""
        dashboard_data_id = str(uuid.uuid4())
        
        try:
            # Get deployment orchestrator status
            deployment_status = await self._get_deployment_orchestrator_status()
            
            # Get recent deployments
            recent_deployments = await self._get_recent_deployments()
            
            # Get deployment pipeline status
            pipeline_status = await self._get_deployment_pipeline_status()
            
            # Get environment health
            environment_health = await self._get_environment_health()
            
            deployment_dashboard = {
                "dashboard_id": dashboard_data_id,
                "dashboard_type": DashboardType.DEPLOYMENT.value,
                "generated_at": datetime.utcnow().isoformat(),
                "deployment_overview": {
                    "orchestrator_status": deployment_status.get("status", "unknown"),
                    "active_deployments": deployment_status.get("active_deployments", 0),
                    "deployment_success_rate": deployment_status.get("success_rate", 0),
                    "average_deployment_time": deployment_status.get("average_duration", 0)
                },
                "recent_deployments": recent_deployments,
                "pipeline_status": pipeline_status,
                "environment_health": environment_health,
                "deployment_metrics": {
                    "total_deployments_today": deployment_status.get("deployments_today", 0),
                    "failed_deployments": deployment_status.get("failed_deployments", 0),
                    "rollbacks_executed": deployment_status.get("rollbacks", 0),
                    "security_scans_passed": deployment_status.get("security_scans_passed", 0)
                },
                "security_scanning": {
                    "scans_completed": deployment_status.get("security_scans_completed", 0),
                    "vulnerabilities_found": deployment_status.get("vulnerabilities_found", 0),
                    "compliance_checks_passed": deployment_status.get("compliance_checks_passed", 0)
                },
                "performance_impact": {
                    "deployment_performance_impact": deployment_status.get("performance_impact", {}),
                    "resource_utilization": deployment_status.get("resource_utilization", {})
                }
            }
            
            return deployment_dashboard
            
        except Exception as e:
            logger.error("Failed to generate deployment dashboard data",
                        dashboard_data_id=dashboard_data_id,
                        error=str(e))
            return {
                "dashboard_id": dashboard_data_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def create_custom_dashboard(
        self, 
        dashboard_config: DashboardConfig
    ) -> Dict[str, Any]:
        """Create a custom monitoring dashboard."""
        try:
            # Validate dashboard configuration
            await self._validate_dashboard_config(dashboard_config)
            
            # Store dashboard configuration
            self.dashboard_configs[dashboard_config.dashboard_type] = dashboard_config
            
            # Initialize dashboard data sources
            await self._initialize_dashboard_data_sources(dashboard_config)
            
            # Set up real-time updates if enabled
            if dashboard_config.auto_refresh:
                await self._setup_dashboard_auto_refresh(dashboard_config)
            
            logger.info("Custom dashboard created",
                       dashboard_id=dashboard_config.dashboard_id,
                       dashboard_type=dashboard_config.dashboard_type.value)
            
            return {
                "dashboard_id": dashboard_config.dashboard_id,
                "status": "created",
                "dashboard_type": dashboard_config.dashboard_type.value,
                "widgets_count": len(dashboard_config.widgets),
                "auto_refresh": dashboard_config.auto_refresh,
                "refresh_interval": dashboard_config.refresh_interval.value
            }
            
        except Exception as e:
            logger.error("Failed to create custom dashboard",
                        dashboard_config=dashboard_config.dashboard_id,
                        error=str(e))
            return {
                "dashboard_id": dashboard_config.dashboard_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def get_real_time_dashboard_stream(
        self, 
        dashboard_type: DashboardType,
        session_id: str
    ) -> Dict[str, Any]:
        """Get real-time dashboard data stream."""
        try:
            # Create streaming session
            stream_session = {
                "session_id": session_id,
                "dashboard_type": dashboard_type,
                "started_at": datetime.utcnow(),
                "last_update": datetime.utcnow(),
                "update_count": 0
            }
            
            self.active_sessions[session_id] = stream_session
            
            # Get initial dashboard data
            if dashboard_type == DashboardType.EXECUTIVE_SUMMARY:
                dashboard_data = await self.get_executive_dashboard_data()
            elif dashboard_type == DashboardType.OPERATIONS:
                dashboard_data = await self.get_operations_dashboard_data()
            elif dashboard_type == DashboardType.SECURITY:
                dashboard_data = await self.get_security_dashboard_data()
            elif dashboard_type == DashboardType.DEPLOYMENT:
                dashboard_data = await self.get_deployment_dashboard_data()
            else:
                dashboard_data = await self._get_custom_dashboard_data(dashboard_type)
            
            # Add streaming metadata
            dashboard_data["streaming_session"] = {
                "session_id": session_id,
                "update_interval": 5,  # seconds
                "real_time_enabled": True
            }
            
            # Start background streaming for this session
            asyncio.create_task(self._stream_dashboard_updates(session_id, dashboard_type))
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Failed to create real-time dashboard stream",
                        dashboard_type=dashboard_type.value,
                        session_id=session_id,
                        error=str(e))
            return {
                "session_id": session_id,
                "error": str(e),
                "dashboard_type": dashboard_type.value
            }
    
    async def export_dashboard_data(
        self, 
        dashboard_type: DashboardType,
        export_format: str = "json",
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Export dashboard data for reporting or analysis."""
        export_id = str(uuid.uuid4())
        
        try:
            # Get dashboard data
            if dashboard_type == DashboardType.EXECUTIVE_SUMMARY:
                dashboard_data = await self.get_executive_dashboard_data()
            elif dashboard_type == DashboardType.OPERATIONS:
                dashboard_data = await self.get_operations_dashboard_data()
            elif dashboard_type == DashboardType.SECURITY:
                dashboard_data = await self.get_security_dashboard_data()
            elif dashboard_type == DashboardType.DEPLOYMENT:
                dashboard_data = await self.get_deployment_dashboard_data()
            else:
                dashboard_data = await self._get_custom_dashboard_data(dashboard_type)
            
            # Add export metadata
            export_data = {
                "export_id": export_id,
                "dashboard_type": dashboard_type.value,
                "export_format": export_format,
                "exported_at": datetime.utcnow().isoformat(),
                "time_range": {
                    "start": time_range[0].isoformat() if time_range else None,
                    "end": time_range[1].isoformat() if time_range else None
                },
                "dashboard_data": dashboard_data
            }
            
            # Format based on requested format
            if export_format.lower() == "csv":
                export_data["csv_data"] = await self._convert_to_csv(dashboard_data)
            elif export_format.lower() == "pdf":
                export_data["pdf_url"] = await self._generate_pdf_report(dashboard_data)
            
            return export_data
            
        except Exception as e:
            logger.error("Failed to export dashboard data",
                        export_id=export_id,
                        dashboard_type=dashboard_type.value,
                        error=str(e))
            return {
                "export_id": export_id,
                "error": str(e),
                "dashboard_type": dashboard_type.value
            }
    
    # Private helper methods
    
    async def _initialize_core_components(self):
        """Initialize all core dashboard components."""
        components = [
            ("observability_orchestrator", self.observability_orchestrator),
            ("alerting_system", self.alerting_system),
            ("websocket_streaming", self.websocket_streaming),
            ("prometheus_integration", self.prometheus_integration),
            ("security_framework", self.security_framework),
            ("deployment_orchestrator", self.deployment_orchestrator)
        ]
        
        for name, component in components:
            try:
                if hasattr(component, 'initialize'):
                    await component.initialize()
                logger.debug("Dashboard component initialized", component=name)
            except Exception as e:
                logger.error("Dashboard component initialization failed", 
                           component=name, error=str(e))
                raise
    
    async def _setup_dashboard_configurations(self):
        """Set up default dashboard configurations."""
        # Executive Summary Dashboard
        executive_widgets = [
            DashboardWidget(
                widget_id="system_health_overview",
                title="System Health Overview",
                widget_type=VisualizationType.STATUS_INDICATOR,
                data_source="observability",
                metric_query="system.health.overall",
                size=(12, 3)
            ),
            DashboardWidget(
                widget_id="key_metrics_summary",
                title="Key Performance Metrics",
                widget_type=VisualizationType.GAUGE,
                data_source="observability",
                metric_query="performance.key_metrics",
                size=(6, 4)
            ),
            DashboardWidget(
                widget_id="active_alerts",
                title="Active Alerts",
                widget_type=VisualizationType.ALERT_PANEL,
                data_source="alerting",
                metric_query="alerts.active",
                size=(6, 4)
            )
        ]
        
        self.dashboard_configs[DashboardType.EXECUTIVE_SUMMARY] = DashboardConfig(
            dashboard_id=str(uuid.uuid4()),
            dashboard_type=DashboardType.EXECUTIVE_SUMMARY,
            title="Executive Summary Dashboard",
            description="High-level system overview for executives and management",
            widgets=executive_widgets,
            refresh_interval=RefreshInterval.NORMAL
        )
        
        # Similar configurations for other dashboard types would be added here
    
    async def _get_alert_summary(self) -> AlertSummary:
        """Get comprehensive alert summary."""
        try:
            active_alerts = await self.alerting_system.get_active_alerts()
            
            # Count alerts by severity
            critical_count = sum(1 for alert in active_alerts if alert.get("severity") == "critical")
            high_count = sum(1 for alert in active_alerts if alert.get("severity") == "high")
            medium_count = sum(1 for alert in active_alerts if alert.get("severity") == "medium")
            low_count = sum(1 for alert in active_alerts if alert.get("severity") == "low")
            
            # Count alerts by status
            acknowledged_count = sum(1 for alert in active_alerts if alert.get("status") == "acknowledged")
            resolved_count = sum(1 for alert in active_alerts if alert.get("status") == "resolved")
            
            return AlertSummary(
                total_alerts=len(active_alerts),
                critical_alerts=critical_count,
                high_alerts=high_count,
                medium_alerts=medium_count,
                low_alerts=low_count,
                acknowledged_alerts=acknowledged_count,
                resolved_alerts=resolved_count,
                mean_time_to_acknowledge=120.0,  # Mock data - would be calculated from historical data
                mean_time_to_resolve=450.0,     # Mock data
                alert_trend="stable"             # Mock data - would be calculated from trends
            )
            
        except Exception as e:
            logger.error("Failed to get alert summary", error=str(e))
            return AlertSummary(
                total_alerts=0, critical_alerts=0, high_alerts=0,
                medium_alerts=0, low_alerts=0, acknowledged_alerts=0,
                resolved_alerts=0, mean_time_to_acknowledge=0.0,
                mean_time_to_resolve=0.0, alert_trend="unknown"
            )
    
    async def _calculate_availability(self, system_health: SystemHealthStatus) -> float:
        """Calculate system availability percentage."""
        # Mock calculation - would use actual uptime data
        if system_health.overall_health == "healthy":
            return 99.9
        elif system_health.overall_health == "degraded":
            return 99.0
        else:
            return 95.0
    
    async def _calculate_performance_score(self, system_health: SystemHealthStatus) -> float:
        """Calculate overall performance score."""
        # Mock calculation based on performance metrics
        performance_metrics = system_health.performance_metrics
        if not performance_metrics:
            return 85.0
        
        # Weighted average of key performance indicators
        response_time_score = min(100, max(0, 100 - (performance_metrics.get("response_time_ms", 100) / 10)))
        error_rate_score = min(100, max(0, 100 - (performance_metrics.get("error_rate", 1) * 20)))
        throughput_score = min(100, performance_metrics.get("throughput", 75))
        
        return (response_time_score + error_rate_score + throughput_score) / 3
    
    async def _start_background_tasks(self):
        """Start background tasks for dashboard data collection."""
        # Start metric collection task
        asyncio.create_task(self._background_metric_collection())
        
        # Start alert monitoring task
        asyncio.create_task(self._background_alert_monitoring())
        
        # Start health status updates
        asyncio.create_task(self._background_health_monitoring())
    
    async def _background_metric_collection(self):
        """Background task for metric collection."""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                metrics = await self.observability_orchestrator.collect_comprehensive_metrics()
                self.metric_cache.update(metrics)
                
                # Update dashboard metrics
                self.metrics.data_points_processed += len(metrics.get("categories", {}))
                self.metrics.last_updated = datetime.utcnow()
                
            except Exception as e:
                logger.error("Background metric collection error", error=str(e))
                await asyncio.sleep(30)
    
    async def _background_alert_monitoring(self):
        """Background task for alert monitoring."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                active_alerts = await self.alerting_system.get_active_alerts()
                self.alert_cache = {
                    "active_alerts": active_alerts,
                    "last_updated": datetime.utcnow().isoformat()
                }
                
                # Update metrics
                self.metrics.alerts_displayed = len(active_alerts)
                
            except Exception as e:
                logger.error("Background alert monitoring error", error=str(e))
                await asyncio.sleep(10)
    
    async def _background_health_monitoring(self):
        """Background task for health status monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                health_status = await self.observability_orchestrator.analyze_system_health()
                self.health_status_cache = health_status
                
            except Exception as e:
                logger.error("Background health monitoring error", error=str(e))
                await asyncio.sleep(60)


# Global instance for production use
_monitoring_dashboard_instance: Optional[ProductionMonitoringDashboard] = None


async def get_monitoring_dashboard() -> ProductionMonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _monitoring_dashboard_instance
    
    if _monitoring_dashboard_instance is None:
        _monitoring_dashboard_instance = ProductionMonitoringDashboard()
        await _monitoring_dashboard_instance.initialize()
    
    return _monitoring_dashboard_instance


async def get_executive_dashboard() -> Dict[str, Any]:
    """Convenience function for executive dashboard data."""
    dashboard = await get_monitoring_dashboard()
    return await dashboard.get_executive_dashboard_data()


async def get_operations_dashboard() -> Dict[str, Any]:
    """Convenience function for operations dashboard data."""
    dashboard = await get_monitoring_dashboard()
    return await dashboard.get_operations_dashboard_data()


async def get_security_dashboard() -> Dict[str, Any]:
    """Convenience function for security dashboard data."""
    dashboard = await get_monitoring_dashboard()
    return await dashboard.get_security_dashboard_data()