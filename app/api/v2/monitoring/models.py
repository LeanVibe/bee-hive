"""
SystemMonitoringAPI v2 Models

Unified response models consolidating all monitoring data structures
from the 9 original monitoring modules.

Epic 4 Phase 2 - Consolidated Models Architecture
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


class MonitoringStatus(str, Enum):
    """System monitoring status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric type classifications."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# ==================== CORE RESPONSE MODELS ====================

class MonitoringResponse(BaseModel):
    """Base response model for all monitoring endpoints."""
    status: str = Field(..., description="Response status")
    timestamp: datetime = Field(..., description="Response timestamp")
    data: Dict[str, Any] = Field(..., description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class SystemHealthStatus(BaseModel):
    """System health status model."""
    overall_status: MonitoringStatus = Field(..., description="Overall system health status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    error_rate: float = Field(..., description="System error rate (0-1)")
    performance_score: float = Field(..., description="Performance score (0-100)")
    components: Dict[str, MonitoringStatus] = Field(default_factory=dict, description="Component health status")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last health check timestamp")


class AlertData(BaseModel):
    """Alert data model."""
    id: str = Field(..., description="Alert ID")
    severity: AlertSeverity = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(..., description="Alert timestamp")
    resolved: bool = Field(False, description="Alert resolution status")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    source: Optional[str] = Field(None, description="Alert source system")
    affected_components: List[str] = Field(default_factory=list, description="Affected system components")


class MetricValue(BaseModel):
    """Individual metric value model."""
    name: str = Field(..., description="Metric name")
    value: Union[float, int, str] = Field(..., description="Metric value")
    type: MetricType = Field(..., description="Metric type")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    timestamp: datetime = Field(..., description="Metric timestamp")
    help_text: Optional[str] = Field(None, description="Metric description")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoints."""
    metrics: List[MetricValue] = Field(..., description="List of metrics")
    total_metrics: int = Field(..., description="Total number of metrics")
    generated_at: datetime = Field(..., description="Metrics generation timestamp")
    format_type: str = Field("json", description="Metrics format type")
    cache_hit: bool = Field(False, description="Whether response was cached")


class PerformanceStats(BaseModel):
    """Performance statistics model."""
    response_time_p95: float = Field(..., description="95th percentile response time (ms)")
    response_time_avg: float = Field(0, description="Average response time (ms)")
    throughput_rps: float = Field(..., description="Throughput in requests per second")
    error_rate: float = Field(..., description="Error rate (0-1)")
    cpu_usage: float = Field(..., description="CPU usage percentage (0-100)")
    memory_usage: float = Field(..., description="Memory usage percentage (0-100)")
    disk_usage: Optional[float] = Field(None, description="Disk usage percentage (0-100)")
    network_io: Optional[Dict[str, float]] = Field(None, description="Network I/O statistics")
    database_connections: Optional[int] = Field(None, description="Active database connections")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate (0-1)")


class BusinessMetrics(BaseModel):
    """Business intelligence metrics model."""
    revenue_growth: float = Field(..., description="Revenue growth percentage")
    customer_satisfaction: float = Field(..., description="Customer satisfaction score (0-100)")
    operational_efficiency: float = Field(..., description="Operational efficiency score (0-100)")
    market_share: float = Field(..., description="Market share percentage")
    cost_savings: Optional[float] = Field(None, description="Cost savings achieved")
    roi_percentage: Optional[float] = Field(None, description="Return on investment percentage")
    user_engagement: Optional[Dict[str, float]] = Field(None, description="User engagement metrics")
    conversion_rates: Optional[Dict[str, float]] = Field(None, description="Conversion rate metrics")


class StrategicIntelligence(BaseModel):
    """Strategic intelligence model."""
    focus_areas: List[str] = Field(default_factory=list, description="Strategic focus areas")
    time_horizon_months: int = Field(6, description="Analysis time horizon in months")
    predictions_included: bool = Field(True, description="Whether predictions are included")
    market_trends: List[str] = Field(default_factory=list, description="Identified market trends")
    competitive_analysis: Dict[str, Any] = Field(default_factory=dict, description="Competitive analysis data")
    opportunities: List[str] = Field(default_factory=list, description="Strategic opportunities")
    risks: List[str] = Field(default_factory=list, description="Strategic risks")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Strategic recommendations")
    confidence_score: float = Field(0.8, description="Analysis confidence score (0-1)")


class ObservabilityEvent(BaseModel):
    """Observability event model."""
    event_id: str = Field(..., description="Unique event ID")
    event_type: str = Field(..., description="Event type")
    event_category: str = Field(..., description="Event category")
    timestamp: datetime = Field(..., description="Event timestamp")
    agent_id: Optional[UUID] = Field(None, description="Associated agent ID")
    session_id: Optional[UUID] = Field(None, description="Associated session ID")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    correlation_id: Optional[str] = Field(None, description="Event correlation ID")
    latency_ms: Optional[int] = Field(None, description="Event latency in milliseconds")
    severity: AlertSeverity = Field(AlertSeverity.INFO, description="Event severity")


class AgentMetrics(BaseModel):
    """Agent metrics model."""
    total_agents: int = Field(..., description="Total number of agents")
    active_agents: int = Field(..., description="Number of active agents")
    inactive_agents: int = Field(..., description="Number of inactive agents")
    error_agents: int = Field(..., description="Number of agents in error state")
    avg_health_score: float = Field(..., description="Average agent health score")
    stale_heartbeats: int = Field(..., description="Agents with stale heartbeats")
    utilization_rate: float = Field(..., description="Agent utilization rate (0-1)")


class TaskMetrics(BaseModel):
    """Task metrics model."""
    total_tasks: int = Field(..., description="Total number of tasks")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    in_progress_tasks: int = Field(..., description="Number of in-progress tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    queue_length: int = Field(..., description="Current task queue length")
    avg_completion_time: float = Field(..., description="Average task completion time (minutes)")
    success_rate: float = Field(..., description="Task success rate (0-1)")
    long_running_tasks: int = Field(..., description="Number of long-running tasks")


class DashboardData(BaseModel):
    """Unified dashboard data model."""
    timestamp: datetime = Field(..., description="Dashboard data timestamp")
    period: str = Field(..., description="Data time period")
    format_type: str = Field(..., description="Dashboard format type")
    
    # Core system data
    system_health: SystemHealthStatus = Field(..., description="System health status")
    agent_metrics: Dict[str, Any] = Field(..., description="Agent metrics data")
    task_metrics: Dict[str, Any] = Field(..., description="Task metrics data")
    
    # Analytics and intelligence
    performance_metrics: PerformanceStats = Field(..., description="Performance statistics")
    business_metrics: BusinessMetrics = Field(..., description="Business intelligence metrics")
    strategic_intelligence: StrategicIntelligence = Field(..., description="Strategic intelligence data")
    
    # Real-time data
    alerts: List[AlertData] = Field(default_factory=list, description="Active system alerts")
    events: List[ObservabilityEvent] = Field(default_factory=list, description="Recent system events")
    
    # Predictive data
    forecasts: Optional[Dict[str, Any]] = Field(None, description="Predictive forecasts")
    
    # Metadata
    cache_hit: bool = Field(False, description="Whether data was served from cache")
    processing_time_ms: Optional[float] = Field(None, description="Data processing time")
    data_freshness_seconds: int = Field(0, description="Age of data in seconds")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    connection_id: Optional[str] = Field(None, description="Connection ID")
    sequence_number: Optional[int] = Field(None, description="Message sequence number")


class MobileAccessData(BaseModel):
    """Mobile access data model."""
    qr_code: str = Field(..., description="Base64 encoded QR code")
    mobile_url: str = Field(..., description="Mobile dashboard URL")
    dashboard_type: str = Field(..., description="Dashboard type")
    api_version: str = Field("2.0", description="API version")
    expires_at: datetime = Field(..., description="QR code expiration timestamp")
    features: Dict[str, bool] = Field(default_factory=dict, description="Available mobile features")


class ConnectionStats(BaseModel):
    """WebSocket connection statistics."""
    total_connections: int = Field(..., description="Total active connections")
    connections_by_type: Dict[str, int] = Field(default_factory=dict, description="Connections grouped by type")
    total_events_sent: int = Field(..., description="Total events sent")
    total_errors: int = Field(..., description="Total connection errors")
    error_rate: float = Field(..., description="Connection error rate (0-1)")
    avg_connection_duration: float = Field(..., description="Average connection duration (seconds)")


class PrometheusMetric(BaseModel):
    """Prometheus-compatible metric model."""
    name: str = Field(..., description="Metric name")
    help_text: str = Field(..., description="Metric help text")
    type: MetricType = Field(..., description="Metric type")
    values: List[Dict[str, Any]] = Field(..., description="Metric values with labels")


class AnalyticsReport(BaseModel):
    """Analytics report model."""
    report_id: str = Field(..., description="Report ID")
    report_type: str = Field(..., description="Report type")
    generated_at: datetime = Field(..., description="Report generation timestamp")
    time_range: str = Field(..., description="Report time range")
    data: Dict[str, Any] = Field(..., description="Report data")
    insights: List[str] = Field(default_factory=list, description="Key insights")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    confidence_level: float = Field(0.8, description="Report confidence level (0-1)")


# ==================== REQUEST MODELS ====================

class EventFilterRequest(BaseModel):
    """Event filtering request model."""
    event_types: Optional[List[str]] = Field(None, description="Filter by event types")
    event_categories: Optional[List[str]] = Field(None, description="Filter by event categories")
    agent_ids: Optional[List[UUID]] = Field(None, description="Filter by agent IDs")
    session_ids: Optional[List[UUID]] = Field(None, description="Filter by session IDs")
    since: Optional[datetime] = Field(None, description="Events since timestamp")
    until: Optional[datetime] = Field(None, description="Events until timestamp")
    limit: int = Field(100, ge=1, le=1000, description="Maximum events to return")
    severity: Optional[AlertSeverity] = Field(None, description="Filter by severity")


class DashboardRequest(BaseModel):
    """Dashboard request model."""
    period: str = Field("current", description="Time period")
    include_forecasts: bool = Field(True, description="Include predictive forecasts")
    format_type: str = Field("standard", description="Response format")
    refresh_cache: bool = Field(False, description="Force cache refresh")
    components: Optional[List[str]] = Field(None, description="Specific components to include")


class MetricsRequest(BaseModel):
    """Metrics request model."""
    format_type: str = Field("prometheus", description="Metrics format")
    categories: Optional[List[str]] = Field(None, description="Metric categories to include")
    time_range: Optional[str] = Field(None, description="Metrics time range")
    aggregation: str = Field("latest", description="Metric aggregation method")


class AlertRequest(BaseModel):
    """Alert management request model."""
    severity: Optional[AlertSeverity] = Field(None, description="Alert severity filter")
    resolved: Optional[bool] = Field(None, description="Filter by resolution status")
    component: Optional[str] = Field(None, description="Filter by component")
    limit: int = Field(50, ge=1, le=200, description="Maximum alerts to return")


# ==================== ERROR MODELS ====================

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    trace_id: Optional[str] = Field(None, description="Error trace ID")


class ValidationError(BaseModel):
    """Validation error model."""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value")


# ==================== CONFIGURATION MODELS ====================

class CacheConfig(BaseModel):
    """Cache configuration model."""
    enabled: bool = Field(True, description="Cache enabled status")
    ttl_seconds: int = Field(300, description="Cache TTL in seconds")
    max_size: Optional[int] = Field(None, description="Maximum cache size")
    hit_rate: Optional[float] = Field(None, description="Current cache hit rate")


class SecurityConfig(BaseModel):
    """Security configuration model."""
    auth_required: bool = Field(True, description="Authentication required")
    permissions_required: List[str] = Field(default_factory=list, description="Required permissions")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    allowed_origins: List[str] = Field(default_factory=list, description="Allowed CORS origins")


class MonitoringConfig(BaseModel):
    """Monitoring configuration model."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    security_config: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    websocket_enabled: bool = Field(True, description="WebSocket streaming enabled")
    mobile_enabled: bool = Field(True, description="Mobile interface enabled")
    analytics_enabled: bool = Field(True, description="Analytics enabled")
    forecasting_enabled: bool = Field(True, description="Forecasting enabled")